"""Generic training entry point for the verified five-model pipeline.

This script intentionally does not create a data allocation. Supply a CSV with
the columns public_image_id, image_path, label, and split (train/val/test).
Tiny-MLP additionally requires feature_00 through feature_77 in the same CSV.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "code" / "models"))
from model_definitions import CLASS_NAMES, MODEL_NAMES, build_model  # noqa: E402

FEATURE_COLUMNS = [f"feature_{index:02d}" for index in range(78)]


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic)


class ImageDataset(Dataset):
    def __init__(self, table: pd.DataFrame, transform):
        self.table = table.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int):
        row = self.table.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        return self.transform(image), int(row["label"])


class FeatureDataset(Dataset):
    def __init__(self, table: pd.DataFrame):
        self.table = table.reset_index(drop=True)
        self.features = self.table[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        self.labels = self.table["label"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, index: int):
        return torch.tensor(self.features[index]), int(self.labels[index])


def make_loader(table: pd.DataFrame, input_mode: str, split: str, batch_size: int, seed: int) -> DataLoader:
    if input_mode == "features":
        dataset = FeatureDataset(table)
    else:
        steps = [transforms.Resize((128, 128))]
        if split == "train":
            steps.append(transforms.ColorJitter(brightness=0.025, contrast=0.025, saturation=0.015))
        steps.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        dataset = ImageDataset(table, transforms.Compose(steps))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=0,
                      generator=generator if split == "train" else None)


def class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.maximum(np.bincount(labels, minlength=3), 1)
    weights = counts.sum() / (3 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels, probabilities = [], []
    with torch.no_grad():
        for inputs, target in loader:
            probability = functional.softmax(model(inputs.to(device)), dim=1).cpu().numpy()
            labels.extend(target.numpy().tolist())
            probabilities.append(probability)
    return np.asarray(labels), np.vstack(probabilities)


def metric_bundle(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predicted = probabilities.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, predicted),
        "macro_precision": precision_score(labels, predicted, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, predicted, average="macro", zero_division=0),
        "macro_f1": f1_score(labels, predicted, average="macro", zero_division=0),
        "macro_auc": roc_auc_score(labels, probabilities, average="macro", multi_class="ovr"),
    }


def train_one(model_name: str, table: pd.DataFrame, config: dict, output_root: Path) -> None:
    training = config["training"]
    seed = int(training["seed"])
    deterministic = bool(training["deterministic_algorithms"])
    set_seed(seed, deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, input_mode = build_model(model_name)
    model.to(device)
    loaders = {
        split: make_loader(table.query("split == @split"), input_mode, split, int(training["batch_size"]), seed)
        for split in ("train", "val", "test")
    }
    labels = loaders["train"].dataset.table["label"].to_numpy(dtype=np.int64)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(labels, device),
        label_smoothing=float(training["label_smoothing"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])
    )
    maximum_epochs = int(training["maximum_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(training["scheduler"]["T_max"]))
    patience = int(training["early_stopping_patience"])
    best_state, best_score, best_loss, wait = None, -1.0, math.inf, 0
    history: list[dict] = []
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for inputs, target in loaders["train"]:
            inputs, target = inputs.to(device), target.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * target.size(0)
            train_correct += (logits.argmax(1) == target).sum().item()
            train_total += target.size(0)
        val_labels, val_probabilities = predict(model, loaders["val"], device)
        val_predicted = val_probabilities.argmax(axis=1)
        val_f1 = f1_score(val_labels, val_predicted, average="macro", zero_division=0)
        model.eval()
        val_loss, val_total = 0.0, 0
        with torch.no_grad():
            for inputs, target in loaders["val"]:
                inputs, target = inputs.to(device), target.to(device)
                loss = criterion(model(inputs), target)
                val_loss += loss.item() * target.size(0)
                val_total += target.size(0)
        mean_val_loss = val_loss / max(1, val_total)
        better = val_f1 > best_score + 1e-12 or (abs(val_f1 - best_score) <= 1e-12 and mean_val_loss < best_loss)
        if better:
            best_score, best_loss, best_state, wait = val_f1, mean_val_loss, deepcopy(model.state_dict()), 0
        else:
            wait += 1
        history.append({
            "epoch": epoch,
            "train_loss": train_loss / max(1, train_total),
            "train_accuracy": train_correct / max(1, train_total),
            "validation_loss": mean_val_loss,
            "validation_macro_f1": val_f1,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        scheduler.step()
        if wait >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    test_table = table.query("split == 'test'").reset_index(drop=True)
    test_labels, test_probabilities = predict(model, loaders["test"], device)
    predicted = test_probabilities.argmax(axis=1)
    output = output_root / model_name.replace("-", "_")
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    prediction = test_table[[column for column in test_table.columns if column in {
        "public_image_id", "dataset", "portion_id", "storage_day", "technical_replicate", "label"
    }]].copy()
    prediction["predicted_label"] = predicted
    for index, class_name in enumerate(CLASS_NAMES):
        prediction[f"prob_{class_name}"] = test_probabilities[:, index]
    prediction.to_csv(output / "test_predictions.csv", index=False)
    pd.DataFrame([{"model": model_name, "n_test": len(test_labels), **metric_bundle(test_labels, test_probabilities)}]).to_csv(
        output / "metrics.csv", index=False
    )
    torch.save(model.state_dict(), output / "model_state.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "config" / "model_and_training_config.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=(*MODEL_NAMES, "all"), default="all")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    table = pd.read_csv(args.manifest)
    required = {"public_image_id", "image_path", "label", "split"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    if set(table["split"]) != {"train", "val", "test"}:
        raise ValueError("split must contain train, val, and test")
    models = MODEL_NAMES if args.model == "all" else (args.model,)
    for model_name in models:
        if model_name == "Tiny-MLP" and not set(FEATURE_COLUMNS).issubset(table.columns):
            raise ValueError("Tiny-MLP requires feature_00 through feature_77")
        train_one(model_name, table, config, args.output)


if __name__ == "__main__":
    main()
