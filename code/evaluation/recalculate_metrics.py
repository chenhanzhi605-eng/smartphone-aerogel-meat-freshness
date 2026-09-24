"""Recalculate image-level classification metrics from released probabilities."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

PROBABILITY_COLUMNS = ["prob_Fresh", "prob_Sub-fresh", "prob_Spoiled"]


def metrics(table: pd.DataFrame) -> dict[str, float]:
    labels = table["label"].astype(int).to_numpy()
    probabilities = table[PROBABILITY_COLUMNS].to_numpy(float)
    predicted = probabilities.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, predicted),
        "macro_precision": precision_score(labels, predicted, average="macro", zero_division=0),
        "macro_recall": recall_score(labels, predicted, average="macro", zero_division=0),
        "macro_f1": f1_score(labels, predicted, average="macro", zero_division=0),
        "macro_auc": roc_auc_score(labels, probabilities, average="macro", multi_class="ovr"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fixed = pd.read_csv(args.repo / "data" / "predictions" / "fixed_test_predictions.csv")
    cross_validation = pd.read_csv(args.repo / "data" / "predictions" / "cv_outer_test_predictions.csv")
    fixed_rows = []
    for (dataset, model), part in fixed.groupby(["dataset", "model"], sort=False):
        fixed_rows.append({"dataset": dataset, "model": model, "n_image_units": len(part), **metrics(part)})
    fold_rows = []
    for (dataset, fold, model), part in cross_validation.groupby(["dataset", "fold", "model"], sort=False):
        fold_rows.append({"dataset": dataset, "fold": fold, "model": model, "n_image_units": len(part), **metrics(part)})
    fixed_metrics = pd.DataFrame(fixed_rows)
    fold_metrics = pd.DataFrame(fold_rows)
    summary = fold_metrics.groupby(["dataset", "model"], as_index=False).agg(
        completed_folds=("fold", "nunique"),
        accuracy_mean=("accuracy", "mean"), accuracy_sample_sd=("accuracy", "std"),
        macro_precision_mean=("macro_precision", "mean"), macro_precision_sample_sd=("macro_precision", "std"),
        macro_recall_mean=("macro_recall", "mean"), macro_recall_sample_sd=("macro_recall", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_sample_sd=("macro_f1", "std"),
        macro_auc_mean=("macro_auc", "mean"), macro_auc_sample_sd=("macro_auc", "std"),
    )
    fixed_metrics.to_csv(args.output / "fixed_test_metrics.csv", index=False)
    fold_metrics.to_csv(args.output / "cv_fold_metrics.csv", index=False)
    summary.to_csv(args.output / "cv_mean_sample_sd.csv", index=False)


if __name__ == "__main__":
    main()
