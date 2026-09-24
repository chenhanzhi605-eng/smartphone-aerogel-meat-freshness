"""Average three technical-replicate probability vectors per portion-day."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

CLASS_NAMES = ["Fresh", "Sub-fresh", "Spoiled"]
PROBABILITY_COLUMNS = [f"prob_{name}" for name in CLASS_NAMES]


def metric_bundle(table: pd.DataFrame) -> dict[str, float]:
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


def aggregate(table: pd.DataFrame, fold_column: bool) -> pd.DataFrame:
    keys = ["dataset", "model"] + (["fold"] if fold_column else []) + ["portion_id", "storage_day"]
    if not table.groupby(keys)["label"].nunique().eq(1).all():
        raise ValueError("A portion-day contains inconsistent true labels")
    counts = table.groupby(keys).size()
    if not counts.eq(3).all():
        raise ValueError("Every portion-day must contain exactly three technical-replicate images")
    aggregated = table.groupby(keys, as_index=False).agg(
        label=("label", "first"),
        n_technical_images=("public_image_id", "size"),
        prob_Fresh=("prob_Fresh", "mean"),
        **{"prob_Sub-fresh": ("prob_Sub-fresh", "mean")},
        prob_Spoiled=("prob_Spoiled", "mean"),
    )
    aggregated["predicted_label"] = aggregated[PROBABILITY_COLUMNS].to_numpy().argmax(axis=1)
    aggregated["predicted_class"] = aggregated["predicted_label"].map(dict(enumerate(CLASS_NAMES)))
    return aggregated


def consistency_rows(table: pd.DataFrame, scope: str, fold_column: bool) -> list[dict]:
    rows = []
    keys = ["portion_id", "storage_day"]
    for group_keys, part in table.groupby((["dataset", "model"] + (["fold"] if fold_column else [])), sort=False):
        group_keys = group_keys if isinstance(group_keys, tuple) else (group_keys,)
        dataset, model = group_keys[:2]
        fold = group_keys[2] if fold_column else None
        predictions = part.assign(_prediction=part[PROBABILITY_COLUMNS].to_numpy().argmax(axis=1))
        unique_counts = predictions.groupby(keys)["_prediction"].nunique()
        rows.append({
            "scope": scope, "dataset": dataset, "model": model, "fold": fold,
            "total_portion_day_groups": len(unique_counts),
            "fully_consistent_groups": int((unique_counts == 1).sum()),
            "consistency_percent": float((unique_counts == 1).mean() * 100),
            "two_to_one_disagreement": int((unique_counts == 2).sum()),
            "three_way_disagreement": int((unique_counts == 3).sum()),
        })
    return rows


def evaluate(aggregated: pd.DataFrame, scope: str, fold_column: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics, matrices = [], []
    group_columns = ["dataset", "model"] + (["fold"] if fold_column else [])
    for group_keys, part in aggregated.groupby(group_columns, sort=False):
        group_keys = group_keys if isinstance(group_keys, tuple) else (group_keys,)
        dataset, model = group_keys[:2]
        fold = group_keys[2] if fold_column else None
        metrics.append({"scope": scope, "dataset": dataset, "model": model, "fold": fold,
                        "n_portion_day_units": len(part), **metric_bundle(part)})
        matrix = confusion_matrix(part["label"], part["predicted_label"], labels=[0, 1, 2])
        for true_index, true_class in enumerate(CLASS_NAMES):
            matrices.append({
                "scope": scope, "dataset": dataset, "model": model, "fold": fold,
                "true_class": true_class,
                "predicted_Fresh": int(matrix[true_index, 0]),
                "predicted_Sub-fresh": int(matrix[true_index, 1]),
                "predicted_Spoiled": int(matrix[true_index, 2]),
            })
    return pd.DataFrame(metrics), pd.DataFrame(matrices)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fixed = pd.read_csv(args.repo / "data" / "predictions" / "fixed_test_predictions.csv")
    cross_validation = pd.read_csv(args.repo / "data" / "predictions" / "cv_outer_test_predictions.csv")
    fixed_aggregated = aggregate(fixed, False)
    cv_aggregated = aggregate(cross_validation, True)
    fixed_metrics, fixed_matrices = evaluate(fixed_aggregated, "fixed_test", False)
    cv_metrics, cv_matrices = evaluate(cv_aggregated, "cross_validation", True)
    cv_summary = cv_metrics.groupby(["dataset", "model"], as_index=False).agg(
        completed_folds=("fold", "nunique"),
        accuracy_mean=("accuracy", "mean"), accuracy_sample_sd=("accuracy", "std"),
        macro_precision_mean=("macro_precision", "mean"), macro_precision_sample_sd=("macro_precision", "std"),
        macro_recall_mean=("macro_recall", "mean"), macro_recall_sample_sd=("macro_recall", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_sample_sd=("macro_f1", "std"),
        macro_auc_mean=("macro_auc", "mean"), macro_auc_sample_sd=("macro_auc", "std"),
    )
    consistency = pd.DataFrame(
        consistency_rows(fixed, "fixed_test", False) + consistency_rows(cross_validation, "cross_validation", True)
    )
    pd.concat([fixed_aggregated.assign(scope="fixed_test"), cv_aggregated.assign(scope="cross_validation")], ignore_index=True).to_csv(
        args.output / "portion_day_aggregated_predictions.csv", index=False
    )
    fixed_metrics.to_csv(args.output / "portion_day_fixed_test_metrics.csv", index=False)
    cv_metrics.to_csv(args.output / "portion_day_cv_fold_metrics.csv", index=False)
    cv_summary.to_csv(args.output / "portion_day_cv_mean_sample_sd.csv", index=False)
    pd.concat([fixed_matrices, cv_matrices], ignore_index=True).to_csv(args.output / "portion_day_confusion_matrices.csv", index=False)
    consistency.to_csv(args.output / "technical_replicate_consistency.csv", index=False)


if __name__ == "__main__":
    main()
