"""Standalone validation checks for the public release candidate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from PIL import Image


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_csv(expected: Path, observed: Path, tolerance: float = 1e-12) -> float:
    first, second = pd.read_csv(expected), pd.read_csv(observed)
    if list(first.columns) != list(second.columns) or len(first) != len(second):
        raise AssertionError(f"CSV structure differs: {expected.name}")
    maximum = 0.0
    for column in first.columns:
        if pd.api.types.is_numeric_dtype(first[column]) and pd.api.types.is_numeric_dtype(second[column]):
            values = np.abs(first[column].to_numpy(float) - second[column].to_numpy(float))
            difference = 0.0 if np.isnan(values).all() else np.nanmax(values)
            maximum = max(maximum, float(difference))
        elif not first[column].fillna("").astype(str).equals(second[column].fillna("").astype(str)):
            raise AssertionError(f"Text differs in {expected.name}: {column}")
    if maximum > tolerance:
        raise AssertionError(f"Numeric difference {maximum} exceeds {tolerance}: {expected.name}")
    return maximum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    repository = args.repo.resolve()
    findings: dict[str, object] = {}

    image_counts = {}
    for dataset in ("beef", "pork"):
        images = sorted((repository / "processed_data" / "dye_grids" / dataset).glob("*.png"))
        if len(images) != 840:
            raise AssertionError(f"Expected 840 {dataset} images, found {len(images)}")
        image_counts[dataset] = len(images)
        features = pd.read_csv(repository / "processed_data" / "features" / f"{dataset}_features_78d.csv")
        columns = [f"feature_{index:02d}" for index in range(78)]
        if len(features) != 840 or not set(columns).issubset(features.columns):
            raise AssertionError(f"Invalid {dataset} feature table")
        if features["public_image_id"].nunique() != 840:
            raise AssertionError(f"Duplicate {dataset} public image identifier")
    findings["processed_image_counts"] = image_counts
    findings["feature_rows"] = 1680
    findings["feature_dimension"] = 78

    fixed = pd.read_csv(repository / "data" / "predictions" / "fixed_test_predictions.csv")
    cross_validation = pd.read_csv(repository / "data" / "predictions" / "cv_outer_test_predictions.csv")
    if len(fixed) != 2100 or len(cross_validation) != 8400:
        raise AssertionError("Unexpected prediction row count")
    findings["fixed_prediction_rows"] = len(fixed)
    findings["cross_validation_prediction_rows"] = len(cross_validation)

    feature_module = load_module("public_features", repository / "code" / "features" / "extract_features.py")
    first = pd.read_csv(repository / "processed_data" / "features" / "beef_features_78d.csv").iloc[0]
    image = repository / "processed_data" / "dye_grids" / "beef" / f"{first['public_image_id']}.png"
    calculated = feature_module.color_features_from_image(image)
    expected = first[[f"feature_{index:02d}" for index in range(78)]].to_numpy(float)
    feature_difference = float(np.max(np.abs(calculated - expected)))
    if not np.allclose(calculated, expected, rtol=0, atol=1e-6):
        raise AssertionError("Feature-extraction smoke test failed")
    findings["feature_smoke_test_maximum_absolute_difference"] = feature_difference

    preprocessing = load_module("public_preprocessing", repository / "code" / "preprocessing" / "roi_preprocessing.py")
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary = Path(temporary_directory)
        source_image = temporary / "synthetic.png"
        annotation = temporary / "synthetic.json"
        canvas = Image.new("RGB", (240, 240), "white")
        pixels = np.asarray(canvas).copy()
        colors = ((220, 40, 40), (230, 210, 40), (70, 170, 70), (50, 80, 210))
        centers = ((70, 70), (170, 70), (70, 170), (170, 170))
        yy, xx = np.ogrid[:240, :240]
        for color, (center_x, center_y) in zip(colors, centers):
            mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= 30 ** 2
            pixels[mask] = color
        Image.fromarray(pixels).save(source_image)
        annotation.write_text(json.dumps({"shapes": [
            {"label": name, "shape_type": "circle", "points": [[x, y], [x + 30, y]]}
            for name, (x, y) in zip(preprocessing.ROI_ORDER, centers)
        ]}), encoding="utf-8")
        grid = preprocessing.construct_dye_grid(source_image, annotation)
        if grid.size != (128, 128):
            raise AssertionError("Preprocessing smoke test produced the wrong image size")
    findings["preprocessing_smoke_test"] = "PASS"

    model_module = load_module("public_models", repository / "code" / "models" / "model_definitions.py")
    parameter_counts = {}
    for model_name in model_module.MODEL_NAMES:
        model, input_mode = model_module.build_model(model_name)
        parameter_counts[model_name] = {"parameters": sum(item.numel() for item in model.parameters()), "input_mode": input_mode}
    findings["model_definition_smoke_test"] = "PASS"
    findings["model_parameter_counts"] = parameter_counts

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary = Path(temporary_directory)
        image_output = temporary / "image_metrics"
        portion_output = temporary / "portion_day"
        subprocess.run([sys.executable, "-I", "-B", str(repository / "code" / "evaluation" / "recalculate_metrics.py"),
                        "--repo", str(repository), "--output", str(image_output)], check=True)
        subprocess.run([sys.executable, "-I", "-B", str(repository / "code" / "evaluation" / "portion_day_aggregation.py"),
                        "--repo", str(repository), "--output", str(portion_output)], check=True)
        image_differences = {
            name: compare_csv(repository / "results" / "image_level_metrics" / name, image_output / name)
            for name in ("fixed_test_metrics.csv", "cv_fold_metrics.csv", "cv_mean_sample_sd.csv")
        }
        portion_names = (
            "portion_day_aggregated_predictions.csv", "portion_day_fixed_test_metrics.csv",
            "portion_day_cv_fold_metrics.csv", "portion_day_cv_mean_sample_sd.csv",
            "portion_day_confusion_matrices.csv", "technical_replicate_consistency.csv",
        )
        portion_differences = {
            name: compare_csv(repository / "results" / "portion_day_analysis" / name, portion_output / name)
            for name in portion_names
        }
    findings["image_level_recalculation_maximum_difference"] = max(image_differences.values())
    findings["portion_day_recalculation_maximum_difference"] = max(portion_differences.values())

    forbidden_paths = []
    absolute_pattern = re.compile("[A-Za-z]" + r":[\\/]" + "(?:Users|1)" + r"[\\/]" + "|/" + "home/", re.IGNORECASE)
    cache_items = []
    for path in repository.rglob("*"):
        relative = path.relative_to(repository).as_posix()
        if path.is_dir() and path.name in {"__pycache__", ".pytest_cache", ".idea", ".vscode"}:
            cache_items.append(relative)
        if path.is_file() and path.suffix.lower() == ".pyc":
            cache_items.append(relative)
        if path.is_file() and path.suffix.lower() in {".py", ".json", ".csv", ".txt", ".md", ".cff"}:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            if absolute_pattern.search(text):
                forbidden_paths.append(relative)
    if forbidden_paths or cache_items:
        raise AssertionError(f"Sanitization failure: absolute={forbidden_paths}, cache={cache_items}")
    findings["absolute_path_scan"] = "PASS"
    findings["cache_scan"] = "PASS"

    checksum_path = repository / "SHA256SUMS.txt"
    if checksum_path.exists():
        checked = 0
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            digest, relative = line.split("  ", 1)
            target = repository / Path(relative)
            if not target.exists() or sha256(target) != digest:
                raise AssertionError(f"Checksum mismatch: {relative}")
            checked += 1
        findings["checksums_validated"] = checked
    else:
        findings["checksums_validated"] = "not yet generated"

    report = args.report or repository / "VALIDATION_REPORT.txt"
    report.write_text("PUBLIC RELEASE VALIDATION\n\nStatus: PASS\n\n" + json.dumps(findings, indent=2), encoding="utf-8")
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
