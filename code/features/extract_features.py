"""Extract the verified 78-dimensional feature vector from 2x2 dye-grids."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROI_SIZE = 64
FEATURE_COLUMNS = [f"feature_{index:02d}" for index in range(78)]


def color_features_from_image(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image.size != (128, 128):
        raise ValueError(f"Expected a 128 x 128 dye-grid, found {image.size}: {path}")
    features: list[float] = []
    lab_means: list[np.ndarray] = []
    for index in range(4):
        row, column = divmod(index, 2)
        patch = image.crop((column * ROI_SIZE, row * ROI_SIZE, (column + 1) * ROI_SIZE, (row + 1) * ROI_SIZE))
        rgb = np.asarray(patch, dtype=np.float32) / 255.0
        mask = ~(np.all(rgb > 0.985, axis=2))
        if int(mask.sum()) < 16:
            mask = np.ones((ROI_SIZE, ROI_SIZE), dtype=bool)
        hsv = np.asarray(patch.convert("HSV"), dtype=np.float32) / 255.0
        lab = np.asarray(patch.convert("LAB"), dtype=np.float32) / 255.0
        lab_means.append(lab[mask].mean(axis=0))
        for values in (rgb, hsv, lab):
            selected = values[mask]
            features.extend(selected.mean(axis=0).tolist())
            features.extend(selected.std(axis=0).tolist())
    for first, second in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        features.append(float(np.linalg.norm(lab_means[first] - lab_means[second])))
    result = np.asarray(features, dtype=np.float32)
    if result.size != 78:
        raise AssertionError(f"Expected 78 features, found {result.size}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True, help="CSV containing image_path and public_image_id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    table = pd.read_csv(args.manifest)
    required = {"image_path", "public_image_id"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    values = np.vstack([color_features_from_image(path) for path in table["image_path"]])
    output = table.copy()
    for index, column in enumerate(FEATURE_COLUMNS):
        output[column] = values[:, index]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
