"""Generic ROI cropping and 2x2 dye-grid construction.

The four annotated dye regions are ordered spatially as MR, BTB, BCP, and
BCG (top-left, top-right, bottom-left, bottom-right). Non-ROI pixels are
replaced by a constant white background. Each ROI is resized to 64 x 64
pixels with bilinear interpolation and assembled into a 128 x 128 image.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROI_ORDER = ("MR", "BTB", "BCP", "BCG")
ROI_SIZE = 64
GRID_SIZE = 128
ROI_SCALE = 1.06


def read_labelme_rois(json_path: str | Path) -> list[dict]:
    """Read four circle/polygon ROIs from a LabelMe annotation."""
    with Path(json_path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    rois: list[dict] = []
    for index, shape in enumerate(data.get("shapes", []), 1):
        points = shape.get("points") or []
        if not points:
            continue
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            continue
        shape_type = str(shape.get("shape_type", "")).lower()
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        if shape_type == "circle" and len(pts) >= 2:
            cx, cy = pts[0]
            ex, ey = pts[1]
            radius = float(math.hypot(ex - cx, ey - cy))
            x1, y1, x2, y2 = cx - radius, cy - radius, cx + radius, cy + radius
        else:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            radius = max(x2 - x1, y2 - y1) / 2.0
        if radius < 2 or x2 <= x1 or y2 <= y1:
            continue
        rois.append({
            "index": index,
            "label": str(shape.get("label", "")),
            "shape_type": shape_type or "polygon",
            "points": pts.tolist(),
            "cx": float(cx),
            "cy": float(cy),
            "bbox": (float(x1), float(y1), float(x2), float(y2)),
        })
    return rois


def order_rois_2x2(rois: list[dict]) -> list[dict]:
    """Return ROIs in row-major order: MR, BTB, BCP, BCG."""
    if len(rois) != 4:
        raise ValueError(f"Expected four ROIs, found {len(rois)}")
    by_y = sorted(rois, key=lambda item: item["cy"])
    return sorted(by_y[:2], key=lambda item: item["cx"]) + sorted(
        by_y[2:], key=lambda item: item["cx"]
    )


def _scaled_points(points: list[list[float]], box: tuple[int, int, int, int]) -> list[tuple[float, float]]:
    left, top, right, bottom = box
    width, height = max(1, right - left), max(1, bottom - top)
    return [((x - left) / width * ROI_SIZE, (y - top) / height * ROI_SIZE) for x, y in points]


def crop_roi(image: Image.Image, roi: dict, fill=(255, 255, 255)) -> Image.Image:
    """Crop one ROI using the verified constant-background procedure."""
    width, height = image.size
    x1, y1, x2, y2 = roi["bbox"]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * ROI_SCALE
    left = max(0, int(math.floor(cx - side / 2.0)))
    top = max(0, int(math.floor(cy - side / 2.0)))
    right = min(width, int(math.ceil(cx + side / 2.0)))
    bottom = min(height, int(math.ceil(cy + side / 2.0)))
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid crop box: {(left, top, right, bottom)}")
    box = (left, top, right, bottom)
    patch = image.crop(box).convert("RGB").resize((ROI_SIZE, ROI_SIZE), Image.Resampling.BILINEAR)
    mask = Image.new("L", (ROI_SIZE, ROI_SIZE), 0)
    draw = ImageDraw.Draw(mask)
    if roi["shape_type"] == "circle":
        x1s, y1s, x2s, y2s = roi["bbox"]
        sx1 = (x1s - left) / (right - left) * ROI_SIZE
        sy1 = (y1s - top) / (bottom - top) * ROI_SIZE
        sx2 = (x2s - left) / (right - left) * ROI_SIZE
        sy2 = (y2s - top) / (bottom - top) * ROI_SIZE
        draw.ellipse((sx1, sy1, sx2, sy2), fill=255)
    elif len(roi["points"]) >= 3:
        draw.polygon(_scaled_points(roi["points"], box), fill=255)
    else:
        draw.rectangle((0, 0, ROI_SIZE, ROI_SIZE), fill=255)
    canvas = Image.new("RGB", (ROI_SIZE, ROI_SIZE), fill)
    canvas.paste(patch, (0, 0), mask)
    return canvas


def construct_dye_grid(image_path: str | Path, annotation_path: str | Path) -> Image.Image:
    """Construct one 128 x 128 dye-grid from an image and four ROIs."""
    image = Image.open(image_path).convert("RGB")
    rois = order_rois_2x2(read_labelme_rois(annotation_path))
    grid = Image.new("RGB", (GRID_SIZE, GRID_SIZE), (255, 255, 255))
    for index, roi in enumerate(rois):
        row, column = divmod(index, 2)
        grid.paste(crop_roi(image, roi), (column * ROI_SIZE, row * ROI_SIZE))
    return grid


def run_manifest(manifest_path: Path, output_root: Path) -> None:
    """Process a user-supplied CSV with image_path, annotation_path, output_name."""
    table = pd.read_csv(manifest_path)
    required = {"image_path", "annotation_path", "output_name"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    output_root.mkdir(parents=True, exist_ok=True)
    for row in table.itertuples(index=False):
        destination = output_root / str(row.output_name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        construct_dye_grid(row.image_path, row.annotation_path).save(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run_manifest(args.manifest, args.output_root)


if __name__ == "__main__":
    main()
