# Meat-freshness sensor-array reproducibility package

## Project overview

This repository contains the computational reproducibility package associated with the article “Smartphone-assisted aerogel colorimetric sensor array coupled with lightweight deep learning for multi-meat freshness monitoring.” It includes generic preprocessing, 78-dimensional feature extraction, five model definitions, evaluation code, processed dye-grid images, saved prediction probabilities, and recalculable result tables.

## Repository structure

```text
code/                 Generic preprocessing, features, models, training, and evaluation
config/               Verified model and training configuration
data/predictions/     Saved fixed-test and four-fold outer-test probabilities
processed_data/       Processed dye-grid images and 78-dimensional feature tables
results/              Recalculated image-level and portion-day-level results
environment/          Requirements and recorded software environment
```

## Included materials

- ROI cropping and 2×2 dye-grid construction code.
- Exact 78-dimensional feature-extraction code.
- Definitions for Spot-CNN, ShuffleNetV2-0.5x, EfficientNet-Lite0, MobileNetV3-Small, and Tiny-MLP.
- Generic training logic using a user-supplied train/validation/test manifest.
- Image-level metric recalculation and portion-day probability aggregation.
- 840 beef and 840 pork processed 2×2 dye-grid PNG files.
- One 78-dimensional feature table per meat type.
- Saved probabilities for the fixed test and four-fold outer-test evaluations.
- Recalculated metrics and example result tables.

## Software environment

The recorded environment used Python 3.10.20, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, timm 1.0.27, scikit-learn 1.7.2, NumPy 1.26.4, pandas 2.3.3, Pillow 12.2.0, SciPy 1.15.3, matplotlib 3.10.8, and CUDA runtime 12.8. Exact package pins are listed in `environment/requirements.txt`.

## Installation

Create a Python 3.10 environment, then run from the repository root:

```bash
python -m pip install -r environment/requirements.txt
```

## Recalculation commands

```bash
python -I -B code/evaluation/recalculate_metrics.py --repo . --output recalculated_image_level
python -I -B code/evaluation/portion_day_aggregation.py --repo . --output recalculated_portion_day
python -I -B validation/validate_release.py --repo .
```

The first two commands use released probabilities and do not train models or change any allocation.

## Portion-day aggregation analysis

For each meat portion and storage day, the three technical-replicate probability vectors are averaged class by class. The class with the largest mean probability is the portion-day prediction. Macro-AUC is recalculated from the averaged three-class probabilities. This reduces each fixed test dataset from 210 image units to 70 portion-day units and each outer-test fold from 210 image units to 70 portion-day units.

## Data description

Each meat type comprises 40 independently stored portions, seven storage days, and three technical-replicate images per portion-day (840 image units). Public identifiers use `dataset`, `portion_id`, `storage_day`, and `technical_replicate`; acquisition-specific filenames and local paths are not retained. Processed dye-grids are 128×128 RGB PNG files with the fixed ROI order MR, BTB, BCP, and BCG.

The same preprocessing, feature-extraction, model-definition, training, and evaluation framework was used for the beef and pork datasets.

## Label mapping

The public labels are `Fresh`, `Sub-fresh`, and `Spoiled`, encoded as 0, 1, and 2. The internal historical token `Borderline` corresponds to the manuscript label `Sub-fresh`.

## What is not included

- Raw smartphone photographs.
- LabelMe annotation JSON files.
- Fixed-split or cross-validation allocation manifests.
- Original specimen-to-role assignment files.
- Model weights.
- Manuscripts, supplementary files, reviewer-response working files, logs, caches, and local development metadata.

Dataset-specific allocation manifests are not included in this public release. Consequently, the released predictions and metrics can be independently recalculated, and the computational pipeline can be inspected and reused, but the exact recorded training allocations cannot be reconstructed from this package alone.

## Reproducibility scope

The package supports verification of processed-image counts, feature values, saved probabilities, image-level metrics, technical-replicate consistency, portion-day probability aggregation, confusion matrices, and four-fold summary statistics. Generic training code is included for transparent inspection and reuse with an author-supplied allocation manifest; no allocation is generated automatically.

## Citation

Please cite the associated article and repository record after publication:

- Article DOI: `[paper DOI]`
- Repository DOI: `[Zenodo DOI]`

## License

Licenses:

- Code: MIT License (`LICENSE_CODE.txt`).
- Processed data: Creative Commons Attribution 4.0 International (`LICENSE_DATA.txt`).
