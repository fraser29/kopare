# Sinus/Airway Detection Quick Guide

This module provides multiple candidate algorithms for identifying internal air
(sinuses and airways) in MRI volumes, plus benchmarking utilities against manual
masks.

Module path: `kopare/sinus_detection.py`

## Inputs

- `image`: 3D MRI array, shape `(z, y, x)`
- `body_mask`: same shape, where `1` = inside body and `0` = external air
- `manual_mask` (for testing): same shape, binary sinus/airway ground truth

## 1) Run one segmentation

```python
import numpy as np
from kopare.sinus_detection import segment_sinus_and_airways

# image, body_mask are numpy arrays with same shape
pred = segment_sinus_and_airways(
    image,
    body_mask,
    method="persistent_dark_after_smoothing",
    sigma_large=2.5,
    min_region_size=300,
    hole_size=400,
)
```

Available methods:
- `threshold_external_reference`
- `thick_region_filter`
- `persistent_dark_after_smoothing` (default)

## 2) Benchmark methods on dataset

```python
from kopare.sinus_detection import benchmark_algorithms

cases = [
    {"image": img1, "body_mask": bm1, "manual_mask": gt1, "case_id": "subj01"},
    {"image": img2, "body_mask": bm2, "manual_mask": gt2, "case_id": "subj02"},
]

results = benchmark_algorithms(cases)
print(results["persistent_dark_after_smoothing"]["summary"])
```

Each summary includes:
- `dice_mean`, `dice_std`, `dice_median`
- `iou_mean`, `precision_mean`, `recall_mean`, `specificity_mean`

## 3) Tune parameters with grid search

```python
from kopare.sinus_detection import grid_search_method_on_cases

grid = {
    "sigma_large": [1.5, 2.0, 2.5, 3.0],
    "min_region_size": [100, 200, 400],
    "hole_size": [150, 300, 600],
}

search = grid_search_method_on_cases(
    cases,
    method="persistent_dark_after_smoothing",
    param_grid=grid,
    objective="dice_mean",
    top_k=5,
)

print("Best params:", search["best"]["params"])
print("Best score:", search["best"]["score"])
```

## 4) Tune and compare multiple methods

```python
from kopare.sinus_detection import grid_search_algorithms

method_param_grids = {
    "threshold_external_reference": {
        "min_region_size": [100, 200, 400],
        "hole_size": [100, 300],
        "closing_radius": [0, 1, 2],
    },
    "thick_region_filter": {
        "opening_radius": [1, 2, 3],
        "regrow_radius": [0, 1, 2],
        "min_region_size": [100, 200, 400],
    },
    "persistent_dark_after_smoothing": {
        "sigma_large": [1.5, 2.0, 2.5, 3.0],
        "min_region_size": [100, 200, 400],
        "hole_size": [150, 300, 600],
    },
}

all_search = grid_search_algorithms(
    cases,
    method_param_grids=method_param_grids,
    objective="dice_mean",
)

print("Best method:", all_search["best_overall"]["method"])
print("Best params:", all_search["best_overall"]["best_params"])
print("Best Dice mean:", all_search["best_overall"]["best_score"])
```

## Notes

- If your manual masks include only sinus/airway regions (not external air),
  these functions are directly compatible.
- The default method is usually a good first option because it prefers regions
  that stay dark after smoothing (often reducing thin-bone false positives).
