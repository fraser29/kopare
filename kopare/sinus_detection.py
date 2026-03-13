"""Experimental sinus/airway detection for 3D MRI volumes.

This module focuses on identifying internal air regions (sinuses and airways)
inside a known body mask where:
    - body_mask == 1: inside body
    - body_mask == 0: external air / outside body

Multiple candidate algorithms are provided so they can be benchmarked against
manual segmentations.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable

import numpy as np
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label
from skimage.morphology import (
    ball,
    binary_closing,
    binary_opening,
    remove_small_holes,
    remove_small_objects,
)

try:
    import vtk
    from vtk.util import numpy_support  # type: ignore[reportMissingImports]
except Exception:  # pragma: no cover - optional dependency at runtime
    vtk = None
    numpy_support = None


NDArrayBool = np.ndarray
NDArrayFloat = np.ndarray


@dataclass
class SinusDetectionCase:
    """Container for one benchmark case."""

    image: np.ndarray
    body_mask: np.ndarray
    manual_mask: np.ndarray
    case_id: str = "case"


def _as_numpy_volume(volume: Any, vtk_array_name: str | None = None) -> np.ndarray:
    """Return a (z, y, x) numpy array from numpy array or vtkImageData."""
    if isinstance(volume, np.ndarray):
        return volume

    if vtk is not None and isinstance(volume, vtk.vtkImageData):
        if vtk_array_name is None:
            vtk_arr = volume.GetPointData().GetScalars()
        else:
            vtk_arr = volume.GetPointData().GetArray(vtk_array_name)
            if vtk_arr is None:
                raise ValueError(f"VTK array '{vtk_array_name}' not found.")

        if vtk_arr is None:
            raise ValueError("No scalar data found in vtkImageData.")

        arr = numpy_support.vtk_to_numpy(vtk_arr)
        dims_xyz = volume.GetDimensions()
        # Convert VTK (x, y, z) storage to numpy (z, y, x)
        arr = np.reshape(arr, dims_xyz, order="F")
        arr = np.transpose(arr, (2, 1, 0))
        return arr

    raise TypeError(
        "Expected a numpy.ndarray or vtk.vtkImageData for volume input."
    )


def _validate_inputs(image: np.ndarray, body_mask: np.ndarray) -> tuple[NDArrayFloat, NDArrayBool]:
    """Validate shapes and return cleaned image/mask arrays."""
    image = np.asarray(image, dtype=float)
    body_mask = np.asarray(body_mask) > 0
    if image.ndim != 3 or body_mask.ndim != 3:
        raise ValueError("image and body_mask must be 3D arrays.")
    if image.shape != body_mask.shape:
        raise ValueError("image and body_mask must have the same shape.")
    if not np.any(body_mask):
        raise ValueError("body_mask has no positive voxels.")
    return image, body_mask


def estimate_air_threshold_from_external_air(
    image: np.ndarray,
    body_mask: np.ndarray,
    external_air_percentile: float = 99.5,
    body_low_percentile: float = 15.0,
    blend_weight_external: float = 0.7,
) -> float:
    """Estimate threshold separating air-like signal from non-air.

    Uses known external air (`body_mask == 0`) as a reference.
    """
    image, body_mask = _validate_inputs(image, body_mask)
    external_air = image[~body_mask]
    inside_body = image[body_mask]
    if external_air.size == 0:
        # Fallback: no external region available, use global inside-body Otsu
        return float(threshold_otsu(inside_body))

    t_external = np.percentile(external_air, external_air_percentile)
    t_body_low = np.percentile(inside_body, body_low_percentile)
    return float(
        blend_weight_external * t_external
        + (1.0 - blend_weight_external) * t_body_low
    )


def _postprocess_internal_air_mask(
    candidate_mask: NDArrayBool,
    min_region_size: int = 200,
    hole_size: int = 300,
    closing_radius: int = 1,
) -> NDArrayBool:
    """Clean a binary candidate mask with basic morphology."""
    mask = candidate_mask.astype(bool)
    if closing_radius > 0:
        mask = binary_closing(mask, footprint=ball(closing_radius))
    if hole_size > 0:
        mask = remove_small_holes(mask, area_threshold=hole_size)
    if min_region_size > 0:
        mask = remove_small_objects(mask, min_size=min_region_size)
    return mask


def algorithm_threshold_external_reference(
    image: np.ndarray,
    body_mask: np.ndarray,
    *,
    threshold: float | None = None,
    min_region_size: int = 200,
    hole_size: int = 300,
    closing_radius: int = 1,
) -> NDArrayBool:
    """Algorithm A: threshold-based internal air detection."""
    image, body_mask = _validate_inputs(image, body_mask)
    if threshold is None:
        threshold = estimate_air_threshold_from_external_air(image, body_mask)
    candidate = body_mask & (image <= threshold)
    return _postprocess_internal_air_mask(
        candidate,
        min_region_size=min_region_size,
        hole_size=hole_size,
        closing_radius=closing_radius,
    )


def algorithm_thick_region_filter(
    image: np.ndarray,
    body_mask: np.ndarray,
    *,
    threshold: float | None = None,
    opening_radius: int = 2,
    regrow_radius: int = 1,
    min_region_size: int = 200,
) -> NDArrayBool:
    """Algorithm B: keep only thicker air-like structures.

    Idea: thin dark structures are more likely cortical bone; opening removes
    them while preserving thicker cavities such as sinus/airway regions.
    """
    image, body_mask = _validate_inputs(image, body_mask)
    if threshold is None:
        threshold = estimate_air_threshold_from_external_air(image, body_mask)

    candidate = body_mask & (image <= threshold)
    if opening_radius > 0:
        candidate = binary_opening(candidate, footprint=ball(opening_radius))
    if regrow_radius > 0:
        candidate = binary_closing(candidate, footprint=ball(regrow_radius))
    candidate = remove_small_objects(candidate, min_size=min_region_size)
    return candidate.astype(bool)


def algorithm_persistent_dark_after_smoothing(
    image: np.ndarray,
    body_mask: np.ndarray,
    *,
    threshold_raw: float | None = None,
    threshold_smooth: float | None = None,
    sigma_large: float = 2.0,
    min_region_size: int = 200,
    hole_size: int = 300,
) -> NDArrayBool:
    """Algorithm C: dark in raw image and still dark after large smoothing.

    The smoothing step suppresses thin structures; if a voxel remains dark
    in the smoothed image, it is more likely inside a larger air cavity.
    """
    image, body_mask = _validate_inputs(image, body_mask)

    if threshold_raw is None:
        threshold_raw = estimate_air_threshold_from_external_air(image, body_mask)

    smooth = gaussian(image, sigma=sigma_large, preserve_range=True)
    if threshold_smooth is None:
        threshold_smooth = estimate_air_threshold_from_external_air(smooth, body_mask)

    candidate = body_mask & (image <= threshold_raw) & (smooth <= threshold_smooth)
    return _postprocess_internal_air_mask(
        candidate,
        min_region_size=min_region_size,
        hole_size=hole_size,
        closing_radius=1,
    )


ALGORITHMS: dict[str, Callable[..., NDArrayBool]] = {
    "threshold_external_reference": algorithm_threshold_external_reference,
    "thick_region_filter": algorithm_thick_region_filter,
    "persistent_dark_after_smoothing": algorithm_persistent_dark_after_smoothing,
}


def segment_sinus_and_airways(
    image: np.ndarray | Any,
    body_mask: np.ndarray | Any,
    *,
    method: str = "persistent_dark_after_smoothing",
    vtk_array_name_image: str | None = None,
    vtk_array_name_mask: str | None = None,
    **method_kwargs: Any,
) -> NDArrayBool:
    """Final segmentation entrypoint.

    Parameters
    ----------
    image:
        3D numpy array or vtkImageData.
    body_mask:
        3D numpy array or vtkImageData, where 1 means inside body.
    method:
        One of keys in `ALGORITHMS`.
    method_kwargs:
        Extra parameters forwarded to the selected method.
    """
    image_np = _as_numpy_volume(image, vtk_array_name=vtk_array_name_image)
    body_mask_np = _as_numpy_volume(body_mask, vtk_array_name=vtk_array_name_mask)
    if method not in ALGORITHMS:
        valid = ", ".join(sorted(ALGORITHMS))
        raise ValueError(f"Unknown method '{method}'. Valid methods: {valid}")
    return ALGORITHMS[method](image_np, body_mask_np, **method_kwargs)


def segmentation_metrics(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict[str, float]:
    """Compute binary segmentation metrics."""
    pred = np.asarray(pred_mask) > 0
    true = np.asarray(true_mask) > 0
    if pred.shape != true.shape:
        raise ValueError("pred_mask and true_mask must have the same shape.")

    tp = int(np.sum(pred & true))
    fp = int(np.sum(pred & ~true))
    fn = int(np.sum(~pred & true))
    tn = int(np.sum(~pred & ~true))

    eps = 1e-8
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def evaluate_algorithm_on_cases(
    cases: list[SinusDetectionCase | tuple[np.ndarray, np.ndarray, np.ndarray] | dict[str, Any]],
    *,
    method: str = "persistent_dark_after_smoothing",
    method_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one method on many cases and summarize metrics."""
    if method_kwargs is None:
        method_kwargs = {}

    per_case: list[dict[str, Any]] = []
    for idx, item in enumerate(cases):
        if isinstance(item, SinusDetectionCase):
            case = item
        elif isinstance(item, dict):
            case = SinusDetectionCase(
                image=item["image"],
                body_mask=item["body_mask"],
                manual_mask=item["manual_mask"],
                case_id=item.get("case_id", f"case_{idx:03d}"),
            )
        else:
            image, body_mask, manual_mask = item
            case = SinusDetectionCase(
                image=image,
                body_mask=body_mask,
                manual_mask=manual_mask,
                case_id=f"case_{idx:03d}",
            )

        pred = segment_sinus_and_airways(
            case.image, case.body_mask, method=method, **method_kwargs
        )
        metrics = segmentation_metrics(pred, case.manual_mask)
        metrics["case_id"] = case.case_id
        metrics["pred_voxels"] = float(np.sum(pred))
        metrics["true_voxels"] = float(np.sum(np.asarray(case.manual_mask) > 0))
        per_case.append(metrics)

    if not per_case:
        return {"method": method, "per_case": [], "summary": {}}

    metric_names = ["dice", "iou", "precision", "recall", "specificity"]
    summary: dict[str, float] = {}
    for key in metric_names:
        vals = np.array([row[key] for row in per_case], dtype=float)
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals))
        summary[f"{key}_median"] = float(np.median(vals))

    return {"method": method, "per_case": per_case, "summary": summary}


def benchmark_algorithms(
    cases: list[SinusDetectionCase | tuple[np.ndarray, np.ndarray, np.ndarray] | dict[str, Any]],
    methods: list[str] | None = None,
    method_kwargs_map: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate several methods and return full benchmark results."""
    if methods is None:
        methods = list(ALGORITHMS.keys())
    if method_kwargs_map is None:
        method_kwargs_map = {}

    results: dict[str, dict[str, Any]] = {}
    for method in methods:
        results[method] = evaluate_algorithm_on_cases(
            cases,
            method=method,
            method_kwargs=method_kwargs_map.get(method, {}),
        )
    return results


def _expand_param_grid(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Expand a dictionary-based parameter grid into list of combinations."""
    if not param_grid:
        return [{}]
    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]
    combos: list[dict[str, Any]] = []
    for vals in product(*value_lists):
        combos.append(dict(zip(keys, vals)))
    return combos


def grid_search_method_on_cases(
    cases: list[SinusDetectionCase | tuple[np.ndarray, np.ndarray, np.ndarray] | dict[str, Any]],
    *,
    method: str,
    param_grid: dict[str, list[Any]],
    fixed_kwargs: dict[str, Any] | None = None,
    objective: str = "dice_mean",
    top_k: int = 5,
) -> dict[str, Any]:
    """Grid-search one method's parameters against manual masks.

    Parameters
    ----------
    cases:
        List of benchmark cases accepted by `evaluate_algorithm_on_cases`.
    method:
        Method name from `ALGORITHMS`.
    param_grid:
        Dict of parameter name to list of candidate values.
    fixed_kwargs:
        Extra method kwargs held constant during search.
    objective:
        Summary metric to maximize (e.g. "dice_mean", "iou_mean").
    top_k:
        Number of highest-ranked settings to keep in output.
    """
    if fixed_kwargs is None:
        fixed_kwargs = {}
    combinations = _expand_param_grid(param_grid)

    ranked_results: list[dict[str, Any]] = []
    for params in combinations:
        all_kwargs = dict(fixed_kwargs)
        all_kwargs.update(params)
        result = evaluate_algorithm_on_cases(
            cases,
            method=method,
            method_kwargs=all_kwargs,
        )
        score = float(result.get("summary", {}).get(objective, float("nan")))
        ranked_results.append(
            {
                "method": method,
                "params": all_kwargs,
                "objective": objective,
                "score": score,
                "summary": result.get("summary", {}),
            }
        )

    ranked_results.sort(key=lambda x: x["score"], reverse=True)
    best = ranked_results[0] if ranked_results else None
    return {
        "method": method,
        "objective": objective,
        "n_evaluated": len(ranked_results),
        "best": best,
        "top_results": ranked_results[: max(1, top_k)],
        "all_results": ranked_results,
    }


def grid_search_algorithms(
    cases: list[SinusDetectionCase | tuple[np.ndarray, np.ndarray, np.ndarray] | dict[str, Any]],
    *,
    method_param_grids: dict[str, dict[str, list[Any]]],
    fixed_kwargs_map: dict[str, dict[str, Any]] | None = None,
    objective: str = "dice_mean",
    top_k_per_method: int = 3,
) -> dict[str, Any]:
    """Grid-search multiple methods and rank methods by best objective score."""
    if fixed_kwargs_map is None:
        fixed_kwargs_map = {}

    per_method: dict[str, Any] = {}
    for method, grid in method_param_grids.items():
        per_method[method] = grid_search_method_on_cases(
            cases,
            method=method,
            param_grid=grid,
            fixed_kwargs=fixed_kwargs_map.get(method, {}),
            objective=objective,
            top_k=top_k_per_method,
        )

    ranking: list[dict[str, Any]] = []
    for method, result in per_method.items():
        best = result.get("best")
        if best is None:
            continue
        ranking.append(
            {
                "method": method,
                "best_score": best["score"],
                "best_params": best["params"],
                "best_summary": best["summary"],
            }
        )
    ranking.sort(key=lambda x: x["best_score"], reverse=True)

    return {
        "objective": objective,
        "per_method": per_method,
        "method_ranking": ranking,
        "best_overall": ranking[0] if ranking else None,
    }


def get_largest_components(mask: np.ndarray, n_components: int = 5) -> NDArrayBool:
    """Keep only the n largest connected components from a binary mask."""
    bw = np.asarray(mask) > 0
    if not np.any(bw):
        return bw

    lbl = label(bw, connectivity=1)
    comp_ids, counts = np.unique(lbl[lbl > 0], return_counts=True)
    if comp_ids.size == 0:
        return np.zeros_like(bw, dtype=bool)
    order = np.argsort(counts)[::-1]
    keep = comp_ids[order[:n_components]]
    return np.isin(lbl, keep)
