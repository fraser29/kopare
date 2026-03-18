
def _ransac_find_plane(
    centroids: np.ndarray,
    cell_normals: np.ndarray,
    active: np.ndarray,
    plane_dist_tol: float,
    cos_tol: float,
    n_iters: int,
    rng: np.random.Generator,
) -> tuple:
    """RANSAC plane fit restricted to *active* cell indices.

    Returns ``(best_inlier_mask, best_normal, best_d)`` where the mask is
    boolean over *all* cells (False for inactive ones).  If no plane is found
    returns an all-False mask.
    """
    active_idx = np.where(active)[0]
    n_active = len(active_idx)
    if n_active < 3:
        return np.zeros(len(centroids), dtype=bool), np.zeros(3), 0.0

    best_mask = np.zeros(len(centroids), dtype=bool)
    best_count = 0
    best_normal = np.zeros(3)
    best_d = 0.0

    for _ in range(n_iters):
        idx = active_idx[rng.choice(n_active, 3, replace=False)]
        p = centroids[idx]
        v1, v2 = p[1] - p[0], p[2] - p[0]
        n_raw = np.cross(v1, v2)
        n_len = np.linalg.norm(n_raw)
        if n_len < 1e-12:
            continue
        n_hat = n_raw / n_len
        d = float(np.dot(n_hat, p[0]))

        dist = np.abs(centroids @ n_hat - d)
        dot_abs = np.abs(cell_normals @ n_hat)
        mask = active & (dist < plane_dist_tol) & (dot_abs > cos_tol)
        count = int(mask.sum())

        if count > best_count:
            best_count = count
            best_mask = mask
            best_normal = n_hat
            best_d = d

    return best_mask, best_normal, best_d


def _refine_plane_pca(
    centroids: np.ndarray,
    cell_normals: np.ndarray,
    cell_areas: np.ndarray,
    active: np.ndarray,
    rough_mask: np.ndarray,
    plane_dist_tol: float,
    cos_tol: float,
    active_area: float,
    bb_min: np.ndarray,
    bb_max: np.ndarray,
    boundary_margin_fraction: float,
    min_flat_area_fraction: float,
) -> tuple:
    """Refine a RANSAC plane via PCA and apply quality checks.

    Returns ``(final_mask, plane_normal, plane_origin, flat_fraction, is_valid)``
    where *is_valid* is True when the plane passes both the area and boundary
    checks.  ``flat_fraction`` is relative to *active_area* (the area of cells
    still in the candidate pool), so each successive cut is judged on a fair
    basis rather than against the shrinking total.
    """
    inlier_pts = centroids[rough_mask]
    if len(inlier_pts) < 3:
        return np.zeros(len(centroids), dtype=bool), np.zeros(3), np.zeros(3), 0.0, False

    centroid_mean = inlier_pts.mean(axis=0)
    cov = np.cov((inlier_pts - centroid_mean).T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    refined_normal = eigenvectors[:, 0]  # smallest eigenvalue → flattest direction
    refined_d = float(np.dot(refined_normal, centroid_mean))

    dist = np.abs(centroids @ refined_normal - refined_d)
    dot_abs = np.abs(cell_normals @ refined_normal)
    final_mask = active & (dist < plane_dist_tol) & (dot_abs > cos_tol)

    # Fraction relative to currently-active area so each pass uses a fair threshold.
    flat_fraction = cell_areas[final_mask].sum() / active_area

    # Boundary check: the plane must sit near an extreme of the bounding box.
    proj_min = float(np.dot(refined_normal, bb_min))
    proj_max = float(np.dot(refined_normal, bb_max))
    margin = boundary_margin_fraction * abs(proj_max - proj_min)
    at_boundary = (abs(refined_d - proj_min) < margin) or (abs(refined_d - proj_max) < margin)

    is_valid = (flat_fraction >= min_flat_area_fraction) and at_boundary
    return final_mask, refined_normal, centroid_mean, flat_fraction, is_valid


def detect_and_label_cut_faces(
    polydata: vtk.vtkPolyData,
    min_flat_area_fraction: float = 0.03,
    normal_angle_tol_deg: float = 15.0,
    plane_dist_tol: float | None = None,
    n_ransac_iters: int = 500,
    boundary_margin_fraction: float = 0.10,
    max_cuts: int = 6,
    array_name: str = "CutFaceLabel",
    random_seed: int = 42,
                            ) -> tuple:

    polydata = vtkfilters.filterTriangulate(polydata)
    centroids, norms = vtkfilters.getPolyDataCenterPtNormal(polydata)

    bounds = polydata.GetBounds()  # (xmin,xmax,ymin,ymax,zmin,zmax)
    diag = np.sqrt(
        (bounds[1] - bounds[0]) ** 2
        + (bounds[3] - bounds[2]) ** 2
        + (bounds[5] - bounds[4]) ** 2
    )
    if plane_dist_tol is None:
        plane_dist_tol = diag * 0.02

    cos_tol = np.cos(np.deg2rad(normal_angle_tol_deg))
    bb_min = np.array([bounds[0], bounds[2], bounds[4]])
    bb_max = np.array([bounds[1], bounds[3], bounds[5]])

    # ------------------------------------------------------------------
    # 4. Sequential RANSAC: find up to max_cuts distinct cut planes
    # ------------------------------------------------------------------
    rng = np.random.default_rng(random_seed)
    labels = np.zeros(n_cells, dtype=np.int32)
    active = np.ones(n_cells, dtype=bool)   # cells still available for fitting
    cuts: list[dict] = []
    consecutive_fails = 0

    for cut_idx in range(1, max_cuts + 1):
        active_area = cell_areas[active].sum()
        if active_area == 0.0:
            break

        rough_mask, _, _ = _ransac_find_plane(
            centroids, cell_normals, active,
            plane_dist_tol, cos_tol, n_ransac_iters, rng,
        )

        if not rough_mask.any():
            break

        final_mask, plane_normal, plane_origin, flat_fraction, is_valid = _refine_plane_pca(
            centroids, cell_normals, cell_areas, active,
            rough_mask, plane_dist_tol, cos_tol,
            active_area, bb_min, bb_max,
            boundary_margin_fraction, min_flat_area_fraction,
        )

        if not is_valid:
            # RANSAC found a plane but it didn't qualify.  Allow a couple of
            # retries in case the sampling landed on a non-cut region by chance,
            # but stop if we keep failing to avoid an infinite search.
            consecutive_fails += 1
            if consecutive_fails >= 2:
                break
            continue

        consecutive_fails = 0
        labels[final_mask] = cut_idx
        active[final_mask] = False   # exclude from subsequent searches

        cuts.append({
            "label": cut_idx,
            "plane_normal": plane_normal,
            "plane_origin": plane_origin,
            "flat_area_fraction": float(flat_fraction),
            "n_cut_cells": int(final_mask.sum()),
        })

    # ------------------------------------------------------------------
    # 5. Build output polydata with CellData label array
    # ------------------------------------------------------------------
    output = vtk.vtkPolyData()
    output.DeepCopy(polydata)

    label_arr = ns.numpy_to_vtk(labels, deep=True)
    label_arr.SetName(array_name)
    output.GetCellData().AddArray(label_arr)

    return output, len(cuts) > 0, cuts
