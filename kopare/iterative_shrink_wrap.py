"""Iterative adaptive shrinkwrap algorithm for VTK polydata surfaces.

Algorithm overview
------------------
1. A wrapping mesh (default: sphere enclosing the target) is initialised.
2. Every vertex of the wrapping mesh is projected to its nearest point on the
   target surface (initial shrinkwrap step).
3. The following refinement loop then runs up to *max_iterations* times:

   a. Every unique edge whose length exceeds *max_edge_length* is examined.
   b. The edge midpoint is projected onto the target surface.
   c. If the displacement from midpoint to its projection is larger than
      *movement_threshold*, the edge is subdivided and the new vertex is
      placed on the surface.
   d. Optionally, all vertices are re-projected to the surface (full
      shrinkwrap pass).
   e. If no splits occurred the loop terminates early (converged).

4. The final mesh is cleaned and returned with recomputed normals.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import vtk
from vtk.util import numpy_support

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_cell_locator(surface: vtk.vtkPolyData) -> vtk.vtkCellLocator:
    """Build and return a vtkCellLocator for *surface*."""
    locator = vtk.vtkCellLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()
    return locator


def _project_to_surface(
    point: np.ndarray,
    locator: vtk.vtkCellLocator,
) -> tuple[np.ndarray, float]:
    """Return *(closest_point, distance)* for *point* projected onto the surface."""
    closest = [0.0, 0.0, 0.0]
    cell_id = vtk.reference(0)
    sub_id = vtk.reference(0)
    dist2 = vtk.reference(0.0)
    locator.FindClosestPoint(point.tolist(), closest, cell_id, sub_id, dist2)
    return np.array(closest), float(dist2) ** 0.5


def _polydata_to_arrays(
    poly: vtk.vtkPolyData,
) -> tuple[list[np.ndarray], list[list[int]]]:
    """Extract points and triangles from *poly* as plain Python lists."""
    pts = [np.array(poly.GetPoint(i), dtype=float)
           for i in range(poly.GetNumberOfPoints())]
    tris: list[list[int]] = []
    for i in range(poly.GetNumberOfCells()):
        cell = poly.GetCell(i)
        if cell.GetNumberOfPoints() == 3:
            tris.append(
                [cell.GetPointId(0), cell.GetPointId(1), cell.GetPointId(2)]
            )
    return pts, tris


def _arrays_to_polydata(
    pts: list[np.ndarray],
    tris: list[list[int] | None],
) -> vtk.vtkPolyData:
    """Build a vtkPolyData from *pts* and *tris* (None entries are skipped)."""
    active_tris = [t for t in tris if t is not None]

    pts_np = np.array(pts, dtype=np.float64)
    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(
        numpy_support.numpy_to_vtk(pts_np, deep=True, array_type=vtk.VTK_DOUBLE)
    )

    cell_array = vtk.vtkCellArray()
    for tri in active_tris:
        cell_array.InsertNextCell(3)
        cell_array.InsertCellPoint(tri[0])
        cell_array.InsertCellPoint(tri[1])
        cell_array.InsertCellPoint(tri[2])

    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_pts)
    poly.SetPolys(cell_array)
    return poly


def _build_pt_to_tris(
    tris: list[list[int] | None],
    n_pts: int,
) -> dict[int, set[int]]:
    """Return a mapping from point index → set of triangle indices containing it."""
    mapping: dict[int, set[int]] = {i: set() for i in range(n_pts)}
    for tri_idx, tri in enumerate(tris):
        if tri is None:
            continue
        for v in tri:
            mapping[v].add(tri_idx)
    return mapping


def _split_edge(
    pts_list: list[np.ndarray],
    tris_list: list[list[int] | None],
    pt_to_tris: dict[int, set[int]],
    p0: int,
    p1: int,
    new_pt: np.ndarray,
) -> int:
    """Split edge *(p0, p1)* by inserting *new_pt*.

    Every triangle that shares this edge is replaced by two triangles using
    the new midpoint vertex.  Updates *pts_list*, *tris_list*, and
    *pt_to_tris* in-place.

    Returns the index of the newly inserted point.
    """
    new_idx = len(pts_list)
    pts_list.append(new_pt.copy())
    pt_to_tris[new_idx] = set()

    # Triangles that contain both p0 and p1
    shared = pt_to_tris.get(p0, set()) & pt_to_tris.get(p1, set())

    for tri_idx in list(shared):
        tri = tris_list[tri_idx]
        if tri is None:
            continue
        third = next(v for v in tri if v != p0 and v != p1)

        # Mark old triangle as deleted and remove from adjacency
        tris_list[tri_idx] = None
        for v in (p0, p1, third):
            pt_to_tris[v].discard(tri_idx)

        # Insert two replacement triangles
        for new_tri in ([p0, new_idx, third], [p1, new_idx, third]):
            idx = len(tris_list)
            tris_list.append(new_tri)
            for v in new_tri:
                pt_to_tris[v].add(idx)

    return new_idx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def iterative_shrink_wrap(
    target_surface: vtk.vtkPolyData,
    initial_wrap: Optional[vtk.vtkPolyData] = None,
    max_edge_length: float = None,
    movement_threshold: float = None,
    max_iterations: int = 50,
    project_all_points: bool = True,
    sphere_resolution: int = 20,
) -> vtk.vtkPolyData:
    """Iterative adaptive shrinkwrap.

    Wraps *initial_wrap* (default: enclosing sphere) around *target_surface*.
    Long edges are subdivided wherever projecting the midpoint to the surface
    displaces it by more than *movement_threshold*.  The loop repeats until no
    qualifying edges remain or *max_iterations* is exceeded.

    Parameters
    ----------
    target_surface:
        The surface to wrap around.
    initial_wrap:
        Starting wrapping mesh.  Defaults to an enclosing sphere.
    max_edge_length:
        Edges longer than this value (in world units) are candidates for
        subdivision.
    movement_threshold:
        Minimum displacement between the edge midpoint and its surface
        projection that triggers a subdivision.
    max_iterations:
        Maximum number of refinement iterations.
    project_all_points:
        If True, re-project every vertex to the surface at the end of each
        iteration (full shrinkwrap pass).  Keeps the mesh tightly fitted as
        new topology is added.
    sphere_resolution:
        Phi/theta resolution of the default enclosing sphere (only used when
        *initial_wrap* is None).

    Returns
    -------
    vtk.vtkPolyData
        Refined wrapping mesh with recomputed point and cell normals.
    """
    locator = _build_cell_locator(target_surface)

    # --- Default initial wrap: sphere that encloses the target ---
    bounds = target_surface.GetBounds()
    maxBound = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    if initial_wrap is None:
        center = [
            (bounds[0] + bounds[1]) * 0.5,
            (bounds[2] + bounds[3]) * 0.5,
            (bounds[4] + bounds[5]) * 0.5,
        ]
        radius = 0.8 * maxBound
        src = vtk.vtkSphereSource()
        src.SetCenter(center)
        src.SetRadius(radius)
        src.SetPhiResolution(sphere_resolution)
        src.SetThetaResolution(sphere_resolution)
        src.Update()
        initial_wrap = src.GetOutput()

    if max_edge_length is None:
        max_edge_length = 0.01 * maxBound
    if movement_threshold is None:
        movement_threshold = 0.001 * maxBound

    pts_list, tris_raw = _polydata_to_arrays(initial_wrap)
    tris_list: list[list[int] | None] = list(tris_raw)

    # --- Initial projection: shrink all vertices onto the target surface ---
    for i in range(len(pts_list)):
        pts_list[i], _ = _project_to_surface(pts_list[i], locator)

    pt_to_tris = _build_pt_to_tris(tris_list, len(pts_list))

    # --- Iterative adaptive subdivision loop ---
    for iteration in range(max_iterations):
        n_split = 0

        # Collect all unique edges that exceed max_edge_length
        edges_seen: set[tuple[int, int]] = set()

        long_edges_with_len: list[tuple[float, tuple[int, int]]] = []

        for tri in tris_list:
            if tri is None:
                continue
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])):
                key: tuple[int, int] = (min(a, b), max(a, b))
                if key not in edges_seen:
                    edges_seen.add(key)
                    length = np.linalg.norm(pts_list[key[1]] - pts_list[key[0]])
                    if length > max_edge_length:
                        long_edges_with_len.append((length, key))

        # Process longest edges first so coarse structure is resolved before fine
        long_edges_with_len.sort(reverse=True)
        long_edges = [key for _, key in long_edges_with_len]

        for e in long_edges:
            # The edge might have been consumed by an earlier split this iteration
            if not (pt_to_tris.get(e[0], set()) & pt_to_tris.get(e[1], set())):
                continue

            midpoint = (pts_list[e[0]] + pts_list[e[1]]) * 0.5
            closest, dist = _project_to_surface(midpoint, locator)

            if dist > movement_threshold:
                _split_edge(pts_list, tris_list, pt_to_tris, e[0], e[1], closest)
                n_split += 1

        # Optional full shrinkwrap pass: keep all vertices on the surface
        if project_all_points and n_split > 0:
            for i in range(len(pts_list)):
                pts_list[i], _ = _project_to_surface(pts_list[i], locator)

        n_active = sum(1 for t in tris_list if t is not None)
        logger.debug(
            "Iteration %d: %d edge(s) split → %d active triangles",
            iteration + 1,
            n_split,
            n_active,
        )

        if n_split == 0:
            logger.info(
                "iterative_shrink_wrap converged after %d iteration(s) "
                "with %d triangles.",
                iteration + 1,
                n_active,
            )
            break
    else:
        logger.warning(
            "iterative_shrink_wrap reached max_iterations=%d without full "
            "convergence (%d triangles).",
            max_iterations,
            sum(1 for t in tris_list if t is not None),
        )

    # --- Build, clean, and return output mesh ---
    result = _arrays_to_polydata(pts_list, tris_list)

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(result)
    cleaner.Update()

    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputConnection(cleaner.GetOutputPort())
    normals_filter.ComputePointNormalsOn()
    normals_filter.ComputeCellNormalsOn()
    normals_filter.SplittingOff()
    normals_filter.Update()

    return normals_filter.GetOutput()
