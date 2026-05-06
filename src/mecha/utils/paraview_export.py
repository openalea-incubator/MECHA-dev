#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#       mecha.utils.paraview_export
#
#       File author(s):
#           Adrien Heymans
#
#       Copyright © by UCLouvain
#       Distributed under the LGPL License.
#       See accompanying file LICENSE.txt or copy at
#           https://www.gnu.org/licenses/lgpl-3.0.en.html
#
# -----------------------------------------------------------------------
"""
Paraview / VTK Export for MECHA Hydraulic Networks
====================================================

Exports a solved MECHA anatomy to a set of Legacy ASCII `.vtk` files readable
by ParaView.  Each geometric "layer" is written to a separate file so that
display properties (colour, opacity, glyph size …) can be controlled
independently inside ParaView.

Generated files
---------------
``<prefix>_cells.vtk``
    One VTK polygon per cell, extruded along Z to represent wall thickness.
    Point data: ``water_potential`` (hPa), ``psi_p`` (hPa), ``os`` (hPa).
    Cell data : ``cell_type`` (int cgroup), ``rank`` (int).

``<prefix>_walls.vtk``
    One quad polygon per wall segment, extruded in Z.
    Cell data : ``K`` (cm³ hPa⁻¹ d⁻¹), ``Q`` (cm³ d⁻¹), ``is_border``,
    ``is_aerenchyma``.

``<prefix>_membranes.vtk``
    One flat quad rectangle between each wall midpoint and connected cell
    centroid.  Cell data : ``K``, ``Q``, ``km``, ``kaqp``.

``<prefix>_plasmodesmata.vtk``
    One line-segment (or thin cylinder approximated as a VTK_LINE) per PD
    connection between two cell symplastic nodes.
    Cell data: ``K``, ``Q``, ``kpl``.

Usage
-----
::

    from mecha.utils.paraview_export import export_to_vtk
    export_to_vtk(mecha_obj, prefix="results/my_sim",
                  maturity_idx=0, scenario_idx="standard water flow",
                  extrude_z=5.0)

Or via the unified ``visualize()`` dispatcher::

    from mecha.utils.visu import visualize
    visualize(mecha_obj, visu_type='paraview',
              prefix='results/my_sim', extrude_z=5.0)

Notes
-----
* All spatial coordinates are in **µm** (same convention as MECHA).
* The ``extrude_z`` parameter sets the thickness of 2-D objects when lifted
  into 3-D (default 5 µm, i.e. typical cell-wall half-thickness).
* VTK Legacy ASCII format is used for maximum compatibility — no additional
  VTK Python binding is required.
* Plasmodesmata radius (``pd_radius``) controls the cylinder approximation
  written in the ``<prefix>_plasmodesmata.vtk`` comment header; the actual
  geometry written is a VTK_LINE because ParaView's *Tube* filter is far
  more efficient for rendering many cylinders than embedding them as polyhedra.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CGROUP_NAME: Dict[int, str] = {
    1: "exodermis",
    2: "epidermis",
    3: "endodermis",
    4: "cortex",
    5: "stele",
    11: "phloem",
    12: "companion",
    13: "xylem",
    16: "pericycle",
}


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _vtk_header(title: str) -> str:
    return (
        "# vtk DataFile Version 3.0\n"
        f"{title}\n"
        "ASCII\n"
        "DATASET POLYDATA\n"
    )


def _write_points(f, pts: List[Tuple[float, float, float]]) -> None:
    f.write(f"POINTS {len(pts)} float\n")
    for x, y, z in pts:
        f.write(f"{x:.4f} {y:.4f} {z:.4f}\n")


def _write_polygons(f, polygons: List[List[int]]) -> None:
    total = sum(len(p) + 1 for p in polygons)
    f.write(f"POLYGONS {len(polygons)} {total}\n")
    for poly in polygons:
        f.write(f"{len(poly)} " + " ".join(str(v) for v in poly) + "\n")


def _write_lines(f, lines: List[Tuple[int, int]]) -> None:
    f.write(f"LINES {len(lines)} {3 * len(lines)}\n")
    for a, b in lines:
        f.write(f"2 {a} {b}\n")


def _write_cell_data_header(f, n: int) -> None:
    f.write(f"\nCELL_DATA {n}\n")


def _write_point_data_header(f, n: int) -> None:
    f.write(f"\nPOINT_DATA {n}\n")


def _write_scalar(f, name: str, values: List[float], dtype: str = "float") -> None:
    f.write(f"SCALARS {name} {dtype} 1\n")
    f.write("LOOKUP_TABLE default\n")
    for v in values:
        f.write(f"{v:.6g}\n")


def _write_vector(f, name: str, vectors: List[Tuple[float, float, float]]) -> None:
    f.write(f"VECTORS {name} float\n")
    for vx, vy, vz in vectors:
        f.write(f"{vx:.6g} {vy:.6g} {vz:.6g}\n")


def _safe(v, default=0.0) -> float:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    return float(v)


# ---------------------------------------------------------------------------
# Solution extraction helper
# ---------------------------------------------------------------------------

def _get_solution(obj: Any, maturity_idx: int = 0,
                  scenario_idx: str = "standard water flow") -> Optional[np.ndarray]:
    """Return the flat solution array for the requested maturity/scenario."""
    if hasattr(obj, "results") and obj.results:
        for res in obj.results:
            if (res.get("maturity stage") == maturity_idx
                    and res.get("scenario") == scenario_idx):
                return np.asarray(res["solution"]).ravel()
    if hasattr(obj, "solution") and obj.solution is not None:
        return np.asarray(obj.solution).ravel()
    return None


def _node_psi(sol: Optional[np.ndarray], node_id: int,
              indice: Dict[int, int]) -> float:
    """Return water potential (hPa) for *node_id* from the solution vector."""
    if sol is None:
        return 0.0
    idx = indice.get(node_id)
    if idx is None or idx >= len(sol):
        return 0.0
    return float(sol[idx])


# ---------------------------------------------------------------------------
# 1. Cell extrusion  →  _cells.vtk
# ---------------------------------------------------------------------------

def _reconstruct_polygon_from_walls(cell) -> Optional[List[Tuple[float, float]]]:
    """Return an ordered coordinate list for a cell whose Shapely polygon is missing.

    Uses the midpoints of the connected walls as vertices and returns their
    convex hull.  Returns ``None`` if fewer than 3 wall midpoints are available.
    """
    from shapely.geometry import MultiPoint

    if not cell.walls:
        return None
    pts = [(w.x, w.y) for w in cell.walls]
    if len(pts) < 3:
        return None
    hull = MultiPoint(pts).convex_hull
    if hull.geom_type != 'Polygon' or hull.is_empty:
        return None
    return list(hull.exterior.coords)[:-1]  # drop closing duplicate


def _export_cells(
    obj: Any,
    filepath: str,
    sol: Optional[np.ndarray],
    indice: Dict[int, int],
    extrude_z: float,
) -> None:
    """Export cell polygons as extruded prisms to *filepath*."""

    cm = obj.network.cell_manager
    if not cm:
        return

    points: List[Tuple[float, float, float]] = []
    polygons: List[List[int]] = []

    # Per-cell data arrays
    water_potentials: List[float] = []
    cgroups: List[int] = []
    ranks: List[int] = []

    for cell in cm:
        # 1) Try stored Shapely polygon
        poly = cell.polygon
        if poly is not None and not poly.is_empty:
            coords = list(poly.exterior.coords)[:-1]  # drop closing duplicate
        else:
            # 2) Reconstruct from connected wall midpoints (GRANAP path)
            coords = _reconstruct_polygon_from_walls(cell)
            if coords is None:
                continue  # no geometry available

        n = len(coords)
        if n < 3:
            continue

        base_idx = len(points)
        # Bottom ring (z=0)
        for x, y in coords:
            points.append((x, y, 0.0))
        # Top ring (z=extrude_z)
        for x, y in coords:
            points.append((x, y, extrude_z))

        # Bottom face
        polygons.append(list(range(base_idx, base_idx + n)))
        # Top face
        polygons.append(list(range(base_idx + n, base_idx + 2 * n)))
        # Side quads
        for i in range(n):
            j = (i + 1) % n
            polygons.append([
                base_idx + i,
                base_idx + j,
                base_idx + n + j,
                base_idx + n + i,
            ])

        n_faces = 2 + n  # bottom + top + sides
        psi_val = _node_psi(sol, cell.node_id, indice)
        for _ in range(n_faces):
            water_potentials.append(psi_val)
            cgroups.append(cell.cgroup)
            ranks.append(cell.rank)

    _ensure_dir(filepath)
    with open(filepath, "w") as f:
        f.write(_vtk_header("MECHA cell cross-sections"))
        _write_points(f, points)
        _write_polygons(f, polygons)
        _write_cell_data_header(f, len(polygons))
        _write_scalar(f, "water_potential", water_potentials)
        _write_scalar(f, "cgroup", [float(v) for v in cgroups])
        _write_scalar(f, "rank", [float(v) for v in ranks])

    print(f"[paraview_export] Cells → {filepath}  ({len(polygons)} polygons)")


# ---------------------------------------------------------------------------
# 2. Thick wall geometry  →  _walls.vtk
# ---------------------------------------------------------------------------

def _oriented_wall_quad(
    x1: float, y1: float,
    x2: float, y2: float,
    half_thickness: float,
    z0: float,
    z1: float,
) -> Tuple[List[Tuple[float, float, float]], List[List[int]]]:
    """Return (points, quads) for an oriented wall strip between two endpoints.

    The strip lies in the XY plane (z = z0), is extruded to z = z1, and has
    total thickness = 2 × ``half_thickness`` in the wall-normal direction.

    Vertex layout (0-7)::

        3──────────────2   z = z1  (top)
        |              |   |
        0──────────────1   z = z0  (bottom)
       pt1            pt2

    with an inner and outer pair displaced ± ``half_thickness`` along the
    wall normal.
    """
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-9:
        # Degenerate wall — produce a tiny axis-aligned quad
        dx, dy, length = half_thickness, 0.0, half_thickness

    # Unit normal perpendicular to the wall (rotated 90° CCW)
    nx_unit = -dy / length
    ny_unit = dx / length

    ox = nx_unit * half_thickness
    oy = ny_unit * half_thickness

    # 8 vertices: 4 on each Z level
    pts = [
        (x1 - ox, y1 - oy, z0),  # 0  bottom, pt1-side, inner
        (x2 - ox, y2 - oy, z0),  # 1  bottom, pt2-side, inner
        (x2 + ox, y2 + oy, z0),  # 2  bottom, pt2-side, outer
        (x1 + ox, y1 + oy, z0),  # 3  bottom, pt1-side, outer
        (x1 - ox, y1 - oy, z1),  # 4  top,    pt1-side, inner
        (x2 - ox, y2 - oy, z1),  # 5  top,    pt2-side, inner
        (x2 + ox, y2 + oy, z1),  # 6  top,    pt2-side, outer
        (x1 + ox, y1 + oy, z1),  # 7  top,    pt1-side, outer
    ]
    quads = [
        [0, 1, 2, 3],   # bottom face (z = z0)
        [4, 5, 6, 7],   # top    face (z = z1)
        [0, 1, 5, 4],   # inner  side
        [2, 3, 7, 6],   # outer  side
        [0, 3, 7, 4],   # pt1    end cap
        [1, 2, 6, 5],   # pt2    end cap
    ]
    return pts, quads


def _export_walls(
    obj: Any,
    filepath: str,
    extrude_z: float,
) -> None:
    """Export wall segments as oriented extruded quads to *filepath*.

    Uses ``network.junction_positions[wall_id]`` (the two real endpoint
    positions of each wall segment) when available, and falls back to an
    approximation centred on the wall midpoint when it is not.
    """

    cm = obj.network.cell_manager
    if not cm:
        return

    # Pre-fetch junction endpoint positions (wall_id → [x1, y1, x2, y2])
    junction_positions = getattr(obj.network, 'junction_positions', {})

    points: List[Tuple[float, float, float]] = []
    polygons: List[List[int]] = []
    K_vals: List[float] = []
    Q_vals: List[float] = []
    is_border_vals: List[float] = []
    is_aero_vals: List[float] = []

    for wall in cm.walls:
        half_thick = max(wall.thickness / 2.0, 0.5)
        z0, z1 = 0.0, extrude_z

        jp = junction_positions.get(wall.node_id)
        if jp is not None and len(jp) >= 4:
            # Use the real endpoints for a properly oriented quad
            x1, y1, x2, y2 = jp[0], jp[1], jp[2], jp[3]
        else:
            # Fallback: approximate by offsetting ± half-length along x
            half_len = max(wall.length / 2.0, 1.0)
            x1, y1 = wall.x - half_len, wall.y
            x2, y2 = wall.x + half_len, wall.y

        local_pts, local_quads = _oriented_wall_quad(
            x1, y1, x2, y2, half_thick, z0, z1
        )
        base = len(points)
        points.extend(local_pts)
        for q in local_quads:
            polygons.append([base + v for v in q])

        K = _safe(wall.kw)
        Q = _safe(wall.Q)
        for _ in local_quads:
            K_vals.append(K)
            Q_vals.append(Q)
            is_border_vals.append(float(wall.is_border))
            is_aero_vals.append(float(wall.is_aerenchyma))

    _ensure_dir(filepath)
    with open(filepath, "w") as f:
        f.write(_vtk_header("MECHA cell walls (thick)"))
        _write_points(f, points)
        _write_polygons(f, polygons)
        _write_cell_data_header(f, len(polygons))
        _write_scalar(f, "K_wall", K_vals)
        _write_scalar(f, "Q_wall", Q_vals)
        _write_scalar(f, "is_border", is_border_vals)
        _write_scalar(f, "is_aerenchyma", is_aero_vals)

    print(f"[paraview_export] Walls → {filepath}  ({len(polygons)} polygons)")


# ---------------------------------------------------------------------------
# 3. Membrane rectangles  →  _membranes.vtk
# ---------------------------------------------------------------------------

def _export_membranes(
    obj: Any,
    filepath: str,
    sol: Optional[np.ndarray],
    indice: Dict[int, int],
    extrude_z: float,
) -> None:
    """Export membrane connections as flat quads between wall and cell."""

    cm = obj.network.cell_manager
    if not cm:
        return

    points: List[Tuple[float, float, float]] = []
    polygons: List[List[int]] = []
    K_vals: List[float] = []
    Q_vals: List[float] = []
    km_vals: List[float] = []
    kaqp_vals: List[float] = []

    # Graph edge attributes carry K and Q after solve
    graph = obj.network.graph

    for mb in cm.membranes:
        wx, wy = mb.wall.x, mb.wall.y
        cx, cy = mb.cell.x, mb.cell.y

        # Midpoint at z=0 and z=extrude_z
        mid_z = extrude_z / 2.0

        base = len(points)
        # Rectangle: 2 points on wall side, 2 on cell side, at two z levels
        points += [
            (wx, wy, 0.0),
            (cx, cy, 0.0),
            (cx, cy, extrude_z),
            (wx, wy, extrude_z),
        ]
        polygons.append([base, base + 1, base + 2, base + 3])

        # Retrieve K/Q from graph edge (wall_node_id ↔ cell_node_id)
        edge_data = graph.edges.get(
            (mb.wall.node_id, mb.cell.node_id),
            graph.edges.get((mb.cell.node_id, mb.wall.node_id), {}),
        )
        K_vals.append(_safe(edge_data.get("K", mb.K_computed)))
        Q_vals.append(_safe(edge_data.get("Q")))
        km_vals.append(_safe(mb.km))
        kaqp_vals.append(_safe(mb.kaqp))

    _ensure_dir(filepath)
    with open(filepath, "w") as f:
        f.write(_vtk_header("MECHA membrane connections"))
        _write_points(f, points)
        _write_polygons(f, polygons)
        _write_cell_data_header(f, len(polygons))
        _write_scalar(f, "K_membrane", K_vals)
        _write_scalar(f, "Q_membrane", Q_vals)
        _write_scalar(f, "km", km_vals)
        _write_scalar(f, "kaqp", kaqp_vals)

    print(f"[paraview_export] Membranes → {filepath}  ({len(polygons)} quads)")


# ---------------------------------------------------------------------------
# 4. Plasmodesmata lines  →  _plasmodesmata.vtk
# ---------------------------------------------------------------------------

def _export_plasmodesmata(
    obj: Any,
    filepath: str,
    sol: Optional[np.ndarray],
    indice: Dict[int, int],
    pd_radius: float = 0.05,
) -> None:
    """
    Export plasmodesmata connections as VTK_LINE segments.

    Each line runs from cell_i centroid to cell_j centroid at z = 0.
    Use ParaView's *Tube* filter (radius = ``pd_radius`` µm) to render
    cylinders.  ``pd_radius`` is embedded in the file comment header.

    Parameters
    ----------
    pd_radius : float
        Cylinder radius (µm) — written as a comment for reference; apply via
        the Tube filter in ParaView.
    """

    cm = obj.network.cell_manager
    if not cm:
        return

    graph = obj.network.graph

    points: List[Tuple[float, float, float]] = []
    lines: List[Tuple[int, int]] = []
    K_vals: List[float] = []
    Q_vals: List[float] = []
    kpl_vals: List[float] = []

    # Build a node_id → point index map
    node_to_pt: Dict[int, int] = {}

    for pd in cm.plasmodesmata:
        ci, cj = pd.cell_i, pd.cell_j

        # Register cell_i
        if ci.node_id not in node_to_pt:
            node_to_pt[ci.node_id] = len(points)
            points.append((ci.x, ci.y, 0.0))

        # Register cell_j
        if cj.node_id not in node_to_pt:
            node_to_pt[cj.node_id] = len(points)
            points.append((cj.x, cj.y, 0.0))

        lines.append((node_to_pt[ci.node_id], node_to_pt[cj.node_id]))

        # K / Q from graph edge
        edge_data = graph.edges.get(
            (ci.node_id, cj.node_id),
            graph.edges.get((cj.node_id, ci.node_id), {}),
        )
        K_vals.append(_safe(edge_data.get("K")))
        Q_vals.append(_safe(edge_data.get("Q")))
        kpl_vals.append(_safe(pd.kpl))

    if not lines:
        print("[paraview_export] No plasmodesmata — skipping.")
        return

    _ensure_dir(filepath)
    with open(filepath, "w") as f:
        f.write(
            "# vtk DataFile Version 3.0\n"
            f"MECHA plasmodesmata  pd_radius={pd_radius:.4f} um\n"
            "ASCII\n"
            "DATASET POLYDATA\n"
        )
        _write_points(f, points)
        _write_lines(f, lines)
        _write_cell_data_header(f, len(lines))
        _write_scalar(f, "K_pd", K_vals)
        _write_scalar(f, "Q_pd", Q_vals)
        _write_scalar(f, "kpl", kpl_vals)

    print(f"[paraview_export] Plasmodesmata → {filepath}  ({len(lines)} lines)")


# ---------------------------------------------------------------------------
# 5. Flow vectors on edges  →  _flow_vectors.vtk
# ---------------------------------------------------------------------------

def _export_flow_vectors(
    obj: Any,
    filepath: str,
    sol: Optional[np.ndarray],
    indice: Dict[int, int],
) -> None:
    """
    Export one point per edge midpoint with a flow vector (Q × radial unit).

    The vector magnitude encodes |Q| and the direction is from the upstream
    node to the downstream node (sign of Q determines direction).
    Useful for the *Glyph* filter in ParaView.
    """

    graph = obj.network.graph
    pos = dict(graph.nodes(data="position", default=(0.0, 0.0)))

    points: List[Tuple[float, float, float]] = []
    vectors: List[Tuple[float, float, float]] = []
    K_vals: List[float] = []
    Q_vals: List[float] = []
    path_ids: List[float] = []

    _PATH_ID = {"wall": 0.0, "membrane": 1.0, "plasmodesmata": 2.0}

    for u, v, eattr in graph.edges(data=True):
        K = eattr.get("K")
        Q = eattr.get("Q")
        path = eattr.get("path", "wall")

        if K is None:
            continue

        pu = pos.get(u, (0.0, 0.0))
        pv = pos.get(v, (0.0, 0.0))
        mx = (pu[0] + pv[0]) / 2.0
        my = (pu[1] + pv[1]) / 2.0

        dx = pv[0] - pu[0]
        dy = pv[1] - pu[1]
        norm = math.hypot(dx, dy)

        q_val = _safe(Q)
        if norm > 1e-12:
            direction = 1.0 if q_val >= 0 else -1.0
            vx = direction * abs(q_val) * dx / norm
            vy = direction * abs(q_val) * dy / norm
        else:
            vx, vy = 0.0, 0.0

        points.append((mx, my, 0.0))
        vectors.append((vx, vy, 0.0))
        K_vals.append(_safe(K))
        Q_vals.append(q_val)
        path_ids.append(_PATH_ID.get(path, -1.0))

    if not points:
        print("[paraview_export] No flow data — skipping flow vectors.")
        return

    _ensure_dir(filepath)
    with open(filepath, "w") as f:
        f.write(
            "# vtk DataFile Version 3.0\n"
            "MECHA edge flow vectors\n"
            "ASCII\n"
            "DATASET POLYDATA\n"
        )
        _write_points(f, points)
        # Write as vertices so the Glyph filter works
        f.write(f"VERTICES {len(points)} {2 * len(points)}\n")
        for i in range(len(points)):
            f.write(f"1 {i}\n")
        _write_point_data_header(f, len(points))
        _write_vector(f, "flow_Q", vectors)
        _write_scalar(f, "K", K_vals)
        _write_scalar(f, "Q_magnitude", [abs(q) for q in Q_vals])
        _write_scalar(f, "path_id", path_ids)

    print(f"[paraview_export] Flow vectors → {filepath}  ({len(points)} points)")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_to_vtk(
    obj: Any,
    prefix: str = "mecha_export",
    maturity_idx: int = 0,
    scenario_idx: str = "standard water flow",
    extrude_z: float = 5.0,
    pd_radius: float = 0.05,
    export_cells: bool = True,
    export_walls: bool = True,
    export_membranes: bool = True,
    export_plasmodesmata: bool = True,
    export_flow_vectors: bool = True,
) -> Dict[str, str]:
    """Export a MECHA simulation result to a set of ParaView `.vtk` files.

    Parameters
    ----------
    obj : Mecha
        A fully solved ``Mecha`` instance.  The network must have a populated
        ``cell_manager`` with membranes and plasmodesmata synced.
    prefix : str
        Path prefix for output files.  Directories are created automatically.
        Example: ``"results/my_sim"`` produces ``results/my_sim_cells.vtk``, etc.
    maturity_idx : int
        Maturity stage index (0-based) to extract the solution from.
    scenario_idx : str
        Scenario key (matches ``res['scenario']`` in ``obj.results``).
    extrude_z : float
        Depth (µm) by which 2-D polygons are extruded in the Z direction.
        Represents the approximate organ section thickness used for display.
    pd_radius : float
        Plasmodesmata cylinder radius (µm) — embedded in the VTK header as a
        comment.  Apply ParaView's *Tube* filter with this radius.
    export_cells, export_walls, export_membranes, export_plasmodesmata,
    export_flow_vectors : bool
        Toggle individual output files.

    Returns
    -------
    dict
        Mapping of output type → file path for each file written.
    """
    if not hasattr(obj, "network") or obj.network is None:
        raise ValueError("obj must be a Mecha instance with a populated network.")
    if obj.network.cell_manager is None:
        raise ValueError("network.cell_manager is None — call build_network first.")

    sol = _get_solution(obj, maturity_idx=maturity_idx, scenario_idx=scenario_idx)
    if sol is None:
        print(
            f"[paraview_export] WARNING: No solution found for maturity={maturity_idx}"
            f" / scenario='{scenario_idx}'.  Scalar fields will be zero."
        )
    indice: Dict[int, int] = getattr(obj, "indice", {})

    written: Dict[str, str] = {}

    if export_cells:
        fp = f"{prefix}_cells.vtk"
        _export_cells(obj, fp, sol, indice, extrude_z)
        written["cells"] = fp

    if export_walls:
        fp = f"{prefix}_walls.vtk"
        _export_walls(obj, fp, extrude_z)
        written["walls"] = fp

    if export_membranes:
        fp = f"{prefix}_membranes.vtk"
        _export_membranes(obj, fp, sol, indice, extrude_z)
        written["membranes"] = fp

    if export_plasmodesmata:
        fp = f"{prefix}_plasmodesmata.vtk"
        _export_plasmodesmata(obj, fp, sol, indice, pd_radius=pd_radius)
        written["plasmodesmata"] = fp

    if export_flow_vectors:
        fp = f"{prefix}_flow_vectors.vtk"
        _export_flow_vectors(obj, fp, sol, indice)
        written["flow_vectors"] = fp

    print(
        f"\n[paraview_export] Done — {len(written)} file(s) written with prefix '{prefix}'."
    )
    return written
