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
    Point data: ``psi_p`` (hPa), ``psi_os`` (hPa), ``psi_total`` (hPa).
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
                  extrude_z=50.0)

Or via the unified ``visualize()`` dispatcher::

    from mecha.utils.visu import visualize
    visualize(mecha_obj, visu_type='paraview',
              prefix='results/my_sim', extrude_z=50.0)

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
from typing import Any, Dict, List, Optional, Tuple, Union

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

DEFAULT_CELL_WALL_THICKNESS: Dict[str, float] = {
    "epidermis": 2,
    "exodermis": 2,
    "hypodermis": 2,
    "endodermis": 1.5,
    "cortex": 1,
    "mesophyll": 1,
    "parenchyma": 1,
    "stele": 1,
    "pericycle": 1,
    "phloem": 1,
    "xylem": 1.5,
    "protoxylem": 1.5,
    "metaxylem": 2,
    "cambium": 1,
    "duct": 5,
    "guard cell": 2,
    "Strasburger cell": 1,
    "outerwall": 2,
    "air space": 0.001,
    "pore": 0.001,
    "aerenchyma": 0.001,
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

def get_thickness(c_type: str, cell_wall_thickness: Union[float, Dict[str, float]] = DEFAULT_CELL_WALL_THICKNESS)-> float:
    if isinstance(cell_wall_thickness, dict):
        val = cell_wall_thickness.get(c_type, cell_wall_thickness.get("default", 1))
    else:
        val = cell_wall_thickness
    # No conversion, assumed scaling to microns
    return val


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
                  scenario_idx: Union[str,int] = "standard water flow") -> Optional[np.ndarray]:
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
    return _safe(sol[idx])


# ---------------------------------------------------------------------------
# 1. Cell extrusion  →  _cells.vtk
# ---------------------------------------------------------------------------

def _reconstruct_polygon_from_walls(cell) -> Optional[List[Tuple[float, float]]]:
    """Return an ordered coordinate list for a cell whose Shapely polygon is missing.

    Uses the midpoints of the connected walls as vertices and returns their
    convex hull.  Returns ``None`` if fewer than 3 wall midpoints are available.
    """
    polygon = cell.polygon
    if polygon is None or polygon.is_empty:
        return None
    if polygon.geom_type != 'Polygon':
        return None
    return list(polygon.exterior.coords)[:-1]


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
    psi_p_vals: List[float] = []
    psi_os_vals: List[float] = []
    psi_total_vals: List[float] = []
    Q_in_vals: List[float] = []
    Q_out_vals: List[float] = []
    Q_total_vals: List[float] = []
    cgroups: List[float] = []
    ranks: List[float] = []

    graph = obj.network.graph

    for cell in cm:
        # 1) Try stored Shapely polygon
        poly = cell.polygon
        c_type = _CGROUP_NAME.get(cell.cgroup)
        if c_type is None:
            c_type = "default"
        wt = get_thickness(c_type)
        if poly is not None and not poly.is_empty:
            poly = poly.buffer(-wt/2)
            if poly is None or poly.is_empty:
                continue
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
            points.append((x, y, extrude_z/2))

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
        
        # Get potentials from graph nodes if available, fallback to solution
        node_data = graph.nodes[cell.node_id]
        psi_p = _safe(node_data.get('psi_p', _node_psi(sol, cell.node_id, indice)))
        psi_os  = _safe(node_data.get('psi_os', 0.0))
        psi_tot = _safe(node_data.get('psi_total', psi_p + psi_os))
        Q_in = _safe(node_data.get('Q_in', 0.0))
        Q_out = _safe(node_data.get('Q_out', 0.0))
        Q_total = Q_in - Q_out

        for _ in range(n_faces):
            psi_p_vals.append(psi_p)
            psi_os_vals.append(psi_os)
            psi_total_vals.append(psi_tot)
            cgroups.append(_safe(cell.cgroup))
            ranks.append(_safe(cell.rank))
            Q_in_vals.append(Q_in)
            Q_out_vals.append(Q_out)
            Q_total_vals.append(Q_total)

    _ensure_dir(filepath)
    with open(filepath, "w") as f:
        f.write(_vtk_header("MECHA cell cross-sections"))
        _write_points(f, points)
        _write_polygons(f, polygons)
        _write_cell_data_header(f, len(polygons))
        _write_scalar(f, "Psi_total", psi_total_vals)
        _write_scalar(f, "Psi_p", psi_p_vals)
        _write_scalar(f, "Psi_os", psi_os_vals)
        _write_scalar(f, "Cell_group", cgroups)
        _write_scalar(f, "Cell_rank", ranks)
        _write_scalar(f, "Q_in", Q_in_vals)
        _write_scalar(f, "Q_out", Q_out_vals)
        _write_scalar(f, "Q", Q_total_vals)

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
        half_thick = max(wall.thickness / 2.0, 0.5)*0.95
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
            Q_vals.append(abs(Q))
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
    """Export membrane connections as flat quads lying ON the wall surface.

    Each membrane is rendered as a rectangle whose long axis follows the
    wall segment (junction endpoint A → junction endpoint B) and whose
    height is ``extrude_z`` (from z=0 to z=extrude_z).  This places the
    membrane physically on the wall face rather than bridging into the
    cell lumen.

    Geometry
    --------
    Given wall endpoints P1=(x1,y1) and P2=(x2,y2) at z=0 and z=extrude_z::

        P1(z=0) ── P2(z=0)
           |              |
        P1(z=Z) ── P2(z=Z)

    The quad is written in CCW order: [P1_bot, P2_bot, P2_top, P1_top].
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
    km_vals: List[float] = []
    kaqp_vals: List[float] = []

    graph = obj.network.graph

    for mb in cm.membranes:
        wall = mb.wall

        # Determine the wall-surface rectangle using real junction endpoints
        jp = junction_positions.get(wall.node_id)
        if jp is not None and len(jp) >= 4:
            x1, y1, x2, y2 = jp[0], jp[1], jp[2], jp[3]
        else:
            # Fallback: +/- half-length along x from wall midpoint
            half_len = max(wall.length / 2.0, 1.0)
            x1, y1 = wall.x - half_len, wall.y
            x2, y2 = wall.x + half_len, wall.y

        # Offset the quad inward toward the connected cell by half the wall
        # thickness.  Direction: wall midpoint -> cell centroid (unit) * (t/2).
        ddx = mb.cell.x - wall.x
        ddy = mb.cell.y - wall.y
        dist = math.hypot(ddx, ddy)
        if dist > 1e-9:
            half_thick = (wall.thickness / 2.0)*1.05
            ox = (ddx / dist) * half_thick
            oy = (ddy / dist) * half_thick
        else:
            ox, oy = 0.0, 0.0

        base = len(points)
        # Four corners of the rectangle, shifted inward by (ox, oy)
        points += [
            (x1 + ox, y1 + oy, 0.0),        # 0 - endpoint-A, bottom
            (x2 + ox, y2 + oy, 0.0),        # 1 - endpoint-B, bottom
            (x2 + ox, y2 + oy, extrude_z),  # 2 - endpoint-B, top
            (x1 + ox, y1 + oy, extrude_z),  # 3 - endpoint-A, top
        ]
        polygons.append([base, base + 1, base + 2, base + 3])

        # Retrieve K/Q from graph edge (wall_node_id ↔ cell_node_id)
        edge_data = graph.edges.get(
            (wall.node_id, mb.cell.node_id),
            graph.edges.get((mb.cell.node_id, wall.node_id), {}),
        )
        K_vals.append(_safe(edge_data.get("K", mb.K_computed)))
        Q_vals.append(_safe(abs(edge_data.get("Q", mb.Q))))
        km_vals.append(_safe(mb.km))
        kaqp_vals.append(_safe(mb.kaqp))

    _ensure_dir(filepath)
    with open(filepath, "w") as f:
        f.write(_vtk_header("MECHA membrane connections (on wall surface)"))
        _write_points(f, points)
        _write_polygons(f, polygons)
        _write_cell_data_header(f, len(polygons))
        _write_scalar(f, "K_membrane", K_vals)
        _write_scalar(f, "Q_membrane", Q_vals)
        _write_scalar(f, "km", km_vals)
        _write_scalar(f, "kaqp", kaqp_vals)

    print(f"[paraview_export] Membranes → {filepath}  ({len(polygons)} quads on wall surface)")


# ---------------------------------------------------------------------------
# 4. Plasmodesmata lines  →  _plasmodesmata.vtk
# ---------------------------------------------------------------------------

def _find_shared_wall_midpoint(
    ci, cj, network
) -> Optional[Tuple[float, float]]:
    """Return the midpoint (x, y) of the wall shared between cell_i and cell_j.

    Shared wall = a wall node that has membrane edges to **both** cells.
    Falls back to the geometric midpoint between the two cell centroids if no
    shared wall is found (e.g. when cells share a junction rather than a wall).
    """
    graph = network.graph
    n_walls = network.n_walls

    walls_i = {nbr for nbr in graph.neighbors(ci.node_id)
               if nbr < n_walls
               and graph.edges[ci.node_id, nbr].get("path") == "membrane"}
    walls_j = {nbr for nbr in graph.neighbors(cj.node_id)
               if nbr < n_walls
               and graph.edges[cj.node_id, nbr].get("path") == "membrane"}

    shared = walls_i & walls_j
    if shared:
        wid = next(iter(shared))  # take the first shared wall
        pos = graph.nodes[wid].get("position", (0.0, 0.0))
        return float(pos[0]), float(pos[1])

    # Fallback: geometric midpoint between the two centroids
    return (ci.x + cj.x) / 2.0, (ci.y + cj.y) / 2.0


def _write_polylines(f, polylines: List[List[int]]) -> None:
    """Write VTK LINES where each entry may have more than 2 points (polyline)."""
    total = sum(len(pl) + 1 for pl in polylines)
    f.write(f"LINES {len(polylines)} {total}\n")
    for pl in polylines:
        f.write(f"{len(pl)} " + " ".join(str(v) for v in pl) + "\n")


def _export_plasmodesmata(
    obj: Any,
    filepath: str,
    sol: Optional[np.ndarray],
    indice: Dict[int, int],
    pd_radius: float = 0.05,
    extrude_z: float = 50.0,
) -> None:
    """
    Export plasmodesmata connections as VTK polylines routed through the shared
    wall midpoint.

    Geometry
    --------
    Each plasmodesmata is represented as a 3-point polyline::

        cell_i centroid  →  shared-wall midpoint  →  cell_j centroid

    All points sit at ``z = extrude_z / 2`` (the mid-plane of the section)
    so that the lines visually thread through the wall face rather than
    floating above or below it.

    Use ParaView's *Tube* filter (radius = ``pd_radius`` µm) to render
    cylinders.  ``pd_radius`` is embedded in the file comment header.

    Parameters
    ----------
    pd_radius : float
        Cylinder radius (µm) — written as a comment for reference.
    extrude_z : float
        Section extrusion depth (µm); plasmodesmata sit at z = extrude_z/2.
    """

    cm = obj.network.cell_manager
    if not cm:
        return

    graph = obj.network.graph
    network = obj.network
    mid_z = extrude_z / 2.0

    points: List[Tuple[float, float, float]] = []
    polylines: List[List[int]] = []
    K_vals: List[float] = []
    Q_vals: List[float] = []
    kpl_vals: List[float] = []

    # node_id → point index (so cell centroids are shared across PDs)
    node_to_pt: Dict[int, int] = {}

    for pd in cm.plasmodesmata:
        ci, cj = pd.cell_i, pd.cell_j

        # -- cell_i centroid --
        if ci.node_id not in node_to_pt:
            node_to_pt[ci.node_id] = len(points)
            points.append((ci.x, ci.y, mid_z))
        idx_i = node_to_pt[ci.node_id]

        # -- shared wall midpoint (new intermediate point, always unique) --
        wx, wy = _find_shared_wall_midpoint(ci, cj, network)
        idx_w = len(points)
        points.append((wx, wy, mid_z))

        # -- cell_j centroid --
        if cj.node_id not in node_to_pt:
            node_to_pt[cj.node_id] = len(points)
            points.append((cj.x, cj.y, mid_z))
        idx_j = node_to_pt[cj.node_id]

        polylines.append([idx_i, idx_w, idx_j])

        # K / Q from graph edge
        edge_data = graph.edges.get(
            (ci.node_id, cj.node_id),
            graph.edges.get((cj.node_id, ci.node_id), {}),
        )
        K_vals.append(_safe(edge_data.get("K")))
        Q_vals.append(_safe(edge_data.get("Q")))
        kpl_vals.append(_safe(pd.kpl))

    if not polylines:
        print("[paraview_export] No plasmodesmata — skipping.")
        return

    _ensure_dir(filepath)
    with open(filepath, "w") as f:
        f.write(
            "# vtk DataFile Version 3.0\n"
            f"MECHA plasmodesmata  pd_radius={pd_radius:.4f} um  z={mid_z:.2f} um\n"
            "ASCII\n"
            "DATASET POLYDATA\n"
        )
        _write_points(f, points)
        _write_polylines(f, polylines)
        _write_cell_data_header(f, len(polylines))
        _write_scalar(f, "K_pd", K_vals)
        _write_scalar(f, "Q_pd", Q_vals)
        _write_scalar(f, "kpl", kpl_vals)

    print(
        f"[paraview_export] Plasmodesmata → {filepath}  "
        f"({len(polylines)} polylines through wall midpoints, z={mid_z:.1f} µm)"
    )


# ---------------------------------------------------------------------------
# 5. Flow vectors on edges  →  _flow_vectors.vtk
# ---------------------------------------------------------------------------

def _export_flow_vectors(
    obj: Any,
    filepath: str,
    sol: Optional[np.ndarray],
    indice: Dict[int, int],
) -> Dict[str, str]:
    """Export edge flow vectors split into three separate files:

    'apoplastic', 'symplastic', and 'transmembrane'.
    The vector magnitude encodes |Q| and the direction is from the upstream
    node to the downstream node (sign of Q determines direction).
    Useful for the *Glyph* filter in ParaView.

    Parameters
    ----------
    obj : Any
        A fully solved Mecha instance.
    filepath : str
        Base file path for export.
    sol : np.ndarray, optional
        Solution vector.
    indice : Dict[int, int]
        Node ID to solution index mapping.

    Returns
    -------
    Dict[str, str]
        Mapping of output type -> file path for successfully exported files.
    """
    import os

    graph = obj.network.graph
    pos = dict(graph.nodes(data="position", default=(0.0, 0.0)))

    # Initialize collections for each category
    data_by_cat = {
        "apoplastic": {"points": [], "vectors": [], "K_vals": [], "Q_vals": [], "path_ids": []},
        "symplastic": {"points": [], "vectors": [], "K_vals": [], "Q_vals": [], "path_ids": []},
        "transmembrane": {"points": [], "vectors": [], "K_vals": [], "Q_vals": [], "path_ids": []},
    }

    _PATH_ID = {"wall": 0.0, "membrane": 1.0, "plasmodesmata": 2.0}
    _PATH_TO_CAT = {
        "wall": "apoplastic",
        "membrane": "transmembrane",
        "plasmodesmata": "symplastic"
    }

    for u, v, eattr in graph.edges(data=True):
        K = eattr.get("K")
        Q = eattr.get("Q")
        path = eattr.get("path", "wall")

        if K is None:
            continue

        cat = _PATH_TO_CAT.get(path, "apoplastic")

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

        data_by_cat[cat]["points"].append((mx, my, 0.0))
        data_by_cat[cat]["vectors"].append((vx, vy, 0.0))
        data_by_cat[cat]["K_vals"].append(_safe(K))
        data_by_cat[cat]["Q_vals"].append(q_val)
        data_by_cat[cat]["path_ids"].append(_PATH_ID.get(path, -1.0))

    base, ext = os.path.splitext(filepath)
    written: Dict[str, str] = {}

    for cat, data in data_by_cat.items():
        points = data["points"]
        if not points:
            continue

        cat_filepath = f"{base}_{cat}{ext}"
        _ensure_dir(cat_filepath)
        with open(cat_filepath, "w") as f:
            f.write(
                "# vtk DataFile Version 3.0\n"
                f"MECHA edge flow vectors - {cat}\n"
                "ASCII\n"
                "DATASET POLYDATA\n"
            )
            _write_points(f, points)
            # Write as vertices so the Glyph filter works
            f.write(f"VERTICES {len(points)} {2 * len(points)}\n")
            for i in range(len(points)):
                f.write(f"1 {i}\n")
            _write_point_data_header(f, len(points))
            _write_vector(f, "flow_Q", data["vectors"])
            _write_scalar(f, "K", data["K_vals"])
            _write_scalar(f, "Q_magnitude", [abs(q) for q in data["Q_vals"]])
            _write_scalar(f, "path_id", data["path_ids"])

        print(f"[paraview_export] Flow vectors ({cat}) → {cat_filepath}  ({len(points)} points)")
        written[f"flow_vectors_{cat}"] = cat_filepath

    return written


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_to_vtk(
    obj: Any,
    prefix: str = "mecha_export",
    maturity_idx: Optional[int] = None,
    scenario_idx: Optional[Union[str, int]] = None,
    extrude_z: float = 50.0,
    pd_radius: float = 0.05,
    export_cells: bool = True,
    export_walls: bool = True,
    export_membranes: bool = True,
    export_plasmodesmata: bool = True,
    export_flow_vectors: bool = True,
) -> Dict[str, str]:
    """Export a MECHA simulation result to a set of ParaView `.vtk` files.

    If ``maturity_idx`` or ``scenario_idx`` are None (default), this function
    will attempt to export all available results from ``obj.results``, adding
    suffixes like ``_mat0_scen1`` to the file prefix.

    Parameters
    ----------
    obj : Mecha
        A fully solved ``Mecha`` instance.
    prefix : str
        Path prefix for output files.
    maturity_idx : int, optional
        Maturity stage index. If None, exports all available.
    scenario_idx : str or int, optional
        Scenario key. If None, exports all available.
    extrude_z : float
        Depth (µm) for extrusion.
    pd_radius : float
        Plasmodesmata radius (µm).
    export_cells, export_walls, export_membranes, export_plasmodesmata,
    export_flow_vectors : bool
        Toggle individual output files.

    Returns
    -------
    dict
        Mapping of output type → file path for the LAST export iteration.
    """
    if not hasattr(obj, "network") or obj.network is None:
        raise ValueError("obj must be a Mecha instance with a populated network.")
    if obj.network.cell_manager is None:
        raise ValueError("network.cell_manager is None — call build_network first.")

    # Determine what to export
    to_export: List[Tuple[int, Union[str, int]]] = []
    if hasattr(obj, "results") and obj.results:
        for res in obj.results:
            m = res.get("maturity stage")
            s = res.get("scenario")
            if (maturity_idx is None or m == maturity_idx) and \
               (scenario_idx is None or s == scenario_idx):
                to_export.append((m, s))

    # Fallback to single export if no results list or no match found
    if not to_export:
        to_export = [(maturity_idx if maturity_idx is not None else 0,
                      scenario_idx if scenario_idx is not None else "standard water flow")]

    last_written: Dict[str, str] = {}
    indice: Dict[int, int] = getattr(obj, "indice", {})

    for m_val, s_val in to_export:
        # Determine prefix for this iteration
        if len(to_export) > 1:
            s_suffix = str(s_val).replace(" ", "_")
            current_prefix = f"{prefix}_mat{m_val}_scen{s_suffix}"
        else:
            current_prefix = prefix

        # Find the result entry to get the specific solution and Kmb
        res_entry = None
        if hasattr(obj, "results") and obj.results:
            for res in obj.results:
                if (res.get("maturity stage") == m_val
                        and res.get("scenario") == s_val):
                    res_entry = res
                    break
        
        if res_entry:
            sol = np.asarray(res_entry["solution"]).ravel()

            # Restore the pre-computed graph state for this scenario.
            # This is faster and avoids the overwriting bug that occurred when
            # initialize_scenarios + compute_edge_flows modified shared graph/
            # object attributes in the wrong order.
            restored = False
            if hasattr(obj, 'restore_scenario_state'):
                restored = obj.restore_scenario_state(m_val, s_val)

            if not restored:
                # Fallback: re-compute (for instances without saved snapshots)
                Kmb = res_entry.get("Kmb")
                s_idx = 0 if (isinstance(s_val, str) and s_val == "standard water flow") else s_val
                if hasattr(obj, "initialize_scenarios") and Kmb is not None:
                    obj.initialize_scenarios(s_idx, m_val, Kmb)
                if hasattr(obj, "compute_edge_flows"):
                    obj.compute_edge_flows(sol, i_maturity=m_val, i_scenario=s_idx)
        else:
            sol = _get_solution(obj, maturity_idx=m_val, scenario_idx=s_val)
        
        if sol is None and (maturity_idx is not None or scenario_idx is not None):
             print(f"[paraview_export] WARNING: No solution found for mat={m_val} / scen={s_val}")

        written: Dict[str, str] = {}

        if export_cells:
            fp = f"{current_prefix}_cells.vtk"
            _export_cells(obj, fp, sol, indice, extrude_z)
            written["cells"] = fp

        if export_walls:
            fp = f"{current_prefix}_walls.vtk"
            _export_walls(obj, fp, extrude_z)
            written["walls"] = fp

        if export_membranes:
            fp = f"{current_prefix}_membranes.vtk"
            _export_membranes(obj, fp, sol, indice, extrude_z)
            written["membranes"] = fp

        if export_plasmodesmata:
            fp = f"{current_prefix}_plasmodesmata.vtk"
            _export_plasmodesmata(obj, fp, sol, indice, pd_radius=pd_radius,
                                  extrude_z=extrude_z)
            written["plasmodesmata"] = fp

        if export_flow_vectors:
            fp = f"{current_prefix}_flow_vectors.vtk"
            written_vectors = _export_flow_vectors(obj, fp, sol, indice)
            written.update(written_vectors)
        
        last_written = written

    print(f"\n[paraview_export] Done — {len(to_export)} scenario(s) exported.")
    return last_written
