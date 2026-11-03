"""
Export a solved MECHA network for ParaView (VTK XML, no VTK library needed).

Works for a 2D cross-section and for a 3D segment (GRANAP Network3D):

``<prefix>_cells.vtu``
    The cells: flat polygons at z = 0 (2D), or prisms between each row's z0
    and z1 (3D).  Cell data: ``psi`` (hPa), ``cgroup``, ``tissue`` (code),
    ``node`` (graph node id); 3D adds ``cell_2d`` and ``row``.
``<prefix>_network.vtp``
    Every graph node as a point, every edge as a line.  Point data: ``psi``,
    ``node_kind`` (code), ``cgroup`` (-1 off cells); line data: ``Q`` (cm^3/d,
    signed from the lower to the higher node id), ``absQ``, ``K``, ``path``
    (code), ``orientation`` (0 lateral, 1 axial; always 0 in 2D).
``<prefix>_legend.txt``
    The integer codes and the units.

Positions are in µm, as in MECHA's graph.  ``Q`` and ``K`` are those stored
on the edges by ``Mecha.solve_W`` (``compute_edge_flows``).
"""

from typing import Dict, Optional

import numpy as np

VTK_POLYGON = 7
VTK_POLYHEDRON = 42


def _array(name: str, values, ncomp: int = 1, kind: str = "Float64") -> str:
    values = np.asarray(values).ravel()
    if kind.startswith("Float"):
        text = " ".join(repr(float(v)) for v in values)
    else:
        text = " ".join(str(int(v)) for v in values)
    return (f'<DataArray type="{kind}" Name="{name}" NumberOfComponents="{ncomp}" '
            f'format="ascii">{text}</DataArray>\n')


def _codes(values) -> Dict[str, int]:
    return {v: i for i, v in enumerate(sorted(set(values)))}


def _node_kind(data: dict) -> str:
    if data.get("type") == "cell":
        return "cell"
    kind = data.get("kind")
    if kind is not None:  # 3D
        return {"side": "side wall", "end": "end wall", "face": "face",
                "vertical": "vertical junction", "rim": "rim junction"}[kind]
    return "wall" if data.get("length", 0) else "junction"


def _ring(polygon) -> np.ndarray:
    """Exterior ring of a cell polygon (largest piece of a MultiPolygon), open, counter-clockwise."""
    from shapely.geometry.polygon import orient
    if polygon.geom_type != "Polygon":
        polygon = max(polygon.geoms, key=lambda p: p.area)
    coords = np.asarray(orient(polygon, 1.0).exterior.coords)
    return coords[:-1]


def export_vtk(mecha, prefix: str, solution: Optional[np.ndarray] = None) -> Dict[str, str]:
    """Write the cells, the network and a legend of a solved ``mecha`` (see module doc).

    Args:
        mecha: a :class:`Mecha` after ``solve_W`` (2D or 3D network).
        prefix: path prefix of the three files.
        solution: water potential per node; default the last ``solve_W`` result.

    Returns:
        ``{"cells": ..., "network": ..., "legend": ...}`` file paths.
    """
    network = mecha.network
    graph = network.graph
    if solution is None:
        if not getattr(mecha, "results", None):
            raise ValueError("export_vtk needs a solved Mecha (call solve_W first) or a solution")
        solution = mecha.results[-1]["solution"]
    psi = np.asarray(solution).ravel()
    is_3d = getattr(network, "is_3d", False)

    cells = sorted(network.cell_manager, key=lambda c: c.node_id)
    tissues = _codes(c.cell_type for c in cells)
    kinds = _codes(_node_kind(d) for _, d in graph.nodes(data=True))
    paths = _codes(d.get("path", "") for *_, d in graph.edges(data=True))

    # ---- cells ------------------------------------------------------------
    points, conn, offsets, types, faces, face_offsets = [], [], [], [], [], []
    for c in cells:
        ring = _ring(c.polygon)
        n, base = len(ring), len(points)
        if is_3d:
            d = graph.nodes[c.node_id]
            points += [(x, y, d["z0"]) for x, y in ring] + [(x, y, d["z1"]) for x, y in ring]
            ids = list(range(base, base + 2 * n))
            stream = [n + 2, n] + [base + i for i in reversed(range(n))] + [n] + [base + n + i for i in range(n)]
            for i in range(n):
                j = (i + 1) % n
                stream += [4, base + i, base + j, base + n + j, base + n + i]
            faces += stream
            face_offsets.append(len(faces))
            types.append(VTK_POLYHEDRON)
        else:
            points += [(x, y, 0.0) for x, y in ring]
            ids = list(range(base, base + n))
            types.append(VTK_POLYGON)
        conn += ids
        offsets.append(len(conn))

    cell_data = (_array("psi", [psi[c.node_id] for c in cells])
                 + _array("cgroup", [c.cgroup for c in cells], kind="Int32")
                 + _array("tissue", [tissues[c.cell_type] for c in cells], kind="Int32")
                 + _array("node", [c.node_id for c in cells], kind="Int64"))
    if is_3d:
        cell_data += (_array("cell_2d", [graph.nodes[c.node_id]["cell_2d"] for c in cells], kind="Int32")
                      + _array("row", [graph.nodes[c.node_id]["row"] for c in cells], kind="Int32"))
    topology = (_array("connectivity", conn, kind="Int64") + _array("offsets", offsets, kind="Int64")
                + _array("types", types, kind="UInt8"))
    if is_3d:
        topology += _array("faces", faces, kind="Int64") + _array("faceoffsets", face_offsets, kind="Int64")

    out = {"cells": prefix + "_cells.vtu", "network": prefix + "_network.vtp", "legend": prefix + "_legend.txt"}
    with open(out["cells"], "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian">\n'
                f'<UnstructuredGrid><Piece NumberOfPoints="{len(points)}" NumberOfCells="{len(cells)}">\n'
                "<Points>" + _array("Points", points, 3) + "</Points>\n"
                "<Cells>\n" + topology + "</Cells>\n"
                '<CellData Scalars="psi">\n' + cell_data + "</CellData>\n"
                "</Piece></UnstructuredGrid></VTKFile>\n")

    # ---- network ----------------------------------------------------------
    nodes = sorted(graph.nodes)
    index = {n: i for i, n in enumerate(nodes)}
    node_points = [tuple(graph.nodes[n]["position"]) + ((0.0,) if len(graph.nodes[n]["position"]) == 2 else ())
                   for n in nodes]
    # compute_edge_flows stores Q = K (psi_u - psi_v) in the graph's own (u, v)
    # order; each line here runs from the lower to the higher node id.
    edges, q = [], []
    for u, v, d in graph.edges(data=True):
        edges.append((min(u, v), max(u, v), d))
        q.append(d.get("Q", 0.0) * (1.0 if u < v else -1.0))
    q = np.asarray(q, dtype=float)
    with open(out["network"], "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian">\n'
                f'<PolyData><Piece NumberOfPoints="{len(nodes)}" NumberOfLines="{len(edges)}">\n'
                "<Points>" + _array("Points", node_points, 3) + "</Points>\n"
                "<Lines>" + _array("connectivity", [index[n] for u, v, _ in edges for n in (u, v)], kind="Int64")
                + _array("offsets", range(2, 2 * len(edges) + 1, 2), kind="Int64") + "</Lines>\n"
                '<PointData Scalars="psi">\n' + _array("psi", [psi[n] for n in nodes])
                + _array("node_kind", [kinds[_node_kind(graph.nodes[n])] for n in nodes], kind="Int32")
                + _array("cgroup", [graph.nodes[n].get("cgroup", -1) if graph.nodes[n].get("type") == "cell" else -1
                                    for n in nodes], kind="Int32") + "</PointData>\n"
                '<CellData Scalars="absQ">\n' + _array("Q", q) + _array("absQ", np.abs(q))
                + _array("K", [d.get("K", 0.0) for *_, d in edges])
                + _array("path", [paths[d.get("path", "")] for *_, d in edges], kind="Int32")
                + _array("orientation", [int(d.get("orientation") == "axial") for *_, d in edges], kind="Int32")
                + "</CellData>\n</Piece></PolyData></VTKFile>\n")

    # ---- legend -----------------------------------------------------------
    with open(out["legend"], "w") as f:
        f.write(f"{'3D segment' if is_3d else '2D cross-section'}: {len(cells)} cells, "
                f"{len(nodes)} nodes, {len(edges)} edges\n")
        for title, codes in (("tissue (cells)", tissues), ("node_kind (network points)", kinds),
                             ("path (network lines)", paths)):
            f.write(title + ":\n" + "".join(f"  {i} {name}\n" for name, i in codes.items()))
        f.write("orientation (network lines): 0 lateral, 1 axial\n"
                "units: positions µm, psi hPa, Q cm^3/d (signed from the lower to the higher node id)\n")
        kr = getattr(mecha, "kr_tot", None)
        if kr is not None:
            f.write(f"kr (first maturity stage) = {kr[0][0]:.4e} cm/hPa/d\n")
    return out
