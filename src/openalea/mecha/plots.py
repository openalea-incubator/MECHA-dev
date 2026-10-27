"""
Plots of one section's hydraulics (the legacy's ``utils/visu`` views, as functions of arrays): its cells by tissue or
by water potential, its conductances by pathway, the radial profile of the potential, the share of each pathway in
the radial flow, the flow vectors; and develop's views of a section partly in contact with the soil: the same
profiles along the wet→dry transect (``axis="signed"``), the maps of the distance to the soil contact and of the
plasmodesmata's aperture. Each takes a SectionFrame, built from the DataStructure (``port_frame``) or, for the
comparison, from the legacy's network (``test/comparison``), so that both are drawn the same way.
"""
from dataclasses import dataclass, field

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon as PolygonPatch

PATHS = {"wall": "apoplastic (walls)", "membrane": "transmembrane", "plasmodesmata": "symplastic (plasmodesmata)"}
PATH_COLOURS = {"wall": "tab:orange", "membrane": "tab:blue", "plasmodesmata": "tab:green"}
_PATH_OF_CODE = {0: "membrane", 2: "wall", 3: "plasmodesmata"}


@dataclass
class SectionFrame:
    """One section: nodes (positions µm, kind 0 cell / 1 wall / 2 junction, rank, tissue), its MECHA edges (ends,
    pathway name, K), the solved potentials and flows, the cells' polygons (µm, or None)."""
    xy: np.ndarray
    kind: np.ndarray
    rank: np.ndarray
    tissue: np.ndarray
    a: np.ndarray
    b: np.ndarray
    path: np.ndarray
    K: np.ndarray
    psi: np.ndarray = None
    Q: np.ndarray = None                     # from a to b
    polygons: list = field(default_factory=list)      # per cell (kind 0, in order): a shapely polygon or None
    centre: tuple = (0., 0.)
    contact: np.ndarray = None               # per node: in contact with the soil (border walls and junctions)
    distance: np.ndarray = None              # per node: the distance to the soil contact along the graph (µm)
    aperture: np.ndarray = None              # per edge: the plasmodesmata's spatial aperture coefficient (NaN: other)


def port_frame(mecha, segment, psi="standard_psi", flow="standard_flow", libraries=()) -> SectionFrame:
    """The section of *segment* from MECHA's component (its DataStructure), with GRANAP's cell polygons (from its
    templates; *libraries*: unused, kept for the earlier calls)."""
    ds = mecha.data_structure
    nodes, edges, geo = mecha.section_geometries()[segment]
    local = np.full(ds.n_nodes(), -1, dtype=np.int64)
    local[nodes] = np.arange(nodes.size)
    view = ds.to_graph_view()
    path_code = np.asarray(ds.get("network_path"))[edges]
    K = np.asarray(ds.get("conductance"))[edges]
    kept = np.isin(path_code, list(_PATH_OF_CODE)) & ((path_code != 2) | geo.junction_edge[np.arange(edges.size)])
    e = edges[kept]
    polygons = []
    g = ds.mtg
    template, row = g.property("template"), g.property("template_row")
    vids = ds.entity_ids("node")[nodes[geo.cells]]
    if vids.size and int(vids[0]) in template:                            # GRANAP's anatomies: their templates
        from shapely import affinity
        from openalea.granap.metafspm.library import find_template
        polygons_of = find_template(libraries, int(template[int(vids[0])])).polygons
        scale = 1000.                                                     # mm -> µm
        polygons = [None if polygons_of[int(row[int(v)])] is None else
                    affinity.scale(polygons_of[int(row[int(v)])], scale, scale, origin=(0, 0)) for v in vids]
    path = np.array([_PATH_OF_CODE[c] for c in np.asarray(ds.get("network_path"))[e].astype(int)])
    aperture = np.full(e.size, np.nan)
    plasmodesmata = path == "plasmodesmata"
    if plasmodesmata.any():
        from openalea.mecha.conductance import plasmodesmata_spatial_coefficients
        aperture[plasmodesmata], _ = plasmodesmata_spatial_coefficients(mecha.network(), mecha.parameter_table("edge"),
                                                                        e[plasmodesmata])
    distance = np.asarray(ds.get("contact_distance"))[nodes] if ds.has("contact_distance") else None
    return SectionFrame(xy=np.c_[geo.x, geo.y], kind=geo.kind, rank=geo.cell_rank, tissue=geo.tissue,
                        a=local[view.tail[e]], b=local[view.head[e]], path=path,
                        K=np.asarray(ds.get("conductance"))[e],
                        psi=None if psi is None else np.asarray(ds.get(psi))[nodes],
                        Q=None if flow is None else np.asarray(ds.get(flow))[e],
                        polygons=polygons, centre=geo.gravity_center,
                        contact=np.asarray(ds.get("soil_conductance"))[nodes] > 0, distance=distance,
                        aperture=aperture)


def _cells(frame):
    return np.flatnonzero(frame.kind == 0)


def _draw_cells(ax, frame, colours, **kwargs):
    patches, faces = [], []
    for cell, polygon, colour in zip(_cells(frame), frame.polygons, colours):
        if polygon is None or polygon.is_empty:
            continue
        for part in (polygon.geoms if polygon.geom_type == "MultiPolygon" else [polygon]):
            patches.append(PolygonPatch(np.asarray(part.exterior.coords), closed=True))
            faces.append(colour)
    ax.add_collection(PatchCollection(patches, facecolor=faces, edgecolor="0.3", linewidth=0.15, **kwargs))
    ax.autoscale_view()
    ax.set_aspect("equal")


def tissue_map(ax, frame, title="tissues"):
    """The cells coloured by tissue (polygons; cell centres without them)."""
    cells = _cells(frame)
    types = sorted({t for t in frame.tissue[cells] if t is not None})
    palette = {t: matplotlib.colormaps["tab20"](i % 20) for i, t in enumerate(types)}
    colours = [palette[t] for t in frame.tissue[cells]]
    if frame.polygons:
        _draw_cells(ax, frame, colours)
    else:
        ax.scatter(*frame.xy[cells].T, c=colours, s=6)
        ax.set_aspect("equal")
    ax.legend(handles=[matplotlib.patches.Patch(color=palette[t], label=t) for t in types], fontsize=6,
              loc="lower right")
    ax.set(title=title, xlabel="µm")


def psi_map(ax, frame, title="water potential", norm=None):
    """The cells coloured by their potential (hPa); the apoplast's nodes as small dots over them."""
    cells = _cells(frame)
    norm = norm or matplotlib.colors.Normalize(np.nanmin(frame.psi), np.nanmax(frame.psi))
    cmap = matplotlib.colormaps["coolwarm_r"]
    if frame.polygons:
        _draw_cells(ax, frame, cmap(norm(frame.psi[cells])))
    apoplast = frame.kind > 0
    points = ax.scatter(*frame.xy[apoplast if frame.polygons else slice(None)].T, s=0.4,
                        c=frame.psi[apoplast if frame.polygons else slice(None)], cmap=cmap, norm=norm)
    ax.set_aspect("equal")
    ax.set(title=title, xlabel="µm")
    return points


def conductance_by_path(axes, frame, norm=None):
    """Three panels: log10 K of the walls, the membranes, the plasmodesmata (cm³ hPa⁻¹ d⁻¹)."""
    logK = np.log10(np.maximum(frame.K, 1e-30))
    norm = norm or matplotlib.colors.Normalize(np.percentile(logK, 1), logK.max())
    out = None
    for ax, path in zip(axes, PATHS):
        chosen = frame.path == path
        lines = LineCollection(np.stack([frame.xy[frame.a[chosen]], frame.xy[frame.b[chosen]]], axis=1),
                               cmap="viridis", norm=norm, linewidths=0.4)
        lines.set_array(logK[chosen])
        ax.add_collection(lines)
        ax.autoscale_view()
        ax.set_aspect("equal")
        ax.set(title=PATHS[path], xlabel="µm")
        out = lines
    return out


def wetdry_axis(frame):
    """(origin, unit vector) of the wet→dry transect (develop's _wetdry_axis): from the centroid of the nodes in contact
    with the soil toward the section's centre, and on to the dry side; None without contact, or when the contact
    surrounds the centre (their centroid on it)."""
    if frame.contact is None or not frame.contact.any():
        return None
    origin = np.asarray(frame.centre, dtype=float)
    towards = origin - frame.xy[frame.contact].mean(axis=0)
    norm = np.hypot(*towards)
    return None if norm < 1e-9 else (origin, towards / norm)


def _coordinate(frame, axis):
    """(coordinate of points, component of a vector along the axis, label): the distance to the centre ("radial") or
    the signed position along the wet→dry transect, negative on the wet side ("signed")."""
    centre = np.asarray(frame.centre, dtype=float)
    if axis == "radial":
        return (lambda points: np.hypot(*(np.asarray(points) - centre).T),
                lambda points, vectors: (vectors * (np.asarray(points) - centre)).sum(axis=1)
                / np.maximum(np.hypot(*(np.asarray(points) - centre).T), 1e-300),
                "distance to the centre (µm)")
    transect = wetdry_axis(frame)
    if transect is None:
        raise ValueError("no wet→dry axis: no node in contact with the soil, or the contact all around")
    origin, unit = transect
    return (lambda points: ((np.asarray(points) - origin) * unit).sum(axis=1),
            lambda points, vectors: (vectors * unit).sum(axis=1), "wet ← transect (µm) → dry")


def radial_profile(ax, frame, label=None, bins=40, axis="radial", **style):
    """The mean potential of the cells and of the apoplast against the distance to the centre, or the position along
    the wet→dry transect (*axis* "signed") (hPa vs µm); NaN values (nodes left out) are skipped."""
    coordinate, _, axis_label = _coordinate(frame, axis)
    r = coordinate(frame.xy)
    edges = np.linspace(0 if axis == "radial" else r.min(), r.max(), bins + 1)
    middle = (edges[1:] + edges[:-1]) / 2
    for chosen, part, marker in ((frame.kind == 0, "cells", "o"), (frame.kind > 0, "apoplast", "")):
        which = np.digitize(r[chosen], edges) - 1
        values = frame.psi[chosen]
        mean = np.array([np.nanmean(values[which == k]) if np.isfinite(values[which == k]).any() else np.nan
                         for k in range(bins)])
        ax.plot(middle, mean, marker=marker, ms=3, label=f"{label + ', ' if label else ''}{part}", **style)
    ax.set(xlabel=axis_label, ylabel="ψ (hPa)", title="radial profile of the potential")


def pathway_flows(frame, axis="radial"):
    """
    (R, ranks, flows): the radial flow carried by each pathway at each layer radius (the legacy's
    ``plot_flow_pathway_breakdown``): each edge's radial component of |Q| assigned to the nearest of the mean radii
    *R* (µm) of the cell layers (by rank: the cells' centres and their inner and outer walls); *ranks* the cell rank
    of each radius; *flows* {pathway: cm³ d⁻¹ per radius}. With *axis* "signed" (develop's ``x_axis='signed'``): the
    layers' positions and the edges' middles along the wet→dry transect, the flow's component along it; each layer's
    wet and dry halves apart (a whole ring's mean position along the transect is near 0, as develop's radial profile
    found).
    """
    coordinate, component, _ = _coordinate(frame, axis)
    centre = np.asarray(frame.centre)
    xy = frame.xy - centre
    cells = _cells(frame)
    walls_of = {}
    for a, b, path in zip(frame.a, frame.b, frame.path):
        if path == "membrane":
            cell, wall = (a, b) if frame.kind[a] == 0 else (b, a)
            walls_of.setdefault(int(cell), []).append(int(wall))
    radii = {}
    for c in cells:
        rc = np.hypot(*xy[c])
        if rc < 1e-6:
            continue
        own = coordinate(frame.xy[c][None])[0]
        half = 0 if axis == "radial" else (own >= 0)                  # the signed axis: the wet and dry halves
        radii.setdefault((frame.rank[c], "mid", half), []).append(own)
        for w in walls_of.get(int(c), []):
            vw = xy[w] - xy[c]
            nw = np.hypot(*vw)
            if nw < 1e-6:
                continue
            cos = (xy[c] @ vw) / (rc * nw)
            if cos > 0.4:
                radii.setdefault((frame.rank[c], "out", half), []).append(coordinate(frame.xy[w][None])[0])
            elif cos < -0.4:
                radii.setdefault((frame.rank[c], "in", half), []).append(coordinate(frame.xy[w][None])[0])
    rank_of = {np.mean(v): key[0] for key, v in radii.items()}
    R = np.unique(list(rank_of))
    middle = (frame.xy[frame.a] + frame.xy[frame.b]) / 2
    vec = xy[frame.b] - xy[frame.a]
    n_vec = np.hypot(*vec.T)
    position = coordinate(middle)
    at_centre = (axis == "radial") & (np.hypot(*(middle - centre).T) < 1e-6)
    along = np.where(at_centre | (n_vec < 1e-6), 1., abs(component(middle, vec)) / np.maximum(n_vec, 1e-300))
    q = abs(frame.Q) * along
    nearest = R[np.argmin(abs(R[None, :] - position[:, None]), axis=1)]
    flows = {path: np.array([q[(frame.path == path) & (nearest == r)].sum() for r in R]) for path in PATHS}
    return R, np.array([rank_of[r] for r in R]), flows


def pathway_breakdown(ax, frame, title="pathways of the radial flow", axis="radial"):
    """The share of each pathway in the radial flow, against the distance to the centre or, *axis* "signed", the
    position along the wet→dry transect (pathway_flows)."""
    R, _, shares = pathway_flows(frame, axis)
    total = sum(shares.values())
    total[total == 0] = 1.
    ax.stackplot(R, *[100 * shares[p] / total for p in PATHS], labels=list(PATHS.values()),
                 colors=[PATH_COLOURS[p] for p in PATHS], alpha=0.7)
    ax.set(xlabel=_coordinate(frame, axis)[2], ylabel="share of the radial flow (%)", ylim=(0, 100), title=title)
    ax.legend(loc="lower left", fontsize=7)


def contact_distance_map(axes, frame):
    """Two panels (develop's plot_contact_distance_map): the cells coloured by their distance to the soil contact
    along the graph (µm), and by the mean spatial aperture coefficient of their plasmodesmata; the cells next to a
    contact node outlined. Returns the two colour mappings."""
    cells = _cells(frame)
    pd = (frame.path == "plasmodesmata") & np.isfinite(frame.aperture)
    total = np.bincount(np.r_[frame.a[pd], frame.b[pd]], weights=np.r_[frame.aperture[pd], frame.aperture[pd]],
                        minlength=frame.kind.size)
    count = np.bincount(np.r_[frame.a[pd], frame.b[pd]], minlength=frame.kind.size)
    aperture = np.divide(total, count, out=np.full(frame.kind.size, np.nan), where=count > 0)
    membranes = frame.path == "membrane"
    wetted = np.zeros(frame.kind.size, dtype=bool)
    for a, b in zip(frame.a[membranes], frame.b[membranes]):
        cell, wall = (a, b) if frame.kind[a] == 0 else (b, a)
        wetted[cell] |= frame.contact[wall]
    out = []
    for ax, values, cmap, label in ((axes[0], frame.distance, "viridis", "distance to the soil contact (µm)"),
                                    (axes[1], aperture, "magma", "plasmodesmata's aperture coefficient")):
        finite = values[cells][np.isfinite(values[cells])]
        norm = matplotlib.colors.Normalize(*(np.nanmin(finite), np.nanmax(finite)) if finite.size else (0., 1.))
        colours = matplotlib.colormaps[cmap](norm(np.where(np.isfinite(values[cells]), values[cells], np.nan)))
        if frame.polygons:
            _draw_cells(ax, frame, colours)
        else:
            ax.scatter(*frame.xy[cells].T, c=colours, s=6)
            ax.set_aspect("equal")
        ax.scatter(*frame.xy[cells[wetted[cells]]].T, s=4, facecolors="none", edgecolors="k", linewidths=0.4)
        ax.set(title=label, xlabel="µm")
        out.append(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap))
    return out


def flow_vectors(ax, frame, path="membrane", title=None, scale=None):
    """The flows of one pathway as arrows from their edges' middles, in their direction, length ∝ |Q|."""
    chosen = (frame.path == path) & (frame.Q != 0)
    a, b, q = frame.a[chosen], frame.b[chosen], frame.Q[chosen]
    direction = frame.xy[b] - frame.xy[a]
    direction /= np.maximum(np.hypot(*direction.T), 1e-12)[:, None]
    direction *= np.sign(q)[:, None]
    middle = (frame.xy[a] + frame.xy[b]) / 2
    ax.quiver(*middle.T, *(direction * abs(q)[:, None]).T, abs(q), cmap="plasma", angles="xy",
              scale=scale or abs(q).max() * 40, width=0.002)
    ax.set_aspect("equal")
    ax.set(title=title or f"flows, {PATHS[path]}", xlabel="µm")
