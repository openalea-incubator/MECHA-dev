"""
MECHA's own geometry of a cross-section (devplan_metafspm.md, §8, step 2): the quantities the legacy derives in
``NetworkBuilder.populate_from_network``, computed from the section's network as the DataStructure holds it (GRANAP's
columns, in mm), in MECHA's units (µm, cm) and with the legacy's summation orders, so that they are exact.

    section = Section(kind, cgroup, tissue, x, y, wall_length, cell_area, a, b, path, length, dist)
    geometry = section_geometry(section)

Nodes are the section's Compartments in the DataStructure's order (protoplasts, then walls, then junctions, each in
GRANAP's order); edges are its Connections in the DataStructure's order (GRANAP's connection order). The legacy numbers
the nodes walls, junctions, cells (``legacy_order``). Its junction rebuild by position only renumbers GRANAP's junctions
(§7), so the port keeps them; it does reproduce the rebuild's rule that a wall with fewer than two junctions loses
its junction edges.
"""
from dataclasses import dataclass, field

import numpy as np

PROTOPLAST, WALL, JUNCTION = 0, 1, 2                                     # GRANAP's kind
MEMBRANE, WALL_AIR, WALL_PATH, PLASMODESMATA, AIR_LINK = 0, 1, 2, 3, 4   # GRANAP's network_path

MM = 1000.                                                               # mm -> µm
N_LAYERS = 62

TYPE_TO_CGROUP = {                                                       # the legacy's type_mapper (cgroup missing)
    'exodermis': 1, 'epidermis': 2, 'endodermis': 3, 'passage': 3, 'cortex': 4, 'air space': 4, 'mesophyll': 4,
    'stele': 5, 'pith': 5, 'parenchyma': 5, 'vascular_parenchyma': 5, 'phloem': 11, 'protosieve': 23,
    'protophloem': 23, 'sieve': 11, 'companion': 12, 'cambium': 12, 'guard cell': 12, 'Strasburger cell': 12,
    'xylem': 13, 'protoxylem': 13, 'metaxylem': 13, 'pericycle': 16,
}
XYLEM_CGROUPS = (13, 19, 20)
STELE_RANKS = (5, 11, 12, 13, 23)
INTERCELLULAR_TYPES = ("intercellular", "air space", "aerenchyma")
XYLEM_TYPES = ("protoxylem", "metaxylem", "xylem")
SIEVE_TYPES = ("sieve", "phloem")
PROTOSIEVE_TYPES = ("protosieve", "protophloem")


@dataclass
class Section:
    """One section's network: node arrays (DataStructure order) and edge arrays (local node indices a, b)."""
    kind: np.ndarray
    cgroup: np.ndarray            # float, NaN on the apoplast
    tissue: np.ndarray            # object, None on the apoplast
    x: np.ndarray                 # mm
    y: np.ndarray
    wall_length: np.ndarray       # mm
    cell_area: np.ndarray         # mm2
    a: np.ndarray
    b: np.ndarray
    path: np.ndarray
    length: np.ndarray            # mm
    dist: np.ndarray              # mm


@dataclass
class SectionGeometry:
    """MECHA's geometry of a section (µm unless stated), arrays over the section's nodes or edges (local indices)."""
    legacy_order: np.ndarray      # local node of each legacy node id (walls, junctions, cells)
    walls: np.ndarray             # local nodes, in legacy order
    junctions: np.ndarray
    cells: np.ndarray             # local nodes, in cell order (cell id = position)
    kind: np.ndarray              # GRANAP's kind per node
    cgroup: np.ndarray            # int per node (cells; -1 on the apoplast)
    tissue: np.ndarray
    x: np.ndarray                 # µm per node (junctions rounded to 1e-6 µm, as the legacy's rebuild)
    y: np.ndarray
    wall_length: np.ndarray       # µm per node: walls; border junctions Σ length/4; else 0
    cell_area: np.ndarray         # µm² per node
    lateral_distance: np.ndarray  # walls: Σ of their membranes' dist (µm); NaN elsewhere
    n_membranes: np.ndarray       # per node (walls)
    border_wall: np.ndarray       # bool per node: a wall with one membrane, to an epidermis cell (the soil interface)
    border_aerenchyma: np.ndarray  # bool per node: a wall with one membrane, to another cell
    border_junction: np.ndarray   # bool per node: a junction between two border walls
    junction_edge: np.ndarray     # bool per edge: a wall edge between a wall and a junction, kept by the rebuild
    edge_length: np.ndarray       # µm per edge: rebuilt wall edges wall length/2; others GRANAP's ×1000
    edge_lateral_distance: np.ndarray  # µm per edge: rebuilt wall edges, the wall's lateral distance
    edge_dist: np.ndarray         # µm per edge (membranes, plasmodesmata)
    gravity_center: tuple         # (x, y) µm: mean of the endodermis cells' positions
    distance_center_grav: np.ndarray  # µm per node (walls)
    distance_max_cortex: float
    distance_min_cortex: float
    distance_avg_epi: float
    perimeter: float              # cm: 2π × the epidermis walls' mean distance to the centre
    cell_rank: np.ndarray         # int per node (cells; -1 elsewhere)
    layer_dist: np.ndarray        # (62,) mean distance of each rank's cells to the centre
    n_layer: np.ndarray           # (62,)
    xylem_80_percentile_distance: float
    outercortex_connec_rank: int
    stele_connec_rank: int
    concentrated_outer_cortex: float
    concentrated_cortex_cortex: float
    concentrated_cortex_endo: float
    xylem_cells: np.ndarray       # local nodes (cgroup 13, 19, 20), cell order
    sieve_cells: np.ndarray
    protosieve_cells: np.ndarray  # with the legacy's duplicates (§2 f)
    passage_cells: np.ndarray     # local nodes (tissue 'passage')
    intercellular_cells: np.ndarray  # local nodes (the network's list: by the elif chain)
    typed: dict = field(default_factory=dict)   # the cell manager's sets by tissue: intercellular, passage, xylem...
    wall_cells: np.ndarray = None # (n, 2) per wall: its membranes' cells (local nodes, in cell order), -1 if none
    junction_order: np.ndarray = None  # the junctions (local nodes) in the legacy's rebuilt order (its junction ids)


def _sequential_sum(values) -> float:
    """Σ in the given order, one addition after another (the legacy's Python loops)."""
    values = np.asarray(values, dtype=float)
    return float(np.cumsum(values)[-1]) if values.size else 0.


def _cgroups(section: Section) -> np.ndarray:
    """The legacy's integer cgroup of each cell (its type's group when the cgroup is missing), -1 on the apoplast."""
    out = np.full(section.kind.size, -1, dtype=np.int64)
    for v in np.flatnonzero(section.kind == PROTOPLAST):
        c = section.cgroup[v]
        try:
            out[v] = int(c)
        except (ValueError, TypeError):
            out[v] = TYPE_TO_CGROUP.get(section.tissue[v], 4)
    return out


def _neighbour_keys(n_nodes, a, b, order_position, edges):
    """
    For each (node, edge) incidence of *edges*, the sort key of the edge in the node's neighbour list, as networkx
    holds it after ``Graph.copy()`` of the legacy's graph: the neighbours earlier in node order first (by node order),
    then the later ones in edge (insertion) order. Returns (node, other, key) arrays, sorted by (node position, key).
    """
    node = np.r_[a[edges], b[edges]]
    other = np.r_[b[edges], a[edges]]
    e = np.r_[edges, edges]
    earlier = order_position[other] < order_position[node]
    key = np.where(earlier, order_position[other], n_nodes + e)
    sort = np.lexsort((key, order_position[node]))
    return node[sort], other[sort], e[sort]


def section_geometry(section: Section) -> SectionGeometry:
    s = section
    n = s.kind.size
    walls = np.flatnonzero(s.kind == WALL)
    junctions = np.flatnonzero(s.kind == JUNCTION)
    cells = np.flatnonzero(s.kind == PROTOPLAST)
    legacy_order = np.r_[walls, junctions, cells]
    position = np.empty(n, dtype=np.int64)
    position[legacy_order] = np.arange(n)
    cgroup = _cgroups(s)
    tissue = np.asarray(s.tissue, dtype=object)
    is_cell = s.kind == PROTOPLAST

    # positions (µm); the rebuilt junctions are rounded to 6 decimals (Python's round, as the legacy)
    x, y = s.x * MM, s.y * MM
    for v in junctions:
        x[v], y[v] = round(float(x[v]), 6), round(float(y[v]), 6)

    # membranes: each wall's count and lateral distance (Σ dist, in edge order)
    membrane = s.path == MEMBRANE
    m_edges = np.flatnonzero(membrane)
    m_wall = np.where(s.kind[s.a[m_edges]] == WALL, s.a[m_edges], s.b[m_edges])
    m_cell = np.where(s.kind[s.a[m_edges]] == WALL, s.b[m_edges], s.a[m_edges])
    edge_dist = s.dist * MM
    n_membranes = np.bincount(m_wall, minlength=n)
    lateral = np.zeros(n)
    np.add.at(lateral, m_wall, edge_dist[m_edges])
    lateral[s.kind != WALL] = np.nan

    one = (s.kind == WALL) & (n_membranes == 1)
    membrane_cell = np.zeros(n, dtype=np.int64)
    membrane_cell[m_wall] = m_cell                                 # a cell of each wall (its only one if one)
    border_wall = one & (cgroup[membrane_cell] == 2)
    border_aerenchyma = one & ~border_wall

    # the wall edges to junctions, as rebuilt: kept when their wall has at least two junctions
    wall_edges = np.flatnonzero(s.path == WALL_PATH)
    a_kind, b_kind = s.kind[s.a[wall_edges]], s.kind[s.b[wall_edges]]
    to_junction = ((a_kind == WALL) & (b_kind == JUNCTION)) | ((a_kind == JUNCTION) & (b_kind == WALL))
    j_edges = wall_edges[to_junction]
    j_wall = np.where(s.kind[s.a[j_edges]] == WALL, s.a[j_edges], s.b[j_edges])
    j_junc = np.where(s.kind[s.a[j_edges]] == WALL, s.b[j_edges], s.a[j_edges])
    kept = np.bincount(j_wall, minlength=n)[j_wall] >= 2
    junction_edge = np.zeros(s.path.size, dtype=bool)
    junction_edge[j_edges[kept]] = True
    j_edges, j_wall, j_junc = j_edges[kept], j_wall[kept], j_junc[kept]

    # the legacy's junction ids: in order of first appearance, scanning the walls in order and each wall's junctions
    # in its neighbour order (connection order: the junctions come after the walls in networkx's node order)
    scan = np.lexsort((j_edges, position[j_wall]))
    first_seen = np.unique(j_junc[scan], return_index=True)[1]
    junction_order = j_junc[scan][np.sort(first_seen)]

    # wall lengths (µm): the walls'; a border junction (two border walls) Σ of their length/4
    wall_length = np.zeros(n)
    wall_length[walls] = s.wall_length[walls] * MM
    border_count = np.bincount(j_junc[border_wall[j_wall]], minlength=n)
    border_junction = (s.kind == JUNCTION) & (border_count == 2)
    quarter = np.zeros(n)
    np.add.at(quarter, j_junc[border_wall[j_wall]], wall_length[j_wall[border_wall[j_wall]]] / 4.)
    wall_length[border_junction] = quarter[border_junction]

    edge_length = s.length * MM
    edge_length[j_edges] = wall_length[j_wall] / 2
    edge_lateral = np.full(s.path.size, np.nan)
    edge_lateral[j_edges] = lateral[j_wall]

    # gravity centre: the endodermis cells, in cell order
    endo = cells[cgroup[cells] == 3]
    if endo.size:
        x_grav = np.float64(_sequential_sum(x[endo]) / endo.size)
        y_grav = np.float64(_sequential_sum(y[endo]) / endo.size)
    else:
        x_grav = y_grav = np.float64(np.nan)

    # the walls' distances to the centre; cortex extent; epidermis mean distance and perimeter
    d_wall = np.full(n, np.nan)
    for v in walls:                    # numpy scalars: the scalar ``** 2`` is libm's pow, not the arrays' x*x (1 ulp)
        d_wall[v] = np.sqrt((x[v] - x_grav) ** 2 + (y[v] - y_grav) ** 2)
    distance_center_grav = np.full(n, np.nan)
    distance_center_grav[walls] = d_wall[walls]
    cortex_walls = np.unique(m_wall[cgroup[m_cell] == 4])
    distance_max_cortex = float(max(0., d_wall[cortex_walls].max())) if cortex_walls.size else 0.
    distance_min_cortex = float(d_wall[cortex_walls].min()) if cortex_walls.size else np.inf
    epi_count = np.bincount(m_wall[cgroup[m_cell] == 2], minlength=n)[walls]
    epi_d = np.repeat(d_wall[walls], epi_count)
    distance_avg_epi = _sequential_sum(epi_d) / epi_d.size if epi_d.size else 0.
    perimeter = 2 * np.pi * distance_avg_epi * 1.0E-04

    # cell ranks
    ranks, layer_dist, n_layer, xylem_distance, outer, stele, extra_protosieve = _ranks(
        s, cells, cgroup, x, y, x_grav, y_grav)

    # cell sets (the network's elif chain, cell order); the cell manager's, by tissue
    xylem, sieve, protosieve, passage, intercellular = [], [], [], [], []
    for v in cells:
        c, t = cgroup[v], tissue[v]
        if c in XYLEM_CGROUPS:
            xylem.append(v)
        elif c == 11 or t in SIEVE_TYPES:
            sieve.append(v)
        elif c == 23 or t in PROTOSIEVE_TYPES:
            protosieve.append(v)
        elif t == 'passage':
            passage.append(v)
        elif t in INTERCELLULAR_TYPES:
            intercellular.append(v)
    protosieve += extra_protosieve
    typed = {name: cells[np.isin(tissue[cells], types)] for name, types in (
        ("intercellular", INTERCELLULAR_TYPES), ("passage", ("passage",)), ("xylem", XYLEM_TYPES),
        ("sieve", SIEVE_TYPES), ("protosieve", PROTOSIEVE_TYPES))}

    wall_cells = np.full((n, 2), -1, dtype=np.int64)
    by_cell = np.lexsort((position[m_cell], m_wall))                # a wall's cells in cell order
    first = np.r_[True, m_wall[by_cell][1:] != m_wall[by_cell][:-1]]
    wall_cells[m_wall[by_cell][first], 0] = m_cell[by_cell][first]
    second = ~first
    wall_cells[m_wall[by_cell][second], 1] = m_cell[by_cell][second]

    ratios = _plasmodesmata_ratios(s, position, cgroup, np.asarray(intercellular, dtype=np.int64), outer, edge_length)

    return SectionGeometry(
        legacy_order=legacy_order, walls=walls, junctions=junctions, cells=cells, kind=s.kind, cgroup=cgroup,
        tissue=tissue,
        x=x, y=y, wall_length=wall_length, cell_area=s.cell_area * 1000 ** 2, lateral_distance=lateral,
        n_membranes=n_membranes, border_wall=border_wall, border_aerenchyma=border_aerenchyma,
        border_junction=border_junction, junction_edge=junction_edge, edge_length=edge_length,
        edge_lateral_distance=edge_lateral, edge_dist=edge_dist, gravity_center=(x_grav, y_grav),
        distance_center_grav=distance_center_grav, distance_max_cortex=distance_max_cortex,
        distance_min_cortex=distance_min_cortex, distance_avg_epi=distance_avg_epi, perimeter=perimeter,
        cell_rank=ranks, layer_dist=layer_dist, n_layer=n_layer,
        xylem_80_percentile_distance=float(np.percentile(xylem_distance, 80)) if len(xylem_distance) else 0,
        outercortex_connec_rank=outer, stele_connec_rank=stele,
        concentrated_outer_cortex=ratios[0], concentrated_cortex_cortex=ratios[1], concentrated_cortex_endo=ratios[2],
        xylem_cells=np.asarray(xylem, dtype=np.int64), sieve_cells=np.asarray(sieve, dtype=np.int64),
        protosieve_cells=np.asarray(protosieve, dtype=np.int64), passage_cells=np.asarray(passage, dtype=np.int64),
        intercellular_cells=np.asarray(intercellular, dtype=np.int64), typed=typed, wall_cells=wall_cells,
        junction_order=junction_order)


def _ranks(s, cells, cgroup, x, y, x_grav, y_grav):
    """
    The legacy's cell ranks (``_rank_cells_from_graph``): the cgroup, then the cortex layers 40–49 from the endodermis
    and from the outer layer, the stele layers 50–61 from the pericycle (or endodermis). Its sweeps update in place,
    but within one sweep a cell's test never reads a rank given in the same sweep, so each sweep is computed at once.
    """
    n = s.kind.size
    rank = np.full(n, -1, dtype=np.int64)
    c = cgroup[cells].copy()
    c[(c == 19) | (c == 20)] = 13
    c[c == 21] = 16
    c[c == 26] = 12
    rank[cells] = c
    dist = np.hypot(x[cells] - x_grav, y[cells] - y_grav)
    contributions = [(c, dist)]                                    # (layer index, distance) in pass order
    xylem_distance = dist[c == 13]
    present = np.bincount(c, minlength=N_LAYERS)
    stele = 3 if present[16] == 0 else 16
    outer = 2 if present[1] == 0 else 1

    # the cells' cell neighbours (any path): a (cells × nodes) incidence, to test "rank r among the neighbours"
    both = (s.kind[s.a] == PROTOPLAST) & (s.kind[s.b] == PROTOPLAST)
    u, v = s.a[both], s.b[both]
    row = np.empty(n, dtype=np.int64)
    row[cells] = np.arange(cells.size)

    def neighbour_has(values):
        """Per cell: whether a neighbour cell's rank is in *values*."""
        hit = np.isin(rank, values)
        out = np.zeros(cells.size, dtype=bool)
        out[row[u[hit[v]]]] = True
        out[row[v[hit[u]]]] = True
        return out

    current = rank[cells]
    cortex, stele_cells = current == 4, np.isin(current, STELE_RANKS)
    to_endo = cortex & neighbour_has([3])
    to_outer = cortex & ~to_endo & neighbour_has([outer])
    to_stele = stele_cells & neighbour_has([stele])
    extra_protosieve = cells[to_stele & (cgroup[cells] == 23)].tolist()
    assign_masks = [(to_endo, 40), (to_outer, 49), (to_stele, 50)]
    for mask, value in assign_masks:
        rank[cells[mask]] = value
    # contributions: the cells in cell order, each to its own layer (one pass)
    _contribute_in_cell_order(contributions, assign_masks, dist)

    for iteration in range(12):
        current = rank[cells]
        masks = []
        if iteration < 4:
            cortex = current == 4
            inner = cortex & neighbour_has([40 + iteration])
            outer_side = cortex & ~inner & neighbour_has([49 - iteration])
            masks += [(inner, 41 + iteration), (outer_side, 48 - iteration)]
        stele_cells = np.isin(current, STELE_RANKS)
        if iteration < 10:
            masks.append((stele_cells & neighbour_has([50 + iteration]), 51 + iteration))
        else:
            masks.append((stele_cells, 61))
        for mask, value in masks:
            rank[cells[mask]] = value
        _contribute_in_cell_order(contributions, masks, dist)

    layer_dist = np.zeros(N_LAYERS)
    n_layer = np.zeros(N_LAYERS, dtype=int)
    for index, d in contributions:
        np.add.at(layer_dist, index, d)                               # in order: sequential per layer
        np.add.at(n_layer, index, 1)
    filled = n_layer > 0
    layer_dist[filled] /= n_layer[filled]
    return rank, layer_dist, n_layer, xylem_distance, outer, stele, extra_protosieve


def _contribute_in_cell_order(contributions, masks, dist):
    """One pass's (layer, distance) contributions, in cell order (the legacy loops over the cells once per pass)."""
    index = np.full(dist.size, -1, dtype=np.int64)
    for mask, value in masks:
        index[mask] = value
    hit = index >= 0
    contributions.append((index[hit], dist[hit]))


def _plasmodesmata_ratios(s, position, cgroup, intercellular, outer, edge_length):
    """
    The legacy's ``compute_cell_surface``: at the cortex interfaces (outer layer–cortex, cortex–cortex,
    cortex–endodermis), the plasmodesmata length over the length between non-intercellular cells, each summed over the
    cells in cell order and over their neighbours in networkx's order.
    """
    n = s.kind.size
    pd_edges = np.flatnonzero(s.path == PLASMODESMATA)
    node, other, e = _neighbour_keys(n, s.a, s.b, position, pd_edges)
    interc = np.zeros(n, dtype=bool)
    interc[intercellular] = True
    gi, gj = cgroup[node], cgroup[other]
    counted = np.isin(gi, (outer, 3, 4)) & ~interc[node] & (s.kind[node] == PROTOPLAST)
    length = edge_length[e]
    interfaces = (((gi == outer) & (gj == 4)) | ((gi == 4) & (gj == outer)),      # disjoint: outer is 1 or 2
                  (gi == 4) & (gj == 4),
                  ((gi == 3) & (gj == 4)) | ((gi == 4) & (gj == 3)))
    ratios = []
    for match in interfaces:
        m = counted & match
        total = _sequential_sum(length[m])
        cross = _sequential_sum(length[m & ~interc[other]])
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios.append(float(np.float64(total) / np.float64(cross)))
    return ratios


AXIAL_JUNCTION = 5                                                       # GRANAP's junctions between sections

NODE_VARIABLES = ("kind", "cgroup", "tissue", "section_x", "section_y", "wall_length", "cell_area")
EDGE_VARIABLES = ("network_path", "edge_length", "edge_dist")


def section_parts(ds) -> dict:
    """
    {SubOrgan vid: (nodes, edges)}: each section's Compartments (DataStructure indices, in order) and its own
    Connections (edge indices, in order; the axial junctions to the neighbouring sections excluded).
    """
    owner = ds.owner("SubOrgan")
    ids = ds.entity_ids("SubOrgan")
    view = ds.to_graph_view()
    path = np.asarray(ds.get("network_path"))
    inner = (owner[view.tail] == owner[view.head]) & (path != AXIAL_JUNCTION)
    node_order = np.argsort(owner, kind="stable")
    node_bounds = np.searchsorted(owner[node_order], np.arange(ids.size + 1))
    edge_owner = owner[view.tail]
    edges = np.flatnonzero(inner)
    edge_order = edges[np.argsort(edge_owner[edges], kind="stable")]
    edge_bounds = np.searchsorted(edge_owner[edge_order], np.arange(ids.size + 1))
    return {int(vid): (node_order[node_bounds[i]:node_bounds[i + 1]], edge_order[edge_bounds[i]:edge_bounds[i + 1]])
            for i, vid in enumerate(ids)}


def section_of(ds, nodes, edges) -> Section:
    """The Section of the Compartments *nodes* and Connections *edges* of the DataStructure *ds*."""
    view = ds.to_graph_view()
    local = np.full(ds.n_nodes(), -1, dtype=np.int64)
    local[nodes] = np.arange(nodes.size)
    node = {name: np.asarray(ds.get(name))[nodes] for name in NODE_VARIABLES}
    edge = {name: np.asarray(ds.get(name))[edges] for name in EDGE_VARIABLES}
    return Section(kind=node["kind"].astype(np.int64), cgroup=node["cgroup"].astype(float), tissue=node["tissue"],
                   x=node["section_x"].astype(float), y=node["section_y"].astype(float),
                   wall_length=node["wall_length"].astype(float), cell_area=node["cell_area"].astype(float),
                   a=local[view.tail[edges]], b=local[view.head[edges]], path=edge["network_path"].astype(np.int64),
                   length=edge["edge_length"].astype(float), dist=edge["edge_dist"].astype(float))


CORTEX_TYPES = ("cortex", "outercortex", "innercortex")


def phi_thick_cells(geometry: SectionGeometry, n_phi_layers: int, phi_type: int) -> np.ndarray:
    """
    The legacy's φ-thickened cortex cells (``tag_phi_thick_cells``), bool per node: *n_phi_layers* rank layers of the
    cortex (by tissue) from its inner side (phi_type 1), its outer side (2), or both sides of its median rank (3).
    """
    out = np.zeros(geometry.cgroup.size, dtype=bool)
    cortex = geometry.cells[np.isin(geometry.tissue[geometry.cells], CORTEX_TYPES)]
    if cortex.size == 0 or n_phi_layers <= 0:
        return out
    rank = geometry.cell_rank[cortex]
    lo, hi, median = rank.min(), rank.max(), np.median(rank.tolist())
    tagged = []
    for i in range(int(n_phi_layers)):
        if phi_type == 1:
            tagged.append(rank == lo + i)
        elif phi_type == 2:
            tagged.append(rank == hi - i)
        elif phi_type == 3:
            tagged += [rank == int(median) - i, rank == int(median) + i]
    if tagged:
        out[cortex[np.logical_or.reduce(tagged)]] = True
    return out
