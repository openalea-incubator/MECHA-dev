"""
MECHA's conductances (devplan_metafspm.md, §8, step 3): the conductance K of every edge of every section, and the
boundary terms of their nodes, as the legacy's ``HydraulicMatrixBuilder`` computes them (its rules and formulas,
element by element, so that the values are exact), vectorised over all the sections' edges at once.

    network = assemble(n_nodes, n_edges, section_geometries)       the sections' geometry on the whole DataStructure
    K, in_matrix = edge_conductances(network, P, barrier, height)  per edge (P: the parameters, per edge)
    soil, axial, axial_rhs = boundary_terms(network, P, barrier, height)   per node (P per node)

Units are MECHA's: lengths in µm, conductances in cm³ hPa⁻¹ d⁻¹ (cm hPa⁻¹ d⁻¹ per wall surface for kw).
"""
from dataclasses import dataclass

import numpy as np

from openalea.mecha.geometry import MEMBRANE, PLASMODESMATA, WALL, WALL_PATH, XYLEM_CGROUPS

# ── The legacy's tables ──
# kw per barrier code (HydraulicData.get_wall_conductivities): each key's source parameter
_KW_KEYS = ("kw_endo_endo", "kw_exo_exo", "kw_exo_epi", "kw_exo_cortex", "kw_cortex_cortex", "kw_endo_peri",
            "kw_endo_cortex", "kw_passage")
_KW_CONFIGS = {   # barrier: sources of _KW_KEYS (kw_septa is always kw_septa, kw_phi_thick always the lignin value)
    0: ("kw", "kw", "kw", "kw", "kw", "kw", "kw", "kw"),
    1: ("kw_casparian", "kw", "kw", "kw", "kw", "kw", "kw", "kw"),
    2: ("kw_casparian", "kw", "kw", "kw", "kw", "kw_suberin_in", "kw_suberin_out", "kw"),
    3: ("kw_casparian", "kw", "kw", "kw", "kw", "kw_suberin_in", "kw_suberin_out", "kw_suberin_in"),
    4: ("kw_casparian", "kw_casparian", "kw", "kw", "kw", "kw_suberin_in", "kw_suberin_out", "kw_suberin_in"),
    5: ("kw_casparian", "kw_casparian", "kw", "kw", "kw", "kw", "kw", "kw"),
    6: ("kw_casparian", "kw_casparian", "kw_suberin_out", "kw_suberin_in", "kw", "kw", "kw", "kw"),
    7: ("kw", "kw_casparian", "kw", "kw", "kw", "kw", "kw", "kw"),
    8: ("kw_casparian", "kw_casparian", "kw_suberin_out", "kw_suberin_in", "kw", "kw_suberin_in", "kw_suberin_out",
        "kw"),
    9: ("kw_casparian", "kw_casparian", "kw_lignin", "kw", "kw", "kw", "kw", "kw"),
}
# a wall's interface (its two cells' cgroups, sorted) -> its kw key (HydraulicData.interface_kw_key_map)
_KW_INTERFACES = {(3, 3): "kw_endo_endo", (3, 4): "kw_endo_cortex", (3, 5): "kw_endo_peri", (3, 11): "kw_endo_peri",
                  (3, 12): "kw_endo_peri", (3, 13): "kw_endo_peri", (3, 16): "kw_endo_peri", (1, 1): "kw_exo_exo",
                  (1, 2): "kw_exo_epi", (1, 4): "kw_exo_cortex", (4, 4): "kw_cortex_cortex"}
# plasmodesmata: the interfaces whose conductance factor applies (HydraulicData.interface_kpl_factor_map, as resolved
# against the legacy's kpl configuration: the other keys are not found and give 1, §2 a)
KPL_FACTOR_INTERFACES = {(12, 12): "kpl_phloem_companion_cell_factor", (11, 12): "kpl_phloem_companion_cell_factor",
                         (11, 16): "kpl_phloem_sieve_tube_factor", (11, 13): "kpl_phloem_sieve_tube_factor",
                         (5, 11): "kpl_phloem_sieve_tube_factor"}
# plasmodesmata frequency per unit height, by interface (HydraulicData.fplxheight_map); others: fplxheight_default
CGROUP_NAMES = {1: "exo", 2: "epi", 3: "endo", 4: "cortex", 5: "stele", 11: "sieve", 12: "comp", 13: "xylem",
                16: "peri", 17: "tparenchyma", 18: "ttracheid", 23: "protophloem"}
FPLXHEIGHT = {(1, 1): 8.0E5, (2, 2): 8.0E5, (1, 2): 1.08E6, (1, 4): 2.28E6, (4, 4): 8.6E5, (3, 4): 8.8E5,
              (3, 3): 6.4E5, (3, 16): 9.6E5, (3, 5): 9.6E5, (3, 17): 1.08E6, (3, 18): 0.0, (5, 16): 1.08E6,
              (5, 11): 9.0E5, (5, 23): 9.0E5, (5, 12): 9.8E5, (5, 13): 6.4E5, (5, 5): 6.4E5, (5, 17): 6.4E5,
              (5, 18): 0.0, (11, 12): 1.76E6, (11, 16): 7.2E5, (11, 13): 0.0, (11, 11): 0.0, (11, 23): 0.0,
              (11, 17): 6.4E5, (11, 18): 0.0, (12, 13): 9.8E5, (12, 16): 7.0E5, (12, 12): 6.8E5, (12, 23): 6.8E5,
              (12, 17): 1.08E6, (12, 18): 0.0, (13, 16): 1.08E6, (13, 13): 6.4E5, (13, 23): 0.0, (13, 17): 1.08E6,
              (13, 18): 1.76E6, (17, 17): 8.0e5, (17, 18): 0.0, (18, 18): 0.0, (23, 23): 0.0}


def fplxheight_name(interface) -> str:
    return "fplxheight_" + "_".join(CGROUP_NAMES[c] for c in interface)


FPLXHEIGHT_PARAMETERS = {fplxheight_name(k): v for k, v in FPLXHEIGHT.items()}


@dataclass
class Network:
    """The sections' MECHA geometry on the whole DataStructure (nodes and edges in its order; µm)."""
    section_of_node: np.ndarray
    section_of_edge: np.ndarray
    a: np.ndarray                 # edges' end nodes
    b: np.ndarray
    path: np.ndarray
    in_section: np.ndarray        # bool per edge: an edge of a section (not an axial junction)
    junction_edge: np.ndarray     # bool per edge: a rebuilt wall–junction edge
    edge_length: np.ndarray
    edge_lateral_distance: np.ndarray
    edge_dist: np.ndarray
    cgroup: np.ndarray            # int per node
    kind: np.ndarray              # GRANAP's kind per node (-1: not in a section)
    rank: np.ndarray
    intercellular: np.ndarray     # bool per node (the cell manager's, by tissue)
    passage: np.ndarray
    phi_thick: np.ndarray
    wall_cells: np.ndarray        # (n, 2) global nodes, -1
    distance_center_grav: np.ndarray
    x: np.ndarray
    wall_length: np.ndarray
    border_wall: np.ndarray
    border_junction: np.ndarray
    xylem: np.ndarray             # int per node: occurrences in the network's xylem list (cgroup)
    protosieve: np.ndarray        # int per node: occurrences in the protosieve list (duplicates, §2 f)
    outercortex_connec_rank: np.ndarray   # per section
    concentrated: np.ndarray              # (sections, 3): outer–cortex, cortex–cortex, cortex–endodermis


def assemble(n_nodes: int, n_edges: int, geometries: list, phi_thick=None) -> Network:
    """
    The Network of the sections *geometries* [(nodes, edges, SectionGeometry)], and of their φ-thickened cells
    (*phi_thick*: a bool per section's node, or None).
    """
    def nodes_full(fill, dtype=float, shape=()):
        return np.full((n_nodes,) + shape, fill, dtype=dtype)

    def edges_full(fill, dtype=float):
        return np.full(n_edges, fill, dtype=dtype)

    net = Network(
        section_of_node=nodes_full(-1, np.int64), section_of_edge=edges_full(-1, np.int64),
        a=edges_full(-1, np.int64), b=edges_full(-1, np.int64), path=edges_full(-1, np.int64),
        in_section=edges_full(False, bool), junction_edge=edges_full(False, bool), edge_length=edges_full(np.nan),
        edge_lateral_distance=edges_full(np.nan), edge_dist=edges_full(np.nan), cgroup=nodes_full(-1, np.int64),
        kind=nodes_full(-1, np.int64),
        rank=nodes_full(-1, np.int64), intercellular=nodes_full(False, bool), passage=nodes_full(False, bool),
        phi_thick=nodes_full(False, bool), wall_cells=nodes_full(-1, np.int64, (2,)),
        distance_center_grav=nodes_full(np.nan), x=nodes_full(np.nan), wall_length=nodes_full(0.),
        border_wall=nodes_full(False, bool), border_junction=nodes_full(False, bool), xylem=nodes_full(0, np.int64),
        protosieve=nodes_full(0, np.int64), outercortex_connec_rank=np.zeros(len(geometries), dtype=np.int64),
        concentrated=np.zeros((len(geometries), 3)))
    for i, (nodes, edges, geo) in enumerate(geometries):
        net.section_of_node[nodes] = i
        net.section_of_edge[edges] = i
        net.in_section[edges] = True
        net.cgroup[nodes] = geo.cgroup
        net.kind[nodes] = geo.kind
        net.rank[nodes] = geo.cell_rank
        net.intercellular[nodes[geo.typed["intercellular"]]] = True
        net.passage[nodes[geo.typed["passage"]]] = True
        if phi_thick is not None:
            net.phi_thick[nodes] = phi_thick[i]
        cells = geo.wall_cells
        net.wall_cells[nodes] = np.where(cells >= 0, nodes[np.maximum(cells, 0)], -1)
        net.distance_center_grav[nodes] = geo.distance_center_grav
        net.x[nodes] = geo.x
        net.wall_length[nodes] = geo.wall_length
        net.border_wall[nodes] = geo.border_wall
        net.border_junction[nodes] = geo.border_junction
        np.add.at(net.xylem, nodes[geo.xylem_cells], 1)
        np.add.at(net.protosieve, nodes[geo.protosieve_cells], 1)
        net.outercortex_connec_rank[i] = geo.outercortex_connec_rank
        net.concentrated[i] = (geo.concentrated_outer_cortex, geo.concentrated_cortex_cortex,
                               geo.concentrated_cortex_endo)
        net.junction_edge[edges] = geo.junction_edge
        net.edge_length[edges] = geo.edge_length
        net.edge_lateral_distance[edges] = geo.edge_lateral_distance
        net.edge_dist[edges] = geo.edge_dist
    return net


def set_edge_ends(net: Network, tail, head, path) -> None:
    net.a, net.b, net.path = np.asarray(tail), np.asarray(head), np.asarray(path, dtype=np.int64)


def _pow2(values) -> np.ndarray:
    """v ** 2 as the legacy's Python scalars compute it (libm's pow), per distinct value."""
    values = np.asarray(values, dtype=float)
    unique, inverse = np.unique(values, return_inverse=True)
    return np.array([float(v) ** 2 for v in unique])[inverse].reshape(values.shape)


def _per_code(codes, table: dict, P: dict, key_index: int) -> np.ndarray:
    """Per entity, the parameter its barrier code's configuration names for key *key_index* (codes not in the table:
    barrier 0's)."""
    out = np.empty(codes.size)
    known = np.isin(codes, list(table))
    for code in np.unique(np.where(known, codes, 0)):
        mask = np.where(known, codes, 0) == code
        out[mask] = np.asarray(P[table[int(code)][key_index]])[mask]
    return out


def wall_configuration(barrier, P) -> dict:
    """{kw key: per entity value}: the wall conductivities of each entity's barrier code."""
    codes = np.asarray(barrier).astype(np.int64)
    config = {key: _per_code(codes, _KW_CONFIGS, P, i) for i, key in enumerate(_KW_KEYS)}
    config["kw_septa"] = np.asarray(P["kw_septa"], dtype=float)
    config["kw_phi_thick"] = np.asarray(P["kw_lignin"], dtype=float)
    return config


def _interface_codes(gi, gj):
    lo, hi = np.minimum(gi, gj), np.maximum(gi, gj)
    return lo * 1000 + hi


def _select(code, table: dict, values: dict, default) -> np.ndarray:
    """Per entity, values[table[interface]] for the interfaces in *table*, else *default* (arrays)."""
    out = np.array(default, dtype=float, copy=True)
    for (i, j), key in table.items():
        mask = code == i * 1000 + j
        if mask.any():
            out[mask] = np.asarray(values[key])[mask]
    return out


def edge_conductances(net: Network, P: dict, barrier, height):
    """
    (K, in_matrix): the conductance of every edge (cm³ hPa⁻¹ d⁻¹), and whether the legacy's matrix has it (its wall,
    membrane and plasmodesmata edges; not the wall_air and air_link edges, §2 i, nor the wall edges its junction
    rebuild drops). P: {parameter: value per edge}; barrier: code per edge; height: µm per edge.
    """
    n = net.path.size
    K = np.zeros(n)
    barrier = np.asarray(barrier).astype(np.int64)
    height = np.asarray(height, dtype=float)
    thickness = np.asarray(P["thickness"], dtype=float)
    kw = np.asarray(P["kw"], dtype=float)
    kw_config = wall_configuration(barrier, P)

    wall_edge = net.in_section & net.junction_edge
    membrane = net.in_section & (net.path == MEMBRANE)
    plasmodesmata = net.in_section & (net.path == PLASMODESMATA)
    in_matrix = wall_edge | membrane | plasmodesmata

    # the wall end of wall and membrane edges, and its cells
    a_is_wall = net.kind[net.a] == WALL
    wall_end = np.where(a_is_wall, net.a, net.b)
    other_end = np.where(a_is_wall, net.b, net.a)
    c0, c1 = net.wall_cells[wall_end, 0], net.wall_cells[wall_end, 1]
    has0, has1 = c0 >= 0, c1 >= 0
    c0s, c1s = np.maximum(c0, 0), np.maximum(c1, 0)
    n_cells = has0.astype(int) + has1

    def count(flag):
        return (has0 & flag[c0s]).astype(int) + (has1 & flag[c1s])

    cg0, cg1 = net.cgroup[c0s], net.cgroup[c1s]
    interface = np.where(n_cells == 2, _interface_codes(cg0, cg1), -1)

    # ── walls (HydraulicMatrixBuilder._fill_wall) ──
    e = np.flatnonzero(wall_edge)
    t, h = thickness[e], height[e]
    temp = 1.0E-04 * ((net.edge_lateral_distance[e] + h) * t - _pow2(t)) / net.edge_length[e]
    n_interc = count(net.intercellular)[e]
    n_xyl = count(np.isin(net.cgroup, XYLEM_CGROUPS))[e]
    has_passage = count(net.passage)[e] > 0
    xylem_pieces = np.asarray(P["xylem_pieces"], dtype=float)[e] > 0
    ghost = ((n_interc >= 2) & (barrier[e] > 0)) | ((n_xyl >= 2) & xylem_pieces)
    phi = (n_cells[e] == 2) & net.phi_thick[c0s[e]] & net.phi_thick[c1s[e]] & (net.rank[c0s[e]] == net.rank[c1s[e]])
    sub = {key: values[e] for key, values in kw_config.items()}
    kw_val = _select(interface[e], _KW_INTERFACES, sub, kw[e])
    kw_val = np.where(phi, sub["kw_phi_thick"], kw_val)
    kw_val = np.where(has_passage, sub["kw_passage"], kw_val)
    kw_val = np.where(ghost, sub["kw_septa"], kw_val)
    K_wall = kw_val * temp
    K_wall = np.where(ghost & (n_interc < 2), 1.0E-16, K_wall)        # xylem pieces' septa
    K[e] = K_wall

    # ── membranes (HydraulicMatrixBuilder._fill_membrane) ──
    e = np.flatnonzero(membrane)
    cell, wall = other_end[e], wall_end[e]
    cg = net.cgroup[cell]
    b = barrier[e]
    kaqp = np.asarray(P["kaqp"], dtype=float)[e]
    kaqp_of = {name: kaqp * np.asarray(P[f"kaqp_{name}_factor"], dtype=float)[e]
               for name in ("stele", "endo", "exo", "epi", "cortex")}
    a_cortex, b_cortex = P["a_cortex"][e], P["b_cortex"][e]
    cortex_value = a_cortex * net.distance_center_grav[wall] * 1.0E-04 + b_cortex
    xylem = np.isin(cg, XYLEM_CGROUPS)
    kaqp_curr = np.select(
        [cg == 1, cg == 2, cg == 3, xylem & (b > 0), xylem, cg > 4, net.intercellular[cell] & (b > 0), cg == 4],
        [kaqp_of["exo"], kaqp_of["epi"], kaqp_of["endo"], kaqp_of["stele"] * 10000, kaqp_of["stele"],
         kaqp_of["stele"], np.asarray(P["k_interc"], dtype=float)[e], cortex_value], 0.0)
    n_endo = count(net.cgroup == 3)[e]
    n_exo = count(net.cgroup == 1)[e]
    n_stele = count(net.cgroup > 4)[e]
    n_passage = count(net.passage)[e]
    sub = {key: values[e] for key, values in kw_config.items()}
    kw_val = _select(interface[e], _KW_INTERFACES, sub, kw[e])
    kw_val = np.where((n_endo > 0) & (n_passage > 0), sub["kw_passage"], kw_val)
    barrier_interface = (n_endo >= 2) | (n_exo >= 2) | ((n_stele > 0) & (n_endo > 0))
    t = thickness[e]
    kmb = np.asarray(P["kmb"], dtype=float)[e]
    with np.errstate(divide="ignore"):
        K_mb = 1 / (1 / (kw_val / (t / 2 * 1.0E-04)) + 1 / (kmb + kaqp_curr)) * 1.0E-08 * (height[e]
                                                                                          + net.edge_dist[e]) * net.edge_length[e]
    K_mb = np.where(kw_val == 0.0, 0.0, K_mb)
    K_mb = np.where(~barrier_interface & (kaqp_curr == 0.0), 1.00E-16, K_mb)
    K[e] = K_mb

    # ── plasmodesmata (HydraulicMatrixBuilder._fill_plasmodesmata) ──
    e = np.flatnonzero(plasmodesmata)
    gi, gj = net.cgroup[net.a[e]], net.cgroup[net.b[e]]
    code = _interface_codes(gi, gj)
    b = barrier[e]
    kpl = np.asarray(P["kpl"], dtype=float)[e]
    sub = {name: np.asarray(P[name], dtype=float)[e] for name in FPLXHEIGHT_PARAMETERS}
    fplx = _select(code, {k: fplxheight_name(k) for k in FPLXHEIGHT}, sub,
                   np.asarray(P["fplxheight_default"], dtype=float)[e])
    base_area = fplx * 1.0E-04 * net.edge_length[e]
    factors = {name: np.asarray(P[name], dtype=float)[e] for name in set(KPL_FACTOR_INTERFACES.values())}
    factor = _select(code, KPL_FACTOR_INTERFACES, factors, np.ones(e.size))
    with np.errstate(divide="ignore", invalid="ignore"):
        aperture = np.where(gi == gj, factor, 2.0 * factor / (factor + 1.0))
    section = net.section_of_edge[e]
    outer = net.outercortex_connec_rank[section]
    modifier = np.select([code == np.minimum(outer, 4) * 1000 + np.maximum(outer, 4), code == 4004, code == 3004],
                         [net.concentrated[section, 0], net.concentrated[section, 1], net.concentrated[section, 2]], 1.0)
    modifier = np.where(b > 0, modifier, 1.0)
    interc = net.intercellular[net.a[e]] | net.intercellular[net.b[e]]
    temp_factor = np.select([interc & (b > 0), code == 13013, (b > 0) & ((gi == 13) | (gj == 13))],
                            [0.0, 10000.0 * base_area, 0.0], aperture * base_area * modifier)
    K[e] = kpl * temp_factor
    return K, in_matrix


def boundary_terms(net: Network, P: dict, barrier, height):
    """
    Per node (cm³ hPa⁻¹ d⁻¹): the soil term of the border walls and junctions in contact (``_apply_soil_boundary``),
    and the axial term of the xylem cells (barrier > 0, ``k_xyl``) or protosieve cells (barrier 0, ``k_sieve``):
    (soil, axial on the matrix diagonal, axial in the right-hand side). P, barrier, height per node.
    """
    n = net.cgroup.size
    kw = np.asarray(P["kw"], dtype=float)
    thickness = np.asarray(P["thickness"], dtype=float)
    height = np.asarray(height, dtype=float)
    contact = net.x >= np.asarray(P["x_contact"], dtype=float)
    soil = np.zeros(n)
    w = np.flatnonzero(net.border_wall & contact)
    soil[w] = kw[w] * (1.0E-04 * (net.wall_length[w] / 2 * height[w]) / (thickness[w] / 2))
    j = np.flatnonzero(net.border_junction & contact)
    soil[j] = kw[j] * (1.0E-04 * (net.wall_length[j] * height[j]) / (thickness[j] / 2))

    barrier = np.asarray(barrier).astype(np.int64)
    k_xyl, k_sieve = np.asarray(P["k_xyl"], dtype=float), np.asarray(P["k_sieve"], dtype=float)
    xylem = (barrier > 0) & (net.xylem > 0)
    protosieve = (barrier == 0) & (net.protosieve > 0)
    axial_rhs = np.where(xylem, k_xyl, np.where(protosieve, k_sieve, 0.))
    axial = axial_rhs.copy()
    twice = protosieve & (net.protosieve > 1)              # the duplicated protosieve cells (§2 f): k_sieve added twice
    for _ in range(int(net.protosieve.max(initial=0)) - 1):
        axial = np.where(twice, axial + k_sieve, axial)
    return soil, axial, axial_rhs


def cortex_parameters(geo, kaqp_cortex: float, ratio_cortex: float, height: float) -> tuple:
    """
    (a_cortex, b_cortex) of one section (``Mecha._calculate_cortex_parameters``): the cortex aquaporin activity's
    linear gradient with the distance to the centre, uniform when ratio_cortex is 1. Its sums run over the cortex cells
    (cgroup 4) in cell order, and over each cell's walls in wall order (the walls come first in networkx's order).
    """
    if ratio_cortex == 1:
        return 0.0, kaqp_cortex
    nodes = geo.cells[geo.cgroup[geo.cells] == 4]
    walls = geo.wall_cells
    shift = (ratio_cortex * geo.distance_max_cortex - geo.distance_min_cortex) / (1 - ratio_cortex)
    tot_surf_cortex, temp = 0.0, 0.0
    by_cell = {}
    for w in geo.walls:
        for c in walls[w]:
            if c >= 0:
                by_cell.setdefault(int(c), []).append(int(w))
    for c in nodes:
        for w in by_cell.get(int(c), ()):
            dist_cell = np.sqrt(np.square(geo.x[w] - geo.x[c]) + np.square(geo.y[w] - geo.y[c]))
            surf = (height + dist_cell) * geo.wall_length[w] * 1.0E-08
            tot_surf_cortex += surf
            temp += surf * 1.0E-04 * (geo.distance_center_grav[w] + shift)
    if temp == 0.0:
        return 0.0, kaqp_cortex
    a_cortex = kaqp_cortex * tot_surf_cortex / temp
    return a_cortex, a_cortex * 1.0E-04 * (ratio_cortex * geo.distance_max_cortex - geo.distance_min_cortex) / (
        1 - ratio_cortex)


def section_matrix(net: Network, K, in_matrix, soil, axial, nodes, edges):
    """
    The legacy's Doussan matrix of one section (its nodes, in the given order): −ΣK on the diagonal, K off it, minus
    the soil and axial terms on the diagonal. Returns (W as CSR, its node order *nodes*).
    """
    from scipy.sparse import coo_matrix
    local = np.full(net.cgroup.size, -1, dtype=np.int64)
    local[nodes] = np.arange(nodes.size)
    e = edges[in_matrix[edges]]
    i, j, k = local[net.a[e]], local[net.b[e]], K[e]
    rows = np.r_[i, i, j, j, np.arange(nodes.size), np.arange(nodes.size)]
    cols = np.r_[i, j, i, j, np.arange(nodes.size), np.arange(nodes.size)]
    data = np.r_[-k, k, k, -k, -soil[nodes], -axial[nodes]]
    return coo_matrix((data, (rows, cols)), shape=(nodes.size, nodes.size)).tocsr()
