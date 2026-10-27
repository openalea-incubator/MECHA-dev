"""
MECHA's conductances (doc/equations.rst): the conductance K of every edge of every section, and the
boundary terms of their nodes, as the legacy's ``HydraulicMatrixBuilder`` computes them (its rules and formulas,
element by element, so that the values are exact), vectorised over all the sections' edges at once.

    network = assemble(n_nodes, n_edges, section_geometries)       the sections' geometry on the whole DataStructure
    K, in_matrix = edge_conductances(network, P, barrier, height)  per edge (P: the parameters, per edge)
    soil, axial, axial_rhs = boundary_terms(network, P, barrier, height)   per node (P per node)

Units are MECHA's: lengths in µm, conductances in cm³ hPa⁻¹ d⁻¹ (cm hPa⁻¹ d⁻¹ per wall surface for kw).
"""
from dataclasses import dataclass

import numpy as np

from openalea.mecha.geometry import (AIR_LINK, MEMBRANE, PLASMODESMATA, STOMATAL, WALL, WALL_AIR, WALL_PATH,
                                     XYLEM_CGROUPS)
from openalea.mecha.postprocess import VISCOSITY_PHLOEM, VISCOSITY_XYLEM, poiseuille

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
    10: ("kw", "kw_casparian", "kw_suberin_out", "kw_suberin_in", "kw", "kw", "kw", "kw"),
    11: ("kw_casparian", "kw_casparian", "kw_suberin_out", "kw_suberin_in", "kw", "kw_suberin_in", "kw_suberin_out",
         "kw"),
}
# the barrier codes lignifying every wall of a sclerenchyma cell (develop's kw_scler), and its source
_KW_SCLER_CONFIGS = {11: "kw_lignin"}
# a wall's interface (its two cells' cgroups, sorted) -> its kw key (HydraulicData.interface_kw_key_map)
_KW_INTERFACES = {(3, 3): "kw_endo_endo", (3, 4): "kw_endo_cortex", (3, 5): "kw_endo_peri", (3, 11): "kw_endo_peri",
                  (3, 12): "kw_endo_peri", (3, 13): "kw_endo_peri", (3, 16): "kw_endo_peri", (3, 22): "kw_endo_peri",
                  (1, 1): "kw_exo_exo",
                  (1, 2): "kw_exo_epi", (1, 4): "kw_exo_cortex", (4, 4): "kw_cortex_cortex",
                  (2, 4): "kw_meso_hypo"}       # the reactions' needle (kw unless kw_meso_hypo is given; legacy r9)
# ── Plasmodesmata ──
# The original MECHA's rules (src_old/6_MECHAv4_SRS2025.py, its radial plasmodesmata; doc/legacy.rst,
# plasmodesmata_rules 1): per interface, a frequency (per unit wall length and height) and an aperture factor, then
# the endodermis side's factor. Its frequencies' parameters and its projects' values (GRANAR):
FPLXHEIGHT_ORIGINAL = {"fplxheight_default": 8.0E5, "fplxheight_epi_exo": 1.08E6, "fplxheight_outer_cortex": 2.28E6,
                       "fplxheight_cortex_cortex": 8.6E5, "fplxheight_cortex_endo": 8.8E5,
                       "fplxheight_endo_endo": 6.4E5, "fplxheight_endo_peri": 9.6E5, "fplxheight_peri_stele": 1.08E6,
                       "fplxheight_stele_stele": 6.4E5, "fplxheight_stele_comp": 9.8E5, "fplxheight_peri_comp": 7.0E5,
                       "fplxheight_comp_comp": 6.8E5, "fplxheight_comp_sieve": 1.76E6, "fplxheight_peri_sieve": 7.2E5,
                       "fplxheight_stele_sieve": 9.0E5}
CGROUP_NAMES = {1: "exo", 2: "epi", 3: "endo", 4: "cortex", 5: "stele", 11: "sieve", 12: "comp", 13: "xylem",
                16: "peri", 17: "tparenchyma", 18: "ttracheid", 22: "cambium", 23: "protophloem"}
# The legacy's per-interface frequencies (HydraulicData.fplxheight_map; others 8e5), plasmodesmata_rules 2
FPLXHEIGHT = {(1, 1): 8.0E5, (2, 2): 8.0E5, (1, 2): 1.08E6, (1, 4): 2.28E6, (4, 4): 8.6E5, (3, 4): 8.8E5,
              (3, 3): 6.4E5, (3, 16): 9.6E5, (3, 5): 9.6E5, (3, 17): 1.08E6, (3, 18): 0.0, (5, 16): 1.08E6,
              (5, 11): 9.0E5, (5, 23): 9.0E5, (5, 12): 9.8E5, (5, 13): 6.4E5, (5, 5): 6.4E5, (5, 17): 6.4E5,
              (5, 18): 0.0, (11, 12): 1.76E6, (11, 16): 7.2E5, (11, 13): 0.0, (11, 11): 0.0, (11, 23): 0.0,
              (11, 17): 6.4E5, (11, 18): 0.0, (12, 13): 9.8E5, (12, 16): 7.0E5, (12, 12): 6.8E5, (12, 23): 6.8E5,
              (12, 17): 1.08E6, (12, 18): 0.0, (13, 16): 1.08E6, (13, 13): 6.4E5, (13, 23): 0.0, (13, 17): 1.08E6,
              (13, 18): 1.76E6, (17, 17): 8.0e5, (17, 18): 0.0, (18, 18): 0.0, (23, 23): 0.0,
              (3, 22): 1.08E6, (5, 22): 1.08E6, (11, 22): 1.2E6, (12, 22): 1.2E6, (13, 22): 6.4E5, (22, 22): 1.0E6}
FPLXHEIGHT_LEGACY_DEFAULT = 8.0E5
# the legacy's aperture factors per interface (develop's interface_kpl_factor_map, its keys fixed: doc/legacy.rst,
# d2); an interface not listed has factor 1
KPL_FACTOR_INTERFACES = {(4, 4): "kpl_cortex_factor", (12, 12): "kpl_PCC_factor", (11, 11): "kpl_PST_factor",
                         (3, 5): "kpl_endo_in_factor", (3, 16): "kpl_endo_in_factor", (3, 4): "kpl_endo_out_factor",
                         (5, 12): "kpl_PPP_factor", (12, 13): "kpl_PPP_factor", (11, 12): "kpl_PCC_factor",
                         (11, 16): "kpl_PST_factor", (11, 13): "kpl_PST_factor", (5, 11): "kpl_PST_factor",
                         (12, 16): "kpl_PPP_factor", (5, 22): "kpl_cambium_stele_factor"}


def fplxheight_name(interface) -> str:
    return "fplxheight_" + "_".join(CGROUP_NAMES[c] for c in interface)


# The interfaces of the tissues the legacy added (transfusion parenchyma 17, tracheids 18, cambium 22, protophloem
# 23), which the original's rules do not know: kept, with the legacy's frequencies, as parameters (in the original
# rules)
ADDED_TISSUES = (17, 18, 22, 23)
FPLXHEIGHT_ADDED = {k: v for k, v in FPLXHEIGHT.items() if set(k) & set(ADDED_TISSUES)}
FPLXHEIGHT_PARAMETERS = {**FPLXHEIGHT_ORIGINAL, **{fplxheight_name(k): v for k, v in FPLXHEIGHT_ADDED.items()}}


GAS_CONSTANT = 8.314                # J mol-1 K-1
WATER_MOLAR_VOLUME = 1.8E-5         # m3 mol-1


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
    sclerenchyma: np.ndarray      # bool per node (by tissue)
    phi_thick: np.ndarray
    wall_cells: np.ndarray        # (n, 2) global nodes, -1
    distance_center_grav: np.ndarray
    x: np.ndarray
    y: np.ndarray
    cell_area: np.ndarray         # µm² per node (0 on the apoplast)
    wall_thickness: np.ndarray    # µm per node: a wall's own, a junction's the mean of its walls' (NaN elsewhere)
    wall_length: np.ndarray
    border_wall: np.ndarray
    border_junction: np.ndarray
    soil_contact_cell: np.ndarray  # bool per node: a protoplast whose border walls touch the soil (MECHA's Contactrange)
    measured_axial_conductivity: np.ndarray  # per node: a conduit's specific axial conductance (cm4 hPa-1 d-1; NaN)
    xylem: np.ndarray             # int per node: occurrences in the network's xylem list (cgroup)
    protosieve: np.ndarray        # int per node: occurrences in the legacy's protosieve list (duplicates, legacy f)
    sieve: np.ndarray             # int per node: occurrences in the sieve tubes' list (by tissue)
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
        sclerenchyma=nodes_full(False, bool),
        phi_thick=nodes_full(False, bool), wall_cells=nodes_full(-1, np.int64, (2,)),
        distance_center_grav=nodes_full(np.nan), x=nodes_full(np.nan), y=nodes_full(np.nan), cell_area=nodes_full(0.), wall_thickness=nodes_full(np.nan), wall_length=nodes_full(0.),
        border_wall=nodes_full(False, bool), border_junction=nodes_full(False, bool),
        soil_contact_cell=nodes_full(False, bool), measured_axial_conductivity=nodes_full(np.nan), xylem=nodes_full(0, np.int64),
        protosieve=nodes_full(0, np.int64), sieve=nodes_full(0, np.int64), outercortex_connec_rank=np.zeros(len(geometries), dtype=np.int64),
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
        net.sclerenchyma[nodes[geo.typed["sclerenchyma"]]] = True
        if phi_thick is not None:
            net.phi_thick[nodes] = phi_thick[i]
        cells = geo.wall_cells
        net.wall_cells[nodes] = np.where(cells >= 0, nodes[np.maximum(cells, 0)], -1)
        net.distance_center_grav[nodes] = geo.distance_center_grav
        net.x[nodes] = geo.x
        net.y[nodes] = geo.y
        net.cell_area[nodes] = np.nan_to_num(geo.cell_area)
        net.wall_length[nodes] = geo.wall_length
        net.border_wall[nodes] = geo.border_wall
        net.border_junction[nodes] = geo.border_junction
        np.add.at(net.xylem, nodes[geo.xylem_cells], 1)
        np.add.at(net.protosieve, nodes[geo.protosieve_cells], 1)
        np.add.at(net.sieve, nodes[geo.sieve_cells], 1)
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


def set_wall_thickness(net: Network, wall_thickness) -> None:
    """Each wall's thickness (µm, GRANAP's), and each junction's: the mean of the walls it joins in its section."""
    out = np.where(net.kind == WALL, wall_thickness, np.nan)
    e = np.flatnonzero(net.junction_edge)
    wall = np.where(net.kind[net.a[e]] == WALL, net.a[e], net.b[e])
    junction = np.where(net.kind[net.a[e]] == WALL, net.b[e], net.a[e])
    total, count = np.zeros(out.size), np.zeros(out.size)
    np.add.at(total, junction, wall_thickness[wall])
    np.add.at(count, junction, 1.)
    joined = count > 0
    out[joined] = total[joined] / count[joined]
    net.wall_thickness = out


def _thickness(net: Network, P: dict, nodes) -> np.ndarray:
    """The thickness (µm) at the wall (or junction) *nodes*, per entity of P: thickness_source 1 MECHA's one value,
    2 the node's own."""
    own = np.asarray(P["thickness_source"]) == 2
    return np.where(own, net.wall_thickness[nodes], np.asarray(P["thickness"], dtype=float))


def _shared_wall_thickness(net: Network, cell_a, cell_b) -> np.ndarray:
    """The thickness (µm) of the walls cells *cell_a* and *cell_b* share (their lengths' weighted mean; NaN where
    they share none, or where their thickness is unknown): the wall a plasmodesmata edge crosses."""
    walls = np.flatnonzero((net.kind == WALL) & (net.wall_cells >= 0).all(axis=1) & np.isfinite(net.wall_thickness))
    n = net.kind.size
    low, high = np.sort(net.wall_cells[walls], axis=1).T
    keys, inverse = np.unique(low * n + high, return_inverse=True)
    if keys.size == 0:
        return np.full(np.size(cell_a), np.nan)
    length = np.bincount(inverse, weights=net.wall_length[walls], minlength=keys.size)
    weighted = np.bincount(inverse, weights=net.wall_length[walls] * net.wall_thickness[walls], minlength=keys.size)
    mean = np.divide(weighted, length, out=np.full(keys.size, np.nan), where=length > 0)
    wanted = np.minimum(cell_a, cell_b) * n + np.maximum(cell_a, cell_b)
    index = np.minimum(np.searchsorted(keys, wanted), keys.size - 1)
    return np.where(keys[index] == wanted, mean[index], np.nan)


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
    config["kw_scler"] = np.full(codes.size, np.nan)
    for code, source in _KW_SCLER_CONFIGS.items():
        config["kw_scler"] = np.where(codes == code, np.asarray(P[source], dtype=float), config["kw_scler"])
    meso_hypo = np.asarray(P["kw_meso_hypo"], dtype=float)
    config["kw_meso_hypo"] = np.where(np.isfinite(meso_hypo), meso_hypo, np.asarray(P["kw"], dtype=float))
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


def edge_conductances(net: Network, P: dict, barrier, height, factors: dict = None):
    """
    (K, in_matrix): the conductance of every edge (cm³ hPa⁻¹ d⁻¹), and whether the legacy's matrix has it (its wall,
    membrane and plasmodesmata edges; not the wall_air and air_link edges, legacy i, nor the wall edges its junction
    rebuild drops). P: {parameter: value per edge}; barrier: code per edge; height: µm per edge.

    With *factors* (a dict), the solutes' diffusion factors per edge are written in it (cm, the reactions' c_flag
    terms of HydraulicMatrixBuilder and SoluteTransport): "apoplastic" (walls: temp × their kw ratio, the
    xylem-pieces septa 1e-16; the membranes of mature vessels: 1e-4 × wall length × height / thickness), "symplastic"
    (plasmodesmata: pd_section × their quantity / the crossed wall's thickness × 1e-4), "membrane" (every membrane: length × height
    × 1e-4 / dist), each to be multiplied by its diffusivity (cm² d⁻¹).
    """
    n = net.path.size
    K = np.zeros(n)
    barrier = np.asarray(barrier).astype(np.int64)
    height = np.asarray(height, dtype=float)
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

    thickness = _thickness(net, P, wall_end)          # per edge: its wall's (walls and membranes)

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
    # a wall of a sclerenchyma cell, lignified at the barrier codes defining kw_scler (develop's barrier 11)
    sclerenchyma = (count(net.sclerenchyma)[e] > 0) & np.isfinite(sub["kw_scler"])
    kw_val = np.where(sclerenchyma, sub["kw_scler"], kw_val)
    kw_val = np.where(has_passage, sub["kw_passage"], kw_val)
    kw_val = np.where(ghost, sub["kw_septa"], kw_val)
    K_wall = kw_val * temp
    K_wall = np.where(ghost & (n_interc < 2), 1.0E-16, K_wall)        # xylem pieces' septa
    K[e] = K_wall
    if factors is not None:
        for name in ("apoplastic", "symplastic", "membrane"):
            factors[name] = np.zeros(n)
        temp_factor = np.where(ghost & (n_interc < 2), 1.0E-16, kw_val / kw[e])
        factors["apoplastic"][e] = temp * temp_factor

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
    # the transfusion tracheids as xylem membranes on request (develop's tracheid_as_xylem), never its boundary
    xylem = np.isin(cg, XYLEM_CGROUPS) | ((cg == 18) & (np.asarray(P["tracheid_as_xylem"], dtype=float)[e] > 0))
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
    # no aquaporin activity outside the barriers: 1e-16 (an air-filled intercellular space, the original's);
    # membrane_floor 2: only where kmb is 0 too (the reactions branch's and develop's)
    floor = ~barrier_interface & (kaqp_curr == 0.0)
    floor &= (np.asarray(P["membrane_floor"], dtype=float)[e] != 2) | (kmb == 0.0)
    K_mb = np.where(floor, 1.00E-16, K_mb)
    K[e] = K_mb
    if factors is not None:
        vessel = xylem & (b > 0)
        factors["apoplastic"][e] = np.where(vessel, 1.0E-04 * (net.wall_length[wall] * height[e]) / t, 0.)
        factors["membrane"][e] = net.edge_length[e] * height[e] * 1e-4 / net.edge_dist[e]

    # ── plasmodesmata ──
    e = np.flatnonzero(plasmodesmata)
    b = barrier[e]
    kpl = np.asarray(P["kpl"], dtype=float)[e]
    original = np.asarray(P["plasmodesmata_rules"], dtype=float)[e] == 1
    temp_factor = np.where(original, _plasmodesmata_original(net, e, P, b), _plasmodesmata_legacy(net, e, P, b))
    # the aperture by the distance to the soil contact (develop's pd_spatial_config; not between two xylem cells)
    coefficient, distance = plasmodesmata_spatial_coefficients(net, P, e)
    both_xylem = np.isin(net.cgroup[net.a[e]], XYLEM_CGROUPS) & np.isin(net.cgroup[net.b[e]], XYLEM_CGROUPS)
    temp_factor = np.where(both_xylem, temp_factor, temp_factor * coefficient)
    K[e] = kpl * temp_factor
    if factors is not None:
        # the crossed wall's thickness with thickness_source 2 (its own), MECHA's one value with 1 or without it
        thickness = np.asarray(P["thickness"], dtype=float)[e]
        own = np.asarray(P["thickness_source"])[e] == 2
        if own.any():
            shared = _shared_wall_thickness(net, net.a[e], net.b[e])
            thickness = np.where(own & np.isfinite(shared), shared, thickness)
        factors["symplastic"][e] = np.asarray(P["pd_section"], dtype=float)[e] * temp_factor / thickness * 1.0E-04
        factors["contact_distance"] = distance
    return K, in_matrix


def soil_contact(net: Network, x_contact, junction_rule) -> np.ndarray:
    """
    Per node, whether it is a border wall or junction in contact with the soil (``HydraulicMatrixBuilder._contact_nodes``):
    at x >= *x_contact* (per node), or along a protoplast in contact (``soil_contact_cell``, MECHA's Contactrange): a
    border wall of such a cell, a border junction whose cells all are (*junction_rule* 1, develop's) or any is (2, the
    reactions branch's). A junction's cells are those of the walls it joins (the legacy's ``junction_wall_cell``).
    """
    contact = (net.border_wall | net.border_junction) & (net.x >= np.asarray(x_contact, dtype=float))
    flags = getattr(net, "soil_contact_cell", None)                 # a stand-in network may have none
    if flags is None or not flags.any():
        return contact
    cells = net.wall_cells
    flagged = np.where(cells >= 0, net.soil_contact_cell[np.maximum(cells, 0)], False)
    contact |= net.border_wall & flagged.any(axis=1)
    e = np.flatnonzero(net.junction_edge)
    wall_is_a = net.kind[net.a[e]] == WALL
    wall, junction = np.where(wall_is_a, net.a[e], net.b[e]), np.where(wall_is_a, net.b[e], net.a[e])
    pairs = np.column_stack((np.repeat(junction, 2), cells[wall].reshape(-1)))
    pairs = np.unique(pairs[pairs[:, 1] >= 0], axis=0)                       # each (junction, cell) once
    n = net.kind.size
    n_cells = np.bincount(pairs[:, 0], minlength=n)
    n_flagged = np.bincount(pairs[:, 0], weights=net.soil_contact_cell[pairs[:, 1]], minlength=n)
    rule = np.asarray(junction_rule, dtype=float) * np.ones(n)
    by_cells = np.where(rule == 2, n_flagged > 0, (n_cells > 0) & (n_flagged == n_cells))
    return contact | (net.border_junction & by_cells)


def contact_distances(net: Network, contact, sections) -> np.ndarray:
    """
    Per node (µm; NaN elsewhere and where unreachable), the shortest distance along each of *sections*' graphs (its
    walls, junctions and cells, joined by their wall, membrane, plasmodesmata and vapour edges, each its Euclidean
    length) to its soil-contact nodes (*contact*, a bool per node: soil_contact) (HydraulicMatrixBuilder._contact_nodes,
    _pd_spatial_coefficients).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    distance = np.full(net.cgroup.size, np.nan)
    graph_edges = net.in_section & (net.junction_edge | np.isin(net.path, (MEMBRANE, PLASMODESMATA, WALL_AIR,
                                                                              AIR_LINK)))
    for section in sections:
        nodes = np.flatnonzero(net.section_of_node == section)
        local = np.full(net.cgroup.size, -1, dtype=np.int64)
        local[nodes] = np.arange(nodes.size)
        e = np.flatnonzero(graph_edges & (net.section_of_edge == section))
        weight = np.hypot(net.x[net.a[e]] - net.x[net.b[e]], net.y[net.a[e]] - net.y[net.b[e]])
        weight = np.where(weight > 0., weight, 1e-300)                   # a zero-length edge is still an edge
        graph = coo_matrix((weight, (local[net.a[e]], local[net.b[e]])), shape=(nodes.size, nodes.size)).tocsr()
        sources = np.flatnonzero(contact[nodes])
        if not sources.size:
            continue
        distance[nodes] = dijkstra(graph, directed=False, indices=sources, min_only=True)
    return np.where(np.isfinite(distance), distance, np.nan)


def plasmodesmata_spatial_coefficients(net: Network, P: dict, e):
    """
    (coefficient per plasmodesmata edge *e*, contact distance per node): with pd_spatial_profile 1 (linear) or 2
    (exponential), each plasmodesma's mean end distance to the soil contact d (contact_distances), relative to the
    section's plasmodesmata's range [d_min, d_max], gives c_wet − (c_wet − c_dry) u with u = (d − d_min) / (d_max −
    d_min), or c_dry + (c_wet − c_dry) exp(−(d − d_min) / L) (L: pd_spatial_length_scale, NaN: a third of the range);
    c_dry where unreachable, c_wet where the range is 0; 1 with profile 0 (develop's _pd_spatial_coefficients).
    """
    coefficient = np.ones(e.size)
    profile = np.asarray(P["pd_spatial_profile"], dtype=float)[e]
    sections = np.unique(net.section_of_edge[e[profile > 0]])
    if sections.size == 0:
        return coefficient, np.full(net.cgroup.size, np.nan)
    def per_node(name):                                                       # a plant's value, per section
        of_section = np.full(int(net.section_of_node.max()) + 1, np.inf)
        of_section[net.section_of_edge[e]] = np.asarray(P[name], dtype=float)[e]
        return np.where(net.section_of_node >= 0, of_section[np.maximum(net.section_of_node, 0)], np.inf)
    distance = contact_distances(net, soil_contact(net, per_node("x_contact"), per_node("contact_junction_rule") if "contact_junction_rule" in P else 1.),
                                 sections)
    d = 0.5 * (distance[net.a[e]] + distance[net.b[e]])
    c_wet = np.asarray(P["pd_spatial_c_wet"], dtype=float)[e]
    c_dry = np.asarray(P["pd_spatial_c_dry"], dtype=float)[e]
    scale = np.asarray(P["pd_spatial_length_scale"], dtype=float)[e]
    for section in sections:
        mine = (net.section_of_edge[e] == section) & (profile > 0)
        reachable = mine & np.isfinite(d)
        d_min = d[reachable].min() if reachable.any() else 0.
        d_max = d[reachable].max() if reachable.any() else 0.
        span = d_max - d_min
        length_scale = np.where(np.isfinite(scale), scale, span / 3.)
        u = np.clip((d - d_min) / span, 0., 1.) if span > 0 else np.zeros(e.size)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            exponential = np.where(length_scale > 0, c_dry + (c_wet - c_dry) * np.exp(-(d - d_min) / length_scale),
                                   c_dry)
        value = np.where(profile == 2, exponential, c_wet - (c_wet - c_dry) * u)
        value = np.where(span > 0, value, c_wet)
        value = np.where(np.isfinite(d), value, c_dry)
        coefficient = np.where(mine, value, coefficient)
    return coefficient, distance


def boundary_terms(net: Network, P: dict, barrier, height):
    """
    Per node (cm³ hPa⁻¹ d⁻¹): the soil term of the border walls and junctions in contact (``_apply_soil_boundary``),
    and the axial term of the xylem cells (barrier > 0) or protosieve cells (barrier 0): (soil, axial on the matrix
    diagonal, axial in the right-hand side). P, barrier, height per node. The axial term (``standard_axial_source``):
    1 each conduit's Poiseuille conductance over the height (water in the xylem, phloem sap in the protosieve: the
    original MECHA's ``Kax_source`` 1), 2 the prescribed ``k_xyl`` and ``k_sieve`` (the legacy's, whatever its source).
    """
    n = net.cgroup.size
    kw = np.asarray(P["kw"], dtype=float)
    thickness = _thickness(net, P, np.arange(n))
    height = np.asarray(height, dtype=float)
    contact = soil_contact(net, P["x_contact"], P.get("contact_junction_rule", 1.))
    soil = np.zeros(n)
    w = np.flatnonzero(net.border_wall & contact)
    soil[w] = kw[w] * (1.0E-04 * (net.wall_length[w] / 2 * height[w]) / (thickness[w] / 2))
    j = np.flatnonzero(net.border_junction & contact)
    soil[j] = kw[j] * (1.0E-04 * (net.wall_length[j] * height[j]) / (thickness[j] / 2))

    barrier = np.asarray(barrier).astype(np.int64)
    k_xyl, k_sieve = np.asarray(P["k_xyl"], dtype=float), np.asarray(P["k_sieve"], dtype=float)
    xylem = (barrier > 0) & (net.xylem > 0)
    protosieve = (barrier == 0) & (net.protosieve > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        measured = getattr(net, "measured_axial_conductivity", np.full(n, np.nan)) / (height / 1.0E04)      # a conduit's own over the height
        k_xyl = np.where(np.asarray(P["standard_axial_source"]) == 1,
                         np.where(np.isfinite(measured), measured, poiseuille(net.cell_area, height, VISCOSITY_XYLEM)),
                         k_xyl)
        k_sieve = np.where(np.asarray(P["standard_axial_source"]) == 1,
                           np.where(np.isfinite(measured), measured,
                                    poiseuille(net.cell_area, height, VISCOSITY_PHLOEM)), k_sieve)
    axial = np.where(xylem, k_xyl, np.where(protosieve, k_sieve, 0.))
    return soil, axial, axial.copy()          # each protosieve cell once (the legacy adds k_sieve per duplicate, legacy f)


def sieve_tube_terms(net: Network, P: dict, barrier, height) -> np.ndarray:
    """
    Per node (cm³ hPa⁻¹ d⁻¹): the conductance between a sieve tube of a section with barriers (barrier > 0) and the
    phloem boundary (the legacy's scenarios ≥ 1 with a phloem pressure, ``Mecha.water_flux``): as the axial terms
    (``standard_axial_source``), 1 its Poiseuille conductance over the height (phloem sap), 2 ``k_sieve``.
    """
    sieve = (np.asarray(barrier) > 0) & (net.sieve > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        k_sieve = np.where(np.asarray(P["standard_axial_source"]) == 1,
                           poiseuille(net.cell_area, np.asarray(height, dtype=float), VISCOSITY_PHLOEM),
                           np.asarray(P["k_sieve"], dtype=float))
    return np.where(sieve, k_sieve, 0.)


def elongation_sink(net: Network, P: dict, barrier, relative_x) -> np.ndarray:
    """
    Per node (cm³ d⁻¹): the water the elongating walls and protoplasts of the sections without barriers (barrier 0,
    the elongation zone) withdraw (``Mecha.elongation_BC``): a wall its length × half its thickness × the elongation
    rate × the apoplast's water fraction, a protoplast its area less its walls' half thickness along its perimeter ×
    the rate × the symplast's water fraction (none where the walls take its whole area). The rate is
    elongation_midpoint_rate + (x_rel − 0.5) × elongation_side_rate_difference at the node's relative x. A cell's
    thickness is its walls' mean, its perimeter their lengths' sum. P per node.
    """
    rate = np.asarray(P["elongation_midpoint_rate"], dtype=float) + \
        (np.asarray(relative_x, dtype=float) - 0.5) * np.asarray(P["elongation_side_rate_difference"], dtype=float)
    thickness = _thickness(net, P, np.arange(net.kind.size))
    walls = np.flatnonzero(net.kind == WALL)
    ends = net.wall_cells[walls]
    has = ends >= 0
    cell_of, wall_of = ends[has], np.repeat(walls, has.sum(axis=1))
    n = net.kind.size
    perimeter = np.bincount(cell_of, weights=net.wall_length[wall_of], minlength=n)
    count = np.bincount(cell_of, minlength=n)
    mean_thickness = np.divide(np.bincount(cell_of, weights=thickness[wall_of], minlength=n), count,
                               out=np.zeros(n), where=count > 0)
    free_area = net.cell_area - perimeter * mean_thickness / 2
    sink = np.where(net.kind == WALL,
                    net.wall_length * thickness / 2 * 1.0E-08 * rate * np.asarray(P["water_fraction_apoplast"]),
                    np.where((net.kind == 0) & (free_area > 0),
                             free_area * 1.0E-08 * rate * np.asarray(P["water_fraction_symplast"]), 0.))
    return np.where((np.asarray(barrier) == 0) & (net.section_of_node >= 0), sink, 0.)


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


def _harmonic(t1, t2):
    """2 t1 t2 / (t1 + t2), 0 where both are 0 (the original's aperture of an interface between two cell types)."""
    t1, t2 = np.broadcast_arrays(np.asarray(t1, dtype=float), np.asarray(t2, dtype=float))
    total = t1 + t2
    return np.divide(2 * t1 * t2, total, out=np.zeros(total.shape), where=total != 0)


def _plasmodesmata_legacy(net: Network, e, P, b):
    """The legacy's plasmodesmata (HydraulicMatrixBuilder._fill_plasmodesmata): quantity × aperture per edge *e*."""
    gi, gj = net.cgroup[net.a[e]], net.cgroup[net.b[e]]
    code = _interface_codes(gi, gj)
    fplx = np.full(e.size, FPLXHEIGHT_LEGACY_DEFAULT)
    for (i, j), value in FPLXHEIGHT.items():
        fplx[code == i * 1000 + j] = value
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
    meso_hypo = np.isfinite(np.asarray(P["kw_meso_hypo"], dtype=float)[e]) & (code == 2004)
    return np.select([interc & (b > 0), meso_hypo, code == 13013, (b > 0) & ((gi == 13) | (gj == 13))],
                     [0.0, 0.0, 10000.0 * base_area, 0.0], aperture * base_area * modifier)


def _phloem_pole_pericycle(net: Network, e):
    """Per node: the phloem-pole pericycle cells, pericycle cells (cgroup 16, 21) joined to a sieve tube (11, 23) by
    plasmodesmata (the original's ``PPP``)."""
    out = np.zeros(net.cgroup.size, dtype=bool)
    a, b = net.a[e], net.b[e]
    ga, gb = net.cgroup[a], net.cgroup[b]
    peri, sieve = (16, 21), (11, 23)
    out[a[np.isin(ga, peri) & np.isin(gb, sieve)]] = True
    out[b[np.isin(gb, peri) & np.isin(ga, sieve)]] = True
    return out


def _plasmodesmata_original(net: Network, e, P, b):
    """
    The original MECHA's plasmodesmata (its radial rules, first match wins): quantity × aperture per edge *e*, then
    × endo_in_factor on the stelar side of the endodermis (both cells endodermis or stele: the wall they share is in
    the original's rows up to the endodermis'), × endo_out_factor on the cortical side. The cgroups of its newer
    CellSet versions are read as it does (19, 20: xylem; 21: pericycle), and develop's companion cells 24 and 26 as
    companion cells (12).
    """
    def param(name):
        return np.asarray(P[name], dtype=float)[e]

    gi, gj = net.cgroup[net.a[e]].copy(), net.cgroup[net.b[e]].copy()
    for g in (gi, gj):
        g[np.isin(g, (19, 20))] = 13
        g[g == 21] = 16
        g[np.isin(g, (24, 26))] = 12

    def pair(x, y):
        x, y = np.atleast_1d(x), np.atleast_1d(y)
        return (np.isin(gi, x) & np.isin(gj, y)) | (np.isin(gj, x) & np.isin(gi, y))

    length = 1.0E-04 * net.edge_length[e]
    section = net.section_of_edge[e]
    outer = net.outercortex_connec_rank[section]
    concentrated = np.where(b[:, None] > 0, net.concentrated[section], 1.0)
    ppp = _phloem_pole_pericycle(net, np.flatnonzero(net.in_section & (net.path == PLASMODESMATA)))
    t_ppp = np.where(ppp[net.a[e]] | ppp[net.b[e]], param("kpl_PPP_factor"), 1.0)
    t_cortex, t_pcc = param("kpl_cortex_factor"), param("kpl_PCC_factor")
    interc = net.intercellular[net.a[e]] | net.intercellular[net.b[e]]
    stele = (5, 13)
    added = np.zeros(e.size, dtype=bool)
    added_value = np.zeros(e.size)
    code = _interface_codes(gi, gj)
    for interface in FPLXHEIGHT_ADDED:
        match = code == interface[0] * 1000 + interface[1]
        added |= match
        added_value = np.where(match, param(fplxheight_name(interface)), added_value)
    rules = [
        (interc & (b > 0), 0.0),
        (np.isfinite(param("kw_meso_hypo")) & pair(2, 4), 0.0),      # the needle's lignified (2, 4) walls
        ((param("xylem_pieces") > 0) & (gi == 13) & (gj == 13), 10000 * param("fplxheight_default") * length),
        ((b > 0) & ((gi == 13) | (gj == 13)), 0.0),
        (pair(2, 1), param("fplxheight_epi_exo") * length),
        ((np.minimum(gi, gj) == np.minimum(outer, 4)) & (np.maximum(gi, gj) == np.maximum(outer, 4)) & (outer != 4),
         _harmonic(t_cortex, 1.0) * param("fplxheight_outer_cortex") * length * concentrated[:, 0]),
        (pair(4, 4), t_cortex * param("fplxheight_cortex_cortex") * length * concentrated[:, 1]),
        (pair(3, 4), _harmonic(t_cortex, 1.0) * param("fplxheight_cortex_endo") * length * concentrated[:, 2]),
        (pair(3, 3), param("fplxheight_endo_endo") * length),
        (pair(3, 16), _harmonic(t_ppp, 1.0) * param("fplxheight_endo_peri") * length),
        (pair(16, stele), _harmonic(t_ppp, 1.0) * param("fplxheight_peri_stele") * length),
        (pair(stele, 12), _harmonic(t_pcc, 1.0) * param("fplxheight_stele_comp") * length),
        (pair(16, 12), _harmonic(t_pcc, t_ppp) * param("fplxheight_peri_comp") * length),
        (pair(12, 12), t_pcc * param("fplxheight_comp_comp") * length),
        (pair(12, 11), _harmonic(t_pcc, 1.0) * param("fplxheight_comp_sieve") * length),
        (pair(16, 11), _harmonic(t_ppp, 1.0) * param("fplxheight_peri_sieve") * length),
        (pair(stele, 11), param("fplxheight_stele_sieve") * length),
        (pair(stele, stele), param("fplxheight_stele_stele") * length),
        (added, added_value * length),
    ]
    quantity = np.select([c for c, _ in rules], [np.broadcast_to(v, e.size) for _, v in rules],
                         param("fplxheight_default") * length)
    inner = lambda g: (g == 3) | (g > 4)                                   # noqa: E731
    has_xylem = np.zeros(int(section.max(initial=-1)) + 1, dtype=bool)
    xylem_sections = net.section_of_node[np.isin(net.cgroup, XYLEM_CGROUPS) & (net.section_of_node >= 0)]
    has_xylem[xylem_sections] = True
    stelar_side = (inner(gi) & inner(gj)) | ~has_xylem[section]       # no xylem: the original's r_rel is NaN
    return quantity * np.where(stelar_side, param("kpl_endo_in_factor"), param("kpl_endo_out_factor"))


def vapour_factor(temperature):
    """e_sat V_w / (R T) at *temperature* (°C): from a vapour conductance (m³ s⁻¹, on a relative vapour pressure) to a
    liquid water flow (m³ s⁻¹), e_sat by Tetens (Pa)."""
    T = np.asarray(temperature, dtype=float)
    e_sat = 610.78 * np.exp(17.27 * T / (T + 237.3))
    return e_sat * WATER_MOLAR_VOLUME / (GAS_CONSTANT * (T + 273.15))


def kelvin_alpha(temperature):
    """α = V_w / (R T) (hPa⁻¹): a liquid at ψ is in equilibrium with a vapour at x = exp(α ψ) (Kelvin)."""
    return WATER_MOLAR_VOLUME / (GAS_CONSTANT * (np.asarray(temperature, dtype=float) + 273.15)) * 100.


def vapour_conductances(net: Network, P: dict, height, pore_width, pore_depth, cell_area, aperture, length):
    """
    The leaves' vapour path, in cm³ d⁻¹ of liquid water per unit of relative vapour pressure x, from
    vapour diffusion, G = D_v A / d × e_sat V_w / (R T):

      wall → air space (wall_air): A the wall's face (its length × the height), d its distance to the air space's
                     centre; the wall's side by Kelvin, x = exp(α ψ), linearised at psi_ref_wall_air: the flow from the
                     wall is K_wall ψ_wall − K_air x_air + offset, K_air = G, K_wall = G e_ref α, offset = G e_ref
                     (1 − α ψ_ref)
      air ↔ air (air_link): A the height × the narrower one (a pore's aperture width; an air space's equivalent
                     diameter √area), d the distance between their centres
      pore → leaf surface (path 6): the pore's file of stomata over the segment, length / stomatal_spacing of them,
                     each a diffusion tube with its end correction (Brown & Escombe, the anatomical g_max),
                     D_v a / (l + (π/2) √(a/π)), a = pore_width × pore_length, l = pore_depth; × its aperture fraction

    *P* per edge; *height*, *length* (µm) per edge; *pore_width*, *pore_depth* (mm), *cell_area* (mm²), *aperture*
    per node. Returns (K per edge for the air links and the stomata, K_wall, K_air, offset, wall end, air end).
    """
    n = net.path.size
    K = np.zeros(n)
    K_wall, K_air, offset = np.zeros(n), np.zeros(n), np.zeros(n)
    wall_end, air_end = np.full(n, -1, dtype=np.int64), np.full(n, -1, dtype=np.int64)
    D = np.asarray(P["vapour_diffusivity"], dtype=float)                       # m² s⁻¹
    F = vapour_factor(P["leaf_temperature"]) * 1.0E6 * 86400.                   # m³ s⁻¹ -> cm³ d⁻¹
    alpha = kelvin_alpha(P["leaf_temperature"])
    psi_ref = np.asarray(P["psi_ref_wall_air"], dtype=float)
    height = np.asarray(height, dtype=float) * 1.0E-6                          # µm -> m

    e = np.flatnonzero(net.in_section & (net.path == WALL_AIR))
    if e.size:
        a_is_wall = net.kind[net.a[e]] == WALL
        wall_end[e] = np.where(a_is_wall, net.a[e], net.b[e])
        air_end[e] = np.where(a_is_wall, net.b[e], net.a[e])
        G = D[e] * (net.edge_length[e] * 1.0E-6 * height[e]) / (net.edge_dist[e] * 1.0E-6) * F[e]
        e_ref = np.exp(alpha[e] * psi_ref[e])
        K_air[e] = G
        K_wall[e] = G * e_ref * alpha[e]
        offset[e] = G * e_ref * (1.0 - alpha[e] * psi_ref[e])

    e = np.flatnonzero(net.in_section & (net.path == AIR_LINK))
    if e.size:
        width = np.fmin(_air_width(net.a[e], pore_width, cell_area), _air_width(net.b[e], pore_width, cell_area))
        K[e] = D[e] * (height[e] * width) / (net.edge_dist[e] * 1.0E-6) * F[e]

    e = np.flatnonzero(net.path == STOMATAL)
    if e.size:
        pore = np.where(np.isfinite(np.asarray(pore_width)[net.a[e]]), net.a[e], net.b[e])
        area = np.asarray(pore_width)[pore] * 1.0E-3 * np.asarray(P["pore_length"], dtype=float)[e] * 1.0E-6
        depth = np.asarray(pore_depth)[pore] * 1.0E-3
        tube = D[e] * area / (depth + np.pi / 2 * np.sqrt(area / np.pi))
        count = np.asarray(length, dtype=float)[e] / np.asarray(P["stomatal_spacing"], dtype=float)[e]
        K[e] = count * np.asarray(aperture, dtype=float)[pore] * tube * F[e]
    return K, K_wall, K_air, offset, wall_end, air_end


def _air_width(nodes, pore_width, cell_area):
    """An air node's width (m) across a link: a pore's aperture, an air space's equivalent diameter √area."""
    width = np.asarray(pore_width, dtype=float)[nodes]
    return np.where(np.isfinite(width), width, np.sqrt(np.asarray(cell_area, dtype=float)[nodes])) * 1.0E-3
