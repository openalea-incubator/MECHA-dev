"""
MECHA's osmotic presets (devplan_metafspm.md, §20): the osmotic potentials of the cells and walls, and the reflection
coefficients of the membranes, of the legacy's scenarios (``Mecha.initialize_scenarios``), on every root section.

MechaHydraulics reads them as inputs (role "Solutes"): this component stands for a metabolism or solute model until
one is coupled (the ``reaction`` branch), and gives the comparison with the legacy's osmotic scenarios.

    presets = MechaOsmoticPresets(data_structure=ds)       # osmotic_potential (nodes), reflection_coefficient (edges)
    mecha = MechaHydraulics(data_structure=ds)

The legacy's rules, per membrane (wall i, cell j), from the cell's rank (and cgroup), the wall's neighbours, the
section's barrier and the wall's position: its radial position relative to the endodermis (r_rel, from the legacy's
layer rows) and its relative x (x_rel). Shoot sections get none (π = 0, σ = 0): the presets are root rules.
"""
from dataclasses import dataclass

import numpy as np

from openalea.metafspm.coupling.component import state_variable
from openalea.metafspm.solve.decorator import rate

from openalea.mecha.component import SCALES, MechaSections, _doc, _p
from openalea.mecha.geometry import MEMBRANE

N_RANKS = 62


def legacy_rows(n_layer, layer_dist):
    """
    The legacy's layer rows (``NetworkBuilder``, its rank-to-row discretisation, transcribed): (rank_to_row (62,),
    the rows' distances to the centre, the outer cortex's row).
    """
    def weighted_avg(indices):
        total_n = sum(n_layer[i] for i in indices)
        return sum(n_layer[i] * layer_dist[i] for i in indices) / total_n

    j = 0
    rank_to_row = np.full(N_RANKS, np.nan, dtype=float)
    distance = []
    for i in range(61, 49, -1):                                          # stele
        if n_layer[i] > 0:
            rank_to_row[i] = j
            distance.append(layer_dist[i])
            j += 1
    if n_layer[16] > 0:                                                  # pericycle
        rank_to_row[16] = j
        distance.append(layer_dist[16])
        j += 1
    rank_to_row[3] = j                                                   # endodermis: 4 rows
    distance.extend([layer_dist[3]] * 4)
    j += 4
    i1 = 40                                                              # cortex
    n_ref, dist_ref = n_layer[i1], layer_dist[i1]
    rank_to_row[i1] = j
    distance.append(layer_dist[i1])
    j += 1
    i1 = 41
    ratio_complete = 0.75
    while i1 < 50:
        if n_layer[i1] > ratio_complete * n_ref:
            rank_to_row[i1] = j
            distance.append(layer_dist[i1])
            j += 1
            n_ref, dist_ref = n_layer[i1], layer_dist[i1]
            i1 += 1
        elif n_layer[i1] > 0:
            i2 = i1 + 1
            while i2 < 50 and n_layer[i2] == 0:
                i2 += 1
            if i2 >= 50:
                i1 += 1
                continue
            if n_layer[i1] + n_layer[i2] > ratio_complete * n_ref:
                if n_layer[i2] > ratio_complete * n_ref:
                    if abs(layer_dist[i1] - layer_dist[i2]) < abs(layer_dist[i1] - dist_ref):
                        rank_to_row[i1] = rank_to_row[i2] = j
                        distance.append(weighted_avg([i1, i2]))
                        j += 1
                    else:
                        rank_to_row[i1] = j - 1
                        distance[j - 1] = (n_layer[i1] * layer_dist[i1] + n_ref * dist_ref) / (n_layer[i1] + n_ref)
                        rank_to_row[i2] = j
                        distance.append(layer_dist[i2])
                        j += 1
                else:
                    rank_to_row[i1] = rank_to_row[i2] = j
                    distance.append(weighted_avg([i1, i2]))
                    j += 1
                i1 = i2 + 1
            else:
                if i2 + 1 < 50 and n_layer[i2 + 1] > 0:
                    if abs(layer_dist[i1] - layer_dist[i2 + 1]) < abs(layer_dist[i1] - dist_ref):
                        rank_to_row[i1] = rank_to_row[i2] = rank_to_row[i2 + 1] = j
                        distance.append(weighted_avg([i1, i2, i2 + 1]))
                        j += 1
                    else:
                        rank_to_row[i1] = j
                        if abs(layer_dist[i2] - layer_dist[i2 + 1]) < abs(layer_dist[i2] - dist_ref):
                            distance[j - 1] = (n_layer[i1] * layer_dist[i1] + n_ref * dist_ref) / (n_layer[i1] + n_ref)
                            rank_to_row[i2] = rank_to_row[i2 + 1] = j
                            distance.append(weighted_avg([i2, i2 + 1]))
                            j += 1
                        else:
                            rank_to_row[i2] = j - 1
                            distance[j] = ((n_layer[i1] * layer_dist[i1] + n_layer[i2] * layer_dist[i2]
                                            + n_ref * dist_ref) / (n_layer[i1] + n_layer[i2] + n_ref))
                            rank_to_row[i2 + 1] = j
                            distance.append(layer_dist[i2 + 1])
                            j += 1
                else:
                    rank_to_row[i1] = rank_to_row[i2] = j
                    distance.append(weighted_avg([i1, i2]))
                    j += 1
                i1 = i2 + 2
        else:
            i1 += 1
    if n_layer[1] > 0:                                                   # exodermis: 2 rows
        rank_to_row[1] = j
        distance.extend([layer_dist[1]] * 2)
        j += 2
    rank_to_row[2] = j                                                   # epidermis
    distance.append(layer_dist[2])
    row_outer_cortex = int(rank_to_row[1] - 1 if not np.isnan(rank_to_row[1]) else rank_to_row[2] - 1)
    return rank_to_row, np.asarray(distance, dtype=float), row_outer_cortex


def relative_positions(geo, rank_to_row, distance):
    """
    (r_rel, x_rel) per local node (``NetworkBuilder.compute_relative_positions``): a wall's radial position, negative
    on the stelar side of the endodermis (from the xylem's 80th-percentile distance, −1, to the endodermis, 0) and
    positive on the cortical side (from the endodermis, 1, to the epidermis, 0), NaN for a wall without a membrane;
    every node's x relative to the cells' x range. The legacy takes that range from GRANAP's cell seeds
    (``_cells_gdf['x']``), which the DataStructure does not carry; here, from the cells' positions (their polygons'
    centroids): x_rel differs by ~0.5 % (it only weighs a left-right soil profile).
    """
    n = geo.kind.size
    r_rel = np.full(n, np.nan)
    d_endo, d_epi, xylem80 = geo.layer_dist[3], geo.layer_dist[2], geo.xylem_80_percentile_distance
    endo_row = rank_to_row[3]

    def row_of(cell):
        return rank_to_row[int(geo.cell_rank[cell])]

    for w in geo.walls:
        cells = [int(c) for c in geo.wall_cells[w] if c >= 0]
        if not cells:
            continue
        row1 = row_of(cells[0])
        row2 = row_of(cells[1]) if len(cells) > 1 else np.nan
        if np.isnan(row1):
            continue
        if not np.isnan(row2) and row1 <= endo_row and row2 <= endo_row:
            r = -((distance[int(row1)] + distance[int(row2)]) / 2.0 - xylem80) / (d_endo - xylem80)
            r_rel[w] = max(min(r, -0.00001), -1.0)
        elif np.isnan(row2) and row1 <= endo_row:
            r = -(distance[int(row1)] - xylem80) / (d_endo - xylem80)
            r_rel[w] = max(min(r, -0.00001), -1.0)
        elif not np.isnan(row2):
            r = (d_epi - (distance[int(row1)] + distance[int(row2)]) / 2.0) / (d_epi - d_endo)
            r_rel[w] = min(max(r, 0.00001), 1.0)
        else:
            r = (d_epi - distance[int(row1)]) / (d_epi - d_endo)
            r_rel[w] = min(max(r, 0.00001), 1.0)
    x_cells = geo.x[geo.cells]
    x_min, x_max = x_cells.min(), x_cells.max()
    x_rel = (geo.x - x_min) / (x_max - x_min) if x_max - x_min > 0 else np.zeros(n)
    return r_rel, x_rel


def wall_counts(geo):
    """
    Per local wall node, its membranes' cells by kind (``Mecha._count_surrounding_cells``, its elif chain):
    {"interc", "xyl", "passage", "endo", "stele", "cortex", "exo", "epi"}.
    """
    n = geo.kind.size
    interc = np.zeros(n, dtype=bool)
    interc[geo.typed["intercellular"]] = True
    passage = np.zeros(n, dtype=bool)
    passage[geo.typed["passage"]] = True
    counts = {k: np.zeros(n, dtype=np.int64) for k in ("interc", "xyl", "passage", "endo", "stele", "cortex", "exo",
                                                       "epi")}
    for w in geo.walls:
        for c in geo.wall_cells[w]:
            if c < 0:
                continue
            g = geo.cgroup[c]
            key = ("interc" if interc[c] else "xyl" if g in (13, 19, 20) else "passage" if passage[c] else
                   "endo" if g == 3 else "stele" if g > 4 else "cortex" if g == 4 else "exo" if g == 1 else
                   "epi" if g == 2 else None)
            if key is not None:
                counts[key][w] += 1
    return counts


def tissue_values(P):
    """The legacy's osmotic potentials (hPa) by tissue (os_hetero), and reflection coefficients (s_hetero)."""
    os_cortex, os_sieve, hetero = P["os_cortex"], P["osmotic_sieve"], int(P["os_hetero"])
    if hetero == 0:
        vals = {k: os_cortex for k in ["epi", "exo", "endo", "peri", "stele"] + [f"c{i}" for i in range(1, 9)]}
        vals["comp"] = (os_sieve + os_cortex) / 2
    elif hetero == 1:
        vals = {"epi": -5000, "exo": -5700, "endo": -6200, "peri": -5000, "stele": -7400}
        vals.update({f"c{i}": v for i, v in enumerate([-6400, -7100, -7800, -8500, -9000, -9300, -9000, -8500], 1)})
        vals["comp"] = (os_sieve - 7400) / 2
    elif hetero == 2:
        vals = {"epi": -11200, "exo": -11500, "endo": -10500, "peri": -9200, "stele": -12100}
        vals.update({f"c{i}": v for i, v in enumerate([-11800, -12100, -12400, -12700, -12850, -12950, -12850,
                                                      -12700], 1)})
        vals["comp"] = (os_sieve - 12100) / 2
    elif hetero == 3:
        vals = {k: os_cortex for k in ["epi", "exo"] + [f"c{i}" for i in range(1, 9)]}
        vals.update({"endo": (os_cortex - 5000.0) / 2.0, "peri": -5000.0, "stele": -5000.0})
        vals["comp"] = (os_sieve - 5000.0) / 2
    else:
        endo = P["osmotic_endo_cell"] if P["presets_rules"] == 1 else P["osmotic_endo"]     # legacy problem q
        vals = {"epi": P["osmotic_epi"], "exo": P["osmotic_exo"], "endo": endo,
                "peri": P["osmotic_peri"], "stele": P["osmotic_stele"], "cortex": os_cortex}
        vals.update({f"c{i}": os_cortex for i in range(1, 9)})
        vals["comp"] = (os_sieve + P["osmotic_stele"]) / 2
    vals = {k: (float(v) if not np.isnan(v) else 0.0) for k, v in vals.items()}
    s_factor, s_hetero = P["s_factor"], int(P["s_hetero"])
    outer = ["epi", "exo_epi", "exo_cortex", "cortex", "endo_cortex"]
    inner = ["endo_peri", "peri", "stele", "comp", "sieve"]
    s_out, s_in = {0: (1.0, 1.0), 1: (1.0, 0.5), 2: (0.5, 1.0)}[s_hetero]
    s_vals = {**{k: s_factor * s_out for k in outer}, **{k: s_factor * s_in for k in inner}}
    return vals, s_vals


def section_presets(geo, P, barrier, membranes, per_membrane=False):
    """
    The legacy's osmotic potentials (per local node, hPa) and reflection coefficients (per membrane, in the order of
    *membranes*, rows (wall, cell) of local nodes) of one root section, with its plant's parameters *P* (scalars).
    A cell's potential is the one its last membrane sets, in the legacy's order (walls by id); junctions have none.
    With *per_membrane*, also each membrane's wall potential: the legacy sets it per membrane, so that a wall between
    a cortical-rule cell and a stelar-rule cell enters its equations with two (problem r10; the port keeps one, the
    last).
    """
    rank_to_row, distance, row_outer_cortex = legacy_rows(geo.n_layer, geo.layer_dist)
    r_rel, x_rel = relative_positions(geo, rank_to_row, distance)
    counts = wall_counts(geo)
    vals, s_vals = tissue_values(P)
    interc = np.zeros(geo.kind.size, dtype=bool)
    interc[geo.typed["intercellular"]] = True
    os_sieve = P["osmotic_sieve"]
    psi_os = np.zeros(geo.kind.size)
    sigma = np.zeros(len(membranes))
    wall_side = np.zeros(len(membranes))
    position = np.empty(geo.legacy_order.size, dtype=np.int64)
    position[geo.legacy_order] = np.arange(geo.legacy_order.size)
    order = np.argsort(position[np.asarray([w for w, _ in membranes], dtype=np.int64)], kind="stable")
    rows_present = rank_to_row[~np.isnan(rank_to_row)]
    legacy = P["presets_rules"] == 2
    cortex_rows = {int(rank_to_row[r]) for r in range(40, 50) if not np.isnan(rank_to_row[r])}
    for k in order:
        i, j = (int(v) for v in membranes[k])
        r_val = float(r_rel[i]) if not np.isnan(r_rel[i]) else 0.0
        x_val = float(x_rel[i]) if not np.isnan(x_rel[i]) else 0.5
        os_soil = 0.0
        if P["osmotic_symmetry_soil"] == 2:
            if P["osmotic_diffusivity_soil"] == 0:
                os_soil = float(P["osmotic_left_soil"] + (P["osmotic_right_soil"] - P["osmotic_left_soil"])
                                * abs(r_val) ** P["osmotic_shape_soil"])
            elif r_val >= 0:
                os_soil = P["osmotic_left_soil"] * np.exp(0.0)              # the convection velocity u: 0
        elif P["osmotic_symmetry_soil"] == 1:
            os_soil = float(P["osmotic_left_soil"] * (1 - x_val) + P["osmotic_right_soil"] * x_val)
        os_xyl = 0.0
        if P["osmotic_symmetry_xyl"] == 2:
            if P["osmotic_diffusivity_xyl"] == 0:
                os_xyl = float(P["osmotic_endo"] + (P["osmotic_xyl"] - P["osmotic_endo"])
                               * (1 - abs(r_val)) ** P["osmotic_shape_xyl"])
            elif r_val < 0:
                os_xyl = P["osmotic_xyl"] * np.exp(-0.0)
        elif P["osmotic_symmetry_xyl"] == 1:
            os_xyl = float((P["osmotic_xyl"] + P["osmotic_endo"]) / 2)
        rank = int(geo.cell_rank[j])
        row = int(rank_to_row[rank]) if rank in rows_present else 0     # the legacy's test (p): a row value, not a rank
        cell_os, wall_os, sig = 0.0, os_soil, 0.0
        cgroup = geo.cgroup[j]
        mature = barrier > 0
        if rank == 1:
            cell_os, sig = vals["exo"], (s_vals["exo_epi"] if counts["epi"][i] == 1 else s_vals["exo_cortex"])
        elif rank == 2:
            cell_os, sig = vals["epi"], s_vals["epi"]
        elif rank == 3:
            cell_os, sig = vals["endo"], s_vals["endo_cortex"]
            if counts["stele"][i] > 0 and counts["endo"][i] > 0:
                sig, wall_os = s_vals["endo_peri"], (os_xyl if mature else os_soil)
            elif counts["stele"][i] == 0 and counts["cortex"][i] > 0:
                wall_os = os_soil
            else:
                wall_os, sig = (os_xyl if mature else os_soil), s_vals["endo_peri"]
        elif 40 <= rank < 50:
            if interc[j]:
                cell_os, wall_os, sig = os_soil, os_soil, 0
            else:
                if legacy:                       # problem p: every cortex cell in row 0, and the layers reversed
                    key = f"c{max(1, min(8, 8 - (row_outer_cortex - row)))}"
                else:                            # the original's: c1 the outer cortex row, c8 the 8th and deeper
                    depth = row_outer_cortex - int(rank_to_row[rank]) + 1
                    key = f"c{depth}" if 1 <= depth <= 8 and len(cortex_rows) > 1 else "c8"
                cell_os, sig = vals.get(key, vals.get("c1", 1)), s_vals["cortex"]
        elif cgroup == 5:
            cell_os, sig, wall_os = vals["stele"], s_vals["stele"], (os_xyl if mature else os_soil)
        elif rank == 16:
            cell_os, sig, wall_os = vals["peri"], s_vals["peri"], (os_xyl if mature else os_soil)
        elif cgroup in (11, 23):
            cell_os = os_sieve if not np.isnan(os_sieve) else vals["stele"]
            sig, wall_os = s_vals["sieve"], (os_xyl if mature else os_soil)
        elif cgroup in (12, 26):
            cell_os, sig, wall_os = vals["comp"], s_vals["comp"], (os_xyl if mature else os_soil)
        elif cgroup in (13, 19, 20):
            if not mature:
                cell_os, sig = vals["stele"], s_vals["stele"]
            else:
                cell_os, sig, wall_os = os_xyl, 0.0, os_xyl
        psi_os[i], psi_os[j], sigma[k], wall_side[k] = wall_os, cell_os, sig, wall_os
    return (psi_os, sigma, wall_side) if per_membrane else (psi_os, sigma)


@dataclass
class MechaOsmoticPresets(MechaSections):
    """
    MECHA's osmotic presets: the osmotic potentials of the root sections' cells and walls, and their membranes'
    reflection coefficients, from the legacy's scenario rules (os_hetero, s_hetero, the soil's and the xylem's
    profiles). Their outputs are MechaHydraulics' "Solutes" inputs.
    """
    presets_sections: float = _p("", "the sections the presets apply to: 1 the roots (MECHA's rules are root rules), "
                                     "2 every section (the reactions' needle, its comparison)", 1.)
    presets_rules: float = _p("", "1: the original MECHA's presets (its cortex layers c1-c8 from the outer cortex "
                                  "inward; the endodermis cells' own osmotic_endo_cell), 2: the legacy's (all cortex "
                                  "cells in c1, problem p; the endodermis cells at osmotic_endo, problem q), for the "
                                  "comparison", 1.)
    os_hetero: float = _p("", "the cells' osmotic potentials: 0 os_cortex everywhere, 1 and 2 the legacy's profiles, "
                              "3 cortex os_cortex and stele -5000 hPa, 4 per tissue (osmotic_epi...)", 0.)
    os_cortex: float = _p("hPa", "the cortex cells' osmotic potential", -4.8E3)
    osmotic_sieve: float = _p("hPa", "the sieve tubes' osmotic potential (NaN: the stele's)", -1.0E4)
    osmotic_epi: float = _p("hPa", "the epidermis cells' osmotic potential (os_hetero 4)", -4.8E3)
    osmotic_exo: float = _p("hPa", "the exodermis cells' osmotic potential (os_hetero 4)", -4.8E3)
    osmotic_endo_cell: float = _p("hPa", "the endodermis cells' osmotic potential (os_hetero 4; presets_rules 2, the "
                                         "legacy's: osmotic_endo, the stelar apoplast's)", -4.8E3)
    osmotic_peri: float = _p("hPa", "the pericycle cells' osmotic potential (os_hetero 4)", -4.8E3)
    osmotic_stele: float = _p("hPa", "the stele cells' osmotic potential (os_hetero 4)", -4.8E3)
    s_hetero: float = _p("", "the reflection coefficients: 0 uniform, 1 the stele's halved, 2 the cortex's halved", 0.)
    s_factor: float = _p("", "the membranes' reflection coefficient (× the s_hetero pattern)", 1.0)
    osmotic_left_soil: float = _p("hPa", "the soil solution's osmotic potential, left (or outer, symmetry 2) side",
                                  -0.5E3)
    osmotic_right_soil: float = _p("hPa", "the soil solution's osmotic potential, right (or inner) side", -0.5E3)
    osmotic_symmetry_soil: float = _p("", "the soil solution's profile in the cortex walls: 1 left to right, 2 radial",
                                      1.)
    osmotic_shape_soil: float = _p("", "the radial profile's exponent (symmetry 2)", 1.0)
    osmotic_diffusivity_soil: float = _p("cm2 d-1", "the soil solutes' diffusivity (symmetry 2; not 0: the legacy's "
                                                    "convective profile, flat as its velocity is 0)", 1.)
    osmotic_xyl: float = _p("hPa", "the xylem sap's osmotic potential", -1.80E3)
    osmotic_endo: float = _p("hPa", "the stelar apoplast's osmotic potential at the endodermis (symmetry 2)", -4.80E3)
    osmotic_symmetry_xyl: float = _p("", "the xylem sap's profile in the stele walls: 1 uniform (mean of osmotic_xyl "
                                         "and osmotic_endo), 2 radial", 2.)
    osmotic_shape_xyl: float = _p("", "the radial profile's exponent (symmetry 2)", 1.0)
    osmotic_diffusivity_xyl: float = _p("cm2 d-1", "the xylem solutes' diffusivity (symmetry 2; not 0: flat, at "
                                                   "osmotic_xyl)", 0.5)

    osmotic_potential: float = state_variable(**_doc("hPa", "osmotic potential of the protoplast or the wall's "
                                                            "solution (0: junctions, shoot sections)"),
                                              initialize=0., scale=SCALES.Compartment, location="node",
                                              state_variable_type="NonInertialIntensive")
    reflection_coefficient: float = state_variable(**_doc("", "the membrane's reflection coefficient (0: other "
                                                              "edges, shoot sections)"),
                                                   initialize=0., scale=SCALES.Connection, location="edge",
                                                   state_variable_type="NonInertialIntensive")

    @rate(include_inactive=True)
    def _osmotic_potential(self, kind, barrier, anatomy_template) -> tuple[np.ndarray, str, np.ndarray]:
        """The presets of every root section, each with its plant's parameters and its barrier."""
        ds = self.data_structure
        psi_os, sigma = np.zeros(ds.n_nodes()), np.zeros(ds.n_edges())
        view = ds.to_graph_view()
        path = np.asarray(ds.get("network_path"))
        barrier = np.asarray(barrier)
        names = [f.name for f in self.__dataclass_fields__.values() if f.metadata.get("variable_type") == "parameter"]
        table = {name: np.asarray(self.parameter_values(name, "node")) for name in names}
        roots = self.section_roles()[1]
        every = np.asarray(table["presets_sections"]) == 2
        for segment, (nodes, edges, geo) in self.section_geometries().items():
            if segment not in roots and not every[nodes[0]]:
                continue
            local = np.full(ds.n_nodes(), -1, dtype=np.int64)
            local[nodes] = np.arange(nodes.size)
            e = edges[path[edges] == MEMBRANE]
            a, b = local[view.tail[e]], local[view.head[e]]
            wall_first = geo.kind[a] == 1
            membranes = np.c_[np.where(wall_first, a, b), np.where(wall_first, b, a)]
            v = nodes[0]
            P = {name: float(values[v]) for name, values in table.items()}
            os_local, sigma_local = section_presets(geo, P, int(barrier[v]), membranes)
            psi_os[nodes] = os_local
            sigma[e] = sigma_local
        return psi_os, "reflection_coefficient", sigma
