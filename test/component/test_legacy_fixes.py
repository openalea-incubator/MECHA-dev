"""The legacy's problems fixed in the port (devplan_metafspm.md, §18): each fix's rule, where the legacy differs."""
from types import SimpleNamespace

import numpy as np
import pytest

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.conductance import boundary_terms

from plants import anatomy_data_structure, root


def test_a1_protosieve_cell_counted_once():
    """§2 f: a protosieve cell listed twice by the legacy gets k_sieve once in the phloem boundary term (barrier 0)."""
    n = 3
    net = SimpleNamespace(cgroup=np.full(n, 23), x=np.zeros(n), border_wall=np.zeros(n, bool),
                          border_junction=np.zeros(n, bool), wall_length=np.zeros(n), cell_area=np.ones(n),
                          wall_thickness=np.full(n, np.nan),
                          xylem=np.zeros(n, np.int64), protosieve=np.array([0, 1, 2]))
    P = {name: np.full(n, value) for name, value in
         (("kw", 2.4e-4), ("thickness", 1.5), ("x_contact", 0.), ("k_xyl", 1e-6), ("k_sieve", 3e-6),
          ("standard_axial_source", 2.), ("thickness_source", 1.))}
    _, axial, rhs = boundary_terms(net, P, barrier=np.zeros(n), height=np.full(n, 200.))
    assert axial.tolist() == [0., 3e-6, 3e-6] and rhs.tolist() == axial.tolist()


def test_a2_each_vessel_gets_its_own_share():
    """§2 o: with a flow at the xylem, each vessel (the network's, by cgroup) receives the flow × its own area fraction,
    and kr's xylem potential is those vessels' mean."""
    g, ds, anatomy = anatomy_data_structure(root(1))
    ds.register("barrier", np.full(ds.n_nodes(), 3.), location="node")
    mecha = MechaHydraulics(data_structure=ds)
    mecha.standard_axial_condition, mecha.standard_axial_flow = 2., 1e-3
    mecha()
    nodes, _, geo = mecha.section_geometries()[int(ds.entity_ids("SubOrgan")[0])]
    vessels = geo.xylem_cells
    inflow = np.asarray(ds.get("standard_axial_inflow"))[nodes]
    area = geo.cell_area[vessels]
    np.testing.assert_allclose(-inflow[vessels], 1e-3 * area / area.sum(), rtol=1e-14)
    assert inflow.sum() == pytest.approx(-1e-3, rel=1e-12)
    psi_xylem = np.asarray(ds.get("standard_psi"))[nodes][vessels].mean()
    kr = 1e-3 / (mecha.standard_psi_soil - psi_xylem) / ds.get("perimeter")[0] / (ds.get("length")[0] * 1e6) / 1e-4
    assert ds.get("kr")[0] == pytest.approx(kr, rel=1e-12)


def _original_plasmodesmata(gi, gj, ppp_i, ppp_j, interc, b, length, outer, conc, P, pieces, stelar):
    """One edge, transcribed from the original MECHA's radial plasmodesmata (src_old/6_MECHAv4_SRS2025.py), with the
    tissues the legacy added (17, 18, 23: their own frequencies, before the default; Q-L1)."""
    from openalea.mecha.conductance import FPLXHEIGHT_ADDED, fplxheight_name
    gi, gj = (13 if g in (19, 20) else 16 if g == 21 else g for g in (gi, gj))
    t = P["kpl_cortex_factor"]
    F = {k: P["fplxheight_" + k] for k in ("default", "epi_exo", "outer_cortex", "cortex_cortex", "cortex_endo",
                                            "endo_endo", "endo_peri", "peri_stele", "stele_stele", "stele_comp",
                                            "peri_comp", "comp_comp", "comp_sieve", "peri_sieve", "stele_sieve")}
    ppp = P["kpl_PPP_factor"] if (ppp_i or ppp_j) else 1
    pcc = P["kpl_PCC_factor"]
    h = lambda x, y: 2 * x * y / (x + y) if x + y else 0.0                        # noqa: E731
    if interc and b > 0:
        f = 0.0
    elif pieces and gi == 13 and gj == 13:
        f = 10000 * F["default"] * length
    elif b > 0 and 13 in (gi, gj):
        f = 0.0
    elif {gi, gj} == {1, 2}:
        f = F["epi_exo"] * length
    elif sorted((gi, gj)) == sorted((outer, 4)) and outer != 4:
        f = h(t, 1) * F["outer_cortex"] * length * (conc[0] if b > 0 else 1)
    elif gi == 4 and gj == 4:
        f = t * F["cortex_cortex"] * length * (conc[1] if b > 0 else 1)
    elif {gi, gj} == {3, 4}:
        f = h(t, 1) * F["cortex_endo"] * length * (conc[2] if b > 0 else 1)
    elif gi == 3 and gj == 3:
        f = F["endo_endo"] * length
    elif {gi, gj} == {3, 16}:
        f = h(ppp, 1) * F["endo_peri"] * length
    elif (gi == 16 and gj in (5, 13)) or (gj == 16 and gi in (5, 13)):
        f = h(ppp, 1) * F["peri_stele"] * length
    elif (gi in (5, 13) and gj == 12) or (gi == 12 and gj in (5, 13)):
        f = h(pcc, 1) * F["stele_comp"] * length
    elif {gi, gj} == {12, 16}:
        f = h(pcc, ppp) * F["peri_comp"] * length
    elif gi == 12 and gj == 12:
        f = pcc * F["comp_comp"] * length
    elif {gi, gj} == {11, 12}:
        f = h(pcc, 1) * F["comp_sieve"] * length
    elif {gi, gj} == {11, 16}:
        f = h(ppp, 1) * F["peri_sieve"] * length
    elif (gi in (5, 13) and gj == 11) or (gi == 11 and gj in (5, 13)):
        f = F["stele_sieve"] * length
    elif gi in (5, 13) and gj in (5, 13):
        f = F["stele_stele"] * length
    elif tuple(sorted((gi, gj))) in FPLXHEIGHT_ADDED:
        f = P[fplxheight_name(tuple(sorted((gi, gj))))] * length
    else:
        f = F["default"] * length
    return f * (P["kpl_endo_in_factor"] if stelar else P["kpl_endo_out_factor"])


def _expected_plasmodesmata(net, e, P, barrier, pieces):
    """The transcription on the edges *e* of *net*, the phloem-pole pericycle found by scanning them."""
    a, b = net.a[e], net.b[e]
    ppp = set()
    for x, y in zip(a, b):
        for p, q in ((x, y), (y, x)):
            if net.cgroup[p] in (16, 21) and net.cgroup[q] in (11, 23):
                ppp.add(int(p))
    inner = lambda c: c == 3 or c > 4                                              # noqa: E731
    expected = np.array([
        _original_plasmodesmata(
            int(net.cgroup[x]), int(net.cgroup[y]), int(x) in ppp, int(y) in ppp,
            bool(net.intercellular[x] or net.intercellular[y]), barrier, 1.0E-04 * net.edge_length[k],
            int(net.outercortex_connec_rank[net.section_of_edge[k]]), net.concentrated[net.section_of_edge[k]], P,
            pieces > 0, inner(net.cgroup[x]) and inner(net.cgroup[y]))
        for k, x, y in zip(e, a, b)])
    return expected, ppp


FACTORS = dict(kpl_cortex_factor=0.5, kpl_PPP_factor=0.3, kpl_PCC_factor=0.7, kpl_endo_in_factor=0.8,
               kpl_endo_out_factor=1.3)


@pytest.fixture(scope="module")
def dicot():
    from plants import anatomy_component
    g, ds, anatomy = anatomy_data_structure(root(1), anatomy_component("for_dicot_root"))
    ds.register("barrier", np.ones(ds.n_nodes()), location="node")
    mecha = MechaHydraulics(data_structure=ds)
    for name, value in FACTORS.items():
        setattr(mecha, name, value)
    return ds, mecha


@pytest.mark.parametrize("barrier, pieces", [(0, 0.), (1, 0.), (3, 1.)])
def test_q_l1_original_plasmodesmata_rules(dicot, barrier, pieces):
    """§18b: the plasmodesmata by the original's rules, every edge of the dicot root against a transcription of its
    code, with factors away from 1 so that each branch shows."""
    from openalea.mecha.geometry import PLASMODESMATA
    ds, mecha = dicot
    ds.set("barrier", np.full(ds.n_nodes(), float(barrier)))
    mecha.xylem_pieces = pieces
    mecha()
    net = mecha.network()
    P = {name: float(np.asarray(mecha.parameter_values(name, "node"))[0]) for name in mecha.PARAMETERS}
    e = np.flatnonzero(net.in_section & (net.path == PLASMODESMATA))
    expected, _ = _expected_plasmodesmata(net, e, P, barrier, pieces)
    np.testing.assert_allclose(np.asarray(ds.get("conductance"))[e], P["kpl"] * expected, rtol=1e-13, atol=0)


@pytest.mark.parametrize("barrier, outer", [(0, 1), (1, 2), (3, 1)])
def test_q_l1_every_branch(dicot, barrier, outer):
    """The same on the dicot's network with its cells given random cgroups (all the tissues the rules name, the
    newer CellSet ones and the added ones), so that the branches GRANAP's presets do not reach are checked too:
    phloem-pole pericycle, pericycle-companion and -sieve, companion-sieve, an epidermis as outer layer."""
    from dataclasses import replace
    from openalea.mecha.conductance import _plasmodesmata_original
    from openalea.mecha.geometry import PLASMODESMATA
    ds, mecha = dicot
    mecha.xylem_pieces = 0.
    net = mecha.network()
    rng = np.random.default_rng(barrier)
    cells = np.flatnonzero(net.cgroup >= 0)
    cgroup = net.cgroup.copy()
    cgroup[cells] = rng.choice([1, 2, 3, 4, 5, 11, 12, 13, 16, 17, 18, 19, 21, 23], cells.size)
    net = replace(net, cgroup=cgroup, outercortex_connec_rank=np.full_like(net.outercortex_connec_rank, outer))
    P_edge = mecha.parameter_table("edge")
    P = {name: float(values[0]) for name, values in P_edge.items()}
    e = np.flatnonzero(net.in_section & (net.path == PLASMODESMATA))
    got = _plasmodesmata_original(net, e, P_edge, np.full(e.size, barrier))
    expected, ppp = _expected_plasmodesmata(net, e, P, barrier, 0.)
    np.testing.assert_allclose(got, expected, rtol=1e-13, atol=0)
    assert len(ppp) > 0


def test_q_l4_each_wall_its_own_thickness():
    """§18e: with thickness_source 2 (default), a wall edge's conductance uses its wall's own thickness (GRANAP's),
    and a border junction's soil term the mean of its walls'; against thickness_source 1 (MECHA's one value)."""
    from openalea.mecha.geometry import WALL
    g, ds, anatomy = anatomy_data_structure(root(1))
    ds.register("barrier", np.full(ds.n_nodes(), 1.), location="node")
    mecha = MechaHydraulics(data_structure=ds)
    mecha()
    K2, soil2 = ds.get("conductance").copy(), ds.get("soil_conductance").copy()
    mecha.thickness_source = 1.
    mecha()
    K1, soil1 = ds.get("conductance").copy(), ds.get("soil_conductance").copy()
    net = mecha.network()
    own = np.asarray(ds.get("wall_thickness"), dtype=float)
    e = np.flatnonzero(net.in_section & net.junction_edge & (K1 > 1e-20))
    wall = np.where(net.kind[net.a[e]] == WALL, net.a[e], net.b[e])
    t, t1, h = own[wall], mecha.thickness, np.asarray(ds.get("length"))[wall] * 1e6
    lateral = net.edge_lateral_distance[e]
    np.testing.assert_allclose(K2[e] / K1[e], ((lateral + h) * t - t ** 2) / ((lateral + h) * t1 - t1 ** 2), rtol=1e-12)
    walls = np.flatnonzero((soil1 > 0) & (net.kind == WALL))
    np.testing.assert_allclose(soil2[walls] / soil1[walls], t1 / own[walls], rtol=1e-12)
    junctions = np.flatnonzero((soil1 > 0) & (net.kind != WALL))
    np.testing.assert_allclose(soil2[junctions] / soil1[junctions], t1 / net.wall_thickness[junctions], rtol=1e-12)
    assert np.isfinite(net.wall_thickness[junctions]).all() and junctions.size > 0


def test_plasmodesmata_diffusion_through_the_crossed_wall():
    """With thickness_source 2 (default), the plasmodesmata's solute diffusion factor divides by the thickness of the
    wall they cross (its own, GRANAP's); with 1, by MECHA's one value (the legacy's, for the comparisons)."""
    from openalea.mecha.geometry import PLASMODESMATA, WALL
    g, ds, anatomy = anatomy_data_structure(root(1))
    ds.register("barrier", np.full(ds.n_nodes(), 1.), location="node")
    mecha = MechaHydraulics(data_structure=ds)
    mecha()
    own_factor = ds.get("symplastic_diffusion_factor").copy()
    mecha.thickness_source = 1.
    mecha()
    one_factor = ds.get("symplastic_diffusion_factor").copy()
    net = mecha.network()
    e = np.flatnonzero(net.in_section & (net.path == PLASMODESMATA) & (one_factor > 0))
    walls = np.flatnonzero(net.kind == WALL)
    crossed = {tuple(sorted(cells)): wall for wall, cells in zip(walls, net.wall_cells[walls].tolist())}
    wall = np.array([crossed[tuple(sorted((a, b)))] for a, b in zip(net.a[e].tolist(), net.b[e].tolist())])
    own = np.asarray(ds.get("wall_thickness"), dtype=float)[wall]
    assert e.size > 0 and not np.allclose(own, mecha.thickness)
    np.testing.assert_allclose(own_factor[e] / one_factor[e], mecha.thickness / own, rtol=1e-12)


def _presets_section(**parameters):
    from openalea.mecha.osmotic import MechaOsmoticPresets, legacy_rows
    g, ds, anatomy = anatomy_data_structure(root(1))
    ds.register("barrier", np.full(ds.n_nodes(), 1.), location="node")
    presets = MechaOsmoticPresets(data_structure=ds)
    for name, value in parameters.items():
        setattr(presets, name, value)
    presets()
    nodes, _, geo = presets.section_geometries()[int(ds.entity_ids("SubOrgan")[0])]
    rank_to_row, _, row_outer = legacy_rows(geo.n_layer, geo.layer_dist)
    return ds, nodes, geo, rank_to_row, row_outer


def test_p_cortex_layers_from_the_outer_cortex():
    """Problem p: with os_hetero 1, each cortex cell takes its layer's value, c1 on the outer cortex row, c2 the next
    inward, ..., c8 the 8th and deeper (the original's); the legacy gives them all c1's."""
    from openalea.mecha.osmotic import tissue_values
    ds, nodes, geo, rank_to_row, row_outer = _presets_section(os_hetero=1., presets_rules=1.)
    pi = np.asarray(ds.get("osmotic_potential"))[nodes]
    vals, _ = tissue_values({"os_hetero": 1, "os_cortex": -4800., "osmotic_sieve": -1e4, "s_factor": 1., "s_hetero": 0})
    interc = np.zeros(geo.kind.size, dtype=bool)
    interc[geo.typed["intercellular"]] = True
    cortex = [c for c in geo.cells if 40 <= geo.cell_rank[c] < 50 and not interc[c]]
    for c in cortex:
        depth = row_outer - int(rank_to_row[geo.cell_rank[c]]) + 1
        assert pi[c] == vals[f"c{depth}" if 1 <= depth <= 8 else "c8"]
    assert len({pi[c] for c in cortex}) > 1
    ds, nodes, *_ = _presets_section(os_hetero=1., presets_rules=2.)
    assert len({np.asarray(ds.get("osmotic_potential"))[nodes][c] for c in cortex}) == 1


def test_q_endodermis_cells_own_potential():
    """Problem q: with os_hetero 4, the endodermis cells take osmotic_endo_cell, whatever the stelar apoplast's
    osmotic_endo."""
    ds, nodes, geo, *_ = _presets_section(os_hetero=4., presets_rules=1., osmotic_endo_cell=-6100., osmotic_endo=-4800.)
    pi = np.asarray(ds.get("osmotic_potential"))[nodes]
    endo = [c for c in geo.cells if geo.cell_rank[c] == 3]
    assert endo and all(pi[c] == -6100. for c in endo)
