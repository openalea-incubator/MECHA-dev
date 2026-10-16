"""
Transpiration by the air spaces' walls (devplan_metafspm.md, §25): the reactions branch's needle water problem
(``simulations/needle.py``: ``InData.needle_defaults()``, its evaporating walls pinned at −1e4 hPa, its wall_air
Kelvin exchange and air links) on GRANAP's ``NeedleAnatomy(seed=42)``, against the port with the reactions' choices
(legacy_options, needle parameters, evaporating_walls 2, membrane_floor_rules 2, presets on every section).

Scenario 0 (the xylem at 0): the pinned walls, the penalty, every edge's conductance, the solution. Scenario 1 (the
default root scenario with s_factor 0.7, the xylem at −200 hPa): the presets' π and σ; the legacy's solution satisfies
the port's equations exactly once the legacy's per-membrane wall potentials (problem r10) are its drives.
"""
import numpy as np
import pytest
import scipy.sparse as sp

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.geometry import MEMBRANE, WALL_AIR, section_parts
from openalea.mecha import osmotic as O

from legacy_run import legacy_options, legacy_reactions_needle
from plants import anatomy_data_structure, needle, needle_anatomy_component, template_organ

NEEDLE = dict(kw=2.5634e-3, kw_meso_hypo=1e-16, kmb=3.6314e-4, x_contact=1e10, k_xyl=1.0, membrane_floor_rules=2.,
              evaporating_walls=2., transpiration_model=2., air_space_psi=-1e4, standard_solve_shoot=1.,
              standard_psi_soil=0., collar_condition=0.)


@pytest.fixture(scope="module")
def needle_case():
    g, ds, anatomy = anatomy_data_structure(needle(), needle_anatomy_component())
    segment = int(ds.entity_ids("SubOrgan")[0])
    reference = legacy_reactions_needle(template_organ(anatomy, g, segment))
    ds.register("barrier", np.ones(ds.n_nodes()), location="node")
    mecha = legacy_options(MechaHydraulics(data_structure=ds))
    for name, value in {**NEEDLE, "standard_psi_xylem": 0.}.items():
        setattr(mecha, name, value)
    mecha()
    scenario0 = {name: np.asarray(ds.get(name)).copy() for name in
                 ("standard_psi", "conductance", "air_wall_conductance", "air_tail_conductance",
                  "air_head_conductance", "air_offset")}
    presets = O.MechaOsmoticPresets(data_structure=ds)
    for name, value in dict(presets_sections=2., presets_rules=2., s_factor=0.7).items():
        setattr(presets, name, value)
    presets()
    mecha.standard_osmotic, mecha.standard_psi_xylem = 1., -200.
    mecha()
    nodes, edges = section_parts(ds)[segment]
    geo = mecha.section_geometries()[segment][2]
    net = reference["network"]
    mecha_index = g.property("mecha_index")
    index = np.array([int(mecha_index[int(v)]) for v in ds.entity_ids("node")[nodes]])
    index[geo.junction_order] = net.n_walls + np.arange(net.n_junctions)
    local = np.full(ds.n_nodes(), -1, dtype=np.int64)
    local[nodes] = np.arange(nodes.size)
    view = ds.to_graph_view()
    ends = np.c_[index[local[view.tail[edges]]], index[local[view.head[edges]]]]
    return dict(ds=ds, mecha=mecha, presets=presets, reference=reference, nodes=nodes, edges=edges, geo=geo,
                index=index, local=local, view=view, ends=ends, scenario0=scenario0)


def _legacy(table, u, v):
    return table.get((u, v), table.get((v, u)))


def test_pinned_walls_and_penalty(needle_case):
    c = needle_case
    penalty = c["scenario0"]["air_wall_conductance"][c["nodes"]]
    assert set(c["index"][penalty > 0].tolist()) == set(c["reference"]["walls"])
    assert np.unique(penalty[penalty > 0]).tolist() == [c["reference"]["k_air"]]


def test_edge_conductances(needle_case):
    """Every edge's K (air links included) and every wall_air edge's K_wall, K_air and offset, bit-exact."""
    c = needle_case
    path = np.asarray(c["ds"].get("network_path"))
    K, s0 = c["scenario0"]["conductance"], c["scenario0"]
    legacy_K = c["reference"]["scenarios"][0]["edge_K"]
    checked = 0
    for e, (u, v) in zip(c["edges"], c["ends"]):
        if int(path[e]) == WALL_AIR:
            K_wall, K_air, offset = _legacy(c["reference"]["wall_air"], int(u), int(v))
            wall_is_tail = u < c["reference"]["network"].n_walls
            tail, head = (K_wall, K_air) if wall_is_tail else (K_air, K_wall)
            assert (s0["air_tail_conductance"][e], s0["air_head_conductance"][e]) == (tail, head)
            assert s0["air_offset"][e] == (offset if wall_is_tail else -offset)
            checked += 1
            continue
        reference = _legacy(legacy_K, int(u), int(v))
        if reference is not None:
            assert K[e] == reference
            checked += 1
    assert checked >= len(legacy_K)


def test_scenario0_solution(needle_case):
    """To 1e-11 of the range: the penalty (1e3 × the largest diagonal term) makes the system less well conditioned
    than the roots' (1e-12, D1)."""
    c = needle_case
    psi = c["scenario0"]["standard_psi"][c["nodes"]]
    reference = c["reference"]["scenarios"][0]["solution"][c["index"]]
    np.testing.assert_allclose(psi, reference, rtol=0, atol=1e-11 * abs(reference).max())


def test_scenario1_presets(needle_case):
    c = needle_case
    pi = np.asarray(c["ds"].get("osmotic_potential"))[c["nodes"]]
    by_legacy = dict(zip(c["index"].tolist(), range(c["nodes"].size)))
    for node, value in c["reference"]["scenarios"][1]["psi_os"].items():
        assert pi[by_legacy[node]] == pytest.approx(0.0 if value is None else value, rel=1e-13, abs=0)
    sigma = np.asarray(c["ds"].get("reflection_coefficient"))
    path = np.asarray(c["ds"].get("network_path"))
    for e, (u, v) in zip(c["edges"], c["ends"]):
        if int(path[e]) == MEMBRANE:
            assert sigma[e] == _legacy(c["reference"]["scenarios"][1]["sigma"], int(u), int(v))


def test_scenario1_legacy_solution_satisfies_the_port_equations(needle_case):
    """
    The legacy's scenario-1 solution in the port's balance (its conductances, wall_air exchange, pinned walls, xylem
    boundary), the membranes' drives with the legacy's per-membrane wall potentials (problem r10): zero residual, to
    the round-off of the penalty's terms. With the port's one potential per wall, the 165 walls where the two differ
    (transfusion tissue against Strasburger and parenchyma cells, the guard cells' walls) leave a residual.
    """
    c = needle_case
    ds, geo, nodes, edges, local, view = c["ds"], c["geo"], c["nodes"], c["edges"], c["local"], c["view"]
    path = np.asarray(ds.get("network_path"))
    P = {f.name: float(np.asarray(c["presets"].parameter_values(f.name, "node"))[nodes[0]])
         for f in O.MechaOsmoticPresets.__dataclass_fields__.values() if f.metadata.get("variable_type") == "parameter"}
    membranes = edges[path[edges] == MEMBRANE]
    a, b = local[view.tail[membranes]], local[view.head[membranes]]
    wall_first = geo.kind[a] == 1
    pairs = np.c_[np.where(wall_first, a, b), np.where(wall_first, b, a)]
    pi, sigma, wall_side = O.section_presets(geo, P, 1, pairs, per_membrane=True)
    drive = np.zeros(ds.n_edges())
    d = sigma * (wall_side - pi[pairs[:, 1]])
    drive[membranes] = np.where(wall_first, d, -d)
    assert (np.abs(wall_side - pi[pairs[:, 0]]) > 0).sum() > 0             # r10 is present on the needle
    x = np.zeros(ds.n_nodes())
    x[nodes] = c["reference"]["scenarios"][1]["solution"][c["index"]]
    B = sp.csr_matrix(ds.incidence_matrix()) if not sp.issparse(ds.incidence_matrix()) else ds.incidence_matrix()
    tail, head = np.asarray(B.maximum(0).T @ x).ravel(), np.asarray((-B).maximum(0).T @ x).ravel()
    flow = ds.get("conductance") * (np.asarray(B.T @ x).ravel() + drive) + ds.get("air_tail_conductance") * tail \
        - ds.get("air_head_conductance") * head + ds.get("air_offset")
    inflow = ds.get("air_wall_conductance") * (ds.get("standard_air_wall_psi") - x) + \
        ds.get("axial_boundary_conductance") * (ds.get("standard_axial_psi") - x)
    residual = (np.asarray(B @ flow).ravel() - inflow)[nodes]
    scale = float((ds.get("air_wall_conductance") * abs(x)).max())
    assert np.abs(residual).max() <= 1e-14 * scale
