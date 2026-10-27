"""
The port's section-scale results against the legacy's. On the legacy's own solution,
the uptake, kr and the edge flows are bit-exact; Kx is bit-exact; from the port's own solve, kr and the flows agree to
1e-12 (decision D1).
"""
import numpy as np
import pytest

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.geometry import section_parts
from openalea.mecha.postprocess import radial_conductivity, soil_uptake

from conftest import BARRIERS
from legacy_run import legacy_options

PSI_SOIL, PSI_XYL = 0.0, -5.0E3


@pytest.fixture(scope="module")
def port(case):
    ds, g = case["ds"], case["g"]
    if not ds.has("barrier"):
        ds.register("barrier", np.ones(ds.n_nodes()), location="node")
    mecha = legacy_options(MechaHydraulics(data_structure=ds))
    nodes, edges = section_parts(ds)[case["segment"]]
    geo = mecha.section_geometries()[case["segment"]][2]
    net = case["legacy"].network
    mecha_index = g.property("mecha_index")
    legacy = np.array([int(mecha_index[int(v)]) for v in ds.entity_ids("node")[nodes]])
    legacy[geo.junction_order] = net.n_walls + np.arange(net.n_junctions)
    stages = {}
    for i, barrier in enumerate(BARRIERS):
        ds.set("barrier", np.full(ds.n_nodes(), float(barrier)))
        mecha()
        stages[barrier] = {name: np.asarray(ds.get(name)).copy() for name in
                           ("conductance", "soil_conductance", "standard_psi", "standard_flow", "standard_uptake", "kr", "Kx",
                            "Kx_phloem")}
        stages[barrier]["legacy"] = case["legacy"].stages[i]
    return dict(mecha=mecha, nodes=nodes, edges=edges, geo=geo, legacy=legacy, stages=stages,
                height=case["height"], net=mecha.network())


@pytest.mark.parametrize("barrier", BARRIERS)
def test_kr_on_the_legacy_solution(port, barrier):
    stage = port["stages"][barrier]
    psi = stage["legacy"]["solution"][port["legacy"]]                     # per local node
    uptake = soil_uptake(port["geo"], stage["soil_conductance"][port["nodes"]], psi, PSI_SOIL)
    kr = radial_conductivity(uptake, PSI_SOIL, PSI_XYL, port["geo"].perimeter, port["height"])
    assert kr == stage["legacy"]["kr"]


@pytest.mark.parametrize("barrier", BARRIERS)
def test_kx(port, barrier):
    stage = port["stages"][barrier]
    assert stage["Kx"][0] == float(np.asarray(stage["legacy"]["Kx"]).reshape(-1)[0])


@pytest.mark.parametrize("barrier", BARRIERS)
def test_flows_on_the_legacy_solution(port, barrier):
    stage, net, legacy = port["stages"][barrier], port["net"], port["legacy"]
    local = np.full(net.cgroup.size, -1, dtype=np.int64)
    local[port["nodes"]] = np.arange(port["nodes"].size)
    solution = stage["legacy"]["solution"]
    reference = stage["legacy"]["edge_Q"]
    K = stage["conductance"]
    checked = 0
    for e in port["edges"]:
        u, v = int(legacy[local[net.a[e]]]), int(legacy[local[net.b[e]]])
        if (u, v) in reference:
            assert K[e] * (solution[u] - solution[v]) == reference[(u, v)]
        elif (v, u) in reference:
            assert K[e] * (solution[v] - solution[u]) == reference[(v, u)]
        else:
            continue
        checked += 1
    assert checked == len(reference)


@pytest.mark.parametrize("barrier", BARRIERS)
def test_port_solve(port, barrier):
    stage, legacy = port["stages"][barrier], port["legacy"]
    reference = stage["legacy"]["solution"]
    psi = stage["standard_psi"][port["nodes"]]
    np.testing.assert_allclose(psi, reference[legacy], rtol=0, atol=1e-12 * abs(reference).max())
    assert stage["kr"][0] == pytest.approx(stage["legacy"]["kr"], rel=1e-12, abs=0)
    flows = stage["standard_flow"][port["edges"]]
    q_max = max(abs(q) for q in stage["legacy"]["edge_Q"].values())
    assert np.isfinite(flows).all() and abs(flows).max() == pytest.approx(q_max, rel=1e-10)


@pytest.mark.parametrize("stage", range(len(BARRIERS)))
def test_axial_flows_by_layer(case, stage):
    """The xylem's flows by layer row (``flow_xyl_layer``) against the legacy's standard problem: postprocess.axial_layers.
    Each is k_xyl (1.0, the comparisons') × a potential difference of 1e-4 hPa: the solutions' agreement to 1e-12 of
    their 5e3 hPa range bounds it to 5e-9 per vessel. Their sum is minus the uptake, to the same."""
    from openalea.mecha.postprocess import axial_layers
    ds = case["ds"]
    if not ds.has("barrier"):
        ds.register("barrier", np.ones(ds.n_nodes()), location="node")
    ds.set("barrier", np.full(ds.n_nodes(), float(BARRIERS[stage])))
    mecha = legacy_options(MechaHydraulics(data_structure=ds))
    mecha()
    layers = axial_layers(mecha, case["segment"])
    reference = np.asarray(case["legacy"].mecha.flow_xyl_layer)[:, stage, 0]
    uptake = float(ds.get("standard_uptake")[0])
    bound = 1e-12 * abs(PSI_XYL) * mecha.k_xyl * len(mecha.section_geometries()[case["segment"]][2].xylem_cells)
    assert abs(layers["xylem"].sum() + uptake) <= bound
    n = min(reference.size, len(layers))
    assert not reference[n:].any() and not layers["xylem"][n:].any()
    np.testing.assert_allclose(layers["xylem"][:n], reference[:n], rtol=0, atol=bound)


class _Element(dict):
    """An XML element's stand-in (its attributes by get)."""


@pytest.mark.parametrize("stage", range(len(BARRIERS)))
def test_measured_axial_conductances(case, stage):
    """MECHA's Kax_source 2 (each vessel's conductance given, K_xyl by cell id): Kx with measured_axial_conductivity
    (their specific values: × the height) against the legacy's calculate_axial_conductance, to 1e-15; here twice
    Poiseuille's, so that Kx doubles."""
    ds, g = case["ds"], case["g"]
    if not ds.has("barrier"):
        ds.register("barrier", np.ones(ds.n_nodes()), location="node")
    ds.set("barrier", np.full(ds.n_nodes(), float(BARRIERS[stage])))
    mecha = legacy_options(MechaHydraulics(data_structure=ds))
    nodes, _ = section_parts(ds)[case["segment"]]
    geo = mecha.section_geometries()[case["segment"]][2]
    from openalea.mecha.postprocess import poiseuille, VISCOSITY_XYLEM
    height = case["height"]
    vessels = geo.xylem_cells
    conductance = 2 * poiseuille(geo.cell_area[vessels], height, VISCOSITY_XYLEM)       # cm3 hPa-1 d-1
    measured = np.full(ds.n_nodes(), np.nan)
    measured[nodes[vessels]] = conductance * height / 1.0E04
    mecha()
    poiseuille_kx = float(ds.get("Kx")[0])
    ds.set("measured_axial_conductivity", measured)
    try:
        mecha()
        kx = float(ds.get("Kx")[0])
    finally:
        ds.set("measured_axial_conductivity", np.full(ds.n_nodes(), np.nan))
    mecha_index = g.property("mecha_index")
    run = case["legacy"]
    node_ids = ds.entity_ids("node")
    elements = [_Element(id=str(int(mecha_index[int(node_ids[nodes[c]])]) - run.network.n_wall_junction), value=str(k))
                for c, k in zip(vessels, conductance)]
    hydraulic = run.mecha.hydraulic
    saved = hydraulic.axial_conductance_source, hydraulic.k_xyl_elems
    hydraulic.axial_conductance_source, hydraulic.k_xyl_elems = 2, elements
    try:
        _, reference = run.mecha.calculate_axial_conductance(stage)
    finally:
        hydraulic.axial_conductance_source, hydraulic.k_xyl_elems = saved
    assert kx == pytest.approx(2 * poiseuille_kx, rel=1e-12)
    assert kx == pytest.approx(float(np.asarray(reference).reshape(-1)[0]), rel=1e-15)
