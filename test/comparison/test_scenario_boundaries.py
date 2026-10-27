"""
The boundary conditions of the legacy's scenarios >= 1 in the standard problem, against its scenario 1
(``Mecha.water_flux``): the elongation sink at barrier 0 (``elongation_BC``: standard_elongation,
elongation_midpoint_rate, elongation_side_rate_difference), the sieve tubes' pressure at barrier > 0
(standard_psi_sieve_tube) or flow (standard_sieve_flow) and the soil's left and right potentials (standard_psi_soil, standard_psi_soil_right), with
the osmotic terms (standard_osmotic) of its default scenario. The legacy's walls take MECHA's thickness for the
elongation (its HydraulicWall's is 1 µm on GRANAP's path, doc/legacy.rst ME-39) and the port's x range (its cells'
centroids, ME-22), and its k_sieve as a one-element array (its scenario loop indexes it, ME-36). Its phloem flow is
split among the sieve tubes as their areas, the port's (the legacy splits it with the protophloem's areas, none on
these roots: NaN, ME-42).
"""
import numpy as np
import pytest

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.geometry import section_parts
from openalea.mecha.osmotic import MechaOsmoticPresets

from legacy_run import legacy_options, legacy_osmotic
from plants import template_organ

THICKNESS = 1.5                                   # µm: the legacy's geometry thickness (the conductances')
CASES = {
    "elongation": dict(barrier=0, legacy=dict(elongation_midpoint_rate=2.8, elongation_side_rate_difference=0.6),
                       port=dict(standard_elongation=1., elongation_midpoint_rate=2.8,
                                 elongation_side_rate_difference=0.6)),
    "sieve_tubes_and_soil": dict(barrier=1, legacy=dict(pressure_sieve_prox=1.1E4, psi_soil_left=-1.0E3,
                                                        psi_soil_right=-3.0E3),
                                 port=dict(standard_psi_sieve_tube=1.1E4, standard_psi_soil=-1.0E3,
                                           standard_psi_soil_right=-3.0E3)),
    "sieve_flow": dict(barrier=1, legacy=dict(psi_soil_left=-1.0E3, psi_soil_right=-1.0E3), sieve_flow=-2.0E-5,
                       port=dict(standard_sieve_flow=-2.0E-5, standard_psi_soil=-1.0E3)),
}
PRESETS = dict(os_hetero=0., s_hetero=0., s_factor=1.0, presets_rules=2.)
RESET = dict(standard_elongation=0., elongation_midpoint_rate=0., elongation_side_rate_difference=0.,
             standard_psi_sieve_tube=float("nan"), standard_psi_soil=0., standard_psi_soil_right=float("nan"),
             standard_osmotic=0., standard_sieve_flow=float("nan"))


@pytest.fixture(scope="module", params=list(CASES))
def both(case, request):
    spec = CASES[request.param]
    ds, g = case["ds"], case["g"]
    if not ds.has("barrier"):
        ds.register("barrier", np.ones(ds.n_nodes()), location="node")
    ds.set("barrier", np.full(ds.n_nodes(), float(spec["barrier"])))
    presets = MechaOsmoticPresets(data_structure=ds)
    for name, value in PRESETS.items():
        setattr(presets, name, value)
    mecha = legacy_options(MechaHydraulics(data_structure=ds))
    mecha.standard_osmotic = 1.
    for name, value in spec["port"].items():
        setattr(mecha, name, value)
    presets()
    mecha()
    port = {name: np.asarray(ds.get(name)).copy() for name in
            ("standard_psi", "standard_flow", "conductance", "standard_uptake", "soil_conductance",
             "standard_soil_psi", "standard_elongation_sink", "osmotic_potential")}
    geo = mecha.section_geometries()[case["segment"]][2]
    x_cells = geo.x[geo.cells]

    def edit(network):
        for wall in network.cell_manager.walls:
            wall.thickness = THICKNESS
        network.x_min, network.x_max = float(x_cells.min()), float(x_cells.max())
        network.r_rel = network.x_rel = None

    def sieve_flow(legacy_mecha):                 # scenario 1's phloem flow, split as the sieve tubes' areas (ME-42)
        if "sieve_flow" not in spec:
            return
        flow, sieve = spec["sieve_flow"], legacy_mecha.network.cell_manager.sieve
        area = np.array([cell.area for cell in sieve])
        legacy_mecha.psi_sieve[1, :, 1] = np.nan
        legacy_mecha.distributed_flow_sieve[1, 0, 1] = flow
        legacy_mecha.distributed_flow_sieve[1, 1:len(sieve) + 1, 1] = flow * area / area.sum()

    legacy = legacy_osmotic(template_organ(case["anatomy"], g, case["segment"]), spec["barrier"],
                            height=case["height"], scenario_zero=spec["legacy"], edit_network=edit,
                            hydraulic=dict(k_sieve=np.array([1.0])), before_solve=sieve_flow, **spec["legacy"])
    nodes, edges = section_parts(ds)[case["segment"]]
    net = case["legacy"].network
    mecha_index = g.property("mecha_index")
    index = np.array([int(mecha_index[int(v)]) for v in ds.entity_ids("node")[nodes]])
    index[geo.junction_order] = net.n_walls + np.arange(net.n_junctions)
    yield dict(port=port, legacy=legacy, nodes=nodes, edges=edges, index=index, ds=ds, spec=spec)
    for name, value in RESET.items():
        setattr(mecha, name, value)
    ds.set("osmotic_potential", np.zeros(ds.n_nodes()))
    ds.set("reflection_coefficient", np.zeros(ds.n_edges()))


def test_elongation_sink(both):
    """The sink per node against the legacy's rhs_e (``elongation_BC``), to 1e-15."""
    if both["spec"]["barrier"] != 0:
        pytest.skip("barrier > 0: no elongation")
    mecha = both["legacy"]["mecha"]
    rhs_e = np.asarray(mecha.elongation_BC(1, 0)).reshape(-1)[both["index"]]
    sink = both["port"]["standard_elongation_sink"][both["nodes"]]
    assert (sink > 0).sum() > 0
    np.testing.assert_allclose(sink, rhs_e, rtol=1e-15, atol=1e-30)


def test_solution(both):
    """The standard problem against the legacy's scenario 1, to 1e-12 of its range."""
    psi = both["port"]["standard_psi"][both["nodes"]]
    reference = both["legacy"]["solution"][both["index"]]
    np.testing.assert_allclose(psi, reference, rtol=0, atol=1e-12 * abs(reference).max())


def test_flows_and_uptake(both):
    """The edge flows, to 1e-12 of the potentials they are differences of; the uptake: Σ K_soil (ψ_soil − ψ) with each
    border node's own soil potential (the legacy's total takes the left one everywhere, doc/legacy.rst h), to
    1e-12."""
    ds, port, legacy, index = both["ds"], both["port"], both["legacy"], both["index"]
    view = ds.to_graph_view()
    local = np.full(ds.n_nodes(), -1, dtype=np.int64)
    local[both["nodes"]] = np.arange(both["nodes"].size)
    potentials = abs(legacy["solution"]).max() + abs(port["osmotic_potential"][both["nodes"]]).max()
    checked = 0
    for edge in both["edges"]:
        a, b = int(index[local[view.tail[edge]]]), int(index[local[view.head[edge]]])
        if (a, b) in legacy["edge_Q"]:
            reference = legacy["edge_Q"][(a, b)]
        elif (b, a) in legacy["edge_Q"]:
            reference = -legacy["edge_Q"][(b, a)]
        else:
            continue
        assert abs(port["standard_flow"][edge] - reference) <= 1e-12 * port["conductance"][edge] * potentials
        checked += 1
    assert checked > 0
    nodes = both["nodes"]
    soil = port["soil_conductance"][nodes]
    expected = float(np.sum(soil * (port["standard_soil_psi"][nodes] - legacy["solution"][index])))
    assert port["standard_uptake"][0] == pytest.approx(expected, rel=1e-11)
    if both["spec"]["barrier"] == 0:
        assert port["standard_uptake"][0] == pytest.approx(legacy["total_flow"], rel=1e-11)
