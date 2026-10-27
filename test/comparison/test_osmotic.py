"""
The osmotic potentials: MECHA's presets (MechaOsmoticPresets) and the standard problem
with the membranes' osmotic terms (standard_osmotic) against the legacy's osmotic scenario (``water_flux``): the
cells' and walls' π, the membranes' σ, the solution and the edge flows, for its os_hetero and s_hetero patterns.
"""
import numpy as np
import pytest

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.geometry import MEMBRANE, section_parts
from openalea.mecha.osmotic import MechaOsmoticPresets

from legacy_run import legacy_options, legacy_osmotic
from plants import template_organ

# every case sets all three (the DataStructure keeps a component's parameters from one case to the next)
SCENARIOS = [dict(os_hetero=0, s_hetero=0, s_factor=1.0), dict(os_hetero=1, s_hetero=1, s_factor=1.0),
             dict(os_hetero=2, s_hetero=2, s_factor=1.0), dict(os_hetero=3, s_hetero=0, s_factor=0.7),
             dict(os_hetero=4, s_hetero=1, s_factor=0.9, osmotic_epi=-5100., osmotic_exo=-5300., osmotic_peri=-5600.,
                  osmotic_stele=-6000.),
             # the walls' solutions: the soil's left to right and the xylem's uniform; both radial, with exponents;
             # both radial with a diffusivity (the legacy's convective profile, flat: ME-24)
             dict(os_hetero=0, s_hetero=0, s_factor=1.0, osmotic_left_soil=-300., osmotic_right_soil=-900.,
                  osmotic_symmetry_soil=1, osmotic_xyl=-1800., osmotic_endo=-4800., osmotic_symmetry_xyl=1),
             dict(os_hetero=1, s_hetero=0, s_factor=1.0, osmotic_left_soil=-300., osmotic_right_soil=-900.,
                  osmotic_symmetry_soil=2, osmotic_shape_soil=2., osmotic_symmetry_xyl=2, osmotic_shape_xyl=0.5),
             dict(os_hetero=0, s_hetero=0, s_factor=1.0, osmotic_left_soil=-300., osmotic_symmetry_soil=2,
                  osmotic_diffusivity_soil=1., osmotic_symmetry_xyl=2, osmotic_diffusivity_xyl=1.)]
# the legacy's presets (its problems p and q, fixed in the port's default rules)
PORT_ONLY = dict(presets_rules=2.)


@pytest.fixture(scope="module", params=[(b, i) for b in (1, 3) for i in range(len(SCENARIOS))],
                ids=lambda p: f"barrier{p[0]}-scenario{p[1]}")
def both(case, request):
    barrier, k = request.param
    scenario = SCENARIOS[k]
    ds, g = case["ds"], case["g"]
    if not ds.has("barrier"):
        ds.register("barrier", np.ones(ds.n_nodes()), location="node")
    ds.set("barrier", np.full(ds.n_nodes(), float(barrier)))
    presets = MechaOsmoticPresets(data_structure=ds)
    for name, value in {**scenario, **PORT_ONLY}.items():
        setattr(presets, name, float(value))
    mecha = legacy_options(MechaHydraulics(data_structure=ds))
    mecha.standard_osmotic = 1.
    presets()
    mecha()
    port = {name: np.asarray(ds.get(name)).copy() for name in
            ("osmotic_potential", "reflection_coefficient", "standard_psi", "standard_flow", "conductance",
             "standard_uptake")}
    geo = mecha.section_geometries()[case["segment"]][2]
    x_cells = geo.x[geo.cells]

    def port_x_range(network):                    # the walls' relative x from the port's cell centroids (ME-22)
        network.x_min, network.x_max = float(x_cells.min()), float(x_cells.max())
        network.r_rel = network.x_rel = None

    legacy = legacy_osmotic(template_organ(case["anatomy"], g, case["segment"]), barrier, height=case["height"],
                            edit_network=port_x_range if scenario.get("osmotic_symmetry_soil") == 1 else None,
                            **scenario)
    nodes, edges = section_parts(ds)[case["segment"]]
    net = case["legacy"].network
    mecha_index = g.property("mecha_index")
    index = np.array([int(mecha_index[int(v)]) for v in ds.entity_ids("node")[nodes]])
    index[geo.junction_order] = net.n_walls + np.arange(net.n_junctions)
    yield dict(port=port, legacy=legacy, nodes=nodes, edges=edges, index=index, ds=ds)
    # the case's DataStructure is shared with the other comparisons: no solutes left on it
    mecha.standard_osmotic = 0.
    defaults = {name: field.default for name, field in MechaOsmoticPresets.__dataclass_fields__.items()}
    for name in {**scenario, **PORT_ONLY}:        # a plant's parameters outlive the component: its defaults back
        setattr(presets, name, defaults[name])
    ds.set("osmotic_potential", np.zeros(ds.n_nodes()))
    ds.set("reflection_coefficient", np.zeros(ds.n_edges()))


def test_osmotic_potentials(both):
    """Every node a legacy membrane sets: the same π, to 1e-13 (the soil's left-right profile weighs each wall by its
    relative x, which the port takes from the cells' centroids, the legacy from GRANAP's cell seeds: one ulp, with
    equal sides)."""
    port, legacy, index = both["port"], both["legacy"], both["index"]
    by_legacy = dict(zip(index.tolist(), both["nodes"].tolist()))
    for node, value in legacy["psi_os"].items():
        expected = 0.0 if value is None else value
        assert port["osmotic_potential"][by_legacy[node]] == pytest.approx(expected, rel=1e-13, abs=0), node


def test_reflection_coefficients(both):
    ds, port, legacy, index = both["ds"], both["port"], both["legacy"], both["index"]
    view = ds.to_graph_view()
    path = np.asarray(ds.get("network_path"))
    local = np.full(ds.n_nodes(), -1, dtype=np.int64)
    local[both["nodes"]] = np.arange(both["nodes"].size)
    e = both["edges"][path[both["edges"]] == MEMBRANE]
    u, v = index[local[view.tail[e]]], index[local[view.head[e]]]
    for edge, a, b in zip(e, u, v):
        key = (int(a), int(b)) if (int(a), int(b)) in legacy["sigma"] else (int(b), int(a))
        assert port["reflection_coefficient"][edge] == legacy["sigma"][key]
    assert len(e) == len(legacy["sigma"])


def test_solution(both):
    """The standard problem with the osmotic terms against the legacy's scenario: to 1e-12 of its range (D1)."""
    psi = both["port"]["standard_psi"][both["nodes"]]
    reference = both["legacy"]["solution"][both["index"]]
    np.testing.assert_allclose(psi, reference, rtol=0, atol=1e-12 * abs(reference).max())


def test_flows_and_uptake(both):
    """The edge flows K (Δψ + σ Δπ), to 1e-12 of the potentials they are differences of (K × max |ψ| + |π|); the
    uptake, to 1e-12."""
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
    assert port["standard_uptake"][0] == pytest.approx(legacy["total_flow"], rel=1e-12)
