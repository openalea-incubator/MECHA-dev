"""
The Casparian bands' reflection coefficient (PR #67's ``sigma_casp_endo``, ``sigma_casp_exo``; the port's
sigma_casparian_endodermis and sigma_casparian_exodermis, doc/legacy.rst, d21) against the legacy's osmotic scenario 1
(``water_flux``): σ on the wall edges of the walls between two endodermal cells (and two exodermal cells, at a
barrier with an exodermal band), the flow K (Δψ + σ Δπ) along them with a junction's π the mean of its walls', the
solutes carried × (1 − σ) there (one Scharfetter–Gummel step on that scenario's flows); and the steady sucrose coupled
with the water (MechaSolutes' solute_solve 2) against the legacy's coupled solver with a band's σ.
"""
import numpy as np
import pytest

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.geometry import WALL_PATH, section_parts
from openalea.mecha.osmotic import MechaOsmoticPresets
from openalea.mecha.solutes import MechaSolutes

from legacy_run import LegacyRun, legacy_coupled, legacy_options, legacy_osmotic, legacy_standard_solute_flux
from plants import template_organ

# (barrier, the endodermal band's σ, the exodermal band's, leaky bands): the endodermis' band alone, with the default
# conductivity (1e-16: a tight band, whose flows are negligible) and with leaky bands (LEAKY_BANDS: the flows along
# them and the solutes they carry count); both bands (barrier 8: the exodermis' Casparian band and suberin, the
# endodermis' too)
CASES = [(3, 0.5, 0.0, False), (3, 0.5, 0.0, True), (8, 0.5, 0.3, True)]
LEAKY_BANDS = dict(kw_casparian=2.4E-5, kw_casparian_exodermis=4.8E-5)           # kw / 10, kw / 5
LEGACY_LEAKY_BANDS = dict(kw_barrier_elems=[{"value": 1.0E-16, "Casp": 2.4E-5, "Sub": 1.0E-16, "Sub_in": 1.0E-16,
                                             "Sub_out": 1.0E-16, "Lig": 1.0E-16, "Casp_exo": 4.8E-5}])
SCENARIO = dict(os_hetero=0, s_hetero=0, s_factor=1.0)
PRESETS = dict(SCENARIO, presets_rules=2.)               # the legacy's presets (its problems p and q)
D_APO, D_PD, D_MEM, DT = 0.1, 0.5, 1e-6, 1. / 24.        # cm2 d-1, d
PULSE = 50e-6                                            # mol cm-3, in the cortex cells and the left half's walls
SOLUTES = dict(d_apoplast=D_APO, d_plasmodesmata=D_PD, d_membrane=D_MEM, solute_flows=2., solute_boundary_advection=0.)
SOLUTES_RESET = dict(solute_flows=1., solute_boundary_advection=1., solute_solve=1.)
# the coupled steady state: as test_coupled_steady.py, with the endodermal band's σ
COUPLED_SIGMA, PRODUCTION = 0.5, 2e-13                   # the membranes' σ; mol d-1 per cortex cell


def _legacy_index(case, mecha, nodes, network):
    """Each of the section's nodes' legacy index, and the section's geometry."""
    data_structure, g = case["data_structure"], case["mtg"]
    geo = mecha.section_geometries()[case["segment"]][2]
    mecha_index = g.property("mecha_index")
    index = np.array([int(mecha_index[int(v)]) for v in data_structure.entity_ids("node")[nodes]])
    index[geo.junction_order] = network.n_walls + np.arange(network.n_junctions)
    return index, geo


def _reset(case, presets, mecha, solutes, parameters, bands):
    """The shared case's parameters and osmotic states back to their defaults (the bands' conductivities to *bands*),
    for the other comparisons."""
    data_structure = case["data_structure"]
    defaults = {name: field.default for name, field in MechaOsmoticPresets.__dataclass_fields__.items()}
    for name in parameters:
        setattr(presets, name, defaults[name])
    for name, value in bands.items():
        setattr(mecha, name, value)
    for name, value in SOLUTES_RESET.items():
        setattr(solutes, name, value)
    mecha.standard_osmotic = 0.
    data_structure.set("barrier", np.ones(data_structure.n_nodes()))
    for name in ("sucrose", "sucrose_source", "osmotic_potential", "sucrose_osmotic_potential"):
        data_structure.set(name, np.zeros(data_structure.n_nodes()))
    data_structure.set("reflection_coefficient", np.zeros(data_structure.n_edges()))
    mecha()


@pytest.fixture(scope="module", params=CASES,
                ids=lambda p: f"barrier{p[0]}-endodermis{p[1]}-exodermis{p[2]}-{'leaky' if p[3] else 'tight'}")
def both(case, request):
    barrier, sigma_endodermis, sigma_exodermis, leaky = request.param
    data_structure, g = case["data_structure"], case["mtg"]
    if not data_structure.has("barrier"):
        data_structure.register("barrier", np.ones(data_structure.n_nodes()), location="node")
    data_structure.set("barrier", np.full(data_structure.n_nodes(), float(barrier)))
    presets = MechaOsmoticPresets(data_structure=data_structure)
    bands = dict(sigma_casparian_endodermis=sigma_endodermis, sigma_casparian_exodermis=sigma_exodermis)
    mecha = legacy_options(MechaHydraulics(data_structure=data_structure))
    mecha.standard_osmotic = 1.
    default_bands = {name: np.array(getattr(mecha, name), copy=True) for name in LEAKY_BANDS}
    solutes = MechaSolutes(data_structure=data_structure, time_step=DT * 86400.)
    for name, value in SOLUTES.items():
        setattr(solutes, name, value)
    nodes, edges = section_parts(data_structure)[case["segment"]]
    names = ("reflection_coefficient", "standard_psi", "standard_flow", "conductance", "standard_uptake",
             "osmotic_potential", "water_potential")

    def solve(**parameters):
        for name, value in dict(PRESETS, **parameters).items():
            setattr(presets, name, float(value))
        for name in ("sucrose", "sucrose_osmotic_potential"):        # the previous step's −R T c: the presets add it
            data_structure.set(name, np.zeros(data_structure.n_nodes()))
        presets()
        # Newton from 0: the bands' osmotic flows (their walls' K) are below the standard problem's absolute tolerance
        # (1e-8 cm3 d-1), which would keep a start at the solution without σ
        data_structure.set("standard_psi", np.zeros(data_structure.n_nodes()))
        mecha()
        data_structure.set("sucrose", start.copy())
        solutes()
        return dict({name: np.asarray(data_structure.get(name)).copy() for name in names},
                    sucrose=np.asarray(data_structure.get("sucrose"))[nodes].copy())

    legacy = legacy_osmotic(template_organ(case["anatomy"], g, case["segment"]), barrier, height=case["height"],
                            hydraulic=dict(LEGACY_LEAKY_BANDS if leaky else {}, sigma_casp_endo=sigma_endodermis,
                                           sigma_casp_exo=sigma_exodermis),
                            **SCENARIO)
    index, geo = _legacy_index(case, mecha, nodes, legacy["mecha"].network)
    c0 = np.where(np.asarray(data_structure.get("cgroup"))[nodes] == 4, PULSE, 0.)
    c0 = np.where((geo.compartment_type != 0) & (geo.x < geo.x[geo.cells].mean()), PULSE, c0)
    start = np.zeros(data_structure.n_nodes())
    start[nodes] = c0
    try:
        for name, value in (LEAKY_BANDS if leaky else {}).items():
            setattr(mecha, name, value)
        without = solve(sigma_casparian_endodermis=0., sigma_casparian_exodermis=0.)
        port = solve(**bands)
        sink = np.asarray(data_structure.get("sucrose_sink"))[nodes] > 0
    finally:
        _reset(case, presets, mecha, solutes, dict(PRESETS, **bands), default_bands)
    c_prev = np.zeros(index.size)
    c_prev[index] = np.where(sink, 0., c0)
    run = LegacyRun(network=legacy["mecha"].network, mecha=legacy["mecha"])
    reference = np.asarray(legacy_standard_solute_flux(run, c_prev, {int(i): 0.0 for i in index[sink]}, d_apo=D_APO,
                                                       d_pd=D_PD, d_mem=D_MEM, dt=DT, conserve_boundary=False,
                                                       i_scenario=1))[index]
    view = data_structure.to_graph_view()
    local = np.full(data_structure.n_nodes(), -1, dtype=np.int64)
    local[nodes] = np.arange(nodes.size)
    ends = np.c_[index[local[view.tail[edges]]], index[local[view.head[edges]]]]
    return dict(port=port, without=without, legacy=legacy, nodes=nodes, edges=edges, index=index, ends=ends, geo=geo,
                sucrose_reference=reference, data_structure=data_structure, sigmas=(sigma_endodermis, sigma_exodermis),
                leaky=leaky)


def test_bands_reflection_coefficients(both):
    """σ on the legacy's band wall edges exactly, 0 on every other wall edge; both bands where the section has an
    exodermis."""
    data_structure, port, legacy, geo = both["data_structure"], both["port"], both["legacy"], both["geo"]
    path = np.asarray(data_structure.get("network_path"))
    sigma_endodermis, sigma_exodermis = both["sigmas"]
    found = set()
    for edge, (a, b) in zip(both["edges"], both["ends"].tolist()):
        if path[edge] != WALL_PATH:
            continue
        expected = legacy["wall_sigma"].get((a, b), legacy["wall_sigma"].get((b, a), 0.0))
        assert port["reflection_coefficient"][edge] == expected
        found.add(expected)
    assert sum(1 for edge in both["edges"] if path[edge] == WALL_PATH and port["reflection_coefficient"][edge] > 0) \
        == len(legacy["wall_sigma"]) > 0
    exodermis = (geo.cgroup[geo.cells] == 1).any()
    assert found == {0.0, sigma_endodermis} | ({sigma_exodermis} if exodermis and sigma_exodermis else set())


def test_solution(both):
    """The standard problem with the bands' osmotic drive against the legacy's scenario: to 1e-12 of its range; the
    bands' σ changes it."""
    psi = both["port"]["standard_psi"][both["nodes"]]
    reference = both["legacy"]["solution"][both["index"]]
    np.testing.assert_allclose(psi, reference, rtol=0, atol=1e-12 * abs(reference).max())
    assert abs(psi - both["without"]["standard_psi"][both["nodes"]]).max() > 1e-6 * abs(reference).max()


def test_flows_and_uptake(both):
    """The edge flows K (Δψ + σ Δπ), to 1e-12 of the potentials they are differences of (K × max |ψ| + |π|); the
    uptake, to 1e-12."""
    port, legacy = both["port"], both["legacy"]
    potentials = abs(legacy["solution"]).max() + abs(port["osmotic_potential"][both["nodes"]]).max()
    checked = 0
    for edge, (a, b) in zip(both["edges"], both["ends"].tolist()):
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


def test_reported_potentials_unchanged(both):
    """The junctions' mean is the drive's only: the osmotic potentials written on the nodes are those without σ."""
    np.testing.assert_array_equal(both["port"]["osmotic_potential"], both["without"]["osmotic_potential"])


def test_one_solute_step(both):
    """One Scharfetter–Gummel step on the scenario's flows, the solutes carried × (1 − σ) along the bands: every
    node's sucrose the legacy's to 1e-11 of the range."""
    reference = both["sucrose_reference"]
    assert abs(reference).max() > 0
    np.testing.assert_allclose(both["port"]["sucrose"], reference, rtol=0, atol=1e-11 * abs(reference).max())
    if both["leaky"]:                                   # the bands' σ counts: another step than without it
        assert abs(both["port"]["sucrose"] - both["without"]["sucrose"]).max() > 1e-6 * abs(reference).max()


@pytest.fixture(scope="module")
def coupled(case):
    data_structure, g = case["data_structure"], case["mtg"]
    if not data_structure.has("barrier"):
        data_structure.register("barrier", np.ones(data_structure.n_nodes()), location="node")
    data_structure.set("barrier", np.ones(data_structure.n_nodes()))
    presets = MechaOsmoticPresets(data_structure=data_structure)
    parameters = dict(os_hetero=0., s_hetero=0., s_factor=COUPLED_SIGMA, presets_rules=2.,
                      sigma_casparian_endodermis=0.5)
    mecha = legacy_options(MechaHydraulics(data_structure=data_structure))
    mecha.standard_osmotic = 1.
    default_bands = {name: np.array(getattr(mecha, name), copy=True) for name in LEAKY_BANDS}
    solutes = MechaSolutes(data_structure=data_structure)
    for name, value in dict(solute_solve=2., d_apoplast=D_APO, d_plasmodesmata=D_PD, d_membrane=D_MEM,
                            solute_boundary_advection=0.).items():
        setattr(solutes, name, value)
    nodes, _ = section_parts(data_structure)[case["segment"]]
    cortex = nodes[np.asarray(data_structure.get("cgroup"))[nodes] == 4]
    source = np.zeros(data_structure.n_nodes())
    source[cortex] = PRODUCTION
    try:
        for name, value in parameters.items():
            setattr(presets, name, value)
        for name, value in LEAKY_BANDS.items():
            setattr(mecha, name, value)
        data_structure.set("sucrose_source", source)
        presets()
        mecha()
        solutes()
        port = {name: np.asarray(data_structure.get(name))[nodes].copy() for name in ("sucrose", "coupled_psi",
                                                                                      "sucrose_sink")}
    finally:
        _reset(case, presets, mecha, solutes, parameters, default_bands)
    index, geo = _legacy_index(case, mecha, nodes, case["legacy"].network)
    x_cells = geo.x[geo.cells]

    def edit(network):
        network.x_min, network.x_max = float(x_cells.min()), float(x_cells.max())
        network.r_rel = network.x_rel = None

    sigma = {cg: COUPLED_SIGMA for cg in range(40)}
    sigma.update({13: 0.0})                             # the mature vessels' (canonical 13)
    legacy = legacy_coupled(template_organ(case["anatomy"], g, case["segment"]), 1,
                            sources={int(i): PRODUCTION for i in index[np.isin(nodes, cortex)]},
                            sinks={int(i): 0.0 for i in index[port["sucrose_sink"] > 0]}, solute_sigma=sigma,
                            height=case["height"], edit_network=edit, os_hetero=0, s_hetero=0, s_factor=COUPLED_SIGMA,
                            cell_sigma={int(i): 0.0 for i in index[np.isin(np.arange(nodes.size),
                                                                           geo.typed["intercellular"])]},
                            hydraulic=dict(LEGACY_LEAKY_BANDS, sigma_casp_endo=0.5))
    return dict(port=port, legacy=legacy, index=index)


def test_coupled_steady_state(coupled):
    """The steady sucrose coupled with the water, the endodermal band leaky with σ 0.5: the sucrose to 1e-11 of its
    range, the water potentials to 1e-12 of theirs (the legacy's Picard iteration converged)."""
    port, legacy, index = coupled["port"], coupled["legacy"], coupled["index"]
    assert legacy["converged"] and legacy["n_iter"] > 1
    assert any("sigma_wall" in attributes for _, _, attributes in legacy["mecha"].network.graph.edges(data=True))
    sucrose, water = legacy["c"][index], legacy["solution"][index]
    np.testing.assert_allclose(port["sucrose"], sucrose, rtol=0, atol=1e-11 * abs(sucrose).max())
    np.testing.assert_allclose(port["coupled_psi"], water, rtol=0, atol=1e-12 * abs(water).max())
