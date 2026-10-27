"""
The solutes on a root section's standard problem (MechaSolutes with solute_flows 2) against the legacy's
``Mecha.standard_solute_flux`` (develop's SoluteTransport on its scenario 0's edge flows): one step from a pulse in
the cortex and the left half's walls, the sinks (phloem, xylem) at 0; in the apoplast or the symplast only
(solute_pathway, its modes 'apo' and 'sym'); upwind (solute_scheme 2) and Crank–Nicolson (solute_theta 0.5); the
steady state on the standard flows with held sources (solute_solve 3, fixed_sucrose, solute_sinks 0: develop's
tutorial 04, its epidermis at 1 and a first-order loss); its operators 'D' and 'A' alone (diffusion only, advection
only), which the port gives with no water flow or no diffusivities.
Develop's Scharfetter–Gummel operator has no liquid boundary (its ``conserve_boundary`` enters only the upwind
operator, doc/legacy.rst d9): solute_boundary_advection 0 there.
"""
import numpy as np
import pytest

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.geometry import section_parts
from openalea.mecha.solutes import MechaSolutes

from legacy_run import legacy_options, legacy_section, legacy_standard_solute_flux
from plants import template_organ

D_APO, D_PD, D_MEM, DT = 0.1, 0.5, 1e-6, 1. / 24.                     # cm2 d-1, d
PULSE = 50e-6                                                         # mol cm-3, in the cortex cells
SOLUTES = dict(d_apoplast=D_APO, d_plasmodesmata=D_PD, d_membrane=D_MEM, solute_flows=2.)
RESET = dict(solute_flows=1., solute_boundary_advection=1., solute_pathway=1., solute_scheme=1., solute_theta=1.,
             solute_solve=1., solute_sinks=1., k_decay=0., d_apoplast=D_APO, d_plasmodesmata=D_PD, d_membrane=D_MEM)


@pytest.fixture(scope="module")
def root_case(case):
    ds, g = case["ds"], case["g"]
    if not ds.has("barrier"):
        ds.register("barrier", np.ones(ds.n_nodes()), location="node")
    ds.set("barrier", np.ones(ds.n_nodes()))
    mecha = legacy_options(MechaHydraulics(data_structure=ds))
    mecha()
    solutes = MechaSolutes(data_structure=ds, time_step=DT * 86400.)
    for name, value in SOLUTES.items():
        setattr(solutes, name, value)
    nodes, _ = section_parts(ds)[case["segment"]]
    geo = mecha.section_geometries()[case["segment"]][2]
    run = legacy_section(template_organ(case["anatomy"], g, case["segment"]), barriers=(1,), height=case["height"])
    net = run.network
    mecha_index = g.property("mecha_index")
    index = np.array([int(mecha_index[int(v)]) for v in ds.entity_ids("node")[nodes]])
    index[geo.junction_order] = net.n_walls + np.arange(net.n_junctions)
    c0 = np.where(np.asarray(ds.get("cgroup"))[nodes] == 4, PULSE, 0.)
    # and in the walls of the left half (for the apoplast alone)
    c0 = np.where((geo.kind != 0) & (geo.x < geo.x[geo.cells].mean()), PULSE, c0)
    yield dict(ds=ds, solutes=solutes, nodes=nodes, index=index, run=run, c0=c0, kind=geo.kind, geo=geo,
               mecha=mecha)
    for name, value in RESET.items():
        setattr(solutes, name, value)
    ds.set("sucrose", np.zeros(ds.n_nodes()))
    ds.set("fixed_sucrose", np.full(ds.n_nodes(), np.nan))


def port_step(root_case, **parameters):
    """The port's step from the pulse (*parameters*: MechaSolutes'); the section's sucrose, its sinks."""
    ds, nodes, solutes = root_case["ds"], root_case["nodes"], root_case["solutes"]
    for name, value in parameters.items():
        setattr(solutes, name, value)
    values = np.zeros(ds.n_nodes())
    values[nodes] = root_case["c0"]
    ds.set("sucrose", values)
    solutes()
    sink = np.asarray(ds.get("sucrose_sink"))[nodes] > 0
    return np.asarray(ds.get("sucrose"))[nodes].copy(), sink


def legacy_step(root_case, sink, mode="full", **options):
    """The legacy's step from the same pulse, the port's sinks at 0; by the port's nodes (NaN outside *mode*'s)."""
    index = root_case["index"]
    c_prev = np.zeros(index.size)
    c_prev[index] = np.where(sink, 0., root_case["c0"])
    bc = {int(i): 0.0 for i in index[sink]}
    c1 = np.asarray(legacy_standard_solute_flux(root_case["run"], c_prev, bc, d_apo=D_APO, d_pd=D_PD, d_mem=D_MEM,
                                                dt=DT, mode=mode, **options))
    nwj = root_case["run"].network.n_wall_junction
    full = np.full(index.size, np.nan)
    if mode == "sym":
        full[nwj:] = c1
    else:
        full[:c1.size] = c1
    return full[index]


def test_standard_solute_flux(root_case):
    """One Scharfetter–Gummel step on the standard flows: every node's sucrose to 1e-11 of the range."""
    sucrose, sink = port_step(root_case, solute_boundary_advection=0.)
    reference = legacy_step(root_case, sink, conserve_boundary=False)
    assert sink.sum() > 0 and abs(sucrose - root_case["c0"]).max() > 1e-3 * PULSE
    np.testing.assert_allclose(sucrose, reference, rtol=0, atol=1e-11 * abs(reference).max())


@pytest.mark.parametrize("mode, pathway", [("apo", 2.), ("sym", 3.)])
def test_pathways(root_case, mode, pathway):
    """In the apoplast (walls and junctions) or the symplast (cells) only: their nodes' sucrose to 1e-11 of the
    range, the others' unchanged."""
    sucrose, sink = port_step(root_case, solute_boundary_advection=0., solute_pathway=pathway)
    reference = legacy_step(root_case, sink, mode=mode, conserve_boundary=False)
    inside = np.isfinite(reference)
    assert np.array_equal(inside, (root_case["kind"] == 0) == (mode == "sym"))
    assert abs(sucrose[inside] - root_case["c0"][inside]).max() > 0
    np.testing.assert_allclose(sucrose[inside], reference[inside], rtol=0, atol=1e-11 * abs(reference[inside]).max())
    np.testing.assert_array_equal(sucrose[~inside], root_case["c0"][~inside])


@pytest.mark.parametrize("scheme, theta", [("upwind", 1.0), ("upwind", 0.5), ("sg", 0.5)])
def test_schemes_and_theta(root_case, scheme, theta):
    """Upwind (with develop's liquid boundary, which its upwind operator has, d9) and Crank–Nicolson: every node's
    sucrose to 1e-11 of the range."""
    advection = 1. if scheme == "upwind" else 0.
    sucrose, sink = port_step(root_case, solute_boundary_advection=advection, solute_pathway=1.,
                              solute_scheme=2. if scheme == "upwind" else 1., solute_theta=theta)
    reference = legacy_step(root_case, sink, conserve_boundary=bool(advection), scheme=scheme, theta=theta)
    np.testing.assert_allclose(sucrose, reference, rtol=0, atol=1e-11 * abs(reference).max())


@pytest.mark.parametrize("k_decay", [0., 48.])
def test_steady_state_with_held_sources(root_case, k_decay):
    """develop's tutorial 04 (its part 2): the epidermis held at 1, no sinks, the steady state on the standard flows
    (upwind, its liquid boundary: the clean soil water dilutes, the xylem's carries the solute out), without and with
    a first-order loss k_deg (48 d-1: MECHA's hormone default): every node to 1e-11."""
    from openalea.mecha.legacy.utils.solute_transport import SoluteTransport
    ds, nodes, geo = root_case["ds"], root_case["nodes"], root_case["geo"]
    epidermis = geo.cells[geo.cgroup[geo.cells] == 2]
    fixed = np.full(ds.n_nodes(), np.nan)
    fixed[nodes[epidermis]] = 1.0
    ds.set("fixed_sucrose", fixed)
    try:
        sucrose, held = port_step(root_case, solute_solve=3., solute_sinks=0., solute_scheme=2., k_decay=k_decay,
                                  solute_boundary_advection=1., solute_pathway=1.)
    finally:
        ds.set("fixed_sucrose", np.full(ds.n_nodes(), np.nan))
        for name in ("solute_solve", "solute_sinks", "solute_scheme", "k_decay"):
            setattr(root_case["solutes"], name, RESET[name])
    index = root_case["index"]
    assert np.array_equal(np.flatnonzero(held), epidermis)
    run = root_case["run"]
    transport = SoluteTransport(run.mecha, {"apo_wall": D_APO, "plasmodesmata": D_PD, "membrane": D_MEM},
                                capacitance_params=None, mode="full", reaction_params={"k_deg": k_decay})
    reference = transport.solve(h=0, i_maturity=0, i_scenario=0, rhs=np.zeros(index.size),
                                boundary_conditions={int(i): 1.0 for i in index[epidermis]}, scheme="upwind")[index]
    assert reference.min() < 0.5
    np.testing.assert_allclose(sucrose, reference, rtol=0, atol=1e-11)


@pytest.mark.parametrize("operators", ["D", "A"])
def test_operators_alone(root_case, operators):
    """The legacy's diffusion-only ('D') and advection-only ('A') steps (upwind): the port's step with the standard
    flows at 0 (the soil and the xylem at one potential) or the diffusivities at 0, to 1e-11 and 5e-5 of the range. They are
    diagnostics of the operator: the port gives them with its physical parameters, so it has no option for them."""
    mecha, solutes = root_case["mecha"], root_case["solutes"]
    if operators == "D":
        mecha.standard_psi_xylem = mecha.standard_psi_soil = 0.
        try:
            mecha()
            assert abs(np.asarray(root_case["ds"].get("standard_flow"))).max() < 1e-15
            sucrose, sink = port_step(root_case, solute_scheme=2., solute_boundary_advection=1., solute_theta=1.,
                                      solute_pathway=1.)
        finally:
            mecha.standard_psi_xylem, mecha.standard_psi_soil = -5.0E3, 0.
            mecha()
    else:
        sucrose, sink = port_step(root_case, solute_scheme=2., solute_boundary_advection=1., solute_theta=1.,
                                  solute_pathway=1., d_apoplast=0., d_plasmodesmata=0., d_membrane=0.)
    for name in ("solute_scheme", "d_apoplast", "d_plasmodesmata", "d_membrane"):
        setattr(solutes, name, RESET[name])
    reference = legacy_step(root_case, sink, scheme="upwind", operators=operators, conserve_boundary=True)
    # advection alone: the concentrations follow the flows' ratios, and the two solves' flows agree to 1e-12 of K x
    # the potentials' range, 1e-5 of the flow on the edges of nearly equal potentials (no diffusion damps it)
    tolerance = 1e-11 if operators == "D" else 5e-5
    np.testing.assert_allclose(sucrose, reference, rtol=0, atol=tolerance * abs(reference).max())
