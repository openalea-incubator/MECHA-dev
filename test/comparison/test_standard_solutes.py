"""
The solutes on a root section's standard problem (MechaSolutes with solute_flows 2) against the legacy's
``Mecha.standard_solute_flux`` (develop's SoluteTransport on its scenario 0's edge flows): one step from a pulse in
the cortex and the left half's walls, the sinks (phloem, xylem) at 0; in the apoplast or the symplast only
(solute_pathway, its modes 'apo' and 'sym'); upwind (solute_scheme 2) and Crank–Nicolson (solute_theta 0.5).
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
RESET = dict(solute_flows=1., solute_boundary_advection=1., solute_pathway=1., solute_scheme=1., solute_theta=1.)


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
    yield dict(ds=ds, solutes=solutes, nodes=nodes, index=index, run=run, c0=c0, kind=geo.kind)
    for name, value in RESET.items():
        setattr(solutes, name, value)
    ds.set("sucrose", np.zeros(ds.n_nodes()))


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
