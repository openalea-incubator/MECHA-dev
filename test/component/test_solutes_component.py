"""
The solutes: MechaSolutes on a root of three sections, on MechaHydraulics' flows with
MECHA's osmotic presets: the Scharfetter–Gummel weights, the sinks, every node's balance and the mass, starch in the
cells only, the sucrose's osmotic potential on the presets' baseline.
"""
import numpy as np
import pytest

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.geometry import MEMBRANE
from openalea.mecha.osmotic import MechaOsmoticPresets
from openalea.mecha.solutes import GAS_CONSTANT, MechaSolutes, edge_weights

from plants import anatomy_data_structure, root

SOIL, DT = -1000., 3600.                               # hPa, s
SOURCE = 1e-12                                         # mol d-1 per cortex cell
STARCH = (2.0, 1.0)                                    # d-1


@pytest.fixture(scope="module")
def stepped():
    g, ds, anatomy = anatomy_data_structure(root(3, length=5e-3))
    owner = ds.owner("SubOrgan")
    ds.register("barrier", np.array([1., 3., 1.])[owner], location="node")
    ds.register("soil_psi", np.full(ds.n_nodes(), SOIL), location="node")
    presets = MechaOsmoticPresets(data_structure=ds)
    presets()
    baseline = np.asarray(ds.get("osmotic_potential")).copy()
    mecha = MechaHydraulics(data_structure=ds)
    mecha()
    solutes = MechaSolutes(data_structure=ds, time_step=DT)
    solutes.k_starch_synthesis, solutes.k_starch_degradation = STARCH
    cortex = (np.asarray(ds.get("kind")) == 0) & (np.asarray(ds.get("cgroup")) == 4)
    ds.set("sucrose_source", np.where(cortex, SOURCE, 0.))
    previous = np.asarray(ds.get("sucrose")).copy(), np.asarray(ds.get("starch")).copy()
    solutes()
    return dict(ds=ds, presets=presets, solutes=solutes, baseline=baseline, previous=previous, cortex=cortex)


def test_edge_weights():
    """Pure diffusion: a = b = D_e; pure advection: upwind; the flow a − b = F at uniform c whatever P."""
    a, b = edge_weights([2., 0., 0., 1., 1.], [0., 3., -3., 50., -800.])
    np.testing.assert_array_equal(a[:3], [2., 3., 0.])
    np.testing.assert_array_equal(b[:3], [2., 0., 3.])
    np.testing.assert_allclose(a[3:] - b[3:], [50., -800.], rtol=1e-12)
    assert (a >= 0).all() and (b >= 0).all()


def test_liquid_edges_only(stepped):
    """No weight on the air edges; on the membranes, the flow carried is (1 − σ) Q."""
    ds = stepped["ds"]
    view = ds.to_graph_view()
    liquid = np.asarray(ds.get("liquid_node")) > 0
    a, b = np.asarray(ds.get("tail_weight")), np.asarray(ds.get("head_weight"))
    off = ~(liquid[view.tail] & liquid[view.head])
    assert ((a[off] == 0) & (b[off] == 0)).all()
    membrane = np.asarray(ds.get("network_path")) == MEMBRANE
    sigma, Q = np.asarray(ds.get("reflection_coefficient")), np.asarray(ds.get("flow"))
    np.testing.assert_allclose((a - b)[membrane], ((1. - sigma) * Q)[membrane],
                               atol=1e-12 * np.abs(Q).max())


def test_sinks_are_the_phloem_and_the_xylem(stepped):
    """The root has no Strasburger cells: its phloem (cgroup 11) and xylem (13) cells hold c_sink."""
    ds = stepped["ds"]
    sink = np.asarray(ds.get("sucrose_sink")) > 0
    kind, cgroup = np.asarray(ds.get("kind")), np.asarray(ds.get("cgroup"))
    liquid = np.asarray(ds.get("liquid_node")) > 0
    np.testing.assert_array_equal(sink, (kind == 0) & np.isin(cgroup, (11, 13)) & liquid)
    assert sink.any()
    np.testing.assert_array_equal(np.asarray(ds.get("sucrose"))[sink], 0.)


def test_balances_and_mass(stepped):
    """Every free node's balance V Δ(c + s) / dt + B q = source; the mass gained is the sources' less what the sinks
    took; the sources reach the sinks (positive concentrations in between)."""
    ds = stepped["ds"]
    B = ds.to_graph_view().incidence
    c, s = np.asarray(ds.get("sucrose")), np.asarray(ds.get("starch"))
    c0, s0 = stepped["previous"]
    V, source = np.asarray(ds.get("node_volume")), np.asarray(ds.get("sucrose_source"))
    q = (np.asarray(ds.get("tail_weight")) * np.asarray(B.maximum(0).T @ c).ravel()
         - np.asarray(ds.get("head_weight")) * np.asarray((-B).maximum(0).T @ c).ravel())
    out = np.asarray(B @ q).ravel()
    dt = DT / 86400.
    free = (np.asarray(ds.get("liquid_node")) > 0) & (np.asarray(ds.get("sucrose_sink")) == 0)
    gained = V * ((c - c0) + (s - s0)) / dt
    np.testing.assert_allclose((gained + out - source)[free], 0., atol=1e-9 * SOURCE)
    sunk = -out[~free & (np.asarray(ds.get("liquid_node")) > 0)].sum()
    assert sunk > 0
    assert gained[free].sum() == pytest.approx(source.sum() - sunk, rel=1e-8)
    assert (c[stepped["cortex"]] > 0).all()


def test_starch_in_the_cells_only(stepped):
    ds = stepped["ds"]
    s, kind = np.asarray(ds.get("starch")), np.asarray(ds.get("kind"))
    assert (s[kind != 0] == 0).all() and (s[stepped["cortex"]] > 0).all()


def test_osmotic_potential_on_the_presets_baseline(stepped):
    """The presets' π is their baseline plus the sucrose's −R T c on the cells; the walls keep the baseline."""
    ds = stepped["ds"]
    c, kind = np.asarray(ds.get("sucrose")), np.asarray(ds.get("kind"))
    cells = (kind == 0) & (np.asarray(ds.get("liquid_node")) > 0)
    expected = np.where(cells, -GAS_CONSTANT * stepped["solutes"].solute_temperature * c, 0.)
    np.testing.assert_allclose(ds.get("sucrose_osmotic_potential"), expected, rtol=1e-12, atol=0)
    stepped["presets"]()
    np.testing.assert_allclose(ds.get("osmotic_potential"), stepped["baseline"] + expected, rtol=1e-12, atol=0)
    assert (np.asarray(ds.get("osmotic_potential"))[cells] < stepped["baseline"][cells]).any()
