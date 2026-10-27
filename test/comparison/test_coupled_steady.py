"""
The steady state of the sucrose coupled with the standard problem's water (MechaSolutes' solute_solve 2: one Newton
system) against the legacy's coupled solver (``coupled_water_solute_solve``, Picard, Scharfetter–Gummel) on its
osmotic scenario 1: a root section at barrier 1, sucrose produced in the cortex cells, the phloem and the xylem its
sinks at 0, every membrane's σ 0.5 (the mature vessels' and the intercellular spaces' 0). The legacy's
reactions-branch problems are removed there: its transport's psi_os held at 0 (r1), its solute's σ the water's (r3:
its σ per cgroup, 0 on the vessels' cgroup, and the intercellular spaces' through their own cgroups).
"""
import numpy as np
import pytest

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.geometry import section_parts
from openalea.mecha.osmotic import MechaOsmoticPresets
from openalea.mecha.solutes import MechaSolutes

from legacy_run import legacy_coupled, legacy_options
from plants import template_organ

SIGMA = 0.5
PRODUCTION = 2e-13                                     # mol d-1 per cortex cell
D_APO, D_PD, D_MEM = 0.1, 0.5, 1e-6
PRESETS = dict(os_hetero=0., s_hetero=0., s_factor=SIGMA, presets_rules=2.)
SOLUTES = dict(solute_solve=2., d_apoplast=D_APO, d_plasmodesmata=D_PD, d_membrane=D_MEM,
               solute_boundary_advection=0.)


@pytest.fixture(scope="module")
def both(case):
    ds, g = case["ds"], case["g"]
    if not ds.has("barrier"):
        ds.register("barrier", np.ones(ds.n_nodes()), location="node")
    ds.set("barrier", np.ones(ds.n_nodes()))
    presets = MechaOsmoticPresets(data_structure=ds)
    for name, value in PRESETS.items():
        setattr(presets, name, value)
    mecha = legacy_options(MechaHydraulics(data_structure=ds))
    mecha.standard_osmotic = 1.
    solutes = MechaSolutes(data_structure=ds)
    for name, value in SOLUTES.items():
        setattr(solutes, name, value)
    nodes, _ = section_parts(ds)[case["segment"]]
    cortex = nodes[np.asarray(ds.get("cgroup"))[nodes] == 4]
    source = np.zeros(ds.n_nodes())
    source[cortex] = PRODUCTION
    ds.set("sucrose_source", source)
    presets()
    mecha()
    solutes()
    port = {name: np.asarray(ds.get(name))[nodes].copy() for name in ("sucrose", "coupled_psi", "sucrose_sink",
                                                                      "standard_psi")}
    geo = mecha.section_geometries()[case["segment"]][2]
    net = case["legacy"].network
    mecha_index = g.property("mecha_index")
    index = np.array([int(mecha_index[int(v)]) for v in ds.entity_ids("node")[nodes]])
    index[geo.junction_order] = net.n_walls + np.arange(net.n_junctions)
    x_cells = geo.x[geo.cells]

    def edit(network):
        network.x_min, network.x_max = float(x_cells.min()), float(x_cells.max())
        network.r_rel = network.x_rel = None

    sigma = {cg: SIGMA for cg in range(40)}
    sigma.update({13: 0.0})                             # the mature vessels' (canonical 13)
    legacy = legacy_coupled(template_organ(case["anatomy"], g, case["segment"]), 1,
                            sources={int(i): PRODUCTION for i in index[np.isin(nodes, cortex)]},
                            sinks={int(i): 0.0 for i in index[port["sucrose_sink"] > 0]}, solute_sigma=sigma,
                            height=case["height"], edit_network=edit, os_hetero=0, s_hetero=0, s_factor=SIGMA,
                            cell_sigma={int(i): 0.0 for i in index[np.isin(np.arange(nodes.size),
                                                                           geo.typed["intercellular"])]})
    yield dict(port=port, legacy=legacy, index=index)
    solutes.solute_solve = 1.
    mecha.standard_osmotic = 0.
    for name in ("sucrose", "sucrose_source", "osmotic_potential", "sucrose_osmotic_potential"):
        ds.set(name, np.zeros(ds.n_nodes()))
    ds.set("reflection_coefficient", np.zeros(ds.n_edges()))


def test_converged(both):
    assert both["legacy"]["converged"] and both["legacy"]["n_iter"] > 1


def test_sucrose(both):
    """The steady sucrose to 1e-11 of its range (the legacy's Picard iteration converged to 1e-10 hPa on ψ)."""
    reference = both["legacy"]["c"][both["index"]]
    assert abs(reference).max() > 0
    np.testing.assert_allclose(both["port"]["sucrose"], reference, rtol=0, atol=1e-11 * abs(reference).max())


def test_water(both):
    """The coupled water potentials to 1e-12 of their range; the sucrose's osmotic feedback changes them."""
    reference = both["legacy"]["solution"][both["index"]]
    np.testing.assert_allclose(both["port"]["coupled_psi"], reference, rtol=0, atol=1e-12 * abs(reference).max())
    assert abs(both["port"]["coupled_psi"] - both["port"]["standard_psi"]).max() > 1e-6 * abs(reference).max()
