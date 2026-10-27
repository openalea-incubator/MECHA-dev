"""
The solutes' liquid boundary (develop's conserve_boundary, doc/legacy.rst d7): the water each node exchanges with the
boundary, the residual of its edges' flows, is the legacy's (SoluteTransport._boundary_flux) on the same water state
(its default osmotic scenario, the port's standard problem with the membranes' osmotic terms).
"""
import numpy as np

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.geometry import MEMBRANE, PLASMODESMATA, WALL_PATH, section_parts
from openalea.mecha.osmotic import MechaOsmoticPresets
from openalea.mecha.solutes import liquid_boundary_exchange

from legacy_run import legacy_options, legacy_osmotic
from plants import template_organ


def test_boundary_exchange_is_the_legacys(case):
    ds, g = case["ds"], case["g"]
    if not ds.has("barrier"):
        ds.register("barrier", np.ones(ds.n_nodes()), location="node")
    ds.set("barrier", np.ones(ds.n_nodes()))
    presets = MechaOsmoticPresets(data_structure=ds)
    for name, value in dict(os_hetero=0., s_hetero=0., s_factor=1., presets_rules=2.).items():
        setattr(presets, name, value)
    mecha = legacy_options(MechaHydraulics(data_structure=ds))
    mecha.standard_osmotic = 1.
    try:
        presets()
        mecha()
        flow = np.asarray(ds.get("standard_flow")).copy()
    finally:                                 # the case's DataStructure is shared with the other comparisons
        mecha.standard_osmotic = 0.
        ds.set("osmotic_potential", np.zeros(ds.n_nodes()))
        ds.set("reflection_coefficient", np.zeros(ds.n_edges()))
    legacy = legacy_osmotic(template_organ(case["anatomy"], g, case["segment"]), 1, height=case["height"],
                            os_hetero=0, s_hetero=0, s_factor=1.0)
    from openalea.mecha.legacy import SoluteTransport
    reference = SoluteTransport(legacy["mecha"], {"apo_wall": 0.1, "plasmodesmata": 0.5, "membrane": 1e-6},
                                {"dt": 0.1})._boundary_flux(0, 1)
    nodes, edges = section_parts(ds)[case["segment"]]
    view = ds.to_graph_view()
    path = np.asarray(ds.get("network_path"))
    counted = np.zeros(ds.n_edges(), dtype=bool)
    counted[edges] = np.isin(path[edges], (WALL_PATH, MEMBRANE, PLASMODESMATA))
    liquid = np.zeros(ds.n_nodes(), dtype=bool)
    liquid[nodes] = True
    outflow, inflow = liquid_boundary_exchange(view.incidence, flow, counted, liquid)
    geo = mecha.section_geometries()[case["segment"]][2]
    net = case["legacy"].network
    mecha_index = g.property("mecha_index")
    index = np.array([int(mecha_index[int(v)]) for v in ds.entity_ids("node")[nodes]])
    index[geo.junction_order] = net.n_walls + np.arange(net.n_junctions)
    expected = reference[index]                                  # its matrix index is the node id
    scale = np.abs(reference).max()
    np.testing.assert_allclose((inflow - outflow)[nodes], expected, rtol=0, atol=1e-9 * scale)
    assert (outflow[nodes] > 1e-6 * scale).sum() > 0 and (inflow[nodes] > 1e-6 * scale).sum() > 0
