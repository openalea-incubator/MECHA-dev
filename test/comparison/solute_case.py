"""
The solutes' comparison case (devplan_metafspm.md, §28): GRANAP's ``NeedleAnatomy(seed=42)`` for the port
(MechaHydraulics with the legacy's options and the needle's parameters, MechaSolutes) and for the reactions branch
(legacy_run.legacy_reactions_step), their nodes and edges matched; the port's step on the reactions' flows.
"""
import numpy as np

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.geometry import section_parts
from openalea.mecha.solutes import MechaSolutes

from legacy_run import legacy_options, legacy_reactions_step
from plants import anatomy_data_structure, needle, needle_anatomy_component, template_organ

D_APO, D_PD, D_MEM, DT = 0.1, 0.5, 1e-6, 1. / 24.             # cm2 d-1, d (the simulation's 3600 s)
NEEDLE = dict(kw=2.5634e-3, kw_meso_hypo=1e-16, kmb=3.6314e-4, x_contact=1e10, k_xyl=1.0, collar_condition=0.)
DIFFUSIVITIES = dict(d_apo=D_APO, d_pd=D_PD, d_mem=D_MEM, dt=DT)


def needle_case():
    """
    dict(g, ds, organ, mecha, solutes, nodes, edges (the section's, DataStructure indices), index (each node's
    legacy index), ends (each edge's legacy ends), reference (the reactions' first step, from its mesophyll pulse),
    liquid (the section's liquid nodes), xy (the port's node positions), legacy_xy (by legacy index)).
    """
    g, ds, anatomy = anatomy_data_structure(needle(), needle_anatomy_component())
    segment = int(ds.entity_ids("SubOrgan")[0])
    organ = template_organ(anatomy, g, segment)
    ds.register("barrier", np.ones(ds.n_nodes()), location="node")
    mecha = legacy_options(MechaHydraulics(data_structure=ds))
    for name, value in NEEDLE.items():
        setattr(mecha, name, value)
    mecha()
    nodes, edges = section_parts(ds)[segment]
    geo = mecha.section_geometries()[segment][2]
    reference = legacy_reactions_step(organ, **DIFFUSIVITIES)
    net = reference["network"]
    mecha_index = g.property("mecha_index")
    index = np.array([int(mecha_index[int(v)]) for v in ds.entity_ids("node")[nodes]])
    index[geo.junction_order] = net.n_walls + np.arange(net.n_junctions)
    local = np.full(ds.n_nodes(), -1, dtype=np.int64)
    local[nodes] = np.arange(nodes.size)
    view = ds.to_graph_view()
    ends = np.c_[index[local[view.tail[edges]]], index[local[view.head[edges]]]]
    solutes = MechaSolutes(data_structure=ds, time_step=DT * 86400.)
    for name, value in dict(d_apoplast=D_APO, d_plasmodesmata=D_PD, d_membrane=D_MEM).items():
        setattr(solutes, name, value)
    legacy = reference["reference"]["mecha"]
    graph = net.graph
    legacy_xy = np.zeros((graph.number_of_nodes(), 2))
    for node, data in graph.nodes(data=True):
        legacy_xy[legacy.indice[node]] = data["position"]
    return dict(g=g, ds=ds, organ=organ, mecha=mecha, solutes=solutes, nodes=nodes, edges=edges, index=index,
                ends=ends, reference=reference, liquid=np.asarray(ds.get("vapour_node"))[nodes] == 0,
                xy=np.c_[geo.x, geo.y], legacy_xy=legacy_xy)


def port_step(case, reference, starch=(0., 0.), source=None, sigma=None, c_start=None, starch_start=None):
    """
    The port's step on the reactions' flows (*reference*: a legacy_reactions_step's), from its initial
    concentrations (or *c_start*, *starch_start*: by legacy index), σ 0 (or *sigma* per edge), the rates *starch*
    (d⁻¹), the source *source* (mol d⁻¹, per DataStructure node). Returns the section's sucrose and starch.
    """
    ds, nodes, edges = case["ds"], case["nodes"], case["edges"]
    flows, Q = reference["flows"], np.zeros(ds.n_edges())
    for e, (a, b) in zip(edges, case["ends"]):
        Q[e] = flows.get((a, b), -flows.get((b, a), 0.))
    ds.set("flow", Q)
    ds.set("reflection_coefficient", np.zeros(ds.n_edges()) if sigma is None else sigma)
    for name, start in (("sucrose", reference["c0"] if c_start is None else c_start), ("starch", starch_start)):
        values = np.zeros(ds.n_nodes())
        if start is not None:
            values[nodes] = np.asarray(start)[case["index"]]
        ds.set(name, values)
    ds.set("sucrose_source", np.zeros(ds.n_nodes()) if source is None else source)
    case["solutes"].k_starch_synthesis, case["solutes"].k_starch_degradation = starch
    case["solutes"]()
    return np.asarray(ds.get("sucrose"))[nodes].copy(), np.asarray(ds.get("starch"))[nodes].copy()


def mesophyll_source(case, reference, increment):
    """The port's source (mol d⁻¹ per DataStructure node) equivalent to the reactions' photosynthetic increment
    *increment* (mol cm⁻³) in its mesophyll cells before the step: V increment / dt."""
    by_legacy = dict(zip(case["index"].tolist(), case["nodes"].tolist()))
    source = np.zeros(case["ds"].n_nodes())
    volume = np.asarray(case["ds"].get("node_volume"))
    for i in reference["kinds"]["meso"]:
        source[by_legacy[i]] = volume[by_legacy[i]] * increment / DT
    return source
