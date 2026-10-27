"""
The legacy's plasmodesmata rules (plasmodesmata_rules 2) are develop's fixed factor map (doc/legacy.rst, d2): with
every factor its own value, the port's plasmodesmata conductances are the legacy's, edge by edge and bit for bit.
"""
import numpy as np
import pytest

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.geometry import PLASMODESMATA, section_parts

from legacy_run import legacy_options, legacy_section
from plants import template_organ

# {the port's parameter: (the key the legacy's solver reads, the value)}
FACTORS = {"kpl_cortex_factor": ("cortex_factor", 0.5), "kpl_endo_in_factor": ("endo_in_factor", 0.7),
           "kpl_endo_out_factor": ("endo_out_factor", 0.8), "kpl_PPP_factor": ("phloem_pericycle_pole_factor", 0.6),
           "kpl_PCC_factor": ("phloem_companion_cell_factor", 0.3), "kpl_PST_factor": ("phloem_sieve_tube_factor", 0.4)}


@pytest.mark.parametrize("barrier", (1,))
def test_each_factor_where_develop_applies_it(case, barrier):
    ds, g = case["ds"], case["g"]
    run = legacy_section(template_organ(case["anatomy"], g, case["segment"]), barriers=(barrier,),
                         height=case["height"], kpl={key: value for key, value in FACTORS.values()})
    if not ds.has("barrier"):
        ds.register("barrier", np.ones(ds.n_nodes()), location="node")
    ds.set("barrier", np.full(ds.n_nodes(), float(barrier)))
    mecha = legacy_options(MechaHydraulics(data_structure=ds))
    defaults = {name: np.array(getattr(mecha, name), copy=True) for name in FACTORS}
    try:                                     # the parameters are the shared case's: restored for the other tests
        for name, (_, value) in FACTORS.items():
            setattr(mecha, name, value)
        mecha()
        K, net = np.asarray(ds.get("conductance")).copy(), mecha.network()
    finally:
        for name, value in defaults.items():
            setattr(mecha, name, value)
    nodes, edges = section_parts(ds)[case["segment"]]
    mecha_index = g.property("mecha_index")
    legacy = {int(v): int(mecha_index[int(v)]) for v in ds.entity_ids("node")[nodes] if int(v) in mecha_index}
    ids = np.asarray(ds.entity_ids("node"))
    expected = {tuple(sorted(k)): v for k, v in run.stages[0]["edge_K"].items()
                if run.network.graph.edges[k]["path"] == "plasmodesmata"}
    plasmodesmata = edges[net.path[edges] == PLASMODESMATA]
    keys = [tuple(sorted((legacy[int(ids[net.a[e]])], legacy[int(ids[net.b[e]])]))) for e in plasmodesmata]
    assert len(keys) == len(expected)
    np.testing.assert_array_equal(K[plasmodesmata], [expected[k] for k in keys])
    graph = run.network.graph
    interfaces = {tuple(sorted((graph.nodes[a]["cgroup"], graph.nodes[b]["cgroup"]))) for a, b in keys}
    assert {(4, 4), (3, 4), (3, 16)} <= interfaces                  # the factors were exercised
