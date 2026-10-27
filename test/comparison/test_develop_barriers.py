"""
develop's barrier codes 10 (the exodermis suberized, with a Casparian strip) and 11 (the endodermis and exodermis
suberized, the sclerenchyma's walls lignified): the port's conductances are the legacy's, edge by edge and bit for
bit (doc/legacy.rst, d4; the sections have no sclerenchyma: its walls are tested in test_develop_additions.py).
"""
import numpy as np
import pytest

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.geometry import section_parts

from legacy_run import legacy_options, legacy_section
from plants import template_organ

DEVELOP_BARRIERS = (10, 11)


@pytest.fixture(scope="module")
def compared(case):
    ds, g = case["ds"], case["g"]
    run = legacy_section(template_organ(case["anatomy"], g, case["segment"]), barriers=DEVELOP_BARRIERS,
                         height=case["height"])
    if not ds.has("barrier"):
        ds.register("barrier", np.ones(ds.n_nodes()), location="node")
    mecha = legacy_options(MechaHydraulics(data_structure=ds))
    nodes, edges = section_parts(ds)[case["segment"]]
    net_legacy = run.network
    mecha_index = g.property("mecha_index")
    legacy = np.array([int(mecha_index[int(v)]) for v in ds.entity_ids("node")[nodes]])
    geo = mecha.section_geometries()[case["segment"]][2]
    by_position = {tuple(net_legacy.graph.nodes[j]["position"]): j
                   for j in range(net_legacy.n_walls, net_legacy.n_wall_junction)}
    for v in geo.junctions:
        legacy[v] = by_position[(geo.x[v], geo.y[v])]
    stages = {}
    for i, barrier in enumerate(DEVELOP_BARRIERS):
        ds.set("barrier", np.full(ds.n_nodes(), float(barrier)))
        mecha()
        stages[barrier] = dict(K=np.asarray(ds.get("conductance")).copy(), net=mecha.network(), legacy=run.stages[i])
    ds.set("barrier", np.ones(ds.n_nodes()))
    return dict(nodes=nodes, edges=edges, legacy=legacy, stages=stages)


@pytest.mark.parametrize("barrier", DEVELOP_BARRIERS)
def test_edge_conductances(compared, barrier):
    stage, legacy, edges = compared["stages"][barrier], compared["legacy"], compared["edges"]
    net = stage["net"]
    local = np.full(net.cgroup.size, -1, dtype=np.int64)
    local[compared["nodes"]] = np.arange(compared["nodes"].size)
    expected = {tuple(sorted(k)): v for k, v in stage["legacy"]["edge_K"].items()}
    key = lambda e: tuple(sorted((int(legacy[local[net.a[e]]]), int(legacy[local[net.b[e]]]))))   # noqa: E731
    in_matrix = [e for e in edges if key(e) in expected]
    assert len(in_matrix) == len(expected)
    np.testing.assert_array_equal([stage["K"][e] for e in in_matrix], [expected[key(e)] for e in in_matrix])
