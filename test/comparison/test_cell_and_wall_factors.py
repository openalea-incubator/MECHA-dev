"""
PR #67's per-cell aquaporin factors and per-wall conductivity factors, as the port's node inputs aquaporin_factor
(here by tissue, as the legacy's aqp_factor_by_type) and wall_conductivity_factor (every third wall, as the legacy's
kw_wall_factor by wall id): every edge conductance the legacy's, bit for bit, at barriers 1 and 3 (doc/legacy.rst,
d18).
"""
import numpy as np
import pytest

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.geometry import WALL, section_parts

from conftest import BARRIERS
from legacy_run import legacy_options, legacy_section
from plants import template_organ

AQUAPORINS = {"cortex": 0.5, "endodermis": 2.0, "stele": 0.7, "exodermis": 1.3}
WALL_FACTOR = 0.3


@pytest.fixture(scope="module")
def compared(case):
    data_structure, g = case["data_structure"], case["mtg"]
    if not data_structure.has("barrier"):
        data_structure.register("barrier", np.ones(data_structure.n_nodes()), location="node")
    mecha = legacy_options(MechaHydraulics(data_structure=data_structure))
    nodes, edges = section_parts(data_structure)[case["segment"]]
    mecha_index = g.property("mecha_index")
    node_ids = np.asarray(data_structure.entity_ids("node"))
    tissue = np.asarray(data_structure.get("tissue"), dtype=object)
    aquaporins = np.array([AQUAPORINS.get(t, 1.) for t in tissue])
    walls = nodes[np.asarray(data_structure.get("compartment_type"))[nodes] == WALL][::3]
    wall_factor = np.ones(data_structure.n_nodes())
    wall_factor[walls] = WALL_FACTOR
    run = legacy_section(template_organ(case["anatomy"], g, case["segment"]), barriers=BARRIERS,
                         height=case["height"],
                         hydraulic=dict(aqp_factor_by_type=AQUAPORINS,
                                        kw_wall_factor={int(mecha_index[int(node_ids[w])]): WALL_FACTOR
                                                        for w in walls}))
    legacy = np.array([int(mecha_index[int(v)]) for v in node_ids[nodes]])
    geo = mecha.section_geometries()[case["segment"]][2]
    by_position = {tuple(run.network.graph.nodes[j]["position"]): j
                   for j in range(run.network.n_walls, run.network.n_wall_junction)}
    for v in geo.junctions:
        legacy[v] = by_position[(geo.x[v], geo.y[v])]
    stages = {}
    try:                                     # the inputs are the shared case's: restored for the other tests
        data_structure.set("aquaporin_factor", aquaporins)
        data_structure.set("wall_conductivity_factor", wall_factor)
        for i, barrier in enumerate(BARRIERS):
            data_structure.set("barrier", np.full(data_structure.n_nodes(), float(barrier)))
            mecha()
            stages[barrier] = dict(K=np.asarray(data_structure.get("conductance")).copy(), net=mecha.network(),
                                   legacy=run.stages[i])
    finally:
        data_structure.set("aquaporin_factor", np.ones(data_structure.n_nodes()))
        data_structure.set("wall_conductivity_factor", np.ones(data_structure.n_nodes()))
        data_structure.set("barrier", np.ones(data_structure.n_nodes()))
    return dict(nodes=nodes, edges=edges, legacy=legacy, stages=stages)


@pytest.mark.parametrize("barrier", BARRIERS)
def test_edge_conductances(compared, barrier):
    stage, legacy, edges = compared["stages"][barrier], compared["legacy"], compared["edges"]
    net = stage["net"]
    assert (net.aquaporin_factor != 1.).any() and (net.wall_conductivity_factor != 1.).any()
    local = np.full(net.cgroup.size, -1, dtype=np.int64)
    local[compared["nodes"]] = np.arange(compared["nodes"].size)
    expected = {tuple(sorted(k)): v for k, v in stage["legacy"]["edge_K"].items()}
    key = lambda e: tuple(sorted((int(legacy[local[net.a[e]]]), int(legacy[local[net.b[e]]]))))   # noqa: E731
    in_matrix = [e for e in edges if key(e) in expected]
    assert len(in_matrix) == len(expected)
    np.testing.assert_array_equal([stage["K"][e] for e in in_matrix], [expected[key(e)] for e in in_matrix])
