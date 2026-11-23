"""
Each wall's own thickness (the port's default, thickness_source 2; PR #67's use_wall_thickness, GRANAP's double wall
thickness, a junction the mean of its walls'): every edge conductance and every soil term the legacy's, bit for bit,
at barriers 0, 1 and 3 (doc/legacy.rst, d12).
"""
import numpy as np
import pytest

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.geometry import section_parts

from conftest import BARRIERS as CASE_BARRIERS
from legacy_run import legacy_options, legacy_section
from plants import template_organ

BARRIERS = (0,) + CASE_BARRIERS


@pytest.fixture(scope="module")
def compared(case):
    data_structure, g = case["data_structure"], case["mtg"]
    run = legacy_section(template_organ(case["anatomy"], g, case["segment"]), barriers=BARRIERS,
                         height=case["height"], geometry=dict(use_wall_thickness=True))
    if not data_structure.has("barrier"):
        data_structure.register("barrier", np.ones(data_structure.n_nodes()), location="node")
    mecha = legacy_options(MechaHydraulics(data_structure=data_structure))
    nodes, edges = section_parts(data_structure)[case["segment"]]
    mecha_index = g.property("mecha_index")
    legacy = np.array([int(mecha_index[int(v)]) for v in data_structure.entity_ids("node")[nodes]])
    geo = mecha.section_geometries()[case["segment"]][2]
    by_position = {tuple(run.network.graph.nodes[j]["position"]): j
                   for j in range(run.network.n_walls, run.network.n_wall_junction)}
    for v in geo.junctions:
        legacy[v] = by_position[(geo.x[v], geo.y[v])]
    stages = {}
    thickness_source = mecha.thickness_source
    try:                                     # the parameters are the shared case's: restored for the other tests
        mecha.thickness_source = 2.
        for i, barrier in enumerate(BARRIERS):
            data_structure.set("barrier", np.full(data_structure.n_nodes(), float(barrier)))
            mecha()
            stages[barrier] = dict(K=np.asarray(data_structure.get("conductance")).copy(),
                                   soil=np.asarray(data_structure.get("soil_conductance")).copy(),
                                   net=mecha.network(), legacy=run.stages[i])
    finally:
        mecha.thickness_source = thickness_source
        data_structure.set("barrier", np.ones(data_structure.n_nodes()))
    return dict(nodes=nodes, edges=edges, legacy=legacy, stages=stages)


@pytest.mark.parametrize("barrier", BARRIERS)
def test_edge_conductances(compared, barrier):
    stage, legacy, edges = compared["stages"][barrier], compared["legacy"], compared["edges"]
    net = stage["net"]
    assert len(np.unique(net.wall_thickness[np.isfinite(net.wall_thickness)])) > 1      # the walls differ
    local = np.full(net.cgroup.size, -1, dtype=np.int64)
    local[compared["nodes"]] = np.arange(compared["nodes"].size)
    expected = {tuple(sorted(k)): v for k, v in stage["legacy"]["edge_K"].items()}
    key = lambda e: tuple(sorted((int(legacy[local[net.a[e]]]), int(legacy[local[net.b[e]]]))))   # noqa: E731
    in_matrix = [e for e in edges if key(e) in expected]
    assert len(in_matrix) == len(expected)
    np.testing.assert_array_equal([stage["K"][e] for e in in_matrix], [expected[key(e)] for e in in_matrix])


@pytest.mark.parametrize("barrier", BARRIERS)
def test_soil_terms(compared, barrier):
    stage, legacy, nodes = compared["stages"][barrier], compared["legacy"], compared["nodes"]
    rhs_s = np.zeros(legacy.size)
    rhs_s[legacy] = -stage["soil"][nodes]                               # in legacy order
    np.testing.assert_array_equal(rhs_s, stage["legacy"]["rhs"].reshape(-1))
