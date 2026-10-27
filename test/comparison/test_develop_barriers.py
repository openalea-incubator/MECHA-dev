"""
Every barrier code, 0 to 11 (MECHA's 0-9; develop's 10, the exodermis suberized with a Casparian strip, and 11, the
endodermis and exodermis suberized, the sclerenchyma's walls lignified): the port's conductances are the legacy's,
edge by edge and bit for bit (doc/legacy.rst, d4; the sections have no sclerenchyma: its walls are tested in
test_develop_additions.py).
"""
import numpy as np
import pytest

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.geometry import section_parts

from legacy_run import legacy_options, legacy_section
from plants import anatomy_data_structure, root, template_organ

DEVELOP_BARRIERS = tuple(range(12))


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


def test_tracheids_as_xylem():
    """Half the stele's cells retagged transfusion tracheids (18) in both, with develop's tracheid_as_xylem: their
    membranes are the xylem's (kaqp_stele × 1e4 at barrier 1); every conductance the legacy's, bit for bit (the
    monocot root, its own section)."""
    g, ds, anatomy = anatomy_data_structure(root(1))
    segment = int(ds.entity_ids("SubOrgan")[0])
    height = float(np.asarray(ds.get("length"))[0]) * 1.0E6
    mecha_index = g.property("mecha_index")
    node_ids = np.asarray(ds.entity_ids("node"))
    cgroup = np.asarray(ds.get("cgroup"), dtype=float).copy()
    stele = np.flatnonzero(cgroup == 5)[::2]
    cgroup[stele] = 18
    ds.set("cgroup", cgroup)
    retagged = {int(mecha_index[int(node_ids[v])]) for v in stele}

    def to_tracheids(network):
        for node in retagged:
            network.graph.nodes[node]["cgroup"] = 18
            network.cell_manager.get_by_node_id(node).cgroup = 18

    run = legacy_section(template_organ(anatomy, g, segment), barriers=(1,), height=height,
                         hydraulic=dict(tracheid_as_xylem=True), edit_network=to_tracheids)
    ds.register("barrier", np.ones(ds.n_nodes()), location="node")
    mecha = legacy_options(MechaHydraulics(data_structure=ds))
    mecha.tracheid_as_xylem = 1.
    mecha()
    K, net = np.asarray(ds.get("conductance")), mecha.network()
    nodes, edges = section_parts(ds)[segment]
    legacy = np.array([int(mecha_index[int(v)]) for v in node_ids[nodes]])
    geo = mecha.section_geometries()[segment][2]
    graph = run.network.graph
    by_position = {tuple(graph.nodes[j]["position"]): j for j in range(run.network.n_walls, run.network.n_wall_junction)}
    for v in geo.junctions:
        legacy[v] = by_position[(geo.x[v], geo.y[v])]
    local = np.full(net.cgroup.size, -1, dtype=np.int64)
    local[nodes] = np.arange(nodes.size)
    expected = {tuple(sorted(k)): v for k, v in run.stages[0]["edge_K"].items()}
    key = lambda e: tuple(sorted((int(legacy[local[net.a[e]]]), int(legacy[local[net.b[e]]]))))   # noqa: E731
    in_matrix = [e for e in edges if key(e) in expected]
    assert len(in_matrix) == len(expected)
    np.testing.assert_array_equal([K[e] for e in in_matrix], [expected[key(e)] for e in in_matrix])
    tracheid_membranes = [e for e in in_matrix if graph.edges[key(e)]["path"] == "membrane"
                          and (set(key(e)) & retagged)]
    assert len(tracheid_membranes) > 0


@pytest.mark.parametrize("profile", ("linear", "exp"))
def test_plasmodesmata_by_distance_to_the_soil(case, profile):
    """develop's pd_spatial_config (c_wet 1, c_dry 0.6; the exponential's length 40 µm): the nodes' distances to the
    soil contact are the legacy's, and so are the plasmodesmata's conductances (to 1e-12: the shortest paths' sums)."""
    config = dict(c_wet=1.0, c_dry=0.6, profile=profile, length_scale=40.0 if profile == "exp" else None)
    ds, g = case["ds"], case["g"]
    run = legacy_section(template_organ(case["anatomy"], g, case["segment"]), barriers=(1,), height=case["height"],
                         hydraulic=dict(pd_spatial_config=config))
    if not ds.has("barrier"):
        ds.register("barrier", np.ones(ds.n_nodes()), location="node")
    ds.set("barrier", np.ones(ds.n_nodes()))
    mecha = legacy_options(MechaHydraulics(data_structure=ds))
    try:                                     # the parameters are the shared case's: restored for the other tests
        mecha.pd_spatial_profile = 1. if profile == "linear" else 2.
        mecha.pd_spatial_length_scale = config["length_scale"] or float("nan")
        mecha()
        K, net = np.asarray(ds.get("conductance")).copy(), mecha.network()
        distance = np.asarray(ds.get("contact_distance")).copy()
    finally:
        mecha.pd_spatial_profile, mecha.pd_spatial_length_scale = 0., float("nan")
    nodes, edges = section_parts(ds)[case["segment"]]
    mecha_index = g.property("mecha_index")
    legacy = np.array([int(mecha_index[int(v)]) for v in ds.entity_ids("node")[nodes]])
    geo = mecha.section_geometries()[case["segment"]][2]
    graph = run.network.graph
    by_position = {tuple(graph.nodes[j]["position"]): j for j in range(run.network.n_walls, run.network.n_wall_junction)}
    for v in geo.junctions:
        legacy[v] = by_position[(geo.x[v], geo.y[v])]
    reference = np.array([run.network.contact_distance.get(int(j), np.nan) for j in legacy])
    np.testing.assert_allclose(distance[nodes], reference, rtol=1e-12, equal_nan=True)
    local = np.full(net.cgroup.size, -1, dtype=np.int64)
    local[nodes] = np.arange(nodes.size)
    expected = {tuple(sorted(k)): v for k, v in run.stages[0]["edge_K"].items() if graph.edges[k]["path"] == "plasmodesmata"}
    key = lambda e: tuple(sorted((int(legacy[local[net.a[e]]]), int(legacy[local[net.b[e]]]))))   # noqa: E731
    pd = [e for e in edges if key(e) in expected]
    assert len(pd) == len(expected)
    np.testing.assert_allclose([K[e] for e in pd], [expected[key(e)] for e in pd], rtol=1e-12)
    coefficients = [graph.edges[key(e)]["pd_spatial_coef"] for e in pd]
    assert min(coefficients) < 0.7 and max(coefficients) == 1.0           # the profile spans c_dry to c_wet
