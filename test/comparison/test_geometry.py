"""
Step 2 (devplan_metafspm.md, §8): MECHA's geometry from the DataStructure is the legacy's, exactly (bit for bit),
quantity by quantity, on the same GRANAP section.
"""
import numpy as np
import pytest

from openalea.mecha.geometry import section_geometry, section_of, section_parts


@pytest.fixture(scope="module")
def both(case):
    ds, g = case["ds"], case["g"]
    nodes, edges = section_parts(ds)[case["segment"]]
    section = section_of(ds, nodes, edges)
    geo = section_geometry(section)
    net = case["legacy"].network
    # legacy node id of each local node: walls and cells by GRANAP's MECHA index, the rebuilt junctions by position
    mecha_index = g.property("mecha_index")
    vids = ds.entity_ids("node")[nodes]
    legacy = np.array([int(mecha_index[int(v)]) for v in vids])
    by_position = {tuple(net.graph.nodes[j]["position"]): j for j in range(net.n_walls, net.n_wall_junction)}
    for v in geo.junctions:
        legacy[v] = by_position[(geo.x[v], geo.y[v])]
    return dict(section=section, geo=geo, net=net, legacy=legacy, local={int(l): i for i, l in enumerate(legacy)})


def test_counts_and_order(both):
    geo, net = both["geo"], both["net"]
    assert (geo.walls.size, geo.junctions.size, geo.cells.size) == (net.n_walls, net.n_junctions, net.n_cells)
    assert sorted(both["legacy"].tolist()) == list(range(net.n_wall_junction + net.n_cells))
    np.testing.assert_array_equal(both["legacy"][geo.walls], np.arange(net.n_walls))
    np.testing.assert_array_equal(both["legacy"][geo.cells], net.n_wall_junction + np.arange(net.n_cells))


def test_positions_lengths_areas(both):
    geo, net, legacy = both["geo"], both["net"], both["legacy"]
    for v in range(legacy.size):
        assert tuple(net.graph.nodes[legacy[v]]["position"]) == (geo.x[v], geo.y[v])
    wall_lengths = np.array([net.wall_lengths[legacy[v]] for v in np.r_[geo.walls, geo.junctions]])
    np.testing.assert_array_equal(geo.wall_length[np.r_[geo.walls, geo.junctions]], wall_lengths)
    np.testing.assert_array_equal(geo.cell_area[geo.cells], net.cell_areas)


def test_walls(both):
    geo, net = both["geo"], both["net"]
    np.testing.assert_array_equal(geo.lateral_distance[geo.walls], net.distance_wall_cell[:, 0])
    assert np.flatnonzero(geo.border_wall[geo.walls]).tolist() == list(net.border_walls)
    assert np.flatnonzero(geo.border_aerenchyma[geo.walls]).tolist() == list(net.border_aerenchyma)
    assert sorted(both["legacy"][geo.border_junction].tolist()) == sorted(net.border_junction)


def test_rebuilt_wall_edges(both):
    geo, net, s, legacy = both["geo"], both["net"], both["section"], both["legacy"]
    edges = np.flatnonzero(geo.junction_edge)
    rebuilt = [(u, v) for u, v, d in net.graph.edges(data=True)
               if d["path"] == "wall" and (net.n_walls <= u < net.n_wall_junction or net.n_walls <= v < net.n_wall_junction)]
    assert edges.size == len(rebuilt)
    for e in edges:
        d = net.graph.edges[legacy[s.a[e]], legacy[s.b[e]]]
        assert (d["length"], d["lateral_distance"]) == (geo.edge_length[e], geo.edge_lateral_distance[e])


def test_membrane_and_plasmodesmata_edges(both):
    geo, net, s, legacy = both["geo"], both["net"], both["section"], both["legacy"]
    for e in np.flatnonzero(np.isin(s.path, (0, 3))):
        d = net.graph.edges[legacy[s.a[e]], legacy[s.b[e]]]
        assert (d["length"], d["dist"]) == (geo.edge_length[e], geo.edge_dist[e])


def test_centre_distances_perimeter(both):
    geo, net = both["geo"], both["net"]
    assert geo.gravity_center == (net.x_grav, net.y_grav)
    np.testing.assert_array_equal(geo.distance_center_grav[geo.walls], net.distance_center_grav[:, 0])
    assert (geo.distance_max_cortex, geo.distance_min_cortex, geo.distance_avg_epi, geo.perimeter) == \
           (net.distance_max_cortex, net.distance_min_cortex, net.distance_avg_epi, net.perimeter)


def test_ranks_and_layers(both):
    geo, net = both["geo"], both["net"]
    np.testing.assert_array_equal(geo.cell_rank[geo.cells], net.cell_ranks)
    np.testing.assert_array_equal(geo.layer_dist, net.layer_dist)
    np.testing.assert_array_equal(geo.n_layer, net.n_layer)
    assert geo.xylem_80_percentile_distance == net.xylem_80_percentile_distance
    assert (geo.outercortex_connec_rank, geo.stele_connec_rank) == (net.outercortex_connec_rank, net.stele_connec_rank)


def test_plasmodesmata_ratios(both):
    geo, net = both["geo"], both["net"]
    assert (geo.concentrated_outer_cortex, geo.concentrated_cortex_cortex, geo.concentrated_cortex_endo) == \
           (net.concentrated_outer_cortex, net.concentrated_cortex_cortex, net.concentrated_cortex_endo)


def test_cell_sets(both):
    geo, net, legacy = both["geo"], both["net"], both["legacy"]
    cell_id = lambda v: legacy[v] - net.n_wall_junction
    assert legacy[geo.xylem_cells].tolist() == list(net.xylem_cells)
    assert legacy[geo.sieve_cells].tolist() == list(net.sieve_cells)
    assert legacy[geo.protosieve_cells].tolist() == list(net.protosieve_cells)
    assert cell_id(geo.passage_cells).tolist() == list(net.passage_cells)
    assert cell_id(geo.intercellular_cells).tolist() == list(net.intercellular_cells)
    cm = net.cell_manager
    for name in ("intercellular", "passage", "xylem", "sieve", "protosieve"):
        assert sorted(legacy[geo.typed[name]].tolist()) == sorted(c.node_id for c in getattr(cm, name)), name


def test_junction_ids(both):
    geo, net = both["geo"], both["net"]
    np.testing.assert_array_equal(both["legacy"][geo.junction_order], net.n_walls + np.arange(net.n_junctions))
