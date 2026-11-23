"""
The sclerenchyma at barrier 11 (PR #67), on the monocot root whose cortex is retagged sclerenchyma in both (the port's
tissue, the legacy's cell type): its lignified walls at kw_sclerenchyma_fraction × kw (doc/legacy.rst, d13), its dead
cells' plasmodesmata closed and its living cells' at sclerenchyma_plasmodesmata_factor (d17: the legacy given the
port's draw, and its product of the two cells' factors, sclerenchyma_plasmodesmata_rule 2); every conductance the
legacy's, bit for bit.
"""
import numpy as np
import pytest

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.geometry import MEMBRANE, PLASMODESMATA, section_parts
from openalea.mecha.legacy.utils.hydraulic_solver import HydraulicMatrixBuilder

from legacy_run import legacy_options, legacy_section
from plants import anatomy_data_structure, root, template_organ

BARRIER = 11


def _to_sclerenchyma(cells):
    """An edit_network of the legacy: its *cells* (node ids) of cell type sclerenchyma."""
    def edit(network):
        manager = network.cell_manager
        for node in cells:
            cell = manager.get_by_node_id(node)
            manager._by_type[cell.cell_type].remove(cell)
            cell.cell_type = "sclerenchyma"
            manager._by_type.setdefault("sclerenchyma", []).append(cell)
            network.graph.nodes[node]["cell_type"] = "sclerenchyma"
    return edit


@pytest.fixture(scope="module")
def compared():
    g, data_structure, anatomy = anatomy_data_structure(root(1))
    segment = int(data_structure.entity_ids("SubOrgan")[0])
    height = float(np.asarray(data_structure.get("length"))[0]) * 1.0E6
    mecha_index = g.property("mecha_index")
    node_ids = np.asarray(data_structure.entity_ids("node"))
    tissue = np.asarray(data_structure.get("tissue"), dtype=object).copy()
    cortex = np.flatnonzero(tissue == "cortex")
    tissue[cortex] = "sclerenchyma"
    data_structure.set("tissue", tissue)
    retagged = [int(mecha_index[int(node_ids[v])]) for v in cortex]
    data_structure.register("barrier", np.full(data_structure.n_nodes(), float(BARRIER)), location="node")
    mecha = legacy_options(MechaHydraulics(data_structure=data_structure))
    mecha()
    K, net = np.asarray(data_structure.get("conductance")).copy(), mecha.network()
    # the port's draw given to the legacy: {cell id: its plasmodesmata factor} (dead 0)
    factor = float(mecha.sclerenchyma_plasmodesmata_factor)
    nodes_scl = cortex[net.sclerenchyma[cortex]]
    draw = {int(mecha_index[int(node_ids[v])]): (factor if net.living_cell[v] else 0.) for v in nodes_scl}
    build = HydraulicMatrixBuilder._build_scl_pd_factors

    def the_ports_draw(builder, kpl_config, barrier):
        if barrier != BARRIER:
            return {}
        offset = builder.network.n_wall_junction
        return {node - offset: value for node, value in draw.items()}
    HydraulicMatrixBuilder._build_scl_pd_factors = the_ports_draw
    try:
        run = legacy_section(template_organ(anatomy, g, segment), barriers=(BARRIER,), height=height,
                             edit_network=_to_sclerenchyma(retagged))
    finally:
        HydraulicMatrixBuilder._build_scl_pd_factors = build
    nodes, edges = section_parts(data_structure)[segment]
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
    in_matrix = np.array([e for e in edges if key(e) in expected])
    assert in_matrix.size == len(expected)
    return dict(K=K, net=net, in_matrix=in_matrix, legacy_K=np.array([expected[key(e)] for e in in_matrix]),
                mecha=mecha, draw=draw)


def test_walls_and_membranes(compared):
    net, in_matrix = compared["net"], compared["in_matrix"]
    assert net.sclerenchyma.sum() > 0
    walls = net.junction_edge[in_matrix]
    membranes = net.path[in_matrix] == MEMBRANE
    for chosen in (walls, membranes):
        np.testing.assert_array_equal(compared["K"][in_matrix[chosen]], compared["legacy_K"][chosen])


def test_plasmodesmata_of_dead_and_living_sclerenchyma(compared):
    """The port's draw (sclerenchyma_dead_fraction 0.8 of the cells dead, exactly) given to the legacy, with its
    product of the two cells' factors: every plasmodesma the legacy's, bit for bit."""
    net, in_matrix, draw = compared["net"], compared["in_matrix"], compared["draw"]
    dead = sum(value == 0. for value in draw.values())
    assert dead == int(round(0.8 * len(draw)))
    plasmodesmata = net.path[in_matrix] == PLASMODESMATA
    np.testing.assert_array_equal(compared["K"][in_matrix[plasmodesmata]], compared["legacy_K"][plasmodesmata])
