"""
The suberin lamella model (PR #67's suberin_model 'lamella'; the port's suberin_model 1, doc/legacy.rst, d19): a
suberized face's secondary layer and the endodermis' tertiary layer, in parallel along the walls and in series with
the membranes. With the inner, outer and radial lamellae, the bands and the exodermis' values all distinct, at the
barrier codes 3, 8 and 11 and with two sets of thicknesses, every edge conductance is the legacy's bit for bit, the
matrix to 4 ulp and the solution to 1e-12 (as test_conductance.py); with the legacy's wall half (lamella_wall_half 2:
kw on every membrane, U9).
"""
import numpy as np
import pytest
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.conductance import section_matrix
from openalea.mecha.geometry import MEMBRANE, section_parts

from legacy_run import legacy_options, legacy_section
from plants import anatomy_data_structure, root, template_organ

LAMELLA_BARRIERS = (3, 8, 11)
# {the port's parameter: value}, and the legacy's kw_barrier entry giving the same
BARRIER_VALUES = dict(kw_casparian=1.0E-9, kw_suberin_in=3.0E-9, kw_suberin_out=5.0E-9, kw_suberin_radial=7.0E-9,
                      kw_casparian_exodermis=2.0E-9, kw_suberin_exodermis=6.0E-9)
LEGACY_BARRIER_VALUES = {"value": 1.0E-16, "Casp": 1.0E-9, "Sub": 7.0E-9, "Sub_in": 3.0E-9, "Sub_out": 5.0E-9,
                         "Lig": 1.0E-16, "Casp_exo": 2.0E-9, "Sub_exo": 6.0E-9}
# the layers' thicknesses (µm): primary (None: the wall's own), secondary (None: a third of it), tertiary
LAYERS = {"thin lamella, tertiary layer": (None, 0.5, 1.0), "given primary wall, default lamella": (1.2, None, 0.0)}
PSI_SOIL, PSI_XYL = 0.0, -5.0E3                    # the legacy's default scenario (hPa)


def _nan(value):
    return float("nan") if value is None else value


def _legacy_nodes(data_structure, g, mecha, segment, network):
    """The legacy's node index of each of the section's nodes (its junctions by their position)."""
    nodes, edges = section_parts(data_structure)[segment]
    mecha_index = g.property("mecha_index")
    legacy = np.array([int(mecha_index[int(v)]) for v in data_structure.entity_ids("node")[nodes]])
    geo = mecha.section_geometries()[segment][2]
    by_position = {tuple(network.graph.nodes[j]["position"]): j for j in range(network.n_walls, network.n_wall_junction)}
    for v in geo.junctions:
        legacy[v] = by_position[(geo.x[v], geo.y[v])]
    return nodes, edges, legacy


def _matched(net, nodes, edges, legacy, reference):
    """(the port's edges the legacy's matrix has, the legacy's K on them)."""
    local = np.full(net.cgroup.size, -1, dtype=np.int64)
    local[nodes] = np.arange(nodes.size)
    expected = {tuple(sorted(k)): v for k, v in reference.items()}
    key = lambda e: tuple(sorted((int(legacy[local[net.a[e]]]), int(legacy[local[net.b[e]]]))))   # noqa: E731
    in_matrix = np.array([e for e in edges if key(e) in expected])
    assert in_matrix.size == len(expected)
    return in_matrix, np.array([expected[key(e)] for e in in_matrix])


@pytest.fixture(scope="module", params=list(LAYERS))
def compared(case, request):
    primary, secondary, tertiary = LAYERS[request.param]
    data_structure, g = case["data_structure"], case["mtg"]
    run = legacy_section(template_organ(case["anatomy"], g, case["segment"]), barriers=LAMELLA_BARRIERS,
                         height=case["height"],
                         hydraulic=dict(kw_barrier_elems=[LEGACY_BARRIER_VALUES], suberin_model="lamella",
                                        prim_wall_thickness=primary, sec_wall_thickness=secondary,
                                        tert_wall_thickness=tertiary))
    if not data_structure.has("barrier"):
        data_structure.register("barrier", np.ones(data_structure.n_nodes()), location="node")
    mecha = legacy_options(MechaHydraulics(data_structure=data_structure))
    nodes, edges, legacy = _legacy_nodes(data_structure, g, mecha, case["segment"], run.network)
    values = dict(BARRIER_VALUES, suberin_model=1., primary_wall_thickness=_nan(primary),
                  secondary_wall_thickness=_nan(secondary), tertiary_wall_thickness=tertiary)
    defaults = {name: np.array(getattr(mecha, name), copy=True) for name in values}
    stages = {}
    try:                                     # the parameters are the shared case's: restored for the other tests
        for name, value in values.items():
            setattr(mecha, name, value)
        for i, barrier in enumerate(LAMELLA_BARRIERS):
            data_structure.set("barrier", np.full(data_structure.n_nodes(), float(barrier)))
            mecha()
            stages[barrier] = dict(
                net=mecha.network(), legacy=run.stages[i],
                **{name: np.asarray(data_structure.get(name)).copy()
                   for name in ("conductance", "soil_conductance", "axial_boundary_conductance",
                                "wall_thickness_total", "membrane_lamella_resistance",
                                "membrane_tertiary_resistance")})
    finally:
        for name, value in defaults.items():
            setattr(mecha, name, value)
        data_structure.set("barrier", np.ones(data_structure.n_nodes()))
    geo = mecha.section_geometries()[case["segment"]][2]
    return dict(nodes=nodes, edges=edges, legacy=legacy, stages=stages, k_xyl=mecha.k_xyl,
                xylem=geo.typed["xylem"], layers=LAYERS[request.param])


@pytest.mark.parametrize("barrier", LAMELLA_BARRIERS)
def test_edge_conductances(compared, barrier):
    stage = compared["stages"][barrier]
    in_matrix, legacy_K = _matched(stage["net"], compared["nodes"], compared["edges"], compared["legacy"],
                                   stage["legacy"]["edge_K"])
    np.testing.assert_array_equal(stage["conductance"][in_matrix], legacy_K)
    # the lamellae are there: suberized membranes, and with a tertiary layer, some behind them
    assert (stage["membrane_lamella_resistance"][in_matrix] > 0).sum() > 0
    assert ((stage["membrane_tertiary_resistance"][in_matrix] > 0).sum() > 0) == (compared["layers"][2] > 0)


@pytest.mark.parametrize("barrier", LAMELLA_BARRIERS)
def test_wall_thickness_total(compared, barrier):
    """The walls' total thickness (their middle, secondary and tertiary layers) is the legacy's t_wall."""
    stage = compared["stages"][barrier]
    walls, reference = _matched(stage["net"], compared["nodes"], compared["edges"], compared["legacy"],
                                stage["legacy"]["edge_t_wall"])
    assert stage["net"].junction_edge[walls].all()
    np.testing.assert_array_equal(stage["wall_thickness_total"][walls], reference)
    assert (reference > reference.min()).any()                           # some walls have layers


def _section_matrix(compared, stage):
    order = compared["nodes"][np.argsort(compared["legacy"])]
    in_matrix = np.zeros(stage["conductance"].size, dtype=bool)
    in_matrix[compared["edges"]] = True
    return order, section_matrix(stage["net"], stage["conductance"], in_matrix, stage["soil_conductance"],
                                 stage["axial_boundary_conductance"], order, compared["edges"])


@pytest.mark.parametrize("barrier", LAMELLA_BARRIERS)
def test_matrix(compared, barrier):
    stage = compared["stages"][barrier]
    _, W = _section_matrix(compared, stage)
    removed = np.zeros(compared["nodes"].size)       # the axial terms the legacy removes in place (test_conductance.py)
    removed[compared["xylem"]] = compared["k_xyl"]
    removed = removed[np.argsort(compared["legacy"])]
    W = W + diags(removed)
    reference = stage["legacy"]["matrix_W"].tocsr()
    W.eliminate_zeros()
    reference.eliminate_zeros()
    eps = np.finfo(float).eps
    excess = abs(W - reference) - 4 * eps * (abs(W) + abs(reference)) / 2 - diags(2 * eps * removed)
    assert excess.data.size == 0 or excess.data.max() <= 0.


@pytest.mark.parametrize("barrier", LAMELLA_BARRIERS)
def test_solution(compared, barrier):
    stage = compared["stages"][barrier]
    order, W = _section_matrix(compared, stage)
    rhs = -stage["soil_conductance"][order] * PSI_SOIL - stage["axial_boundary_conductance"][order] * PSI_XYL
    reference = stage["legacy"]["solution"]
    np.testing.assert_allclose(spsolve(W, rhs), reference, rtol=0, atol=1e-12 * abs(reference).max())


def test_sclerenchyma_secondary_walls():
    """At barrier 11, a fibre's lignified wall is the secondary layer of its side of each of its walls (at
    kw_sclerenchyma), and its membranes have no lamella: the monocot root's cortex retagged sclerenchyma in both, the
    walls' and the membranes' conductances the legacy's, bit for bit."""
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

    def to_sclerenchyma(network):
        manager = network.cell_manager
        for node in retagged:
            cell = manager.get_by_node_id(node)
            manager._by_type[cell.cell_type].remove(cell)
            cell.cell_type = "sclerenchyma"
            manager._by_type.setdefault("sclerenchyma", []).append(cell)
            network.graph.nodes[node]["cell_type"] = "sclerenchyma"

    run = legacy_section(template_organ(anatomy, g, segment), barriers=(11,), height=height,
                         edit_network=to_sclerenchyma,
                         hydraulic=dict(kw_barrier_elems=[LEGACY_BARRIER_VALUES], suberin_model="lamella",
                                        sec_wall_thickness=0.5, tert_wall_thickness=1.0))
    data_structure.register("barrier", np.full(data_structure.n_nodes(), 11.), location="node")
    mecha = legacy_options(MechaHydraulics(data_structure=data_structure))
    for name, value in dict(BARRIER_VALUES, suberin_model=1., secondary_wall_thickness=0.5,
                            tertiary_wall_thickness=1.0).items():
        setattr(mecha, name, value)
    mecha()
    K, net = np.asarray(data_structure.get("conductance")), mecha.network()
    nodes, edges, legacy = _legacy_nodes(data_structure, g, mecha, segment, run.network)
    in_matrix, legacy_K = _matched(net, nodes, edges, legacy, run.stages[0]["edge_K"])
    assert net.sclerenchyma.sum() > 0
    for chosen in (net.junction_edge[in_matrix], net.path[in_matrix] == MEMBRANE):
        assert chosen.sum() > 0
        np.testing.assert_array_equal(K[in_matrix[chosen]], legacy_K[chosen])
