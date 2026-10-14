"""
Step 3 (devplan_metafspm.md, §8): the port's conductances are the legacy's, edge by edge and bit for bit, as are the
soil terms; its matrix agrees to 4 ulp and its solution to 1e-12 (decision D1: summation and node orders differ).
"""
import numpy as np
import pytest
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.conductance import section_matrix
from openalea.mecha.geometry import section_parts

from conftest import BARRIERS

PSI_SOIL, PSI_XYL = 0.0, -5.0E3                    # the legacy's default scenario (hPa)


@pytest.fixture(scope="module")
def port(case):
    ds, g = case["ds"], case["g"]
    if not ds.has("barrier"):
        ds.register("barrier", np.ones(ds.n_nodes()), location="node")
    mecha = MechaHydraulics(data_structure=ds)
    nodes, edges = section_parts(ds)[case["segment"]]
    net = case["legacy"].network
    mecha_index = g.property("mecha_index")
    legacy = np.array([int(mecha_index[int(v)]) for v in ds.entity_ids("node")[nodes]])
    geo = mecha.section_geometries()[case["segment"]][2]
    by_position = {tuple(net.graph.nodes[j]["position"]): j for j in range(net.n_walls, net.n_wall_junction)}
    for v in geo.junctions:
        legacy[v] = by_position[(geo.x[v], geo.y[v])]
    stages = {}
    for i, barrier in enumerate(BARRIERS):
        ds.set("barrier", np.full(ds.n_nodes(), float(barrier)))
        mecha()
        stages[barrier] = dict(K=np.asarray(ds.get("conductance")).copy(), soil=np.asarray(ds.get("soil_conductance")).copy(),
                               axial=np.asarray(ds.get("axial_boundary_conductance")).copy(), net=mecha.network(),
                               legacy=case["legacy"].stages[i])
    return dict(mecha=mecha, nodes=nodes, edges=edges, legacy=legacy, stages=stages, segment=case["segment"])


@pytest.mark.parametrize("barrier", BARRIERS)
def test_edge_conductances(port, barrier):
    stage, legacy, edges = port["stages"][barrier], port["legacy"], port["edges"]
    net = stage["net"]
    local = np.full(net.cgroup.size, -1, dtype=np.int64)
    local[port["nodes"]] = np.arange(port["nodes"].size)
    reference = stage["legacy"]["edge_K"]
    key = lambda e: tuple(sorted((int(legacy[local[net.a[e]]]), int(legacy[local[net.b[e]]]))))
    expected = {tuple(sorted(k)): v for k, v in reference.items()}
    in_matrix = [e for e in edges if key(e) in expected]
    assert len(in_matrix) == len(expected)
    port_K = np.array([stage["K"][e] for e in in_matrix])
    legacy_K = np.array([expected[key(e)] for e in in_matrix])
    np.testing.assert_array_equal(port_K, legacy_K)
    others = np.setdiff1d(edges, in_matrix)
    assert (stage["K"][others] == 0).all()


@pytest.mark.parametrize("barrier", BARRIERS)
def test_soil_terms(port, barrier):
    stage, legacy, nodes = port["stages"][barrier], port["legacy"], port["nodes"]
    rhs_s = np.zeros(legacy.size)
    rhs_s[legacy] = -stage["soil"][nodes]                               # in legacy order
    np.testing.assert_array_equal(rhs_s, stage["legacy"]["rhs"].reshape(-1))


def _legacy_order(port):
    return port["nodes"][np.argsort(port["legacy"])]


@pytest.mark.parametrize("barrier", BARRIERS)
def test_matrix(port, barrier):
    stage = port["stages"][barrier]
    in_matrix = np.zeros(stage["K"].size, dtype=bool)
    in_matrix[port["edges"]] = True
    W = section_matrix(stage["net"], stage["K"], in_matrix, stage["soil"], stage["axial"], _legacy_order(port),
                       port["edges"])
    # the legacy returns its matrix after removing the axial terms in place (remove_xyl_phloem_BC), at its cell
    # manager's xylem cells (by tissue): the same removal on the port's
    geo = port["mecha"].section_geometries()[port["segment"]][2]
    removed = np.zeros(port["nodes"].size)
    if barrier > 0:
        removed[geo.typed["xylem"]] = port["mecha"].k_xyl
    W = W + diags(removed[np.argsort(port["legacy"])])
    reference = stage["legacy"]["matrix_W"].tocsr()
    W.eliminate_zeros()
    reference.eliminate_zeros()
    eps = np.finfo(float).eps
    excess = abs(W - reference) - 4 * eps * (abs(W) + abs(reference)) / 2
    assert excess.data.size == 0 or excess.data.max() <= 0.


@pytest.mark.parametrize("barrier", BARRIERS)
def test_solution(port, barrier):
    stage = port["stages"][barrier]
    order = _legacy_order(port)
    in_matrix = np.zeros(stage["K"].size, dtype=bool)
    in_matrix[port["edges"]] = True
    W = section_matrix(stage["net"], stage["K"], in_matrix, stage["soil"], stage["axial"], order, port["edges"])
    rhs = -stage["soil"][order] * PSI_SOIL - stage["axial"][order] * PSI_XYL       # barrier > 0: no duplicates
    psi = spsolve(W, rhs)
    reference = stage["legacy"]["solution"]
    np.testing.assert_allclose(psi, reference, rtol=0, atol=1e-12 * abs(reference).max())
