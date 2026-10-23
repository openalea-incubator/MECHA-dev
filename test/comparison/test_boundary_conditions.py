"""
The legacy's tutorial 02 (boundary conditions), ported: the standard problem with a flow at the xylem instead of its
potential. The port's solution, uptake, recovered xylem potential and kr against the legacy's (to 1e-12, D1); the
flow condition with the Dirichlet case's uptake gives back its kr.

Fix A2 (§18, legacy problem o): the port gives each vessel the share of its own area; the legacy computes the shares
in its cell manager's xylem order and gives them in its network's. Where the two orders agree (the dicot root) the
comparison stays exact; where they do not (the monocot root) the uptake still agrees, and the potentials differ.
"""
import numpy as np
import pytest

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.geometry import XYLEM_TYPES, section_parts

from legacy_run import legacy_neumann, legacy_options
from plants import template_organ

BARRIER = 3


@pytest.fixture(scope="module")
def both(case):
    ds, g = case["ds"], case["g"]
    if not ds.has("barrier"):
        ds.register("barrier", np.ones(ds.n_nodes()), location="node")
    ds.set("barrier", np.full(ds.n_nodes(), float(BARRIER)))
    mecha = legacy_options(MechaHydraulics(data_structure=ds))
    mecha()                                                         # Dirichlet: the uptake
    dirichlet = dict(uptake=float(ds.get("standard_uptake")[0]), kr=float(ds.get("kr")[0]))
    mecha.standard_axial_condition, mecha.standard_axial_flow = 2., dirichlet["uptake"]
    try:
        mecha()
        port = dict(psi=np.asarray(ds.get("standard_psi")).copy(), uptake=float(ds.get("standard_uptake")[0]),
                    kr=float(ds.get("kr")[0]))
    finally:
        mecha.standard_axial_condition, mecha.standard_axial_flow = 1., 0.
    legacy = legacy_neumann(template_organ(case["anatomy"], g, case["segment"]), BARRIER, dirichlet["uptake"],
                            height=case["height"])
    nodes, _ = section_parts(ds)[case["segment"]]
    geo = mecha.section_geometries()[case["segment"]][2]
    net = case["legacy"].network
    mecha_index = g.property("mecha_index")
    index = np.array([int(mecha_index[int(v)]) for v in ds.entity_ids("node")[nodes]])
    index[geo.junction_order] = net.n_walls + np.arange(net.n_junctions)
    typed = [c for t in XYLEM_TYPES for c in geo.cells[geo.tissue[geo.cells] == t]]
    same_order = list(typed) == list(geo.xylem_cells)               # the legacy's shares are then the right ones
    return dict(port=port, legacy=legacy, dirichlet=dirichlet, nodes=nodes, index=index, same_order=same_order)


def test_solution(both):
    """To 1e-11 of the range: with a flow at the xylem only the soil anchors the potentials, a less well-conditioned
    system than with the xylem's potential (1e-12 there). Only where the legacy's shares are right (fix A2)."""
    if not both["same_order"]:
        pytest.skip("the legacy gives the vessels mis-ordered shares (§2 o), fixed in the port (A2)")
    psi, reference = both["port"]["psi"][both["nodes"]], both["legacy"]["solution"][both["index"]]
    np.testing.assert_allclose(psi, reference, rtol=0, atol=1e-11 * abs(reference).max())


def test_uptake_and_kr(both):
    """The uptake is the given flow in both; kr (through the vessels' mean potential) differs with the shares (A2):
    by 1.6e-4 on the monocot root."""
    assert both["port"]["uptake"] == pytest.approx(both["legacy"]["total_flow"], rel=1e-12)
    assert both["port"]["kr"] == pytest.approx(both["legacy"]["kr"], rel=1e-10 if both["same_order"] else 1e-3)


def test_same_physical_state_as_dirichlet(both):
    """The flow of the Dirichlet case gives back its uptake; kr differs only by the vessels' spread of potential."""
    assert both["port"]["uptake"] == pytest.approx(both["dirichlet"]["uptake"], rel=1e-9)
    assert both["port"]["kr"] == pytest.approx(both["dirichlet"]["kr"], rel=0.05)
