"""
The solutes' liquid boundary (solute_boundary_advection 1; develop's conserve_boundary, doc/legacy.rst d7), on the
seedling of test_shoot.py: the root takes up soil water, which crosses the stem and evaporates from the leaf. Water
leaving the liquid network as vapour leaves its solutes behind: the leaf gives the boundary no water, its solute only
reaches its sinks, and its evaporating walls concentrate. The root's soil water brings c_boundary (as does the water
each section receives from its neighbour: the sections are not connected for the solutes).
"""
import numpy as np
import pytest

from openalea.metafspm.scene.population import Planting, planting

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.geometry import WALL_AIR
from openalea.mecha.solutes import MechaSolutes
from openalea.mecha.stand_ins import Transpiration

from plants import initiated
from test_shoot import SOIL, TRANSPIRATION, Anatomy, Seedling

DT = 3600.                                                 # s
C0 = 1e-6                                                  # mol cm-3, the starting sucrose
C_SOIL = 2e-6                                              # mol cm-3, the soil water's (c_boundary)


@pytest.fixture(scope="module")
def stepped():
    data_structure = initiated(planting([Planting()], None), Seedling, Anatomy, time_step=DT).data_structure
    mtg = data_structure.structure
    Anatomy(data_structure=data_structure)()
    data_structure.register("barrier", np.ones(data_structure.n_nodes()), location="node")
    data_structure.register("soil_psi", np.full(data_structure.n_nodes(), SOIL), location="node")
    Transpiration(data_structure=data_structure, transpiration=TRANSPIRATION)()       # its one leaf organ's
    mecha = MechaHydraulics(data_structure=data_structure)
    mecha.collar_condition = 0.
    mecha()
    solutes = MechaSolutes(data_structure=data_structure, time_step=DT)
    solutes.c_boundary, solutes.c_sink = C_SOIL, C0
    liquid = np.asarray(data_structure.get("vapour_node")) == 0         # the liquid nodes (MechaSolutes' liquid_node)
    data_structure.set("sucrose", np.where(liquid, C0, 0.))
    solutes()
    np.testing.assert_array_equal(np.asarray(data_structure.get("liquid_node")) > 0, liquid)
    labels = mtg.property("label")
    owners = data_structure.entity_ids("SubOrgan")[data_structure.owner("SubOrgan")]
    leaf = np.isin(owners, [int(v) for v in data_structure.entity_ids("SubOrgan") if labels[int(v)] == mtg.labels.SubOrgan.LeafElement])
    root = np.isin(owners, [int(v) for v in data_structure.entity_ids("SubOrgan") if labels[int(v)] == mtg.labels.SubOrgan.RootSegment])
    return dict(data_structure=data_structure, liquid=liquid, leaf=leaf, root=root)


def test_evaporation_leaves_the_solutes_behind(stepped):
    data_structure, liquid, leaf = stepped["data_structure"], stepped["liquid"], stepped["leaf"]
    view = data_structure.to_graph_view()
    flow, path = np.asarray(data_structure.get("flow")), np.asarray(data_structure.get("network_path"))
    evaporating = (path == WALL_AIR) & liquid[view.tail] & leaf[view.tail]
    evaporated = flow[evaporating].sum() - flow[(path == WALL_AIR) & liquid[view.head] & leaf[view.head]].sum()
    assert evaporated > 0.5 * TRANSPIRATION                 # most of the leaf's water leaves as vapour
    outflow = np.asarray(data_structure.get("boundary_outflow"))
    assert outflow[leaf & liquid].sum() < 1e-9 * evaporated  # none of it given to the boundary
    walls = np.unique(view.tail[evaporating & (flow > 0)])
    assert (np.asarray(data_structure.get("sucrose"))[walls] > C0).all()  # the evaporating walls concentrate


def test_balances_with_the_boundary(stepped):
    """Every free liquid node's balance: V Δc / dt + B q + O c − b = 0 (no sources, no starch); the leaf's free nodes
    lose solute only to its sinks."""
    data_structure, liquid = stepped["data_structure"], stepped["liquid"]
    B = data_structure.to_graph_view().incidence
    c, V = np.asarray(data_structure.get("sucrose")), np.asarray(data_structure.get("node_volume"))
    q = (np.asarray(data_structure.get("tail_weight")) * np.asarray(B.maximum(0).T @ c).ravel()
         - np.asarray(data_structure.get("head_weight")) * np.asarray((-B).maximum(0).T @ c).ravel())
    out = np.asarray(B @ q).ravel()
    O, b = np.asarray(data_structure.get("boundary_outflow")), np.asarray(data_structure.get("boundary_source"))
    gained = V * (c - np.where(liquid, C0, 0.)) / (DT / 86400.)
    free = liquid & (np.asarray(data_structure.get("sucrose_sink")) == 0)
    residual = gained + out + O * c - b
    np.testing.assert_allclose(residual[free], 0., atol=1e-9 * np.abs(out[free]).max())


def test_soil_water_brings_its_solute(stepped):
    """The root's inflow, its soil uptake (the transpiration), brings c_boundary. The sections are not connected for
    the solutes: the water a section receives from its neighbour (its axial junctions) is boundary water too."""
    data_structure, root, liquid = stepped["data_structure"], stepped["root"], stepped["liquid"]
    b = np.asarray(data_structure.get("boundary_source"))
    assert b[root & liquid].sum() == pytest.approx(TRANSPIRATION * C_SOIL, rel=1e-6)
