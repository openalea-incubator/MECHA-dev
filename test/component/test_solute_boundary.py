"""
The solutes' liquid boundary (solute_boundary_advection 1; develop's conserve_boundary, doc/legacy.rst d7), on the
seedling of test_shoot.py: the root takes up soil water, which crosses the stem and evaporates from the leaf. Water
leaving the liquid network as vapour leaves its solutes behind: the leaf gives the boundary no water, its solute only
reaches its sinks, and its evaporating walls concentrate. The root's soil water brings c_boundary (as does the water
each section receives from its neighbour: the sections are not connected for the solutes).
"""
import numpy as np
import pandas as pd
import pytest

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.scene.population import build_population

from openalea.granap.metafspm.component import junction_rules
from openalea.mecha.component import MechaHydraulics
from openalea.mecha.geometry import WALL_AIR
from openalea.mecha.solutes import MechaSolutes

from test_shoot import SOIL, TRANSPIRATION, Anatomy, Seedling

DT = 3600.                                                 # s
C0 = 1e-6                                                  # mol cm-3, the starting sucrose
C_SOIL = 2e-6                                              # mol cm-3, the soil water's (c_boundary)


@pytest.fixture(scope="module")
def stepped():
    Choregrapher().add_simulation_time_step(DT)
    table = pd.DataFrame([dict(plant="p0", model=None, x=0., y=0., z=0., rotation=0., scenario={"parameters": {}})])
    g, _ = build_population(table, initiators=(Seedling, Anatomy))
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=junction_rules(g))
    Anatomy(data_structure=ds)()
    ds.register("barrier", np.ones(ds.n_nodes()), location="node")
    ds.register("soil_psi", np.full(ds.n_nodes(), SOIL), location="node")
    for organ in ds.entity_ids("Organ"):
        leaf = g.property("label").get(int(organ)) == g.labels.Organ.Leaf
        g.property("leaf_transpiration")[int(organ)] = TRANSPIRATION if leaf else 0.
    mecha = MechaHydraulics(data_structure=ds)
    mecha.collar_condition = 0.
    mecha()
    solutes = MechaSolutes(data_structure=ds, time_step=DT)
    solutes.c_boundary, solutes.c_sink = C_SOIL, C0
    liquid = np.asarray(ds.get("vapour_node")) == 0         # the liquid nodes (MechaSolutes' liquid_node)
    ds.set("sucrose", np.where(liquid, C0, 0.))
    solutes()
    np.testing.assert_array_equal(np.asarray(ds.get("liquid_node")) > 0, liquid)
    labels = g.property("label")
    owners = ds.entity_ids("SubOrgan")[ds.owner("SubOrgan")]
    leaf = np.isin(owners, [int(v) for v in ds.entity_ids("SubOrgan") if labels[int(v)] == g.labels.SubOrgan.LeafElement])
    root = np.isin(owners, [int(v) for v in ds.entity_ids("SubOrgan") if labels[int(v)] == g.labels.SubOrgan.RootSegment])
    return dict(ds=ds, liquid=liquid, leaf=leaf, root=root)


def test_evaporation_leaves_the_solutes_behind(stepped):
    ds, liquid, leaf = stepped["ds"], stepped["liquid"], stepped["leaf"]
    view = ds.to_graph_view()
    flow, path = np.asarray(ds.get("flow")), np.asarray(ds.get("network_path"))
    evaporating = (path == WALL_AIR) & liquid[view.tail] & leaf[view.tail]
    evaporated = flow[evaporating].sum() - flow[(path == WALL_AIR) & liquid[view.head] & leaf[view.head]].sum()
    assert evaporated > 0.5 * TRANSPIRATION                 # most of the leaf's water leaves as vapour
    outflow = np.asarray(ds.get("boundary_outflow"))
    assert outflow[leaf & liquid].sum() < 1e-9 * evaporated  # none of it given to the boundary
    walls = np.unique(view.tail[evaporating & (flow > 0)])
    assert (np.asarray(ds.get("sucrose"))[walls] > C0).all()  # the evaporating walls concentrate


def test_balances_with_the_boundary(stepped):
    """Every free liquid node's balance: V Δc / dt + B q + O c − b = 0 (no sources, no starch); the leaf's free nodes
    lose solute only to its sinks."""
    ds, liquid = stepped["ds"], stepped["liquid"]
    B = ds.to_graph_view().incidence
    c, V = np.asarray(ds.get("sucrose")), np.asarray(ds.get("node_volume"))
    q = (np.asarray(ds.get("tail_weight")) * np.asarray(B.maximum(0).T @ c).ravel()
         - np.asarray(ds.get("head_weight")) * np.asarray((-B).maximum(0).T @ c).ravel())
    out = np.asarray(B @ q).ravel()
    O, b = np.asarray(ds.get("boundary_outflow")), np.asarray(ds.get("boundary_source"))
    gained = V * (c - np.where(liquid, C0, 0.)) / (DT / 86400.)
    free = liquid & (np.asarray(ds.get("sucrose_sink")) == 0)
    residual = gained + out + O * c - b
    np.testing.assert_allclose(residual[free], 0., atol=1e-9 * np.abs(out[free]).max())


def test_soil_water_brings_its_solute(stepped):
    """The root's inflow, its soil uptake (the transpiration), brings c_boundary. The sections are not connected for
    the solutes: the water a section receives from its neighbour (its axial junctions) is boundary water too."""
    ds, root, liquid = stepped["ds"], stepped["root"], stepped["liquid"]
    b = np.asarray(ds.get("boundary_source"))
    assert b[root & liquid].sum() == pytest.approx(TRANSPIRATION * C_SOIL, rel=1e-6)
