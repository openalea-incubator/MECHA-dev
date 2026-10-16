"""
The shoot in MECHA's plant problem (devplan_metafspm.md, §15, E1–E3): a stem element bearing a leaf element and a
root segment, GRANAP's anatomies (monocot stem and leaf presets, the root preset). Only the root exchanges with the
soil, only the leaf's stomatal pores with the air; what the roots take up, the leaf transpires.
"""
import numpy as np
import pandas as pd
import pytest

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import StructuralComponent
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.scene.population import build_population

from openalea.granap.leaf_class import LeafAnatomy
from openalea.granap.stem_class import StemAnatomy
from openalea.granap.metafspm.component import GranapAnatomy, OrganAnatomy, junction_rules
from openalea.granap.metafspm.library import AnatomyLibrary, preset_parameters
from openalea.mecha.component import MechaHydraulics

SOIL = -1000.                                            # hPa


class Seedling(StructuralComponent):
    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        s, L = g.scales, g.labels

        def organ(axis_label, kind, organ_label):
            axis = g.add_component(plant, **PropsConfig(scale=s.Axis, edge_type='/', label=getattr(L.Axis, axis_label)))
            unit = g.add_component(axis, **PropsConfig(scale=s.GrowthUnit, edge_type='/',
                                                       label=getattr(L.GrowthUnit, kind)))
            phytomer = g.add_component(unit, **PropsConfig(scale=s.Phytomer, edge_type='/',
                                                           label=getattr(L.Phytomer, kind)))
            return g.add_component(phytomer, **PropsConfig(scale=s.Organ, edge_type='/',
                                                           label=getattr(L.Organ, organ_label)))

        stem = g.add_component(organ("Shoot", "Shoot", "StemInternode"),
                               **PropsConfig(scale=s.SubOrgan, edge_type='/', label=L.SubOrgan.StemElement))
        leaf = g.add_component_with_topo(organ("Shoot", "Shoot", "Leaf"), stem,
                                         **PropsConfig(scale=s.SubOrgan, edge_type='+', label=L.SubOrgan.LeafElement))
        root = g.add_component_with_topo(organ("Root", "Root", "RootInternode"), stem,
                                         **PropsConfig(scale=s.SubOrgan, edge_type='+', label=L.SubOrgan.RootSegment))
        for v, radius in ((stem, 1e-3), (leaf, 1e-3), (root, 0.55e-3)):
            g.property("radius")[v] = radius
            g.property("length")[v] = 5e-3
            g.property("distance_from_tip")[v] = 0.


class Anatomy(GranapAnatomy):
    shoot = {"StemElement": OrganAnatomy(AnatomyLibrary(parameters=preset_parameters("for_monocot_stem"),
                                                        organ_class=StemAnatomy)),
             "LeafElement": OrganAnatomy(AnatomyLibrary(parameters=preset_parameters("for_monocot_leaf"),
                                                        organ_class=LeafAnatomy), size=None)}


@pytest.fixture(scope="module")
def plant():
    Choregrapher().add_simulation_time_step(3600.)
    table = pd.DataFrame([dict(plant="p0", model=None, x=0., y=0., z=0., rotation=0., scenario={"parameters": {}})])
    g, _ = build_population(table, initiators=(Seedling, Anatomy))
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=junction_rules(g))
    Anatomy(data_structure=ds)()
    ds.register("barrier", np.ones(ds.n_nodes()), location="node")
    ds.register("soil_psi", np.full(ds.n_nodes(), SOIL), location="node")
    mecha = MechaHydraulics(data_structure=ds)
    mecha.collar_condition = 0.
    mecha()
    labels = g.property("label")
    kinds = {int(v): labels[int(v)] for v in ds.entity_ids("SubOrgan")}
    return g, ds, mecha, kinds


def _organ_nodes(g, ds, kinds, label):
    owners = ds.entity_ids("SubOrgan")[ds.owner("SubOrgan")]
    return np.isin(owners, [v for v, k in kinds.items() if k == label])


def test_only_the_root_meets_the_soil_and_only_the_leaf_the_air(plant):
    g, ds, mecha, kinds = plant
    L = g.labels.SubOrgan
    root, leaf = _organ_nodes(g, ds, kinds, L.RootSegment), _organ_nodes(g, ds, kinds, L.LeafElement)
    soil = ds.get("soil_conductance")
    assert (soil[root] > 0).any() and (soil[~root] == 0).all()
    # transpiration_model 2 (default, §25): the walls of the leaf's air spaces, pinned; no pore evaporates
    pinned = ds.get("air_wall_conductance") > 0
    assert pinned.any() and leaf[pinned].all() and (ds.get("kind")[pinned] == 1).all()
    assert (ds.get("evaporation_conductance") == 0).all()


def test_stomatal_pores_as_an_option(plant):
    """transpiration_model 1: the leaf's stomatal pores meet the air (E3), no wall is pinned."""
    g, ds, mecha, kinds = plant
    leaf = _organ_nodes(g, ds, kinds, g.labels.SubOrgan.LeafElement)
    mecha.transpiration_model = 1.
    try:
        mecha()
        pores = ds.get("evaporation_conductance") > 0
        assert pores.any() and leaf[pores].all() and (ds.get("tissue")[pores] == "pore").all()
        assert (ds.get("air_wall_conductance") == 0).all()
        assert ds.get("evaporation").sum() == pytest.approx(ds.get("root_uptake").sum(), rel=1e-8)
    finally:
        mecha.transpiration_model = 2.
        mecha()


def test_root_uptake_is_transpired(plant):
    """To 1e-6: the pinned walls' penalty (1e3 × the largest diagonal term, the reactions' device) puts their terms'
    round-off at ~1e-9 cm3 d-1 each."""
    g, ds, mecha, kinds = plant
    uptake, evaporation = ds.get("root_uptake").sum(), ds.get("evaporation").sum()
    assert uptake > 0 and evaporation == pytest.approx(uptake, rel=1e-6)
    psi = ds.get("psi")
    liquid = ds.get("tissue") != "air space"               # on the air spaces, ψ holds their vapour pressure (§25)
    assert np.isfinite(psi).all() and psi[liquid].max() <= SOIL + 1e-6


def test_root_quantities_on_roots_only(plant):
    g, ds, mecha, kinds = plant
    L = g.labels.SubOrgan
    for i, v in enumerate(ds.entity_ids("SubOrgan")):
        root = kinds[int(v)] == L.RootSegment
        assert np.isfinite(ds.get("perimeter")[i]) == root and np.isfinite(ds.get("kr")[i]) == root
        assert np.isfinite(ds.get("Kx")[i])
