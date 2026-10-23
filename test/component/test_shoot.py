"""
The shoot in MECHA's plant problem (devplan_metafspm.md, §15, E1–E3): a stem element bearing a leaf element and a
root segment, GRANAP's anatomies (monocot stem and leaf presets, the root preset). Only the root exchanges with the
soil. The leaf transpires its organ's flow (an Organ-scale input, §27), from its surface node; it reaches it from the
evaporating walls through the air spaces and the stomata (GRANAP §60): what the leaf transpires, the root takes up.
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
TRANSPIRATION = 0.05                                     # cm3 d-1, the leaf organ's


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
    for organ in ds.entity_ids("Organ"):                      # the leaf organ's transpiration, an MTG property
        leaf = g.property("label").get(int(organ)) == g.labels.Organ.Leaf
        g.property("leaf_transpiration")[int(organ)] = TRANSPIRATION if leaf else 0.
    mecha = MechaHydraulics(data_structure=ds)                # (read when its inputs are registered)
    mecha.collar_condition = 0.
    mecha()
    labels = g.property("label")
    kinds = {int(v): labels[int(v)] for v in ds.entity_ids("SubOrgan")}
    return g, ds, mecha, kinds


def _organ_nodes(g, ds, kinds, label):
    owners = ds.entity_ids("SubOrgan")[ds.owner("SubOrgan")]
    return np.isin(owners, [v for v, k in kinds.items() if k == label])


def test_only_the_root_meets_the_soil_and_the_leaf_has_its_vapour_path(plant):
    g, ds, mecha, kinds = plant
    L = g.labels.SubOrgan
    root, leaf = _organ_nodes(g, ds, kinds, L.RootSegment), _organ_nodes(g, ds, kinds, L.LeafElement)
    soil = ds.get("soil_conductance")
    assert (soil[root] > 0).any() and (soil[~root] == 0).all()
    kind, path, K = ds.get("kind"), ds.get("network_path"), ds.get("conductance")
    surface = np.flatnonzero(kind == 3)
    assert surface.size == 1 and ds.get("surface_inflow")[surface[0]] == pytest.approx(-TRANSPIRATION)
    stomatal = path == 6
    assert stomatal.sum() == (ds.get("tissue") == "pore").sum() > 0 and (K[stomatal] > 0).all()
    assert (K[path == 4] > 0).all() and (ds.get("air_tail_conductance")[path == 1] > 0).all()
    vapour = ds.get("vapour_node") > 0
    assert vapour[surface].all() and leaf[vapour & (kind != 3)].all()


def test_root_uptake_is_transpired(plant):
    """The organ's transpiration leaves through its stomata, from what the root takes up."""
    g, ds, mecha, kinds = plant
    uptake, evaporation = ds.get("root_uptake").sum(), ds.get("evaporation").sum()
    assert uptake == pytest.approx(TRANSPIRATION, rel=1e-8) and evaporation == pytest.approx(TRANSPIRATION, rel=1e-8)
    pores = ds.get("tissue") == "pore"
    assert (ds.get("evaporation")[pores] > 0).all() and (ds.get("evaporation")[~pores] == 0).all()


def test_potentials_along_the_path(plant):
    """Liquid potentials below the soil's, the air spaces' (Kelvin) below the walls', the leaf surface the driest."""
    g, ds, mecha, kinds = plant
    psi, water = ds.get("psi"), ds.get("water_potential")
    vapour = ds.get("vapour_node") > 0
    assert np.isfinite(water).all() and psi[~vapour].max() <= SOIL + 1e-6
    x = psi[vapour]
    assert ((x > 0) & (x <= 1)).all()                                   # relative vapour pressures
    surface = ds.get("kind") == 3
    assert water[surface][0] <= water[vapour].min() + 1e-6
    assert water[surface][0] < psi[~vapour].min()


def test_root_quantities_on_roots_only(plant):
    g, ds, mecha, kinds = plant
    L = g.labels.SubOrgan
    for i, v in enumerate(ds.entity_ids("SubOrgan")):
        root = kinds[int(v)] == L.RootSegment
        assert np.isfinite(ds.get("perimeter")[i]) == root and np.isfinite(ds.get("kr")[i]) == root
        assert np.isfinite(ds.get("Kx")[i])
