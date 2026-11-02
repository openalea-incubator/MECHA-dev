"""
The shoot in MECHA's plant problem: a stem element bearing a leaf element and a
root segment, GRANAP's anatomies (monocot stem and leaf presets, the root preset). Only the root exchanges with the
soil. The leaf transpires its organ's flow (an Organ-scale input, §27), from its surface node; it reaches it from the
evaporating walls through the air spaces and the stomata (GRANAP §60): what the leaf transpires, the root takes up.
"""
import numpy as np
import pytest

from openalea.metafspm.coupling.component import StructuralComponent
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.scene.population import Planting, planting

from openalea.granap.leaf_class import LeafAnatomy
from openalea.granap.stem_class import StemAnatomy
from openalea.granap.metafspm.component import GranapAnatomy, OrganAnatomy
from openalea.granap.metafspm.library import AnatomyLibrary, preset_parameters
from openalea.mecha.component import MechaHydraulics
from openalea.mecha.stand_ins import Transpiration, UniformSoil

from plants import initiated

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
        g.property("stem_radius")[stem] = 1e-3                     # GRANAP's stems' class input
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
    ds = initiated(planting([Planting()], None), Seedling, Anatomy, time_step=3600.).data_structure
    g = ds.structure
    Anatomy(data_structure=ds)()
    ds.register("barrier", np.ones(ds.n_nodes()), location="node")
    ds.register("soil_psi", np.full(ds.n_nodes(), SOIL), location="node")
    Transpiration(data_structure=ds, transpiration=TRANSPIRATION)()       # its one leaf organ's
    mecha = MechaHydraulics(data_structure=ds)
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


def test_the_leaf_organ_transpires_its_plant_s_transpiration(plant):
    """Transpiration's share: the plant's on its one leaf organ, none on the stem's and the root's."""
    _, ds, _, _ = plant
    leaf = ds.labels(location="Organ", per_node=False) == "Leaf"
    np.testing.assert_array_equal(np.asarray(ds.get("leaf_transpiration")), np.where(leaf, TRANSPIRATION, 0.))


def test_root_quantities_on_roots_only(plant):
    g, ds, mecha, kinds = plant
    L = g.labels.SubOrgan
    for i, v in enumerate(ds.entity_ids("SubOrgan")):
        root = kinds[int(v)] == L.RootSegment
        assert np.isfinite(ds.get("perimeter")[i]) == root and np.isfinite(ds.get("kr")[i]) == root
        assert np.isfinite(ds.get("Kx")[i])


def _seedling(reduced=False, labels=("RootSegment", "StemElement"), **scenario):
    """The seedling as a plant model (GRANAP's anatomies and MechaHydraulics, the leaf transpiring); *reduced*: its
    stem and root sections reduced (plant_reduction: their conduits and the root's soil node kept), the leaf whole."""
    from openalea.metafspm.coupling.composite_wrapper import CompositeModel
    from openalea.metafspm.scene.population import population_data_structure
    from openalea.granap.metafspm.component import junction_rules
    from openalea.mecha.component import plant_reduction

    class SeedlingModel(CompositeModel):
        from_scale, nodes, initiators = "SubOrgan", "Compartment", (Seedling, Anatomy)
        wiring = staticmethod(junction_rules)
        reduction = plant_reduction(labels=labels) if reduced else None

        def __init__(self, data_structure, time_step):
            self.anatomy = Anatomy(data_structure=data_structure, time_step=time_step)
            self.soil = UniformSoil(data_structure=data_structure, time_step=time_step)
            self.transpiration = Transpiration(data_structure=data_structure, time_step=time_step)
            self.hydraulics = MechaHydraulics(data_structure=data_structure, time_step=time_step,
                                              soil_boundary="node_dirichlet")
            self.couple(self.anatomy, self.soil, self.transpiration, self.hydraulics)

        def run(self):
            self.soil()
            self.transpiration()
            self.anatomy()
            self.hydraulics()

    scenario = dict(scenario, collar_condition=0., uniform_soil_psi=SOIL, transpiration=TRANSPIRATION)
    population = population_data_structure(planting([Planting()], SeedlingModel, scenario), SeedlingModel)
    ds = population.data_structure
    population.instance.run()
    return ds, population.instance.hydraulics


def test_a_condensed_stem_and_root_with_a_whole_leaf_are_the_full_seedling():
    """A plant model's reduction on the shoot too: the stem element reduced onto its conduits, the root segment onto
    its conduits and soil node, the transpiring leaf element whole (its vapour path, an explicit term): the
    potentials of every node they share with the full seedling, the uptake (the transpiration), as the full one's."""
    full, _ = _seedling(instanced_anatomy=True, soil_node=True)
    small, _ = _seedling(reduced=True, soil_node=True)
    assert small.n_nodes() < full.n_nodes() / 3

    def places(ds):
        node = ds.structure.values_of("template_node", ds.entity_ids("node"))
        return [(int(o), node.get(int(v), -int(v))) for o, v in zip(ds.owner("SubOrgan"), ds.entity_ids("node"))]
    row = {place: i for i, place in enumerate(places(full))}
    shared = [(i, row[place]) for i, place in enumerate(places(small)) if place in row]
    b, a = np.array(shared).T
    labels = full.labels(location="SubOrgan")
    assert {"StemElement", "LeafElement", "RootSegment"} <= set(labels[a].tolist())
    np.testing.assert_allclose(small.get("psi")[b], full.get("psi")[a], rtol=1e-9)
    for ds in (full, small):
        assert float(np.sum(ds.get("root_uptake"))) == pytest.approx(TRANSPIRATION, rel=1e-8)


def test_a_reduced_leaf_keeps_its_vapour_path_and_gives_the_full_seedling():
    """The leaf element reduced too: its conduits and its vapour path kept (its air spaces, pores and evaporating
    walls; its wall-air and air-link Connections explicit, their conductances computed on the reduced section as on
    the full one), the rest eliminated: the same potentials on every shared node, the same transpiration taken up."""
    full, _ = _seedling(instanced_anatomy=True, soil_node=True)
    small, _ = _seedling(reduced=True, labels=("RootSegment", "StemElement", "LeafElement"), soil_node=True)
    leaf = list(full.labels(location="SubOrgan", per_node=False)).index("LeafElement")
    full_leaf = int((np.asarray(full.owner("SubOrgan")) == leaf).sum())
    small_leaf = int((np.asarray(small.owner("SubOrgan")) == leaf).sum())
    assert small_leaf < full_leaf

    def places(ds):
        node = ds.structure.values_of("template_node", ds.entity_ids("node"))
        return [(int(o), node.get(int(v), -int(v))) for o, v in zip(ds.owner("SubOrgan"), ds.entity_ids("node"))]
    row = {place: i for i, place in enumerate(places(full))}
    shared = [(i, row[place]) for i, place in enumerate(places(small)) if place in row]
    b, a = np.array(shared).T
    np.testing.assert_allclose(small.get("psi")[b], full.get("psi")[a], rtol=1e-8)
    for ds in (full, small):
        assert float(np.sum(ds.get("root_uptake"))) == pytest.approx(TRANSPIRATION, rel=1e-8)
