"""
The shoot in MECHA's plant problem: a stem element bearing a leaf element and a
root segment, GRANAP's anatomies (monocot stem and leaf presets, the root preset). Only the root exchanges with the
soil. The leaf transpires its organ's flow (an Organ-scale input, §27), from its surface node; it reaches it from the
evaporating walls through the air spaces and the stomata (GRANAP §60): what the leaf transpires, the root takes up.
The transpiring organs are those GRANAP gave a leaf surface, not a label: Adel's blade and sheath transpire alike.
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


def test_the_liquid_volume_weighs_a_leaf_element_s_water_potential(plant):
    """liquid_volume is the nodes' volume but on the vapour nodes (the air spaces, the pores, the surface): weighted by
    it, a leaf element's mean water potential is its liquid's (walls and protoplasts), not the air's Kelvin values."""
    g, ds, mecha, kinds = plant
    leaf = _organ_nodes(g, ds, kinds, g.labels.SubOrgan.LeafElement)
    liquid, volume = ds.get("liquid_volume"), ds.get("node_volume")
    vapour = ds.get("vapour_node") > 0
    assert (liquid[vapour] == 0).all() and np.array_equal(liquid[~vapour], volume[~vapour])
    water = ds.get("water_potential")
    mean = np.sum(water[leaf] * liquid[leaf]) / np.sum(liquid[leaf])
    assert water[leaf & ~vapour].min() <= mean <= water[leaf & ~vapour].max()


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
    stem and root sections reduced (plant_reduction: their conduits and the root's soil node kept), the leaf whole: the
    scenario's entries "reduced" and "reduced_where" (scenario_reduction), one model class for both."""
    from openalea.metafspm.coupling.composite_wrapper import CompositeModel, Part
    from openalea.metafspm.scene.population import population_data_structure
    from openalea.granap.metafspm.component import junction_rules
    from openalea.mecha.component import scenario_reduction

    class SeedlingModel(CompositeModel):
        from_scale, nodes, initiators = "SubOrgan", "Compartment", (Seedling, Anatomy)
        wiring = staticmethod(junction_rules)
        reduction = staticmethod(scenario_reduction)

        parts = {"anatomy": Anatomy, "soil": UniformSoil, "transpiration": Transpiration,
                 "hydraulics": Part(MechaHydraulics, soil_boundary="node_dirichlet")}
        schedule = ("soil", "anatomy", "transpiration", "hydraulics")

    scenario = dict(scenario, collar_condition=0., uniform_soil_psi=SOIL, transpiration=TRANSPIRATION)
    if reduced:                                  # the exactness's reference: a kind per length (no classes)
        scenario.update(reduced=True, reduced_where={"label": list(labels)},
                        length_classes=scenario.get("length_classes", "exact"))
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


def test_a_reduced_section_s_kept_nodes_hold_its_volumes_and_its_liquid_mean():
    """Every section reduced (the root, the stem, the leaf): its kept nodes hold the volumes of the nodes they condense
    (the Kron reduction's recovery weights), so each section's node_volume and liquid_volume are the full anatomy's,
    and each section's liquid-weighted mean water potential the full one's (a leaf element's: its stomata's)."""
    full, _ = _seedling(instanced_anatomy=True, soil_node=True)
    small, _ = _seedling(reduced=True, labels=("RootSegment", "StemElement", "LeafElement"), soil_node=True)
    assert small.n_nodes() < full.n_nodes() / 3
    sections = range(len(full.entity_ids("SubOrgan")))

    def per_section(ds, values):
        owner = np.asarray(ds.owner("SubOrgan"))
        return np.array([float(np.sum(np.asarray(values)[owner == i])) for i in sections])
    for name in ("node_volume", "liquid_volume"):
        assert (per_section(small, small.get(name)) > 0).all()
        np.testing.assert_allclose(per_section(small, small.get(name)), per_section(full, full.get(name)), rtol=1e-12)
    for ds in (full, small):
        assert (ds.get("liquid_volume")[ds.get("vapour_node") > 0] == 0).all()

    def liquid_mean(ds):
        liquid = np.asarray(ds.get("liquid_volume"))
        return per_section(ds, liquid * np.asarray(ds.get("water_potential"))) / per_section(ds, liquid)
    np.testing.assert_allclose(liquid_mean(small), liquid_mean(full), rtol=1e-8)


def test_the_condensed_volumes_are_each_kinds_derived_values():
    """The kept nodes' condensed volumes per unit height are computed once per kind, on its library while its
    factorisation is live (metafspm's kind_derived, kron.shares: two single solves), and kept in the kind's outputs:
    the same as the dense recovery weights (the interior recovered for each kept node's unit potential, its rows over
    the liquid kept nodes normalised), and a kind evaluated again gives them again."""
    small, hydraulics = _seedling(reduced=True, labels=("RootSegment", "StemElement", "LeafElement"), soil_node=True)
    reduction = hydraulics._reduction()
    system = hydraulics._plant_problem_name(reduction)
    groups = reduction.reduced_groups(small)
    kinds = reduction.group_kinds(small, groups)
    g = small.structure
    vapour = np.asarray(small.get("vapour_node")) > 0
    P_node = hydraulics.parameter_table("node")
    for i, (owner, key) in groups.items():
        kind = kinds[i][system]
        first, _ = g.instances.block(owner)
        kept = small.index_of(first + g.instances.templates[key].rows_at_scale(g.scales.Compartment))
        full = reduction.full_template(key)
        unit = hydraulics._template_unit_volumes(full, (float(P_node["thickness_source"][kept[0]]),
                                                         float(P_node["thickness"][kept[0]])))
        compartments = full.rows_at_scale(g.scales.Compartment)
        kept_positions = np.searchsorted(compartments, kind.kept_rows)
        interior_positions = np.searchsorted(compartments, kind.interior_rows)
        targets = ~vapour[kept]
        expected = unit[kept_positions].copy()
        if interior_positions.size and targets.any():
            weights = np.asarray(kind.reduction.recover(np.eye(kept_positions.size)))[:, targets]
            total = weights.sum(axis=1)
            weights = np.where((total > 0)[:, None], weights / np.where(total > 0, total, 1.)[:, None],
                               1. / targets.sum())
            expected[targets] += weights.T @ unit[interior_positions]
        np.testing.assert_allclose(kind.outputs["condensed_unit_volume"], expected, rtol=1e-10, atol=1e-30)
    volumes = np.array(small.get("node_volume"))
    for kind in [k for by_system in reduction._kinds.values() for k in by_system.values()]:
        kind.release()                                   # evaluated again: new kind objects, the same values
    reduction._kinds.clear()
    hydraulics.__dict__.pop("_step_memos", None)         # the conductances computed again (a memoized step)
    hydraulics()
    np.testing.assert_allclose(small.get("node_volume"), volumes, rtol=1e-12)


class BladeAndSheath(StructuralComponent):
    """Adel's leaf: an internode's stem element bearing a sheath (an Organ Sheath, its element a StemElement, GRANAP's
    sheath anatomy: a bent leaf) and the sheath its blade (an Organ Blade, a LeafElement), and a root segment."""

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
        sheath = g.add_component_with_topo(organ("Shoot", "Shoot", "Sheath"), stem,
                                           **PropsConfig(scale=s.SubOrgan, edge_type='+', label=L.SubOrgan.StemElement))
        blade = g.add_component_with_topo(organ("Shoot", "Shoot", "Blade"), sheath,
                                          **PropsConfig(scale=s.SubOrgan, edge_type='/', label=L.SubOrgan.LeafElement))
        root = g.add_component_with_topo(organ("Root", "Root", "RootInternode"), stem,
                                         **PropsConfig(scale=s.SubOrgan, edge_type='+', label=L.SubOrgan.RootSegment))
        for v, radius in ((stem, 1e-3), (sheath, 1e-3), (blade, 1e-3), (root, 0.55e-3)):
            g.property("stem_radius")[v] = 1e-3
            g.property("radius")[v] = radius
            g.property("length")[v] = 5e-3
            g.property("distance_from_tip")[v] = 0.


def test_a_blade_and_a_sheath_both_transpire():
    """Adel's blade and sheath, each given a leaf surface by GRANAP (the sheath's kind Sheath, GRANAP's sheath
    anatomy), are both transpiring organs: Transpiration shares the plant's between them, MechaHydraulics takes each
    organ's as a Neumann outflow on its surface node, and each organ's pores evaporate it; the root takes up the sum."""
    from openalea.metafspm.coupling.composite_wrapper import CompositeModel
    from openalea.metafspm.scene.population import population_data_structure
    from openalea.granap.metafspm.component import junction_rules

    class AdelLeafModel(CompositeModel):
        from_scale, nodes, initiators = "SubOrgan", "Compartment", (BladeAndSheath, GranapAnatomy)
        wiring = staticmethod(junction_rules)

        parts = {"anatomy": GranapAnatomy, "soil": UniformSoil, "transpiration": Transpiration,
                 "hydraulics": MechaHydraulics}
        schedule = ("soil", "anatomy", "transpiration", "hydraulics")

    scenario = {"stem_anatomy": {}, "leaf_anatomy": {}, "sheath_anatomy": {}, "collar_condition": 0.,
                "uniform_soil_psi": SOIL, "transpiration": TRANSPIRATION}
    population = population_data_structure(planting([Planting()], AdelLeafModel, scenario), AdelLeafModel)
    ds = population.data_structure
    population.instance.run()
    organs = ds.labels(location="Organ", per_node=False)
    blade, sheath = list(organs).index("Blade"), list(organs).index("Sheath")
    expected = np.zeros(len(organs))
    expected[[blade, sheath]] = TRANSPIRATION / 2
    np.testing.assert_array_equal(np.asarray(ds.get("leaf_transpiration")), expected)
    kind, organ = np.asarray(ds.get("kind")), np.asarray(ds.owner("Organ"))
    surface = np.flatnonzero(kind == 3)
    assert sorted(organ[surface].tolist()) == sorted([blade, sheath])
    np.testing.assert_allclose(np.asarray(ds.get("surface_inflow"))[surface], -TRANSPIRATION / 2, rtol=1e-12)
    assert (np.asarray(ds.get("surface_inflow"))[kind != 3] == 0).all()
    evaporation = np.asarray(ds.get("evaporation"))
    for transpiring in (blade, sheath):
        assert evaporation[organ == transpiring].sum() == pytest.approx(TRANSPIRATION / 2, rel=1e-8)
    assert float(np.sum(ds.get("root_uptake"))) == pytest.approx(TRANSPIRATION, rel=1e-8)
    assert evaporation.sum() == pytest.approx(TRANSPIRATION, rel=1e-8)


def test_the_reduction_s_scenario_entries_are_accepted_with_the_reduction_off():
    """scenario_reduction owns its scenario entries (scenario_keys): with "reduced" false, its labels and classes are
    inactive entries of the plant model, not unknown ones; the seedling is whole."""
    from openalea.mecha.component import scenario_reduction
    assert set(scenario_reduction.scenario_keys) == {"reduced", "reduced_where", "length_classes", "osmotic_reduction",
                                                     "leaf_ports", "vapour_path"}
    ds, hydraulics = _seedling(reduced=False, soil_node=True, reduced_where={"label": "RootSegment"},
                               length_classes="exact")
    assert ds.reduction is None and hydraulics._reduction() is None
