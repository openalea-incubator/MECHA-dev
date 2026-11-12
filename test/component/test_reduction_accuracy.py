"""
The reduction's length classes against its exact kinds (metafspm's devplan_release_optimisation.md §5, O-4): a
seedling as use case 1's plant is on its first days (a stem element, a sheath and its blade, a root axis of
segments of several lengths with a lateral; GRANAP's anatomies, every section reduced) growing for 3 days, its
sections' kinds by length class (length_classes=0.1, the default: a section between two classes their kinds' blend)
or one per length ("exact"): the roots' uptake, the blade's water potential and the root system's conductance (Krs,
root_system) within 0.5 % of the exact kinds'; each class's kind evaluated once. The kinds evaluated in bounded batches, in
the reduction's library kept from one batch to the next, are those of one evaluation. The shoot's lumped kept nodes
(plant_reduction's leaf_ports="bundles" and vapour_path="air_spaces", metafspm's devplan O-7): the uptake and the
blade's water potential within 0.5 % of each conduit and each evaporating wall its own node, on far fewer nodes.
"""
import numpy as np
import pytest

from openalea.metafspm.compute import Compute
from openalea.metafspm.coupling.component import StructuralComponent
from openalea.metafspm.coupling.composite_wrapper import CompositeModel, Part
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.scene.population import Planting, planting, population_data_structure
from openalea.metafspm.solve import reduction as reduction_module

from openalea.granap.metafspm.component import GranapAnatomy, junction_rules
from openalea.mecha.component import MechaHydraulics, scenario_reduction
from openalea.mecha.simulation import root_system
from openalea.mecha.stand_ins import Transpiration, UniformSoil

pytestmark = pytest.mark.slow    # minutes long: run before a release or a merge to main (pytest -m slow)

SOIL = -1000.                                            # hPa
TRANSPIRATION = 0.05                                     # cm3 d-1
LABELS = ("RootSegment", "StemElement", "LeafElement")       # the sheaths' elements are StemElements
ROOT_LENGTHS = (11e-3, 6.1e-3, 4.4e-3, 2.7e-3)          # m, from the collar to the tip
LATERAL_LENGTHS = (3.6e-3,)                              # m, on the main root's first segment
SHOOT_LENGTHS = {"stem": 4.2e-3, "sheath": 17e-3, "blade": 33e-3}
GROWTH = 0.3                                             # d-1: each section's relative elongation rate (the
                                                         # youngest the fastest: GROWTH / (1 + rank / 2)); the
                                                         # stem element's none (no stem extension in 3 days)
STEP, DAYS = 24 * 3600., 3
TOLERANCE = 0.005


class GrowingSeedling(StructuralComponent):
    """The seedling: a stem element bearing a sheath (an Organ Sheath, its element a StemElement) and its blade (a
    LeafElement), and a root axis of segments with a lateral, each section of its length."""

    @classmethod
    def initiate_plant(cls, mtg, plant, parameters):
        s, L = mtg.scales, mtg.labels

        def organ(axis_label, compartment_type, organ_label):
            axis = mtg.add_component(plant, **PropsConfig(scale=s.Axis, edge_type='/', label=getattr(L.Axis, axis_label)))
            unit = mtg.add_component(axis, **PropsConfig(scale=s.GrowthUnit, edge_type='/',
                                                       label=getattr(L.GrowthUnit, compartment_type)))
            phytomer = mtg.add_component(unit, **PropsConfig(scale=s.Phytomer, edge_type='/',
                                                           label=getattr(L.Phytomer, compartment_type)))
            return mtg.add_component(phytomer, **PropsConfig(scale=s.Organ, edge_type='/',
                                                           label=getattr(L.Organ, organ_label)))

        stem = mtg.add_component(organ("Shoot", "Shoot", "StemInternode"),
                               **PropsConfig(scale=s.SubOrgan, edge_type='/', label=L.SubOrgan.StemElement))
        sheath = mtg.add_component_with_topo(organ("Shoot", "Shoot", "Sheath"), stem,
                                           **PropsConfig(scale=s.SubOrgan, edge_type='+', label=L.SubOrgan.StemElement))
        blade = mtg.add_component_with_topo(organ("Shoot", "Shoot", "Blade"), sheath,
                                          **PropsConfig(scale=s.SubOrgan, edge_type='/', label=L.SubOrgan.LeafElement))
        for v, length in ((stem, SHOOT_LENGTHS["stem"]), (sheath, SHOOT_LENGTHS["sheath"]),
                          (blade, SHOOT_LENGTHS["blade"])):
            mtg.set_values("stem_radius", [v], [1e-3])
            mtg.set_values("radius", [v], [1e-3])
            mtg.set_values("length", [v], [length])
            mtg.set_values("distance_from_tip", [v], [0.])
        root_organ = organ("Root", "Root", "RootInternode")
        segments, previous = [], stem
        for rank, length in enumerate(ROOT_LENGTHS):
            props = PropsConfig(scale=s.SubOrgan, edge_type='+' if previous is stem else '<',
                                label=L.SubOrgan.RootSegment)
            previous = (mtg.add_component_with_topo(root_organ, previous, **props) if previous is stem
                        else mtg.add_child(previous, **props))
            segments.append(previous)
            mtg.set_values("radius", [previous], [0.55e-3])
            mtg.set_values("length", [previous], [length])
            mtg.set_values("distance_from_tip", [previous], [sum(ROOT_LENGTHS[rank + 1:])])
        for bearer, length in zip(segments, LATERAL_LENGTHS):
            lateral = mtg.add_component_with_topo(organ("Root", "Root", "RootInternode"), bearer,
                                                **PropsConfig(scale=s.SubOrgan, edge_type='+',
                                                              label=L.SubOrgan.RootSegment))
            mtg.set_values("radius", [lateral], [0.25e-3])
            mtg.set_values("length", [lateral], [length])
            mtg.set_values("distance_from_tip", [lateral], [0.])


class SeedlingModel(CompositeModel):
    from_scale, nodes, initiators = "SubOrgan", "Compartment", (GrowingSeedling, GranapAnatomy)
    wiring = staticmethod(junction_rules)
    reduction = staticmethod(scenario_reduction)

    parts = {"anatomy": GranapAnatomy, "soil": UniformSoil, "transpiration": Transpiration,
             "hydraulics": Part(MechaHydraulics, soil_boundary="node_dirichlet")}
    schedule = ("soil", "anatomy", "transpiration", "hydraulics")


def _seedling(length_classes, **options):
    scenario = {"stem_anatomy": {}, "leaf_anatomy": {}, "sheath_anatomy": {}, "soil_node": True,
                "instanced_anatomy": True, "collar_condition": 0., "uniform_soil_psi": SOIL,
                "transpiration": TRANSPIRATION, "reduced": True, "reduced_where": {"label": list(LABELS)},
                "length_classes": length_classes, **options}
    return population_data_structure(planting([Planting()], SeedlingModel, scenario), SeedlingModel, time_step=STEP)


def _grow(population, initial, hours) -> None:
    """Each growing section's length after *hours* of growth (from *initial*, {SubOrgan vid: length}), the youngest
    the fastest, the others' unchanged: on the MPG and the DataStructure."""
    data_structure = population.data_structure
    mtg = data_structure.structure
    vids = data_structure.entity_ids("SubOrgan").tolist()
    lengths = {}
    for rank, vid in enumerate(sorted(initial, key=lambda v: initial[v])):
        rate = GROWTH / (1. + (len(initial) - 1 - rank) / 2.)
        lengths[vid] = initial[vid] * np.exp(rate * hours / 24.)
        mtg.set_values("length", [vid], [lengths[vid]])
    per_group = np.array([lengths.get(int(v), float(mtg.view("length")[int(v)])) for v in vids])
    if data_structure.location("length") == "node":
        data_structure.set("length", per_group[np.asarray(data_structure.owner("SubOrgan"))])
    else:
        data_structure.set("length", per_group)


def _measures(data_structure) -> tuple:
    """(each root section's uptake, the blade's liquid-weighted water potential)."""
    owner = np.asarray(data_structure.owner("SubOrgan"))
    labels = np.asarray(data_structure.labels(location="SubOrgan", per_node=False))
    uptake = np.bincount(owner, weights=np.asarray(data_structure.get("root_uptake"), dtype=float), minlength=labels.size)
    blade = labels[owner] == "LeafElement"
    liquid, psi = np.asarray(data_structure.get("liquid_volume")), np.asarray(data_structure.get("psi"))
    return uptake[labels == "RootSegment"], float((liquid * psi)[blade].sum() / liquid[blade].sum())


class _Counter:
    def __init__(self, monkeypatch):
        self.kinds, self.keys = 0, []
        evaluate = reduction_module._evaluate

        def counted(reduction, parameters, requests):
            self.kinds += len(requests)
            self.keys += list(requests)
            return evaluate(reduction, parameters, requests)
        monkeypatch.setattr(reduction_module, "_evaluate", counted)


def test_the_length_classes_give_the_exact_kinds_uptake_and_leaf_potential_within_half_a_percent(monkeypatch):
    counter = _Counter(monkeypatch)
    results, keys = {}, {}
    for length_classes in ("exact", 0.1):
        population = _seedling(length_classes)
        mtg = population.data_structure.structure
        initial = {int(v): float(mtg.view("length")[int(v)]) for v in population.data_structure.entity_ids("SubOrgan")
                   if mtg.label(mtg.complex(int(v))) != mtg.labels.Organ.StemInternode}
        counter.kinds, counter.keys, series = 0, [], []
        for step in range(int(DAYS * 24 * 3600 / STEP) + 1):
            _grow(population, initial, step * STEP / 3600.)
            population.instance.run()
            series.append(_measures(population.data_structure))
        results[length_classes], keys[length_classes] = series, list(counter.keys)
    for (uptake, leaf), (exact_uptake, exact_leaf) in zip(results[0.1], results["exact"]):
        assert np.abs(uptake - exact_uptake).max() <= TOLERANCE * np.abs(exact_uptake).sum()
        assert abs(leaf - exact_leaf) <= TOLERANCE * abs(exact_leaf - SOIL)
        assert exact_uptake.sum() == pytest.approx(TRANSPIRATION, rel=1e-8)
    assert len(set(keys[0.1])) == len(keys[0.1])                  # each class's kind evaluated once
    length = list(scenario_reduction({"reduced": True}).archetype_variables).index("length")
    classes = {key[1][length] for key in keys[0.1]}
    width = np.log1p(0.1)
    np.testing.assert_allclose(np.log(sorted(classes)) / width, np.round(np.log(sorted(classes)) / width),
                               atol=1e-9)                              # on the classes (1.1)^k


def _nodes_by_label(data_structure) -> dict:
    labels = np.asarray(data_structure.labels(location="SubOrgan", per_node=False))
    nodes = np.bincount(np.asarray(data_structure.owner("SubOrgan")), minlength=labels.size)
    return {label: int(nodes[labels == label].sum()) for label in LABELS}


def test_the_shoots_bundles_and_air_spaces_give_the_uptake_and_leaf_potential_within_half_a_percent():
    """leaf_ports="bundles" and vapour_path="air_spaces" (the shoot's conduits per vascular bundle, a leaf's
    evaporating walls per air space: each one node) against each conduit and wall its own node, the same 3 days'
    growth: the roots' uptake and the blade's water potential within 0.5 % (of the transpiration, of the blade's
    drop below the soil); the stem's, sheath's and blade's kept nodes a quarter of theirs, the roots' the same."""
    results, nodes = {}, {}
    for name, options in (("cells", {}), ("lumped", {"leaf_ports": "bundles", "vapour_path": "air_spaces"})):
        population = _seedling(0.1, **options)
        mtg = population.data_structure.structure
        initial = {int(v): float(mtg.view("length")[int(v)]) for v in population.data_structure.entity_ids("SubOrgan")
                   if mtg.label(mtg.complex(int(v))) != mtg.labels.Organ.StemInternode}
        series = []
        for step in range(int(DAYS * 24 * 3600 / STEP) + 1):
            _grow(population, initial, step * STEP / 3600.)
            population.instance.run()
            series.append(_measures(population.data_structure))
        results[name], nodes[name] = series, _nodes_by_label(population.data_structure)
    for (uptake, leaf), (cells_uptake, cells_leaf) in zip(results["lumped"], results["cells"]):
        assert np.abs(uptake - cells_uptake).max() <= TOLERANCE * np.abs(cells_uptake).sum()
        assert abs(leaf - cells_leaf) <= TOLERANCE * abs(cells_leaf - SOIL)
        assert uptake.sum() == pytest.approx(TRANSPIRATION, rel=1e-8)
    assert nodes["lumped"]["RootSegment"] == nodes["cells"]["RootSegment"]
    for label in ("StemElement", "LeafElement"):
        assert nodes["lumped"][label] < nodes["cells"][label] / 4


def test_the_length_classes_give_the_exact_kinds_root_system_conductance_within_half_a_percent():
    """Krs (MARSHAL's, root_system) of a condensed root system of segments between two classes."""
    options = dict(main=(3, 3, 1, 1), laterals=((1, (1, 1)),), segment_length=4.3e-3, condensed=True)
    exact = root_system(length_classes="exact", **options)
    classes = root_system(length_classes=0.1, **options)
    assert classes.conductance == pytest.approx(exact.conductance, rel=TOLERANCE)
    np.testing.assert_allclose(classes.sections["suf"], exact.sections["suf"], atol=TOLERANCE)


def test_batches_in_the_kept_library_give_the_kinds_of_one_evaluation():
    """The kinds of the seedling evaluated one per batch (Compute(archetype_batch) of one section's interior cells), each
    batch in the library the former left (its groups replaced), are those of one evaluation in a fresh library."""
    def kinds(population):
        data_structure = population.data_structure
        reduction = data_structure.structure.reduction
        system = population.instance.hydraulics._plant_problem_name(reduction)
        return {i: by_system[system] for i, by_system in reduction.group_archetypes(data_structure, reduction.reduced_groups(data_structure)).items()}

    reference = kinds(_seedling("exact"))
    population = _seedling("exact")
    with Compute(archetype_batch=1).limits():
        batched = kinds(population)
    reduction = population.data_structure.structure.reduction
    assert len(reduction._libraries) == 1
    assert len({compartment_type.key for compartment_type in batched.values()}) == len(reduction._archetypes) > 2
    for i, compartment_type in reference.items():
        np.testing.assert_allclose(batched[i].D, compartment_type.D, rtol=1e-12, atol=1e-14 * np.abs(compartment_type.D).max())
        np.testing.assert_allclose(batched[i].e, compartment_type.e, rtol=1e-12, atol=1e-14 * max(np.abs(compartment_type.e).max(), 1e-300))
        for name, value in compartment_type.outputs.items():
            np.testing.assert_allclose(batched[i].outputs[name], value, rtol=1e-12)



def _grown(population) -> list:
    """The seedling grown for DAYS (_grow, as the tests above): each day's measures."""
    mtg = population.data_structure.structure
    initial = {int(v): float(mtg.view("length")[int(v)]) for v in population.data_structure.entity_ids("SubOrgan")
               if mtg.label(mtg.complex(int(v))) != mtg.labels.Organ.StemInternode}
    series = []
    for step in range(int(DAYS * 24 * 3600 / STEP) + 1):
        _grow(population, initial, step * STEP / 3600.)
        population.instance.run()
        series.append(_measures(population.data_structure))
    return series


def test_growth_and_kinds_pass_the_topology_and_kinds_checks(monkeypatch):
    """The growing seedling (its sections lengthening: new kinds every day, the reduction's library patched with
    their sections in place of the former) under metafspm's checks: every patched structure-derived value equal to
    its build and every kind evaluated equal to its evaluation alone in a fresh library (Compute(checks={"topology",
    "archetypes"})); the results those of the unchecked run, bit for bit."""
    counter = _Counter(monkeypatch)
    with Compute().limits():                                           # the same thread limits as the checked run
        unchecked = _grown(_seedling(0.1))
    assert len(set(counter.keys)) > 2                                  # kinds evaluated as the sections grow
    with Compute(checks={"topology", "archetypes"}).limits():
        checked = _grown(_seedling(0.1))
    for (uptake, leaf), (checked_uptake, checked_leaf) in zip(unchecked, checked):
        np.testing.assert_array_equal(uptake, checked_uptake)
        assert leaf == checked_leaf
