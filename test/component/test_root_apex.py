"""
A root apex (SubOrgan label RootApex, ArchiSimple's axis tip) in MECHA's plant problem: a root section as a segment
(ROOT_LABELS), with GRANAP's anatomy of the root once it has a length: its soil contact, kr, perimeter and uptake;
its barriers Development's by its distance from the tip (ArchiSimple's: from the section's base, an apex's its
length); reduced when the reduction lists RootApex, the same uptake as the full plant. At barrier 0 (a tip zone
reaching the apex), MECHA's xylem is immature (no xylem junction, D3): the apex then takes up almost nothing, its only
axial path the protophloem's.
"""
import numpy as np
import pytest

from openalea.metafspm.coupling.component import StructuralComponent
from openalea.metafspm.coupling.composite_wrapper import CompositeModel, Part
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.scene.population import Planting, planting, population_data_structure

from openalea.granap.metafspm.component import junction_rules
from openalea.mecha.component import MechaHydraulics, scenario_reduction
from openalea.mecha.stand_ins import Development, Transpiration, UniformSoil

from test_shoot import SOIL, TRANSPIRATION, Anatomy

SEGMENT, APEX, RADIUS = 4.5e-3, 2e-3, 0.55e-3          # m
BARRIERS = {"tip_zone": 0., "tip_barrier": 0., "young_zone": 0.1, "young_barrier": 1., "mature_barrier": 3.}
TIP_ZONE_ON_THE_APEX = dict(BARRIERS, tip_zone=APEX)                        # the apex at barrier 0


class SeedlingWithApex(StructuralComponent):
    """A stem element bearing a leaf element and a root of one segment ending in an apex APEX long; their distances
    from the tip ArchiSimple's (from the section's base)."""

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
        apex = g.add_child(root, **PropsConfig(scale=s.SubOrgan, edge_type='<', label=L.SubOrgan.RootApex))
        g.set_values("stem_radius", [stem], [1e-3])
        for vertex, radius, length, distance in ((stem, 1e-3, 5e-3, 0.), (leaf, 1e-3, 5e-3, 0.),
                                                 (root, RADIUS, SEGMENT, SEGMENT + APEX),
                                                 (apex, RADIUS, APEX, APEX)):
            g.set_values("radius", [vertex], [radius])
            g.set_values("length", [vertex], [length])
            g.set_values("distance_from_tip", [vertex], [distance])


def _seedling(reduced=False, barriers=BARRIERS):
    class SeedlingModel(CompositeModel):
        from_scale, nodes, initiators = "SubOrgan", "Compartment", (SeedlingWithApex, Anatomy)
        wiring = staticmethod(junction_rules)
        reduction = staticmethod(scenario_reduction)

        parts = {"anatomy": Anatomy, "barriers": Development, "soil": UniformSoil, "transpiration": Transpiration,
                 "hydraulics": Part(MechaHydraulics, soil_boundary="node_dirichlet")}
        schedule = ("soil", "anatomy", "barriers", "transpiration", "hydraulics")

    scenario = dict(barriers, collar_condition=0., uniform_soil_psi=SOIL, transpiration=TRANSPIRATION,
                    soil_node=True, instanced_anatomy=True)
    if reduced:
        scenario.update(reduced=True, reduced_where={"label": ["RootSegment", "RootApex", "StemElement"]},
                        length_classes="exact")                 # the exactness's reference: a kind per length
    population = population_data_structure(planting([Planting()], SeedlingModel, scenario), SeedlingModel)
    population.instance.run()
    return population.data_structure


@pytest.fixture(scope="module")
def plants():
    return _seedling(), _seedling(reduced=True)


def _section(ds, label):
    return int(np.flatnonzero(ds.labels(location="SubOrgan", per_node=False) == label)[0])


def test_the_apex_is_a_root_section_that_takes_up_water(plants):
    full, _ = plants
    apex, segment = _section(full, "RootApex"), _section(full, "RootSegment")
    owner = np.asarray(full.owner("SubOrgan"))
    soil = full.get("soil_conductance")
    assert (owner == apex).sum() > 0 and (soil[owner == apex] > 0).any()        # its anatomy, its soil contact
    for name in ("kr", "perimeter"):                                              # root quantities
        assert np.isfinite(full.get(name)[apex]) and full.get(name)[apex] > 0
    uptake = np.asarray(full.get("root_uptake"), dtype=float)
    by_section = np.bincount(owner, weights=uptake, minlength=len(full.entity_ids("SubOrgan")))
    assert by_section[apex] > 0 and by_section[segment] > 0
    assert float(uptake.sum()) == pytest.approx(TRANSPIRATION, rel=1e-8)


def test_an_apex_in_the_tip_zone_has_no_barrier_and_no_mature_xylem():
    """The tip zone reaching the apex: its barrier 0 (Development's, by its distance from the tip), MECHA's immature
    xylem: no xylem junction to its segment, so it takes up (almost) nothing; the segment takes up the transpiration."""
    full = _seedling(barriers=TIP_ZONE_ON_THE_APEX)
    barrier, owner = full.get("barrier"), np.asarray(full.owner("SubOrgan"))
    apex, segment = _section(full, "RootApex"), _section(full, "RootSegment")
    assert (barrier[owner == apex] == 0.).all() and (barrier[owner == segment] == 1.).all()
    by_section = np.bincount(owner, weights=np.asarray(full.get("root_uptake"), dtype=float),
                             minlength=len(full.entity_ids("SubOrgan")))
    assert abs(by_section[apex]) < 1e-9 * TRANSPIRATION
    assert by_section[segment] == pytest.approx(TRANSPIRATION, rel=1e-8)


def test_a_reduced_apex_takes_up_the_full_one_s_water(plants):
    full, small = plants
    assert small.n_nodes() < full.n_nodes()
    apex = _section(full, "RootApex")
    reduced_groups = small.reduction.reduced_groups(small)
    assert apex in reduced_groups

    def by_section(ds):
        return np.bincount(np.asarray(ds.owner("SubOrgan")), weights=np.asarray(ds.get("root_uptake"), dtype=float),
                           minlength=len(ds.entity_ids("SubOrgan")))
    np.testing.assert_allclose(by_section(small), by_section(full), rtol=1e-7, atol=1e-12)


def test_an_apex_in_the_tip_zone_has_kr_against_its_protophloem():
    """At barrier 0, MECHA's kr is against the protophloem (the sieve tubes of GRANAP's roots, at standard_psi_sieve):
    defined, positive, its standard uptake negative (the phloem's pressure above the soil's); a reduced apex takes it
    from its kind (kr, standard_uptake, perimeter), the full one's. kr_effective stays NaN there: the plant problem
    gives the tip zone no stele boundary (no xylem junction, the protophloem's potential not imposed)."""
    full, small = _seedling(barriers=TIP_ZONE_ON_THE_APEX), _seedling(reduced=True, barriers=TIP_ZONE_ON_THE_APEX)
    apex = _section(full, "RootApex")
    assert apex in small.reduction.reduced_groups(small)
    assert np.isfinite(full.get("kr")[apex]) and full.get("kr")[apex] > 0 and full.get("standard_uptake")[apex] < 0
    for name in ("kr", "standard_uptake", "perimeter"):
        assert small.get(name)[apex] == pytest.approx(full.get(name)[apex], rel=1e-9)
    assert np.isnan(full.get("kr_effective")[apex]) and np.isnan(small.get("kr_effective")[apex])
