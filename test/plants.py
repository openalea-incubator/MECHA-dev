"""
The plants of the tests: a root of one or more segments (lengths in m, as metafspm's plant models), its GRANAP
anatomies, and the DataStructure over their Compartments (anatomy mode).
"""
from openalea.metafspm.coupling.component import StructuralComponent
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.scene.population import Planting, planting, population_data_structure

from openalea.granap.metafspm.component import GranapAnatomy, junction_rules
from openalea.granap.metafspm.library import AnatomyLibrary, preset_parameters

RADIUS = 0.55e-3                     # m: GRANAP's monocot root preset
HEIGHT = 200e-6                      # m: the legacy's default maturity height, 200 µm


def root(n_segments=1, length=HEIGHT, radius=RADIUS):
    """A root axis of *n_segments* segments, as an architecture model would build it."""

    class Root(StructuralComponent):
        @classmethod
        def initiate_plant(cls, g, plant, parameters):
            s, L = g.scales, g.labels
            complex_ = plant
            for scale, label in ((s.Axis, L.Axis.Root), (s.GrowthUnit, L.GrowthUnit.Root),
                                 (s.Phytomer, L.Phytomer.Root), (s.Organ, L.Organ.RootInternode)):
                complex_ = g.add_component(complex_, **PropsConfig(scale=scale, edge_type='/', label=label))
            segment = g.add_component(complex_, **PropsConfig(scale=s.SubOrgan, edge_type='/',
                                                              label=L.SubOrgan.RootSegment))
            for rank in range(n_segments):
                if rank:
                    segment = g.add_child(segment, **PropsConfig(scale=s.SubOrgan, edge_type='<',
                                                                 label=L.SubOrgan.RootSegment))
                g.set_values("radius", [segment], [radius])
                g.set_values("length", [segment], [length])
                g.set_values("distance_from_tip", [segment], [(n_segments - 1 - rank) * length])

    return Root


class Anatomy(GranapAnatomy):
    """GRANAP's anatomy component, its default library (the monocot root preset)."""


def anatomy_component(preset=None):
    """GRANAP's anatomy component on an OrganInputData preset ("for_dicot_root"...), None: the monocot root."""
    if preset is None:
        return Anatomy
    return type(f"Anatomy_{preset}", (GranapAnatomy,), {"library": AnatomyLibrary(parameters=preset_parameters(preset))})


def initiated(table, *initiators, time_step: float = 3600.):
    """The plants of *table* built by *initiators* (an architecture, then GRANAP's anatomies), their Compartments
    wired at the segments' junctions, in one MPGDataStructure: metafspm's population_data_structure with a model of no
    component (the tests build theirs on population.data_structure)."""
    def __init__(self, data_structure, time_step):
        self.components = []

    model = type("Initiated", (), dict(initiators=tuple(initiators), nodes="Compartment",
                                       wiring=staticmethod(junction_rules), __init__=__init__, run=lambda self: None))
    return population_data_structure(table, model, time_step=time_step)


def anatomy_data_structure(architecture, anatomy_class=Anatomy):
    """The plant, its GRANAP anatomies, and the DataStructure over their Compartments: (g, ds, anatomy)."""
    ds = initiated(planting([Planting()], None), architecture, anatomy_class).data_structure
    g = ds.structure
    anatomy = anatomy_class(data_structure=ds)
    anatomy()
    return g, ds, anatomy


def template_organ(anatomy, g, segment):
    """The GRANAP organ the segment's anatomy was copied from (its template's, the population's: g.objects)."""
    from openalea.granap.metafspm.library import template_by_index
    return template_by_index(g, int(g.property("anatomy_template")[segment])).organ


def needle(length=HEIGHT):
    """A needle: one leaf element (a SubOrgan LeafElement), *length* long, its anatomy GRANAP's needle (NeedleAnatomy)."""

    class Needle(StructuralComponent):
        @classmethod
        def initiate_plant(cls, g, plant, parameters):
            s, L = g.scales, g.labels
            complex_ = plant
            for scale, label in ((s.Axis, L.Axis.Shoot), (s.GrowthUnit, L.GrowthUnit.Shoot),
                                 (s.Phytomer, L.Phytomer.Shoot), (s.Organ, L.Organ.Leaf)):
                complex_ = g.add_component(complex_, **PropsConfig(scale=scale, edge_type='/', label=label))
            segment = g.add_component(complex_, **PropsConfig(scale=s.SubOrgan, edge_type='/',
                                                              label=L.SubOrgan.LeafElement))
            g.set_values("radius", [segment], [0.6e-3])
            g.set_values("length", [segment], [length])
            g.set_values("distance_from_tip", [segment], [0.])

    return Needle


def needle_anatomy_component(seed=42):
    """GRANAP's anatomy component with its needle for leaf elements (the reactions branch's NeedleAnatomy(seed=42))."""
    from openalea.granap.metafspm.component import OrganAnatomy
    from openalea.granap.metafspm.library import preset_parameters
    from openalea.granap.needle_class import NeedleAnatomy
    shoot = {"LeafElement": OrganAnatomy(AnatomyLibrary(parameters=preset_parameters("for_needle"),
                                                        organ_class=NeedleAnatomy, seed=seed), size=None)}
    return type(f"NeedleAnatomy_{seed}", (GranapAnatomy,), {"shoot": shoot})
