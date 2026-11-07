"""
The seedling of metafspm's soil–plant–atmosphere example (examples/soil_plant_atmosphere), its anatomies populated by
GRANAP and its hydraulics by MECHA (devplan_metafspm.md, §15):

    SeedlingArchitecture   the architecture of metafspm's seedling (a stem of three phytomers, each with a leaf of
                           three elements; three root axes from the collar, a pivot with two laterals and two oblique
                           roots with one lateral each), with each segment's length, radius, distance from its tip and
                           3D extent (x1..z2, for the mapping onto the soil); its anatomies are left to GRANAP
    SeedlingHydraulics     the plant population model: GRANAP's anatomies (roots by the root preset, classes by
                           diameter; stems and leaves by the monocot presets, turned on by the plant's scenario:
                           SEEDLING_SCENARIO), MECHA's stand-ins for a root development model (Development: each root
                           segment's apoplastic barriers from its distance to the tip, none on the tip segment, a
                           Casparian strip up to 10 cm, suberin beyond; the shoot's mature xylem, 1) and for a
                           transpiration model (Transpiration: the plant's, shared by the organs with a leaf
                           surface), MECHA's osmotic presets standing for a solute model (the roots' cells and walls'
                           osmotic potentials, the membranes' reflection coefficients; ψ is then the pressure
                           potential and water_potential = ψ + π), and MECHA's hydraulics
    SEEDLING_SCENARIO      the seedling's scenario: its components' parameters (its barriers, its transpiration)
    SOIL_SCENARIO          metafspm's test soil (Soil) at the example's 2.5 cm cells
    SCENE_TRANSLATOR       the plant–soil exchange with metafspm's test soil, in each model's units: the soil's
                           potential (MPa) to MECHA's soil_psi (hPa), MECHA's root uptake (cm³ d⁻¹) to the soil's sink
                           (mm³ s⁻¹)
"""
from dataclasses import dataclass

import numpy as np

from openalea.metafspm.coupling.component import StructuralComponent, planting_variable, state_variable
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.configs import PropsConfig, ScalesConfig
from openalea.metafspm.testing.soil import Soil

from openalea.granap.metafspm.component import GranapAnatomy, junction_rules
from openalea.mecha.component import MechaHydraulics
from openalea.mecha.osmotic import MechaOsmoticPresets
from openalea.mecha.stand_ins import Development, Transpiration

# metafspm's test soil (openalea.metafspm.testing.soil, not an official soil model), with the 2.5 cm cells of
# metafspm's soil–plant–atmosphere example: the Scene's environment_scenarios entry for Soil
SOIL_SCENARIO = {"voxel": 0.025}

SCALES = ScalesConfig()
ROOT_RADIUS, LATERAL_RADIUS, STEM_RADIUS = 0.55e-3, 0.3e-3, 1.0e-3        # m
MPA_TO_HPA = 1e4
CM3_PER_DAY_TO_MM3_PER_S = 1e3 / 86400.


def _doc(unit, description):
    return dict(unit=unit, unit_comment="", description=description, min_value=-1e12, max_value=1e12,
                value_comment="", references="", DOI=[])


def _segment_coordinate(name):
    return state_variable(**_doc("m", f"the segment's {name}"), default=0., scale=SCALES.SubOrgan, location="node",
                          aggregation="broadcast", kind="descriptor")


def _segment_size(description, kind="intensive"):
    return state_variable(**_doc("m", description), default=0., scale=SCALES.SubOrgan, kind=kind)


def build_architecture(g, plant, parameters):
    """
    metafspm's seedling (its seedling.build_seedling), its architecture only: every segment with its length, radius,
    distance from its tip and 3D extent.
    """
    p = dict(n_phytomers=3, n_leaf_elements=3, n_root_axes=3, n_root_segments=6, n_lateral_segments=4,
             stem_length=0.02, leaf_length=0.03, root_length=0.025, lateral_length=0.02, emergence_angle=15.,
             gravitropism=0.35, lateral_gravitropism=0.08, tortuosity=0.08, pivot_emergence_angle=80.,
             pivot_gravitropism=0.8, n_pivot_segments=8, pivot_lateral_ranks=(2, 5))
    p.update({key: value for key, value in parameters.items() if key in p})
    s, labels = g.scales, g.labels
    x0, y0, z0 = (float(g.view(name)[plant]) for name in ("x", "y", "z"))
    rotation = float(g.view("rotation").get(plant, 0.))
    segments = {}                                    # vid -> (length, radius, start, end)
    chains = []                                      # each axis' segments, base to tip

    def segment(vid, length, radius, start, direction):
        end = np.asarray(start) + length * np.asarray(direction) / np.linalg.norm(direction)
        segments[vid] = (length, radius, np.asarray(start, dtype=float), end)
        return end

    shoot = g.add_component(plant, **PropsConfig(scale=s.Axis, edge_type='/', label=labels.Axis.Shoot))
    unit = g.add_component(shoot, **PropsConfig(scale=s.GrowthUnit, edge_type='/', label=labels.GrowthUnit.Shoot))
    top, phytomer, stem_element, collar, stem = (x0, y0, z0), None, None, None, []
    for rank in range(int(p["n_phytomers"])):
        phytomer = (g.add_component(unit, **PropsConfig(scale=s.Phytomer, edge_type='/', label=labels.Phytomer.Shoot))
                    if phytomer is None else
                    g.add_child(phytomer, **PropsConfig(scale=s.Phytomer, edge_type='<', label=labels.Phytomer.Shoot)))
        internode = g.add_component(phytomer, **PropsConfig(scale=s.Organ, edge_type='/',
                                                             label=labels.Organ.StemInternode))
        props = PropsConfig(scale=s.SubOrgan, edge_type='<' if stem_element else '/', label=labels.SubOrgan.StemElement)
        stem_element = (g.add_component_with_topo(internode, stem_element, **props) if stem_element
                        else g.add_component(internode, **props))
        collar = collar or stem_element
        stem.append(stem_element)
        top = segment(stem_element, p["stem_length"], STEM_RADIUS, top, (0., 0., 1.))
        leaf = g.add_child(internode, **PropsConfig(scale=s.Organ, edge_type='+', label=labels.Organ.Leaf))
        azimuth = rotation + rank * 2. * np.pi / 3.
        start, parent, blade = top, stem_element, []
        for element in range(int(p["n_leaf_elements"])):
            props = PropsConfig(scale=s.SubOrgan, edge_type='+' if element == 0 else '<',
                                label=labels.SubOrgan.LeafElement)
            vid = (g.add_component_with_topo(leaf, parent, **props) if element == 0 else g.add_child(parent, **props))
            start = segment(vid, p["leaf_length"], STEM_RADIUS, start, (np.cos(azimuth), np.sin(azimuth),
                                                                       0.6 - 0.4 * element))
            parent = vid
            blade.append(vid)
        chains.append(blade)
    chains.append(stem)

    rng = np.random.default_rng(int(plant))

    def tropism(direction, coefficient):
        bent = np.asarray(direction) + coefficient * np.array([0., 0., -1.]) + rng.normal(0., p["tortuosity"], 3)
        return bent / np.linalg.norm(bent)

    for axis_rank in range(int(p["n_root_axes"])):
        axis = g.add_child(shoot, **PropsConfig(scale=s.Axis, edge_type='+', label=labels.Axis.Root))
        unit_r = g.add_component(axis, **PropsConfig(scale=s.GrowthUnit, edge_type='/', label=labels.GrowthUnit.Root))
        phytomer_r = g.add_component(unit_r, **PropsConfig(scale=s.Phytomer, edge_type='/',
                                                           label=labels.Phytomer.Root))
        organ = g.add_component(phytomer_r, **PropsConfig(scale=s.Organ, edge_type='/',
                                                          label=labels.Organ.RootInternode))
        pivot = axis_rank == 0
        others = max(int(p["n_root_axes"]) - 1, 1)
        azimuth = rotation + (axis_rank - 1) * 2. * np.pi / others + rng.normal(0., 0.25)
        dip = np.radians(p["pivot_emergence_angle"] if pivot else p["emergence_angle"]) + rng.normal(0., 0.1)
        gravitropism = p["pivot_gravitropism"] if pivot else p["gravitropism"]
        direction = np.array([np.cos(azimuth) * np.cos(dip), np.sin(azimuth) * np.cos(dip), -np.sin(dip)])
        start, parent, axis_segments = (x0, y0, z0), collar, []
        for rank in range(int(p["n_pivot_segments"] if pivot else p["n_root_segments"])):
            props = PropsConfig(scale=s.SubOrgan, edge_type='+' if rank == 0 else '<',
                                label=labels.SubOrgan.RootSegment)
            vid = (g.add_component_with_topo(organ, parent, **props) if rank == 0 else g.add_child(parent, **props))
            start = segment(vid, p["root_length"], ROOT_RADIUS, start, direction)
            direction = tropism(direction, gravitropism)
            parent = vid
            axis_segments.append(vid)
        chains.append(axis_segments)
        bearers = p["pivot_lateral_ranks"] if pivot else (2,)
        for number, bearer_rank in enumerate(bearers):
            bearer = axis_segments[min(int(bearer_rank), len(axis_segments) - 1)]
            lateral = g.add_child(organ, **PropsConfig(scale=s.Organ, edge_type='+',
                                                       label=labels.Organ.RootInternode))
            side = azimuth + np.pi / 2. + number * np.pi + rng.normal(0., 0.3)
            direction = np.array([np.cos(side), np.sin(side), -0.2])
            direction /= np.linalg.norm(direction)
            start, parent, lateral_segments = segments[bearer][3], bearer, []
            for rank in range(int(p["n_lateral_segments"])):
                props = PropsConfig(scale=s.SubOrgan, edge_type='+' if rank == 0 else '<',
                                    label=labels.SubOrgan.RootSegment)
                vid = (g.add_component_with_topo(lateral, parent, **props) if rank == 0
                       else g.add_child(parent, **props))
                start = segment(vid, p["lateral_length"], LATERAL_RADIUS, start, direction)
                direction = tropism(direction, p["lateral_gravitropism"])
                parent = vid
                lateral_segments.append(vid)
            chains.append(lateral_segments)

    for name in ("length", "radius", "distance_from_tip", "x1", "x2", "y1", "y2", "z1", "z2"):
        g.properties().setdefault(name, {})
    for vid, (length, radius, start, end) in segments.items():
        for name, value in (("length", length), ("radius", radius), ("x1", start[0]), ("x2", end[0]),
                            ("y1", start[1]), ("y2", end[1]), ("z1", start[2]), ("z2", end[2])):
            g.set_values(name, [vid], [float(value)])
        if g.view("label")[vid] == g.labels.SubOrgan.StemElement:     # GRANAP's stems' class input
            g.set_values("stem_radius", [vid], [float(radius)])
    for chain in chains:                                               # distance from each axis' tip
        lengths = [segments[v][0] for v in chain]
        for k, vid in enumerate(chain):
            g.set_values("distance_from_tip", [vid], [float(sum(lengths[k + 1:]))])


@dataclass
class SeedlingArchitecture(StructuralComponent):
    """The architecture initiator: metafspm's seedling, at its planting's position and rotation, its segments'
    geometry (x1..z2 seen by their Compartments), the sizes GRANAP's anatomies and MECHA read."""
    x: float = planting_variable("x")
    y: float = planting_variable("y")
    z: float = planting_variable("z")
    rotation: float = planting_variable("rotation")
    length: float = _segment_size("the segment's length")
    radius: float = _segment_size("the segment's radius")
    stem_radius: float = _segment_size("a stem element's radius (0: not a stem element)", kind="descriptor")
    distance_from_tip: float = _segment_size("the segment's distance from its axis' tip")
    x1: float = _segment_coordinate("start, x")
    x2: float = _segment_coordinate("end, x")
    y1: float = _segment_coordinate("start, y")
    y2: float = _segment_coordinate("end, y")
    z1: float = _segment_coordinate("start, z")
    z2: float = _segment_coordinate("end, z")

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        build_architecture(g, plant, parameters)


# The plant's transpiration (mm3 s-1), split equally between its leaf organs: a constant for now (a transpiration
# model will give each leaf's, plan §26–27). 0.04 mm3 s-1 is 2 mmol m-2 s-1, a well-watered cereal seedling's daytime
# rate, over its 10.8 cm2 of leaf blade (9 leaf elements of 3 cm, GRANAP's monocot leaf 4.0 mm wide).
TRANSPIRATION = 0.04
MM3_PER_S_TO_CM3_PER_DAY = 1e-3 * 86400.
# The root segments' barriers by their distance from the tip (Development): none on the tip segment, a Casparian
# strip up to 10 cm, suberin beyond
BARRIERS = {"tip_zone": 0., "tip_barrier": 0., "young_zone": 0.1, "young_barrier": 1., "mature_barrier": 3.}
SEEDLING_SCENARIO = {"stem_anatomy": {}, "leaf_anatomy": {},   # the shoot's anatomies, its presets
                     "collar_condition": 0.,                   # the shoot above the roots: no collar
                     **BARRIERS,
                     "transpiration": TRANSPIRATION * MM3_PER_S_TO_CM3_PER_DAY}     # Transpiration's (cm3 d-1)


class SeedlingHydraulics(CompositeModel):
    """
    The plant population model: the seedlings' architecture and GRANAP's anatomies (the initiators; the Compartments
    are the graph's nodes, the vessels joined between sections by GRANAP's junction rules), the barriers'
    (Development) and the leaves' transpiration (Transpiration: its plant's, shared by the organs with a leaf
    surface), MECHA's osmotic presets and hydraulics, from the soil around the roots to the air at the leaves' stomata.
    """
    initiators = (SeedlingArchitecture, GranapAnatomy)
    from_scale, nodes, wiring = "SubOrgan", "Compartment", junction_rules

    parts = {"architecture": SeedlingArchitecture, "anatomy": GranapAnatomy, "development": Development,
             "transpiration": Transpiration, "osmotic": MechaOsmoticPresets,   # MECHA's presets as "Solutes" (§20,
             "hydraulics": MechaHydraulics}                                    # Q-O3) until a metabolism is coupled
    schedule = ("development", "anatomy", "transpiration", "osmotic", "hydraulics")

    def __init__(self, data_structure, time_step):
        super().__init__(data_structure, time_step)
        # the roots' soil contacts (MECHA's entity set): the soil links' mapping maps them only
        data_structure.define_mask("root_surface", entities="MechaHydraulics.soil_contacts")


# the plants and the soil exchange through the roots' surface only (the mask root_surface, by the overlap of its
# sections with the cells): the Scene's mapping of these links
ROOT_SURFACE = {"mask": "root_surface", "method": "overlap"}
SCENE_TRANSLATOR = (
    Translator()
    .link("MechaHydraulics", "soil_psi", "SoilWaterTransport", {"water_potential": MPA_TO_HPA}, mapping=ROOT_SURFACE)
    .link("SoilWaterTransport", "plant_uptake", "MechaHydraulics", {"root_uptake": CM3_PER_DAY_TO_MM3_PER_S},
          mapping=ROOT_SURFACE)
)


def largest_change(tolerance=1e-4):
    """
    A stop_when condition: the largest change of the soil's and the plants' water potentials between two steps is
    below *tolerance* (MPa; MECHA's ψ in hPa). The history of the changes is kept on the condition.
    """
    previous = {}

    def condition(scene):
        change = 0.
        for key, (ds, _, _) in scene.data_structures().items():
            if ds.has("psi"):
                values = np.array(ds.get("psi"), dtype=float).reshape(-1) / MPA_TO_HPA
            elif ds.has("water_potential"):
                values = np.array(ds.get("water_potential"), dtype=float).reshape(-1)
            else:
                continue
            if key in previous and previous[key].shape == values.shape:
                change = max(change, float(np.abs(values - previous[key]).max()))
            else:
                change = np.inf
            previous[key] = values
        condition.history.append(change)
        return change < tolerance

    condition.history = []
    return condition
