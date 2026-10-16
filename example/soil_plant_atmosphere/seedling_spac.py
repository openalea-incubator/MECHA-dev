"""
The seedling of metafspm's soil–plant–atmosphere example (examples/soil_plant_atmosphere), its anatomies populated by
GRANAP and its hydraulics by MECHA (devplan_metafspm.md, §15):

    SeedlingArchitecture   the architecture of metafspm's seedling (a stem of three phytomers, each with a leaf of
                           three elements; three root axes from the collar, a pivot with two laterals and two oblique
                           roots with one lateral each), with each segment's radius, distance from its tip and 3D
                           extent (x1..z2, for the mapping onto the soil); its anatomies are left to GRANAP
    SeedlingAnatomy        GRANAP's anatomy component: roots by the root preset (classes by diameter and distance from
                           the tip), stems by the monocot stem preset, leaves by the monocot leaf preset
    development            a stand-in for a root development model: each root segment's apoplastic barriers from its
                           distance to the tip (none on the tip segment, a Casparian strip up to 10 cm, suberin
                           beyond); the shoot's mature xylem (1)
    SeedlingHydraulics     the plant population model: GRANAP's and MECHA's components on the seedlings' anatomies,
                           MECHA's osmotic presets standing for a solute model (the roots' cells and walls' osmotic
                           potentials, the membranes' reflection coefficients; ψ is then the pressure potential and
                           water_potential = ψ + π)
    SCENE_TRANSLATOR       the plant–soil exchange with metafspm's soil model, in each model's units: the soil's
                           potential (MPa) to MECHA's soil_psi (hPa), MECHA's root uptake (cm³ d⁻¹) to the soil's sink
                           (mm³ s⁻¹)
"""
import importlib.util
import os
import sys
from dataclasses import dataclass

import numpy as np

import openalea.metafspm
from openalea.metafspm.coupling.component import StructuralComponent, state_variable
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.cross import CrossMapping
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.configs import PropsConfig, ScalesConfig

from openalea.granap.leaf_class import LeafAnatomy
from openalea.granap.stem_class import StemAnatomy
from openalea.granap.metafspm.component import GranapAnatomy, OrganAnatomy, junction_rules
from openalea.granap.metafspm.library import AnatomyLibrary, preset_parameters
from openalea.mecha.component import MechaHydraulics
from openalea.mecha.osmotic import MechaOsmoticPresets

# metafspm's soil model, from its example (examples/soil_plant_atmosphere/models.py)
SPAC = os.path.join(os.path.dirname(openalea.metafspm.__file__), "..", "..", "..", "examples", "soil_plant_atmosphere")
sys.path.insert(0, os.path.abspath(SPAC))
_spec = importlib.util.spec_from_file_location("spac_models", os.path.join(SPAC, "models.py"))
spac_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(spac_models)
Soil = spac_models.Soil

SCALES = ScalesConfig()
ROOT_RADIUS, LATERAL_RADIUS, STEM_RADIUS = 0.55e-3, 0.3e-3, 1.0e-3        # m
MPA_TO_HPA = 1e4
CM3_PER_DAY_TO_MM3_PER_S = 1e3 / 86400.


def _doc(unit, description):
    return dict(unit=unit, unit_comment="", description=description, min_value=-1e12, max_value=1e12,
                value_comment="", references="", DOI=[])


def _segment_coordinate(name):
    return state_variable(**_doc("m", f"the segment's {name}"), initialize=0., scale=SCALES.SubOrgan, location="node",
                          mapping="broadcast", state_variable_type="descriptor")


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
    x0, y0, z0 = (float(g.property(name)[plant]) for name in ("x", "y", "z"))
    rotation = float(g.property("rotation").get(plant, 0.))
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
            g.property(name)[vid] = float(value)
    for chain in chains:                                               # distance from each axis' tip
        lengths = [segments[v][0] for v in chain]
        for k, vid in enumerate(chain):
            g.property("distance_from_tip")[vid] = float(sum(lengths[k + 1:]))


@dataclass
class SeedlingArchitecture(StructuralComponent):
    """The architecture initiator: metafspm's seedling, its segments' geometry (x1..z2 seen by their Compartments)."""
    x1: float = _segment_coordinate("start, x")
    x2: float = _segment_coordinate("end, x")
    y1: float = _segment_coordinate("start, y")
    y2: float = _segment_coordinate("end, y")
    z1: float = _segment_coordinate("start, z")
    z2: float = _segment_coordinate("end, z")

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        build_architecture(g, plant, parameters)


class SeedlingAnatomy(GranapAnatomy):
    """GRANAP's anatomies: roots (default library: the root preset), stems and leaves (monocot presets)."""
    shoot = {"StemElement": OrganAnatomy(AnatomyLibrary(parameters=preset_parameters("for_monocot_stem"),
                                                        organ_class=StemAnatomy)),
             "LeafElement": OrganAnatomy(AnatomyLibrary(parameters=preset_parameters("for_monocot_leaf"),
                                                        organ_class=LeafAnatomy), size=None)}


def development(ds, is_root, tip_segment=1e-9, strip=0.1):
    """
    The barrier code of every Compartment (MECHA's maturity): root segments by their distance from the tip (0 on the
    tip segment, 1 up to *strip* m, 3 beyond), the shoot's 1 (mature xylem).
    """
    distance = np.asarray(ds.get("distance_from_tip"), dtype=float)
    if distance.size != ds.n_nodes():                                  # stored at the segments
        distance = distance[ds.owner("SubOrgan")]
    code = np.where(distance < tip_segment, 0., np.where(distance <= strip, 1., 3.))
    return np.where(is_root, code, 1.)


# the potential of the leaves' air spaces (hPa): RH 97 % at 20 °C (Kelvin's equation, R T / V_w ln RH)
AIR_SPACE_PSI = 8.314 * 293.15 / 1.8e-5 * np.log(0.97) * 1e-2


class SeedlingHydraulics(CompositeModel):
    """
    The plant population model: the seedlings' architecture (initiators), their GRANAP anatomies (anatomy mode: the
    Compartments are the graph's nodes, the vessels joined between sections by GRANAP's junction rules), and MECHA's
    hydraulics on them, from the soil around the roots to the air at the leaves' stomata.
    """
    initiators = (SeedlingArchitecture, SeedlingAnatomy)
    from_scale = "SubOrgan"
    nodes = "Compartment"
    wiring = junction_rules

    def __init__(self, data_structure, time_step):
        self.architecture = SeedlingArchitecture(data_structure=data_structure, time_step=time_step)
        self.anatomy = SeedlingAnatomy(data_structure=data_structure, time_step=time_step)
        self.anatomy()                                                 # the 3D quantities
        if not data_structure.has("barrier"):
            data_structure.register("barrier", np.ones(data_structure.n_nodes()), location="node")
        # MECHA's osmotic presets as the "Solutes" model (§20, Q-O3), until a metabolism model is coupled
        self.osmotic = MechaOsmoticPresets(data_structure=data_structure, time_step=time_step)
        self.hydraulics = MechaHydraulics(data_structure=data_structure, time_step=time_step)
        self.hydraulics.collar_condition = 0.                          # the shoot above the roots
        # the leaves transpire through their stomatal pores (transpiration_model 1) for now: with their air spaces'
        # walls pinned (2, §25), nothing limits the leaves' outflow, and the scene's plant-soil iteration diverges
        # (§25a); AIR_SPACE_PSI is their pinned potential when they are used
        self.hydraulics.transpiration_model = 1.
        self.hydraulics.air_space_psi = AIR_SPACE_PSI
        self.declare_data_and_couple_components(translator_path=Translator(),
                                                components=(self.architecture, self.anatomy, self.osmotic,
                                                            self.hydraulics))
        # the root surface, as a mask: the soil mapping maps it only
        data_structure.define_mask("root_surface", {"soil_conductance": ">0"})
        self._is_root = None

    def run(self):
        self.anatomy()
        ds = self.hydraulics.data_structure
        if self._is_root is None or self._is_root.size != ds.n_nodes():
            roots = self.hydraulics.section_roles()[1]
            owner_vids = ds.entity_ids("SubOrgan")[ds.owner("SubOrgan")]
            self._is_root = np.isin(owner_vids, list(roots))
        ds.set("barrier", development(ds, self._is_root))
        self.osmotic()
        self.hydraulics()


SCENE_TRANSLATOR = (
    Translator()
    .link("MechaHydraulics", "soil_psi", "SoilWaterTransport", {"water_potential": MPA_TO_HPA})
    .link("SoilWaterTransport", "plant_uptake", "MechaHydraulics", {"root_uptake": CM3_PER_DAY_TO_MM3_PER_S})
)


def root_surface_mappings(scene):
    """The roots' surface (MECHA's soil-contact walls and junctions) onto the soil grid."""
    soil = next(model for model in scene.environment if isinstance(model, Soil))
    return [CrossMapping(population.data_structure, soil.grid, method="overlap", mask="root_surface")
            for population in scene.populations]


def largest_change(tolerance=1e-4):
    """
    A stop_when condition: the largest change of the soil's and the plants' water potentials between two steps is
    below *tolerance* (MPa; MECHA's ψ in hPa). The history of the changes is kept on the condition.
    """
    previous = {}

    def condition(scene):
        change = 0.
        for key, ds in enumerate(scene._all_data_structures()):
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
