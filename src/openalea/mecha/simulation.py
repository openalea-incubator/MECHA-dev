"""
MECHA for users: the water flow across one root cross section in a few lines, its results as tables.

    from openalea.mecha.simulation import simulate, parameters

    result = simulate()                           a monocot root section (GRANAP's default), MECHA's defaults
    result.summary                                kr, the water uptake, the pathways' shares in the cortex
    result.tissues()                              per tissue: cells, area, mean water potential
    result.pathways()                             each pathway's share of the radial flow, layer by layer
    result.draw("potential")                      the section coloured by water potential ("tissues", "pathways")
    parameters()                                  MECHA's parameters: value, unit, description and bounds
    simulate({"kw": 2.4e-5})                      another cell wall conductivity
    simulate(anatomy={"preset": "for_dicot_root"})    another anatomy (GRANAP's parameters)
    simulate(barrier=3)                           a fully suberized endodermis
    simulate(osmotic={"os_hetero": 4, "osmotic_epi": -6800., ...})   with osmotic potentials (MECHA's presets)
    simulate_root(barriers=(3, 3, 1, 1, 0))       a root axis: its sections from the collar to the tip, joined by
                                                  their xylem, in a soil at soil_psi; a table per section
    simulate(barrier=3, outputs="outputs")        the run saved in ./outputs: run.json (its status), scenario.json,
                                                  Plant/<model>/summaries.csv and frames/ (the solved section, encoding3d)
    seedling = root_system()                      a seedling's root system, a main root and a lateral, one network:
    seedling.conductance, seedling.sections       its conductance (Krs) and its sections' uptake fractions (SUF)
    fit_root_system(observations)                 MECHA's membrane conductivities (kaqp, kmb) fitted to measured Krs,
                                                  with and without the aquaporins inhibited

The section is solved in MECHA's standard problem: the soil at soil_psi around it, its xylem vessels at xylem_psi.
Quantities are in MECHA's units: cm, hPa, days (kr in cm hPa-1 d-1); positions in µm.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.scene.population import Planting, planting
from openalea.metafspm.scene.recorder import SceneRecorder
from openalea.metafspm.scene.scenarios import parameter_table
from openalea.metafspm.scene.scene import Scene
from openalea.granap.metafspm.architectures import RootAxis
from openalea.granap.metafspm.component import GranapAnatomy, junction_rules
from openalea.mecha import plots
from openalea.mecha.component import MechaHydraulics, scenario_reduction
from openalea.mecha.geometry import XYLEM_CGROUPS
from openalea.mecha.osmotic import MechaOsmoticPresets
from openalea.mecha.root_system import RootSystemConductance
from openalea.mecha.stand_ins import Development, UniformSoil, development_parameters

# MECHA's main parameters, by the pathway they act on
MAIN_PARAMETERS = ("kw", "kmb", "kaqp", "kpl", "kw_casparian", "kw_suberin_in", "kw_suberin_out", "thickness",
                   "kaqp_endo_factor", "kaqp_exo_factor", "kaqp_cortex_factor", "kaqp_stele_factor", "ratio_cortex")
CORTEX_RANKS = range(40, 50)                       # MECHA's cell ranks of the cortex layers
# A tomato's young root (develop's tutorials/root_example.py and dicot_tomato.py, Solanum lycopersicum), GRANAP's
# dicot root tuned: a two-armed xylem star, its phloem and cambium, its cell sizes. Their inner and outer cortex
# layers are one cortex of 4 layers here, and the exodermis is kept: GRANAP's anatomy parameters set its sections'
# fields, they neither add layers nor remove one.
TOMATO_ROOT = {
    "preset": "for_dicot_root",
    "aerenchyma": {"aerenchyma_proportion": 0.0},
    "stele": {"thickness": 0.116, "cell_diameter": 0.01, "cell_diameter_center": 0.030},
    "xylem": {"n_vascular_peak": 2, "radius_valley_side": 0.015, "radius_peak_side": 0.058, "arc_peak_side": 0.008,
              "arc_valley_side": 0.015, "vessel_diameter": 0.0166, "vessel_diameter_min": 0.01, "pith_radius": 0.0,
              "gradient_inflection": 0.8, "gradient_steepness": 5, "allow_ellipse": True},
    "phloem": {"sieve_diameter": 0.005, "relative_distance": 0.8, "cluster_height": 0.013, "cluster_width": 0.035},
    "cambium": {"radius_valley_side": 0.017, "radius_peak_side": 0.059, "visible_distance": 0.02,
                "arc_peak_side": 0.013, "arc_valley_side": 0.016, "cell_diameter": 0.005, "cell_width": 0.01},
    "endodermis": {"cell_diameter": 0.015, "cell_width": 0.019},
    "pericycle": {"cell_diameter": 0.0141, "cell_width": 0.0086},
    "cortex": {"cell_diameter": 0.025, "cell_width": 0.024, "n_layers": 4},
    "epidermis": {"cell_diameter": 0.027, "cell_width": 0.0255},
    "inter_cellular_spaces": {"smoothness": 0.05, "tissue": ["cortex"]},
}
# cm hPa-1 d-1 to m s-1 MPa-1: 1e-2 m / 1e-4 MPa / 86400 s
KR_TO_M_PER_S_PER_MPA = 1e-2 / 1e-4 / 86400.


def parameters(values: dict = None, main_only: bool = False) -> pd.DataFrame:
    """MECHA's parameters, one row each: its name, its value (the default, or the one in *values*), its unit, its
    description, its declared bounds (min_value, max_value), its component, and whether it is one of the main
    parameters (those first)."""
    table = parameter_table(MechaHydraulics, values=values, main=MAIN_PARAMETERS, roles=("parameter",))
    table["value"] = pd.to_numeric(table["value"])                   # numbers, for display and arithmetic
    return table[table["main"]].reset_index(drop=True) if main_only else table


class _RootSection(CompositeModel):
    """One root axis: GRANAP's anatomy on each segment, its barriers by their distance from the tip (Development), a
    uniform soil around it (UniformSoil), MECHA's hydraulics (the plant model of developer tutorial 1)."""
    initiators = (RootAxis, GranapAnatomy)
    from_scale, nodes, wiring = "SubOrgan", "Compartment", junction_rules

    def __init__(self, data_structure, time_step):
        self.anatomy = GranapAnatomy(data_structure=data_structure, time_step=time_step)
        self.development = Development(data_structure=data_structure, time_step=time_step)
        self.soil = UniformSoil(data_structure=data_structure, time_step=time_step)
        self.hydraulics = MechaHydraulics(data_structure=data_structure, time_step=time_step)
        self.couple(self.anatomy, self.development, self.soil, self.hydraulics)

    def run(self):
        self.development()
        self.soil()
        self.anatomy()
        self.hydraulics()


class _OsmoticRootSection(CompositeModel):
    """One root axis as _RootSection, with MECHA's osmotic presets (the cells', walls' and soil's osmotic potentials)
    before the hydraulics."""
    initiators = (RootAxis, GranapAnatomy)
    from_scale, nodes, wiring = "SubOrgan", "Compartment", junction_rules

    def __init__(self, data_structure, time_step):
        self.anatomy = GranapAnatomy(data_structure=data_structure, time_step=time_step)
        self.development = Development(data_structure=data_structure, time_step=time_step)
        self.soil = UniformSoil(data_structure=data_structure, time_step=time_step)
        self.presets = MechaOsmoticPresets(data_structure=data_structure, time_step=time_step)
        self.hydraulics = MechaHydraulics(data_structure=data_structure, time_step=time_step)
        self.couple(self.anatomy, self.development, self.soil, self.presets, self.hydraulics)

    def run(self):
        self.development()
        self.soil()
        self.anatomy()
        self.presets()
        self.hydraulics()


@dataclass
class Section:
    """A solved root section: its *summary* (a pandas Series), its *scene* (metafspm's: its DataStructure holds every
    variable of every node and edge of the section's network)."""
    scene: object
    summary: pd.Series
    parameters: dict

    @property
    def data_structure(self):
        return self.scene.populations[0].data_structure

    @property
    def hydraulics(self):
        return self.scene.populations[0].instance.hydraulics

    def frame(self) -> plots.SectionFrame:
        """The section as MECHA's plots draw it (positions, tissues, edges, potentials, flows)."""
        return plots.port_frame(self.hydraulics, int(self.data_structure.entity_ids("SubOrgan")[0]))

    def tissues(self) -> pd.DataFrame:
        """Per tissue: its number of cells, their total area (µm²) and their mean water potential (hPa), from the
        outside in."""
        frame = self.frame()
        cells = np.flatnonzero(frame.kind == 0)
        distance = np.hypot(*(frame.xy[cells] - np.asarray(frame.centre)).T)
        area = np.asarray(self.data_structure.get("cell_area"))[
            self.hydraulics.section_geometries()[int(self.data_structure.entity_ids("SubOrgan")[0])][0]][cells]
        table = pd.DataFrame({"tissue": frame.tissue[cells], "area": area * 1e6, "potential": frame.psi[cells],
                              "distance": distance})
        grouped = table.groupby("tissue").agg(cells=("area", "size"), area=("area", "sum"),
                                              potential=("potential", "mean"), distance=("distance", "mean"))
        return grouped.sort_values("distance", ascending=False).drop(columns="distance")

    def pathways(self) -> pd.DataFrame:
        """Each pathway's share of the radial flow (fractions) at each cell layer's radius (µm), with the layer's
        MECHA cell rank: walls (apoplastic), membranes (transmembrane), plasmodesmata (symplastic)."""
        radius, ranks, flows = plots.pathway_flows(self.frame())
        total = sum(flows.values())
        total = np.where(total == 0, 1., total)
        return pd.DataFrame({"radius": radius, "rank": ranks,
                             **{name: flows[name] / total for name in ("wall", "membrane", "plasmodesmata")}})

    def draw(self, what: str = "potential", ax=None):
        """The section: "potential" (cells coloured by water potential), "tissues", or "pathways" (each pathway's
        share of the radial flow against the distance to the centre); returns the matplotlib axes."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))
        frame = self.frame()
        if what == "potential":
            points = plots.psi_map(ax, frame)
            ax.figure.colorbar(points, ax=ax, label="water potential (hPa)", shrink=0.8)
        elif what == "tissues":
            plots.tissue_map(ax, frame)
        elif what == "pathways":
            plots.pathway_breakdown(ax, frame)
        else:
            raise ValueError(f"draw: what is 'potential', 'tissues' or 'pathways', not {what!r}")
        return ax


def _summary(section: Section) -> pd.Series:
    ds = section.data_structure
    kr = float(np.asarray(ds.get("kr"))[0])
    _, ranks, flows = plots.pathway_flows(section.frame())
    in_cortex = np.isin(ranks, list(CORTEX_RANKS))
    totals = {name: flows[name][in_cortex].sum() for name in flows}
    total = sum(totals.values()) or 1.
    return pd.Series({
        "kr": kr, "kr_SI": kr * KR_TO_M_PER_S_PER_MPA,
        "uptake": float(np.asarray(ds.get("standard_uptake"))[0]),
        "perimeter": float(np.asarray(ds.get("perimeter"))[0]),
        "cells": int(np.sum(np.asarray(ds.get("kind")) == 0)),
        "cortex_wall_share": totals["wall"] / total, "cortex_membrane_share": totals["membrane"] / total,
        "cortex_plasmodesmata_share": totals["plasmodesmata"] / total,
    })


def simulate(parameters: dict = None, anatomy: dict = None, barrier: float = 1., soil_psi: float = 0.,
             xylem_psi: float = -5000., osmotic: dict = None, outputs: str = None, compute=None) -> Section:
    """Solve the water flow across one root cross section: GRANAP's anatomy (*anatomy*: its parameters, e.g.
    {"preset": "for_dicot_root", "cortex": {"n_layers": 3}}; a monocot root by default), MECHA's hydraulics with its
    defaults changed by *parameters* ({name: value}), the apoplastic barriers *barrier* (MECHA's maturity code: 1
    an endodermal Casparian strip, 2 + suberin with passage cells, 3 a fully suberized endodermis, 4–10 the
    exodermis' variants, 11 both suberized and the sclerenchyma's walls lignified; 0, none, is a root tip whose water goes to the protophloem: the standard problem then needs
    parameters={"standard_psi_sieve": ...}), the soil at *soil_psi* and the xylem at *xylem_psi* (hPa). With
    *osmotic* ({name: value} of MECHA's osmotic presets, MechaOsmoticPresets: os_hetero, osmotic_epi, os_cortex,
    osmotic_left_soil...), the cells', walls' and soil's osmotic potentials act across the membranes (their reflection
    coefficients); *soil_psi* and *xylem_psi* are then their pressure potentials. With *outputs* (a folder), the run is
    saved there: run.json (its status, timing and versions), scenario.json (the parameters it ran with),
    Plant/<model>/summaries.csv (the section's state variables) and frames/ (the section before and after its solve,
    encoding3d files, openalea.encoding3d.read reads them).

    The summary: kr, the radial conductivity (cm hPa-1 d-1; kr_SI in m s-1 MPa-1), the water uptake (cm³ d-1), the
    perimeter (cm), the number of cells, and the share of the radial flow carried across the cortex by the walls,
    the membranes and the plasmodesmata.
    *compute*: the Scene's run configuration (metafspm's Compute: its threads, its memory trim; None: Compute())."""
    scenario = dict(parameters or {}, standard_psi_soil=soil_psi, standard_psi_xylem=xylem_psi,
                    mature_barrier=float(barrier))
    if anatomy is not None:
        scenario["root_anatomy"] = dict(anatomy)
    if osmotic is not None:
        scenario.update(osmotic, standard_osmotic=1.)
    recorder = SceneRecorder(outputs, frame_period=1, display_period=1) if outputs is not None else None
    scene = Scene(planting([Planting()], _OsmoticRootSection if osmotic is not None else _RootSection, scenario),
                  recorder=recorder, compute=compute)
    scene.simulate(1)                                         # one solve, its state recorded with outputs
    section = Section(scene=scene, summary=pd.Series(dtype=float), parameters=dict(parameters or {}))
    section.summary = _summary(section)
    return section


def simulate_root(barriers=(3, 3, 3, 1, 1, 0), segment_length: float = 5e-3, radius: float = 0.55e-3,
                  parameters: dict = None, anatomy: dict = None, soil_psi: float = -1000.,
                  collar_psi: float = -5000., outputs: str = None, compute=None) -> pd.DataFrame:
    """Solve the water flow in a root axis (MECHA's plant problem): one section per entry of *barriers* (their
    maturity codes, from the collar to the tip), each *segment_length* m long and of *radius* m, joined by their
    xylem vessels; the soil at *soil_psi* around them, the collar's xylem at *collar_psi* (hPa). Returns a table, one
    row per section: its distance from the collar (mm), barrier, mean xylem potential (hPa), uptake (cm³ d-1), kr
    (its standard problem's) and kr_effective (in the root), both in cm hPa-1 d-1; table.attrs["scene"] is the
    metafspm scene. The barriers are Development's zones (at most three: tip, young, mature), by each section's
    distance from the tip; the soil UniformSoil's. With *outputs* (a folder), the run is saved there as simulate saves it.
    *compute*: the Scene's run configuration (metafspm's Compute: its threads, its memory trim; None: Compute())."""
    n = len(barriers)
    distances = [(n - 1 - i) * segment_length for i in range(n)]            # from the tip, the collar's first
    scenario = dict(parameters or {}, collar_condition=1., collar_psi=collar_psi, uniform_soil_psi=soil_psi,
                    **development_parameters(distances, barriers),
                    root_segments=[[radius, segment_length, distance] for distance in distances])
    if anatomy is not None:
        scenario["root_anatomy"] = dict(anatomy)
    recorder = SceneRecorder(outputs, frame_period=1, display_period=1) if outputs is not None else None
    scene = Scene(planting([Planting()], _RootSection, scenario), recorder=recorder, compute=compute)
    scene.simulate(1)                                         # one solve, its state recorded with outputs
    ds = scene.populations[0].data_structure
    from_collar = np.argsort(-np.asarray(ds.get("distance_from_tip"), dtype=float), kind="stable")
    xylem = np.isin(np.asarray(ds.get("cgroup")), XYLEM_CGROUPS).astype(float)
    xylem_potential = ds.map_to(ds.get("psi"), "node", "SubOrgan", "weighted_mean", weights=xylem)
    table = pd.DataFrame({
        "distance": (np.arange(n) + 0.5) * segment_length * 1e3,
        "barrier": ds.map_to(ds.get("barrier"), "node", "SubOrgan", "max")[from_collar].astype(int),
        "xylem_potential": xylem_potential[from_collar],
        "uptake": np.asarray(ds.get("uptake"))[from_collar], "kr": np.asarray(ds.get("kr"))[from_collar],
        "kr_effective": np.asarray(ds.get("kr_effective"))[from_collar]})
    table.attrs["scene"] = scene
    return table


# ── A root system's conductance (Krs) and its calibration ──

# The experiments root_system knows: {name: function of the parameters' values -> the values it runs with};
# "inhibited": the aquaporins blocked at 90 % (e.g. by azide or HgCl2), their contribution kaqp divided by 10
EXPERIMENTS = {"control": lambda values: dict(values),
               "inhibited": lambda values: dict(values, kaqp=0.1 * values.get("kaqp", MechaHydraulics.kaqp))}


class _RootSystem(CompositeModel):
    """A root system: GRANAP's anatomy on each segment, its barriers (Development), a uniform soil (UniformSoil),
    MECHA's hydraulics on the whole network, and its conductance (RootSystemConductance); its sections reduced by
    its scenario's "reduced" (scenario_reduction: "length_classes", "osmotic_reduction")."""
    initiators = (RootAxis, GranapAnatomy)
    from_scale, nodes, wiring = "SubOrgan", "Compartment", junction_rules
    reduction = staticmethod(scenario_reduction)

    def __init__(self, data_structure, time_step):
        self.anatomy = GranapAnatomy(data_structure=data_structure, time_step=time_step)
        self.development = Development(data_structure=data_structure, time_step=time_step)
        self.soil = UniformSoil(data_structure=data_structure, time_step=time_step)
        self.hydraulics = MechaHydraulics(data_structure=data_structure, time_step=time_step)
        self.conductance = RootSystemConductance(data_structure=data_structure, time_step=time_step)
        self.couple(self.anatomy, self.development, self.soil, self.hydraulics, self.conductance)

    def run(self):
        self.development()
        self.soil()
        self.anatomy()
        self.hydraulics()
        self.conductance()


class _OsmoticRootSystem(CompositeModel):
    """A root system as _RootSystem, with MECHA's osmotic presets before the hydraulics."""
    initiators = (RootAxis, GranapAnatomy)
    from_scale, nodes, wiring = "SubOrgan", "Compartment", junction_rules
    reduction = staticmethod(scenario_reduction)

    def __init__(self, data_structure, time_step):
        self.anatomy = GranapAnatomy(data_structure=data_structure, time_step=time_step)
        self.development = Development(data_structure=data_structure, time_step=time_step)
        self.soil = UniformSoil(data_structure=data_structure, time_step=time_step)
        self.presets = MechaOsmoticPresets(data_structure=data_structure, time_step=time_step)
        self.hydraulics = MechaHydraulics(data_structure=data_structure, time_step=time_step)
        self.conductance = RootSystemConductance(data_structure=data_structure, time_step=time_step)
        self.couple(self.anatomy, self.development, self.soil, self.presets, self.hydraulics, self.conductance)

    def run(self):
        self.development()
        self.soil()
        self.anatomy()
        self.presets()
        self.hydraulics()
        self.conductance()


@dataclass
class RootSystem:
    """A solved root system: its *scene* (metafspm's), the *parameters* it ran with, and its sections' layout."""
    scene: object
    parameters: dict
    layout: pd.DataFrame

    @property
    def data_structure(self):
        return self.scene.populations[0].data_structure

    @property
    def conductance(self) -> float:
        """Krs, the root system's conductance (cm3 hPa-1 d-1): its uptake per hPa between the soil, uniform around
        it, and the collar's xylem."""
        return float(np.asarray(self.data_structure.get("root_system_conductance"))[0])

    @property
    def sections(self) -> pd.DataFrame:
        """One row per section: its axis ("main", "lateral 1", ...), its rank on the axis from its base, its barrier,
        its standard uptake fraction (SUF: its share of the uptake under a uniform soil), its uptake (cm³ d-1) and its
        kr (cm hPa-1 d-1, the section alone)."""
        ds = self.data_structure
        built = np.argsort(ds.entity_ids("SubOrgan"), kind="stable")    # the sections as RootAxis built them
        return self.layout.assign(suf=np.asarray(ds.get("standard_uptake_fraction"))[built],
                                  uptake=np.asarray(ds.get("uptake"))[built], kr=np.asarray(ds.get("kr"))[built])

    def update(self, parameters: dict) -> "RootSystem":
        """The same root system with MECHA's *parameters* changed ({name: value}), solved again (its anatomy kept)."""
        hydraulics = self.scene.populations[0].instance.hydraulics
        for name, value in parameters.items():
            setattr(hydraulics, name, value)
        self.parameters = dict(self.parameters, **parameters)
        self.scene.run()
        return self


def root_system(main=(3, 3, 1, 1), laterals=((1, (1, 1)),), segment_length: float = 5e-3,
                main_radius: float = 0.5e-3, lateral_radius: float = 0.25e-3, parameters: dict = None,
                anatomy: dict = None, soil_psi: float = -1000., collar_psi: float = -5000.,
                experiment: str = "control", instanced: bool = False, plant_solver: str = "full",
                soil_boundary: str = "per_node", soil_bulk_conductance: float = 0., condensed: bool = False,
                length_classes="exact", osmotic: dict = None, compute=None) -> RootSystem:
    """
    A seedling's root system, every section a GRANAP anatomy and all of them one MECHA network joined by their xylem:
    a main root of len(*main*) segments (their barriers, MECHA's maturity codes, from the collar to the tip) and its
    *laterals*, ((bearer segment's index on the main root, their barriers from their base), ...); segments
    *segment_length* m long, of *main_radius* or *lateral_radius* m; the soil at *soil_psi* around them, the collar's
    xylem at *collar_psi* (hPa). A segment's barriers follow its distance from its tip (Development's three zones):
    the main root's and the laterals' segments at one distance have the same. *parameters*: MECHA's, changed from
    its defaults; *experiment*: one of EXPERIMENTS ("inhibited": the aquaporins blocked); *instanced*: each
    segment's anatomy an instance of its class's template (GRANAP's instanced_anatomy: no vertex of its own, the
    same results); *plant_solver*: MechaHydraulics'
    ("condensed": each section's interior eliminated, one factorisation per kind of section); *soil_boundary*:
    MechaHydraulics' ("node_dirichlet", "node_robin": one soil node per section, GRANAP's soil_node), with
    *soil_bulk_conductance* (cm3 hPa-1 d-1) on each soil node (node_robin); *condensed*: each section reduced onto
    its conduits and soil node (plant_reduction, solve.reduction; a soil node boundary, node_dirichlet by default),
    its kind evaluated at its length or its class (*length_classes*), the scenario's entries "reduced" and
    "length_classes" (scenario_reduction); *osmotic*: MECHA's osmotic presets with these parameters ({name: value},
    MechaOsmoticPresets'; {}: its defaults), reduced sections through their kinds.

    Its conductance (Krs, cm3 hPa-1 d-1) and its sections' standard uptake fractions are MARSHAL's (the soil
    uniform, the collar fixed), computed on the whole anatomical network. *compute*: the Scene's run configuration
    (metafspm's Compute: its threads, its memory trim; None: Compute()).
    """
    if experiment not in EXPERIMENTS:
        raise ValueError(f"root_system(experiment=...): one of {sorted(EXPERIMENTS)}, got {experiment!r}")
    if condensed and soil_boundary == "per_node":
        soil_boundary = "node_dirichlet"
    values = EXPERIMENTS[experiment](dict(parameters or {}))
    n = len(main)
    rows = [("main", rank, barrier, main_radius, (n - 1 - rank) * segment_length) for rank, barrier in enumerate(main)]
    branched = []
    for k, (bearer, barriers) in enumerate(laterals, start=1):
        if not 0 <= bearer < n:
            raise ValueError(f"root_system: lateral {k} on segment {bearer} of a main root of {n}")
        m = len(barriers)
        branched.append([bearer, [[lateral_radius, segment_length, (m - 1 - i) * segment_length] for i in range(m)]])
        rows += [(f"lateral {k}", rank, barrier, lateral_radius, (m - 1 - rank) * segment_length)
                 for rank, barrier in enumerate(barriers)]
    layout = pd.DataFrame(rows, columns=["axis", "rank", "barrier", "radius", "distance_from_tip"])
    scenario = dict(values, collar_condition=1., collar_psi=collar_psi, root_laterals=branched,
                    instanced_anatomy=bool(instanced), plant_solver=plant_solver, soil_boundary=soil_boundary,
                    soil_node=soil_boundary != "per_node" or condensed, **dict(osmotic or {}),
                    uniform_soil_psi=soil_psi,
                    **development_parameters(layout["distance_from_tip"], layout["barrier"]),
                    root_segments=[[main_radius, segment_length, (n - 1 - i) * segment_length] for i in range(n)])
    if soil_boundary == "node_robin":
        scenario["uniform_soil_bulk_conductance"] = soil_bulk_conductance
    if anatomy is not None:
        scenario["root_anatomy"] = dict(anatomy)
    if condensed:                                  # the plant model's reduction (scenario_reduction)
        scenario.update(reduced=True, length_classes=length_classes, osmotic_reduction=osmotic is not None)
    scene = Scene(planting([Planting()], _OsmoticRootSystem if osmotic is not None else _RootSystem, scenario),
                  recorder=SceneRecorder(), compute=compute)
    ds = scene.populations[0].data_structure
    if len(ds.entity_ids("SubOrgan")) != len(layout):
        raise RuntimeError("root_system: the sections do not match the layout")
    scene.run()
    return RootSystem(scene=scene, parameters=values,
                      layout=layout.drop(columns="distance_from_tip").assign(radius=layout["radius"] * 1e3))


@dataclass
class Fit:
    """A calibration's result: the fitted *values* ({name: value}), the observations *compared* with the model at
    them, and the objective's *history* (every evaluation: its values, residuals and cost)."""
    values: dict
    compared: pd.DataFrame
    history: list


def fit_root_system(observations, parameters=("kaqp", "kmb"), start: dict = None, tolerance: float = 1e-8,
                    **options) -> Fit:
    """
    Fit MECHA's *parameters* so that the root system (root_system(**options)) matches the *observations* (a table:
    variable, e.g. root_system_conductance, its value and optionally its experiment, "control" or "inhibited", its
    uncertainty, its unit; metafspm's read_observations), within their declared bounds, from *start* ({name: value};
    MECHA's defaults by default), by scipy's least squares (*tolerance*: its ftol, xtol and gtol). Without an uncertainty, an observation's residual is its
    relative difference (the uncertainty is its value). Each experiment's root system is built once, then solved
    again for each trial.
    """
    from scipy.optimize import least_squares

    from openalea.metafspm.scene.calibration import compare, objective, read_observations

    observations = read_observations(observations)
    observations["experiment"] = observations["experiment"].map(lambda e: "control" if e is None else e)
    observations["uncertainty"] = [abs(value) if u is None else u for value, u in
                                   zip(observations["value"], observations["uncertainty"])]
    built = {}

    def run(values, experiment):
        if experiment not in built:
            built[experiment] = root_system(parameters=values, experiment=experiment, **options)
        else:
            built[experiment].update(EXPERIMENTS[experiment](values))
        return built[experiment].scene

    start = {name: (start or {}).get(name, getattr(MechaHydraulics, name)) for name in parameters}
    cost = objective(run, list(parameters), observations, declared=[MechaHydraulics],
                     scale=[getattr(MechaHydraulics, name) for name in parameters])
    fitted = least_squares(cost, x0=np.array([start[name] for name in parameters]) / cost.scale, bounds=cost.bounds,
                           x_scale="jac", ftol=tolerance, xtol=tolerance, gtol=tolerance)
    values = cost.values(fitted.x)
    cost(fitted.x)                                       # the root systems left at the fitted values
    return Fit(values=values, compared=compare({e: s.scene for e, s in built.items()}, observations),
               history=cost.history)
