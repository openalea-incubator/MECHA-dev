"""
Stand-ins for the models MECHA's inputs come from, until a plant model couples the real ones: each is a metafspm
component declaring, as its own state, an input of MechaHydraulics, computed from its parameters (set by the plant's
scenario, or by a Scene forcing of the same name):

    Development      the root sections' (segments and apices) apoplastic barriers (barrier, MECHA's maturity code) by their distance from
                     their axis' tip, in up to three zones: mature_barrier everywhere, but up to young_zone
                     (young_barrier) and up to tip_zone (tip_barrier) from the tip
    UniformSoil      a soil uniform around the roots: its water potential (soil_psi) and the bulk soil's conductance
                     on each soil node (soil_bulk_conductance)
    Transpiration    each transpiring organ's transpiration (leaf_transpiration): the plant's, shared equally by the
                     organs the anatomy gave a leaf surface (GRANAP's leaves, Adel's blades and sheaths)

    class RootHydraulics(CompositeModel):
        initiators = (RootAxis, GranapAnatomy)
        ...
        def __init__(self, data_structure, time_step):
            self.anatomy = GranapAnatomy(data_structure=data_structure, time_step=time_step)
            self.development = Development(data_structure=data_structure, time_step=time_step)
            self.soil = UniformSoil(data_structure=data_structure, time_step=time_step)
            self.hydraulics = MechaHydraulics(data_structure=data_structure, time_step=time_step)
            self.couple(self.anatomy, self.development, self.soil, self.hydraulics)

    planting([Planting()], RootHydraulics, {"tip_zone": 0., "tip_barrier": 0., "young_zone": 0.1,
                                            "mature_barrier": 3., "uniform_soil_psi": -1000.})
"""
from dataclasses import dataclass

import numpy as np

from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, parameter, state_variable
from openalea.metafspm.solve.decorator import rate

from openalea.mecha.component import ROOT_LABELS, SCALES, _doc
from openalea.mecha.geometry import LEAF_SURFACE


@dataclass
class Development(FunctionalComponent):
    """
    A root development model's stand-in: each root section's (a root segment's or apex's, ROOT_LABELS) apoplastic
    barriers (MECHA's maturity code) from its distance to its axis' tip, in three zones of its plant's parameters:
    the tip's (up to tip_zone), the young root's (up to young_zone), the mature root's beyond (without zones,
    mature_barrier on every root section). The shoot's segments keep the default, 1 (their mature xylem).
    """
    label: int = input_variable(**_doc("", "the segment's MTG label (the root sections: RootSegment, RootApex)"),
                                by="Architecture", default=0, scale=SCALES.SubOrgan, dtype="int", kind="descriptor")
    distance_from_tip: float = input_variable(**_doc("m", "the segment's distance from its axis' tip"),
                                              by="Architecture", default=0., scale=SCALES.SubOrgan, location="node",
                                              aggregation="broadcast", kind="intensive")
    tip_zone: float = parameter(**_doc("m", "the distance from the tip up to which the segments have tip_barrier "
                                            "(NaN: no tip zone)"), by="Development", default=float("nan"))
    tip_barrier: float = parameter(**_doc("", "the barriers near the tip (e.g. 0: none)"), by="Development",
                                   default=1.)
    young_zone: float = parameter(**_doc("m", "the distance from the tip up to which the other segments have "
                                              "young_barrier (NaN: no young zone)"), by="Development",
                                  default=float("nan"))
    young_barrier: float = parameter(**_doc("", "the barriers of the young root (e.g. 1: a Casparian strip)"),
                                     by="Development", default=1.)
    mature_barrier: float = parameter(**_doc("", "the barriers beyond the young zone (e.g. 3: a suberized "
                                                 "endodermis; every segment's without zones)"), by="Development",
                                      default=1.)
    barrier: float = state_variable(**_doc("", "the segment's apoplastic barriers (MECHA's maturity code: 0 none, 1 "
                                                 "endodermal Casparian strip, 2 + suberin with passage cells, 3 "
                                                 "endodermis fully suberized, 4-9 exodermis variants, 10 exodermis "
                                                 "suberized with a Casparian strip, 11 endodermis and exodermis "
                                                 "suberized and the sclerenchyma's walls lignified)"),
                                    default=1., scale=SCALES.SubOrgan, location="node", aggregation="broadcast",
                                    kind="descriptor")

    @rate(filters={"label": ROOT_LABELS}, include_inactive=True)
    def _barrier(self, distance_from_tip, tip_zone, tip_barrier, young_zone, young_barrier, mature_barrier):
        """Each root section's code, by the zone its distance from the tip is in (an apex's: the tip's)."""
        distance = np.asarray(distance_from_tip, dtype=float)
        return np.where(distance <= tip_zone, tip_barrier, np.where(distance <= young_zone, young_barrier,
                                                                    mature_barrier))


def development_parameters(distances, barriers) -> dict:
    """Development's parameters giving each segment, at its distance from its tip, its barrier code; raises when two
    segments at one distance (a main root's and a lateral's) differ, or when the codes along the root make more than
    three zones (a single code: mature_barrier alone)."""
    codes = {}
    for distance, barrier in zip(distances, barriers):
        if codes.setdefault(float(distance), float(barrier)) != float(barrier):
            raise ValueError(f"barriers: the segments at {distance} m from their tips have different barriers "
                             f"({codes[float(distance)]:g} and {float(barrier):g}): a segment's barriers follow its "
                             "distance from its tip (Development)")
    zones = []                                         # [up to distance, code], from the tip
    for distance in sorted(codes):
        if zones and zones[-1][1] == codes[distance]:
            zones[-1][0] = distance
        else:
            zones.append([distance, codes[distance]])
    if len(zones) > 3:
        raise ValueError(f"barriers: {len(zones)} zones along the root ({[code for _, code in zones]} from the tip), "
                         "Development has three (tip, young, mature)")
    parameters = {"mature_barrier": zones[-1][1]}
    for (distance, code), (zone, barrier) in zip(zones[-2::-1], (("young_zone", "young_barrier"),
                                                                 ("tip_zone", "tip_barrier"))):
        parameters.update({zone: distance, barrier: code})
    return parameters


@dataclass
class UniformSoil(FunctionalComponent):
    """A soil model's stand-in: the soil uniform around every root section, at its plant's uniform_soil_psi, and the
    bulk soil's conductance on each section's soil node (MechaHydraulics' soil_boundary node_robin)."""
    uniform_soil_psi: float = parameter(**_doc("hPa", "the soil's water potential around the roots"), by="Soil",
                                        default=0.)
    uniform_soil_bulk_conductance: float = parameter(**_doc("cm3 hPa-1 d-1", "the bulk soil's conductance between a "
                                                            "root section's soil node and the soil"), by="Soil",
                                                     default=0.)
    soil_psi: float = state_variable(**_doc("hPa", "the soil's water potential around the Compartment"), default=0.,
                                     scale=SCALES.Compartment, location="node", kind="intensive")
    soil_bulk_conductance: float = state_variable(**_doc("cm3 hPa-1 d-1", "on a root section's soil node: the "
                                                         "conductance between it and the soil at soil_psi"),
                                                  default=0., scale=SCALES.Compartment, location="node",
                                                  kind="intensive")

    @rate(include_inactive=True)
    def _soil_psi(self, uniform_soil_psi, uniform_soil_bulk_conductance) -> dict:
        """The plant's values on every Compartment (its parameters come broadcast to them)."""
        return {"soil_psi": np.array(uniform_soil_psi, dtype=float),
                "soil_bulk_conductance": np.array(uniform_soil_bulk_conductance, dtype=float)}


@dataclass
class Transpiration(FunctionalComponent):
    """A transpiration model's stand-in: the plant's transpiration (its scenario's, or a Scene forcing of the same
    name, a time series), shared equally by its transpiring organs: those the anatomy gave a leaf surface (a
    Compartment of kind 3, GRANAP's for its leaf kinds: a leaf's elements, Adel's blades' and sheaths'), the organs
    whose transpiration MechaHydraulics takes as a Neumann flux on that surface."""
    kind: float = input_variable(**_doc("", "the Compartment's kind (3: an organ's leaf surface, GRANAP's)"),
                                 by="Anatomy", default=0., scale=SCALES.Compartment, location="node",
                                 kind="descriptor")
    transpiration: float = parameter(**_doc("cm3 d-1", "the plant's transpiration"), by="Transpiration", default=0.)
    transpiring: float = state_variable(**_doc("", "1 on an organ the anatomy gave a leaf surface (it transpires), 0 "
                                                   "elsewhere"),
                                        default=0., scale=SCALES.Organ, location="Organ", kind="descriptor")
    leaf_transpiration: float = state_variable(**_doc("cm3 d-1", "the organ's transpiration (0: no leaf surface)"),
                                               default=0., scale=SCALES.Organ, location="Organ", kind="extensive")

    @rate(include_inactive=True)
    def _transpiring(self, kind) -> dict:
        """The transpiring organs: those owning a leaf surface node (the anatomy's, so that it runs before)."""
        surface = (np.asarray(kind) == LEAF_SURFACE).astype(float)
        return {"transpiring": (self.data_structure.map_to(surface, "node", "Organ", "sum") > 0).astype(float)}

    @rate(include_inactive=True)
    def _leaf_transpiration(self, transpiring, transpiration) -> dict:
        """Each transpiring organ's share of its plant's transpiration. The parameter comes per Organ (the location of
        the step's output), each its plant's."""
        ds = self.data_structure
        leaf = np.asarray(transpiring) > 0
        leaves = ds.map_to(ds.map_to(leaf, "Organ", "Plant", "sum"), "Plant", "Organ", "broadcast")
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.asarray(transpiration, dtype=float) / leaves
        return {"leaf_transpiration": np.where(leaf, share, 0.)}
