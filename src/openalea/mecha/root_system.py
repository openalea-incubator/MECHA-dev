"""
A root system's conductance and its standard uptake fractions, upscaled from MECHA's network of every section
(walls, membranes, plasmodesmata, the xylem vessels and their junctions between sections), in MARSHAL's sense
(Couvreur et al. 2012; Meunier et al. 2020): the soil water potential uniform at the root's surface, the collar's
xylem potential fixed::

    Krs   = Q_collar / (psi_soil - psi_collar)        the root system's conductance (cm3 hPa-1 d-1)
    SUF_i = q_i / Q_collar                            each section's share of the uptake (sums to 1 per plant)

RootSystemConductance computes them from MechaHydraulics' conductances on the same DataStructure, as one linear
solve of that network (metafspm's solve.network.equivalent_conductance), without changing its state: it is a
virtual experiment, MECHA's plant problem under its protocol. Under a uniform soil, without osmotic potentials, the
plant problem's uptake is Krs (psi_soil - psi_collar).
"""
from dataclasses import dataclass

import numpy as np

from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, state_variable, upscaling
from openalea.metafspm.solve.decorator import totalstate
from openalea.metafspm.solve.network import equivalent_conductance
from openalea.mecha.component import SCALES

MARSHAL = "Couvreur et al. 2012, Plant Cell Environ.; Meunier et al. 2020, in silico Plants (MARSHAL)"
NETWORK = ("kw", "kaqp", "kpl", "kmb", "kw_casparian", "kw_suberin_in", "kw_suberin_out", "xylem_axial_conductance")


def _input(unit: str, description: str, location: str, scale) -> dict:
    return dict(unit=unit, description=description, by="MechaHydraulics", default=0., scale=scale,
                location=location)


@dataclass
class RootSystemConductance(FunctionalComponent):
    """The root system's conductance (Krs) and its sections' standard uptake fractions (SUF), upscaled from the
    anatomical network MechaHydraulics assembles; the collar condition must be a potential (collar_condition 1)."""

    conductance: float = input_variable(**_input("cm3 hPa-1 d-1", "the edge's hydraulic conductance", "edge",
                                                 SCALES.Connection))
    xylem_axial_conductance: float = input_variable(**_input("cm3 hPa-1 d-1", "a xylem junction's conductance",
                                                             "edge", SCALES.Connection))
    soil_conductance: float = input_variable(**_input("cm3 hPa-1 d-1", "conductance between a border wall or "
                                                      "junction and the soil", "node", SCALES.Compartment))
    collar_kind: float = input_variable(**_input("", "the collar condition of the node: 1 potential", "node",
                                                 SCALES.Compartment))

    root_system_conductance: float = state_variable(
        unit="cm3 hPa-1 d-1", description="the root system's conductance, Krs", default=np.nan, location="Plant",
        kind="NonInertialIntensive", references=MARSHAL,
        upscaling=upscaling(of=NETWORK, law="network", references=MARSHAL,
                            protocol="the soil water potential uniform at the root's surface, the collar's xylem "
                                     "potential fixed: Krs = Q_collar / (psi_soil - psi_collar)"))
    standard_uptake_fraction: float = state_variable(
        unit="adim", description="the section's standard uptake fraction, SUF: its share of the root system's uptake "
                                 "under Krs' protocol", default=np.nan, scale=SCALES.SubOrgan,
        kind="NonInertialExtensive", references=MARSHAL,
        upscaling=upscaling(of=NETWORK, law="network", references=MARSHAL,
                            protocol="SUF_i = q_i / Q_collar under the protocol of root_system_conductance"))

    @totalstate(include_inactive=True)
    def _root_system_conductance(self, conductance, xylem_axial_conductance, soil_conductance, collar_kind) -> dict:
        """Krs and the SUF of each plant: one solve of its network at a unit potential difference, from its root
        surface (each surface node joined to the soil through its soil conductance) to its collar's vessels."""
        ds = self.data_structure
        collar = np.asarray(collar_kind) == 1.
        if not collar.any():
            raise ValueError("RootSystemConductance: no collar at a potential (collar_condition 1)")
        surface = np.asarray(soil_conductance) > 0.
        total, inflow = equivalent_conductance(ds.to_graph_view(), np.asarray(conductance) +
                                               np.asarray(xylem_axial_conductance), sources=surface, sinks=collar,
                                               source_conductance=soil_conductance)
        plant = np.asarray(ds.owner("Plant"))
        per_plant = np.bincount(plant, weights=inflow, minlength=len(ds.entity_ids("Plant")))
        share = np.divide(inflow, per_plant[plant], out=np.zeros_like(inflow), where=per_plant[plant] > 0.)
        return {"root_system_conductance": per_plant,
                "standard_uptake_fraction": ds.map_to(share, "node", "SubOrgan", "sum")}
