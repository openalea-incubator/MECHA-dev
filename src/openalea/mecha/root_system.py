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

from openalea.metafspm.coupling.component import (FunctionalComponent, input_variable, setting, state_variable,
                                                   upscaling)
from openalea.metafspm.solve.decorator import totalstate
from openalea.metafspm.solve.network import equivalent_conductance
from openalea.mecha.component import SCALES
from openalea.mecha.geometry import SOIL, SOIL_NODE

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
    plant_solver: str = setting(description="the solve of Krs' network (as MechaHydraulics' plant problem's): "
                                            "\"full\", one sparse solve; \"condensed\", each section's interior "
                                            "eliminated (solve.network.condensed_solve)",
                                default="full", choices=("full", "condensed"))
    network_path: float = input_variable(**_input("", "the Connection's kind of path (7: to a soil node, not part of "
                                                  "the root's network)", "edge", SCALES.Connection))
    kind: float = input_variable(**dict(_input("", "the Compartment's kind (4: a root section's soil node, GRANAP's "
                                               "soil_node)", "node", SCALES.Compartment), by="Anatomy"),
                                 kind="descriptor")

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

    @totalstate(include_inactive=True, memo=True)
    def _root_system_conductance(self, conductance, xylem_axial_conductance, soil_conductance, collar_kind,
                                 network_path, kind) -> dict:
        """Krs and the SUF of each plant: one solve of its network at a unit potential difference, from its root
        surface (each surface node joined to the soil through its soil conductance) to its collar's vessels. A soil
        node's Connections (GRANAP's soil_node) are not part of the root: MARSHAL's protocol puts the soil at the
        surface. Computed again only when its network changed (memo=True)."""
        ds = self.data_structure
        collar = np.asarray(collar_kind) == 1.
        if not collar.any():
            raise ValueError("RootSystemConductance: no collar at a potential (collar_condition 1)")
        network = np.asarray(conductance) + np.asarray(xylem_axial_conductance)
        soil_nodes = np.asarray(kind) == SOIL_NODE
        if soil_nodes.any():                       # each section's soil node held at the soil (its contacts in the
            sources, through = soil_nodes, None    # network, or in its reduced section's interior)
        else:                                      # each surface node joined to the soil through its conductance
            sources, through = np.asarray(soil_conductance) > 0., soil_conductance
            network = np.where(np.asarray(network_path) == SOIL, 0., network)
        reduction = ds.reduction                   # its plant problem's system (MechaHydraulics' or a subclass's)
        system = next((name for name in reduction.systems if name.endswith("._plant_problem")), None) \
            if reduction is not None and not reduction.library else None
        terms = reduction.terms(self, system, ds, ds.entity_ids("node")) if system is not None else None
        total, inflow = equivalent_conductance(ds.to_graph_view(), network, sources=sources, sinks=collar,
                                               source_conductance=through,
                                               groups=ds.owner("SubOrgan") if self.plant_solver == "condensed" else None,
                                               cache=self.caches().setdefault("krs_condensation", {}),
                                               extra=terms[0] if terms is not None else None)
        per_plant = ds.map_to(inflow, "node", "Plant", "sum")
        plant_total = ds.map_to(per_plant, "Plant", "node", "broadcast")
        share = np.divide(inflow, plant_total, out=np.zeros_like(inflow), where=plant_total > 0.)
        return {"root_system_conductance": per_plant,
                "standard_uptake_fraction": ds.map_to(share, "node", "SubOrgan", "sum")}
