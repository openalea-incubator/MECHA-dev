"""
MECHA for users: the water flow across one root cross section in a few lines, its results as tables.

    from openalea.mecha.simulation import simulate, parameters

    result = simulate()                           a monocot root section (GRANAP's default), MECHA's defaults
    result.summary                                kr, the water uptake, the pathways' shares in the cortex
    result.tissues()                              per tissue: cells, area, mean water potential
    result.pathways()                             each pathway's share of the radial flow, layer by layer
    result.draw("potential")                      the section coloured by water potential ("tissues", "pathways")
    parameters()                                  MECHA's parameters: value, unit and description
    simulate({"kw": 2.4e-5})                      another cell wall conductivity
    simulate(anatomy={"preset": "for_dicot_root"})    another anatomy (GRANAP's parameters)
    simulate(barrier=3)                           a fully suberized endodermis
    simulate_root(barriers=(3, 3, 1, 1, 0))       a root axis: its sections from the collar to the tip, joined by
                                                  their xylem, in a soil at soil_psi; a table per section

The section is solved in MECHA's standard problem: the soil at soil_psi around it, its xylem vessels at xylem_psi.
Quantities are in MECHA's units: cm, hPa, days (kr in cm hPa-1 d-1); positions in µm.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.scene.scene import Scene
from openalea.mecha import plots
from openalea.mecha.component import MechaHydraulics

# MECHA's main parameters, by the pathway they act on
MAIN_PARAMETERS = ("kw", "kmb", "kaqp", "kpl", "kw_casparian", "kw_suberin_in", "kw_suberin_out", "thickness",
                   "kaqp_endo_factor", "kaqp_exo_factor", "kaqp_cortex_factor", "kaqp_stele_factor", "ratio_cortex")
CORTEX_RANKS = range(40, 50)                       # MECHA's cell ranks of the cortex layers
# cm hPa-1 d-1 to m s-1 MPa-1: 1e-2 m / 1e-4 MPa / 86400 s
KR_TO_M_PER_S_PER_MPA = 1e-2 / 1e-4 / 86400.


def parameters(values: dict = None, main_only: bool = False) -> pd.DataFrame:
    """MECHA's parameters, one row each: its name, its value (the default, or the one in *values*), its unit, its
    description, and whether it is one of the main parameters (those first)."""
    declarations = MechaHydraulics.declarations()
    table = declarations[declarations["kind"] == "parameter"][["name", "default", "unit", "description"]]
    table = table.rename(columns={"default": "value"}).reset_index(drop=True)
    for name, value in (values or {}).items():
        table.loc[table["name"] == name, "value"] = value
    table["main"] = table["name"].isin(MAIN_PARAMETERS)
    order = {name: rank for rank, name in enumerate(MAIN_PARAMETERS)}
    table = table.assign(rank=table["name"].map(order).fillna(len(order))).sort_values("rank", kind="stable")
    table = table.drop(columns="rank").reset_index(drop=True)
    return table[table["main"]].reset_index(drop=True) if main_only else table


class _RootSection(CompositeModel):
    """One root segment: GRANAP's anatomy, MECHA's hydraulics (the plant model of developer tutorial 1)."""
    from_scale, nodes = "SubOrgan", "Compartment"

    def __init__(self, data_structure, time_step):
        from openalea.granap.metafspm.component import GranapAnatomy

        self.anatomy = GranapAnatomy(data_structure=data_structure, time_step=time_step)
        self.hydraulics = MechaHydraulics(data_structure=data_structure, time_step=time_step)
        self.declare_data_and_couple_components(translator_path=Translator(),
                                                components=(self.anatomy, self.hydraulics))

    def run(self):
        self.anatomy()
        self.hydraulics()


def _plant_model():
    from openalea.granap.metafspm.architectures import RootAxis
    from openalea.granap.metafspm.component import GranapAnatomy, junction_rules

    _RootSection.initiators = (RootAxis, GranapAnatomy)
    _RootSection.wiring = junction_rules
    return _RootSection


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
             xylem_psi: float = -5000.) -> Section:
    """Solve the water flow across one root cross section: GRANAP's anatomy (*anatomy*: its parameters, e.g.
    {"preset": "for_dicot_root", "cortex": {"n_layers": 3}}; a monocot root by default), MECHA's hydraulics with its
    defaults changed by *parameters* ({name: value}), the apoplastic barriers *barrier* (MECHA's maturity code: 1
    an endodermal Casparian strip, 2 + suberin with passage cells, 3 a fully suberized endodermis, 4–10 the
    exodermis' variants, 11 both suberized and the sclerenchyma's walls lignified; 0, none, is a root tip whose water goes to the protophloem: the standard problem then needs
    parameters={"standard_psi_sieve": ...}), the soil at *soil_psi* and the xylem at *xylem_psi* (hPa).

    The summary: kr, the radial conductivity (cm hPa-1 d-1; kr_SI in m s-1 MPa-1), the water uptake (cm³ d-1), the
    perimeter (cm), the number of cells, and the share of the radial flow carried across the cortex by the walls,
    the membranes and the plasmodesmata."""
    scenario = dict(parameters or {}, standard_psi_soil=soil_psi, standard_psi_xylem=xylem_psi)
    if anatomy is not None:
        scenario["root_anatomy"] = dict(anatomy)
    table = pd.DataFrame([dict(plant="plant", model=_plant_model(), x=0., y=0., z=0., rotation=0.,
                               scenario={"parameters": scenario})])
    scene = Scene(table)
    ds = scene.populations[0].data_structure
    ds.set("barrier", np.full(ds.n_nodes(), float(barrier)))
    scene.run()
    section = Section(scene=scene, summary=pd.Series(dtype=float), parameters=dict(parameters or {}))
    section.summary = _summary(section)
    return section


def simulate_root(barriers=(3, 3, 3, 1, 1, 0), segment_length: float = 5e-3, radius: float = 0.55e-3,
                  parameters: dict = None, anatomy: dict = None, soil_psi: float = -1000.,
                  collar_psi: float = -5000.) -> pd.DataFrame:
    """Solve the water flow in a root axis (MECHA's plant problem): one section per entry of *barriers* (their
    maturity codes, from the collar to the tip), each *segment_length* m long and of *radius* m, joined by their
    xylem vessels; the soil at *soil_psi* around them, the collar's xylem at *collar_psi* (hPa). Returns a table, one
    row per section: its distance from the collar (mm), barrier, mean xylem potential (hPa), uptake (cm³ d-1), kr
    (its standard problem's) and kr_effective (in the root), both in cm hPa-1 d-1; table.attrs["scene"] is the
    metafspm scene."""
    n = len(barriers)
    scenario = dict(parameters or {}, collar_condition=1., collar_psi=collar_psi,
                    root_segments=[[radius, segment_length, (n - 1 - i) * segment_length] for i in range(n)])
    if anatomy is not None:
        scenario["root_anatomy"] = dict(anatomy)
    scene = Scene(pd.DataFrame([dict(plant="root", model=_plant_model(), x=0., y=0., z=0., rotation=0.,
                                     scenario={"parameters": scenario})]))
    ds = scene.populations[0].data_structure
    owner = np.asarray(ds.owner("SubOrgan"))                  # the sections in order, the collar's first
    ds.set("barrier", np.asarray(barriers, dtype=float)[owner])
    ds.set("soil_psi", np.full(ds.n_nodes(), float(soil_psi)))
    scene.run()
    psi, xylem = np.asarray(ds.get("psi")), np.isin(np.asarray(ds.get("cgroup")), (13, 19, 20))
    table = pd.DataFrame({
        "distance": (np.arange(n) + 0.5) * segment_length * 1e3, "barrier": list(barriers),
        "xylem_potential": [psi[(owner == i) & xylem].mean() for i in range(n)],
        "uptake": np.asarray(ds.get("uptake")), "kr": np.asarray(ds.get("kr")),
        "kr_effective": np.asarray(ds.get("kr_effective"))})
    table.attrs["scene"] = scene
    return table
