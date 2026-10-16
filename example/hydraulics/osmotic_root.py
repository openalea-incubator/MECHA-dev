"""
The osmotic potentials in MECHA's component (devplan_metafspm.md, §20–21): one root section (GRANAP's monocot root
preset, 200 µm), its cells' and walls' osmotic potentials π and its membranes' reflection coefficients σ given by
MECHA's presets (MechaOsmoticPresets, standing for a solute model: MechaHydraulics' "Solutes" inputs), and MECHA's
standard problem solved with the membranes' osmotic terms (standard_osmotic): a membrane passes K (Δψ + σ Δπ), ψ the
pressure potential, ψ + π the water potential.

Three cases, at barriers 1 (endodermal Casparian strip) and 3 (suberized endodermis):
    no solutes     π = σ = 0 (no "Solutes" model): MECHA's standard problem
    os_hetero 0    the cells at os_cortex (−4800 hPa), the sieve tubes at −10 000, the soil solution at −500, the
                   xylem sap at −1800 (σ = 1)
    os_hetero 1    the legacy's measured profiles: the cortex graded from −6400 hPa (outer row) to −9300 (inner)

    python osmotic_root.py          (from MECHA-dev/example/hydraulics; figures in ./outputs/osmotic_root_*.png: the maps
                                      of π, ψ and ψ + π; their radial profiles; the pathways' shares of the radial flow)
"""
import os
from dataclasses import replace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import StructuralComponent
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.scene.population import build_population

from openalea.granap.metafspm.component import GranapAnatomy, junction_rules
from openalea.mecha import plots
from openalea.mecha.component import MechaHydraulics
from openalea.mecha.osmotic import MechaOsmoticPresets

HEIGHT, RADIUS = 200e-6, 0.55e-3                  # m
BARRIERS = (1, 3)
CASES = {"no solutes": None, "os_hetero 0": 0., "os_hetero 1": 1.}
HERE = os.path.dirname(os.path.abspath(__file__))


class Root(StructuralComponent):
    """One root segment, 200 µm long (one MECHA section)."""

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        s, L = g.scales, g.labels
        complex_ = plant
        for scale, label in ((s.Axis, L.Axis.Root), (s.GrowthUnit, L.GrowthUnit.Root), (s.Phytomer, L.Phytomer.Root),
                             (s.Organ, L.Organ.RootInternode)):
            complex_ = g.add_component(complex_, **PropsConfig(scale=scale, edge_type='/', label=label))
        segment = g.add_component(complex_, **PropsConfig(scale=s.SubOrgan, edge_type='/',
                                                          label=L.SubOrgan.RootSegment))
        g.property("radius")[segment] = RADIUS
        g.property("length")[segment] = HEIGHT
        g.property("distance_from_tip")[segment] = 0.


def section():
    """The plant, its GRANAP anatomy, the DataStructure, MECHA's osmotic presets and hydraulics."""
    Choregrapher().add_simulation_time_step(3600.)
    plants = pd.DataFrame([dict(plant="p0", model=None, x=0., y=0., z=0., rotation=0., scenario={"parameters": {}})])
    g, _ = build_population(plants, initiators=(Root, GranapAnatomy))
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=junction_rules(g))
    GranapAnatomy(data_structure=ds)()
    ds.register("barrier", np.ones(ds.n_nodes()), location="node")
    presets = MechaOsmoticPresets(data_structure=ds)
    mecha = MechaHydraulics(data_structure=ds)
    mecha.standard_osmotic = 1.
    return ds, presets, mecha


def solve(ds, presets, mecha, barrier, os_hetero):
    """One case: the presets (or none), the hydraulics; its section frames (π, pressure, water potential) and
    results."""
    ds.set("barrier", np.full(ds.n_nodes(), float(barrier)))
    if os_hetero is None:
        ds.set("osmotic_potential", np.zeros(ds.n_nodes()))
        ds.set("reflection_coefficient", np.zeros(ds.n_edges()))
    else:
        presets.os_hetero = os_hetero
        presets()
    mecha()
    segment = int(ds.entity_ids("SubOrgan")[0])
    libraries = (GranapAnatomy.library,)
    frames = {name: plots.port_frame(mecha, segment, psi=variable, libraries=libraries) for name, variable in
              (("osmotic", "osmotic_potential"), ("pressure", "standard_psi"), ("water", "standard_water_potential"))}
    nodes, _, geo = mecha.section_geometries()[segment]
    for frame in frames.values():                     # the profiles: cells without the air spaces (π: the soil's, σ 0)
        frame.profile_psi = frame.psi.copy()
        frame.profile_psi[geo.typed["intercellular"]] = np.nan
    return frames, dict(uptake=float(ds.get("standard_uptake")[0]), kr=float(ds.get("kr")[0]))


def main():
    ds, presets, mecha = section()
    results = {(barrier, case): solve(ds, presets, mecha, barrier, value) for barrier in BARRIERS
               for case, value in CASES.items()}
    print(f"{'barrier':>8s} {'case':>12s} {'uptake (cm3/d)':>15s} {'kr (cm/hPa/d)':>14s}")
    for (barrier, case), (_, r) in results.items():
        print(f"{barrier:8d} {case:>12s} {r['uptake']:15.4e} {r['kr']:14.4e}")

    folder = os.path.join(HERE, "outputs")
    os.makedirs(folder, exist_ok=True)
    rows = (("osmotic", "osmotic potential π"), ("pressure", "pressure potential ψ"), ("water", "water potential ψ + π"))
    for barrier in BARRIERS:
        frames = {case: results[(barrier, case)][0] for case in CASES}
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        for i, (key, title) in enumerate(rows):
            values = np.concatenate([frames[case][key].psi for case in CASES])
            norm = matplotlib.colors.Normalize(values.min(), values.max())
            for j, case in enumerate(CASES):
                points = plots.psi_map(axes[i, j], frames[case][key], title=f"{case}: {title}", norm=norm)
            fig.colorbar(points, ax=axes[i, :], label=f"{title} (hPa)", shrink=0.8)
        fig.suptitle(f"One root section, barrier {barrier}: MECHA's standard problem with the osmotic terms")
        fig.savefig(os.path.join(folder, f"osmotic_root_barrier{barrier}_maps.png"), dpi=300)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
        for ax, (key, title) in zip(axes, rows):
            for case, style in zip(CASES, ("-", "--", ":")):
                frame = frames[case][key]
                plots.radial_profile(ax, replace(frame, psi=frame.profile_psi), label=case, linestyle=style)
            ax.set(title=f"{title}, radial profile", ylabel=f"{title} (hPa)")
            ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(folder, f"osmotic_root_barrier{barrier}_profiles.png"), dpi=300)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
        for ax, case in zip(axes, CASES):
            plots.pathway_breakdown(ax, frames[case]["pressure"], title=f"{case}: pathways of the radial flow")
        fig.tight_layout()
        fig.savefig(os.path.join(folder, f"osmotic_root_barrier{barrier}_pathways.png"), dpi=300)
        plt.close(fig)
    print(folder)


if __name__ == "__main__":
    main()
