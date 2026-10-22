"""
The osmotic potentials in MECHA's component (devplan_metafspm.md, §20–21): one root section (GRANAP's monocot root
preset, 200 µm), its cells' and walls' osmotic potentials π and its membranes' reflection coefficients σ given by
MECHA's presets (MechaOsmoticPresets, standing for a solute model: MechaHydraulics' "Solutes" inputs), and MECHA's
standard problem solved with the membranes' osmotic terms (standard_osmotic): a membrane passes K (Δψ + σ Δπ), ψ the
pressure potential, ψ + π the water potential.

Three plants in one scene, at barriers 1 (endodermal Casparian strip) and 3 (suberized endodermis):
    no solutes     the presets' σ factor 0 and their potentials 0: MECHA's standard problem
    os_hetero 0    the cells at os_cortex (−4800 hPa), the sieve tubes at −10 000, the soil solution at −500, the
                   xylem sap at −1800 (σ = 1)
    os_hetero 1    the legacy's measured profiles: the cortex graded from −6400 hPa (outer row) to −9300 (inner)

    python osmotic_root.py          (from MECHA-dev/example/hydraulics; figures in ./outputs/osmotic_root_*.png: the maps
                                      of π, ψ and ψ + π; their radial profiles; the pathways' shares of the radial flow)
"""
import os
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.scene.scene import Scene

from openalea.granap.metafspm.architectures import RootAxis
from openalea.granap.metafspm.component import GranapAnatomy, junction_rules
from openalea.mecha import plots
from openalea.mecha.component import MechaHydraulics
from openalea.mecha.osmotic import MechaOsmoticPresets

BARRIERS = (1, 3)
NO_SOLUTES = dict(s_factor=0., os_cortex=0., osmotic_sieve=0., osmotic_left_soil=0., osmotic_right_soil=0.,
                  osmotic_xyl=0., osmotic_endo=0.)
CASES = {"no solutes": NO_SOLUTES, "os_hetero 0": {"os_hetero": 0.}, "os_hetero 1": {"os_hetero": 1.}}
HERE = os.path.dirname(os.path.abspath(__file__))


class OsmoticRoot(CompositeModel):
    """GRANAP's anatomies, MECHA's osmotic presets (the "Solutes" role) and hydraulics with the osmotic terms."""
    initiators = (RootAxis, GranapAnatomy)
    from_scale, nodes, wiring = "SubOrgan", "Compartment", junction_rules

    def __init__(self, data_structure, time_step):
        ds = data_structure
        self.anatomy = GranapAnatomy(data_structure=ds, time_step=time_step)
        self.osmotic = MechaOsmoticPresets(data_structure=ds, time_step=time_step)
        self.hydraulics = MechaHydraulics(data_structure=ds, time_step=time_step)
        self.declare_data_and_couple_components(translator_path=Translator(),
                                                components=(self.anatomy, self.osmotic, self.hydraulics))

    def run(self):
        self.anatomy()
        self.osmotic()
        self.hydraulics()


def main():
    table = pd.DataFrame([dict(plant=case, model=OsmoticRoot, x=float(i), y=0., z=0., rotation=0.,
                               scenario={"parameters": {"standard_osmotic": 1., **parameters}})
                          for i, (case, parameters) in enumerate(CASES.items())])
    scene = Scene(table)
    ds = scene.populations[0].data_structure
    mecha = scene.populations[0].instance.hydraulics
    segments = [int(s) for s in ds.entity_ids("SubOrgan")]            # one per plant, in the table's order

    results = {}
    for barrier in BARRIERS:
        ds.set("barrier", np.full(ds.n_nodes(), float(barrier)))      # a development model's role
        scene.run()
        for i, case in enumerate(CASES):
            frames = {name: plots.port_frame(mecha, segments[i], psi=variable) for name, variable in
                      (("osmotic", "osmotic_potential"), ("pressure", "standard_psi"),
                       ("water", "standard_water_potential"))}
            geo = mecha.section_geometries()[segments[i]][2]
            for frame in frames.values():             # the profiles: cells without the air spaces
                frame.profile_psi = frame.psi.copy()
                frame.profile_psi[geo.typed["intercellular"]] = np.nan
            results[(barrier, case)] = frames, dict(uptake=ds.get("standard_uptake")[i], kr=ds.get("kr")[i])
    print(f"{'barrier':>8s} {'case':>12s} {'uptake (cm3/d)':>15s} {'kr (cm/hPa/d)':>14s}")
    for (barrier, case), (_, r) in results.items():
        print(f"{barrier:8d} {case:>12s} {r['uptake']:15.4e} {r['kr']:14.4e}")

    folder = os.path.join(HERE, "outputs")
    os.makedirs(folder, exist_ok=True)
    rows = (("osmotic", "osmotic potential π"), ("pressure", "pressure potential ψ"), ("water", "water potential ψ + π"))
    for barrier in BARRIERS:
        frames = {case: results[(barrier, case)][0] for case in CASES}
        fig, axes = plt.subplots(3, 3, figsize=(15, 15), layout="constrained")
        for i, (key, title) in enumerate(rows):
            values = np.concatenate([frames[case][key].psi for case in CASES])
            norm = plt.Normalize(values.min(), values.max())
            for j, case in enumerate(CASES):
                points = plots.psi_map(axes[i, j], frames[case][key], title=f"{case}: {title}", norm=norm)
            fig.colorbar(points, ax=axes[i, :], label=f"{title} (hPa)", shrink=0.8)
        fig.suptitle(f"One root section, barrier {barrier}: MECHA's standard problem with the osmotic terms")
        fig.savefig(os.path.join(folder, f"osmotic_root_barrier{barrier}_maps.png"), dpi=300)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), layout="constrained")
        for ax, (key, title) in zip(axes, rows):
            for case, style in zip(CASES, ("-", "--", ":")):
                frame = frames[case][key]
                plots.radial_profile(ax, replace(frame, psi=frame.profile_psi), label=case, linestyle=style)
            ax.set(title=f"{title}, radial profile", ylabel=f"{title} (hPa)")
            ax.legend(fontsize=6)
        fig.savefig(os.path.join(folder, f"osmotic_root_barrier{barrier}_profiles.png"), dpi=300)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), layout="constrained")
        for ax, case in zip(axes, CASES):
            plots.pathway_breakdown(ax, frames[case]["pressure"], title=f"{case}: pathways of the radial flow")
        fig.savefig(os.path.join(folder, f"osmotic_root_barrier{barrier}_pathways.png"), dpi=300)
        plt.close(fig)
    print(folder)


if __name__ == "__main__":
    main()
