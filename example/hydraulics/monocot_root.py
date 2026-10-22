"""
The simplest use of MECHA's component in metafspm: a monocot root of three segments gets its cross-section anatomies
from GRANAP's component, and MECHA's component computes, on each section, the conductance of every wall, membrane and
plasmodesma, then solves MECHA's standard problem (soil at 0 hPa around the section, xylem at -5000 hPa) for the
water potentials, the flows, and the section's radial conductivity kr and axial conductance Kx. The segments differ
by their apoplastic barriers (as a development model would give them): an endodermal Casparian strip near the tip,
then a suberized endodermis, then an exodermal Casparian strip too.

    python monocot_root.py            (from MECHA-dev/example/hydraulics; figure in ./outputs/monocot_root.png)
"""
import os

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

SEGMENT, RADIUS = 5e-3, 0.55e-3                   # m
BARRIERS = (4, 3, 1)          # base to tip: + exodermal strip, suberized endodermis, endodermal Casparian strip


class RootHydraulics(CompositeModel):
    """GRANAP's anatomies and MECHA's hydraulics on a root's segments."""
    initiators = (RootAxis, GranapAnatomy)
    from_scale, nodes, wiring = "SubOrgan", "Compartment", junction_rules

    def __init__(self, data_structure, time_step):
        self.anatomy = GranapAnatomy(data_structure=data_structure, time_step=time_step)
        self.hydraulics = MechaHydraulics(data_structure=data_structure, time_step=time_step)
        self.declare_data_and_couple_components(translator_path=Translator(),
                                                components=(self.anatomy, self.hydraulics))

    def run(self):
        self.anatomy()
        self.hydraulics()


def main():
    scenario = {"root_segments": [[RADIUS, SEGMENT, (2 - i) * SEGMENT] for i in range(3)]}
    scene = Scene(pd.DataFrame([dict(plant="root", model=RootHydraulics, x=0., y=0., z=0., rotation=0.,
                                     scenario={"parameters": scenario})]))
    ds = scene.populations[0].data_structure
    mecha = scene.populations[0].instance.hydraulics
    ds.set("barrier", np.array(BARRIERS, dtype=float)[ds.owner("SubOrgan")])   # a development model's role
    scene.run()

    segments = [int(s) for s in ds.entity_ids("SubOrgan")]
    for i, segment in enumerate(segments):
        print(f"segment {segment}: barrier {BARRIERS[i]}, kr {ds.get('kr')[i]:.3e} cm/hPa/d, "
              f"Kx {ds.get('Kx')[i]:.3e} cm4/hPa/d, uptake {ds.get('standard_uptake')[i]:.3e} cm3/d")

    fig, axes = plt.subplots(2, len(segments), figsize=(5 * len(segments), 10), layout="constrained")
    norm = plt.Normalize(mecha.standard_psi_xylem, mecha.standard_psi_soil)
    for col, (i, segment) in enumerate(list(enumerate(segments))[::-1]):        # tip first
        frame = plots.port_frame(mecha, segment)
        points = plots.psi_map(axes[0, col], frame, norm=norm,
                               title=f"barrier {BARRIERS[i]}: kr {ds.get('kr')[i]:.2e} cm hPa⁻¹ d⁻¹")
        plots.pathway_breakdown(axes[1, col], frame, title=f"barrier {BARRIERS[i]}: pathways of the radial flow")
    fig.colorbar(points, ax=axes[0], label="ψ (hPa)", shrink=0.8)
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, "monocot_root.png"), dpi=130)
    print(os.path.join(folder, "monocot_root.png"))


if __name__ == "__main__":
    main()
