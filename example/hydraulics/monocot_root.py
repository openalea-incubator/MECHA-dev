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

from openalea.metafspm import CompositeModel, Planting, planting, Scene

from openalea.granap.metafspm.architectures import RootAxis
from openalea.granap.metafspm.component import GranapAnatomy, junction_rules
from openalea.mecha import plots
from openalea.mecha.component import MechaHydraulics
from openalea.mecha.stand_ins import Development, development_parameters

SEGMENT, RADIUS = 5e-3, 0.55e-3                   # m
BARRIERS = (4, 3, 1)          # base to tip: + exodermal strip, suberized endodermis, endodermal Casparian strip


class RootHydraulics(CompositeModel):
    """GRANAP's anatomies on a root's segments, their barriers by their distance from the tip (Development, a root
    development model's stand-in) and MECHA's
    hydraulics."""
    initiators = (RootAxis, GranapAnatomy)
    from_scale, nodes, wiring = "SubOrgan", "Compartment", junction_rules

    parts = {"anatomy": GranapAnatomy, "development": Development, "hydraulics": MechaHydraulics}
    schedule = ("development", "anatomy", "hydraulics")


def main():
    distances = [(2 - i) * SEGMENT for i in range(3)]                   # from the tip, base first
    scenario = {"root_segments": [[RADIUS, SEGMENT, distance] for distance in distances],
                **development_parameters(distances, BARRIERS)}               # the barriers by distance from the tip
    scene = Scene(planting([Planting()], RootHydraulics, scenario))
    data_structure = scene.populations[0].data_structure
    mecha = scene.populations[0].instance.hydraulics
    scene.run()

    segments = [int(s) for s in data_structure.entity_ids("SubOrgan")]
    barriers = data_structure.map_to(data_structure.get("barrier"), "node", "SubOrgan", "max").astype(int)     # each section's
    for i, segment in enumerate(segments):
        print(f"segment {segment}: barrier {barriers[i]}, kr {data_structure.get('kr')[i]:.3e} cm/hPa/d, "
              f"Kx {data_structure.get('Kx')[i]:.3e} cm4/hPa/d, uptake {data_structure.get('standard_uptake')[i]:.3e} cm3/d")

    fig, axes = plt.subplots(2, len(segments), figsize=(5 * len(segments), 10), layout="constrained")
    norm = plt.Normalize(mecha.standard_psi_xylem, mecha.standard_psi_soil)
    tip_first = np.argsort(np.asarray(data_structure.get("distance_from_tip"), dtype=float), kind="stable")
    for col, i in enumerate(tip_first):
        frame = plots.port_frame(mecha, segments[i])
        points = plots.psi_map(axes[0, col], frame, norm=norm,
                               title=f"barrier {barriers[i]}: kr {data_structure.get('kr')[i]:.2e} cm hPa⁻¹ d⁻¹")
        plots.pathway_breakdown(axes[1, col], frame, title=f"barrier {barriers[i]}: pathways of the radial flow")
    fig.colorbar(points, ax=axes[0], label="ψ (hPa)", shrink=0.8)
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, "monocot_root.png"), dpi=130)
    print(os.path.join(folder, "monocot_root.png"))


if __name__ == "__main__":
    main()
