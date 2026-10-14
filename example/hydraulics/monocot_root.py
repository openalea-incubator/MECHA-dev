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
from matplotlib.collections import LineCollection

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import StructuralComponent
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.scene.population import build_population

from openalea.granap.metafspm.component import GranapAnatomy, junction_rules
from openalea.mecha.component import MechaHydraulics

SEGMENT, RADIUS = 5e-3, 0.55e-3                   # m
BARRIERS = (1, 3, 4)          # tip to base: endodermal Casparian strip, suberized endodermis, + exodermal strip


# 1. The architecture: one root axis of three segments (what a root architecture model would build)
class Root(StructuralComponent):
    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        s, L = g.scales, g.labels
        complex_ = plant
        for scale, label in ((s.Axis, L.Axis.Root), (s.GrowthUnit, L.GrowthUnit.Root), (s.Phytomer, L.Phytomer.Root),
                             (s.Organ, L.Organ.RootInternode)):
            complex_ = g.add_component(complex_, **PropsConfig(scale=scale, edge_type='/', label=label))
        segment = g.add_component(complex_, **PropsConfig(scale=s.SubOrgan, edge_type='/',
                                                          label=L.SubOrgan.RootSegment))
        for rank in range(3):
            if rank:
                segment = g.add_child(segment, **PropsConfig(scale=s.SubOrgan, edge_type='<',
                                                             label=L.SubOrgan.RootSegment))
            g.property("radius")[segment] = RADIUS
            g.property("length")[segment] = SEGMENT
            g.property("distance_from_tip")[segment] = (2 - rank) * SEGMENT


def main():
    # 2. Instantiate: the plant and its anatomies (GRANAP), the DataStructure over their Compartments, MECHA
    Choregrapher().add_simulation_time_step(3600.)
    plants = pd.DataFrame([dict(plant="p0", model=None, x=0., y=0., z=0., rotation=0., scenario={"parameters": {}})])
    g, _ = build_population(plants, initiators=(Root, GranapAnatomy))
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=junction_rules(g))
    anatomy = GranapAnatomy(data_structure=ds)
    anatomy()
    segments = ds.entity_ids("SubOrgan")                              # base to tip
    owner = ds.owner("SubOrgan")
    ds.register("barrier", np.array(BARRIERS[::-1], dtype=float)[owner], location="node")
    mecha = MechaHydraulics(data_structure=ds)

    # 3. Run: conductances, then each section's standard problem
    mecha()
    for i, segment in enumerate(segments):
        print(f"segment {segment}: barrier {BARRIERS[::-1][i]}, kr {ds.get('kr')[i]:.3e} cm/hPa/d, "
              f"Kx {ds.get('Kx')[i]:.3e} cm4/hPa/d, uptake {ds.get('uptake')[i]:.3e} cm3/d")

    # 4. Plot each section: conductances and flows on the edges, potentials on the nodes
    x, y = ds.get("section_x"), ds.get("section_y")                   # mm
    view = ds.to_graph_view()
    K, Q, psi = ds.get("conductance"), ds.get("flow"), ds.get("psi")
    drawn = (K > 0) & (owner[view.tail] == owner[view.head])
    fig, axes = plt.subplots(3, len(segments), figsize=(4.4 * len(segments), 13))
    for col, i in enumerate(range(len(segments))[::-1]):              # tip first
        edges = np.flatnonzero(drawn & (owner[view.tail] == i))
        lines = np.stack([np.c_[x[view.tail[edges]], y[view.tail[edges]]],
                          np.c_[x[view.head[edges]], y[view.head[edges]]]], axis=1)
        for row, values, label, cmap in ((0, np.log10(K[edges]), "log10 K (cm³ hPa⁻¹ d⁻¹)", "viridis"),
                                         (2, np.log10(np.maximum(abs(Q[edges]), 1e-30)), "log10 |Q| (cm³ d⁻¹)",
                                          "plasma")):
            collection = LineCollection(lines, cmap=cmap, linewidths=0.4)
            collection.set_array(values)
            collection.set_clim(*np.percentile(values, (2, 100)))
            axes[row, col].add_collection(collection)
            axes[row, col].autoscale_view()
            fig.colorbar(collection, ax=axes[row, col], label=label, shrink=0.8)
        nodes = owner == i
        points = axes[1, col].scatter(x[nodes], y[nodes], c=psi[nodes], s=0.8, cmap="coolwarm_r", vmin=-5000, vmax=0)
        fig.colorbar(points, ax=axes[1, col], label="ψ (hPa)", shrink=0.8)
        axes[0, col].set_title(f"barrier {BARRIERS[::-1][i]}: kr {ds.get('kr')[i]:.2e} cm hPa⁻¹ d⁻¹")
        for ax in axes[:, col]:
            ax.set_aspect("equal")
            ax.set_xlabel("mm")
    fig.tight_layout()
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, "monocot_root.png"), dpi=130)
    print(os.path.join(folder, "monocot_root.png"))


if __name__ == "__main__":
    main()
