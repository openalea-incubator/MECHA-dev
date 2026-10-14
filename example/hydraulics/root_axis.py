"""
MECHA's plant problem on a root axis: six 5 mm segments, each with its GRANAP anatomy and MECHA's conductances,
joined by their xylem vessels (the axial junctions' Poiseuille conductances), the soil around them at -1000 hPa and
the collar's xylem held at -5000 hPa. The barriers mature away from the tip (none at the tip: its xylem does not
conduct yet). Shown: along the axis, the xylem's potential, each section's uptake, and its kr (MECHA's standard
problem) against its effective kr in the plant; then three sections coloured by the solved potential, with the flows.

    python root_axis.py               (from MECHA-dev/example/hydraulics; figure in ./outputs/root_axis.png)
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

N, SEGMENT, RADIUS = 6, 5e-3, 0.55e-3            # segments, m, m
BARRIERS = (3, 3, 3, 1, 1, 0)                    # collar to tip
SOIL_PSI = -1000.                                # hPa


class Root(StructuralComponent):
    """The architecture: one root axis of N segments, from the collar to the tip."""

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        s, L = g.scales, g.labels
        complex_ = plant
        for scale, label in ((s.Axis, L.Axis.Root), (s.GrowthUnit, L.GrowthUnit.Root), (s.Phytomer, L.Phytomer.Root),
                             (s.Organ, L.Organ.RootInternode)):
            complex_ = g.add_component(complex_, **PropsConfig(scale=scale, edge_type='/', label=label))
        segment = g.add_component(complex_, **PropsConfig(scale=s.SubOrgan, edge_type='/',
                                                          label=L.SubOrgan.RootSegment))
        for rank in range(N):
            if rank:
                segment = g.add_child(segment, **PropsConfig(scale=s.SubOrgan, edge_type='<',
                                                             label=L.SubOrgan.RootSegment))
            g.property("radius")[segment] = RADIUS
            g.property("length")[segment] = SEGMENT
            g.property("distance_from_tip")[segment] = (N - 1 - rank) * SEGMENT


def draw_section(ax, fig, ds, i, owner, x, y, view, psi, flow):
    """Section *i*: its edges by log |Q|, its nodes by ψ."""
    edges = np.flatnonzero((owner[view.tail] == i) & (owner[view.head] == i) & (flow != 0))
    lines = np.stack([np.c_[x[view.tail[edges]], y[view.tail[edges]]],
                      np.c_[x[view.head[edges]], y[view.head[edges]]]], axis=1)
    q = np.log10(abs(flow[edges]))
    collection = LineCollection(lines, cmap="Greys", linewidths=0.35, alpha=0.8)
    collection.set_array(q)
    collection.set_clim(np.percentile(q, 5), q.max())
    ax.add_collection(collection)
    nodes = owner == i
    points = ax.scatter(x[nodes], y[nodes], c=psi[nodes], s=0.8, cmap="coolwarm_r", vmin=-5000, vmax=SOIL_PSI)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return points


def main():
    Choregrapher().add_simulation_time_step(3600.)
    plants = pd.DataFrame([dict(plant="p0", model=None, x=0., y=0., z=0., rotation=0., scenario={"parameters": {}})])
    g, _ = build_population(plants, initiators=(Root, GranapAnatomy))
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=junction_rules(g))
    anatomy = GranapAnatomy(data_structure=ds)
    anatomy()
    owner = ds.owner("SubOrgan")                                      # sections in order, collar first
    ds.register("barrier", np.array(BARRIERS, dtype=float)[owner], location="node")
    ds.register("soil_psi", np.full(ds.n_nodes(), SOIL_PSI), location="node")
    mecha = MechaHydraulics(data_structure=ds)
    mecha()

    psi, flow, cgroup = ds.get("psi"), ds.get("flow"), ds.get("cgroup")
    xylem = np.isin(cgroup, (13, 19, 20))
    distance = (np.arange(N) + 0.5) * SEGMENT * 1e3                     # mm from the collar
    psi_xylem = np.array([psi[(owner == i) & xylem].mean() for i in range(N)])
    uptake, kr, kr_eff = ds.get("uptake"), ds.get("kr"), ds.get("kr_effective")
    for i in range(N):
        print(f"section {i}: barrier {BARRIERS[i]}, ψ xylem {psi_xylem[i]:8.1f} hPa, uptake {uptake[i]:.3e} cm3/d, "
              f"kr {kr[i]:.3e}, kr effective {kr_eff[i]:.3e} cm/hPa/d")

    fig = plt.figure(figsize=(15, 9.5))
    top = [fig.add_subplot(2, 3, k) for k in (1, 2, 3)]
    top[0].plot(distance, psi_xylem, "o-")
    top[0].set(xlabel="distance from the collar (mm)", ylabel="mean xylem ψ (hPa)", title="xylem potential")
    top[1].bar(distance, uptake, width=SEGMENT * 1e3 * 0.8)
    top[1].set(xlabel="distance from the collar (mm)", ylabel="uptake (cm³ d⁻¹)", title="uptake per section")
    top[2].semilogy(distance, kr, "o-", label="kr (standard problem)")
    top[2].semilogy(distance, kr_eff, "s--", label="kr effective (plant)")
    top[2].set(xlabel="distance from the collar (mm)", ylabel="cm hPa⁻¹ d⁻¹", title="radial conductivity")
    top[2].legend()
    for k in range(3):
        for i, b in enumerate(BARRIERS):
            top[k].annotate(f"b{b}", (distance[i], 0.02), xycoords=("data", "axes fraction"), ha="center", fontsize=7)
    x, y = ds.get("section_x"), ds.get("section_y")
    view = ds.to_graph_view()
    for k, i in enumerate((0, N // 2, N - 1)):
        ax = fig.add_subplot(2, 3, 4 + k)
        points = draw_section(ax, fig, ds, i, owner, x, y, view, psi, flow)
        ax.set_title(f"section {i} ({distance[i]:.1f} mm, barrier {BARRIERS[i]})")
    fig.colorbar(points, ax=fig.axes[3:], label="ψ (hPa); edges: log |Q| (grey)", shrink=0.7)
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "root_axis.png")
    fig.savefig(path, dpi=130)
    print(path)


if __name__ == "__main__":
    main()
