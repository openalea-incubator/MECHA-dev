"""
MECHA's plant problem on a root axis: six 5 mm segments, each with its GRANAP anatomy and MECHA's conductances,
joined by their xylem vessels (the axial junctions' Poiseuille conductances), the soil around them at -1000 hPa and
the collar's xylem held at -5000 hPa. The barriers mature away from the tip (none at the tip: its xylem does not
conduct yet). Shown: along the axis, the xylem's potential, each section's uptake, and its kr (MECHA's standard
problem) against its effective kr in the plant; then three sections coloured by the solved potential.

    python root_axis.py               (from MECHA-dev/example/hydraulics; figure in ./outputs/root_axis.png)
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

N, SEGMENT, RADIUS = 6, 5e-3, 0.55e-3            # segments, m, m
BARRIERS = (3, 3, 3, 1, 1, 0)                    # collar to tip
SOIL_PSI = -1000.                                # hPa


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
    scenario = {"root_segments": [[RADIUS, SEGMENT, (N - 1 - i) * SEGMENT] for i in range(N)],
                "collar_condition": 1., "collar_psi": -5000.}
    scene = Scene(pd.DataFrame([dict(plant="root", model=RootHydraulics, x=0., y=0., z=0., rotation=0.,
                                     scenario={"parameters": scenario})]))
    ds = scene.populations[0].data_structure
    mecha = scene.populations[0].instance.hydraulics
    owner = ds.owner("SubOrgan")                                      # sections in order, collar first
    ds.set("barrier", np.array(BARRIERS, dtype=float)[owner])         # a development model's role
    ds.set("soil_psi", np.full(ds.n_nodes(), SOIL_PSI))               # a soil model's
    scene.run()

    psi = ds.get("psi")
    xylem = np.isin(ds.get("cgroup"), (13, 19, 20))
    distance = (np.arange(N) + 0.5) * SEGMENT * 1e3                     # mm from the collar
    psi_xylem = np.array([psi[(owner == i) & xylem].mean() for i in range(N)])
    uptake, kr, kr_eff = ds.get("uptake"), ds.get("kr"), ds.get("kr_effective")
    for i in range(N):
        print(f"section {i}: barrier {BARRIERS[i]}, ψ xylem {psi_xylem[i]:8.1f} hPa, uptake {uptake[i]:.3e} cm3/d, "
              f"kr {kr[i]:.3e}, kr effective {kr_eff[i]:.3e} cm/hPa/d")

    fig = plt.figure(figsize=(15, 9.5), layout="constrained")
    top = [fig.add_subplot(2, 3, k) for k in (1, 2, 3)]
    top[0].plot(distance, psi_xylem, "o-")
    top[0].set(xlabel="distance from the collar (mm)", ylabel="mean xylem ψ (hPa)", title="xylem potential")
    top[1].bar(distance, uptake, width=SEGMENT * 1e3 * 0.8)
    top[1].set(xlabel="distance from the collar (mm)", ylabel="uptake (cm³ d⁻¹)", title="uptake per section")
    top[2].semilogy(distance, kr, "o-", label="kr (standard problem)")
    top[2].semilogy(distance, kr_eff, "s--", label="kr effective (plant)")
    top[2].set(xlabel="distance from the collar (mm)", ylabel="cm hPa⁻¹ d⁻¹", title="radial conductivity")
    top[2].legend()
    for ax in top:
        for i, b in enumerate(BARRIERS):
            ax.annotate(f"b{b}", (distance[i], 0.02), xycoords=("data", "axes fraction"), ha="center", fontsize=7)
    segments = [int(s) for s in ds.entity_ids("SubOrgan")]
    bottom = [fig.add_subplot(2, 3, 4 + k) for k in range(3)]
    for ax, i in zip(bottom, (0, N // 2, N - 1)):
        frame = plots.port_frame(mecha, segments[i], psi="psi", flow="flow")
        points = plots.psi_map(ax, frame, norm=plt.Normalize(-5000., SOIL_PSI),
                               title=f"section {i} ({distance[i]:.1f} mm, barrier {BARRIERS[i]})")
    fig.colorbar(points, ax=bottom, label="ψ (hPa)", shrink=0.7)
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "root_axis.png")
    fig.savefig(path, dpi=130)
    print(path)


if __name__ == "__main__":
    main()
