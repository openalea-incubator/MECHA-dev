"""
Tutorial 01 against the legacy MECHA: the same section (GRANAP's root preset, or GRANAP's CellSet organ from a CellSet
file) through the port and through the legacy (its NetworkBuilder on the same organ), kr and Kx at barriers 1 and 3,
and the tutorial's views drawn from both by the same functions (openalea.mecha.plots).

    python tutorial_01_input.py [cellset]      (from MECHA-dev/example/comparison; figures in ./outputs/)
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "tutorials"), os.path.join(HERE, "..", "..", "test", "comparison")]

from legacy_run import legacy_frame, legacy_section                             # noqa: E402
from tutorial_01_input import BARRIERS, section                                  # noqa: E402

from openalea.mecha import plots                                                # noqa: E402


def main(source="granap"):
    ds, mecha, library = section(source)
    segment = int(ds.entity_ids("SubOrgan")[0])
    organ = library.by_index(int(ds.get("anatomy_template")[0])).organ
    height = float(np.asarray(ds.get("length"))[0]) * 1.0E6
    run = legacy_section(organ, barriers=BARRIERS, height=height)
    port, legacy = {}, {}
    print(f"{source}: {'barrier':>8s} {'kr legacy':>12s} {'kr port':>12s} {'Kx legacy':>12s} {'Kx port':>12s}")
    for i, barrier in enumerate(BARRIERS):
        ds.set("barrier", np.full(ds.n_nodes(), float(barrier)))
        mecha()
        port[barrier] = plots.port_frame(mecha, segment, libraries=(library,))
        legacy[barrier] = legacy_frame(run, i, organ)
        Kx = float(np.ravel(run.stages[i]["Kx"])[0])
        print(f"{'':{len(source) + 1}s} {barrier:8d} {run.stages[i]['kr']:12.6e} {ds.get('kr')[0]:12.6e} {Kx:12.6e} "
              f"{ds.get('Kx')[0]:12.6e}")

    folder = os.path.join(HERE, "outputs")
    os.makedirs(folder, exist_ok=True)
    last = BARRIERS[-1]
    norm = plt.Normalize(min(port[last].psi.min(), legacy[last].psi.min()), 0.)
    fig, axes = plt.subplots(3, 2, figsize=(13, 18))
    for column, (name, frames) in enumerate((("legacy", legacy), ("port", port))):
        plots.tissue_map(axes[0, column], frames[last], title=f"{name}: the section ({source})")
        points = plots.psi_map(axes[1, column], frames[last], title=f"{name}: ψ, barrier {last}", norm=norm)
        plots.pathway_breakdown(axes[2, column], frames[last], title=f"{name}: pathways of the radial flow")
    fig.colorbar(points, ax=axes[1, :], label="ψ (hPa)", shrink=0.8)
    fig.savefig(os.path.join(folder, f"tutorial_01_{source}_legacy_port.png"), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for barrier in BARRIERS:
        plots.radial_profile(ax, legacy[barrier], label=f"legacy, barrier {barrier}", linestyle="-", alpha=0.6, lw=3)
        plots.radial_profile(ax, port[barrier], label=f"port, barrier {barrier}", linestyle="--", color="k", lw=0.8)
    ax.legend(fontsize=6)
    fig.savefig(os.path.join(folder, f"tutorial_01_{source}_profiles.png"), dpi=300)
    plt.close(fig)
    print(folder)


if __name__ == "__main__":
    main("cellset" if "cellset" in sys.argv[1:] else "granap")
