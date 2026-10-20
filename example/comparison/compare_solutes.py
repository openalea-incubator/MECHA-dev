"""
The solute port against the reactions branch's transport, drawn (devplan_metafspm.md, §28e): on GRANAP's
``NeedleAnatomy(seed=42)`` and the reactions' needle water state (its flows given to both), one implicit Euler step
of an hour, legacy | port | difference, on the liquid nodes:

    pulse              from 50 µmol cm⁻³ in the mesophyll, σ_s 0, the air edges' flows 0 (r1, r2, r12 vanish)
    photosynthesis     the reactions' increment in the mesophyll (the port's source V Δc / dt), with starch (2, 1 d⁻¹)
                       in the cells only (its reaction's volumes restricted to them, the port's Q-U4): the sucrose,
                       then the starch
    r1                 a reviewed difference: the solute's σ 0.7 on every membrane, the reactions' flows with K σ Δπ
                       added a second time, against the port's (1 − σ) Q
    r4                 a reviewed difference: the reactions' starch reaction in the walls and junctions too

    python compare_solutes.py          (from MECHA-dev/example/comparison; figure in ./outputs/solutes_needle.png)
"""
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "..", "test"), os.path.join(HERE, "..", "..", "test", "comparison")]

from legacy_run import legacy_reactions_step                                   # noqa: E402
from solute_case import DIFFUSIVITIES, mesophyll_source, needle_case, port_step  # noqa: E402

from openalea.mecha.geometry import MEMBRANE                                  # noqa: E402

STARCH = (2.0, 1.0)                    # d-1: synthesis, degradation (the reactions' diurnal runs')
PHOTOSYNTHESIS = 5e-6                  # mol cm-3: their c_photo, at full light
SIGMA = 0.7


def cases(case):
    """[(title, quantity, legacy by legacy index, port per section node)] of the figure's rows."""
    organ, reference = case["organ"], case["reference"]
    rows = []
    sucrose, _ = port_step(case, reference)
    rows.append(("pulse", "sucrose", reference["c1"], sucrose))

    photosynthesis = legacy_reactions_step(organ, **DIFFUSIVITIES, starch=STARCH, cells_only=True,
                                           source={i: PHOTOSYNTHESIS for i in reference["kinds"]["meso"]})
    sucrose, starch = port_step(case, photosynthesis, starch=STARCH,
                                source=mesophyll_source(case, photosynthesis, PHOTOSYNTHESIS))
    rows.append(("photosynthesis", "sucrose", photosynthesis["c1"], sucrose))
    rows.append(("photosynthesis", "starch", photosynthesis["starch1"], starch))

    r1 = legacy_reactions_step(organ, **DIFFUSIVITIES, solute_sigma={g: SIGMA for g in range(40)})
    ds = case["ds"]
    sucrose, _ = port_step(case, r1, sigma=np.where(np.asarray(ds.get("network_path")) == MEMBRANE, SIGMA, 0.))
    rows.append(("r1: K σ Δπ counted twice", "sucrose", r1["c1"], sucrose))

    r4 = legacy_reactions_step(organ, **DIFFUSIVITIES, starch=STARCH, cells_only=False,
                               source={i: PHOTOSYNTHESIS for i in reference["kinds"]["meso"]})
    rows.append(("r4: starch in the walls", "starch", r4["starch1"], rows[2][3]))
    return rows


def draw(ax, xy, values, cell, **style):
    """The walls and junctions as small points, the cells over them as larger ones."""
    ax.scatter(xy[~cell, 0], xy[~cell, 1], c=values[~cell], s=0.6, **style)
    return ax.scatter(xy[cell, 0], xy[cell, 1], c=values[cell], s=9, edgecolors="none", **style)


def main():
    case = needle_case()
    liquid, index = case["liquid"], case["index"]
    port_xy, legacy_xy = case["xy"][liquid], case["legacy_xy"][index[liquid]]
    cell = (np.asarray(case["ds"].get("kind"))[case["nodes"]] == 0)[liquid]
    rows = cases(case)
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 3.3 * len(rows)), layout="constrained")
    for (title, quantity, legacy, port), ax in zip(rows, axes):
        legacy_values, port_values = legacy[index[liquid]], port[liquid]
        top = max(legacy_values.max(), port_values.max(), 1e-30)
        style = dict(cmap="viridis", norm=matplotlib.colors.Normalize(0., top * 1e6))
        points = draw(ax[0], legacy_xy, legacy_values * 1e6, cell, **style)
        draw(ax[1], port_xy, port_values * 1e6, cell, **style)
        fig.colorbar(points, ax=ax[:2], label=f"{quantity} (µmol cm⁻³)", shrink=0.9)
        difference = abs(port_values - legacy_values) / top
        points = draw(ax[2], port_xy, difference, cell, cmap="magma",
                      norm=matplotlib.colors.Normalize(0., max(difference.max(), 1e-16)))
        fig.colorbar(points, ax=ax[2], label=f"|Δ{quantity}| / max", shrink=0.9)
        ax[0].set_ylabel(f"{title}\nmax |Δ| / max {difference.max():.1e}")
        print(f"{title:28s} {quantity:8s} max |Δ| / max {difference.max():.2e}")
        for a in ax:
            a.set_aspect("equal")
            a.set_xticks([])
            a.set_yticks([])
    for a, title in zip(axes[0], ("reactions branch", "port", "difference")):
        a.set_title(title)
    fig.suptitle("Sucrose and starch after one hour on the needle (the reactions' water flows given to both)")
    folder = os.path.join(HERE, "outputs")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "solutes_needle.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
