"""
The solute port against the reactions branch over a day (devplan_metafspm.md, §28e): on GRANAP's
``NeedleAnatomy(seed=42)`` and the reactions' needle water state (its flows, constant, given to both), 24 implicit
Euler steps of an hour from no sugar, each chained on its own previous concentrations:
  - photosynthesis by day: the reactions' increment c_photo × max(0, sin(π (t − 6) / 12)) in the mesophyll before each
    step (its diurnal runs'), the port's equivalent source V Δc / dt;
  - starch (2, 1 d⁻¹) in the cells only (the reactions' reaction volumes restricted to them, the port's Q-U4);
  - σ_s 0 and the air edges' flows 0 (r1, r2, r12 vanish); the sinks (phloem loading, xylem) at 0.
The masses (sucrose in the cells and in the walls, starch, taken by the sinks) along the day, legacy and port, their
relative differences, and the maps at the end of the day.

    python compare_solutes_day.py      (from MECHA-dev/example/comparison; figures in ./outputs/solutes_needle_day_*.png)
"""
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "..", "test"), os.path.join(HERE, "..", "..", "test", "comparison")]

from compare_solutes import draw                                              # noqa: E402
from legacy_run import legacy_reactions_step                                   # noqa: E402
from solute_case import DIFFUSIVITIES, DT, mesophyll_source, needle_case, port_step  # noqa: E402

STARCH = (2.0, 1.0)                    # d-1: synthesis, degradation
PHOTOSYNTHESIS = 5e-6                  # mol cm-3 per hour at full light: the reactions' c_photo
HOURS = 24


def light(hour):
    """The reactions' diurnal factor: a half sine from 6 h to 18 h."""
    return max(0., np.sin(np.pi * (hour - 6.) / 12.))


def run(case):
    """Both models through the day; their sucrose and starch by legacy index after each step, and the inputs."""
    organ, water = case["organ"], case["reference"]["reference"]
    index, n = case["index"], case["legacy_xy"].shape[0]
    legacy = dict(c=np.zeros(n), s=np.zeros(n))
    port = dict(c=np.zeros(n), s=np.zeros(n))
    history = []
    for hour in range(HOURS):
        increment = PHOTOSYNTHESIS * light(hour)
        step = legacy_reactions_step(organ, **DIFFUSIVITIES, starch=STARCH, cells_only=True, reference=water,
                                     c_start=legacy["c"], starch_start=legacy["s"],
                                     source={i: increment for i in case["reference"]["kinds"]["meso"]})
        legacy = dict(c=step["c1"], s=step["starch1"])
        sucrose, starch = port_step(case, step, starch=STARCH, source=mesophyll_source(case, step, increment),
                                    c_start=port["c"], starch_start=port["s"])
        port = dict(c=np.zeros(n), s=np.zeros(n))
        port["c"][index], port["s"][index] = sucrose, starch
        history.append(dict(hour=hour + 1, legacy=legacy, port=port, produced=increment, bc=step["bc"]))
    return history


def masses(case, history):
    """Per model and hour (mol): sucrose in the cells, in the walls and junctions, starch, and the cumulated
    production and uptake by the sinks (production − what the section holds)."""
    volume = case["reference"]["volumes"]
    nwj = case["reference"]["network"].n_wall_junction
    meso = np.asarray(case["reference"]["kinds"]["meso"])
    liquid = np.zeros(volume.size, dtype=bool)
    liquid[case["index"][case["liquid"]]] = True
    sink = np.zeros(volume.size, dtype=bool)
    sink[list(history[0]["bc"])] = True
    free = liquid & ~sink
    walls = np.arange(volume.size) < nwj
    out = {}
    produced = np.cumsum([h["produced"] for h in history]) * volume[meso].sum()
    for model in ("legacy", "port"):
        c = np.array([h[model]["c"] for h in history])
        s = np.array([h[model]["s"] for h in history])
        held = dict(cells=(c * volume)[:, free & ~walls].sum(1), walls=(c * volume)[:, free & walls].sum(1),
                    starch=(s * volume)[:, free].sum(1))
        held["sinks"] = produced - held["cells"] - held["walls"] - held["starch"]
        out[model] = held
    return out, produced


def main():
    case = needle_case()
    history = run(case)
    m, produced = masses(case, history)
    hours = [h["hour"] for h in history]
    folder = os.path.join(HERE, "outputs")
    os.makedirs(folder, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), layout="constrained")
    for name, color in zip(("cells", "walls", "starch", "sinks"), ("C0", "C1", "C2", "C3")):
        axes[0].plot(hours, m["legacy"][name] * 1e12, color=color, label=f"{name}: reactions branch")
        axes[0].plot(hours, m["port"][name] * 1e12, color=color, linestyle="--", marker=".", label=f"{name}: port")
        difference = abs(m["port"][name] - m["legacy"][name]) / max(abs(m["legacy"][name]).max(), 1e-300)
        axes[1].semilogy(hours, np.maximum(difference, 1e-18), color=color, label=name)
        print(f"{name:7s} end of day: legacy {m['legacy'][name][-1]:.6e} port {m['port'][name][-1]:.6e} mol, "
              f"max relative difference {difference.max():.1e}")
    axes[0].plot(hours, produced * 1e12, color="k", linestyle=":", label="produced")
    axes[0].set(xlabel="hour", ylabel="sucrose (pmol)", title="where the photosynthate is")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set(xlabel="hour", ylabel="|port − legacy| / max legacy", title="relative difference of the masses")
    axes[1].legend(fontsize=7)
    fig.suptitle("A day on the needle, the reactions' water flows given to both: photosynthesis by day, starch in "
                 "the cells")
    path = os.path.join(folder, "solutes_needle_day_masses.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(path)

    liquid, index = case["liquid"], case["index"]
    cell = (np.asarray(case["ds"].get("kind"))[case["nodes"]] == 0)[liquid]
    port_xy, legacy_xy = case["xy"][liquid], case["legacy_xy"][index[liquid]]
    last = history[-1]
    fig, axes = plt.subplots(2, 3, figsize=(15, 6.6), layout="constrained")
    for ax, (key, quantity) in zip(axes, (("c", "sucrose"), ("s", "starch"))):
        legacy_values, port_values = last["legacy"][key][index[liquid]], last["port"][key][index[liquid]]
        top = max(legacy_values.max(), 1e-30)
        style = dict(cmap="viridis", norm=matplotlib.colors.Normalize(0., top * 1e6))
        points = draw(ax[0], legacy_xy, legacy_values * 1e6, cell, **style)
        draw(ax[1], port_xy, port_values * 1e6, cell, **style)
        fig.colorbar(points, ax=ax[:2], label=f"{quantity} (µmol cm⁻³)", shrink=0.9)
        difference = abs(port_values - legacy_values) / top
        points = draw(ax[2], port_xy, difference, cell, cmap="magma",
                      norm=matplotlib.colors.Normalize(0., max(difference.max(), 1e-16)))
        fig.colorbar(points, ax=ax[2], label=f"|Δ{quantity}| / max", shrink=0.9)
        ax[0].set_ylabel(f"{quantity}\nmax |Δ| / max {difference.max():.1e}")
        for a in ax:
            a.set_aspect("equal")
            a.set_xticks([])
            a.set_yticks([])
    for a, title in zip(axes[0], ("reactions branch", "port", "difference")):
        a.set_title(title)
    fig.suptitle(f"The needle at the end of the day ({HOURS} h)")
    path = os.path.join(folder, "solutes_needle_day_maps.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(path)


if __name__ == "__main__":
    main()
