"""
Tutorial 02, ported (example/legacy/tutorial_02_bc.ipynb): boundary conditions of MECHA's standard problem. The same
mature section (barrier 3, suberized endodermis) solved twice:

    potential   the soil at 0 hPa and the xylem at -5000 hPa (through k_xyl): MECHA gives the uptake Q and kr
    flow        the soil at 0 hPa and the uptake Q of the first case leaving through the xylem, split among the
                vessels as their areas: MECHA gives the potentials, the xylem's (their mean) and kr

Both describe the same physical state: they give the same uptake, and kr up to the spread of the vessels' potential.

    python tutorial_02_bc.py          (from MECHA-dev/example/tutorials; figure in ./outputs/)
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from openalea.mecha import plots

from tutorial_01_input import section

BARRIER = 3


def main():
    ds, mecha, library = section("granap")
    segment = int(ds.entity_ids("SubOrgan")[0])
    ds.set("barrier", np.full(ds.n_nodes(), float(BARRIER)))
    xylem = np.isin(ds.get("cgroup"), (13, 19, 20))

    mecha()                                                       # case 1: the xylem's potential
    potential = dict(uptake=float(ds.get("standard_uptake")[0]), kr=float(ds.get("kr")[0]), Kx=float(ds.get("Kx")[0]),
                     psi_xylem=mecha.standard_psi_xylem, frame=plots.port_frame(mecha, segment, libraries=(library,)))

    mecha.standard_axial_condition, mecha.standard_axial_flow = 2., potential["uptake"]
    mecha()                                                       # case 2: the uptake, as a flow into the xylem
    flow = dict(uptake=float(ds.get("standard_uptake")[0]), kr=float(ds.get("kr")[0]), Kx=float(ds.get("Kx")[0]),
                psi_xylem=float(np.asarray(ds.get("standard_psi"))[xylem].mean()),
                frame=plots.port_frame(mecha, segment, libraries=(library,)))

    print(f"{'':16s} {'potential':>12s} {'flow':>12s} {'Δ (%)':>8s}")
    for name, unit in (("psi_xylem", "hPa"), ("uptake", "cm3/d"), ("kr", "cm/hPa/d"), ("Kx", "cm4/hPa/d")):
        a, b = potential[name], flow[name]
        print(f"{name + ' (' + unit + ')':16s} {a:12.4e} {b:12.4e} {abs(a - b) / abs(a) * 100:8.3f}")

    norm = plt.Normalize(min(potential["frame"].psi.min(), flow["frame"].psi.min()), 0.)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, case, title in ((axes[0], potential, "xylem potential (-5000 hPa)"),
                            (axes[1], flow, f"xylem flow ({potential['uptake']:.3e} cm³/d)")):
        points = plots.psi_map(ax, case["frame"], title=f"ψ, {title}", norm=norm)
    fig.colorbar(points, ax=axes[:2], label="ψ (hPa)", shrink=0.8)
    plots.radial_profile(axes[2], potential["frame"], label="potential")
    plots.radial_profile(axes[2], flow["frame"], label="flow", linestyle="--")
    axes[2].legend(fontsize=7)
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, "tutorial_02_bc.png"), dpi=300)
    print(os.path.join(folder, "tutorial_02_bc.png"))


if __name__ == "__main__":
    main()
