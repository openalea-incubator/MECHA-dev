"""
The seedling SPAC example's figures: the plant at SubOrgan scale (its segments coloured by their xylem's ψ) over
the soil's ψ, one section per organ type coloured by ψ, the potential along the water's path (soil, root, stem, leaf,
air), the roots' kr and effective kr; and a text summary of the potential drops.
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

HPA_PER_MPA = 1e4
XYLEM = (13, 19, 20)


def _segments(ds):
    """Per segment: (label name, its nodes' mask index, x1, x2, z1, z2, y1, y2)."""
    g = ds.mtg
    labels = g.property("label")
    names = {g.labels.SubOrgan.RootSegment: "root", g.labels.SubOrgan.StemElement: "stem",
             g.labels.SubOrgan.LeafElement: "leaf"}
    segments = ds.entity_ids("SubOrgan")
    coords = {name: np.array([g.property(name)[int(v)] for v in segments]) for name in
              ("x1", "x2", "y1", "y2", "z1", "z2")}
    return segments, [names.get(labels[int(v)], "?") for v in segments], coords


def section_table(ds):
    """Per segment: its organ, its xylem's mean ψ and its mean ψ (MPa)."""
    segments, kinds, coords = _segments(ds)
    owner, psi, cgroup = ds.owner("SubOrgan"), np.asarray(ds.get("psi")) / HPA_PER_MPA, ds.get("cgroup")
    xylem = np.isin(cgroup, XYLEM)
    psi_x = np.array([psi[(owner == i) & xylem].mean() for i in range(segments.size)])
    psi_mean = np.array([psi[owner == i].mean() for i in range(segments.size)])
    return segments, kinds, coords, psi_x, psi_mean


def plant_over_soil(built, path):
    plant = built.populations[0].data_structure
    soil = next(model for model in built.environment if hasattr(model, "grid")).grid
    segments, kinds, coords, psi_x, _ = section_table(plant)
    fig, ax = plt.subplots(figsize=(7, 8))
    soil_psi = np.asarray(soil.get("water_potential")).reshape(soil.shape)
    y_index = soil.shape[1] // 2
    extent = (0, soil.shape[0] * soil.dx[0], -soil.shape[2] * soil.dx[2], 0)
    image = ax.imshow(soil_psi[:, y_index, :].T, extent=extent, origin="upper", cmap="Blues_r", aspect="equal")
    fig.colorbar(image, ax=ax, label="soil ψ (MPa), slice through the plant", shrink=0.6, location="bottom")
    lines = np.stack([np.c_[coords["x1"], coords["z1"]], np.c_[coords["x2"], coords["z2"]]], axis=1)
    widths = np.array([{"root": 1.5, "stem": 4., "leaf": 3.}[k] for k in kinds])
    collection = LineCollection(lines, cmap="coolwarm_r", linewidths=widths)
    collection.set_array(psi_x)
    ax.add_collection(collection)
    fig.colorbar(collection, ax=ax, label="xylem ψ (MPa)", shrink=0.6)
    ax.set(xlabel="x (m)", ylabel="z (m)", title="the seedling (side view) over the soil")
    ax.set_ylim(extent[2], max(0.1, coords["z2"].max() * 1.1))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def sections_by_organ(built, path):
    """One section per organ type (the root's at the collar, the stem's base, the first leaf element) by ψ."""
    plant = built.populations[0].data_structure
    segments, kinds, _, _, _ = section_table(plant)
    owner, psi = plant.owner("SubOrgan"), np.asarray(plant.get("psi")) / HPA_PER_MPA
    x, y = plant.get("section_x"), plant.get("section_y")
    view = plant.to_graph_view()
    flow = np.asarray(plant.get("flow"))
    chosen = [kinds.index(k) for k in ("root", "stem", "leaf")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8))
    for ax, i in zip(axes, chosen):
        nodes = owner == i
        edges = np.flatnonzero((owner[view.tail] == i) & (owner[view.head] == i) & (flow != 0))
        q = np.log10(abs(flow[edges]))
        lines = np.stack([np.c_[x[view.tail[edges]], y[view.tail[edges]]],
                          np.c_[x[view.head[edges]], y[view.head[edges]]]], axis=1)
        collection = LineCollection(lines, cmap="Greys", linewidths=0.3, alpha=0.7)
        collection.set_array(q)
        collection.set_clim(np.percentile(q, 5), q.max())
        ax.add_collection(collection)
        points = ax.scatter(x[nodes], y[nodes], c=psi[nodes], s=0.6, cmap="coolwarm_r")
        fig.colorbar(points, ax=ax, label="ψ (MPa)", shrink=0.8)
        ax.set_aspect("equal")
        ax.set_title(f"{kinds[i]} section {int(segments[i])} (edges: log |Q|, grey)")
        ax.set_xlabel("mm")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def potential_profile(built, path):
    """Mean ψ along the path: soil at the roots, root xylem, stem xylem, leaf xylem, leaf cells, stomatal pores, air."""
    plant = built.populations[0].data_structure
    segments, kinds, _, psi_x, _ = section_table(plant)
    owner, psi = plant.owner("SubOrgan"), np.asarray(plant.get("psi")) / HPA_PER_MPA
    kind_of_node = np.array(kinds)[owner]
    soil = plant.get("soil_conductance") > 0
    barrier = np.asarray(plant.get("barrier"))
    conducting = np.array([barrier[owner == i][0] > 0 for i in range(segments.size)])   # mature xylem (not tips)
    pores = plant.get("evaporation_conductance") > 0
    leaf = kind_of_node == "leaf"
    steps = [("soil (at the roots)", float(np.mean(plant.get("soil_psi")[soil])) / HPA_PER_MPA),
             ("root surface", psi[soil].mean()),
             ("root xylem", psi_x[(np.array(kinds) == "root") & conducting].mean()),
             ("stem xylem", psi_x[np.array(kinds) == "stem"].mean()),
             ("leaf xylem", psi_x[np.array(kinds) == "leaf"].mean()),
             ("leaf cells", psi[leaf & (plant.get("kind") == 0) & ~pores].mean()),
             ("stomatal pores", psi[pores].mean())]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(len(steps)), [v for _, v in steps], "o-")
    ax.set_xticks(range(len(steps)), [name for name, _ in steps], rotation=25, ha="right")
    ax.set_ylabel("mean ψ (MPa)")
    ax.set_title("the water potential along the path (the air at "
                 f"{float(np.asarray(plant.get('air_psi'))[0]) / HPA_PER_MPA:.1f} MPa)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return steps


def root_kr(built, path):
    plant = built.populations[0].data_structure
    segments, kinds, _, _, _ = section_table(plant)
    roots = np.array(kinds) == "root"
    distance = np.asarray(plant.get("distance_from_tip"))
    if distance.size != segments.size:
        distance = np.array([distance[plant.owner("SubOrgan") == i][0] for i in range(segments.size)])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(distance[roots] * 100, plant.get("kr")[roots], "o", label="kr (MECHA's standard problem)")
    ax.semilogy(distance[roots] * 100, plant.get("kr_effective")[roots], "s", mfc="none", label="kr effective (plant)")
    ax.set(xlabel="distance from the root tip (cm)", ylabel="cm hPa⁻¹ d⁻¹",
           title="the roots' radial conductivity (barriers: 0 at the tip, 1 to 10 cm, 3 beyond)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def all_figures(built, folder):
    os.makedirs(folder, exist_ok=True)
    plant_over_soil(built, os.path.join(folder, "plant_over_soil.png"))
    sections_by_organ(built, os.path.join(folder, "sections_by_organ.png"))
    steps = potential_profile(built, os.path.join(folder, "potential_profile.png"))
    root_kr(built, os.path.join(folder, "root_kr.png"))
    print("ψ along the path (MPa): " + ", ".join(f"{name} {value:.2f}" for name, value in steps))
    print(folder)
