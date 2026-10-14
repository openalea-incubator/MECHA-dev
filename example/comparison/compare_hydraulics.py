"""
The port of MECHA against the legacy, drawn (devplan_metafspm.md, §8): on the same GRANAP section (a one-segment root,
200 µm long), each edge's conductance K, each node's solved potential ψ and each edge's flow Q, legacy | port |
difference, at each barrier stage; besides the quantitative comparison tests, a check of their visual coherence.

    python compare_hydraulics.py [dicot]       (from MECHA-dev/example/comparison; figures in ./outputs/)
"""
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "..", "test"), os.path.join(HERE, "..", "..", "test", "comparison")]

from legacy_run import legacy_section                                          # noqa: E402
from plants import anatomy_component, anatomy_data_structure, root, template_organ  # noqa: E402

from openalea.mecha.component import MechaHydraulics                          # noqa: E402
from openalea.mecha.geometry import section_parts                             # noqa: E402

BARRIERS = (1, 3)


def port_stage(mecha, segment, barrier):
    """The port's K, ψ and Q on the segment's section at *barrier*: the component's own (its standard problem)."""
    ds = mecha.data_structure
    ds.set("barrier", np.full(ds.n_nodes(), float(barrier)))
    mecha()
    nodes, edges = section_parts(ds)[segment]
    net = mecha.network()
    e = edges[np.isin(net.path[edges], (0, 3)) | net.junction_edge[edges]]       # the edges of MECHA's matrix
    return dict(net=net, nodes=nodes, edges=e, K=np.asarray(ds.get("conductance"))[e], psi=np.asarray(ds.get("psi")),
                Q=np.asarray(ds.get("flow"))[e], kr=float(np.asarray(ds.get("kr"))[0]))


def draw_edges(ax, segments, values, cmap, norm, width=0.4):
    lines = LineCollection(segments, cmap=cmap, norm=norm, linewidths=width)
    lines.set_array(values)
    ax.add_collection(lines)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return lines


def main(preset=None):
    g, ds, anatomy = anatomy_data_structure(root(1), anatomy_component(preset))
    segment = int(ds.entity_ids("SubOrgan")[0])
    height = float(np.asarray(ds.get("length"))[0]) * 1.0E6
    run = legacy_section(template_organ(anatomy, g, segment), barriers=BARRIERS, height=height)
    ds.register("barrier", np.ones(ds.n_nodes()), location="node")
    mecha = MechaHydraulics(data_structure=ds)
    graph = run.network.graph
    position = {n: graph.nodes[n]["position"] for n in graph.nodes}
    folder = os.path.join(HERE, "outputs")
    os.makedirs(folder, exist_ok=True)
    name = "dicot_root" if preset else "monocot_root"
    for i, barrier in enumerate(BARRIERS):
        port = port_stage(mecha, segment, barrier)
        stage = run.stages[i]
        geo = mecha.section_geometries()[segment][2]
        xy = np.zeros((ds.n_nodes(), 2))
        xy[port["nodes"], 0], xy[port["nodes"], 1] = geo.x, geo.y
        net, e = port["net"], port["edges"]
        port_segments = np.stack([xy[net.a[e]], xy[net.b[e]]], axis=1)
        keys = list(stage["edge_K"])
        legacy_segments = np.array([[position[u], position[v]] for u, v in keys])
        legacy_K = np.array([stage["edge_K"][k] for k in keys])
        legacy_Q = np.array([stage["edge_Q"].get(k, np.nan) for k in keys])
        # the port's edges matched to the legacy's (by their ends' positions), for the differences
        legacy_by_ends = {tuple(sorted((tuple(position[u]), tuple(position[v])))): j for j, (u, v) in enumerate(keys)}
        match = np.array([legacy_by_ends.get(tuple(sorted((tuple(xy[a]), tuple(xy[b])))), -1)
                          for a, b in zip(net.a[e], net.b[e])])
        same_direction = np.array([j >= 0 and tuple(xy[a]) == tuple(position[keys[j][0]])
                                   for j, a in zip(match, net.a[e])])
        legacy_Q_on_port = np.where(same_direction, 1., -1.) * legacy_Q[np.maximum(match, 0)]
        legacy_psi = stage["solution"]
        node_xy_legacy = np.array([position[n] for n in range(legacy_psi.size)])

        fig, axes = plt.subplots(3, 3, figsize=(15, 14.5))
        logK = matplotlib.colors.Normalize(*np.percentile(np.log10(np.maximum(legacy_K, 1e-30)), (1, 100)))
        lines = draw_edges(axes[0, 0], legacy_segments, np.log10(np.maximum(legacy_K, 1e-30)), "viridis", logK)
        draw_edges(axes[0, 1], port_segments, np.log10(np.maximum(port["K"], 1e-30)), "viridis", logK)
        fig.colorbar(lines, ax=axes[0, :2], label="log10 K (cm³ hPa⁻¹ d⁻¹)", shrink=0.8)
        rel_K = abs(port["K"] - legacy_K[np.maximum(match, 0)]) / np.maximum(abs(legacy_K[np.maximum(match, 0)]),
                                                                            1e-300)
        lines = draw_edges(axes[0, 2], port_segments, rel_K, "magma", matplotlib.colors.Normalize(0, max(rel_K.max(),
                                                                                                          1e-16)))
        fig.colorbar(lines, ax=axes[0, 2], label="|ΔK| / K", shrink=0.8)

        psi_norm = matplotlib.colors.Normalize(np.nanmin(legacy_psi), np.nanmax(legacy_psi))
        sc = axes[1, 0].scatter(node_xy_legacy[:, 0], node_xy_legacy[:, 1], c=legacy_psi, s=1.5, cmap="coolwarm_r",
                                norm=psi_norm)
        axes[1, 1].scatter(xy[port["nodes"], 0], xy[port["nodes"], 1], c=port["psi"][port["nodes"]], s=1.5,
                           cmap="coolwarm_r", norm=psi_norm)
        fig.colorbar(sc, ax=axes[1, :2], label="ψ (hPa)", shrink=0.8)
        legacy_id = np.empty(ds.n_nodes(), dtype=np.int64)
        by_position = {tuple(position[n]): n for n in graph.nodes}
        legacy_id[port["nodes"]] = [by_position[tuple(p)] for p in xy[port["nodes"]]]
        d_psi = abs(port["psi"][port["nodes"]] - legacy_psi[legacy_id[port["nodes"]]])
        sc = axes[1, 2].scatter(xy[port["nodes"], 0], xy[port["nodes"], 1], c=d_psi, s=1.5, cmap="magma")
        fig.colorbar(sc, ax=axes[1, 2], label="|Δψ| (hPa)", shrink=0.8)

        q_norm = matplotlib.colors.Normalize(*np.nanpercentile(np.log10(np.maximum(abs(legacy_Q), 1e-30)), (5, 100)))
        lines = draw_edges(axes[2, 0], legacy_segments, np.log10(np.maximum(abs(legacy_Q), 1e-30)), "plasma", q_norm)
        draw_edges(axes[2, 1], port_segments, np.log10(np.maximum(abs(port["Q"]), 1e-30)), "plasma", q_norm)
        fig.colorbar(lines, ax=axes[2, :2], label="log10 |Q| (cm³ d⁻¹)", shrink=0.8)
        d_Q = abs(port["Q"] - legacy_Q_on_port)
        lines = draw_edges(axes[2, 2], port_segments, d_Q, "magma", matplotlib.colors.Normalize(0, max(np.nanmax(d_Q),
                                                                                                    1e-30)))
        fig.colorbar(lines, ax=axes[2, 2], label="|ΔQ| (cm³ d⁻¹)", shrink=0.8)

        for ax, title in zip(axes[0], ("legacy", "port", "difference")):
            ax.set_title(title)
        for ax, label in zip(axes[:, 0], ("conductance", "potential", "flow")):
            ax.set_ylabel(label)
        fig.suptitle(f"{name}, barrier {barrier}: max |ΔK|/K {rel_K.max():.1e}, max |Δψ| {d_psi.max():.1e} hPa, "
                     f"max |ΔQ| {np.nanmax(d_Q):.1e} cm³/d (|Q| up to {np.nanmax(abs(legacy_Q)):.1e})\n"
                     f"kr: legacy {stage['kr']:.6e}, port {port['kr']:.6e} cm hPa⁻¹ d⁻¹")
        path = os.path.join(folder, f"hydraulics_{name}_barrier{barrier}.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main("for_dicot_root" if "dicot" in sys.argv[1:] else None)
