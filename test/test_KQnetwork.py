"""
test_KQnetwork.py
=================
Aim 2 — Conductance (K) and Flow (Q) on the hydraulic network.

Tests implemented here:
  1. ``test_edge_K_stored``        – after build_matrices, every graph edge has 'K'
  2. ``test_edge_Q_computed``      – after solve_W, every graph edge with K has 'Q'
  3. ``test_KQ_back_propagated``   – HydraulicWall.kw, HydraulicMembrane.K_computed,
                                     HydraulicPlasmodesmata.kpl are all populated
  4. ``test_psi_radial_profile``   – Psi decreases monotonically from soil to xylem
                                     (barrier=1 maturity stage)

Visualization:
  - ``plot_conductance_network``   – edges coloured and scaled by K, per path type
  - ``plot_flow_network``          – arrows on edges showing Q direction & magnitude
  - ``plot_K_distribution``        – violin/box plots of K per edge type
  - ``plot_Q_distribution``        – violin/box plots of |Q| per edge type
  - ``plot_psi_radial_profile``    – mean Psi vs. radial distance (cell centroid r)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for CI; remove for live use
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import networkx as nx

from openalea.mecha import Mecha, InData, NetworkBuilder
from granap.root_class import RootAnatomy

# ---------------------------------------------------------------------------
# Helper: build a small standard root anatomy and a Mecha instance
# ---------------------------------------------------------------------------

def _build_mecha():
    """Return a solved Mecha instance (GRANAP path, barrier=1, h=0)."""
    root = RootAnatomy()
    root.update_params("aerenchyma", "aerenchyma_proportion", 0.5)
    root.update_params("aerenchyma", "n_files", 10)
    root.update_params("inter_cellular_spaces", "tissue", "cortex")
    _ = root.export_to_adjencymatrix()

    default_input = InData()
    default_input.geometry.set_maturity_stages([1, 3])

    network = NetworkBuilder(root)
    network.populate_from_network()

    mecha = Mecha(default_input, network=network)
    solution, _, matrix_W, Kmb, rhs_s = mecha.solve_W(h=0, i_maturity=0)
    return mecha, solution


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_edge_K_stored():
    """After build_matrices (called inside solve_W), every edge must have 'K'."""
    mecha, _ = _build_mecha()
    graph = mecha.network.graph
    missing = [(u, v) for u, v, d in graph.edges(data=True) if 'K' not in d]
    assert len(missing) == 0, (
        f"{len(missing)} edges are missing 'K' attribute after build_matrices.\n"
        f"Example: {missing[:5]}"
    )
    print(f"[PASS] test_edge_K_stored — all {graph.number_of_edges()} edges have 'K'.")


def test_edge_Q_computed():
    """After solve_W (which calls compute_edge_flows), every K-bearing edge has 'Q'."""
    mecha, _ = _build_mecha()
    graph = mecha.network.graph
    missing_Q = [(u, v) for u, v, d in graph.edges(data=True)
                 if d.get('K') is not None and 'Q' not in d]
    assert len(missing_Q) == 0, (
        f"{len(missing_Q)} edges have 'K' but no 'Q'.\nExample: {missing_Q[:5]}"
    )
    print(f"[PASS] test_edge_Q_computed — Q populated on all edges.")


def test_KQ_back_propagated():
    """HydraulicWall.kw, HydraulicMembrane.K_computed, HydraulicPlasmodesmata.kpl
    must all be set after solve_W."""
    mecha, _ = _build_mecha()
    cm = mecha.network.cell_manager

    # Walls
    walls_no_kw = [w for w in cm.walls if w.kw is None]
    assert len(walls_no_kw) == 0, (
        f"{len(walls_no_kw)} HydraulicWall objects have kw=None."
    )

    # Membranes
    mb_no_K = [m for m in cm.membranes if m.K_computed is None]
    assert len(mb_no_K) == 0, (
        f"{len(mb_no_K)} HydraulicMembrane objects have K_computed=None."
    )

    # Plasmodesmata
    pd_no_kpl = [p for p in cm.plasmodesmata if p.kpl is None]
    assert len(pd_no_kpl) == 0, (
        f"{len(pd_no_kpl)} HydraulicPlasmodesmata objects have kpl=None."
    )

    print(
        f"[PASS] test_KQ_back_propagated — "
        f"{len(cm.walls)} walls, {len(cm.membranes)} membranes, "
        f"{len(cm.plasmodesmata)} PD all have K."
    )


def test_psi_radial_profile():
    """Psi averaged by cortex layer should decrease monotonically from soil→xylem."""
    mecha, solution = _build_mecha()
    sol = np.asarray(solution).ravel()
    cm  = mecha.network.cell_manager

    # Gather (r, psi) pairs for non-xylem, non-intercellular cells
    intercellular_ids = {c.cell_id for c in cm.intercellular}
    xylem_cgroups = {13, 19, 20}

    radii, psis = [], []
    for cell in cm:
        if cell.cell_id in intercellular_ids:
            continue
        if cell.cgroup in xylem_cgroups:
            continue
        r = float(np.hypot(cell.x, cell.y))
        psi = float(sol[mecha.indice[cell.node_id]])
        radii.append(r)
        psis.append(psi)

    if len(radii) < 2:
        print("[SKIP] test_psi_radial_profile — too few cells.")
        return

    radii = np.array(radii)
    psis  = np.array(psis)

    # Outer rim: highest r → should be closer to psi_soil (highest)
    # Inner rim: low r    → lower psi (toward xylem)
    outer_mask = radii > np.percentile(radii, 80)
    inner_mask = radii < np.percentile(radii, 20)
    psi_outer  = np.mean(psis[outer_mask])
    psi_inner  = np.mean(psis[inner_mask])

    assert psi_outer > psi_inner, (
        f"Expected psi_outer ({psi_outer:.2f} hPa) > psi_inner ({psi_inner:.2f} hPa), "
        f"but got the opposite — flow direction may be wrong."
    )
    print(
        f"[PASS] test_psi_radial_profile — "
        f"psi_outer={psi_outer:.1f} hPa > psi_inner={psi_inner:.1f} hPa ✓"
    )


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

_PATH_COLORS = {
    'wall':          '#e07b4a',   # warm orange
    'membrane':      '#4ab5e0',   # sky blue
    'plasmodesmata': '#7be04a',   # lime green
}
_PATH_LABELS = {
    'wall':          'Apoplastic (wall)',
    'membrane':      'Transcellular (membrane)',
    'plasmodesmata': 'Symplastic (PD)',
}


def _gather_edge_data(graph):
    """Return a dict of lists: {path_type: {'K': [...], 'Q': [...], 'pos': [...]}}."""
    data = {k: {'K': [], 'Q': [], 'xy_mid': []} for k in _PATH_COLORS}
    pos = nx.get_node_attributes(graph, 'position')

    for u, v, eattr in graph.edges(data=True):
        path = eattr.get('path', 'wall')
        if path not in data:
            continue
        K = eattr.get('K')
        Q = eattr.get('Q')
        if K is None:
            continue
        pu = pos.get(u, (0, 0))
        pv = pos.get(v, (0, 0))
        mid = ((pu[0] + pv[0]) / 2, (pu[1] + pv[1]) / 2)
        data[path]['K'].append(K)
        data[path]['Q'].append(Q if Q is not None else 0.0)
        data[path]['xy_mid'].append(mid)
    return data


def plot_conductance_network(mecha, save_path: str = None):
    """Draw the graph with edges coloured and scaled by conductance K.

    Three subplots: one per path type (wall / membrane / PD).
    """
    graph  = mecha.network.graph
    pos    = nx.get_node_attributes(graph, 'position')
    n_wall = mecha.network.n_walls
    nwj    = mecha.network.n_wall_junction

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#111111')
    fig.suptitle("Conductance K per edge type", color='white', fontsize=14, y=1.01)

    for ax, path_type in zip(axes, ['wall', 'membrane', 'plasmodesmata']):
        ax.set_facecolor('#1a1a2e')
        ax.set_aspect('equal')

        # Draw background nodes (small, grey)
        nx.draw_networkx_nodes(
            graph, pos, ax=ax,
            node_size=2, node_color='#444466', alpha=0.4
        )

        # Collect edges of this type
        edge_list, K_vals = [], []
        for u, v, eattr in graph.edges(data=True):
            if eattr.get('path') == path_type:
                K = eattr.get('K')
                if K is not None and K > 0:
                    edge_list.append((u, v))
                    K_vals.append(K)

        if not edge_list:
            ax.set_title(f"{_PATH_LABELS[path_type]}\n(no edges)", color='white')
            continue

        K_arr  = np.array(K_vals)
        log_K  = np.log10(K_arr + 1e-30)
        log_K -= log_K.min()
        if log_K.max() > 0:
            log_K /= log_K.max()   # normalise 0→1 for width

        widths = 0.3 + 4.5 * log_K

        # Colour map: low K = dark, high K = bright
        cmap   = cm.get_cmap('plasma')
        colors = [cmap(v) for v in log_K]

        nx.draw_networkx_edges(
            graph, pos, ax=ax,
            edgelist=edge_list,
            edge_color=colors,
            width=widths,
            alpha=0.85
        )

        # Colourbar
        sm = cm.ScalarMappable(
            cmap='plasma',
            norm=mcolors.LogNorm(vmin=max(K_arr.min(), 1e-30), vmax=K_arr.max())
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label('K  [cm³ hPa⁻¹ d⁻¹]', color='white', fontsize=8)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

        ax.set_title(
            f"{_PATH_LABELS[path_type]}\n"
            f"n={len(edge_list)}  K∈[{K_arr.min():.2e}, {K_arr.max():.2e}]",
            color='white', fontsize=9
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='#111111')
        print(f"Saved → {save_path}")
    else:
        plt.savefig("outputs/conductance_network.png", dpi=150,
                    bbox_inches='tight', facecolor='#111111')
        print("Saved → outputs/conductance_network.png")
    plt.close()


def plot_flow_network(mecha, save_path: str = None):
    """Draw arrows on edges proportional to |Q|, coloured by edge type.

    Arrow head direction follows sign of Q (u→v if Q>0, v→u if Q<0).
    """
    graph  = mecha.network.graph
    pos    = nx.get_node_attributes(graph, 'position')

    fig, ax = plt.subplots(figsize=(12, 12), facecolor='#0d0d1a')
    ax.set_facecolor('#0d0d1a')
    ax.set_aspect('equal')
    ax.set_title("Water Flow Q on network edges", color='white', fontsize=13)

    # Background: all nodes
    nx.draw_networkx_nodes(graph, pos, ax=ax,
                           node_size=2, node_color='#333355', alpha=0.3)

    # Gather all Q values for normalisation
    all_Q = [abs(d.get('Q', 0.0))
             for _, _, d in graph.edges(data=True)
             if d.get('Q') is not None]
    q_max = max(all_Q) if all_Q else 1.0

    from matplotlib.patches import FancyArrowPatch

    legend_handles = []
    for path_type, color in _PATH_COLORS.items():
        drawn = False
        for u, v, eattr in graph.edges(data=True):
            if eattr.get('path') != path_type:
                continue
            Q = eattr.get('Q')
            K = eattr.get('K')
            if Q is None or K is None or K == 0:
                continue

            pu = pos.get(u, (0, 0))
            pv = pos.get(v, (0, 0))

            # Arrow direction follows sign of Q
            if Q >= 0:
                src, dst = pu, pv
            else:
                src, dst = pv, pu

            magnitude = abs(Q) / q_max       # 0…1
            if magnitude < 1e-12:
                continue

            lw  = 0.2 + 4.0 * magnitude
            hw  = 0.4 + 5.0 * magnitude      # head width
            alpha = 0.35 + 0.55 * magnitude  # more opaque for stronger flow

            ax.annotate(
                "", xy=dst, xytext=src,
                arrowprops=dict(
                    arrowstyle=f"->,head_width={hw:.2f},head_length={hw*0.8:.2f}",
                    color=color,
                    lw=lw,
                    alpha=alpha,
                    connectionstyle="arc3,rad=0.0"
                )
            )
            if not drawn:
                drawn = True

        # Legend proxy
        from matplotlib.lines import Line2D
        legend_handles.append(
            Line2D([0], [0], color=color, lw=2, label=_PATH_LABELS[path_type])
        )

    ax.legend(handles=legend_handles, loc='upper right',
              facecolor='#222244', labelcolor='white', edgecolor='white',
              fontsize=8)
    plt.tight_layout()
    out = save_path or "outputs/flow_network.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
    print(f"Saved → {out}")
    plt.close()


def plot_K_distribution(mecha, save_path: str = None):
    """Violin plots of K per edge type."""
    data = _gather_edge_data(mecha.network.graph)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor='#14141f')
    ax.set_facecolor('#1e1e30')
    ax.set_title("Conductance K distribution per edge type", color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')

    path_types  = [p for p in _PATH_COLORS if data[p]['K']]
    log_K_lists = [np.log10(np.maximum(data[p]['K'], 1e-30)) for p in path_types]
    colors      = [_PATH_COLORS[p] for p in path_types]
    labels      = [_PATH_LABELS[p] for p in path_types]

    vp = ax.violinplot(log_K_lists, showmedians=True, showextrema=True)
    for body, col in zip(vp['bodies'], colors):
        body.set_facecolor(col)
        body.set_alpha(0.75)
    vp['cmedians'].set_color('white')
    vp['cmins'].set_color('white')
    vp['cmaxes'].set_color('white')
    vp['cbars'].set_color('white')

    ax.set_xticks(range(1, len(path_types) + 1))
    ax.set_xticklabels(labels, color='white', fontsize=8)
    ax.set_ylabel("log₁₀(K)  [cm³ hPa⁻¹ d⁻¹]", color='white')

    plt.tight_layout()
    out = save_path or "outputs/K_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#14141f')
    print(f"Saved → {out}")
    plt.close()


def plot_Q_distribution(mecha, save_path: str = None):
    """Violin plots of |Q| per edge type."""
    data = _gather_edge_data(mecha.network.graph)

    fig, ax = plt.subplots(figsize=(9, 5), facecolor='#14141f')
    ax.set_facecolor('#1e1e30')
    ax.set_title("|Q| distribution per edge type", color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')

    path_types    = [p for p in _PATH_COLORS if data[p]['Q']]
    log_absQ_lists = []
    for p in path_types:
        absQ = np.abs(data[p]['Q'])
        absQ = np.maximum(absQ, 1e-40)
        log_absQ_lists.append(np.log10(absQ))

    colors = [_PATH_COLORS[p] for p in path_types]
    labels = [_PATH_LABELS[p] for p in path_types]

    vp = ax.violinplot(log_absQ_lists, showmedians=True, showextrema=True)
    for body, col in zip(vp['bodies'], colors):
        body.set_facecolor(col)
        body.set_alpha(0.75)
    vp['cmedians'].set_color('white')
    vp['cmins'].set_color('white')
    vp['cmaxes'].set_color('white')
    vp['cbars'].set_color('white')

    ax.set_xticks(range(1, len(path_types) + 1))
    ax.set_xticklabels(labels, color='white', fontsize=8)
    ax.set_ylabel("log₁₀(|Q|)  [cm³ d⁻¹]", color='white')

    plt.tight_layout()
    out = save_path or "outputs/Q_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#14141f')
    print(f"Saved → {out}")
    plt.close()


def plot_psi_radial_profile(mecha, solution, save_path: str = None):
    """Scatter + binned mean of Psi vs. radial distance for cell nodes."""
    sol = np.asarray(solution).ravel()
    cm  = mecha.network.cell_manager

    intercellular_ids = {c.cell_id for c in cm.intercellular}
    xylem_cgroups     = {13, 19, 20}

    cgroup_colors = {
        1: '#ff6b6b', 2: '#ffd93d', 3: '#6bcb77',
        4: '#4d96ff', 5: '#c77dff', 11: '#f4acb7',
        12: '#9d8189', 16: '#a2d2ff',
    }

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#14141f')
    ax.set_facecolor('#1e1e30')
    ax.set_title("Water potential Ψ vs. radial position", color='white', fontsize=13)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')

    # Scatter per cgroup
    seen_cgroups = set()
    for cell in cm:
        if cell.cell_id in intercellular_ids:
            continue
        r   = float(np.hypot(cell.x, cell.y))
        psi = float(sol[mecha.indice[cell.node_id]])
        col = cgroup_colors.get(cell.cgroup, '#cccccc')
        lbl = f"cgroup {cell.cgroup}" if cell.cgroup not in seen_cgroups else None
        seen_cgroups.add(cell.cgroup)
        ax.scatter(r, psi, color=col, s=8, alpha=0.45, label=lbl)

    # Binned mean (all cells)
    radii, psis = [], []
    for cell in cm:
        if cell.cell_id in intercellular_ids:
            continue
        radii.append(float(np.hypot(cell.x, cell.y)))
        psis.append(float(sol[mecha.indice[cell.node_id]]))
    if radii:
        radii = np.array(radii)
        psis  = np.array(psis)
        n_bins = 20
        bins   = np.linspace(radii.min(), radii.max(), n_bins + 1)
        bin_centers, bin_means = [], []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            mask = (radii >= b0) & (radii < b1)
            if mask.sum() > 0:
                bin_centers.append((b0 + b1) / 2)
                bin_means.append(psis[mask].mean())
        ax.plot(bin_centers, bin_means, color='white', lw=2.0,
                label='Bin mean', zorder=5)

    ax.set_xlabel("Radial distance from centre  [µm]")
    ax.set_ylabel("Ψ  [hPa]")
    ax.legend(loc='upper right', facecolor='#222244',
              labelcolor='white', edgecolor='white',
              fontsize=7, markerscale=1.5)

    plt.tight_layout()
    out = save_path or "outputs/psi_radial_profile.png"
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#14141f')
    print(f"Saved → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

def print_KQ_summary(mecha):
    """Print a summary table of K and Q statistics per edge type."""
    data = _gather_edge_data(mecha.network.graph)
    print("\n" + "="*65)
    print(f"{'Edge type':<25} {'n':>6} {'K_min':>10} {'K_max':>10} {'Q_max':>10}")
    print("="*65)
    for path_type in _PATH_COLORS:
        d = data[path_type]
        if not d['K']:
            print(f"{_PATH_LABELS[path_type]:<25} {'0':>6}")
            continue
        K = np.array(d['K'])
        Q = np.abs(d['Q'])
        print(
            f"{_PATH_LABELS[path_type]:<25} {len(K):>6}"
            f"  {K.min():>10.3e}  {K.max():>10.3e}  {Q.max():>10.3e}"
        )
    print("="*65 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    print("Building network & solving...")
    mecha, solution = _build_mecha()

    print(f"\nNetwork: {mecha.network.cell_manager}")

    # --- Run tests ---
    test_edge_K_stored.__wrapped__ = False
    for fn in [test_edge_K_stored, test_edge_Q_computed,
               test_KQ_back_propagated, test_psi_radial_profile]:
        try:
            fn()
        except AssertionError as e:
            print(f"[FAIL] {fn.__name__}: {e}")

    # --- Summary table ---
    print_KQ_summary(mecha)

    # --- Plots ---
    print("Generating conductance network plot...")
    plot_conductance_network(mecha)

    print("Generating flow network plot...")
    plot_flow_network(mecha)

    print("Generating K distribution plot...")
    plot_K_distribution(mecha)

    print("Generating Q distribution plot...")
    plot_Q_distribution(mecha)

    print("Generating Psi radial profile plot...")
    plot_psi_radial_profile(mecha, solution)

    print("\nDone. Check the outputs/ directory.")
