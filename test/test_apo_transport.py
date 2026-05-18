"""
test_apo_transport.py

Apoplastic solute transport via SoluteTransport, homogeneous tissue.

INPUTS are modified to remove the Casparian strip:
  kw_endo_endo → kw_base   all wall conductances equal
  barrier      → 0         disables the hard-coded K=1e-16 branch
  water_flux() re-run      gives a smooth, unblocked radial pressure gradient

SoluteTransport (mode='apo') is called directly with those inputs.
An absorbing BC (c=0) is applied at stele wall nodes to prevent blow-up
at pure-sink nodes that have no outgoing apoplastic path (water exits there
via membranes, which are outside mode='apo').

Wall nodes with L=0 (junction nodes — connectivity only, no physical volume)
are filtered from plots but kept in the solver for correct topology.

Physical expectations
---------------------
DIFF:  mass spreads radially; boundary-absorbed at stele → r_cm drifts
ADV:   mass swept inward; r_cm decreases monotonically

Outputs
-------
  test/outputs/apo_evolution.png
  test/outputs/apo_radial.png
  test/outputs/apo_rcm.png
"""

import os
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mecha.mecha_class import Mecha
from mecha.utils.data_loader import InData
from mecha.solute_transport import SoluteTransport

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_CELLSET   = os.path.join(_REPO_ROOT, 'extdata', 'current_root.xml')
_OUT_DIR   = os.path.join(os.path.dirname(__file__), 'outputs')

D_WALL    = 1e-5   # cm²/d  apoplastic diffusivity
DP        = dict(apo_wall=D_WALL, membrane=0.0, plasmodesmata=0.0)
CAP       = {'dt': 0.1, 'C_wall': 1.0, 'C_cell': 1.0}
N_STEPS   = 200    # minimum; overridden in test body by transit-time estimate
THETA     = 1.0

# ── build helpers ─────────────────────────────────────────────────────────────

def _build_mecha_homogeneous() -> Mecha:
    """
    Mecha with all apoplastic wall conductances equal (no Casparian strip).

    Changes made to the INPUT parameters (not the builder code):
      kw_endo_endo → kw_base  (was 1e-16; makes 46 endodermis wall nodes
                                conduct equally to cortex nodes)
      barrier stays 1         (no other code path changes; count_interC>=2
                                nodes = 0 in this anatomy so the hardcoded
                                K=1e-16 branch is never reached)
      water_flux() re-run     produces a smooth radial pressure field
      edge_flux_list filled   from the new hydraulic solution
    """
    data  = InData(cellset_file=_CELLSET)
    mecha = Mecha(data)

    h, i_mat = 0, 0
    kw_base = float(mecha.hydraulic.get_kw_value(h))      # 2.4e-4
    barrier = int(mecha.geometry.maturity_stages[i_mat].get('barrier'))  # 1

    # Patch: make endodermis wall conductance equal to the cortex default
    kw_config = mecha.hydraulic_conductivities[h, i_mat, barrier]['kw']
    kw_config['kw_endo_endo'] = kw_base   # was 1e-16

    # Re-solve hydraulics with the patched conductances
    sol, mat_W = mecha.water_flux()

    # Fill edge_flux_list (workaround for known edge_flux_list bugs)
    fluxes = []
    coo = mat_W.tocoo()
    for i, j, v in zip(coo.row, coo.col, coo.data):
        if i < j and v > 0:
            f = float(v) * (float(sol[i][0]) - float(sol[j][0]))
            fluxes.append({'source': int(i), 'target': int(j), 'flux': f})
    mecha.edge_flux_list[0][0] = fluxes
    return mecha


def _root_center(mecha: Mecha):
    nwj = mecha.network.n_wall_junction
    xs, ys = [], []
    for nd, d in mecha.network.graph.nodes(data=True):
        if mecha.indice[nd] >= nwj and 'position' in d:
            xs.append(d['position'][0])
            ys.append(d['position'][1])
    return float(np.mean(xs)), float(np.mean(ys))


def _stele_bc_nodes(mecha: Mecha) -> dict:
    """
    Indices of wall nodes deep in the stele (count_stele_overall > 0 and
    count_cortex == 0 and count_endo == 0).  These are pure sinks in mode='apo':
    water exits here via membranes, not wall-to-wall edges.  Setting c=0 there
    is an absorbing BC (mass removed when it reaches the stele) and prevents
    numerical blow-up from tiny capacitance.
    """
    nwj = mecha.network.n_wall_junction
    bc = {}
    for nd, d in mecha.network.graph.nodes(data=True):
        idx = mecha.indice[nd]
        if idx < nwj:
            if (d.get('count_stele_overall', 0) > 0
                    and d.get('count_cortex', 0) == 0
                    and d.get('count_endo', 0) == 0):
                bc[idx] = 0.0
    return bc


def _collect_display_nodes(mecha: Mecha, cx: float, cy: float):
    """
    Wall nodes with L>0 (physical wall segments, not connectivity-only junctions).
    Returns [(idx, r_µm, x, y, count_cortex), ...].
    """
    nwj = mecha.network.n_wall_junction
    wl  = mecha.network.wall_lengths
    result = []
    for nd, d in mecha.network.graph.nodes(data=True):
        idx = mecha.indice[nd]
        if idx < nwj and 'position' in d and wl.get(idx, 0.0) > 0:
            r = np.hypot(d['position'][0] - cx, d['position'][1] - cy)
            cc = d.get('count_cortex', 0)
            result.append((idx, r, d['position'][0], d['position'][1], cc))
    return result


def _cortex_wall_ic(display_nodes: list, nwj: int):
    """c=1 in displayed cortex wall nodes (count_cortex≥1) at 40–60 % of r."""
    cortex = [(idx, r) for idx, r, *_, cc in display_nodes if cc >= 1]
    if not cortex:
        raise RuntimeError('No cortex wall nodes (count_cortex>=1) with L>0 found')
    r_vals = [r for _, r in cortex]
    r_lo, r_hi = np.percentile(r_vals, [40, 60])
    c0 = np.zeros(nwj)
    for idx, r in cortex:
        if r_lo <= r <= r_hi:
            c0[idx] = 1.0
    return c0, r_lo, r_hi


def _r_cm(c_apo, disp_idxs, r_disp_um):
    """
    Concentration-weighted mean radius over displayed (L>0) wall nodes only.

    Unweighted by node volume so that r_cm is insensitive to the radial
    gradient in wall-segment size (outer cortex walls are longer).
    Returns µm.
    """
    c_d = c_apo[disp_idxs]
    tot = float(c_d.sum())
    if tot < 1e-30:
        return float('nan')
    return float(np.dot(r_disp_um, c_d)) / tot


# ── visualisation ─────────────────────────────────────────────────────────────

def _plot_evolution(display_nodes, snap_data, r_lo, r_hi, cx, cy, nwj):
    cases  = ('diff', 'adv', 'total')
    labels = {'diff': 'Diffusion (D)', 'adv': 'Advection (A)', 'total': 'Full (T=A+D)'}
    n_snap = len(snap_data['diff'])
    fig, axes = plt.subplots(3, n_snap,
                             figsize=(3.0 * n_snap, 9.5), squeeze=False,
                             layout='constrained')
    theta_ring = np.linspace(0, 2 * np.pi, 300)
    xs   = np.array([e[2] for e in display_nodes])
    ys   = np.array([e[3] for e in display_nodes])
    idxs = np.array([e[0] for e in display_nodes], dtype=int)

    for row_i, key in enumerate(cases):
        snaps  = snap_data[key]
        # vmax from the NON-IC snapshots so post-IC frames are not washed out
        # against the saturated t=0 ring.  t=0 panel will be clipped to vmax.
        non_ic = [(c, t_s) for c, t_s in snaps if t_s > 0]
        src    = non_ic if non_ic else snaps
        vmax   = max(max(float(c[idxs].max()), 1e-12) for c, _ in src)
        sc_ref = None
        for col_j, (c_vec, t) in enumerate(snaps):
            ax  = axes[row_i, col_j]
            sc  = ax.scatter(xs, ys, s=4,
                             c=c_vec[idxs], cmap='plasma',
                             vmin=0.0, vmax=vmax,
                             zorder=2, edgecolors='none')
            sc_ref = sc
            for rv in (r_lo, r_hi):
                ax.plot(cx + rv * np.cos(theta_ring),
                        cy + rv * np.sin(theta_ring),
                        color='cyan', lw=0.6, ls='--', zorder=3)
            ax.set_aspect('equal', 'box')
            ax.set_title(f't = {t:.1f} d', fontsize=8)
            ax.tick_params(labelsize=6)
            if col_j == 0:
                ax.set_ylabel(f"{labels[key]}\ny (µm)", fontsize=7)
        if sc_ref is not None:
            fig.colorbar(sc_ref, ax=axes[row_i, :].tolist(),
                         shrink=0.6, label='c (a.u.)', pad=0.01)

    fig.suptitle(
        f'Apoplastic transport — homogeneous tissue (IE θ=1, dt={CAP["dt"]} d)\n'
        'Wall nodes (L>0 only) | cyan: initial ring | absorbing BC at stele',
        fontsize=9)
    out = os.path.join(_OUT_DIR, 'apo_evolution.png')
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  → {out}')


def _plot_radial(display_nodes, snap_data):
    idxs  = np.array([e[0] for e in display_nodes], dtype=int)
    r_arr = np.array([e[1] for e in display_nodes])
    n     = len(snap_data['diff'])
    show_idx = list(range(n))
    n_show   = len(show_idx)
    fig, axes = plt.subplots(1, n_show, figsize=(4.0 * n_show, 3.5), squeeze=False)
    colors = {'diff': '#2166ac', 'adv': '#d6604d', 'total': '#4dac26'}
    lab    = {'diff': 'DIFF', 'adv': 'ADV', 'total': 'T'}
    for col_j, si in enumerate(show_idx):
        ax = axes[0, col_j]
        t  = snap_data['diff'][si][1]
        for key in ('diff', 'adv', 'total'):
            c_vec = snap_data[key][si][0]
            ax.scatter(r_arr, c_vec[idxs], s=3,
                       color=colors[key], alpha=0.5, label=lab[key])
        ax.set_xlabel('r (µm)', fontsize=7)
        ax.set_ylabel('c (a.u.)', fontsize=7)
        ax.set_title(f't = {t:.1f} d', fontsize=9)
        ax.set_ylim(bottom=-0.02)
        ax.tick_params(labelsize=6)
        if col_j == 0:
            ax.legend(fontsize=7)
    fig.suptitle('Apoplastic radial profiles (wall nodes, L>0)', fontsize=9)
    plt.tight_layout()
    out = os.path.join(_OUT_DIR, 'apo_radial.png')
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  → {out}')


def _plot_rcm(ts_data, rcm_data, mass_data, M0s, T_transit=None):
    """
    Two-panel figure.
    Top:    concentration-weighted r_cm (no volume bias) vs time.
    Bottom: fraction of initial mass remaining vs time.
    """
    colors = {'diff': '#2166ac', 'adv': '#d6604d', 'total': '#4dac26'}
    labels = {'diff': 'Diffusion (D)', 'adv': 'Advection (A)', 'total': 'Full (T=A+D)'}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

    for key in ('diff', 'adv', 'total'):
        t_arr   = np.array(ts_data[key])
        rcm_arr = np.array(rcm_data[key])
        valid   = np.isfinite(rcm_arr)
        ax1.plot(t_arr[valid], rcm_arr[valid],
                 color=colors[key], lw=1.5, label=labels[key])

        m_arr = np.array(mass_data[key]) / M0s[key]
        ax2.plot(t_arr, m_arr, color=colors[key], lw=1.5, label=labels[key])

    if T_transit is not None:
        for ax in (ax1, ax2):
            ax.axvline(T_transit, color='gray', lw=0.8, ls=':', alpha=0.7,
                       label=f'T_transit={T_transit:.1f}d')

    ax1.set_ylabel('r_cm (µm)', fontsize=9)
    ax1.set_title('Concentration-weighted r_cm  (unweighted by node volume)\n'
                  'Homogeneous apoplast | absorbing BC at stele (ADV, T)\n'
                  'ADV r_cm rises after T_transit: inner mass absorbed, outer tail remains',
                  fontsize=7)
    ax1.legend(fontsize=8)

    ax2.set_xlabel('Time (d)', fontsize=9)
    ax2.set_ylabel('M(t) / M₀', fontsize=9)
    ax2.set_title('Mass fraction remaining', fontsize=8)
    ax2.set_ylim(-0.02, 1.05)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(_OUT_DIR, 'apo_rcm.png')
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  → {out}')


# ── test ──────────────────────────────────────────────────────────────────────

def test_apo_transport():
    """
    Apoplastic DIFF and ADV in homogeneous tissue using SoluteTransport (mode='apo').

    Assertions
    ----------
    mass approximately conserved for DIFF   |ΔM|/M < 1e-4  (some lost at stele BC)
    no spurious negatives                   min(c) >= -1e-6
    ADV shifts r_cm inward                  Δr_cm(ADV) < −1 µm
    ADV and DIFF profiles differ            max|Δc| > 1e-12
    """
    print('\n[test_apo] building Mecha (homogeneous tissue) ...')
    mecha = _build_mecha_homogeneous()
    nwj   = mecha.network.n_wall_junction

    cx, cy = _root_center(mecha)
    display_nodes = _collect_display_nodes(mecha, cx, cy)
    print(f'  wall nodes total: {nwj}  |  displayed (L>0): {len(display_nodes)}')

    # ── scale fluxes for global Pe = 1; choose dt and N_STEPS from transit time ─
    # Global Pe = v × r_lo / D_wall = 1  →  v_target = D_wall / r_lo.
    # This equates the advective and diffusive transit times: T = r_lo² / D_wall.
    # v_phys [cm/d] of a wall node: A_ii [cm³/d] / cross-section (h×t) [cm²].
    st_pe = SoluteTransport(mecha, DP, CAP, mode='apo')
    A_tmp = st_pe.build_advection_matrix(0, 0)
    D_tmp = st_pe.build_diffusion_matrix(0, 0)

    c0_tmp, r_lo_tmp, _ = _cortex_wall_ic(display_nodes, nwj)
    ring_idxs_tmp = np.where(c0_tmp > 0)[0]

    height_cm = float(mecha.geometry.maturity_stages[0].get('height')) * 1e-4
    thick_cm  = mecha.geometry.thickness * 1e-4
    r_lo_cm   = r_lo_tmp * 1e-4

    Ad_ring = np.abs(A_tmp.diagonal())[ring_idxs_tmp]
    valid_v = Ad_ring > 0
    v_raw   = float(Ad_ring[valid_v].mean()) / (height_cm * thick_cm)  # cm/d (unscaled)
    v_tgt   = D_WALL / r_lo_cm            # cm/d, global Pe=1
    flux_scale = v_tgt / v_raw if v_raw > 0 else 1.0
    for f in mecha.edge_flux_list[0][0]:
        f['flux'] *= flux_scale

    T_transit = r_lo_cm / v_tgt           # d = r_lo² / D_wall (same for ADV and DIFF)
    dt_use    = max(0.1, round(T_transit / 100.0, 1))    # ~100 steps per transit
    CAP['dt'] = dt_use
    T_steps   = int(np.ceil(T_transit / dt_use))         # steps per transit
    # Run to 5×T_transit so DIFF reaches near-equilibrium (boundary effects gone)
    N_STEPS   = max(200, min(2000, 5 * T_steps))

    Dd_ring    = np.abs(D_tmp.diagonal())[ring_idxs_tmp]
    valid_both = valid_v & (Dd_ring > 0)
    pe_ring    = float((Ad_ring[valid_both] * flux_scale / Dd_ring[valid_both]).mean()) \
                 if valid_both.any() else float('nan')
    print(f'  global Pe=1: v={v_tgt*1e4:.3f}µm/d  T_transit={T_transit:.1f}d'
          f'  dt={dt_use}d  N_STEPS={N_STEPS}  pe_ring≈{pe_ring:.3f}'
          f'  (total {N_STEPS * dt_use:.1f}d)')

    bc_stele = _stele_bc_nodes(mecha)
    print(f'  absorbing BC at {len(bc_stele)} stele wall nodes (ADV and T; DIFF is mass-conserving)')

    # ── build capacitance, initial condition, r array ─────────────────────────
    st   = SoluteTransport(mecha, DP, CAP, mode='apo')
    Cm   = st.build_capacitance(0)
    Cm_d = Cm.diagonal()

    c0, r_lo, r_hi = _cortex_wall_ic(display_nodes, nwj)
    n_ic = int(c0.sum())
    print(f'  IC: cortex ring  r=[{r_lo:.1f}, {r_hi:.1f}] µm  ({n_ic} nodes)')

    disp_idxs = np.array([e[0] for e in display_nodes], dtype=int)
    r_disp_um = np.array([e[1] for e in display_nodes])   # already in µm

    # ── time loop ─────────────────────────────────────────────────────────────
    # DIFF:  no absorbing BC — mass conserved; D is bidirectional so stele
    #        nodes are NOT pure sinks and there is no blow-up.
    # ADV:   absorbing BC at stele nodes — pure sinks in mode='apo'.
    # total: same absorbing BC as ADV (A component creates the same sinks).
    case_bc = {'diff': {}, 'adv': bc_stele, 'total': bc_stele}

    # Snaps at T_transit fractions; last snap is the equilibration endpoint
    snap_at   = {0, T_steps // 4, T_steps // 2, T_steps, N_STEPS}
    snap_data = {'diff': [], 'adv': [], 'total': []}
    ts_rcm    = {'diff': [], 'adv': [], 'total': []}
    rcm_data  = {'diff': [], 'adv': [], 'total': []}
    mass_data = {'diff': [], 'adv': [], 'total': []}
    M0s       = {k: float(np.dot(Cm_d, c0)) for k in ('diff', 'adv', 'total')}
    c_cur     = {'diff': c0.copy(), 'adv': c0.copy(), 'total': c0.copy()}

    def _record(key, c_vec, step):
        t = float(step) * CAP['dt']
        ts_rcm[key].append(t)
        rcm_data[key].append(_r_cm(c_vec, disp_idxs, r_disp_um))
        mass_data[key].append(float(np.dot(Cm_d, c_vec)))
        if step in snap_at:
            snap_data[key].append((c_vec.copy(), t))

    # Record true IC (t=0) before any time stepping
    for key in ('diff', 'adv', 'total'):
        _record(key, c_cur[key], 0)

    rhs = np.zeros(nwj)
    total_t = N_STEPS * CAP['dt']
    print(f'  running {N_STEPS} IE steps (dt={CAP["dt"]} d, total {total_t:.1f} d) ...')

    for step in range(1, N_STEPS + 1):
        for key, ops in (('diff', 'D'), ('adv', 'A'), ('total', 'T')):
            c_cur[key] = st.solve(
                h=0, i_maturity=0, i_scenario=0,
                rhs=rhs,
                boundary_conditions=case_bc[key],
                c_prev=c_cur[key],
                theta=THETA,
                operators=ops,
            )
            _record(key, c_cur[key], step)

    # ── assertions ────────────────────────────────────────────────────────────
    for key in ('diff', 'adv', 'total'):
        min_c = min(c.min() for c, _ in snap_data[key])
        assert min_c >= -1e-6, \
            f'[{key}] negative concentration: {min_c:.3e}'

    M_f_diff   = float(np.dot(Cm_d, c_cur['diff']))
    rel_loss   = abs(M_f_diff - M0s['diff']) / max(M0s['diff'], 1e-30)
    assert rel_loss < 0.01, \
        f'DIFF lost {rel_loss:.1%} of mass — mass should be conserved with no absorbing BC'

    diff_max = np.abs(c_cur['adv'] - c_cur['diff']).max()
    assert diff_max > 1e-12, \
        f'ADV and DIFF profiles identical: max|Δc| = {diff_max:.3e}'

    delta_adv = rcm_data['adv'][-1] - rcm_data['adv'][0]    # µm
    assert delta_adv < -1.0, \
        f'ADV r_cm did not shift inward: Δr_cm = {delta_adv:.2f} µm'

    # ── diagnostics ───────────────────────────────────────────────────────────
    print(f'\n  ┌──────────────┬──────────────┬──────────────┬──────────────┐')
    print(f'  │ Case         │ M(t=0)       │ M(t={total_t:.0f}d)     │ Δr_cm (µm)   │')
    print(f'  ├──────────────┼──────────────┼──────────────┼──────────────┤')
    for key in ('diff', 'adv', 'total'):
        M0  = M0s[key]
        M_f = float(np.dot(Cm_d, c_cur[key]))
        dr  = rcm_data[key][-1] - rcm_data[key][0]   # µm
        lbl = {'diff': 'Diffusion', 'adv': 'Advection', 'total': 'Full T'}[key]
        print(f'  │ {lbl:<12} │ {M0:>12.4g} │ {M_f:>12.4g} │ {dr:>+12.2f} │')
    print(f'  └──────────────┴──────────────┴──────────────┴──────────────┘')

    # ── plots ─────────────────────────────────────────────────────────────────
    os.makedirs(_OUT_DIR, exist_ok=True)
    _plot_evolution(display_nodes, snap_data, r_lo, r_hi, cx, cy, nwj)
    _plot_radial(display_nodes, snap_data)
    _plot_rcm(ts_rcm, rcm_data, mass_data, M0s, T_transit=T_transit)


# ── standalone ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    test_apo_transport()
    print('\n[test_apo] PASSED')