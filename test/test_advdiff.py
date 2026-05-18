"""
test_advdiff.py

Combined apoplastic (mode='apo') and symplastic (mode='sym') solute transport
using SoluteTransport with Implicit Euler stepping.

Both pathways use the same homogeneous tissue (Casparian strip removed) so
that transport is driven by a smooth radial pressure gradient.  The hydraulic
fluxes are scaled independently for each mode to achieve global Pe = 1.

  APO  mode='apo'   wall-to-wall diffusion/advection      D_wall = 1e-5 cm²/d
  SYM  mode='sym'   plasmodesmata diffusion/advection      D_PD   = 1e-4 cm²/d

Three operators per mode:
  D  diffusion only
  A  advection only
  T  full advection + diffusion

Physical expectations
---------------------
APO DIFF  mass conserved; cortex ring spreads and drifts to stele BC
APO ADV   mass absorbed at stele; front sweeps inward; r_cm drops then rises
APO T     ADV + DIFF combined; fastest depletion
SYM DIFF  mass conserved (no absorbing BC); ring spreads slowly via PD
SYM ADV   mass absorbed at innermost cells; front sweeps inward
SYM T     combined; essentially full depletion by T_diff

Outputs
-------
  test/outputs/advdiff_apo_evolution.png
  test/outputs/advdiff_sym_evolution.png
  test/outputs/advdiff_radial.png
  test/outputs/advdiff_rcm.png
"""

import os
import sys
import numpy as np
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

D_WALL = 1e-5   # cm²/d  apoplastic wall diffusivity
D_PD   = 1e-4   # cm²/d  plasmodesmata diffusivity

DP_APO = dict(apo_wall=D_WALL, membrane=0.0, plasmodesmata=0.0)
DP_SYM = dict(apo_wall=0.0,   membrane=0.0, plasmodesmata=D_PD)

THETA = 1.0

CASES  = ['diff', 'adv', 'total']
COLORS = {'diff': '#2166ac', 'adv': '#d6604d', 'total': '#4dac26'}
LABELS = {'diff': 'Diffusion (D)', 'adv': 'Advection (A)', 'total': 'Full (T=A+D)'}
OPS    = {'diff': 'D', 'adv': 'A', 'total': 'T'}

_NONPARENCHYMA_CGROUPS = frozenset({13, 19, 20, 11, 12})


# ── shared build helpers ───────────────────────────────────────────────────────

def _build_mecha_homogeneous() -> Mecha:
    """Homogeneous apoplast: kw_endo_endo = kw_base, water_flux() re-solved."""
    data  = InData(cellset_file=_CELLSET)
    mecha = Mecha(data)
    h, i_mat = 0, 0
    kw_base   = float(mecha.hydraulic.get_kw_value(h))
    kw_config = mecha.hydraulic_conductivities[h, i_mat, 1]['kw']
    kw_config['kw_endo_endo'] = kw_base
    sol, mat_W = mecha.water_flux()
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


def _r_cm(c, disp_vals_fn, disp_r_um):
    vals = disp_vals_fn(c)
    tot  = float(vals.sum())
    if tot < 1e-30:
        return float('nan')
    return float(np.dot(disp_r_um, vals)) / tot


# ── APO helpers ───────────────────────────────────────────────────────────────

def _stele_bc_nodes(mecha: Mecha) -> dict:
    nwj = mecha.network.n_wall_junction
    bc  = {}
    for nd, d in mecha.network.graph.nodes(data=True):
        idx = mecha.indice[nd]
        if idx < nwj:
            if (d.get('count_stele_overall', 0) > 0
                    and d.get('count_cortex', 0) == 0
                    and d.get('count_endo', 0) == 0):
                bc[idx] = 0.0
    return bc


def _collect_display_nodes(mecha: Mecha, cx, cy):
    nwj = mecha.network.n_wall_junction
    wl  = mecha.network.wall_lengths
    result = []
    for nd, d in mecha.network.graph.nodes(data=True):
        idx = mecha.indice[nd]
        if idx < nwj and 'position' in d and wl.get(idx, 0.0) > 0:
            r  = np.hypot(d['position'][0] - cx, d['position'][1] - cy)
            cc = d.get('count_cortex', 0)
            result.append((idx, r, d['position'][0], d['position'][1], cc))
    return result


def _cortex_wall_ic(display_nodes, nwj):
    cortex = [(idx, r) for idx, r, *_, cc in display_nodes if cc >= 1]
    r_vals = [r for _, r in cortex]
    r_lo, r_hi = np.percentile(r_vals, [40, 60])
    c0 = np.zeros(nwj)
    for idx, r in cortex:
        if r_lo <= r <= r_hi:
            c0[idx] = 1.0
    return c0, r_lo, r_hi


# ── SYM helpers ───────────────────────────────────────────────────────────────

def _collect_display_cells(mecha: Mecha, cx, cy):
    nwj = mecha.network.n_wall_junction
    result = []
    for nd, d in mecha.network.graph.nodes(data=True):
        idx = mecha.indice[nd]
        if idx < nwj or 'position' not in d:
            continue
        cell_id = idx - nwj
        ca   = mecha.network.cell_areas
        area = float(ca[cell_id]) if cell_id < len(ca) else 0.0
        if area > 0:
            r  = np.hypot(d['position'][0] - cx, d['position'][1] - cy)
            cg = d.get('cgroup', -1)
            result.append((idx, cell_id, r, d['position'][0], d['position'][1], cg))
    return result


def _cortex_cell_ic(display_cells, n_cells):
    cands = [(cell_id, r)
             for _, cell_id, r, _, _, cg in display_cells
             if cg not in _NONPARENCHYMA_CGROUPS]
    if not cands:
        cands = [(cell_id, r) for _, cell_id, r, *_ in display_cells]
    r_vals = [r for _, r in cands]
    r_lo, r_hi = np.percentile(r_vals, [48, 52])
    c0 = np.zeros(n_cells)
    for cell_id, r in cands:
        if r_lo <= r <= r_hi:
            c0[cell_id] = 1.0
    return c0, r_lo, r_hi


def _inner_stele_bc(display_cells, r_thresh_um) -> dict:
    return {node_id: 0.0
            for node_id, _, r, *_ in display_cells
            if r < r_thresh_um}


# ── generic evolution plot ─────────────────────────────────────────────────────

def _plot_evolution(xs, ys, get_vals, snap_data, r_lo, r_hi, cx, cy,
                    dot_size, vmax, suptitle, out_name):
    """
    3-row × n_snap scatter evolution plot.

    xs, ys      : position arrays for display nodes (µm)
    get_vals(c) : extracts concentrations at those nodes from the c vector
    vmax        : colormap ceiling (use 1.0 to keep IC ring visible)
    """
    n_snap = len(snap_data[CASES[0]])
    fig, axes = plt.subplots(3, n_snap,
                             figsize=(3.0 * n_snap, 9.5), squeeze=False,
                             layout='constrained')
    theta_ring = np.linspace(0, 2 * np.pi, 300)

    for row_i, key in enumerate(CASES):
        snaps  = snap_data[key]
        sc_ref = None
        for col_j, (c_vec, t) in enumerate(snaps):
            ax  = axes[row_i, col_j]
            sc  = ax.scatter(xs, ys, s=dot_size,
                             c=get_vals(c_vec), cmap='plasma',
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
                ax.set_ylabel(f"{LABELS[key]}\ny (µm)", fontsize=7)
        if sc_ref is not None:
            fig.colorbar(sc_ref, ax=axes[row_i, :].tolist(),
                         shrink=0.6, label='c (a.u.)', pad=0.01)

    fig.suptitle(suptitle, fontsize=9)
    out = os.path.join(_OUT_DIR, out_name)
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  → {out}')


def _plot_radial(apo_nodes, apo_get_vals, snap_apo,
                 sym_cells, sym_get_vals, snap_sym):
    """
    2-row × n_snap radial scatter.  Top row: APO; bottom row: SYM.
    Both rows share the same number of panels (n_snap must be identical).
    """
    r_apo = np.array([e[1] for e in apo_nodes])
    r_sym = np.array([e[2] for e in sym_cells])
    n     = len(snap_apo[CASES[0]])
    fig, axes = plt.subplots(2, n, figsize=(4.0 * n, 7.0), squeeze=False)

    for col_j in range(n):
        t_apo = snap_apo[CASES[0]][col_j][1]
        t_sym = snap_sym[CASES[0]][col_j][1]
        for key in CASES:
            c_a = snap_apo[key][col_j][0]
            axes[0, col_j].scatter(r_apo, apo_get_vals(c_a),
                                   s=3, color=COLORS[key], alpha=0.5,
                                   label=LABELS[key])
            c_s = snap_sym[key][col_j][0]
            axes[1, col_j].scatter(r_sym, sym_get_vals(c_s),
                                   s=5, color=COLORS[key], alpha=0.5,
                                   label=LABELS[key])
        for row_i, (ax, t, ylabel) in enumerate(zip(
                [axes[0, col_j], axes[1, col_j]],
                [t_apo, t_sym],
                ['c (APO)', 'c (SYM)'])):
            ax.set_xlabel('r (µm)', fontsize=7)
            ax.set_ylabel(ylabel, fontsize=7)
            ax.set_title(f't = {t:.1f} d', fontsize=8)
            ax.set_ylim(bottom=-0.02)
            ax.tick_params(labelsize=6)
            if col_j == 0:
                ax.legend(fontsize=6)

    axes[0, 0].set_ylabel('c  APO (wall nodes)', fontsize=7)
    axes[1, 0].set_ylabel('c  SYM (cell nodes)', fontsize=7)
    fig.suptitle('Radial profiles — APO (top) and SYM (bottom)', fontsize=9)
    plt.tight_layout()
    out = os.path.join(_OUT_DIR, 'advdiff_radial.png')
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  → {out}')


def _plot_rcm(ts_apo, rcm_apo, mass_apo, M0_apo, T_tr_apo,
              ts_sym, rcm_sym, mass_sym, M0_sym, T_tr_sym):
    """
    2×2 panel figure.
    Col 0: APO, Col 1: SYM.
    Row 0: r_cm(t), Row 1: M(t)/M0.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex='col')

    for col_i, (ts, rcm, mass, M0s, T_tr, mode_label) in enumerate([
        (ts_apo, rcm_apo, mass_apo, M0_apo, T_tr_apo, 'APO (wall nodes)'),
        (ts_sym, rcm_sym, mass_sym, M0_sym, T_tr_sym, 'SYM (cell nodes)'),
    ]):
        ax_r = axes[0, col_i]
        ax_m = axes[1, col_i]
        for key in CASES:
            t_arr   = np.array(ts[key])
            rcm_arr = np.array(rcm[key])
            valid   = np.isfinite(rcm_arr)
            ax_r.plot(t_arr[valid], rcm_arr[valid],
                      color=COLORS[key], lw=1.5, label=LABELS[key])
            m_arr = np.array(mass[key]) / M0s[key]
            ax_m.plot(t_arr, m_arr,
                      color=COLORS[key], lw=1.5, label=LABELS[key])
        if T_tr is not None:
            for ax in (ax_r, ax_m):
                ax.axvline(T_tr, color='gray', lw=0.8, ls=':', alpha=0.7,
                           label=f'T_transit={T_tr:.1f}d')
        ax_r.set_ylabel('r_cm (µm)', fontsize=9)
        ax_r.set_title(mode_label, fontsize=9)
        ax_r.legend(fontsize=7)
        ax_m.set_xlabel('Time (d)', fontsize=9)
        ax_m.set_ylabel('M(t) / M₀', fontsize=9)
        ax_m.set_ylim(-0.02, 1.05)
        ax_m.legend(fontsize=7)

    fig.suptitle('Concentration-weighted r_cm and mass fraction\n'
                 'Homogeneous tissue | absorbing BC at stele (ADV, T)',
                 fontsize=9)
    plt.tight_layout()
    out = os.path.join(_OUT_DIR, 'advdiff_rcm.png')
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  → {out}')


# ── test ──────────────────────────────────────────────────────────────────────

def test_advdiff():
    """
    Combined APO and SYM transport: D, A, T operators on homogeneous tissue.

    Assertions
    ----------
    APO DIFF  |ΔM|/M < 0.01   mass conserved (no absorbing BC)
    SYM DIFF  |ΔM|/M < 0.01   mass conserved (no absorbing BC)
    no negatives               min(c) >= -1e-6 for every case
    APO ADV   Δr_cm < -1 µm   front sweeps inward
    SYM ADV   Δr_cm < -1 µm   front sweeps inward
    """
    print('\n[test_advdiff] building Mecha (homogeneous tissue) ...')
    mecha   = _build_mecha_homogeneous()
    nwj     = mecha.network.n_wall_junction
    n_cells = mecha.network.n_cells
    cx, cy  = _root_center(mecha)

    # Save raw fluxes; each mode will rescale them independently
    raw_fluxes = [f['flux'] for f in mecha.edge_flux_list[0][0]]

    display_nodes = _collect_display_nodes(mecha, cx, cy)
    display_cells = _collect_display_cells(mecha, cx, cy)
    print(f'  wall nodes (L>0): {len(display_nodes)}  |  '
          f'cell nodes (area>0): {len(display_cells)}')

    # ─────────────────────────────────────────────────────────────────────────
    # APO RUN
    # ─────────────────────────────────────────────────────────────────────────
    print('\n[APO] scaling fluxes for global Pe = 1 ...')

    # Restore raw fluxes before APO scaling
    for f, v in zip(mecha.edge_flux_list[0][0], raw_fluxes):
        f['flux'] = v

    cap_apo = {'dt': 0.1, 'C_wall': 1.0, 'C_cell': 1.0}

    st_pe  = SoluteTransport(mecha, DP_APO, cap_apo, mode='apo')
    A_tmp  = st_pe.build_advection_matrix(0, 0)
    D_tmp  = st_pe.build_diffusion_matrix(0, 0)

    c0_tmp, r_lo_tmp, _ = _cortex_wall_ic(display_nodes, nwj)
    ring_w  = np.where(c0_tmp > 0)[0]

    height_cm = float(mecha.geometry.maturity_stages[0].get('height')) * 1e-4
    thick_cm  = mecha.geometry.thickness * 1e-4
    r_lo_cm   = r_lo_tmp * 1e-4

    Ad_ring  = np.abs(A_tmp.diagonal())[ring_w]
    valid_v  = Ad_ring > 0
    v_raw    = float(Ad_ring[valid_v].mean()) / (height_cm * thick_cm)
    v_tgt    = D_WALL / r_lo_cm            # Pe=1 target velocity
    apo_scale = v_tgt / v_raw if v_raw > 0 else 1.0
    for f in mecha.edge_flux_list[0][0]:
        f['flux'] *= apo_scale

    T_tr_apo  = r_lo_cm / v_tgt           # d
    dt_apo    = max(0.1, round(T_tr_apo / 100.0, 1))
    cap_apo['dt'] = dt_apo
    T_steps_apo   = int(np.ceil(T_tr_apo / dt_apo))
    N_apo         = max(200, min(2000, 5 * T_steps_apo))

    Dd_ring    = np.abs(D_tmp.diagonal())[ring_w]
    valid_both = valid_v & (Dd_ring > 0)
    pe_ring_apo = float((Ad_ring[valid_both] * apo_scale / Dd_ring[valid_both]).mean()) \
                  if valid_both.any() else float('nan')
    print(f'  v={v_tgt*1e4:.3f}µm/d  T_transit={T_tr_apo:.1f}d  '
          f'dt={dt_apo}d  N={N_apo}  pe_ring≈{pe_ring_apo:.3f}')

    bc_apo_stele = _stele_bc_nodes(mecha)
    print(f'  absorbing BC at {len(bc_apo_stele)} stele wall nodes')

    st_apo   = SoluteTransport(mecha, DP_APO, cap_apo, mode='apo')
    Cm_apo   = st_apo.build_capacitance(0)
    Cm_d_apo = Cm_apo.diagonal()

    c0_apo, r_lo_apo, r_hi_apo = _cortex_wall_ic(display_nodes, nwj)
    print(f'  IC: cortex ring r=[{r_lo_apo:.1f}, {r_hi_apo:.1f}] µm  '
          f'({int(c0_apo.sum())} wall nodes)')

    disp_idxs = np.array([e[0] for e in display_nodes], dtype=int)
    r_disp_apo = np.array([e[1] for e in display_nodes])
    apo_get_vals = lambda c: c[disp_idxs]

    apo_bc   = {'diff': {}, 'adv': bc_apo_stele, 'total': bc_apo_stele}
    snap_at_apo = {0, T_steps_apo // 4, T_steps_apo // 2, T_steps_apo, N_apo}
    snap_apo  = {k: [] for k in CASES}
    ts_apo    = {k: [] for k in CASES}
    rcm_apo   = {k: [] for k in CASES}
    mass_apo  = {k: [] for k in CASES}
    M0_apo    = {k: float(np.dot(Cm_d_apo, c0_apo)) for k in CASES}
    c_apo     = {k: c0_apo.copy() for k in CASES}

    def _rec_apo(key, c_vec, step):
        t = float(step) * cap_apo['dt']
        ts_apo[key].append(t)
        rcm_apo[key].append(_r_cm(c_vec, apo_get_vals, r_disp_apo))
        mass_apo[key].append(float(np.dot(Cm_d_apo, c_vec)))
        if step in snap_at_apo:
            snap_apo[key].append((c_vec.copy(), t))

    for key in CASES:
        _rec_apo(key, c_apo[key], 0)

    rhs_apo = np.zeros(nwj)
    print(f'  running {N_apo} APO steps ...')
    for step in range(1, N_apo + 1):
        for key in CASES:
            c_apo[key] = st_apo.solve(
                h=0, i_maturity=0, i_scenario=0,
                rhs=rhs_apo,
                boundary_conditions=apo_bc[key],
                c_prev=c_apo[key],
                theta=THETA,
                operators=OPS[key],
            )
            _rec_apo(key, c_apo[key], step)

    # ─────────────────────────────────────────────────────────────────────────
    # SYM RUN
    # ─────────────────────────────────────────────────────────────────────────
    print('\n[SYM] scaling fluxes for global Pe = 1 ...')

    # Restore raw fluxes before SYM scaling
    for f, v in zip(mecha.edge_flux_list[0][0], raw_fluxes):
        f['flux'] = v

    cap_sym = {'dt': 0.1, 'C_wall': 1.0, 'C_cell': 1.0}

    st_pe_s = SoluteTransport(mecha, DP_SYM, cap_sym, mode='sym')
    A_tmp_s = st_pe_s.build_advection_matrix(0, 0)
    D_tmp_s = st_pe_s.build_diffusion_matrix(0, 0)

    c0_tmp_s, r_lo_tmp_s, _ = _cortex_cell_ic(display_cells, n_cells)
    ring_c   = np.where(c0_tmp_s > 0)[0]
    r_lo_cm_s = r_lo_tmp_s * 1e-4

    Ad_diag_s = np.abs(A_tmp_s.diagonal())
    has_adv_s = Ad_diag_s.max() > 0 and len(ring_c) > 0 and Ad_diag_s[ring_c].max() > 0

    Dd_ring_s = np.abs(D_tmp_s.diagonal())[ring_c]
    vols_all  = st_pe_s._compute_node_volumes(0)
    V_ring    = vols_all[nwj + ring_c]
    mask_tau  = (Dd_ring_s > 0) & (V_ring > 0)
    tau_cell  = float(np.mean(V_ring[mask_tau] / Dd_ring_s[mask_tau]))

    ca        = mecha.network.cell_areas
    areas_ring = np.array([float(ca[cid]) for cid in ring_c if cid < len(ca)])
    L_cell_cm = float(np.sqrt(np.mean(areas_ring[areas_ring > 0]))) * 1e-4
    N_hops    = r_lo_cm_s / L_cell_cm

    dt_raw    = tau_cell / 10.0
    mag       = 10.0 ** np.floor(np.log10(max(dt_raw, 1e-10)))
    dt_sym    = max(1e-4, round(dt_raw / mag) * mag)

    T_diff_steps = max(10, int(np.ceil(N_hops ** 2 * tau_cell / dt_sym)))

    if has_adv_s:
        Ad_ring_s  = Ad_diag_s[ring_c]
        valid_vs   = Ad_ring_s > 0
        v_raw_s    = float(Ad_ring_s[valid_vs].mean()) / (height_cm * thick_cm)
        v_tgt_s    = D_PD / r_lo_cm_s
        sym_scale  = v_tgt_s / v_raw_s if v_raw_s > 0 else 1.0
        for f in mecha.edge_flux_list[0][0]:
            f['flux'] *= sym_scale
        valid_bs   = valid_vs & (Dd_ring_s > 0)
        pe_ring_sym = float((Ad_ring_s[valid_bs] * sym_scale / Dd_ring_s[valid_bs]).mean()) \
                      if valid_bs.any() else 1.0
        T_adv_steps = max(5, int(np.ceil(N_hops * tau_cell / (pe_ring_sym * dt_sym))))
        T_tr_sym    = T_adv_steps * dt_sym
        print(f'  v={v_tgt_s*1e4:.3f}µm/d  pe_ring={pe_ring_sym:.3f}')
    else:
        T_adv_steps = T_diff_steps // max(1, int(N_hops))
        T_tr_sym    = T_adv_steps * dt_sym
        pe_ring_sym = float('nan')

    N_sym = min(2000, int(T_diff_steps))
    cap_sym['dt'] = dt_sym
    print(f'  τ_cell={tau_cell:.1f}d  L_cell={L_cell_cm*1e4:.1f}µm  N_hops={N_hops:.1f}  '
          f'dt={dt_sym}d  T_adv≈{T_tr_sym:.0f}d ({T_adv_steps}st)  run={N_sym}st')

    r_all_sym = np.array([r for _, _, r, *_ in display_cells])
    r_inner   = float(np.percentile(r_all_sym, 20))
    bc_sym_stele = _inner_stele_bc(display_cells, r_inner) if has_adv_s else {}
    print(f'  absorbing BC at {len(bc_sym_stele)} innermost cells (r < {r_inner:.1f} µm)')

    st_sym   = SoluteTransport(mecha, DP_SYM, cap_sym, mode='sym')
    Cm_sym   = st_sym.build_capacitance(0)
    Cm_d_sym = Cm_sym.diagonal()

    c0_sym, r_lo_sym, r_hi_sym = _cortex_cell_ic(display_cells, n_cells)
    print(f'  IC: cortex cell ring r=[{r_lo_sym:.1f}, {r_hi_sym:.1f}] µm  '
          f'({int((c0_sym > 0).sum())} cells)')

    disp_cids  = np.array([e[1] for e in display_cells], dtype=int)
    r_disp_sym = np.array([e[2] for e in display_cells])
    sym_get_vals = lambda c: c[disp_cids]

    sym_bc = {'diff': {}, 'adv': bc_sym_stele, 'total': bc_sym_stele}
    snap_at_sym = {0, T_adv_steps // 4, T_adv_steps // 2, T_adv_steps, N_sym}
    snap_sym  = {k: [] for k in CASES}
    ts_sym    = {k: [] for k in CASES}
    rcm_sym   = {k: [] for k in CASES}
    mass_sym  = {k: [] for k in CASES}
    M0_sym    = {k: float(np.dot(Cm_d_sym, c0_sym)) for k in CASES}
    c_sym_cur = {k: c0_sym.copy() for k in CASES}

    def _rec_sym(key, c_vec, step):
        t = float(step) * cap_sym['dt']
        ts_sym[key].append(t)
        rcm_sym[key].append(_r_cm(c_vec, sym_get_vals, r_disp_sym))
        mass_sym[key].append(float(np.dot(Cm_d_sym, c_vec)))
        if step in snap_at_sym:
            snap_sym[key].append((c_vec.copy(), t))

    for key in CASES:
        _rec_sym(key, c_sym_cur[key], 0)

    rhs_sym = np.zeros(n_cells)
    print(f'  running {N_sym} SYM steps ...')
    for step in range(1, N_sym + 1):
        for key in CASES:
            c_sym_cur[key] = st_sym.solve(
                h=0, i_maturity=0, i_scenario=0,
                rhs=rhs_sym,
                boundary_conditions=sym_bc[key],
                c_prev=c_sym_cur[key],
                theta=THETA,
                operators=OPS[key],
            )
            _rec_sym(key, c_sym_cur[key], step)

    # ── assertions ────────────────────────────────────────────────────────────

    # APO
    for key in CASES:
        min_c = min(c.min() for c, _ in snap_apo[key])
        assert min_c >= -1e-6, f'[APO {key}] negative concentration: {min_c:.3e}'
    M_f_apo = float(np.dot(Cm_d_apo, c_apo['diff']))
    rel_apo  = abs(M_f_apo - M0_apo['diff']) / max(M0_apo['diff'], 1e-30)
    assert rel_apo < 0.01, f'APO DIFF mass error: {rel_apo:.1%}'
    d_adv_apo = rcm_apo['adv'][-1] - rcm_apo['adv'][0]
    assert d_adv_apo < -1.0, f'APO ADV r_cm did not shift inward: {d_adv_apo:.2f} µm'

    # SYM
    for key in CASES:
        min_c = min(c.min() for c, _ in snap_sym[key])
        assert min_c >= -1e-6, f'[SYM {key}] negative concentration: {min_c:.3e}'
    M_f_sym  = float(np.dot(Cm_d_sym, c_sym_cur['diff']))
    rel_sym  = abs(M_f_sym - M0_sym['diff']) / max(M0_sym['diff'], 1e-30)
    assert rel_sym < 0.01, f'SYM DIFF mass error: {rel_sym:.1%}'
    if has_adv_s:
        d_adv_sym = rcm_sym['adv'][-1] - rcm_sym['adv'][0]
        assert d_adv_sym < -1.0, f'SYM ADV r_cm did not shift inward: {d_adv_sym:.2f} µm'

    # ── diagnostics ───────────────────────────────────────────────────────────
    total_apo = N_apo * cap_apo['dt']
    total_sym = N_sym * cap_sym['dt']
    print(f'\n  ┌──────────┬──────────────┬──────────────┬──────────────┐')
    print(f'  │ APO case │ M(t=0)       │ M(t={total_apo:.0f}d)    │ Δr_cm (µm)   │')
    print(f'  ├──────────┼──────────────┼──────────────┼──────────────┤')
    for key in CASES:
        M_f = float(np.dot(Cm_d_apo, c_apo[key]))
        dr  = rcm_apo[key][-1] - rcm_apo[key][0]
        lbl = {'diff': 'Diff', 'adv': 'Adv', 'total': 'T'}[key]
        print(f'  │ {lbl:<8} │ {M0_apo[key]:>12.4g} │ {M_f:>12.4g} │ {dr:>+12.2f} │')
    print(f'  └──────────┴──────────────┴──────────────┴──────────────┘')

    print(f'\n  ┌──────────┬──────────────┬──────────────┬──────────────┐')
    print(f'  │ SYM case │ M(t=0)       │ M(t={total_sym:.0f}d)   │ Δr_cm (µm)   │')
    print(f'  ├──────────┼──────────────┼──────────────┼──────────────┤')
    for key in CASES:
        M_f = float(np.dot(Cm_d_sym, c_sym_cur[key]))
        dr  = rcm_sym[key][-1] - rcm_sym[key][0]
        lbl = {'diff': 'Diff', 'adv': 'Adv', 'total': 'T'}[key]
        print(f'  │ {lbl:<8} │ {M0_sym[key]:>12.4g} │ {M_f:>12.4g} │ {dr:>+12.2f} │')
    print(f'  └──────────┴──────────────┴──────────────┴──────────────┘')

    # ── plots ─────────────────────────────────────────────────────────────────
    os.makedirs(_OUT_DIR, exist_ok=True)

    xs_apo = np.array([e[2] for e in display_nodes])
    ys_apo = np.array([e[3] for e in display_nodes])
    _plot_evolution(
        xs_apo, ys_apo, apo_get_vals, snap_apo,
        r_lo_apo, r_hi_apo, cx, cy,
        dot_size=4, vmax=1.0,
        suptitle=(f'APO transport — homogeneous tissue (IE θ=1, dt={cap_apo["dt"]} d)\n'
                  'Wall nodes (L>0) | cyan: IC ring | absorbing BC at stele (ADV, T)'),
        out_name='advdiff_apo_evolution.png',
    )

    xs_sym = np.array([e[3] for e in display_cells])
    ys_sym = np.array([e[4] for e in display_cells])
    _plot_evolution(
        xs_sym, ys_sym, sym_get_vals, snap_sym,
        r_lo_sym, r_hi_sym, cx, cy,
        dot_size=10, vmax=1.0,
        suptitle=(f'SYM transport — homogeneous tissue (IE θ=1, dt={cap_sym["dt"]} d)\n'
                  'Cell nodes (area>0) | cyan: IC ring | absorbing BC at stele (ADV, T)'),
        out_name='advdiff_sym_evolution.png',
    )

    _plot_radial(display_nodes, apo_get_vals, snap_apo,
                 display_cells, sym_get_vals, snap_sym)

    _plot_rcm(ts_apo, rcm_apo, mass_apo, M0_apo, T_tr_apo,
              ts_sym, rcm_sym, mass_sym, M0_sym, T_tr_sym)


# ── standalone ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    test_advdiff()
    print('\n[test_advdiff] PASSED')
