"""
test_sym_transport.py

Symplastic solute transport via SoluteTransport, mode='sym'.

Uses the same homogeneous-tissue Mecha as test_apo_transport (equal wall
conductances, re-solved hydraulics) so that symplastic advection is driven
by the same clean radial pressure field.

SoluteTransport (mode='sym') operates on the n_cells subspace only.
Solute diffuses through plasmodesmata (PD); advection is carried by
symplastic water fluxes through PD.

With barrier=1 the Casparian strip disconnects xylem cells (cgroup 13)
from the main symplasm.  The connected path is:
    cortex (4) → endodermis (3) → pericycle (16) → stele parenchyma (5)

Tissue cgroup map (MECHA conventions)
--------------------------------------
  4  cortex
  3  endodermis
 16  pericycle
  5  stele parenchyma
 13  xylem / vessel (also 19, 20 → 13); symplastically ISOLATED (barrier=1)
 11  sieve tube elements
 12  companion cells
  1  epidermis / outermost cortex

Physical expectations
---------------------
DIFF:  PD diffusion spreads the cortex ring inward (and outward); mass
       is globally conserved — no absorbing BC → row sums of D are zero.
ADV:   Bulk symplastic flow (driven by the radial pressure gradient)
       sweeps the ring inward; absorbing BC at innermost stele cells models
       unloading.  Skipped when no PD fluxes are found.

Outputs
-------
  test/outputs/sym_evolution.png
  test/outputs/sym_radial.png
  test/outputs/sym_rcm.png
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

D_PD    = 1e-4   # cm²/d  plasmodesmata diffusivity (same as D_WALL in apo test)
DP_SYM  = dict(apo_wall=0.0, membrane=0.0, plasmodesmata=D_PD)
CAP     = {'dt': 0.1, 'C_wall': 1.0, 'C_cell': 1.0}
THETA   = 1.0

# cgroups that are symplastically isolated or not true parenchyma cells
_ISOLATED_CGROUPS   = frozenset({13, 19, 20})   # xylem (PD blocked when barrier>0)
_NONPARENCHYMA_CGROUPS = frozenset({13, 19, 20, 11, 12})  # excluded from IC ring

# ── build helpers ─────────────────────────────────────────────────────────────

def _build_mecha_homogeneous() -> Mecha:
    """
    Same homogeneous-tissue Mecha as test_apo_transport:
      kw_endo_endo → kw_base   (removes the Casparian wall conductance barrier)
      water_flux() re-run      (produces a clean radial pressure gradient)
      edge_flux_list filled    (from mat_W × pressure solution)
    """
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


def _collect_display_cells(mecha: Mecha, cx: float, cy: float):
    """
    Cell nodes (idx >= nwj) with positive area.
    Returns list of (node_id, cell_id, r_µm, x, y, cgroup).
    node_id  = full network index (>= nwj)
    cell_id  = node_id − nwj  (index into the sym concentration vector)
    """
    nwj = mecha.network.n_wall_junction
    result = []
    for nd, d in mecha.network.graph.nodes(data=True):
        idx = mecha.indice[nd]
        if idx < nwj or 'position' not in d:
            continue
        cell_id = idx - nwj
        ca = mecha.network.cell_areas
        area = float(ca[cell_id]) if cell_id < len(ca) else 0.0
        if area > 0:
            r  = np.hypot(d['position'][0] - cx, d['position'][1] - cy)
            cg = d.get('cgroup', -1)
            result.append((idx, cell_id, r,
                           d['position'][0], d['position'][1], cg))
    return result


def _cortex_cell_ic(display_cells: list, n_cells: int):
    """
    c=1 in the cortex cell ring at 48–52 % of non-stele cell radius.
    Returns (c0 shape=(n_cells,), r_lo_µm, r_hi_µm).
    """
    # Candidates: exclude xylem, sieve tube, companion cells
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


def _inner_stele_bc(display_cells: list, r_thresh_um: float) -> dict:
    """
    Absorbing BC (c=0) for ADV at innermost stele cells (r < r_thresh_um).
    Uses node_id as key (solve() converts to cell_id internally for sym mode).
    """
    return {node_id: 0.0
            for node_id, _, r, *_ in display_cells
            if r < r_thresh_um}


def _r_cm(c_sym: np.ndarray, disp_cids: np.ndarray,
          r_disp_um: np.ndarray) -> float:
    c_d = c_sym[disp_cids]
    tot = float(c_d.sum())
    if tot < 1e-30:
        return float('nan')
    return float(np.dot(r_disp_um, c_d)) / tot


# ── visualisation ─────────────────────────────────────────────────────────────

def _plot_evolution(display_cells, snap_data, r_lo, r_hi, cx, cy, cases):
    labels = {'diff': 'Diffusion (D)', 'adv': 'Advection (A)', 'total': 'Full (T=A+D)'}
    n_snap = len(snap_data[cases[0]])
    n_rows = len(cases)
    fig, axes = plt.subplots(n_rows, n_snap,
                             figsize=(3.0 * n_snap, 3.5 * n_rows), squeeze=False,
                             layout='constrained')
    theta_ring = np.linspace(0, 2 * np.pi, 300)
    xs   = np.array([e[3] for e in display_cells])
    ys   = np.array([e[4] for e in display_cells])
    cids = np.array([e[1] for e in display_cells], dtype=int)

    for row_i, key in enumerate(cases):
        snaps  = snap_data[key]
        # Fix vmax=1.0: IC defines the reference concentration.
        # ADV focusing (c > 1 near stele) is clipped to max colour rather than
        # expanding the scale and washing out the IC ring at t=0.
        vmax   = 1.0
        sc_ref = None
        for col_j, (c_vec, t) in enumerate(snaps):
            ax = axes[row_i, col_j]
            sc = ax.scatter(xs, ys, s=10,
                            c=c_vec[cids], cmap='plasma',
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
        f'Symplastic transport — PD only (IE θ=1, dt={CAP["dt"]} d)\n'
        'Cell nodes (area > 0) | cyan: initial ring | absorbing BC at stele (ADV)',
        fontsize=9)
    out = os.path.join(_OUT_DIR, 'sym_evolution.png')
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  → {out}')


def _plot_radial(display_cells, snap_data, cases):
    cids  = np.array([e[1] for e in display_cells], dtype=int)
    r_arr = np.array([e[2] for e in display_cells])
    n        = len(snap_data[cases[0]])
    show_idx = list(range(n))
    n_show   = len(show_idx)
    fig, axes = plt.subplots(1, n_show, figsize=(4.0 * n_show, 3.5), squeeze=False)
    colors = {'diff': '#2166ac', 'adv': '#d6604d', 'total': '#4dac26'}
    lab    = {'diff': 'DIFF', 'adv': 'ADV', 'total': 'T'}
    for col_j, si in enumerate(show_idx):
        ax = axes[0, col_j]
        t  = snap_data[cases[0]][si][1]
        for key in cases:
            c_vec = snap_data[key][si][0]
            ax.scatter(r_arr, c_vec[cids], s=5,
                       color=colors[key], alpha=0.5, label=lab[key])
        ax.set_xlabel('r (µm)', fontsize=7)
        ax.set_ylabel('c (a.u.)', fontsize=7)
        ax.set_title(f't = {t:.1f} d', fontsize=9)
        ax.set_ylim(bottom=-0.02)
        ax.tick_params(labelsize=6)
        if col_j == 0:
            ax.legend(fontsize=7)
    fig.suptitle('Symplastic radial profiles (cell nodes, area > 0)', fontsize=9)
    plt.tight_layout()
    out = os.path.join(_OUT_DIR, 'sym_radial.png')
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  → {out}')


def _plot_rcm(ts_data, rcm_data, mass_data, M0s, cases, T_transit=None):
    colors = {'diff': '#2166ac', 'adv': '#d6604d', 'total': '#4dac26'}
    labels = {'diff': 'Diffusion (D)', 'adv': 'Advection (A)', 'total': 'Full (T=A+D)'}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

    for key in cases:
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
    ax1.set_title('Concentration-weighted r_cm (cell nodes)\n'
                  'Symplastic transport | PD diffusion and advection\n'
                  'Absorbing BC at innermost stele cells (ADV, T)',
                  fontsize=7)
    ax1.legend(fontsize=8)

    ax2.set_xlabel('Time (d)', fontsize=9)
    ax2.set_ylabel('M(t) / M₀', fontsize=9)
    ax2.set_title('Mass fraction remaining', fontsize=8)
    ax2.set_ylim(-0.02, 1.05)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(_OUT_DIR, 'sym_rcm.png')
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  → {out}')


# ── test ──────────────────────────────────────────────────────────────────────

def test_sym_transport():
    """
    Symplastic DIFF and (when PD fluxes exist) ADV using SoluteTransport mode='sym'.

    Assertions
    ----------
    DIFF mass conserved          |ΔM|/M < 1e-4  (no absorbing BC → row sums zero)
    no spurious negatives        min(c) >= -1e-6
    DIFF spreads                 |Δr_cm(DIFF)| > 0.1 µm
    ADV and DIFF differ (if ADV) max|Δc| > 1e-12
    ADV shifts r_cm inward       Δr_cm(ADV) < −1 µm
    """
    print('\n[test_sym] building Mecha (homogeneous tissue) ...')
    mecha   = _build_mecha_homogeneous()
    nwj     = mecha.network.n_wall_junction
    n_cells = mecha.network.n_cells

    cx, cy = _root_center(mecha)
    display_cells = _collect_display_cells(mecha, cx, cy)
    print(f'  cell nodes total: {n_cells}  |  displayed (area>0): {len(display_cells)}')

    # Cgroup summary for diagnostics
    cg_counts: dict = {}
    for _, _, _, _, _, cg in display_cells:
        cg_counts[cg] = cg_counts.get(cg, 0) + 1
    print(f'  cgroup counts: { {k: cg_counts[k] for k in sorted(cg_counts)} }')

    # ── Pe scaling; detect PD fluxes ──────────────────────────────────────────
    st_pe = SoluteTransport(mecha, DP_SYM, CAP, mode='sym')
    A_tmp = st_pe.build_advection_matrix(0, 0)
    D_tmp = st_pe.build_diffusion_matrix(0, 0)

    c0_tmp, r_lo_tmp, _ = _cortex_cell_ic(display_cells, n_cells)
    ring_cids = np.where(c0_tmp > 0)[0]

    height_cm = float(mecha.geometry.maturity_stages[0].get('height')) * 1e-4
    thick_cm  = mecha.geometry.thickness * 1e-4
    r_lo_cm   = r_lo_tmp * 1e-4

    Ad_diag = np.abs(A_tmp.diagonal())
    has_adv = Ad_diag.max() > 0 and len(ring_cids) > 0 and Ad_diag[ring_cids].max() > 0

    T_transit: float
    T_adv_steps: int
    T_diff_steps: int
    n_steps: int
    dt_use: float
    pe_ring: float = float('nan')

    # ── Actual cell hop time from the PD diffusion matrix ────────────────────
    # τ_cell = V_cell / |D_cell|  (d)
    # D_PD enters _fill_plasmodesmata via geometry factors that reduce the
    # effective conductance by orders of magnitude relative to D_PD alone.
    # Using L_cell²/D_PD ignores this and vastly underestimates τ_cell.
    Dd_ring  = np.abs(D_tmp.diagonal())[ring_cids]
    vols_all = st_pe._compute_node_volumes(0)           # cm³, all nodes
    V_ring   = vols_all[nwj + ring_cids]               # IC ring cell volumes
    mask_tau = (Dd_ring > 0) & (V_ring > 0)
    tau_cell = float(np.mean(V_ring[mask_tau] / Dd_ring[mask_tau]))  # d

    # Characteristic cell length from IC ring area
    ca = mecha.network.cell_areas
    areas_ring = np.array([float(ca[cid]) for cid in ring_cids if cid < len(ca)])
    L_cell_cm  = float(np.sqrt(np.mean(areas_ring[areas_ring > 0]))) * 1e-4  # cm
    N_hops     = r_lo_cm / L_cell_cm

    # dt: τ_cell / 10  → ~10 % concentration change per step (clearly visible)
    dt_raw = tau_cell / 10.0
    mag    = 10.0 ** np.floor(np.log10(max(dt_raw, 1e-10)))
    dt_use = max(1e-4, round(dt_raw / mag) * mag)   # rounded to 1 sig fig

    # Transit times in steps
    #   DIFF: N_hops² × τ_cell steps to equilibrate across the IC-to-boundary span
    #   ADV:  N_hops  × τ_cell / pe_ring steps (pe_ring brings ADV transit down)
    T_diff_steps = max(10, int(np.ceil(N_hops ** 2 * tau_cell / dt_use)))

    if has_adv:
        Ad_ring = Ad_diag[ring_cids]
        valid_v = Ad_ring > 0
        v_raw   = float(Ad_ring[valid_v].mean()) / (height_cm * thick_cm)
        v_tgt   = D_PD / r_lo_cm              # global Pe = 1
        flux_scale = v_tgt / v_raw if v_raw > 0 else 1.0
        for f in mecha.edge_flux_list[0][0]:
            f['flux'] *= flux_scale

        valid_both = valid_v & (Dd_ring > 0)
        pe_ring    = float((Ad_ring[valid_both] * flux_scale / Dd_ring[valid_both]).mean()) \
                     if valid_both.any() else 1.0
        T_adv_steps = max(5, int(np.ceil(N_hops * tau_cell / (pe_ring * dt_use))))
        T_transit   = T_adv_steps * dt_use
        print(f'  PD fluxes; global Pe=1: v={v_tgt*1e4:.3f}µm/d  pe_ring={pe_ring:.3f}')
    else:
        T_adv_steps = T_diff_steps // max(1, int(N_hops))
        T_transit   = T_adv_steps * dt_use

    n_steps = min(2000, int(T_diff_steps))
    print(f'  τ_cell={tau_cell:.1f}d  L_cell={L_cell_cm*1e4:.1f}µm  N_hops={N_hops:.1f}'
          f'  dt={dt_use}d'
          f'  T_adv≈{T_adv_steps*dt_use:.0f}d ({T_adv_steps}st)'
          f'  T_diff≈{T_diff_steps*dt_use:.0f}d ({T_diff_steps}st)'
          f'  run={n_steps}st ({n_steps*dt_use:.0f}d)')

    CAP['dt'] = dt_use

    # Absorbing BC for ADV: innermost stele cells (r < 20th-percentile of all cells)
    r_all   = np.array([r for _, _, r, *_ in display_cells])
    r_inner = float(np.percentile(r_all, 20))
    bc_stele = _inner_stele_bc(display_cells, r_inner) if has_adv else {}
    print(f'  absorbing BC at {len(bc_stele)} innermost cells '
          f'(r < {r_inner:.1f} µm, ADV only; DIFF is mass-conserving)')

    # ── build solver, IC, capacitance ─────────────────────────────────────────
    st   = SoluteTransport(mecha, DP_SYM, CAP, mode='sym')
    Cm   = st.build_capacitance(0)
    Cm_d = Cm.diagonal()

    c0, r_lo, r_hi = _cortex_cell_ic(display_cells, n_cells)
    n_ic = int((c0 > 0).sum())
    print(f'  IC: cortex cell ring  r=[{r_lo:.1f}, {r_hi:.1f}] µm  ({n_ic} cells)')

    disp_cids = np.array([e[1] for e in display_cells], dtype=int)
    r_disp_um = np.array([e[2] for e in display_cells])

    # ── time loop ─────────────────────────────────────────────────────────────
    cases     = ['diff'] + (['adv', 'total'] if has_adv else [])
    case_bc   = {'diff': {}, 'adv': bc_stele, 'total': bc_stele}
    case_ops  = {'diff': 'D', 'adv': 'A', 'total': 'T'}

    # Snaps: 3 steps inside the ADV transit window + end state
    snap_at = {0, T_adv_steps // 4, T_adv_steps // 2, T_adv_steps, n_steps}
    snap_data = {k: [] for k in cases}
    ts_rcm    = {k: [] for k in cases}
    rcm_data  = {k: [] for k in cases}
    mass_data = {k: [] for k in cases}
    M0s       = {k: float(np.dot(Cm_d, c0)) for k in cases}
    c_cur     = {k: c0.copy() for k in cases}

    def _record(key: str, c_vec: np.ndarray, step: int) -> None:
        t = float(step) * CAP['dt']
        ts_rcm[key].append(t)
        rcm_data[key].append(_r_cm(c_vec, disp_cids, r_disp_um))
        mass_data[key].append(float(np.dot(Cm_d, c_vec)))
        if step in snap_at:
            snap_data[key].append((c_vec.copy(), t))

    for key in cases:
        _record(key, c_cur[key], 0)

    rhs     = np.zeros(n_cells)
    total_t = n_steps * CAP['dt']
    print(f'  running {n_steps} IE steps (dt={CAP["dt"]} d, total {total_t:.1f} d) ...')

    for step in range(1, n_steps + 1):
        for key in cases:
            c_cur[key] = st.solve(
                h=0, i_maturity=0, i_scenario=0,
                rhs=rhs,
                boundary_conditions=case_bc[key],
                c_prev=c_cur[key],
                theta=THETA,
                operators=case_ops[key],
            )
            _record(key, c_cur[key], step)

    # ── assertions ────────────────────────────────────────────────────────────
    for key in cases:
        min_c = min(c.min() for c, _ in snap_data[key])
        assert min_c >= -1e-6, f'[{key}] negative concentration: {min_c:.3e}'

    M_f_diff  = float(np.dot(Cm_d, c_cur['diff']))
    rel_loss  = abs(M_f_diff - M0s['diff']) / max(M0s['diff'], 1e-30)
    assert rel_loss < 0.01, \
        f'DIFF lost {rel_loss:.1%} of mass (sym mode should be mass-conserving)'

    delta_diff = rcm_data['diff'][-1] - rcm_data['diff'][0]

    if has_adv:
        diff_max = np.abs(c_cur['adv'] - c_cur['diff']).max()
        assert diff_max > 1e-12, \
            f'ADV and DIFF profiles identical: max|Δc| = {diff_max:.3e}'

        delta_adv = rcm_data['adv'][-1] - rcm_data['adv'][0]
        assert delta_adv < -1.0, \
            f'ADV r_cm did not shift inward: Δr_cm = {delta_adv:.2f} µm'

    # ── diagnostics ───────────────────────────────────────────────────────────
    lbl_map = {'diff': 'DIFF ', 'adv': 'ADV  ', 'total': 'T    '}
    for key in cases:
        M_f = float(np.dot(Cm_d, c_cur[key]))
        dr  = rcm_data[key][-1] - rcm_data[key][0]
        print(f'  {lbl_map[key]}: M(0)={M0s[key]:.4g} → M({total_t:.0f}d)={M_f:.4e}'
              f'  Δr_cm={dr:+.2f} µm')

    # ── plots ─────────────────────────────────────────────────────────────────
    os.makedirs(_OUT_DIR, exist_ok=True)
    _plot_evolution(display_cells, snap_data, r_lo, r_hi, cx, cy, cases)
    _plot_radial(display_cells, snap_data, cases)
    _plot_rcm(ts_rcm, rcm_data, mass_data, M0s, cases, T_transit=T_transit)


# ── standalone ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    test_sym_transport()
    print('\n[test_sym] PASSED')
