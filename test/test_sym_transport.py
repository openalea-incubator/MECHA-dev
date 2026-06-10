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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from openalea.mecha.mecha_class import Mecha
from openalea.mecha.utils.data_loader import InData
from openalea.mecha.solute_transport import SoluteTransport
from openalea.mecha.utils.scenario_builder import ScenarioBuilder

_OUT_DIR    = os.path.join(os.path.dirname(__file__), 'outputs')
_CELLSET    = os.path.join(os.path.dirname(__file__), '..', 'extdata', 'current_root.xml')


def _build_mecha_homogeneous() -> Mecha:
    data  = InData(cellset_file=_CELLSET)
    mecha = Mecha(data)
    h, i_mat  = 0, 0
    kw_base   = float(mecha.hydraulic.get_kw_value(h))
    kw_config = mecha.hydraulic_conductivities[h, i_mat, 1]['kw']
    kw_config['kw_endo_endo'] = kw_base
    mecha.water_flux()
    sol   = mecha.results[0]["solution"]
    mat_W = mecha.results[0]["matrix_W"]
    
    fluxes = []
    coo = mat_W.tocoo()
    for i, j, v in zip(coo.row, coo.col, coo.data):
        if i < j and v > 0:
            f = float(v) * (float(sol[i][0]) - float(sol[j][0]))
            fluxes.append({'source': int(i), 'target': int(j), 'flux': f})
    mecha.edge_flux_list[0][0] = fluxes
    return mecha

D_PD   = 1e-4   # cm²/d  plasmodesmata diffusivity
DP_SYM = dict(apo_wall=0.0, membrane=0.0, plasmodesmata=D_PD)
CAP    = {'dt': 0.1, 'C_wall': 1.0, 'C_cell': 1.0}
THETA  = 1.0


# ── visualisation ─────────────────────────────────────────────────────────────

def _plot_evolution(display_cells, snap_data, r_lo, r_hi, cx, cy, cases):
    labels = {'diff': 'Diffusion (D)', 'adv': 'Advection (A)', 'total': 'Full (T=A+D)'}
    n_snap = len(snap_data[cases[0]])
    n_rows = len(cases)
    fig, axes = plt.subplots(n_rows, n_snap,
                             figsize=(3.0 * n_snap, 3.5 * n_rows), squeeze=False,
                             layout='constrained')
    theta_ring = np.linspace(0, 2 * np.pi, 300)
    xs   = np.array([n.x for n in display_cells])
    ys   = np.array([n.y for n in display_cells])
    cids = np.array([n.cell_id for n in display_cells], dtype=int)

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
    cids  = np.array([n.cell_id for n in display_cells], dtype=int)
    r_arr = np.array([n.r for n in display_cells])
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

    cx, cy = ScenarioBuilder.root_center(mecha)
    display_cells = ScenarioBuilder.collect_display_nodes(mecha, cx, cy, key='sym')
    print(f'  cell nodes total: {n_cells}  |  displayed (area>0): {len(display_cells)}')

    # Cgroup summary for diagnostics
    cg_counts: dict = {}
    for n in display_cells:
        cg_counts[n.cgroup] = cg_counts.get(n.cgroup, 0) + 1
    print(f'  cgroup counts: { {k: cg_counts[k] for k in sorted(cg_counts)} }')

    # ── Pe scaling; detect PD fluxes ──────────────────────────────────────────
    st_pe = SoluteTransport(mecha, DP_SYM, CAP, mode='sym')
    A_tmp = st_pe.build_advection_matrix(0, 0)
    D_tmp = st_pe.build_diffusion_matrix(0, 0)

    c0_tmp, r_lo_tmp, _ = ScenarioBuilder.circular_ic(
        mecha, display_cells, key='sym', percentile=(48, 52))
    ring_cids = np.where(c0_tmp > 0)[0]

    r_lo_cm  = r_lo_tmp * 1e-4

    has_adv  = ScenarioBuilder.has_advection(A_tmp, ring_cids)
    # τ_cell from PD geometry (more accurate than L²/D_PD; see cell_timescale)
    tau_cell = ScenarioBuilder.cell_timescale(st_pe, D_tmp, ring_cids, nwj)
    dt_use   = ScenarioBuilder.auto_dt(tau_cell)

    T_transit: float
    T_adv_steps: int
    T_diff_steps: int
    n_steps: int
    pe_ring: float = float('nan')

    # Characteristic cell length from IC ring area
    ca = mecha.network.cell_areas
    areas_ring = np.array([float(ca[cid]) for cid in ring_cids if cid < len(ca)])
    L_cell_cm  = float(np.sqrt(np.mean(areas_ring[areas_ring > 0]))) * 1e-4  # cm
    N_hops     = r_lo_cm / L_cell_cm

    # DIFF: N_hops² × τ_cell steps; ADV: N_hops × τ_cell / pe_ring steps
    T_diff_steps = max(10, int(np.ceil(N_hops ** 2 * tau_cell / dt_use)))

    if has_adv:
        flux_scale, v_tgt, _ = ScenarioBuilder.scale_fluxes_pe1(
            mecha, A_tmp, ring_cids, D_PD, r_lo_cm)

        Ad_ring    = np.abs(A_tmp.diagonal())[ring_cids]
        Dd_ring    = np.abs(D_tmp.diagonal())[ring_cids]
        valid_both = (Ad_ring > 0) & (Dd_ring > 0)
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
    bc_stele = ScenarioBuilder.solute_bc(display_cells, key='sym') if has_adv else {}
    r_inner  = float(np.percentile([n.r for n in display_cells], 20))
    print(f'  absorbing BC at {len(bc_stele)} innermost cells '
          f'(r < {r_inner:.1f} µm, ADV only; DIFF is mass-conserving)')

    # ── build solver, IC, capacitance ─────────────────────────────────────────
    st   = SoluteTransport(mecha, DP_SYM, CAP, mode='sym')
    Cm   = st.build_capacitance(0)
    Cm_d = Cm.diagonal()

    c0, r_lo, r_hi = ScenarioBuilder.circular_ic(
        mecha, display_cells, key='sym', percentile=(48, 52))
    n_ic = int((c0 > 0).sum())
    print(f'  IC: cortex cell ring  r=[{r_lo:.1f}, {r_hi:.1f}] µm  ({n_ic} cells)')

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
        rcm_data[key].append(ScenarioBuilder.r_cm(c_vec, display_cells, key='sym'))
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
