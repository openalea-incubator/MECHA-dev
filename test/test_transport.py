"""
test_transport.py

Combined APO+SYM transport test using a Neumann (flow) BC at the xylem
to set the velocity field — no artificial flux scaling.

Strategy
--------
Fix total xylem uptake F_total as a Neumann BC at the xylem boundary.
By mass conservation this same flux F_total enters through the outer ring,
directly setting the mean radial velocity:

    ū(r_lo) = F_total / (2π · r_lo · h)          [L·ū = F_total / (2π·h)]

Choose D (one value for both APO and SYM) to achieve Pe = 1:

    Pe = ū · r_lo / D  = F_total / (2π · h · D) = 1
    →  D = F_total / (2π · h)

MECHA is then solved once with this Neumann BC, and the resulting
edge-flux field is used directly for both transport pathways without
any post-hoc rescaling.

Three operators per mode:
  D  diffusion only
  A  advection only
  T  full advection + diffusion (A + D)

Outputs
-------
  test/outputs/transport_apo_evolution.png
  test/outputs/transport_sym_evolution.png
  test/outputs/transport_radial.png
  test/outputs/transport_rcm.png
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

# Target Pe = 1; D is computed from F_total = 2π·h·D after the Neumann solve.
# Choose a single D for both APO (apo_wall) and SYM (plasmodesmata).
PE_TARGET  = 1.0          # Péclet number to enforce
THETA      = 1.0          # implicit Euler

CASES  = ['diff', 'adv', 'total']
COLORS = {'diff': '#2166ac', 'adv': '#d6604d', 'total': '#4dac26'}
LABELS = {'diff': 'Diffusion (D)', 'adv': 'Advection (A)', 'total': 'Full (T=A+D)'}
OPS    = {'diff': 'D', 'adv': 'A', 'total': 'T'}

_NONPARENCHYMA_CGROUPS = frozenset({13, 19, 20, 11, 12})


# ── Neumann BC hydraulic solve ─────────────────────────────────────────────────

def _build_mecha_neumann():
    """
    Build Mecha with homogeneous tissue and Neumann BC at xylem.

    The xylem pressure BC is replaced by a prescribed total radial flow
    F_total (Neumann BC).  psi_soil = 0 remains the reference Dirichlet
    at the outer ring (already the XML default).

    Returns
    -------
    mecha   : Mecha with edge_flux_list[0][0] populated
    D       : diffusivity (cm²/d) that gives Pe = PE_TARGET
    F_total : total xylem uptake used as Neumann BC (cm³/d)
    h_cm    : root segment height (cm)
    """
    data  = InData(cellset_file=_CELLSET)
    mecha = Mecha(data)
    h_idx, i_mat = 0, 0

    # Homogeneous tissue: remove Casparian-strip conductance barrier
    kw_base   = float(mecha.hydraulic.get_kw_value(h_idx))
    kw_config = mecha.hydraulic_conductivities[h_idx, i_mat, 1]['kw']
    kw_config['kw_endo_endo'] = kw_base

    h_cm = float(mecha.geometry.maturity_stages[0].get('height')) * 1e-4  # µm → cm

    # F_total chosen so that D = F_total/(2π·h) gives Pe=1 at the cortex ring.
    # We pick D first (a round number), then derive F_total.
    # D = 5e-5 cm²/d gives a transit time of ~10–50 d depending on r_lo.
    D = 5e-5            # cm²/d  (target diffusivity for Pe = PE_TARGET)
    F_total = 2.0 * np.pi * h_cm * D * PE_TARGET   # cm³/d

    # Switch xylem BC from Dirichlet (psi_xyl) to Neumann (flow):
    mecha.psi_xyl[1, 0, 0] = np.nan                   # disable pressure BC
    mecha.distributed_flow_xyl[1, 0, 0] = F_total     # total xylem uptake (cm³/d)
    mecha._distribute_xylem_flow()                     # distribute across xylem cells

    # Solve: psi_soil = 0 (reference), xylem takes up F_total
    sol, mat_W = mecha.water_flux()

    # Build edge_flux_list[0][0] from the Neumann solution
    fluxes = []
    coo = mat_W.tocoo()
    for i, j, v in zip(coo.row, coo.col, coo.data):
        if i < j and v > 0:
            f = float(v) * (float(sol[i][0]) - float(sol[j][0]))
            fluxes.append({'source': int(i), 'target': int(j), 'flux': f})
    mecha.edge_flux_list[0][0] = fluxes

    return mecha, D, F_total, h_cm


# ── shared geometry helpers ────────────────────────────────────────────────────

def _root_center(mecha):
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

def _stele_bc_nodes(mecha):
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


def _collect_display_nodes(mecha, cx, cy):
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

def _collect_display_cells(mecha, cx, cy):
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


def _inner_stele_bc(display_cells, r_thresh_um):
    return {node_id: 0.0
            for node_id, _, r, *_ in display_cells
            if r < r_thresh_um}


# ── plots ─────────────────────────────────────────────────────────────────────

def _plot_evolution(xs, ys, get_vals, snap_data, r_lo, r_hi, cx, cy,
                    dot_size, vmax, suptitle, out_name):
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
                             vmin=0.0, vmax=vmax, zorder=2, edgecolors='none')
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
        for ax, t, ylabel in zip(
                [axes[0, col_j], axes[1, col_j]],
                [t_apo, t_sym],
                ['c  APO (wall nodes)', 'c  SYM (cell nodes)']):
            ax.set_xlabel('r (µm)', fontsize=7)
            ax.set_ylabel(ylabel, fontsize=7)
            ax.set_title(f't = {t:.1f} d', fontsize=8)
            ax.set_ylim(bottom=-0.02)
            ax.tick_params(labelsize=6)
            if col_j == 0:
                ax.legend(fontsize=6)

    fig.suptitle('Radial profiles — APO (top) and SYM (bottom)', fontsize=9)
    plt.tight_layout()
    out = os.path.join(_OUT_DIR, 'transport_radial.png')
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  → {out}')


def _plot_rcm(ts_apo, rcm_apo, mass_apo, M0_apo, T_tr_apo,
              ts_sym, rcm_sym, mass_sym, M0_sym, T_tr_sym):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex='col')

    for col_i, (ts, rcm, mass, M0s, T_tr, lbl) in enumerate([
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
        ax_r.set_title(lbl, fontsize=9)
        ax_r.legend(fontsize=7)
        ax_m.set_xlabel('Time (d)', fontsize=9)
        ax_m.set_ylabel('M(t) / M₀', fontsize=9)
        ax_m.set_ylim(-0.02, 1.05)
        ax_m.legend(fontsize=7)

    fig.suptitle('Concentration-weighted r_cm and mass fraction\n'
                 'Neumann xylem BC | absorbing BC at stele (ADV, T)',
                 fontsize=9)
    plt.tight_layout()
    out = os.path.join(_OUT_DIR, 'transport_rcm.png')
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  → {out}')


# ── test ──────────────────────────────────────────────────────────────────────

def test_transport():
    """
    Combined APO and SYM transport with Neumann xylem BC.

    Assertions
    ----------
    APO DIFF  |ΔM|/M < 0.01   mass conserved
    SYM DIFF  |ΔM|/M < 0.01   mass conserved
    no negatives               min(c) >= -1e-6 for every case
    APO ADV   Δr_cm < -1 µm   front sweeps inward
    SYM ADV   Δr_cm < -1 µm   front sweeps inward
    """
    print('\n[test_transport] building Mecha with Neumann xylem BC ...')
    mecha, D, F_total, h_cm = _build_mecha_neumann()
    nwj     = mecha.network.n_wall_junction
    n_cells = mecha.network.n_cells
    cx, cy  = _root_center(mecha)

    print(f'  h = {h_cm*1e4:.0f} µm  '
          f'F_total = {F_total:.3e} cm³/d  '
          f'D = {D:.2e} cm²/d  (Pe = {PE_TARGET:.0f})')

    # Verify that fluxes are non-trivial (Neumann BC produced a real flow)
    nonzero = [f for f in mecha.edge_flux_list[0][0] if abs(f['flux']) > 0]
    print(f'  edge_flux_list: {len(mecha.edge_flux_list[0][0])} edges, '
          f'{len(nonzero)} with |flux| > 0')
    assert len(nonzero) > 0, 'Neumann BC produced no non-zero fluxes'

    # Verify sign: net inward flow (source > target means higher r → lower r)
    total_inward = sum(f['flux'] for f in mecha.edge_flux_list[0][0])
    print(f'  Σ flux = {total_inward:.4e}  (should reflect radial bias)')

    display_nodes = _collect_display_nodes(mecha, cx, cy)
    display_cells = _collect_display_cells(mecha, cx, cy)
    print(f'  wall nodes (L>0): {len(display_nodes)}  |  '
          f'cell nodes (area>0): {len(display_cells)}')

    # ─────────────────────────────────────────────────────────────────────────
    # APO RUN
    # ─────────────────────────────────────────────────────────────────────────
    print('\n[APO] Neumann velocity field, Pe = 1 by construction ...')

    # D gives Pe=1 by design; verify by checking advection vs diffusion matrices
    cap_apo = {'dt': 0.1, 'C_wall': 1.0, 'C_cell': 1.0}
    DP_APO  = dict(apo_wall=D, membrane=0.0, plasmodesmata=0.0)

    st_pe   = SoluteTransport(mecha, DP_APO, cap_apo, mode='apo')
    A_tmp   = st_pe.build_advection_matrix(0, 0)
    D_tmp   = st_pe.build_diffusion_matrix(0, 0)

    c0_tmp, r_lo_tmp, _ = _cortex_wall_ic(display_nodes, nwj)
    ring_w  = np.where(c0_tmp > 0)[0]
    r_lo_cm = r_lo_tmp * 1e-4

    Ad_ring  = np.abs(A_tmp.diagonal())[ring_w]
    Dd_ring  = np.abs(D_tmp.diagonal())[ring_w]
    valid    = (Ad_ring > 0) & (Dd_ring > 0)
    pe_ring   = float((Ad_ring[valid] / Dd_ring[valid]).mean()) if valid.any() else float('nan')
    pe_global = F_total / (2.0 * np.pi * h_cm * D)   # = 1 by construction
    print(f'  Pe_global={pe_global:.3f} (by construction)  Pe_local_ring={pe_ring:.3f}')

    # Derive time-stepping from transit time T = r_lo²/D
    T_tr_apo     = r_lo_cm ** 2 / D                         # days
    dt_apo       = max(0.1, round(T_tr_apo / 100.0, 1))
    T_steps_apo  = int(np.ceil(T_tr_apo / dt_apo))
    N_apo        = max(200, min(2000, 5 * T_steps_apo))
    cap_apo['dt'] = dt_apo

    print(f'  r_lo={r_lo_cm*1e4:.1f}µm  T_transit={T_tr_apo:.1f}d  '
          f'dt={dt_apo}d  N={N_apo}')

    bc_apo_stele = _stele_bc_nodes(mecha)
    print(f'  absorbing BC at {len(bc_apo_stele)} stele wall nodes')

    st_apo    = SoluteTransport(mecha, DP_APO, cap_apo, mode='apo')
    Cm_apo    = st_apo.build_capacitance(0)
    Cm_d_apo  = Cm_apo.diagonal()

    c0_apo, r_lo_apo, r_hi_apo = _cortex_wall_ic(display_nodes, nwj)
    print(f'  IC: cortex ring r=[{r_lo_apo:.1f}, {r_hi_apo:.1f}] µm  '
          f'({int(c0_apo.sum())} wall nodes)')

    disp_idxs    = np.array([e[0] for e in display_nodes], dtype=int)
    r_disp_apo   = np.array([e[1] for e in display_nodes])
    apo_get_vals = lambda c: c[disp_idxs]

    apo_bc      = {'diff': {}, 'adv': bc_apo_stele, 'total': bc_apo_stele}
    # Snapshots: IC, one very-early (step 1), ~T/20, ~T/4, T, 5T
    # Step 1 (t=dt≈0.2d) shows the ring before diffusion has spread it significantly.
    snap_at_apo = {0, 1,
                   max(2, T_steps_apo // 20),
                   max(3, T_steps_apo // 4),
                   T_steps_apo, N_apo}
    snap_apo    = {k: [] for k in CASES}
    ts_apo      = {k: [] for k in CASES}
    rcm_apo     = {k: [] for k in CASES}
    mass_apo    = {k: [] for k in CASES}
    M0_apo      = {k: float(np.dot(Cm_d_apo, c0_apo)) for k in CASES}
    c_apo       = {k: c0_apo.copy() for k in CASES}

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
    # SYM RUN  (same velocity field, D used as plasmodesmata conductance)
    # ─────────────────────────────────────────────────────────────────────────
    print('\n[SYM] Neumann velocity field, same D → Pe = 1 by construction ...')

    cap_sym = {'dt': 0.1, 'C_wall': 1.0, 'C_cell': 1.0}
    DP_SYM  = dict(apo_wall=0.0, membrane=0.0, plasmodesmata=D)

    st_pe_s  = SoluteTransport(mecha, DP_SYM, cap_sym, mode='sym')
    A_tmp_s  = st_pe_s.build_advection_matrix(0, 0)
    D_tmp_s  = st_pe_s.build_diffusion_matrix(0, 0)

    c0_tmp_s, r_lo_tmp_s, _ = _cortex_cell_ic(display_cells, n_cells)
    ring_c    = np.where(c0_tmp_s > 0)[0]
    r_lo_cm_s = r_lo_tmp_s * 1e-4

    Ad_diag_s  = np.abs(A_tmp_s.diagonal())
    has_adv_s  = (Ad_diag_s.max() > 0
                  and len(ring_c) > 0
                  and Ad_diag_s[ring_c].max() > 0)

    Dd_ring_s  = np.abs(D_tmp_s.diagonal())[ring_c]
    if has_adv_s:
        Ad_ring_s  = Ad_diag_s[ring_c]
        valid_s    = (Ad_ring_s > 0) & (Dd_ring_s > 0)
        pe_ring_s  = float((Ad_ring_s[valid_s] / Dd_ring_s[valid_s]).mean()) \
                     if valid_s.any() else float('nan')
        print(f'  Pe_global={pe_global:.3f}  Pe_local_ring(SYM)={pe_ring_s:.3f}')

    # Time-stepping from cell-level diffusion timescale
    vols_all  = st_pe_s._compute_node_volumes(0)
    V_ring    = vols_all[nwj + ring_c]
    mask_tau  = (Dd_ring_s > 0) & (V_ring > 0)
    tau_cell  = float(np.mean(V_ring[mask_tau] / Dd_ring_s[mask_tau])) \
                if mask_tau.any() else 1.0

    ca         = mecha.network.cell_areas
    areas_ring = np.array([float(ca[cid]) for cid in ring_c if cid < len(ca)])
    L_cell_cm  = float(np.sqrt(np.mean(areas_ring[areas_ring > 0]))) * 1e-4 \
                 if areas_ring.size > 0 else r_lo_cm_s / 10
    N_hops     = r_lo_cm_s / L_cell_cm

    T_tr_sym = r_lo_cm_s ** 2 / D      # global transit time (same D as APO)

    dt_raw   = tau_cell / 10.0
    mag      = 10.0 ** np.floor(np.log10(max(dt_raw, 1e-10)))
    dt_sym   = max(1e-4, round(dt_raw / mag) * mag)
    # Cap dt so the transit window contains at least 10 steps
    dt_sym   = max(1e-4, min(dt_sym, T_tr_sym / 10.0))

    T_diff_steps = max(10, int(np.ceil(N_hops ** 2 * tau_cell / dt_sym)))
    T_adv_steps  = max(5, int(np.ceil(T_tr_sym / dt_sym)))
    N_sym        = max(100, min(500, 10 * T_adv_steps))
    cap_sym['dt'] = dt_sym

    print(f'  τ_cell={tau_cell:.1f}d  L_cell={L_cell_cm*1e4:.1f}µm  '
          f'N_hops={N_hops:.1f}  dt={dt_sym:.2g}d  '
          f'T_adv≈{T_tr_sym:.0f}d ({T_adv_steps}st)  run={N_sym}st')

    r_all_sym    = np.array([r for _, _, r, *_ in display_cells])
    r_inner      = float(np.percentile(r_all_sym, 20))
    bc_sym_stele = _inner_stele_bc(display_cells, r_inner) if has_adv_s else {}
    print(f'  absorbing BC at {len(bc_sym_stele)} innermost cells (r < {r_inner:.1f} µm)')

    st_sym    = SoluteTransport(mecha, DP_SYM, cap_sym, mode='sym')
    Cm_sym    = st_sym.build_capacitance(0)
    Cm_d_sym  = Cm_sym.diagonal()

    c0_sym, r_lo_sym, r_hi_sym = _cortex_cell_ic(display_cells, n_cells)
    print(f'  IC: cortex cell ring r=[{r_lo_sym:.1f}, {r_hi_sym:.1f}] µm  '
          f'({int((c0_sym > 0).sum())} cells)')

    disp_cids    = np.array([e[1] for e in display_cells], dtype=int)
    r_disp_sym   = np.array([e[2] for e in display_cells])
    sym_get_vals = lambda c: c[disp_cids]

    sym_bc      = {'diff': {}, 'adv': bc_sym_stele, 'total': bc_sym_stele}
    # Snapshots: IC, step 1, ~T/4, ~T/2, T, N  (6 panels matching APO)
    snap_at_sym = {0, 1,
                   max(2, T_adv_steps // 4),
                   max(3, T_adv_steps // 2),
                   max(4, T_adv_steps), N_sym}
    snap_sym    = {k: [] for k in CASES}
    ts_sym      = {k: [] for k in CASES}
    rcm_sym     = {k: [] for k in CASES}
    mass_sym    = {k: [] for k in CASES}
    M0_sym      = {k: float(np.dot(Cm_d_sym, c0_sym)) for k in CASES}
    c_sym_cur   = {k: c0_sym.copy() for k in CASES}

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

    for key in CASES:
        min_c = min(c.min() for c, _ in snap_apo[key])
        assert min_c >= -1e-6, f'[APO {key}] negative concentration: {min_c:.3e}'
    M_f_apo = float(np.dot(Cm_d_apo, c_apo['diff']))
    rel_apo  = abs(M_f_apo - M0_apo['diff']) / max(M0_apo['diff'], 1e-30)
    assert rel_apo < 0.01, f'APO DIFF mass error: {rel_apo:.1%}'
    _adv_arr = np.array(rcm_apo['adv'])
    _adv_fin = _adv_arr[np.isfinite(_adv_arr)]
    dr_apo   = float(_adv_fin[-1] - _adv_fin[0]) if len(_adv_fin) >= 2 else float('nan')
    assert np.isfinite(dr_apo) and dr_apo < -1.0, \
        f'APO ADV r_cm did not shift inward: {dr_apo}'

    for key in CASES:
        min_c = min(c.min() for c, _ in snap_sym[key])
        assert min_c >= -1e-6, f'[SYM {key}] negative concentration: {min_c:.3e}'
    M_f_sym = float(np.dot(Cm_d_sym, c_sym_cur['diff']))
    rel_sym  = abs(M_f_sym - M0_sym['diff']) / max(M0_sym['diff'], 1e-30)
    assert rel_sym < 0.01, f'SYM DIFF mass error: {rel_sym:.1%}'
    if has_adv_s:
        _sadv_arr = np.array(rcm_sym['adv'])
        _sadv_fin = _sadv_arr[np.isfinite(_sadv_arr)]
        dr_sym    = float(_sadv_fin[-1] - _sadv_fin[0]) if len(_sadv_fin) >= 2 else float('nan')
        assert np.isfinite(dr_sym) and dr_sym < -1.0, \
            f'SYM ADV r_cm did not shift inward: {dr_sym}'

    # ── diagnostics ───────────────────────────────────────────────────────────
    total_apo = N_apo * cap_apo['dt']
    total_sym = N_sym * cap_sym['dt']
    print(f'\n  ┌──────────┬──────────────┬──────────────┬──────────────┐')
    print(f'  │ APO case │ M(t=0)       │ M(t={total_apo:.0f}d)     │ Δr_cm (µm)   │')
    print(f'  ├──────────┼──────────────┼──────────────┼──────────────┤')
    for key in CASES:
        M_f = float(np.dot(Cm_d_apo, c_apo[key]))
        _ra = np.array(rcm_apo[key]); _rf = _ra[np.isfinite(_ra)]
        dr  = float(_rf[-1] - _rf[0]) if len(_rf) >= 2 else float('nan')
        lbl = {'diff': 'Diff', 'adv': 'Adv', 'total': 'T'}[key]
        dr_s = f'{dr:>+12.2f}' if np.isfinite(dr) else f'{"absorbed":>12s}'
        print(f'  │ {lbl:<8} │ {M0_apo[key]:>12.4g} │ {M_f:>12.4g} │ {dr_s} │')
    print(f'  └──────────┴──────────────┴──────────────┴──────────────┘')

    print(f'\n  ┌──────────┬──────────────┬──────────────┬──────────────┐')
    print(f'  │ SYM case │ M(t=0)       │ M(t={total_sym:.0f}d)    │ Δr_cm (µm)   │')
    print(f'  ├──────────┼──────────────┼──────────────┼──────────────┤')
    for key in CASES:
        M_f = float(np.dot(Cm_d_sym, c_sym_cur[key]))
        _rs = np.array(rcm_sym[key]); _rsf = _rs[np.isfinite(_rs)]
        dr  = float(_rsf[-1] - _rsf[0]) if len(_rsf) >= 2 else float('nan')
        lbl = {'diff': 'Diff', 'adv': 'Adv', 'total': 'T'}[key]
        dr_s = f'{dr:>+12.2f}' if np.isfinite(dr) else f'{"absorbed":>12s}'
        print(f'  │ {lbl:<8} │ {M0_sym[key]:>12.4g} │ {M_f:>12.4g} │ {dr_s} │')
    print(f'  └──────────┴──────────────┴──────────────┴──────────────┘')

    # ── plots ─────────────────────────────────────────────────────────────────
    os.makedirs(_OUT_DIR, exist_ok=True)

    xs_apo = np.array([e[2] for e in display_nodes])
    ys_apo = np.array([e[3] for e in display_nodes])
    _plot_evolution(
        xs_apo, ys_apo, apo_get_vals, snap_apo,
        r_lo_apo, r_hi_apo, cx, cy,
        dot_size=4, vmax=1.0,
        suptitle=(f'APO transport — Neumann BC (F={F_total:.2e} cm³/d, Pe={pe_ring:.2f})\n'
                  f'D = {D:.2e} cm²/d | absorbing BC at stele (ADV, T)'),
        out_name='transport_apo_evolution.png',
    )

    xs_sym = np.array([e[3] for e in display_cells])
    ys_sym = np.array([e[4] for e in display_cells])
    _plot_evolution(
        xs_sym, ys_sym, sym_get_vals, snap_sym,
        r_lo_sym, r_hi_sym, cx, cy,
        dot_size=10, vmax=1.0,
        suptitle=(f'SYM transport — Neumann BC (F={F_total:.2e} cm³/d)\n'
                  f'D = {D:.2e} cm²/d | absorbing BC at stele (ADV, T)'),
        out_name='transport_sym_evolution.png',
    )

    _plot_radial(display_nodes, apo_get_vals, snap_apo,
                 display_cells, sym_get_vals, snap_sym)

    _plot_rcm(ts_apo, rcm_apo, mass_apo, M0_apo, T_tr_apo,
              ts_sym, rcm_sym, mass_sym, M0_sym, T_tr_sym)


# ── standalone ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    test_transport()
    print('\n[test_transport] PASSED')
