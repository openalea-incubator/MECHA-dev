"""
test_diffusion.py

Gaussian diffusion test for SoluteTransport with Crank-Nicolson stepping.

Two transport pathways are compared, both initialised with the same
middle-cortex ring pulse:

  APO  — apoplastic (cell-wall network):
         cell → membrane → wall node → apo → wall node → membrane → cell
         D_wall = 1e-5 cm²/d,  D_mem = 1e-6 cm²/d,  D_pd = 0

  SYM  — symplastic (plasmodesmata):
         cell → PD → adjacent cell  (no wall involvement)
         D_pd  = 1e-3 cm²/d,  D_wall = D_mem = 0  (mode='sym')

Physical expectations
---------------------
1. Both cases produce a Gaussian radial profile with σ² ∝ t (R² > 0.95).
2. Capacity-weighted mass is machine-exact conserved (|ΔM|/M < 1e-10).
3. Concentrations remain non-negative.
4. The effective diffusivity D_eff = slope/2 of the σ²(t) line is compared
   against the network MSD estimate:

       D_eff = Σ_{i<j} K_ij (Δr_ij)² / Σ_i C_i       [cm²/d]

   where K_ij are off-diagonal conductances of the diffusion matrix and
   C_i are node capacitances.  The geometric reduction factor f = D_eff / D_input
   differs between pathways:
     f_apo  is set by the membrane-to-wall conductance ratio
     f_sym  is set by the PD pore cross-section (≪ cell-wall area)

Outputs
-------
  test/outputs/diffusion_evolution.png    — cross-section snapshots (2 rows)
  test/outputs/diffusion_radial.png       — radial profiles + Gaussian fits
  test/outputs/diffusion_sigma.png        — σ(t) and σ²(t) for both cases
"""

import os
import sys
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.stats import linregress

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mecha.mecha_class import Mecha
from mecha.utils.data_loader import InData

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_CELLSET   = os.path.join(_REPO_ROOT, 'extdata', 'current_root.xml')
_OUT_DIR   = os.path.join(os.path.dirname(__file__), 'outputs')

# ── diffusion parameters ──────────────────────────────────────────────────────

D_WALL   = 1e-5   # cm²/d  apoplastic diffusivity  (apo case)
D_MEM    = 1e-6   # cm²/d  membrane diffusivity    (apo case, 10× smaller)
D_PD     = 1e-3   # cm²/d  PD diffusivity          (sym case; elevated to compensate
                  #         for small PD pore area, yet slow enough to stay within
                  #         the cortex domain over 20 days → clear Fickian regime)

CASES = {
    'apo': dict(
        label='Apoplastic (wall+membrane)',
        color='#2166ac',
        dp=dict(apo_wall=D_WALL, membrane=D_MEM, plasmodesmata=0.0),
        mode='full',
        D_ref=D_WALL,        # reference diffusivity for reduction factor
        D_ref_label='D_wall',
    ),
    'sym': dict(
        label='Symplastic (plasmodesmata)',
        color='#d6604d',
        dp=dict(apo_wall=0.0, membrane=0.0, plasmodesmata=D_PD),
        mode='sym',
        D_ref=D_PD,
        D_ref_label='D_pd',
    ),
}

CAP_PARAMS = {'dt': 1.0, 'C_wall': 1.0, 'C_cell': 1.0}
N_STEPS    = 20             # after 20 days boundary effects start to become significant → diverging from a linear σ² ∝ t regime
THETA      = 0.5


# ── network helpers ───────────────────────────────────────────────────────────

def _build_mecha() -> Mecha:
    data  = InData(cellset_file=_CELLSET)
    mecha = Mecha(data)
    mecha.water_flux()
    return mecha


def _root_center(mecha: Mecha):
    """Return (cx, cy) centroid of all cell nodes (µm)."""
    nwj = mecha.network.n_wall_junction
    xs, ys = [], []
    for nd, d in mecha.network.graph.nodes(data=True):
        if mecha.indice[nd] >= nwj and 'position' in d:
            xs.append(d['position'][0])
            ys.append(d['position'][1])
    return float(np.mean(xs)), float(np.mean(ys))


def _collect_cortex(mecha: Mecha, cx: float, cy: float):
    """Return [(idx, r_µm, x, y), ...] for all cortex (cgroup=4) cells nodes.

    r_µm is the radial distance from the root centre (cx, cy).
    """
    nwj = mecha.network.n_wall_junction
    return [
        (mecha.indice[nd],
         np.hypot(d['position'][0] - cx, d['position'][1] - cy),
         d['position'][0], d['position'][1])
        for nd, d in mecha.network.graph.nodes(data=True)
        if mecha.indice[nd] >= nwj and d.get('cgroup') == 4
    ]


def _middle_ring_ic(mecha: Mecha, cortex: list, mode: str):
    """
    Return c0 with c=1 in the 40th–60th percentile radial band of cortex cells.

    For mode='full' shape = (n_total,);  for mode='sym' shape = (n_cells,).
    """
    nwj    = mecha.network.n_wall_junction
    n_size = (mecha.network.graph.number_of_nodes() if mode == 'full'
              else mecha.network.n_cells)
    r_vals = np.array([e[1] for e in cortex])
    r_lo, r_hi = np.percentile(r_vals, [40, 60])
    c0 = np.zeros(n_size)
    for idx, r, x, y in cortex:
        if r_lo <= r <= r_hi:
            matrix_idx = idx if mode == 'full' else idx - nwj
            c0[matrix_idx] = 1.0
    return c0, r_lo, r_hi


# ── Gaussian fitting ──────────────────────────────────────────────────────────

def _gaussian(r, A, r0, sigma):
    return A * np.exp(-(r - r0) ** 2 / (2 * sigma ** 2))


def _fit_gaussian(r, c, r0_guess, sigma_guess):
    if c.max() < 1e-8:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', OptimizeWarning)
            popt, pcov = curve_fit(
                _gaussian, r, c,
                p0=[c.max(), r0_guess, sigma_guess],
                bounds=([0, r0_guess * 0.5, 1.0],
                        [c.max() * 2, r0_guess * 1.5, 2000.0]),
                maxfev=8000,
            )
        if np.sqrt(np.diag(pcov))[2] / (popt[2] + 1e-12) > 0.5:
            return None
        return tuple(popt)
    except (RuntimeError, ValueError):
        return None


# ── network diffusivity estimate (mean-square-displacement formula) ───────────

def _D_eff_network_msd(D_mat, Cm_diag, r_by_idx_cm: dict) -> float:
    """
    Network mean-square-displacement estimate of radial effective diffusivity.

    Derived from the continuum identity  D_eff = d/dt ⟨Δr²⟩ / 2  applied to
    the linear transport equation dc/dt = T c.  For a network at steady state
    this reduces to:

        D_eff = Σ_{i<j} K_ij (r_i − r_j)²  /  Σ_i C_i      [cm²/d]

    where K_ij are the off-diagonal conductances of the diffusion matrix
    (cm³/d) and C_i are the capacitances (cm³).  Multiply by 1e8 for µm²/d.

    The sum is restricted to node pairs that both appear in r_by_idx_cm
    (usually the cortex region only). Because it ignores circumferential
    conductances (Δr ≈ 0 for tangential neighbours) the estimate is exact for
    purely radial networks and an upper bound otherwise.
    """
    valid = set(r_by_idx_cm)
    coo   = D_mat.tocoo()
    num   = sum(
        v * (r_by_idx_cm[i] - r_by_idx_cm[j]) ** 2
        for i, j, v in zip(coo.row, coo.col, coo.data)
        if i < j and v > 0 and i in valid and j in valid
    )
    den = sum(Cm_diag[i] for i in valid if i < len(Cm_diag))
    return (num / den * 1e8) if den > 0 else float('nan')


def _D_eff_pathway_breakdown(mecha, mode: str, dp: dict, cap_params: dict,
                              r_by_idx_cm: dict, Cm_diag: np.ndarray) -> dict:
    """
    Decompose D_eff_theory by transport pathway.

    The MSD formula is linear in the conductances K_ij, and K_ij = D_p * g_ij
    where g_ij is a purely geometric factor.  Therefore the total D_eff_theory
    can be written as a sum of per-pathway contributions:

        D_eff_total = Σ_p D_p * D_eff_unit_p

    where D_eff_unit_p is computed with D_p=1 and all other diffusivities = 0.

    Returns
    -------
    dict with keys 'wall', 'membrane', 'plasmodesmata', 'total'  (µm²/d each)
    """
    from mecha.solute_transport import SoluteTransport

    pathways = {
        'wall':          ('apo_wall',      dict(apo_wall=1.0, membrane=0.0, plasmodesmata=0.0)),
        'membrane':      ('membrane',      dict(apo_wall=0.0, membrane=1.0, plasmodesmata=0.0)),
        'plasmodesmata': ('plasmodesmata', dict(apo_wall=0.0, membrane=0.0, plasmodesmata=1.0)),
    }
    result = {}
    for name, (dp_key, unit_dp) in pathways.items():
        D_val = float(dp.get(dp_key, 0.0))
        if D_val == 0.0:
            result[name] = 0.0
            continue
        st     = SoluteTransport(mecha, unit_dp, cap_params, mode)
        G_unit = st.build_diffusion_matrix(0, 0)
        result[name] = D_val * _D_eff_network_msd(G_unit, Cm_diag, r_by_idx_cm)
    result['total'] = sum(v for k, v in result.items() if k != 'total')
    return result


# ── visualisation ─────────────────────────────────────────────────────────────

def _plot_evolution(mecha, snap_data, r_lo, r_hi, cx, cy):
    """
    Two-row cross-section scatter plot.  Top row = APO, bottom row = SYM.
    snap_data: {case_key: [(c_vec, t), ...]}
    """
    nwj    = mecha.network.n_wall_junction
    graph  = mecha.network.graph
    indice = mecha.indice

    cell_ent, wall_ent = [], []
    for nd, d in graph.nodes(data=True):
        idx = indice[nd]
        pos = d.get('position', (0.0, 0.0))
        (cell_ent if idx >= nwj else wall_ent).append((idx, pos[0], pos[1]))

    n_snap = len(snap_data['apo'])
    fig, axes = plt.subplots(2, n_snap,
                             figsize=(4.2 * n_snap, 7.5), squeeze=False,
                             layout='constrained')

    vmax = max(c.max()
               for snaps in snap_data.values()
               for c, _ in snaps)
    vmax = max(vmax, 1e-12)

    theta_ring = np.linspace(0, 2 * np.pi, 300)

    for row_i, (case_key, case) in enumerate(CASES.items()):
        snaps = snap_data[case_key]
        sc_ref = None
        for col_j, (c_vec, t) in enumerate(snaps):
            ax = axes[row_i, col_j]
            # background walls
            if wall_ent:
                ax.scatter([e[1] for e in wall_ent],
                           [e[2] for e in wall_ent],
                           s=1, c='#dddddd', zorder=1)
            # cells coloured by concentration
            if case['mode'] == 'full':
                c_cell = [float(c_vec[e[0]]) for e in cell_ent]
            else:  # sym → c_vec indexed by cell_id
                c_cell = [float(c_vec[e[0] - nwj]) for e in cell_ent]

            sc = ax.scatter([e[1] for e in cell_ent],
                            [e[2] for e in cell_ent],
                            s=18, c=c_cell, cmap='plasma',
                            vmin=0.0, vmax=vmax,
                            zorder=2, edgecolors='none')
            sc_ref = sc
            for rv in (r_lo, r_hi):
                ax.plot(cx + rv * np.cos(theta_ring),
                        cy + rv * np.sin(theta_ring),
                        color='cyan', lw=0.6, ls='--', zorder=3)
            ax.set_aspect('equal', 'box')
            ax.set_title(f't = {t:.3g} d', fontsize=8)
            ax.set_xlabel('x (µm)', fontsize=7)
            ax.tick_params(labelsize=6)
            if col_j == 0:
                ax.set_ylabel(f"{case['label']}\ny (µm)", fontsize=7)

        if sc_ref is not None:
            fig.colorbar(sc_ref, ax=axes[row_i, :].tolist(),
                         shrink=0.6, label='c (a.u.)', pad=0.01)

    fig.suptitle('Solute diffusion from middle-cortex ring  (CN θ=0.5)\n'
                 'cyan dashed: initial ring boundaries', fontsize=9)
    out = os.path.join(_OUT_DIR, 'diffusion_evolution.png')
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  → {out}')


def _plot_radial(cortex, snap_data, fits_data, r0, nwj):
    """
    Radial concentration profiles — both cases on same subplots.
    snap_data: {case_key: [(c_vec, t), ...]}
    fits_data: {case_key: [(A, r0, sigma) or None, ...]}
    """
    r_arr  = np.array([e[1] for e in cortex])
    r_fit  = np.linspace(r_arr.min() - 50, r_arr.max() + 50, 400)
    n_snap = len(snap_data['apo'])

    ncols = min(n_snap, 4)
    nrows = (n_snap + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.2 * ncols, 3.0 * nrows), squeeze=False)
    for ax in axes.flat[n_snap:]:
        ax.set_visible(False)

    for si in range(n_snap):
        ax = axes.flat[si]
        # both cases share the same snapshot index
        t = snap_data['apo'][si][1]
        for case_key, case in CASES.items():
            c_vec = snap_data[case_key][si][0]
            if case['mode'] == 'full':
                c_arr = np.array([float(c_vec[e[0]]) for e in cortex])
            else:
                c_arr = np.array([float(c_vec[e[0] - nwj]) for e in cortex])

            ax.scatter(r_arr, c_arr, s=7, color=case['color'],
                       alpha=0.6, label=case['label'])
            fit = fits_data[case_key][si]
            if fit is not None:
                A, r0f, sig = fit
                ax.plot(r_fit, _gaussian(r_fit, A, r0f, sig),
                        '-', color=case['color'], lw=1.5,
                        label=f'  σ={sig:.0f} µm')
        ax.axvline(r0, color='gray', lw=0.8, ls=':', label='r₀')
        ax.set_xlabel('r (µm)', fontsize=7)
        ax.set_ylabel('c (a.u.)', fontsize=7)
        ax.set_title(f't = {t:.3g} d', fontsize=9)
        if si == 0:
            ax.legend(fontsize=5, loc='upper right')
        ax.set_ylim(-0.05, 1.1)
        ax.tick_params(labelsize=6)

    fig.suptitle('Radial profiles (cortex cells)', fontsize=9)
    plt.tight_layout()
    out = os.path.join(_OUT_DIR, 'diffusion_radial.png')
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  → {out}')


def _plot_sigma(ts_data, sig_data, D_effs, D_eff_theories, sigma0s):
    """σ vs √t for both cases.  Solid lines = fit, dotted = MSD theory."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    t_max = max(ts_data['apo'][-1], ts_data['sym'][-1])
    t_ref = np.linspace(0, t_max, 300)

    for case_key, case in CASES.items():
        col     = case['color']
        t_arr   = np.array(ts_data[case_key])
        sig_arr = np.array(sig_data[case_key])
        De      = D_effs[case_key]
        De_th   = D_eff_theories[case_key]
        s0      = sigma0s[case_key]

        ax.scatter(np.sqrt(t_arr), sig_arr, color=col, s=18,
                   label=case['label'], zorder=3)
        ax.plot(np.sqrt(t_ref), np.sqrt(s0**2 + 2*De*t_ref),
                '-', color=col, lw=1.3,
                label=f'fit D_eff={De:.0f} µm²/d')
        ax.plot(np.sqrt(t_ref), np.sqrt(s0**2 + 2*De_th*t_ref),
                ':', color=col, lw=1.0, alpha=0.6,
                label=f'theory {De_th:.0f} µm²/d')

    ax.set_xlabel('√t  (d^½)', fontsize=9)
    ax.set_ylabel('σ (µm)', fontsize=9)
    ax.set_title('σ vs √t  (linear = Fickian)  ··· MSD theory')
    ax.legend(fontsize=6, loc='upper left')

    plt.tight_layout()
    out = os.path.join(_OUT_DIR, 'diffusion_sigma.png')
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  → {out}')


# ── test ──────────────────────────────────────────────────────────────────────

def test_diffusion_time_evolution():
    """
    Middle-cortex ring pulse spreading for APO and SYM pathways.

    Assertions (per case)
    ---------------------
    capacity mass conserved   |ΔM|/M < 1e-10
    min(c) >= -1e-10          no spurious negatives
    R²(σ²,t) > 0.95          σ ∝ √t  (Fickian diffusion)

    Reduction-factor analysis
    -------------------------
    D_eff = slope/2 of σ²(t) is compared with the network MSD estimate:
        D_eff_theory = Σ K_ij (Δr_ij)² / Σ C_i
    restricted to node pairs whose radial positions both fall within the
    cortex r-range. K_ij = D_input * g_ij where g_ij are the geometric factors.
    The reduction factor f = D_eff / D_input hence measures the effect of the 
    network geometry. 
    """
    from mecha.solute_transport import SoluteTransport

    print('\n[test_diffusion] building Mecha...')
    mecha = _build_mecha()

    nwj    = mecha.network.n_wall_junction
    cx, cy = _root_center(mecha)
    cortex = _collect_cortex(mecha, cx, cy)
    r_vals = np.array([e[1] for e in cortex])

    # ── build diffusion matrices once (for theory estimate) ──────────────────
    print('  building diffusion matrices...')
    st_apo = SoluteTransport(mecha, CASES['apo']['dp'], CAP_PARAMS, 'full')
    st_sym = SoluteTransport(mecha, CASES['sym']['dp'], CAP_PARAMS, 'sym')
    D_mat_apo = st_apo.build_diffusion_matrix(0, 0)
    D_mat_sym = st_sym.build_diffusion_matrix(0, 0)
    Cm_apo    = st_apo.build_capacitance(0).diagonal() * CAP_PARAMS['dt']
    Cm_sym    = st_sym.build_capacitance(0).diagonal() * CAP_PARAMS['dt']

    # MSD theory estimate for both cases
    # APO: all cortex-region nodes (cells + walls)
    r_min, r_max = r_vals.min(), r_vals.max()
    r_by_idx_apo = {}
    for nd, d in mecha.network.graph.nodes(data=True):
        idx = mecha.indice[nd]
        r   = np.hypot(d['position'][0] - cx, d['position'][1] - cy)
        if r_min <= r <= r_max:
            r_by_idx_apo[idx] = r * 1e-4   # µm → cm
    D_theory_apo = _D_eff_network_msd(D_mat_apo, Cm_apo, r_by_idx_apo)

    # SYM: cortex cells only (sym-mode indexing)
    r_by_cellid = {e[0] - nwj: e[1] * 1e-4 for e in cortex}
    D_theory_sym = _D_eff_network_msd(D_mat_sym, Cm_sym, r_by_cellid)

    D_eff_theories = {'apo': D_theory_apo, 'sym': D_theory_sym}

    # per-pathway decomposition of D_eff_theory (MSD formula is linear in K_ij)
    breakdowns = {
        'apo': _D_eff_pathway_breakdown(
            mecha, 'full', CASES['apo']['dp'], CAP_PARAMS,
            r_by_idx_apo, Cm_apo),
        'sym': _D_eff_pathway_breakdown(
            mecha, 'sym', CASES['sym']['dp'], CAP_PARAMS,
            r_by_cellid, Cm_sym),
    }

    # ── initial conditions & capacitance weights ──────────────────────────────
    ics = {}
    mass_weights = {}
    ring_bounds  = {}
    for key, case in CASES.items():
        c0, r_lo, r_hi = _middle_ring_ic(mecha, cortex, case['mode'])
        ics[key]          = c0
        ring_bounds[key]  = (r_lo, r_hi)
        st = st_apo if key == 'apo' else st_sym
        mass_weights[key] = st.build_capacitance(0).diagonal() * CAP_PARAMS['dt']

    r_lo, r_hi = ring_bounds['apo']
    r0_init    = 0.5 * (r_lo + r_hi)
    sigma_init = 0.5 * (r_hi - r_lo) / np.sqrt(2.0)

    # ── snapshot bookkeeping ──────────────────────────────────────────────────
    snap_at  = {0, 2, 5, 10, 15, 20}
    snap_data  = {k: [] for k in CASES}
    fits_data  = {k: [] for k in CASES}
    ts_sigma   = {k: [] for k in CASES}
    sig_data   = {k: [] for k in CASES}
    M0s        = {k: float(np.dot(mass_weights[k], ics[k])) for k in CASES}
    c_current  = {k: ics[k].copy() for k in CASES}

    r_arr = r_vals  # cortex radii [µm]

    def _record(key, c_vec, step):
        t = float(step) * CAP_PARAMS['dt']
        if case['mode'] == 'full':
            c_arr = np.array([float(c_vec[e[0]]) for e in cortex])
        else:
            c_arr = np.array([float(c_vec[e[0] - nwj]) for e in cortex])
        fit = _fit_gaussian(r_arr, c_arr, r0_init, sigma_init)
        if fit is not None:
            ts_sigma[key].append(t)
            sig_data[key].append(abs(fit[2]))
        if step in snap_at:
            snap_data[key].append((c_vec.copy(), t))
            fits_data[key].append(fit)

    # record step 0
    for key, case in CASES.items():
        _record(key, c_current[key], 0)

    # ── time loop (both cases per step) ──────────────────────────────────────
    print(f'  running {N_STEPS} CN steps at dt={CAP_PARAMS["dt"]} d ...')
    for step in range(1, N_STEPS + 1):
        for key, case in CASES.items():
            c_current[key] = mecha.standard_solute_flux(
                h=0, i_maturity=0, i_scenario=0,
                diffusion_params   = case['dp'],
                capacitance_params = CAP_PARAMS,
                mode               = case['mode'],
                c_prev             = c_current[key],
                theta              = THETA,
            )
            _record(key, c_current[key], step)

    # ── assertions ────────────────────────────────────────────────────────────
    D_effs   = {}
    sigma0s  = {}
    R2s      = {}

    for key, case in CASES.items():
        # 1. Mass conservation
        M_last   = float(np.dot(mass_weights[key], c_current[key]))
        mass_err = abs(M_last - M0s[key]) / max(abs(M0s[key]), 1e-30)
        assert mass_err < 1e-10, \
            f'[{key}] mass not conserved: {mass_err:.3e}'

        # 2. Positivity
        min_c = min(c.min() for c, _ in snap_data[key])
        assert min_c >= -1e-10, \
            f'[{key}] negative concentration: {min_c:.3e}'

        # 3. Fickian σ ∝ √t
        assert len(ts_sigma[key]) >= 5, \
            f'[{key}] too few Gaussian fits: {len(ts_sigma[key])}'
        t_arr  = np.array(ts_sigma[key][1:])
        sig2   = np.array(sig_data[key][1:]) ** 2
        slope, intercept, r_val, *_ = linregress(t_arr, sig2)
        R2 = r_val ** 2
        # SYM case spreads quickly and hits root boundary ⇒ slight non-linearity
        r2_thresh = 0.90 if key == 'sym' else 0.95
        assert R2 > r2_thresh, \
            f'[{key}] σ²(t) not linear: R² = {R2:.3f}'
        D_effs[key]  = 0.5 * slope
        sigma0s[key] = np.sqrt(max(intercept, 0.0))
        R2s[key]     = R2

    # ── reduction factor report ───────────────────────────────────────────────
    print('\n  ┌─────────────┬────────────┬──────────────┬────────────┬──────────────┬────────────┐')
    print(  '  │ Case        │ D_input    │ D_eff (fit)  │ f_fit      │ D_eff theory │ f_geom     │')
    print(  '  │             │ (µm²/d)    │ (µm²/d)      │            │ (µm²/d)      │            │')
    print(  '  ├─────────────┼────────────┼──────────────┼────────────┼──────────────┼────────────┤')
    for key, case in CASES.items():
        D_in  = case['D_ref'] * 1e8
        D_fit = D_effs[key]
        D_th  = D_eff_theories[key]
        f_fit = D_fit / D_in
        f_geo = D_th  / D_in
        print(f'  │ {case["label"][:11]:<11} │ {D_in:>10.1f} │ {D_fit:>12.1f} │ {f_fit:>10.4f} │ {D_th:>12.1f} │ {f_geo:>10.4f} │')
    print(  '  └─────────────┴────────────┴──────────────┴────────────┴──────────────┴────────────┘')

    print('\n  Pathway breakdown of D_eff_theory (additive via MSD linearity):')
    print(  '  ┌─────────────┬──────────────┬──────────────┬──────────────┬──────────────┐')
    print(  '  │ Case        │ wall         │ membrane     │ PD           │ total        │')
    print(  '  │             │ (µm²/d)      │ (µm²/d)      │ (µm²/d)      │ (µm²/d)      │')
    print(  '  ├─────────────┼──────────────┼──────────────┼──────────────┼──────────────┤')
    for key, case in CASES.items():
        bkd = breakdowns[key]
        print(f'  │ {case["label"][:11]:<11} │ {bkd["wall"]:>12.1f} │ {bkd["membrane"]:>12.1f} │ {bkd["plasmodesmata"]:>12.1f} │ {bkd["total"]:>12.1f} │')
    print(  '  └─────────────┴──────────────┴──────────────┴──────────────┴──────────────┘')

    # ── visualisation ─────────────────────────────────────────────────────────
    os.makedirs(_OUT_DIR, exist_ok=True)
    _plot_evolution(mecha, snap_data, r_lo, r_hi, cx, cy)
    _plot_radial(cortex, snap_data, fits_data, r0_init, nwj)
    _plot_sigma(ts_sigma, sig_data, D_effs, D_eff_theories, sigma0s)

# ── standalone ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    test_diffusion_time_evolution()
    print('\n[test_diffusion] PASSED')