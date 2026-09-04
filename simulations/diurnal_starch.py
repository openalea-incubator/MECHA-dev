"""
diurnal_starch.py
=================
Diurnal starch sweep.  All simulation logic lives in ``needle.py``;
this module provides the sweep configuration, the comparison plot, and the
runnable entry point.

Use ``DiurnalParams(starch=True, k_starch_syn=k_f, k_starch_deg=k_r)`` to
activate the reversible sucrose <-> starch chemistry.  ``NeedleDiurnalSimulation``
branches on ``params.starch`` automatically.
"""

import os

import numpy as np

from needle import (
    _OUT_DIR,
    DiurnalParams,
    SimResult,
    run_experiments,
    g_photosynthesis,
    diurnal_flux_plot,
)


# ══════════════════════════════════════════════════════════════════════════════
# Comparison plot
# ══════════════════════════════════════════════════════════════════════════════
def starch_comparison_plot(results: list[SimResult], path: str) -> str:
    """Four-panel comparison: j_w, j_s, mean starch, j_s/j_w for all (k_f, k_r)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    ax_jw, ax_js, ax_st, ax_rat = axes

    period  = float(results[0].params.get('period', 1.0))
    g_phase = float(results[0].params.get('g_phase', 0.0))

    def _night_bands(ax, t):
        # Night = dark period (no photosynthesis); f no longer reaches zero.
        tt = np.linspace(t[0], t[-1], 2000)
        night = g_photosynthesis(tt, period, g_phase) <= 0.0
        ax.fill_between(tt, 0, 1, where=night, transform=ax.get_xaxis_transform(),
                        color='0.88', zorder=0, step='mid')

    colors = plt.cm.viridis(np.linspace(0, 0.95, len(results)))

    for res, col in zip(results, colors):
        t   = np.asarray(res.times)
        j_w = np.asarray(res.get_extra('j_w'))
        j_s = np.asarray(res.get_extra('j_s'))
        m_s = res.get_extra('m_starch')
        m_s = np.asarray(m_s) if m_s is not None else np.zeros_like(j_s)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(np.abs(j_w) > 0, j_s / j_w, np.nan)
        kf  = res.params.get('k_starch_syn', 0.0)
        kr  = res.params.get('k_starch_deg', 0.0)
        lbl = f"k_f={kf:.1f}, k_r={kr:.1f}"

        for ax in axes:
            _night_bands(ax, t)

        ax_jw.plot(t, j_w,          color=col, lw=1.2, ms=2, marker='o', label=lbl)
        ax_js.plot(t, j_s  * 1e12,  color=col, lw=1.2, ms=2, marker='o')
        ax_st.plot(t, m_s  * 1e6,   color=col, lw=1.2, ms=2, marker='o')
        ax_rat.plot(t, ratio * 1e3, color=col, lw=1.2, ms=2, marker='o')

    ax_jw.set_ylabel('j_w  (cm³/d)');   ax_jw.set_title('Transpirational water flux j_w')
    ax_jw.legend(fontsize=7, loc='upper right')
    ax_js.set_ylabel('j_s  (pmol/d)');  ax_js.set_title('Sucrose export flux j_s (to sink)')
    ax_st.set_ylabel('mean starch (µM)'); ax_st.set_title('Interior starch (cell mean)')
    ax_rat.set_ylabel('j_s / j_w  (mM)'); ax_rat.set_xlabel('t  (d)')
    ax_rat.set_title('Export ratio j_s / j_w')

    fig.suptitle('Diurnal needle: sucrose ↔ starch sweep  (k_f, k_r)')
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Sweep configuration
# ══════════════════════════════════════════════════════════════════════════════
def build_grid(seed: int = 42,
               kf_values=None,
               kr_values=None,
               max_steps: int = 48) -> list[DiurnalParams]:
    """Build a k_f × k_r grid of DiurnalParams(starch=True) experiments.

    All share ``seed`` so every worker rebuilds the identical anatomy once and
    caches it — the more grid points, the better the parallel amortisation.
    """
    kf_values = (np.linspace(0.0, 5.0, 6) if kf_values is None
                 else np.asarray(kf_values, dtype=float))
    kr_values = (np.linspace(0.0, 5.0, 6) if kr_values is None
                 else np.asarray(kr_values, dtype=float))
    grid = []
    for kf in kf_values:
        for kr in kr_values:
            label  = f"diurnal_starch_kf{kf:.2f}_kr{kr:.2f}".replace('.', 'p')
            starch = (kf != 0.0 or kr != 0.0)
            grid.append(DiurnalParams(
                seed=seed, label=label, max_steps=max_steps,
                starch=starch, k_starch_syn=float(kf), k_starch_deg=float(kr),
            ))
    return grid


def run_sweep(seed: int = 42,
              kf_values=None,
              kr_values=None,
              max_steps: int = 48,
              max_workers=None) -> list[SimResult]:
    """Run the k_f × k_r grid in parallel via the portable run_experiments path."""
    grid = build_grid(seed, kf_values, kr_values, max_steps)
    print(f"Running {len(grid)} experiments (grid) …")
    results = run_experiments(grid, max_workers=max_workers)
    for res in results:
        m = res.get_extra('m_starch')
        m_max = (np.max(m) * 1e6) if m is not None else 0.0
        print(f"  [{res.params['label']}]  "
              f"j_s_max={np.max(res.get_extra('j_s'))*1e12:.1f} pmol/d"
              f"  m_starch_max={m_max:.2f} µM")
    return results


if __name__ == '__main__':
    SEED      = 42
    MAX_STEPS = 48

    print("=== Diurnal starch sweep ===")
    # Curated 6-pair subset for a quick comparison plot (kept small & legible).
    kf_kr = [(0.0, 0.0), (1.0, 0.5), (2.0, 0.5),
             (2.0, 2.0), (5.0, 0.5), (2.0, 5.0)]
    grid = [
        DiurnalParams(seed=SEED, max_steps=MAX_STEPS,
                      starch=(kf != 0.0 or kr != 0.0),
                      k_starch_syn=kf, k_starch_deg=kr,
                      label=f"diurnal_starch_kf{kf:.1f}_kr{kr:.1f}".replace('.', 'p'))
        for kf, kr in kf_kr
    ]
    print(f"Running {len(grid)} experiments in parallel …")
    results = run_experiments(grid, max_workers=None)
    print("Done.")

    stem = os.path.join(_OUT_DIR, 'needle_diurnal_starch')
    starch_comparison_plot(results, stem + '_comparison.png')
    print(f"Saved: {stem}_comparison.png")

    for res in results:
        s = os.path.join(_OUT_DIR, f"needle_{res.params['label']}")
        res.save(s + '.npz')
        diurnal_flux_plot(res, s + '_fluxes.png')
    print("Done.")
