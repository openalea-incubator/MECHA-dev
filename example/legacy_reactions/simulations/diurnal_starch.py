"""
diurnal_starch.py
=================
Diurnal starch sweep.  Simulation logic lives in ``needle.py`` and the plots in
``post_sim.py``; this module provides the sweep configuration and the runnable
entry point.

Use ``DiurnalParams(starch=True, k_starch_syn=k_f, k_starch_deg=k_r)`` to
activate the reversible sucrose <-> starch chemistry.  ``NeedleDiurnalSimulation``
branches on ``params.starch`` automatically.
"""

import os

import numpy as np

from needle import _OUT_DIR, DiurnalParams, run_experiments
from post_sim import (diurnal_flux_plot, starch_comparison_plot,
                      concentration_gif, starch_concentration_gif,
                      load_results)


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
              max_workers=None) -> list:
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
    MAX_STEPS = 72

    # Set RUN_SIM=True to rerun the sweep; False to load the last saved results.
    RUN_SIM = True

    KF_KR = [(0.0, 0.0), (1.0, 0.5), (2.0, 0.5),
             (2.0, 2.0), (5.0, 0.5), (2.0, 5.0)]

    if RUN_SIM:
        print("=== Diurnal starch sweep (running) ===")
        grid = [
            DiurnalParams(seed=SEED, max_steps=MAX_STEPS,
                          starch=(kf != 0.0 or kr != 0.0),
                          k_starch_syn=kf, k_starch_deg=kr,
                          label=f"diurnal_starch_kf{kf:.1f}_kr{kr:.1f}".replace('.', 'p'))
            for kf, kr in KF_KR
        ]
        print(f"Running {len(grid)} experiments in parallel …")
        results = run_experiments(grid, max_workers=None)
        for res in results:
            s = os.path.join(_OUT_DIR, f"needle_{res.params['label']}")
            res.save(s + '.npz')
            print(f"  Saved {s}.npz")
    else:
        print("=== Diurnal starch sweep (loading saved results) ===")
        labels = [f"diurnal_starch_kf{kf:.1f}_kr{kr:.1f}".replace('.', 'p')
                  for kf, kr in KF_KR]
        paths  = [os.path.join(_OUT_DIR, f"needle_{lbl}.npz") for lbl in labels]
        results = load_results(paths)
        print(f"  Loaded {len(results)} result(s).")

    # ── Post-processing ───────────────────────────────────────────────────────
    stem = os.path.join(_OUT_DIR, 'needle_diurnal_starch')

    starch_comparison_plot(results, stem + '_comparison.png')
    print(f"Saved: {stem}_comparison.png")

    for res in results:
        lbl = res.params['label']
        s   = os.path.join(_OUT_DIR, f"needle_{lbl}")
        diurnal_flux_plot(res,              s + '_fluxes.png')
        concentration_gif(res,              s + '_sucrose.gif', max_frames=60, fps=8)
        if res.concentration_starch is not None:
            starch_concentration_gif(res,   s + '_starch.gif',  max_frames=60, fps=8)
        print(f"  [{lbl}] plots + GIFs done")

    print("Done.")
