"""
diurnal_starch.py
=================
Diurnal starch sweep: the ``diurnal.py`` set-up with a constant-rate
sucrose ↔ starch conversion, swept over (k_syn, k_deg).  (0, 0) runs without
starch as the reference.
"""

import os
from dataclasses import replace

import numpy as np

from needle import _OUT_DIR, Starch, run_experiments
from diurnal import EXPERIMENT as DIURNAL
from post_sim import (diurnal_flux_plot, starch_comparison_plot,
                      concentration_gif, starch_concentration_gif,
                      load_results)

MAX_STEPS = 72
KF_KR = [(0.0, 0.0), (1.0, 0.5), (2.0, 0.5),
         (2.0, 2.0), (5.0, 0.5), (2.0, 5.0)]


def label(kf: float, kr: float) -> str:
    return f"diurnal_starch_kf{kf:.1f}_kr{kr:.1f}".replace('.', 'p')


BASE = DIURNAL.replace(dynamics=replace(DIURNAL.dynamics, max_steps=MAX_STEPS))

GRID = [
    BASE.replace(label=label(kf, kr),
                 starch=Starch(k_syn=kf, k_deg=kr) if (kf or kr) else None)
    for kf, kr in KF_KR
]


if __name__ == '__main__':
    RUN_SIM = True    # False: load the last saved results

    if RUN_SIM:
        print(f"=== Diurnal starch sweep: {len(GRID)} experiments ===")
        results = run_experiments(GRID, max_workers=None)
        for res in results:
            s = os.path.join(_OUT_DIR, f"needle_{res.params['label']}")
            res.save(s + '.npz')
            m = res.get_extra('m_starch')
            print(f"  [{res.params['label']}]  "
                  f"j_s_max={np.max(res.j_s)*1e12:.1f} pmol/d  "
                  f"m_starch_max={(np.max(m)*1e6 if m is not None else 0.0):.2f} µM")
    else:
        results = load_results([os.path.join(_OUT_DIR, f"needle_{e.label}.npz") for e in GRID])
        print(f"  Loaded {len(results)} result(s).")

    stem = os.path.join(_OUT_DIR, 'needle_diurnal_starch')
    starch_comparison_plot(results, stem + '_comparison.png')
    print(f"Saved: {stem}_comparison.png")
    for res in results:
        s = os.path.join(_OUT_DIR, f"needle_{res.params['label']}")
        diurnal_flux_plot(res, s + '_fluxes.png')
        concentration_gif(res, s + '_sucrose.gif', max_frames=60, fps=8)
        if res.concentration_starch is not None:
            starch_concentration_gif(res, s + '_starch.gif', max_frames=60, fps=8)
    print("Done.")
