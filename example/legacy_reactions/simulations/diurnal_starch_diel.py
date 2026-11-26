"""
diurnal_starch_diel.py
======================
Diurnal needle transport with a **time-dependent** sucrose ↔ starch conversion.

A single run lets the conversion follow the day/night rhythm 
in phase with the transpirational drive and the photosynthetic sucrose
source:

  * daytime   → forward-dominated  sucrose → starch  (``k_f(t)`` large, ``k_r``≈0)
  * night     → backward-dominated starch → sucrose  (``k_r(t)`` large, ``k_f``≈0)

The rates are driven by ``needle.s_diel(t)`` (the ``NeedleDiurnalDielStarchSimulation``
rebuilds ``self.mst.K`` every step).  The simulation runs for many diel cycles so
the system settles into a repeating (limit-cycle) steady state.

Outputs (in ``simulations/outputs``):
  * ``…_anatomy.png``    GRANAR-generated needle anatomy, coloured by tissue
  * ``…_diel.png``       k_f/k_r, j_w, j_s, mean starch, j_s/j_w time series
  * ``…_fluxes.png``     the standard diurnal flux plot (auto starch panel)
  * ``…_sucrose.gif``    transient sucrose concentration field
  * ``…_starch.gif``     transient starch concentration field
"""

import os

import numpy as np

from needle import _OUT_DIR, DielStarchParams, run_experiments
from post_sim import (anatomy_plot, diel_starch_plot, diurnal_flux_plot,
                      concentration_gif, starch_concentration_gif, load_results)


if __name__ == '__main__':
    SEED = 42

    # Run long enough to reach a repeating diel steady state (limit cycle): the
    # interior sucrose pool grows until its gradient to the phloem/xylem sink
    # makes daily export balance daily photosynthetic input.  dt = 1/24 d
    # (hourly), 30 full days → 720 steps.
    N_DAYS    = 30
    DT        = 1.0 / 24.0
    MAX_STEPS = int(round(N_DAYS / DT))

    # Set RUN_SIM=True to (re)run; False to load the last saved result.
    RUN_SIM = True

    params = DielStarchParams(
        seed=SEED,
        dt=DT,
        max_steps=MAX_STEPS,
        k_f_day=6.0,        # 1/d peak daytime synthesis  (sucrose → starch)
        k_r_night=6.0,      # 1/d peak night remobilisation (starch → sucrose)
        label='diurnal_starch_diel',
    )

    stem = os.path.join(_OUT_DIR, f"needle_{params.label}")

    if RUN_SIM:
        print(f"=== Diel starch run: {N_DAYS} days, {MAX_STEPS} steps ===")
        (res,) = run_experiments([params], max_workers=1)
        res.save(stem + '.npz')
        print(f"  Saved {stem}.npz")
    else:
        print("=== Diel starch run (loading saved result) ===")
        (res,) = load_results([stem + '.npz'])
        print(f"  Loaded {stem}.npz")

    m = res.get_extra('m_starch')
    j_s = res.get_extra('j_s')
    print(f"  j_s_max = {np.max(j_s) * 1e12:.1f} pmol/d"
          f"   m_starch: min={np.min(m) * 1e6:.2f} µM"
          f"  max={np.max(m) * 1e6:.2f} µM")

    # Steady-state (limit-cycle) check: compare the daily peak starch of the
    # last two full days; a small relative change means the cycle has settled.
    steps_per_day = int(round(1.0 / DT))
    if len(m) > 2 * steps_per_day:
        last  = float(np.max(m[-steps_per_day:]))
        prev  = float(np.max(m[-2 * steps_per_day:-steps_per_day]))
        drift = abs(last - prev) / (abs(prev) + 1e-30)
        print(f"  last-day peak starch drift vs previous day: {drift * 100:.2f}%"
              f"  → {'steady (limit cycle)' if drift < 0.02 else 'still drifting'}")

    # ── Post-processing ───────────────────────────────────────────────────────
    anatomy_plot(res,                stem + '_anatomy.png')
    print(f"  Saved {stem}_anatomy.png")
    diel_starch_plot(res,            stem + '_diel.png')
    print(f"  Saved {stem}_diel.png")
    diurnal_flux_plot(res,           stem + '_fluxes.png')
    print(f"  Saved {stem}_fluxes.png")
    concentration_gif(res,           stem + '_sucrose.gif', max_frames=80, fps=10)
    print(f"  Saved {stem}_sucrose.gif")
    starch_concentration_gif(res,    stem + '_starch.gif',  max_frames=80, fps=10)
    print(f"  Saved {stem}_starch.gif")

    print("Done.")
