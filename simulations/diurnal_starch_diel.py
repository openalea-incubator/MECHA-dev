"""
diurnal_starch_diel.py
======================
Diurnal needle transport with a **time-dependent** sucrose ↔ starch conversion,
in phase with transpiration and photosynthesis (``needle.s_diel``):

  * day   → sucrose → starch    (k_f(t) = k_f_day · max(0, s))
  * night → starch → sucrose    (k_r(t) = k_r_night · max(0, −s))

Runs for many diel cycles so the system settles into a repeating cycle.

Outputs (``simulations/outputs``): _anatomy.png, _diel.png, _fluxes.png,
_sucrose.gif, _starch.gif.
"""

import os

import numpy as np

from needle import (_OUT_DIR, D_PD_SUCROSE, Anatomy, DielStarch, DirichletBC,
                    Diurnal, Experiment, JFNK, Solute, TimeStepping, run_experiments)
from post_sim import (anatomy_plot, diel_starch_plot, diurnal_flux_plot,
                      concentration_gif, starch_concentration_gif, load_results)

N_DAYS = 30
DT = 1.0 / 24.0

EXPERIMENT = Experiment(
    label='diurnal_starch_diel',
    anatomy=Anatomy(seed=42),
    solute=Solute(d_pd=D_PD_SUCROSE),
    bc=DirichletBC(
        psi_xyl=-200.0, air='psi', psi_air=-1.0e3, c_photo=5.0e-6,
        sinks=('phloem_loading', 'xylem'),
        diurnal=Diurnal(period=1.0, f_baseline=0.1),
    ),
    dynamics=TimeStepping(dt=DT, max_steps=int(round(N_DAYS / DT))),
    coupling=JFNK(),
    starch=DielStarch(k_f_day=6.0, k_r_night=6.0, period=1.0),
)


if __name__ == '__main__':
    RUN_SIM = True    # False: load the last saved result
    stem = os.path.join(_OUT_DIR, f"needle_{EXPERIMENT.label}")

    if RUN_SIM:
        print(f"=== Diel starch run: {N_DAYS} days ===")
        (res,) = run_experiments([EXPERIMENT], max_workers=1)
        res.save(stem + '.npz')
    else:
        (res,) = load_results([stem + '.npz'])
    print(f"  {stem}.npz")

    m, j_s = res.m_starch, res.j_s
    print(f"  j_s_max = {np.max(j_s) * 1e12:.1f} pmol/d"
          f"   m_starch: min={np.min(m) * 1e6:.2f} µM  max={np.max(m) * 1e6:.2f} µM")

    # Limit-cycle check: daily starch peak of the last vs previous day.
    spd = int(round(1.0 / DT))
    if len(m) > 2 * spd:
        last, prev = float(np.max(m[-spd:])), float(np.max(m[-2 * spd:-spd]))
        drift = abs(last - prev) / (abs(prev) + 1e-30)
        print(f"  last-day peak starch drift: {drift * 100:.2f}%"
              f"  → {'steady (limit cycle)' if drift < 0.02 else 'still drifting'}")

    anatomy_plot(res, stem + '_anatomy.png')
    diel_starch_plot(res, stem + '_diel.png')
    diurnal_flux_plot(res, stem + '_fluxes.png')
    concentration_gif(res, stem + '_sucrose.gif', max_frames=80, fps=10)
    starch_concentration_gif(res, stem + '_starch.gif', max_frames=80, fps=10)
    print("Done.")
