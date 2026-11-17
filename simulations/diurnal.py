"""
diurnal.py
==========
Diurnal (day/night) needle scenario with Dirichlet drivers.

    Ψ_xyl = −200 hPa
    Ψ_air = −1e3 · f(t) hPa              (peaks at noon, night floor 0.1)
    mesophyll += 5e-6 · g(t) mol/cm³     per step (photosynthesis, daytime only)

Measured outputs: j_w, j_s, j_s/j_w — see post_sim.diurnal_flux_plot.
"""

import os

from needle import (_OUT_DIR, D_PD_SUCROSE, Anatomy, DirichletBC, Diurnal,
                    Experiment, JFNK, NeedleSimulation, Solute, TimeStepping)
from post_sim import diurnal_flux_plot

EXPERIMENT = Experiment(
    label='diurnal',
    anatomy=Anatomy(seed=42),
    solute=Solute(d_pd=D_PD_SUCROSE),
    bc=DirichletBC(
        psi_xyl=-200.0, air='psi', psi_air=-1.0e3, c_photo=5.0e-6,
        sinks=('phloem_loading', 'xylem'),
        diurnal=Diurnal(period=1.0, f_baseline=0.1),
    ),
    dynamics=TimeStepping(dt=1.0 / 24.0, max_steps=48),
    coupling=JFNK(),               # NoCoupling() | Picard() | JFNK()
    starch=None,
)


if __name__ == '__main__':
    res = NeedleSimulation(EXPERIMENT, verbose=True).run()
    print(res.summary())
    stem = os.path.join(_OUT_DIR, f"needle_{EXPERIMENT.label}")
    res.save(stem + '.npz')
    diurnal_flux_plot(res, stem + '_fluxes.png')
    print(f"Saved: {stem}_fluxes.png")
