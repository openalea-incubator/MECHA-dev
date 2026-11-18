"""
flux_budget.py
==============
Constant-flux needle scenario (Neumann BCs, no diurnal variation).

    Ψ_xyl = −200 hPa                    (Dirichlet, sets the potential level)
    E     = const  mmol m⁻² s⁻¹         outward through the air spaces
    A     = const  µmol m⁻² s⁻¹         → sucrose, evenly over the mesophyll

Starting from a sucrose-free needle, the march runs until sucrose export into
the phloem loading zone (Strasburger + sieve cells) balances production on
average over ``balance_window``.  The xylem is NOT a sink (σ_xylem = 1).
Output: export / production over time (instantaneous and cumulative).
"""

import os

import numpy as np

from needle import (_OUT_DIR, D_PD_SUCROSE, D_APO_SUCROSE, Anatomy, Experiment, JFNK, NeumannBC,
                    NoCoupling, Picard, SimResult, Solute, TimeStepping, SteadyState, run_experiments)
from post_sim import anatomy_plot

#COUPLINGS = {'none': NoCoupling(), 'picard': Picard(), 'jfnk': JFNK()}

EXPERIMENT = Experiment(
    label='flux_budget_steady',
    anatomy=Anatomy(seed=42),
    solute=Solute(d_pd=D_PD_SUCROSE, d_apo=D_APO_SUCROSE, sigma_xylem=1.0),
    bc=NeumannBC(
        psi_xyl=-200.0,
        transpiration_E=1.7,       # mmol m⁻² s⁻¹
        assimilation_A=3.6,        # µmol m⁻² s⁻¹
        ref_area=None,             # m²; None → outer perimeter × slice height
        sinks=('strasburger', 'phloem'),
    ),
    dynamics=SteadyState(),
    coupling=JFNK(max_iter=20),
)

if __name__ == '__main__':
    RUN_SIMULATION = False    # False: skip the run, re-plot from the stored .npz
    stem = os.path.join(_OUT_DIR, f"needle_{EXPERIMENT.label}")
    if RUN_SIMULATION:
        (res,) = run_experiments([EXPERIMENT], max_workers=1)
        res.save(stem + '.npz')
    else:
        if not os.path.exists(stem + '.npz'):
            raise FileNotFoundError(f"{stem}.npz not found; run with RUN_SIMULATION = True first.")
        res = SimResult.load(stem + '.npz')
        print(f"Loaded: {stem}.npz")
    e = res.extras
    print(res.summary())

    #print(f"Saved: {anatomy_plot(res, f'{stem}_anatomy.pdf')}")
    #for quantity, tag in (('solute', 'sucrose_flow'), ('water', 'water_flow'), ('peclet', 'peclet')):
    #    print(f"Saved: {res.plot_flow_network(f'{stem}_{tag}.pdf', quantity=quantity, tol_decades=2.0)}")