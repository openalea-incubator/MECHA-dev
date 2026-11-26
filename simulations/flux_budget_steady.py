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
from dataclasses import replace
from itertools import product

from needle import (_OUT_DIR, DIFF_SUCROSE, Anatomy, Experiment, JFNK, NeumannBC,
                    NoCoupling, Picard, SimResult, Solute, TimeStepping, SteadyState, run_experiments)
from post_sim import (_STEADY_METRIC_LABELS, anatomy_plot, flow_decomposition, flow_decomposition_plot,
                      steady_metric_vs_ae_plot)

#COUPLINGS = {'none': NoCoupling(), 'picard': Picard(), 'jfnk': JFNK()}

D_PD_SUCROSE = 0.6 * DIFF_SUCROSE
_OUT_DIR = os.path.join(_OUT_DIR, 'steady_state', 'nig')


EXPERIMENT = Experiment(
    label='flux_budget_steady',
    anatomy=Anatomy(seed=42),
    solute=Solute(d_pd=D_PD_SUCROSE, d_apo=1e-20, d_mem=1e-20, sigma=1.0, sigma_xylem=1.0),
    bc=NeumannBC(
        psi_xyl=-200.0,
        transpiration_E=1.7,       # mmol m⁻² s⁻¹
        assimilation_A=3.6,        # µmol m⁻² s⁻¹
        ref_area=None,             # m²; None → outer perimeter × slice height
        sinks=('strasburger', 'phloem'),
    ),
    dynamics=SteadyState(),
    coupling=JFNK(max_iter=25),
)

# (E, A) in (mmol m⁻² s⁻¹, µmol m⁻² s⁻¹)
VALUES_n17 = ((0.3475854, 2.195323), (0.4061078, 5.528739), (0.4000667, 6.861486), (0.4019172, 9.696612), (0.3962713, 11.92927))
VALUES_s49 = ((1.71, 6.24), (1.66, 12.90), (1.62, 15.37), (1.57, 19.14), (1.51, 20.64))
VALUES_s38 = ((0.8965178, 4.055882), (0.8325153, 8.113755), (0.7711562,	9.495983), (0.7405828, 12.24407), (0.700708, 13.64662))
VALUES_n6 = ((0.21, 1.26), (0.20, 2.72), (0.18, 3.15), (0.17, 4.14), (0.16, 4.86))

VALUES_NIG = (*VALUES_n6, *VALUES_n17)
VALUES_SYL = (*VALUES_s38, *VALUES_s49)

VALUES_ALL = (*VALUES_NIG, *VALUES_SYL)

EXPS = [
    EXPERIMENT.replace(
        label=f'flux_budget_steady_E{E:.1f}_A{A:.1f}',
        bc=replace(EXPERIMENT.bc, transpiration_E=E, assimilation_A=A),
    )
    for E, A in VALUES_NIG
]

if __name__ == '__main__':
    RUN_SIMULATION = True    # False: skip the run, re-plot from the stored .npz
    DECOMPOSE_FLOW = False    # split water flux / Péclet into transpiration and osmotic parts
    PLOT_ANATOMY = False
    PLOT_FLOWS = False        # per-link network plots (total + decomposed if available)
    FIG_FMT = 'pdf'


    if RUN_SIMULATION:
        results = run_experiments(EXPS, max_workers=None)
        for res in results:
            stem = os.path.join(_OUT_DIR, f"needle_{res.params['label']}")
            res.save(stem + '.npz')
    else:
        results = []
        stem = []
        for experiment in EXPS:        
            stem = os.path.join(_OUT_DIR, f"needle_{experiment.__getattribute__('label')}")

            if not os.path.exists(stem + '.npz'):
                raise FileNotFoundError(f"{stem}.npz not found; run with RUN_SIMULATION = True first.")
            res = SimResult.load(stem + '.npz')
            print(f"Loaded: {stem}.npz")
            results.append(res)
    
    for res in results:
        e = res.extras
        print(res.summary())
        stem = os.path.join(_OUT_DIR, f"needle_{res.params['label']}")

        if DECOMPOSE_FLOW and 'edge_evapo' not in e:
            flow_decomposition(res)
            res.save(stem + '.npz')
        if 'edge_evapo' in e:
            print(f"  Péclet split: transpiration {float(e['peclet_evapo']):+.4f}  "
                f"sucrose-osmotic {float(e['peclet_osmo']):+.4f}  "
                f"background {float(e['peclet_base']):+.4f}  "
                f"(total {float(e['mean_peclet']):+.4f}, closure {float(e['closure_residual']):.1e})")
            print(f"Saved: {flow_decomposition_plot(res, f'{stem}_peclet_split.{FIG_FMT}')}")

        if PLOT_ANATOMY:
            print(f"Saved: {anatomy_plot(res, f'{stem}_anatomy.{FIG_FMT}')}")
        if PLOT_FLOWS:
            quantities = [('solute', 'sucrose_flow'), ('water', 'water_flow'), ('peclet', 'peclet')]
            if 'edge_evapo' in e:
                quantities += [('water_evapo', 'water_flow_evapo'), ('water_osmo', 'water_flow_osmo'),
                            ('peclet_evapo', 'peclet_evapo'), ('peclet_osmo', 'peclet_osmo')]
            for quantity, tag in quantities:
                print(f"Saved: {res.plot_flow_network(f'{stem}_{tag}.{FIG_FMT}', quantity=quantity, tol_decades=2.0)}")

    for key in _STEADY_METRIC_LABELS:

        print(f"Saved: {steady_metric_vs_ae_plot(results, os.path.join(_OUT_DIR, f'{key}_vs_AE.{FIG_FMT}'), metric=key)}")