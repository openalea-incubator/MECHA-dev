"""
misco_steady.py
===============
Steady constant-flux needle scenario (as ``flux_budget_steady.py``) on the
microscopy-derived anatomy ``S11 4h 40c air spaces_CellSet.xml`` instead of a
GRANAP draw.  Cells named 'air space' in the cellset are the evaporating
mesophyll air spaces (wall_air links).

    Ψ_xyl = −200 hPa                    (Dirichlet, sets the potential level)
    E     = const  mmol m⁻² s⁻¹         outward through the air spaces
    A     = const  µmol m⁻² s⁻¹         → sucrose, evenly over the mesophyll
"""

import os
from dataclasses import replace

from needle import (_OUT_DIR, _SCRIPT_DIR, DIFF_SUCROSE, Anatomy, Experiment, JFNK, NeumannBC,
                    Solute, SimResult, SteadyState, run_experiments)
from post_sim import (_STEADY_METRIC_LABELS, anatomy_plot, flow_decomposition, flow_decomposition_plot,
                      steady_metric_vs_ae_plot)
#from flux_budget_steady import VALUES_ALL

CELLSET = os.path.join(_SCRIPT_DIR, 'S11 4h 40c air spaces_CellSet.xml')

D_PD_SUCROSE = 0.6 * DIFF_SUCROSE
_OUT_DIR = os.path.join(_OUT_DIR, 'steady_state', 'misco')
os.makedirs(_OUT_DIR, exist_ok=True)


EXPERIMENT = Experiment(
    label='misco_steady',
    # Transfusion parenchyma/tracheids (17/18) → 4 as in GRANAP; MECHA gives 17/18 σ = 0,
    # which leaks sucrose into walls with no steady exit.
    anatomy=Anatomy(seed=None, cellset=CELLSET, im_scale=1.0, cgroup_remap=((17, 4), (18, 4))),
    solute=Solute(d_pd=D_PD_SUCROSE, d_apo=1e-20, d_mem=1e-20, sigma=1.0, sigma_xylem=1.0),
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

VALUES_s49 = ((1.71, 6.24), (1.66, 12.90), (1.62, 15.37), (1.57, 19.14), (1.51, 20.64))

EXPS = [
    EXPERIMENT.replace(
        label=f'misco_steady_E{E:.1f}_A{A:.1f}',
        bc=replace(EXPERIMENT.bc, transpiration_E=E, assimilation_A=A),
    )
    for E, A in VALUES_s49
]

if __name__ == '__main__':
    RUN_SIMULATION = True    # False: skip the run, re-plot from the stored .npz
    DECOMPOSE_FLOW = True    # split water flux / Péclet into transpiration and osmotic parts
    PLOT_ANATOMY = True
    PLOT_FLOWS = True        # per-link network plots (total + decomposed if available)
    FIG_FMT = 'pdf'

    if RUN_SIMULATION:
        results = run_experiments(EXPS, max_workers=None)
        for res in results:
            stem = os.path.join(_OUT_DIR, f"needle_{res.params['label']}")
            res.save(stem + '.npz')
    else:
        results = []
        for experiment in EXPS:
            stem = os.path.join(_OUT_DIR, f"needle_{experiment.label}")
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
