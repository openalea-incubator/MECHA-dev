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

from needle import (_OUT_DIR, DIFF_SUCROSE, Anatomy, Experiment, JFNK, NeumannBC,
                    NoCoupling, Picard, SimResult, Solute, TimeStepping, run_experiments)
from post_sim import concentration_gif, flow_network_gif, anatomy_plot, budget_plot

#COUPLINGS = {'none': NoCoupling(), 'picard': Picard(), 'jfnk': JFNK()}
D_PD_SUCROSE = 0.6 * DIFF_SUCROSE



EXPERIMENT = Experiment(
    label='flux_budget_xyopen',
    anatomy=Anatomy(seed=42),
    solute=Solute(d_pd=D_PD_SUCROSE, d_apo=1e-20, d_mem=1e-20, sigma=1.0, sigma_xylem=0.0),
    bc=NeumannBC(
        psi_xyl=-200.0,
        transpiration_E=0.5,       # mmol m⁻² s⁻¹
        assimilation_A=5.3,        # µmol m⁻² s⁻¹
        ref_area=None,             # m²; None → outer perimeter × slice height
        sinks=('strasburger', 'phloem'),
    ),
    dynamics=TimeStepping(
        dt=2.0 / 24.0, max_steps=24 * 20,
        balance_tol=1.0e-3, balance_window=2.0,
    ),
    coupling=Picard(max_iter=20),
)

if __name__ == '__main__':
    RUN_SIMULATION = True    # False: skip the march, re-plot from the stored .npz
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
    print(f"  production     = {e['q_prod'][-1]*1e9:.3f} nmol/d")
    print(f"  export         = {e['j_s'][-1]*1e9:.3f} nmol/d "
          f"(Strasburger {e['j_strasburger'][-1]*1e9:.3f}, sieve {e['j_phloem'][-1]*1e9:.3f})")
    print(f"  mean xylem sucrose = {e['c_xylem'][-1]*1e6:.3f} mM (not a sink)")
    print(f"  ratio inst/cum = {e['ratio_inst'][-1]:.4f} / {e['ratio_cum'][-1]:.4f}")
    print(f"  max|mass balance error| = {np.max(np.abs(e['mass_balance']))*1e12:.3e} pmol")
    if 'n_couple' in e:
        print(f"  coupling iters/step = {np.mean(e['n_couple'][1:]):.1f}, "
              f"unconverged steps = {int(np.sum(~e['couple_ok'].astype(bool)))}")

    FIG_FMT = 'pdf'          # static figures as vectors; 'png' for raster
    TOL_DECADES = 2.0        # hide links below max·10^-TOL_DECADES

    print(f"Saved: {budget_plot(res, f'{stem}_budget.{FIG_FMT}')}")
    print(f"Saved: {concentration_gif(res, stem + '_sucrose.gif', max_frames=120, fps=10)}")
    print(f"Saved: {anatomy_plot(res, f'{stem}_anatomy.{FIG_FMT}')}")
    for quantity, tag in (('solute', 'sucrose_flow'), ('water', 'water_flow'), ('peclet', 'peclet')):
        print(f"Saved: {res.plot_flow_network(f'{stem}_{tag}.{FIG_FMT}', quantity=quantity, tol_decades=TOL_DECADES)}")
        print(f"Saved: {flow_network_gif(res, f'{stem}_{tag}.gif', quantity=quantity, tol_decades=TOL_DECADES)}")
