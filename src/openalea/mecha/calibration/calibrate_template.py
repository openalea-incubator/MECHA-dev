"""Template: calibrate needle transport coefficients against transpiration data.

Copy this file, replace the values in the ``# >>> DATA PLACEHOLDER`` blocks with
your own experimental numbers, and run it::

    python calibrate_template.py

The script wires up the existing calibration pipeline:

    NeedleAnatomy  ->  NetworkBuilder  ->  ForwardModel
                                              |
        Measurement[] --+--> CostFunction --> Optimizer --> theta*
                        |
                        (E measured in mmol H2O m^-2 s^-1)

Nothing here needs editing except the placeholder blocks.

Calibration target: radial conductance k_r
"""

from __future__ import annotations

# GRANAP builds the needle anatomy; MECHA consumes it.
from openalea.granap.needle_class import NeedleAnatomy

from openalea.mecha import NetworkBuilder
from openalea.mecha.calibration import (
    CostFunction,
    ForwardModel,
    Measurement,
    Optimizer,
    ParamSpace,
    FlatnessConstraint,
)

# ══════════════════════════════════════════════════════════════════════════════
# 1. Experimental data
# ══════════════════════════════════════════════════════════════════════════════
# >>> DATA PLACEHOLDER — measured transpiration points ──────────────────────────
# One entry per measurement.  Each needs:
#   psi_xyl      xylem water potential during the measurement          [hPa]
#   rh_airspace  substomatal air-space relative humidity, in (0, 1]    [-]
#   provide either E_obs or kr_obs
#   E_obs        measured transpiration flux density                   [mmol m^-2 s^-1]
#   kr_obs       measured radial conductance (if available)            [cm hPa^-1 d^-1] 
#   weight       optional; use 1/sigma to down-weight noisy points     [-]
#   label        optional free-form tag (e.g. drought stage)
#
# Replace the illustrative numbers below with your own data.
MEASUREMENTS = [
    Measurement(psi_xyl=-200, rh_airspace=0.98, E_obs=6.7, label="well-watered"),
    #Measurement(psi_xyl=-200.0, rh_airspace=0.999, kr_obs=7.7e-4, label="well-watered"),
]

CONSTRAINTS = [
    FlatnessConstraint(rh_airspace=0.98, psi_range=(-1000.0, -200.0),
                       n_anchors=3, weight=1.0, relative=True, label="flat kr"),
]

# ══════════════════════════════════════════════════════════════════════════════
# 2. Which coefficients to calibrate
# ══════════════════════════════════════════════════════════════════════════════
# >>> DATA PLACEHOLDER — parameters to fit ──────────────────────────────────────
# Any subset of PARAM_KEYS = ("kw", "kmb", "kaqp", "kpl").
# Fewer parameters => a better-posed problem; add more only if your data can
# constrain them.
FIT_NAMES = ["kw", "kmb"]

# >>> DATA PLACEHOLDER (optional) — override the default search bounds ───────────
# Physical (linear) bounds per parameter. Leave a name out to keep ParamSpace's
# DEFAULT_BOUNDS.  Units: kw/kmb/kaqp [cm hPa^-1 d^-1], kpl [cm^3 hPa^-1 d^-1].
BOUNDS = {
    # "kaqp": (1.0e-6, 1.0e-2),
    # "kmb":  (1.0e-7, 1.0e-3),
}

# >>> DATA PLACEHOLDER (optional) — coefficients held FIXED at known values ──────
# Merged into every evaluated theta; not optimized.
FIXED = {
    # "kw":  2.4e-4,
    # "kpl": 5.3e-12,
    # "kwa": 1.0e-4,
}


# ══════════════════════════════════════════════════════════════════════════════
# 3. Optimizer settings
# ══════════════════════════════════════════════════════════════════════════════
# >>> DATA PLACEHOLDER (optional) — solver knobs ────────────────────────────────
TEMPERATURE_K = 298.15   # room temperature for p_sat / Kelvin conversion
I_MATURITY = 0           # maturity-stage index to solve
RELATIVE_RESIDUALS = True  # normalize residuals by k_r_obs (balances magnitudes)
RANDOM_SEED = 0          # reproducible global search
GLOBAL_MAXITER = 60      # differential_evolution iterations (bigger = slower)
GLOBAL_POPSIZE = 12      # population multiplier
REFINE_LOCALLY = True    # follow the global search with a least-squares polish
WORKERS = 12             # -1 = all CPUs
# NeedleAnatomy is STOCHASTIC: each build draws a new random geometry. A fixed
# seed is REQUIRED so every worker process rebuilds the *identical* anatomy;
# otherwise each worker would calibrate against a different needle and the
# objective becomes inconsistent across processes. Set to None only for a
# single-process, single-anatomy run.
ANATOMY_SEED = 42


def build_forward_model() -> ForwardModel:
    """Build the needle forward model for a fixed (reproducible) anatomy.

    Must be a top-level, picklable callable with no per-call randomness so each
    worker process reconstructs the same model (see ``ANATOMY_SEED``).
    """
    anatomy = NeedleAnatomy(seed=ANATOMY_SEED)
    anatomy.export_to_adjencymatrix()
    network = NetworkBuilder(anatomy)
    network.populate_from_network()
    return ForwardModel(network, T=TEMPERATURE_K, i_maturity=I_MATURITY)


def main() -> None:
    # --- forward model + inverse problem ------------------------------------
    fm = build_forward_model()
    space = ParamSpace(names=FIT_NAMES, bounds=BOUNDS, fixed=FIXED)
    # `forward_model_factory` (a module-level, picklable callable) lets each
    # worker PROCESS rebuild its own model: the MECHA solve is GIL-bound and the
    # network is mutated per solve, so process parallelism is required (threads
    # neither speed it up nor are thread-safe).
    cost = CostFunction(
        fm, space, MEASUREMENTS, relative=True, constraints=CONSTRAINTS,
        forward_model_factory=build_forward_model,
    )
    optimizer = Optimizer(cost, workers=WORKERS)

    # --- run global search then local refinement ----------------------------
    print(f"Calibrating {space.names} against {len(MEASUREMENTS)} measurement(s)...")
    result = optimizer.run(
        seed=RANDOM_SEED,
        global_maxiter=GLOBAL_MAXITER,
        global_popsize=GLOBAL_POPSIZE,
        refine=REFINE_LOCALLY,
    )

    # --- report --------------------------------------------------------------
    print("\n=== Calibrated transport coefficients ===")
    for name in space.names:
        print(f"  {name:5s} = {result.theta[name]:.4e}")
    if space.fixed:
        print("  (fixed:", {k: f"{v:.3e}" for k, v in space.fixed.items()}, ")")
    print(f"\ncost            = {result.cost:.4e}")
    print(f"forward solves  = {result.n_evals}")
    print(f"success         = {result.success}")
    print(f"message         = {result.message}")

    # --- measured vs modelled at the optimum ---------------------------------
    print("\n=== Fit quality (k_r, cm hPa^-1 d^-1; E_obs, mmol m^-2 s^-1; E_model, mmol m^-2 s^-1 hPa^-1) ===")
    kr_model = cost.predict_kr(result.theta)
    kr_obs = cost.observed_kr()
    E_obs = cost.observed_E_obs()
    E_model = cost.predict_E(result.theta)

    print(f"  {'label':<14} {'psi_xyl':>9} {'RH':>6} {'ΔΨ':>4}"
          f" {'kr_obs':>11} {'kr_model':>11} {'E_obs':>11} {'E_model':>11}")
    for m, ko, km, eo, em in zip(MEASUREMENTS, kr_obs, kr_model, E_obs, E_model):
        print(f"  {m.label:<14} {m.psi_xyl:9.1f} {m.rh_airspace:6.2f} {cost._dpsi(m):2.4e}"
              f"{ko:11.4e} {km:11.4e} {eo:11.4e} {em:11.4e}")


if __name__ == "__main__":
    main()
