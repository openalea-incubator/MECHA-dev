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
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Experimental data
# ══════════════════════════════════════════════════════════════════════════════
# >>> DATA PLACEHOLDER — measured transpiration points ──────────────────────────
# One entry per measurement.  Each needs:
#   psi_xyl      xylem water potential during the measurement          [hPa]
#   rh_airspace  substomatal air-space relative humidity, in (0, 1]    [-]
#   E_obs        measured transpiration flux density                   [mmol m^-2 s^-1]
#   weight       optional; use 1/sigma to down-weight noisy points     [-]
#   label        optional free-form tag (e.g. drought stage)
#
# Replace the illustrative numbers below with your own data.
MEASUREMENTS = [
    Measurement(psi_xyl=-200.0, rh_airspace=0.99, E_obs=6.7, label="well-watered"),
    #Measurement(psi_xyl=-700.0, rh_airspace=0.99, E_obs=0.75, label="mild-drought"),
    #Measurement(psi_xyl=-1500.0, rh_airspace=0.99, E_obs=0.0038, label="drought"),
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


def build_forward_model() -> ForwardModel:
    """Build the (cached) needle forward model once."""
    anatomy = NeedleAnatomy()
    anatomy.export_to_adjencymatrix()
    network = NetworkBuilder(anatomy)
    network.populate_from_network()
    return ForwardModel(network, T=TEMPERATURE_K, i_maturity=I_MATURITY)


def main() -> None:
    # --- forward model + inverse problem ------------------------------------
    fm = build_forward_model()
    space = ParamSpace(names=FIT_NAMES, bounds=BOUNDS, fixed=FIXED)
    cost = CostFunction(fm, space, MEASUREMENTS, relative=RELATIVE_RESIDUALS)
    optimizer = Optimizer(cost)

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
    print("\n=== Fit quality (k_r, cm hPa^-1 d^-1) ===")
    kr_model = cost.predict(result.theta)
    kr_obs = cost.observed()
    print(f"  {'label':<14} {'psi_xyl':>9} {'RH':>6} {'kr_obs':>11} {'kr_model':>11}")
    for m, ko, km in zip(MEASUREMENTS, kr_obs, kr_model):
        print(f"  {m.label:<14} {m.psi_xyl:9.1f} {m.rh_airspace:6.2f} "
              f"{ko:11.4e} {km:11.4e}")


if __name__ == "__main__":
    main()
