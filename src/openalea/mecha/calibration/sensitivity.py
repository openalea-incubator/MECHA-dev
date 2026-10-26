"""Study the sensitivity of the theta transport coefficients under normal
(well-watered) conditions.

This script reuses the calibration pipeline (:mod:`optimizer`) to quantify how
strongly each unknown transport coefficient ``theta`` in
:data:`~openalea.mecha.calibration.forward_model.PARAM_KEYS`
``("kw", "kmb", "kaqp", "kpl")`` influences the modelled radial conductance
``k_r`` under a single, representative "normal" boundary condition.

    NeedleAnatomy  ->  NetworkBuilder  ->  ForwardModel
                                              |
        Measurement[] --+--> CostFunction --> SensitivityAnalyzer
                        |                          |
                        |                   local()  -> Jacobian / identifiability
                        |                   morris() -> global screening
                        (normal conditions: well-watered, high RH)

Two complementary analyses are run:

* **Local** — a finite-difference Jacobian of ``k_r`` with respect to the
  log10 coefficients, evaluated at the needle root defaults. It gives a
  per-parameter sensitivity ranking plus identifiability diagnostics (Fisher
  information ``J^T J`` eigen-spectrum, condition number and Brun collinearity
  index).
* **Morris** — a derivative-free global screening over the full parameter
  bounds, ranking parameters by their global influence (``mu_star``) and
  flagging non-linearity / interaction (``sigma``).
"""

from __future__ import annotations

# GRANAP builds the needle anatomy; MECHA consumes it.
from openalea.granap.needle_class import NeedleAnatomy

from openalea.mecha import NetworkBuilder
from openalea.mecha.calibration import (
    PARAM_KEYS,
    CostFunction,
    ForwardModel,
    Measurement,
    ParamSpace,
    SensitivityAnalyzer,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Normal boundary condition
# ══════════════════════════════════════════════════════════════════════════════
# A single representative "normal conditions" operating point. The observed
# value is irrelevant for a sensitivity study (only the *model* response to
# perturbing theta matters), but a Measurement is required to drive the forward
# model, so a nominal well-watered flux is used.
#   psi_xyl      xylem water potential under normal conditions   [hPa]
#   rh_airspace  substomatal air-space relative humidity, (0, 1] [-]
#   E_obs        nominal transpiration flux density              [mmol m^-2 s^-1]
MEASUREMENTS = [
    Measurement(psi_xyl=-4000.0, rh_airspace=0.90, E_obs=6.7, label="well-watered"),
]


# ══════════════════════════════════════════════════════════════════════════════
# 2. Which coefficients to study
# ══════════════════════════════════════════════════════════════════════════════
# Study the sensitivity of ALL theta transport coefficients.
FIT_NAMES = list(PARAM_KEYS)  # ("kw", "kmb", "kaqp", "kpl")

# Optional physical (linear) bound overrides; empty -> ParamSpace DEFAULT_BOUNDS.
BOUNDS: dict = {}


# ══════════════════════════════════════════════════════════════════════════════
# 3. Model / analysis settings
# ══════════════════════════════════════════════════════════════════════════════
TEMPERATURE_K = 298.15   # room temperature for p_sat / Kelvin conversion
I_MATURITY = 0           # maturity-stage index to solve
RELATIVE_RESIDUALS = True  # (unused for sensitivity, kept for pipeline parity)

# Local Jacobian: absolute perturbation of each log10 parameter (~2.3 % change
# in the physical coefficient) — large enough to see through MECHA's numerical
# noise.
LOCAL_DIFF_STEP = 1.0e-2

# Morris global screening.
MORRIS_TRAJECTORIES = 12
MORRIS_LEVELS = 4
MORRIS_SEED = 0


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
    space = ParamSpace(names=FIT_NAMES, bounds=BOUNDS)
    cost = CostFunction(fm, space, MEASUREMENTS, relative=RELATIVE_RESIDUALS)
    analyzer = SensitivityAnalyzer(cost)

    print(
        f"Studying sensitivity of {space.names} at "
        f"{len(MEASUREMENTS)} normal-condition point(s)..."
    )

    # --- local (Jacobian-based) sensitivity ---------------------------------
    # Evaluated at the geometric midpoint of the bounds (the default operating
    # point when no theta is supplied).
    local = analyzer.local(diff_step=LOCAL_DIFF_STEP)

    print("\n=== Local sensitivity (per decade of each coefficient) ===")
    print("Evaluation point theta:")
    for name in local.names:
        print(f"  {name:5s} = {local.theta[name]:.4e}")

    print("\nSensitivity ranking  (||d k_r / d log10(theta)||):")
    for name, s in local.ranking:
        print(f"  {name:5s} : {s:.4e}")

    print("\nIdentifiability diagnostics:")
    print(f"  Fisher eigenvalues     = "
          f"{', '.join(f'{v:.3e}' for v in local.eigvals)}")
    print(f"  condition number       = {local.condition_number:.4e}")
    print(f"  collinearity index     = {local.collinearity_index:.4e}")
    if local.condition_number > 1.0e6:
        print("  -> parameters are jointly hard to identify from this point.")
    if local.collinearity_index > 10.0:
        print("  -> strong parameter interaction (collinearity) detected.")

    # --- global (Morris) screening ------------------------------------------
    morris = analyzer.morris(
        n_trajectories=MORRIS_TRAJECTORIES,
        n_levels=MORRIS_LEVELS,
        seed=MORRIS_SEED,
    )

    print("\n=== Morris global screening ===")
    print(f"  {'param':<6} {'mu_star':>12} {'mu':>12} {'sigma':>12}")
    idx = {n: i for i, n in enumerate(morris.names)}
    for name, _ in morris.ranking:
        i = idx[name]
        print(f"  {name:<6} {morris.mu_star[i]:12.4e}"
              f" {morris.mu[i]:12.4e} {morris.sigma[i]:12.4e}")
    print("\n  mu_star : overall global influence (larger = more influential)")
    print("  sigma   : non-linearity / interaction (larger = more interaction)")

    print(f"\nforward solves = {cost.n_evals}")


if __name__ == "__main__":
    main()
