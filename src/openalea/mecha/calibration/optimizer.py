"""Parameter-space, cost function and optimiser wrapper for the needle
transpiration calibration.

This module turns the :class:`~openalea.mecha.calibration.forward_model.ForwardModel`
into an inverse problem: given a set of measured transpiration rates under
controlled boundary conditions ``(psi_xyl, RH)`` it searches for the unknown
transport coefficients ``theta`` (a subset of
:data:`~openalea.mecha.calibration.forward_model.PARAM_KEYS`) that best
reproduce the measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from openalea.mecha.calibration.forward_model import (
    PARAM_KEYS,
    ForwardModel,
    M_W,
    rh_to_water_potential,
)

# ── Water physical constants for the flux-unit conversion ─────────────────────
RHO_W_G_CM3 = 1.0
#: Seconds per day.
SECONDS_PER_DAY = 86_400.0
#: mmol per mol.
MMOL_PER_MOL = 1.0e3

#: Prefactor converting a volumetric flux [cm^3 d^-1] to a molar flux [mmol s^-1]
Q_TO_MMOL_PER_S: float = (RHO_W_G_CM3 / M_W) * MMOL_PER_MOL / SECONDS_PER_DAY


def q_to_flux_density(q_cm3_per_day: float, area_m2: float) -> float:
    """Convert a MECHA volumetric flux to a transpiration flux density.

    Parameters
    ----------
    q_cm3_per_day : float
        Volumetric water flux leaving the cross-section [cm^3 H2O d^-1], i.e.
        the output of :meth:`ForwardModel.transpiration_flux`.
    area_m2 : float
        Transpiring surface area [m^2] that the modelled cross-section
        represents (experimental normalization area of the measured ``E``).

    Returns
    -------
    float
        Molar transpiration flux density ``E`` [mmol H2O m^-2 s^-1].
    """
    if area_m2 <= 0.0:
        raise ValueError(f"area_m2 must be positive, got {area_m2}.")
    return Q_TO_MMOL_PER_S * q_cm3_per_day / area_m2


def flux_density_to_q(e_mmol_m2_s: float, area_m2: float) -> float:
    """Inverse of :func:`q_to_flux_density` (mmol m^-2 s^-1 → cm^3 d^-1).

    Provided for convenience / diagnostics.
    """
    if area_m2 <= 0.0:
        raise ValueError(f"area_m2 must be positive, got {area_m2}.")
    return e_mmol_m2_s * area_m2 / Q_TO_MMOL_PER_S


#: Prefactor converting a molar flux density [mmol H2O m^-2 s^-1] to a
#: volumetric flux density [cm^3 cm^-2 d^-1] = [cm d^-1]::
#:     v[cm/d] = E[mmol/m2/s] * (M_w/rho_w)[cm^3/mmol] / 1e4[cm2/m2] * 86400[s/d]
#: With M_w = 18.01528 g/mol => 18.01528e-3 cm^3/mmol (rho_w = 1 g/cm^3).
#: Value ≈ 1.5566e-1.
FLUX_DENSITY_TO_CM_PER_DAY: float = (
    (M_W * 1.0e-3 / RHO_W_G_CM3) / 1.0e4 * SECONDS_PER_DAY
)


def flux_density_to_kr(e_mmol_m2_s: float, dpsi_hpa: float) -> float:
    """Convert a measured transpiration flux density to a radial conductance.

    Turns an experimentally reported ``E`` [mmol H2O m^-2 s^-1] into the same
    radial-conductance quantity the forward model returns
    (:meth:`ForwardModel.transpiration_flux` with ``output="k_r"``)::

        k_r = (E converted to a volumetric flux density [cm d^-1]) / |dpsi|
            = FLUX_DENSITY_TO_CM_PER_DAY * E / |dpsi|        [cm hPa^-1 d^-1]

    Parameters
    ----------
    e_mmol_m2_s : float
        Measured transpiration flux density [mmol H2O m^-2 s^-1].
    dpsi_hpa : float
        Magnitude of the water-potential drop driving the flux [hPa], i.e.
        ``|psi_xyl - psi_air|`` for the measurement's boundary conditions.

    Returns
    -------
    float
        Radial conductance ``k_r`` [cm hPa^-1 d^-1].
    """
    dpsi = abs(float(dpsi_hpa))
    if dpsi == 0.0:
        raise ValueError("dpsi_hpa must be non-zero to form a conductance.")
    return FLUX_DENSITY_TO_CM_PER_DAY * e_mmol_m2_s / dpsi


# ── Measurements ──────────────────────────────────────────────────────────────
@dataclass
class Measurement:
    """A single measured transpiration data point.

    The calibration target is the **radial conductance** ``k_r``
    [cm hPa⁻¹ d⁻¹].

    You may specify the observed conductance either directly (``kr_obs``) or as
    a measured flux density (``E_obs``); in the latter case it is converted with
    :func:`flux_density_to_kr` using the water-potential drop of this
    measurement's boundary conditions.

    Parameters
    ----------
    psi_xyl : float
        Xylem water-potential boundary condition during the measurement [hPa].
    rh_airspace : float
        Substomatal air-space relative humidity during the measurement, in
        ``(0, 1]``.
    kr_obs : float, optional
        Measured radial conductance [cm hPa⁻¹ d⁻¹]. Provide this *or*
        ``E_obs``.
    E_obs : float, optional
        Measured transpiration flux density [mmol H2O m⁻² s⁻¹].  Converted to a
        conductance internally. Provide this *or* ``kr_obs``.
    weight : float, optional
        Relative weight of this point in the cost function (default 1.0).
        Use e.g. ``1 / sigma`` to down-weight noisy points.
    label : str, optional
        Free-form identifier.
    """

    psi_xyl: float
    rh_airspace: float
    kr_obs: Optional[float] = None
    E_obs: Optional[float] = None
    weight: float = 1.0
    label: str = ""

    def __post_init__(self) -> None:
        if (self.kr_obs is None) == (self.E_obs is None):
            raise ValueError(
                "Provide exactly one of `kr_obs` or `E_obs` for the measurement."
            )

    def observed_kr(self, dpsi_hpa: float) -> float:
        """Observed radial conductance [cm hPa⁻¹ d⁻¹] for this point.

        ``dpsi_hpa`` is the water-potential drop the forward model reports for
        this measurement's boundary conditions; it is only used when the target
        was given as a flux density ``E_obs``.
        """
        if self.kr_obs is not None:
            return float(self.kr_obs)
        return flux_density_to_kr(float(self.E_obs), dpsi_hpa)


# ── Parameter space ───────────────────────────────────────────────────────────
@dataclass
class ParamSpace:
    """Definition of the calibrated parameter subspace.

    The transport coefficients span many orders of magnitude
    (``kpl ~ 1e-12`` … ``kw ~ 1e-4``), so the optimizer works in **base-10 log
    space**.  A candidate vector ``x`` produced by an optimizer is therefore
    ``log10(theta)``; :meth:`to_theta` maps it back to the physical dictionary
    consumed by the forward model.

    Parameters
    ----------
    names : sequence of str
        Ordered subset of :data:`PARAM_KEYS` to calibrate.
    bounds : dict, optional
        Mapping ``name -> (lo, hi)`` giving the **physical** (linear) bounds of
        each parameter.  Missing entries fall back to :data:`DEFAULT_BOUNDS`.
    fixed : dict, optional
        Physical values for parameters held constant (not calibrated). These
        are merged into every ``theta`` returned by :meth:`to_theta`.
    """

    #: Physically plausible linear bounds (min, max) per coefficient, used when
    #: a bound is not supplied explicitly. These bracket the maize-root
    #: defaults by ~2 orders of magnitude each and should be widened/narrowed
    #: as needle-specific evidence accumulates.
    DEFAULT_BOUNDS: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "kw":   (1.0e-6, 1.0e-2),   # bulk wall conductivity  [cm hPa^-1 d^-1]
            "kmb":  (1.0e-7, 1.0e-3),   # background membrane     [cm hPa^-1 d^-1]
            "kaqp": (1.0e-6, 1.0e-2),   # aquaporin membrane      [cm hPa^-1 d^-1]
            "kpl":  (1.0e-14, 1.0e-9),  # plasmodesmata           [cm^3 hPa^-1 d^-1]
        },
        repr=False,
    )

    names: Sequence[str] = field(default_factory=lambda: list(PARAM_KEYS))
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    fixed: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = [n for n in self.names if n not in PARAM_KEYS]
        if unknown:
            raise ValueError(
                f"Unknown parameter name(s) {unknown}; allowed: {PARAM_KEYS}."
            )
        # Resolve effective bounds (explicit → default) and validate.
        resolved: Dict[str, Tuple[float, float]] = {}
        for name in self.names:
            if name not in self.bounds and name not in self.DEFAULT_BOUNDS:
                raise ValueError(
                    f"No bounds for '{name}': it has no entry in DEFAULT_BOUNDS, "
                    f"so an explicit (lo, hi) must be given in `bounds`."
                )
            lo, hi = self.bounds.get(name, self.DEFAULT_BOUNDS.get(name))
            if not (lo > 0.0 and hi > lo):
                raise ValueError(
                    f"Bounds for '{name}' must satisfy 0 < lo < hi, got ({lo}, {hi})."
                )
            resolved[name] = (float(lo), float(hi))
        self._eff_bounds = resolved

    # -- dimensions --
    @property
    def ndim(self) -> int:
        """Number of calibrated parameters."""
        return len(self.names)

    @property
    def log_bounds(self) -> List[Tuple[float, float]]:
        """Per-parameter ``(log10 lo, log10 hi)`` bounds in optimizer order."""
        return [
            (float(np.log10(lo)), float(np.log10(hi)))
            for lo, hi in (self._eff_bounds[n] for n in self.names)
        ]

    @property
    def linear_bounds(self) -> List[Tuple[float, float]]:
        """Per-parameter physical ``(lo, hi)`` bounds in optimizer order."""
        return [self._eff_bounds[n] for n in self.names]

    # -- mappings between the optimizer vector (log10) and theta --
    def to_theta(self, x_log: Sequence[float]) -> Dict[str, float]:
        """Map a log10 optimizer vector to a physical ``theta`` dict.

        The ``fixed`` parameters are always merged in so the returned dict is a
        complete specification for the forward model.
        """
        x_log = np.asarray(x_log, dtype=float)
        if x_log.shape != (self.ndim,):
            raise ValueError(
                f"x_log has shape {x_log.shape}, expected ({self.ndim},)."
            )
        theta: Dict[str, float] = dict(self.fixed)
        for name, xl in zip(self.names, x_log):
            theta[name] = float(10.0 ** xl)
        return theta

    def to_log_vector(self, theta: Dict[str, float]) -> np.ndarray:
        """Map a physical ``theta`` dict to the log10 optimizer vector."""
        return np.array(
            [np.log10(float(theta[name])) for name in self.names], dtype=float
        )

    def clip_log(self, x_log: Sequence[float]) -> np.ndarray:
        """Clip a log10 vector to the (log) bounds."""
        lo = np.array([b[0] for b in self.log_bounds])
        hi = np.array([b[1] for b in self.log_bounds])
        return np.clip(np.asarray(x_log, dtype=float), lo, hi)

    def initial_log_vector(
        self, theta0: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """Return a starting log10 vector.

        If ``theta0`` is given it is used (clipped to bounds); otherwise the
        geometric midpoint of each bound is returned.
        """
        if theta0 is not None:
            return self.clip_log(self.to_log_vector(theta0))
        return np.array(
            [0.5 * (lo + hi) for lo, hi in self.log_bounds], dtype=float
        )


# ── Cost function ─────────────────────────────────────────────────────────────
class CostFunction:
    """Least-squares cost coupling the forward model to measurements.

    For each :class:`Measurement` the forward model is solved once and its
    **radial conductance** ``k_r`` [cm hPa⁻¹ d⁻¹] is compared with the observed
    conductance, forming the weighted residual ``w * (k_r_model - k_r_obs)``.
    The full residual vector feeds :func:`scipy.optimize.least_squares`;
    its sum-of-squares feeds scalar optimizers.

    Parameters
    ----------
    forward_model : ForwardModel
        Pre-built forward model.
    param_space : ParamSpace
        Definition of the calibrated coefficients.
    measurements : sequence of Measurement
        Experimental transpiration data points.
    relative : bool, optional
        If ``True`` the residuals are normalized by ``k_r_obs`` (dimensionless
        relative error), which balances points spanning several orders of
        magnitude. Defaults to ``False`` (absolute residuals in
        cm hPa⁻¹ d⁻¹).
    """

    def __init__(
        self,
        forward_model: ForwardModel,
        param_space: ParamSpace,
        measurements: Sequence[Measurement],
        relative: bool = False,
    ) -> None:
        if not measurements:
            raise ValueError("At least one measurement is required.")
        self.fm = forward_model
        self.space = param_space
        self.measurements = list(measurements)
        self.relative = bool(relative)
        #: Number of forward-model solves performed (diagnostics).
        self.n_evals = 0

    # -- driving potential drop for a measurement (matches the forward model) --
    def _dpsi(self, m: Measurement) -> float:
        """Water-potential drop ``|psi_xyl - psi_air|`` [hPa] for a point."""
        psi_air = rh_to_water_potential(m.rh_airspace, self.fm.T)
        return abs(float(m.psi_xyl) - psi_air)

    # -- single forward evaluation --
    def model_kr(self, theta: Dict[str, float], m: Measurement) -> float:
        """Forward-model radial conductance ``k_r`` [cm hPa⁻¹ d⁻¹] for a point."""
        return self.fm.transpiration_flux(
            theta, m.psi_xyl, m.rh_airspace, output="k_r"
        )

    # -- residual vector (for least_squares) --
    def residuals(self, x_log: Sequence[float]) -> np.ndarray:
        """Weighted residual vector ``w * (k_r_model - k_r_obs)``.

        A non-finite forward solve is mapped to a large finite residual so the
        optimizer can retreat from an infeasible region without crashing.
        """
        theta = self.space.to_theta(x_log)
        self.n_evals += 1
        res = np.empty(len(self.measurements), dtype=float)
        for i, m in enumerate(self.measurements):
            kr_obs = m.observed_kr(self._dpsi(m))
            try:
                kr_model = self.model_kr(theta, m)
            except Exception:
                kr_model = np.nan
            r = kr_model - kr_obs
            if self.relative:
                denom = kr_obs if kr_obs != 0.0 else 1.0
                r = r / denom
            r = m.weight * r
            res[i] = r if np.isfinite(r) else 1.0e6
        return res

    # -- scalar objective (for differential_evolution / minimize) --
    def objective(self, x_log: Sequence[float]) -> float:
        """Sum-of-squared weighted residuals (scalar cost)."""
        r = self.residuals(x_log)
        return float(np.dot(r, r))

    def observed(self) -> np.ndarray:
        """Observed radial conductances ``k_r_obs`` for all measurements."""
        return np.array(
            [m.observed_kr(self._dpsi(m)) for m in self.measurements],
            dtype=float,
        )

    def predict(self, theta: Dict[str, float]) -> np.ndarray:
        """Model radial conductances ``k_r`` for all measurements (diagnostics)."""
        return np.array(
            [self.model_kr(theta, m) for m in self.measurements],
            dtype=float,
        )


# ── Optimiser wrapper ─────────────────────────────────────────────────────────
@dataclass
class OptimizeResult:
    """Outcome of a calibration run."""

    theta: Dict[str, float]
    x_log: np.ndarray
    cost: float
    success: bool
    message: str
    n_evals: int
    #: Raw scipy result object(s) for inspection.
    raw: Dict[str, object] = field(default_factory=dict)


class Optimizer:
    """Wrapper around SciPy optimizers.

    Two complementary strategies are exposed:

    * :meth:`run_global` — :func:`scipy.optimize.differential_evolution`, a
      derivative-free global search over the (log) bounds. Robust to the
      multi-modal, order-of-magnitude landscape of transport coefficients but
      comparatively many forward solves.
    * :meth:`run_local` — :func:`scipy.optimize.least_squares`
      (Trust-Region-Reflective) on the residual vector, giving fast local
      refinement and a Jacobian that the sensitivity layer can reuse.

    :meth:`run` chains them (global → local) for a practical default.
    """

    def __init__(self, cost: CostFunction) -> None:
        self.cost = cost
        self.space = cost.space

    # -- global search --
    def run_global(
        self,
        seed: Optional[int] = None,
        maxiter: int = 100,
        popsize: int = 15,
        tol: float = 1.0e-4,
        polish: bool = False,
    ) -> OptimizeResult:
        """Global search with differential evolution over the log bounds."""
        from scipy.optimize import differential_evolution

        self.cost.n_evals = 0
        res = differential_evolution(
            self.cost.objective,
            bounds=self.space.log_bounds,
            seed=seed,
            maxiter=maxiter,
            popsize=popsize,
            tol=tol,
            polish=polish,
            workers=1,          # no parallelism
            updating="immediate",
        )
        x_log = self.space.clip_log(res.x)
        return OptimizeResult(
            theta=self.space.to_theta(x_log),
            x_log=x_log,
            cost=float(res.fun),
            success=bool(res.success),
            message=str(res.message),
            n_evals=self.cost.n_evals,
            raw={"global": res},
        )

    # -- local refinement --
    def run_local(
        self,
        x0_log: Optional[Sequence[float]] = None,
        theta0: Optional[Dict[str, float]] = None,
        max_nfev: Optional[int] = None,
        xtol: float = 1.0e-8,
        ftol: float = 1.0e-8,
        diff_step: float = 1.0e-2,
    ) -> OptimizeResult:
        """Local least-squares refinement (Trust-Region-Reflective).

        Provide either ``x0_log`` (a log10 vector) or ``theta0`` (a physical
        dict); if neither is given the geometric midpoint of the bounds is used.

        The Jacobian is estimated by finite differences of the (noisy) forward
        model. ``diff_step`` is the **relative** perturbation applied to each
        log10 parameter; the default ``1e-2`` (≈ 2.3 % change in the physical
        coefficient) is deliberately much larger than SciPy's ``sqrt(eps)``
        default so the resulting flux change rises well above MECHA's
        solve-to-solve numerical noise (~1e-6 cm^3 d^-1). A too-small step
        produces a spurious zero gradient and premature termination.
        """
        from scipy.optimize import least_squares

        if x0_log is None:
            x0_log = self.space.initial_log_vector(theta0)
        x0_log = self.space.clip_log(x0_log)
        lo = [b[0] for b in self.space.log_bounds]
        hi = [b[1] for b in self.space.log_bounds]

        self.cost.n_evals = 0
        res = least_squares(
            self.cost.residuals,
            x0=x0_log,
            bounds=(lo, hi),
            method="trf",
            xtol=xtol,
            ftol=ftol,
            diff_step=diff_step,
            max_nfev=max_nfev,
        )
        x_log = self.space.clip_log(res.x)
        cost = float(2.0 * res.cost)  # least_squares reports 0.5*||r||^2
        return OptimizeResult(
            theta=self.space.to_theta(x_log),
            x_log=x_log,
            cost=cost,
            success=bool(res.success),
            message=str(res.message),
            n_evals=self.cost.n_evals,
            raw={"local": res},
        )

    # -- combined global → local --
    def run(
        self,
        seed: Optional[int] = None,
        global_maxiter: int = 100,
        global_popsize: int = 15,
        refine: bool = True,
    ) -> OptimizeResult:
        """Global search followed by an optional local refinement."""
        g = self.run_global(
            seed=seed, maxiter=global_maxiter, popsize=global_popsize
        )
        if not refine:
            return g
        l = self.run_local(x0_log=g.x_log)
        total_evals = g.n_evals + l.n_evals
        # Keep whichever stage reached the lower cost.
        best = l if l.cost <= g.cost else g
        return OptimizeResult(
            theta=best.theta,
            x_log=best.x_log,
            cost=best.cost,
            success=best.success,
            message=f"global: {g.message} | local: {l.message}",
            n_evals=total_evals,
            raw={"global": g.raw.get("global"), "local": l.raw.get("local")},
        )


# ══════════════════════════════════════════════════════════════════════════════
# Sensitivity analysis
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class LocalSensitivity:
    """Result of a local (Jacobian-based) sensitivity analysis.

    All quantities are expressed with respect to the **log10 parameters**, i.e.
    a sensitivity of ``s`` means "the model output changes by ``s`` per decade
    of the coefficient", which is the natural, scale-free measure for
    coefficients that span many orders of magnitude.

    Attributes
    ----------
    names : list of str
        Parameter names, in Jacobian-column order.
    x_log : numpy.ndarray
        Log10 parameter vector the Jacobian was evaluated at.
    theta : dict
        Physical parameter values at ``x_log``.
    jacobian : numpy.ndarray, shape (n_measurements, n_params)
        ``J[i, j] = d E_model_i / d log10(theta_j)`` [mmol m^-2 s^-1 / decade].
    ranking : list of (str, float)
        Parameters sorted by descending column-norm sensitivity
        ``||J[:, j]||`` — how strongly each parameter moves the model output.
    fisher : numpy.ndarray, shape (n_params, n_params)
        Gauss-Newton / Fisher information matrix ``J^T J``.
    eigvals : numpy.ndarray
        Eigenvalues of ``J^T J`` (descending). Near-zero eigenvalues flag
        directions in parameter space the data cannot constrain.
    condition_number : float
        ``max(eigval) / min(eigval)`` of ``J^T J``. Large (≫ 1e6) means the
        parameters are jointly non-identifiable from the given measurements.
    collinearity_index : float
        ``1 / sqrt(min eigenvalue of the correlation-scaled J^T J)``.
        Values ≳ 10-20 indicate strong parameter interaction.
    """

    names: List[str]
    x_log: np.ndarray
    theta: Dict[str, float]
    jacobian: np.ndarray
    ranking: List[Tuple[str, float]]
    fisher: np.ndarray
    eigvals: np.ndarray
    condition_number: float
    collinearity_index: float


@dataclass
class MorrisSensitivity:
    """Result of a Morris elementary-effects (global screening) analysis.

    Attributes
    ----------
    names : list of str
        Parameter names, in result order.
    mu_star : numpy.ndarray
        Mean of the absolute elementary effects per parameter — the overall
        (global) influence measure; larger = more influential.
    mu : numpy.ndarray
        Mean of the (signed) elementary effects; sign indicates the direction
        of the average effect.
    sigma : numpy.ndarray
        Standard deviation of the elementary effects — large values indicate
        non-linear response and/or interactions with other parameters.
    ranking : list of (str, float)
        Parameters sorted by descending ``mu_star``.
    n_trajectories : int
        Number of Morris trajectories used.
    """

    names: List[str]
    mu_star: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    ranking: List[Tuple[str, float]]
    n_trajectories: int


class SensitivityAnalyzer:
    """Sensitivity / identifiability diagnostics for the calibration problem.

    Two complementary analyses are offered:

    * :meth:`local` — a finite-difference Jacobian of the model outputs with
      respect to the log10 coefficients, plus the derived identifiability
      diagnostics (Fisher information ``J^T J``, its eigen-spectrum, condition
      number and Brun collinearity index). Cheap (``2 * n_params + 1`` solves
      per measurement set) and reuses the same robust ``diff_step`` that the
      local optimizer needs to see through MECHA's numerical noise.

    * :meth:`morris` — a derivative-free global screening (Morris elementary
      effects) over the full parameter bounds. It ranks parameters by their
      global influence (``mu_star``) and flags non-linearity / interaction
      (``sigma``) without assuming a linearization point.

    Parameters
    ----------
    cost : CostFunction
        The calibration cost (provides the forward model, parameter space and
        measurements).
    """

    def __init__(self, cost: CostFunction) -> None:
        self.cost = cost
        self.space = cost.space
        self.measurements = cost.measurements

    # -- vector of model outputs (radial conductance) across all measurements --
    def _model_vector(self, x_log: Sequence[float]) -> np.ndarray:
        """Model radial conductances ``k_r`` for all measurements at ``x_log``."""
        theta = self.space.to_theta(x_log)
        self.cost.n_evals += 1
        out = np.empty(len(self.measurements), dtype=float)
        for i, m in enumerate(self.measurements):
            try:
                out[i] = self.cost.model_kr(theta, m)
            except Exception:
                out[i] = np.nan
        return out

    # ── local Jacobian-based sensitivity ──────────────────────────────────────
    def local(
        self,
        x_log: Optional[Sequence[float]] = None,
        theta: Optional[Dict[str, float]] = None,
        diff_step: float = 1.0e-2,
    ) -> LocalSensitivity:
        """Finite-difference Jacobian and identifiability diagnostics.

        Provide either ``x_log`` (log10 vector) or ``theta`` (physical dict) to
        set the evaluation point; if neither is given the geometric midpoint of
        the bounds is used.

        ``diff_step`` is the **absolute** perturbation of each log10 parameter
        (a step of ``1e-2`` ≈ 2.3 % change in the physical coefficient). It is
        deliberately much larger than machine precision so the resulting output
        change rises above MECHA's ~1e-6 solve-to-solve noise; too small a step
        yields a spurious zero gradient.
        """
        if x_log is None:
            x_log = self.space.initial_log_vector(theta)
        x_log = self.space.clip_log(x_log)
        n = self.space.ndim
        m = len(self.measurements)

        self.cost.n_evals = 0
        f0 = self._model_vector(x_log)
        J = np.zeros((m, n), dtype=float)
        for j in range(n):
            xp = np.array(x_log, dtype=float)
            xm = np.array(x_log, dtype=float)
            xp[j] += diff_step
            xm[j] -= diff_step
            # Clip to bounds; fall back to one-sided if a bound is hit.
            xp = self.space.clip_log(xp)
            xm = self.space.clip_log(xm)
            fp = self._model_vector(xp)
            fm_ = self._model_vector(xm)
            denom = xp[j] - xm[j]
            J[:, j] = (fp - fm_) / denom if denom != 0.0 else 0.0

        # Column-norm sensitivity ranking.
        col_norms = np.linalg.norm(J, axis=0)
        order = np.argsort(col_norms)[::-1]
        ranking = [(self.space.names[k], float(col_norms[k])) for k in order]

        # Fisher information and identifiability diagnostics.
        fisher = J.T @ J
        eigvals = np.linalg.eigvalsh(fisher)[::-1]  # descending
        eig_min = float(eigvals[-1])
        eig_max = float(eigvals[0])
        cond = eig_max / eig_min if eig_min > 0.0 else np.inf

        # collinearity index on the column-normalised sensitivity
        # matrix: gamma = 1 / sqrt(min eigenvalue of the normalised J^T J).
        nz = col_norms > 0.0
        if nz.any():
            Jn = J[:, nz] / col_norms[nz]
            corr_eig = np.linalg.eigvalsh(Jn.T @ Jn)
            min_corr = float(corr_eig[0])
            collinearity = 1.0 / np.sqrt(min_corr) if min_corr > 0.0 else np.inf
        else:
            collinearity = np.inf

        return LocalSensitivity(
            names=list(self.space.names),
            x_log=x_log,
            theta=self.space.to_theta(x_log),
            jacobian=J,
            ranking=ranking,
            fisher=fisher,
            eigvals=eigvals,
            condition_number=float(cond),
            collinearity_index=float(collinearity),
        )

    # ── global Morris screening ────────────────────────────────────────────────
    def morris(
        self,
        n_trajectories: int = 10,
        n_levels: int = 4,
        seed: Optional[int] = None,
        aggregate: str = "rms",
    ) -> MorrisSensitivity:
        """Morris elementary-effects global screening over the log bounds.

        The Morris method perturbs one parameter at a time along randomly
        placed trajectories through a discretised ``n_levels`` grid on the
        (log10) parameter box, measuring the resulting change in the model
        output (an *elementary effect*). The mean absolute effect ``mu_star``
        ranks global influence; the spread ``sigma`` flags non-linearity /
        interactions. No linearization point is assumed, so it complements
        :meth:`local`.

        Because the model output here is a *vector* (one ``E`` per
        measurement), each elementary effect is reduced to a scalar via
        ``aggregate`` (``"rms"`` root-mean-square, or ``"mean"`` of absolute
        components) before the Morris statistics are formed.

        Cost: ``n_trajectories * (n_params + 1)`` forward-model solves.

        References
        ----------
        Morris (1991) Technometrics 33:161-174;
        Campolongo et al. (2007) Environ. Model. Softw. 22:1509-1518.
        """
        rng = np.random.default_rng(seed)
        n = self.space.ndim
        lo = np.array([b[0] for b in self.space.log_bounds])
        hi = np.array([b[1] for b in self.space.log_bounds])
        span = hi - lo
        delta = 1.0 / (n_levels - 1) if n_levels > 1 else 1.0  # grid step (fraction)

        def scalarise(vec: np.ndarray) -> float:
            v = np.asarray(vec, dtype=float)
            if not np.all(np.isfinite(v)):
                return np.nan
            if aggregate == "mean":
                return float(np.mean(np.abs(v)))
            return float(np.sqrt(np.mean(v * v)))  # rms

        self.cost.n_evals = 0
        effects: List[List[float]] = [[] for _ in range(n)]
        for _ in range(n_trajectories):
            # Random starting point on the discrete grid (fractional 0..1).
            base = rng.integers(0, n_levels, size=n) / (n_levels - 1) \
                if n_levels > 1 else np.zeros(n)
            x_frac = base.astype(float)
            f_prev = self._model_vector(lo + x_frac * span)
            # Random order in which parameters are perturbed.
            for j in rng.permutation(n):
                x_new = x_frac.copy()
                # Step +delta, or -delta if that would leave the unit box.
                step = delta if x_new[j] + delta <= 1.0 else -delta
                x_new[j] += step
                f_new = self._model_vector(lo + x_new * span)
                # Elementary effect in physical decade units:
                # d(output) / d(log10 param) = d(output) / (step * span_j).
                dparam = step * span[j]
                ee_vec = (f_new - f_prev) / dparam if dparam != 0.0 else \
                    np.zeros_like(f_new)
                effects[j].append(scalarise(ee_vec))
                x_frac = x_new
                f_prev = f_new

        mu = np.array([np.nanmean(e) if e else np.nan for e in effects])
        mu_star = np.array(
            [np.nanmean(np.abs(e)) if e else np.nan for e in effects]
        )
        sigma = np.array(
            [np.nanstd(e) if len(e) > 1 else 0.0 for e in effects]
        )
        order = np.argsort(np.nan_to_num(mu_star, nan=-np.inf))[::-1]
        ranking = [(self.space.names[k], float(mu_star[k])) for k in order]

        return MorrisSensitivity(
            names=list(self.space.names),
            mu_star=mu_star,
            mu=mu,
            sigma=sigma,
            ranking=ranking,
            n_trajectories=n_trajectories,
        )
