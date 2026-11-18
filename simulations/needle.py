"""
needle.py
=========
Building blocks for needle water–sucrose simulations.  A run is composed from
independent, immutable option objects and executed by one ``NeedleSimulation``:

    Experiment(
        label    = 'my_run',
        anatomy  = Anatomy(seed=42),                       # 1) geometry / anatomy
        solute   = Solute(d_pd=D_PD_SUCROSE),              #    transport properties
        bc       = DirichletBC(...) | NeumannBC(...),      # 2) boundary conditions
        dynamics = SteadyState() | TimeStepping(...),      # 3) steady or transient
        coupling = NoCoupling() | Picard() | JFNK(),       # 4) osmotic coupling
        starch   = None | Starch(...) | DielStarch(...),   # 5) optional starch
    )
    res = NeedleSimulation(exp).run()                      # 7) SimResult

Switch one option with ``exp.replace(coupling=Picard())``; ``run_experiments``
runs a list of experiments in parallel (anatomy built once per seed and worker).

Units: lengths µm/cm, volumes cm³, conc mol/cm³ (=10³ M), rates mol/d, time d.
"""

from __future__ import annotations

import os
import sys
import copy
import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Optional

import numpy as np
from scipy.sparse import csgraph

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR    = os.path.join(_SCRIPT_DIR, 'outputs')
os.makedirs(_OUT_DIR, exist_ok=True)

_MECHA_SRC  = os.path.join(_SCRIPT_DIR, '..', '..', 'MECHA-dev', 'src')
_GRANAP_SRC = os.path.join(_SCRIPT_DIR, '..', '..', 'GRANAP-dev', 'src')
for _p in (_MECHA_SRC, _GRANAP_SRC):
    sys.path.insert(0, os.path.abspath(_p))

from openalea.granap.needle_class import NeedleAnatomy
from openalea.mecha.mecha_class import Mecha
from openalea.mecha.utils.data_loader import InData, BoundaryData
from openalea.mecha.utils.network_builder import NetworkBuilder
from openalea.mecha.utils.solute_transport import (
    SoluteTransport, SoluteGeometry, MultiSoluteTransport)
from openalea.mecha.utils.coupled_solver import (
    coupled_water_solute_solve, coupled_implicit_step)
from openalea.mecha.calibration.forward_model import rh_to_water_potential

# Gas constant used by MECHA for van't Hoff Ψ_os = −R·T·c  [hPa cm³ mol⁻¹ K⁻¹].
_R_HPA = 8.314e4

# Network link types recorded per edge; extras['edge_path'] indexes this tuple.
EDGE_PATHS = ('wall', 'membrane', 'plasmodesmata')

_SEC_PER_DAY = 86400.0
_V_WATER = 18.015          # cm³/mol  molar volume of liquid water (M_w / ρ_w)
_C_PER_SUCROSE = 12        # C atoms per sucrose (C12H22O11)

# Default base plasmodesmata frequency (Fplxheight);
# the whole per-interface density profile is a ratio to this.
_FPLXHEIGHT_BASE = 8.0e5

# Sucrose diffusion in water @ 25 C: 5.2e-6 cm²/s × 86400 s/d → cm²/d.
DIFF_SUCROSE = 5.2e-6 * 86400.0

# Sucrose plasmodesmatal diffusivity
D_PD_SUCROSE = 0.1 * DIFF_SUCROSE

# Sucrose plasmodesmatal diffusivity
D_APO_SUCROSE = 0.8 * DIFF_SUCROSE

# Sucrose membrane diffusivity
D_MEM_SUCROSE = 0.3 * DIFF_SUCROSE


def transpiration_to_flow(E: float, area_m2: float) -> float:
    """E [mmol m⁻² s⁻¹] → total liquid water flux [cm³/d] over ``area_m2``."""
    return E * 1e-3 * _V_WATER * _SEC_PER_DAY * area_m2


def assimilation_to_sucrose(A: float, area_m2: float) -> float:
    """A [µmol CO₂ m⁻² s⁻¹] → sucrose production [mol/d] over ``area_m2``.

    Upper bound: all fixed carbon is assumed to end up as sucrose.
    """
    return A * 1e-6 / _C_PER_SUCROSE * _SEC_PER_DAY * area_m2


# ══════════════════════════════════════════════════════════════════════════════
# Diurnal modulation helpers
# ══════════════════════════════════════════════════════════════════════════════
def f_transpiration(t, period: float = 1.0, phase: float = 0.0,
                    baseline: float = 0.1):
    """Transpirational driver f(t) ∈ [baseline, 1]: smooth cosine, peaks at noon.

    Night floor ``baseline`` (stomata never fully close), 1 at solar noon.
    """
    wave = 0.5 * (1.0 - np.cos(2.0 * np.pi * (t / period) + phase))
    return baseline + (1.0 - baseline) * wave


def g_photosynthesis(t, period: float = 1.0, phase: float = 0.0):
    """Photosynthetic source g(t) = max(0, −cos) ∈ [0, 1]: zero at night."""
    return np.maximum(0.0, -np.cos(2.0 * np.pi * (t / period) + phase))


def s_diel(t, period: float = 1.0, phase: float = 0.0):
    """Diel conversion signal s(t) ∈ [−1, 1]: +1 at noon, −1 at midnight."""
    return -np.cos(2.0 * np.pi * (t / period) + phase)


# ══════════════════════════════════════════════════════════════════════════════
# 1) Geometry / anatomy
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Anatomy:
    """Which anatomy draw and which MECHA stage/scenario indices to use."""

    seed: Optional[int] = 42       # None → random draw
    # Base PD frequency (Fplxheight); rescales the whole per-interface map by
    # pm_den / 8.0E5 (both PD hydraulic K and solute DF). None → XML defaults.
    pm_den: Optional[float] = None
    h_idx: int = 0
    i_mat: int = 0
    i_sce: int = 1


@dataclass
class GeometryBase:
    """The expensive, stochastic ``NeedleAnatomy`` draw shared across runs.

    Deterministic for a fixed ``seed``, so parallel workers rebuild it from the
    seed rather than pickling the live network object.
    """

    seed: Optional[int]
    anatomy: NeedleAnatomy

    @classmethod
    def build(cls, seed: Optional[int] = None) -> "GeometryBase":
        anatomy = NeedleAnatomy(seed=seed) if seed is not None else NeedleAnatomy()
        anatomy.export_to_adjencymatrix()
        return cls(seed=seed, anatomy=anatomy)


_BASE_CACHE: dict = {}


def get_geometry_base(seed: Optional[int]) -> GeometryBase:
    """Per-process cached ``GeometryBase`` for ``seed`` (built on first use)."""
    if seed not in _BASE_CACHE:
        _BASE_CACHE[seed] = GeometryBase.build(seed)
    return _BASE_CACHE[seed]


@dataclass(frozen=True)
class Solute:
    """Sucrose transport properties and discretisation."""

    d_pd: float = D_PD_SUCROSE     # cm²/d  plasmodesmatal diffusivity
    d_apo: float = D_APO_SUCROSE   # cm²/d  apoplastic wall diffusivity
    d_mem: float = D_MEM_SUCROSE   # cm²/d  passive transmembrane diffusivity
    sigma: float = 0.97            # -      membrane reflection coefficient
    # Effective xylem reflection (solute only; 1 → xylem impermeable).
    sigma_xylem: Optional[float] = None
    ops: str = 'T'                 # 'D' | 'T'  (T = advection + diffusion)
    scheme: str = 'sg'             # 'upwind' | 'sg' (Scharfetter–Gummel)

    def diffusion_params(self) -> dict:
        return dict(
            apo_wall=self.d_apo,
            membrane=self.d_mem,
            plasmodesmata=self.d_pd,
            sigma={cg: self.sigma for cg in range(1, 20)},
            sigma_xylem=self.sigma_xylem,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2) Boundary conditions
# ══════════════════════════════════════════════════════════════════════════════
# Tissue groups that may be used as Dirichlet sucrose sinks (BoundaryConditions.sinks).
SINK_GROUPS = ('strasburger', 'phloem', 'phloem_loading', 'xylem')


@dataclass(frozen=True)
class Diurnal:
    """Day/night modulation: water drive × f(t), sucrose source × g(t)."""

    period: float = 1.0            # d
    f_phase: float = 0.0           # rad  transpiration driver
    g_phase: float = 0.0           # rad  photosynthesis source
    f_baseline: float = 0.1        # -    night transpiration floor

    def f(self, t: float) -> float:
        return float(f_transpiration(t, self.period, self.f_phase, self.f_baseline))

    def g(self, t: float) -> float:
        return float(g_photosynthesis(t, self.period, self.g_phase))


@dataclass(frozen=True)
class BoundaryConditions:
    """Shared part of the water + sucrose boundary conditions.

    The xylem water potential is always Dirichlet (sets the potential level);
    the sucrose sinks are Dirichlet ``c_sink`` on the listed ``sinks`` groups.
    """

    psi_xyl: float = -200.0        # hPa
    sinks: tuple = ('phloem_loading', 'xylem')
    c_sink: float = 0.0            # mol/cm³
    diurnal: Optional[Diurnal] = None   # None → constant drive (f = g = 1)

    def f(self, t: float) -> float:
        return 1.0 if self.diurnal is None else self.diurnal.f(t)

    def g(self, t: float) -> float:
        return 1.0 if self.diurnal is None else self.diurnal.g(t)

    # ---- hooks used by NeedleSimulation ---------------------------------------
    def setup(self, sim: "NeedleSimulation") -> None:
        """Register the water BCs on ``sim.mecha`` (before any solve)."""
        a = sim.exp.anatomy
        sim.mecha.psi_xyl[1, a.i_mat, a.i_sce] = self.psi_xyl
        sim.mecha.psi_sieve[1, a.i_mat, a.i_sce] = np.nan

    def drive(self, sim: "NeedleSimulation", t: float) -> None:
        """Update the water drive at time ``t`` (no-op when constant)."""

    def target_flow(self, sim: "NeedleSimulation", t: float) -> Optional[float]:
        """Imposed transpiration [cm³/d] at ``t``; None when not prescribed."""
        return None

    def source(self, sim: "NeedleSimulation", c: np.ndarray, t: float):
        """Sucrose input for the step ending at ``t``.

        Returns ``(c, rhs, q)``: start field (possibly with injected mass), nodal
        source vector [mol/d] and the production rate q [mol/d].
        """
        return c, np.zeros(sim.st._matrix_size), 0.0

    def steady_rhs(self, sim: "NeedleSimulation") -> np.ndarray:
        """Steady-state rhs (T c = rhs, i.e. −source)."""
        return np.zeros(sim.st._matrix_size)

    def fixed_concentrations(self, sim: "NeedleSimulation") -> dict:
        """Extra sucrose Dirichlet values {node: c} beyond the sinks."""
        return {}


@dataclass(frozen=True)
class DirichletBC(BoundaryConditions):
    """Prescribed potentials/concentrations.

    Water: air spaces held at ``psi_air`` (or the Kelvin potential of ``rh_air``),
    scaled by f(t).  Sucrose: optional fixed mesophyll ``c_meso`` and/or a
    per-step injection ``c_photo``·g(t) into the (free) mesophyll.
    """

    air: str = 'psi'                # 'psi' | 'rh'
    psi_air: float = -1.0e4         # hPa  (air='psi')
    rh_air: float = 0.99            # -    (air='rh')
    c_meso: Optional[float] = None  # mol/cm³  hold mesophyll at this value
    c_photo: float = 0.0            # mol/cm³  per-step mesophyll increment at g=1

    def setup(self, sim):
        super().setup(sim)
        if self.air == 'rh':
            sim.psi_air = rh_to_water_potential(self.rh_air, sim.exp.t_kelvin)
        elif self.air == 'psi':
            sim.psi_air = self.psi_air
        else:
            raise ValueError(f"air must be 'psi' or 'rh'; got {self.air!r}.")
        sim.mecha.set_air_wall_bc(sim.psi_air)
        sim._log(f"  Air Dirichlet BC Ψ_air={sim.psi_air:.1f} hPa ({self.air})")

    def drive(self, sim, t):
        if self.diurnal is not None:
            sim.mecha.set_air_wall_bc(sim.psi_air * self.f(t))

    def source(self, sim, c, t):
        rhs = np.zeros(sim.st._matrix_size)
        if self.c_photo == 0.0:
            return c, rhs, 0.0
        dc = self.c_photo * self.g(t)
        c = c.copy()
        c[sim.meso_nodes] += dc
        injected = dc * float(np.sum(sim.node_vols[sim.meso_nodes]))
        return c, rhs, injected / sim.exp.dynamics.dt

    def fixed_concentrations(self, sim):
        if self.c_meso is None:
            return {}
        return {int(n): self.c_meso for n in sim.meso_nodes}


@dataclass(frozen=True)
class NeumannBC(BoundaryConditions):
    """Prescribed gas-exchange fluxes.

    Water: transpiration E·f(t) leaves through the air spaces (split by area).
    Sucrose: assimilation A·g(t) → sucrose (12 C, upper bound), spread per
    mesophyll volume.  E and A refer to ``ref_area`` (None → outer perimeter ×
    slice height).
    """

    transpiration_E: float = 1.0   # mmol m⁻² s⁻¹
    assimilation_A: float = 5.0    # µmol m⁻² s⁻¹
    ref_area: Optional[float] = None   # m²

    def setup(self, sim):
        super().setup(sim)
        sim.ref_area = sim.reference_area(self.ref_area)
        sim.flow_trans = transpiration_to_flow(self.transpiration_E, sim.ref_area)
        sim.mecha._distribute_transpiration_flow(sim.flow_trans)
        sim._log(f"  Air Neumann BC E={self.transpiration_E} mmol/m²/s → "
                 f"{sim.flow_trans:.3e} cm³/d over {sim.ref_area:.3e} m²")

    def drive(self, sim, t):
        if self.diurnal is not None:
            sim.mecha._distribute_transpiration_flow(sim.flow_trans * self.f(t))

    def target_flow(self, sim, t):
        return sim.flow_trans * self.f(t)

    def _production(self, sim, t: Optional[float]) -> float:
        g = 1.0 if t is None else self.g(t)
        return assimilation_to_sucrose(self.assimilation_A * g, sim.ref_area)

    def source(self, sim, c, t):
        q = self._production(sim, t)
        rhs = sim.st._distribute_assimilates(q, sim.meso_cell_ids, sim.exp.anatomy.i_mat)
        return c, rhs, q

    def steady_rhs(self, sim):
        q = self._production(sim, None)
        return -sim.st._distribute_assimilates(q, sim.meso_cell_ids, sim.exp.anatomy.i_mat)


# ══════════════════════════════════════════════════════════════════════════════
# 3) Dynamics
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class SteadyState:
    """Solve the (coupled) steady state directly."""

    continuation: Optional[tuple] = None   # homotopy λ schedule (Picard/JFNK only)


@dataclass(frozen=True)
class TimeStepping:
    """θ-method time march; any enabled stop criterion ends the run early."""

    dt: float = 1.0 / 24.0         # d
    max_steps: int = 48
    theta: float = 1.0             # 1.0 implicit Euler | 0.5 Crank–Nicolson
    c0_meso: float = 0.0           # mol/cm³  initial (free) mesophyll sucrose
    # ---- stop criteria (None → disabled) -------------------------------------
    sink_fraction: Optional[float] = None  # cum. outflow to sinks ≥ fraction × initial mass
    equil_tol: Optional[float] = None      # max|Δc| < equil_tol  [mol/cm³]
    balance_tol: Optional[float] = None    # |mean(export/production) − 1| < tol
    balance_window: float = 0.0            # d  averaging window (0 → last step)


# ══════════════════════════════════════════════════════════════════════════════
# 4) Osmotic coupling  (Ψ_os = baseline − R·T·c on the cells)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class NoCoupling:
    """No iteration: Ψ_os is read from the start-of-step concentration (lagged)."""

    def step(self, sim, c, baseline, rhs, transport_step):
        c_new, n, _ = sim._implicit_step(c, baseline, rhs, transport_step,
                                         method='picard', max_iter=1, tol=np.inf)
        return c_new, n, True

    def steady(self, sim, rhs):
        a, s = sim.exp.anatomy, sim.exp.solute
        sim.mecha.water_flux(h=a.h_idx, verbose=False)
        c = sim.st.solve(h=a.h_idx, i_maturity=a.i_mat, i_scenario=a.i_sce,
                         rhs=rhs.copy(), boundary_conditions=sim.bc,
                         operators=s.ops, scheme=s.scheme)
        return c, 1, True


@dataclass(frozen=True)
class Picard:
    """Fixed-point iteration of water ↔ solute, optionally under-relaxed."""

    tol: float = 10.0              # hPa  on max|ΔΨ_os| (steady: max|ΔΨ_total|)
    max_iter: int = 12
    relax: float = 1.0             # ω ∈ (0, 1]

    def step(self, sim, c, baseline, rhs, transport_step):
        return sim._implicit_step(c, baseline, rhs, transport_step, method='picard',
                                  tol=self.tol, max_iter=self.max_iter,
                                  relaxation=self.relax)

    def steady(self, sim, rhs):
        return sim._coupled_steady(rhs, method='picard', tol=self.tol,
                                   max_iter=self.max_iter, relaxation=self.relax)


@dataclass(frozen=True)
class JFNK:
    """Jacobian-free Newton–Krylov on the cell concentrations."""

    tol: float = 10.0              # hPa  (residual tolerance = tol / R·T)
    max_iter: int = 10             # Newton iterations
    inner: int = 30                # GMRES iterations per Newton step
    line_search: Optional[str] = 'armijo'
    rdiff: float = 1.0e-3

    def _kw(self) -> dict:
        return dict(tol=self.tol, max_iter=self.max_iter, jfnk_maxiter=self.max_iter,
                    jfnk_inner_maxiter=self.inner, jfnk_line_search=self.line_search,
                    jfnk_rdiff=self.rdiff)

    def step(self, sim, c, baseline, rhs, transport_step):
        return sim._implicit_step(c, baseline, rhs, transport_step, method='jfnk',
                                  **self._kw())

    def steady(self, sim, rhs):
        return sim._coupled_steady(rhs, method='jfnk', **self._kw())


# ══════════════════════════════════════════════════════════════════════════════
# 5) Starch (optional immobile second solute, sucrose ↔ starch)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Starch:
    """Constant-rate reversible sucrose ↔ starch conversion."""

    k_syn: float = 2.0             # 1/d  sucrose → starch
    k_deg: float = 1.0             # 1/d  starch → sucrose
    c0: float = 0.0                # mol/cm³  initial starch in cells

    def rates(self, t: float) -> tuple:
        return self.k_syn, self.k_deg


@dataclass(frozen=True)
class DielStarch:
    """Day/night rates: k_f = k_f_day·max(0, s), k_r = k_r_night·max(0, −s)."""

    k_f_day: float = 6.0           # 1/d  peak noon synthesis
    k_r_night: float = 6.0         # 1/d  peak midnight remobilisation
    period: float = 1.0            # d
    phase: float = 0.0             # rad
    c0: float = 0.0                # mol/cm³

    def rates(self, t: float) -> tuple:
        s = float(s_diel(t, self.period, self.phase))
        return self.k_f_day * max(0.0, s), self.k_r_night * max(0.0, -s)


# ══════════════════════════════════════════════════════════════════════════════
# Experiment = one complete, immutable set-up
# ══════════════════════════════════════════════════════════════════════════════
def _describe(obj):
    """Nested, JSON-friendly dict of an option tree (keeps the option type)."""
    if is_dataclass(obj):
        out = {'type': type(obj).__name__}
        out.update({f.name: _describe(getattr(obj, f.name)) for f in fields(obj)})
        return out
    if isinstance(obj, tuple):
        return [_describe(v) for v in obj]
    return obj


@dataclass(frozen=True)
class Experiment:
    label: str = 'needle'
    anatomy: Anatomy = Anatomy()
    solute: Solute = Solute()
    bc: BoundaryConditions = DirichletBC()
    dynamics: object = TimeStepping()          # SteadyState | TimeStepping
    coupling: object = JFNK()                  # NoCoupling | Picard | JFNK
    starch: Optional[object] = None            # None | Starch | DielStarch
    t_kelvin: float = 298.15                   # K
    edge_snapshot_dt: Optional[float] = 1.0    # d  per-link snapshots; None → every step

    def __post_init__(self):
        if self.starch is not None and not isinstance(self.dynamics, TimeStepping):
            raise ValueError("Starch requires TimeStepping dynamics.")
        bad = [s for s in self.bc.sinks if s not in SINK_GROUPS]
        if bad:
            raise ValueError(f"Unknown sink group(s) {bad}; choose from {SINK_GROUPS}.")

    def replace(self, **changes) -> "Experiment":
        return dataclasses.replace(self, **changes)

    def describe(self) -> dict:
        return _describe(self)


# ══════════════════════════════════════════════════════════════════════════════
# 7) Result container (raw data only; plotting lives in post_sim.py)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class SimResult:
    """Plain-array, picklable result of a single simulation run.

    Frames are ``float32``; ``concentration`` is 1-D (steady) or 2-D
    ``(n_steps, N)`` (dynamic).  Per-link snapshots live in ``extras['edge_*']``.
    """

    mode: str                                  # 'steady' | 'dynamic'
    params: dict                               # Experiment.describe()
    concentration: np.ndarray                  # float32 (N,) or (n_steps, N)

    cell_polygons: Optional[object] = None     # WKB per cell
    cell_ids: Optional[np.ndarray] = None
    node_of_cell: Optional[np.ndarray] = None  # full-network index per cell id

    # ---- Dynamic-only --------------------------------------------------------
    times: Optional[np.ndarray] = None
    interior_mass: Optional[np.ndarray] = None
    frac_to_sink: Optional[np.ndarray] = None
    sink_label: str = 'to sink'
    sink_unit: str = 'pmol'
    sink_scale: float = 1.0e12
    stop_reason: Optional[str] = None
    concentration_starch: Optional[np.ndarray] = None

    # Transpiration [cm³/d] through the wall_air walls at the sucrose-free baseline.
    water_flux_total: Optional[float] = None

    # ---- Steady-only ---------------------------------------------------------
    q_load: Optional[float] = None
    n_iterations: Optional[int] = None
    converged: Optional[bool] = None

    # Series-specific outputs, round-tripped by save/load as 'extra__<key>'.
    extras: dict = field(default_factory=dict)

    @property
    def j_w(self):
        return self.extras.get('j_w')

    @property
    def j_s(self):
        return self.extras.get('j_s')

    @property
    def m_starch(self):
        return self.extras.get('m_starch')

    @property
    def f_drive(self):
        return self.extras.get('f_drive')

    @property
    def g_drive(self):
        return self.extras.get('g_drive')

    def get_extra(self, name: str, default=None):
        return self.extras.get(name, default)

    def param(self, name: str, default=None):
        """First value named ``name`` anywhere in the (nested) params tree."""
        stack = [self.params]
        while stack:
            d = stack.pop(0)
            if not isinstance(d, dict):
                continue
            if name in d:
                return d[name]
            stack.extend(v for v in d.values() if isinstance(v, dict))
        return default

    def plot_flow_network(self, path: Optional[str] = None, quantity: str = 'solute',
                          step: int = -1, **kwargs):
        """Per-link arrows of 'solute' | 'water' | 'peclet' (see post_sim.flow_network_plot)."""
        from post_sim import flow_network_plot
        return flow_network_plot(self, path, quantity=quantity, step=step, **kwargs)

    # ─────────────────────────────────────────────────────────────────────────
    def save(self, path: str) -> str:
        """Persist to a compressed ``.npz`` (params stored as a JSON sidecar)."""
        import json
        arrays = dict(
            mode=self.mode,
            concentration=self.concentration,
            cell_polygons=(np.array(self.cell_polygons, dtype=object)
                           if self.cell_polygons is not None else np.empty(0)),
            cell_ids=(self.cell_ids if self.cell_ids is not None else np.empty(0)),
            node_of_cell=(self.node_of_cell if self.node_of_cell is not None
                          else np.empty(0)),
            times=(self.times if self.times is not None else np.empty(0)),
            interior_mass=(self.interior_mass if self.interior_mass is not None
                           else np.empty(0)),
            frac_to_sink=(self.frac_to_sink if self.frac_to_sink is not None
                          else np.empty(0)),
            sink_label=self.sink_label,
            sink_unit=self.sink_unit,
            sink_scale=self.sink_scale,
            q_load=(np.nan if self.q_load is None else self.q_load),
            n_iterations=(-1 if self.n_iterations is None else self.n_iterations),
            converged=(self.converged if self.converged is not None else False),
            stop_reason=(self.stop_reason or ''),
            water_flux_total=(np.nan if self.water_flux_total is None
                              else self.water_flux_total),
            concentration_starch=(self.concentration_starch
                                  if self.concentration_starch is not None
                                  else np.empty(0)),
        )
        for key, val in self.extras.items():
            arrays[f'extra__{key}'] = np.asarray(val)
        np.savez_compressed(path, **arrays)
        sidecar = os.path.splitext(path)[0] + '.params.json'
        with open(sidecar, 'w') as fh:
            json.dump(self.params, fh, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path: str) -> "SimResult":
        import json
        with np.load(path, allow_pickle=True) as z:
            res = cls(
                mode=str(z['mode']),
                params={},
                concentration=z['concentration'],
                cell_polygons=(z['cell_polygons'] if z['cell_polygons'].size else None),
                cell_ids=(z['cell_ids'] if z['cell_ids'].size else None),
                node_of_cell=(z['node_of_cell'] if z['node_of_cell'].size else None),
                times=(z['times'] if z['times'].size else None),
                interior_mass=(z['interior_mass'] if z['interior_mass'].size else None),
                frac_to_sink=(z['frac_to_sink'] if z['frac_to_sink'].size else None),
                **({k: (float(z[k]) if k == 'sink_scale' else str(z[k]))
                    for k in ('sink_label', 'sink_unit', 'sink_scale') if k in z}),
                q_load=(None if np.isnan(z['q_load']) else float(z['q_load'])),
                n_iterations=(None if int(z['n_iterations']) < 0
                              else int(z['n_iterations'])),
                converged=bool(z['converged']),
                stop_reason=(str(z['stop_reason']) or None),
                water_flux_total=(None if ('water_flux_total' not in z
                                           or np.isnan(z['water_flux_total']))
                                  else float(z['water_flux_total'])),
                concentration_starch=(z['concentration_starch']
                                      if ('concentration_starch' in z
                                          and z['concentration_starch'].size)
                                      else None),
                extras={k[len('extra__'):]: z[k]
                        for k in z.files if k.startswith('extra__')},
            )
        sidecar = os.path.splitext(path)[0] + '.params.json'
        if os.path.exists(sidecar):
            with open(sidecar) as fh:
                res.params = json.load(fh)
        return res

    def summary(self) -> dict:
        """Key scalar outputs as a flat dict (for tabulating experiments)."""
        out = {'label': self.params.get('label', ''), 'mode': self.mode}
        if self.mode == 'dynamic':
            out.update(
                n_steps=(0 if self.times is None else int(self.times.size - 1)),
                t_final=(None if self.times is None else float(self.times[-1])),
                mass_to_sink=(None if self.frac_to_sink is None
                              else float(self.frac_to_sink[-1])),
                stop_reason=self.stop_reason,
            )
        else:
            out.update(q_load=self.q_load, n_iterations=self.n_iterations,
                       converged=self.converged)
        return out


# ══════════════════════════════════════════════════════════════════════════════
# Simulation engine
# ══════════════════════════════════════════════════════════════════════════════
class NeedleSimulation:
    """Build a private MECHA scaffold for ``exp`` and run it."""

    def __init__(self, exp: Experiment,
                 geometry: Optional[GeometryBase] = None,
                 verbose: bool = True):
        self.exp = exp
        self.verbose = verbose
        self.geometry = geometry or get_geometry_base(exp.anatomy.seed)
        self.psi_air = self.flow_trans = self.ref_area = None

        self._build_network()
        self._classify_tissue()
        exp.bc.setup(self)
        self._build_transport()
        self._build_solute_bc()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # ---- setup ----------------------------------------------------------------
    def _build_network(self) -> None:
        a = self.exp.anatomy
        self._log(f"=== Building MECHA network (label={self.exp.label!r}) ===")
        network = NetworkBuilder(self.geometry.anatomy)
        network.populate_from_network()

        data = InData.needle_defaults()
        # Full default scenario (valid osmotic-profile fields) with σ on; the only
        # osmotic magnitude is the dynamic van't Hoff sucrose term.
        s1 = copy.deepcopy(BoundaryData().scenarios[0])
        s1['s_factor'] = self.exp.solute.sigma
        data.boundary.add_scenario(s1)

        if a.pm_den is not None:
            scale = a.pm_den / _FPLXHEIGHT_BASE
            data.hydraulic.fplxheight_map = {
                interface: height * scale
                for interface, height in data.hydraulic.fplxheight_map.items()
            }

        self.mecha = Mecha(data, network=network)
        self.nwj     = self.mecha.network.n_wall_junction
        self.n_cells = self.mecha.network.n_cells
        self.n_total = self.nwj + self.n_cells
        self._log(f"  n_wall_junction={self.nwj}  n_cells={self.n_cells}")

    def _classify_tissue(self) -> None:
        nwj = self.nwj
        meso, strasburger, phloem, xylem, airspace = [], [], [], [], []
        for nd, d in self.mecha.network.graph.nodes(data=True):
            idx = self.mecha.indice[nd]
            if idx < nwj:
                continue
            cid = idx - nwj
            cg  = int(d.get('cgroup', -1))
            ct  = str(d.get('cell_type', ''))
            if self.mecha.network.is_wall_air_cell(nd):
                airspace.append(cid)
            elif cg == 4 and ct == 'mesophyll':
                meso.append(cid)
            elif cg == 12 and ct == 'Strasburger cell':
                strasburger.append(cid)
            elif cg == 11:
                phloem.append(cid)
            elif cg == 13:
                xylem.append(cid)
        if not meso:                       # fallback: any cgroup-4 non-air cell
            for nd, d in self.mecha.network.graph.nodes(data=True):
                idx = self.mecha.indice[nd]
                if (idx >= nwj and int(d.get('cgroup', -1)) == 4
                        and not self.mecha.network.is_wall_air_cell(nd)):
                    meso.append(idx - nwj)
        self.meso_cell_ids        = meso
        self.strasburger_cell_ids = strasburger
        self.phloem_cell_ids      = phloem
        self.xylem_cell_ids       = xylem
        self.airspace_cell_ids    = airspace
        self.phloem_loading_ids   = strasburger if strasburger else phloem
        self.meso_nodes = nwj + np.asarray(meso, dtype=int)
        self._log(f"  mesophyll={len(meso)}  phloem-loading={len(self.phloem_loading_ids)}"
                  f"  xylem={len(xylem)}  air={len(airspace)}")

    def tissue_ids(self, group: str) -> list:
        """Cell ids of a sink group (see SINK_GROUPS)."""
        return {
            'strasburger':    sorted(self.strasburger_cell_ids),
            'phloem':         sorted(set(self.phloem_cell_ids) - set(self.strasburger_cell_ids)),
            'phloem_loading': sorted(self.phloem_loading_ids),
            'xylem':          sorted(self.xylem_cell_ids),
        }[group]

    def reference_area(self, ref_area: Optional[float] = None) -> float:
        """Needle surface area [m²] for per-area fluxes (default perimeter × height)."""
        if ref_area is not None:
            return float(ref_area)
        perimeter = self.mecha.network._cells_gdf.geometry.union_all().exterior.length   # µm
        height = float(self.mecha.geometry.maturity_stages[self.exp.anatomy.i_mat].get('height'))
        return perimeter * height * 1e-12

    def _build_transport(self) -> None:
        a, dyn = self.exp.anatomy, self.exp.dynamics
        cap = {'dt': dyn.dt} if isinstance(dyn, TimeStepping) else None
        geom = SoluteGeometry(self.mecha)
        self.st = SoluteTransport(self.mecha, self.exp.solute.diffusion_params(),
                                  cap, mode='full', geometry=geom)
        self.D_mat = self.st.build_diffusion_matrix(a.h_idx, a.i_mat)
        self.mst = None
        if self.exp.starch is not None:
            st_starch = SoluteTransport(
                self.mecha, {'apo_wall': 0.0, 'plasmodesmata': 0.0, 'membrane': 0.0},
                cap, mode='full', mobile=False, geometry=geom)
            self.mst = MultiSoluteTransport(geom, [self.st, st_starch],
                                            self._reaction_matrix(0.0))
        self.node_vols = self.st._compute_node_volumes(a.i_mat)

    def _reaction_matrix(self, t: float) -> np.ndarray:
        # Species [sugar, starch]; K[s, r] = production of s from r [1/d].
        k_f, k_r = self.exp.starch.rates(t)
        return np.array([[-k_f,  k_r],
                         [ k_f, -k_r]])

    def _build_solute_bc(self) -> None:
        """Sucrose Dirichlet set: sinks, BC-fixed values and anchors for any
        disconnected component so the solve is non-singular."""
        bcfg, nwj = self.exp.bc, self.nwj
        D = self.D_mat
        isolated = list(np.where(np.array(np.abs(D).sum(axis=1)).ravel() == 0)[0])
        n_comp, labels = csgraph.connected_components(D, directed=False,
                                                      connection='weak')
        sink_ids = sorted({cid for g in bcfg.sinks for cid in self.tissue_ids(g)})
        anchored = {nwj + c for c in sink_ids} | {nwj + c for c in self.meso_cell_ids}
        iso_set = set(isolated)
        extra = []
        for comp in range(n_comp):
            members = np.where(labels == comp)[0]
            if any(m in anchored for m in members):
                continue
            non_iso = [m for m in members if m not in iso_set]
            if non_iso:
                extra.append(non_iso[0])

        bc = {nid: bcfg.c_sink for nid in isolated + extra}
        bc.update({nwj + cid: bcfg.c_sink for cid in sink_ids})
        bc.update(bcfg.fixed_concentrations(self))
        self.bc = bc

        mask = np.ones(self.n_total, dtype=bool)
        mask[[n for n in bc if 0 <= n < self.n_total]] = False
        self.interior_mask = mask
        self.sink_groups = {g: nwj + np.asarray(self.tissue_ids(g), dtype=int)
                            for g in bcfg.sinks}
        self.sink_nodes = nwj + np.asarray(sink_ids, dtype=int)
        self._log(f"  sucrose Dirichlet nodes: {len(bc)}  sinks={bcfg.sinks}")

    # ---- solver glue used by the coupling options ----------------------------
    def _implicit_step(self, c, baseline, rhs, transport_step, **kw):
        a, s = self.exp.anatomy, self.exp.solute
        return coupled_implicit_step(
            self.mecha, self.st, c_prev=c, T=self.exp.t_kelvin,
            psi_os_baseline=baseline, boundary_conditions=self.bc, rhs=rhs,
            i_scenario=a.i_sce, i_maturity=a.i_mat, h=a.h_idx,
            theta=self.exp.dynamics.theta, operators=s.ops, scheme=s.scheme,
            transport_step=transport_step, **kw)

    def _coupled_steady(self, rhs, **kw):
        a, s = self.exp.anatomy, self.exp.solute
        c, n, ok = coupled_water_solute_solve(
            self.mecha, self.st, T=self.exp.t_kelvin,
            boundary_conditions=self.bc, rhs=rhs,
            i_scenario=a.i_sce, i_maturity=a.i_mat, h=a.h_idx,
            operators=s.ops, scheme=s.scheme,
            continuation_steps=self.exp.dynamics.continuation,
            verbose=self.verbose, **kw)
        return c, n, ok

    def _spatial_operator(self):
        a, s = self.exp.anatomy, self.exp.solute
        return self.st.spatial_operator(h=a.h_idx, i_maturity=a.i_mat, i_scenario=a.i_sce,
                                        operators=s.ops, scheme=s.scheme).tocsr()

    def _step(self, c, c_sta, rhs, baseline):
        """One coupled time step; sugar+starch as one block when starch is on."""
        a, s, ts = self.exp.anatomy, self.exp.solute, self.exp.dynamics
        transport_step, sta = None, [None]
        if c_sta is not None:
            def transport_step():
                c_new, sta[0] = self.mst.solve_coupled(
                    h=a.h_idx, i_maturity=a.i_mat, i_scenario=a.i_sce,
                    c_prev=[c, c_sta], boundary_conditions=[self.bc, None],
                    rhs=[rhs, None], theta=ts.theta, operators=s.ops, scheme=s.scheme)
                return c_new
        c_new, n_iter, ok = self.exp.coupling.step(self, c, baseline, rhs, transport_step)
        return c_new, sta[0], n_iter, ok

    # ---- run -----------------------------------------------------------------
    def run(self) -> SimResult:
        if isinstance(self.exp.dynamics, SteadyState):
            return self._run_steady()
        return self._run_transient()

    def _run_steady(self) -> SimResult:
        self._log(f"=== Steady solve ({type(self.exp.coupling).__name__}) ===")
        c, n_iter, ok = self.exp.coupling.steady(self, self.exp.bc.steady_rhs(self))
        c = np.asarray(c, dtype=float)
        Tc = self._spatial_operator().dot(c)
        q_load = float(np.sum(Tc[self.sink_nodes]))
        extras = {f'j_{g}': np.asarray(float(np.sum(Tc[n])))
                  for g, n in self.sink_groups.items()}
        self._init_edge_record()
        extras.update(self._edge_extras(0, 0.0, c))
        extras.update(self._tissue_extras())
        polys, cell_ids, node_of_cell = self._cell_geometry()
        self._log(f"  converged={ok}  iters={n_iter}  Q_load={q_load*1e12:.2f} pmol/d")
        return SimResult(
            mode='steady', params=self.exp.describe(),
            concentration=c.astype(np.float32),
            cell_polygons=polys, cell_ids=cell_ids, node_of_cell=node_of_cell,
            q_load=q_load, n_iterations=int(n_iter), converged=bool(ok),
            water_flux_total=self._total_transpiration_flux(), extras=extras,
        )

    def _run_transient(self) -> SimResult:
        exp, ts = self.exp, self.exp.dynamics
        nwj, n_cells = self.nwj, self.n_cells
        manager = self.mecha.network.cell_manager

        # Sucrose-free baseline: Ψ_os background and reference transpiration.
        self.mecha.water_flux(h=exp.anatomy.h_idx, verbose=False)
        baseline = np.array([c.psi_os if c.psi_os is not None else 0.0 for c in manager])
        j_w0 = self._total_transpiration_flux()

        c = np.zeros(self.n_total)
        c[self.meso_nodes] = ts.c0_meso
        for nid, val in self.bc.items():
            if 0 <= nid < self.n_total:
                c[nid] = val
        c_sta = None
        if self.mst is not None:
            c_sta = np.zeros(self.n_total)
            c_sta[nwj: nwj + n_cells] = exp.starch.c0

        dir_nodes = np.asarray([n for n in self.bc if 0 <= n < self.n_total], dtype=int)
        xyl_nodes = nwj + np.asarray(self.xylem_cell_ids, dtype=int)

        def sucrose_mass(c):
            return float(np.sum(self.node_vols[self.interior_mask] * c[self.interior_mask]))

        def starch_mass(c_sta):
            return float(np.sum(self.node_vols * c_sta)) if c_sta is not None else 0.0

        mass0 = sucrose_mass(c)
        sta_masses = [starch_mass(c_sta)]
        rec = {k: [0.0] for k in ('j_s', 'q_prod', 'cum_prod', 'cum_export', 'cum_out',
                                  'n_couple')}
        rec.update({f'j_{g}': [0.0] for g in self.sink_groups})
        rec['j_w'] = [j_w0]
        rec['f_drive'] = [exp.bc.f(0.0)]
        rec['g_drive'] = [exp.bc.g(0.0)]
        rec['couple_ok'] = [True]
        rec['c_xylem'] = [float(np.mean(c[xyl_nodes])) if xyl_nodes.size else 0.0]
        target0 = exp.bc.target_flow(self, 0.0)
        if target0 is not None:
            rec['j_w_target'] = [target0]
        if c_sta is not None:
            k_f0, k_r0 = exp.starch.rates(0.0)
            rec.update(m_starch=[float(np.mean(c_sta[nwj: nwj + n_cells]))],
                       k_f=[k_f0], k_r=[k_r0])
        times, frames, masses = [0.0], [c.copy()], [mass0]
        sta_frames = [c_sta.copy()] if c_sta is not None else None
        ratios = []

        self._init_edge_record(ts.dt)
        n_win = max(1, int(round(ts.balance_window / ts.dt)))
        log_every = max(1, ts.max_steps // 20)
        self._log(f"=== Time march: {type(exp.bc).__name__}, "
                  f"{type(exp.coupling).__name__}, starch={type(exp.starch).__name__}, "
                  f"dt={ts.dt:.4f} d, ≤{ts.max_steps} steps ===")
        stop_reason = f"reached max_steps={ts.max_steps}"

        for step in range(1, ts.max_steps + 1):
            t = step * ts.dt
            exp.bc.drive(self, t)
            if self.mst is not None:
                self.mst.K = self._reaction_matrix(t)
            c, rhs, q = exp.bc.source(self, c, t)
            c_new, c_sta_new, n_iter, ok = self._step(c, c_sta, rhs, baseline)
            dmax = float(np.max(np.abs(c_new - c)))
            c = c_new
            if c_sta is not None:
                c_sta = c_sta_new

            Tc = self._spatial_operator().dot(c)
            j_s = float(np.sum(Tc[self.sink_nodes]))
            mass = sucrose_mass(c)
            sta_masses.append(starch_mass(c_sta))
            self._maybe_record_edges(step, t, c)

            times.append(t);  frames.append(c.copy());  masses.append(mass)
            for g, nodes in self.sink_groups.items():
                rec[f'j_{g}'].append(float(np.sum(Tc[nodes])))
            rec['j_s'].append(j_s)
            rec['q_prod'].append(q)
            rec['cum_prod'].append(rec['cum_prod'][-1] + q * ts.dt)
            rec['cum_export'].append(rec['cum_export'][-1] + j_s * ts.dt)
            rec['cum_out'].append(rec['cum_out'][-1] + float(np.sum(Tc[dir_nodes])) * ts.dt)
            rec['n_couple'].append(n_iter)
            rec['couple_ok'].append(bool(ok))
            rec['j_w'].append(self._total_transpiration_flux())
            rec['f_drive'].append(exp.bc.f(t))
            rec['g_drive'].append(exp.bc.g(t))
            rec['c_xylem'].append(float(np.mean(c[xyl_nodes])) if xyl_nodes.size else 0.0)
            if 'j_w_target' in rec:
                rec['j_w_target'].append(exp.bc.target_flow(self, t))
            if c_sta is not None:
                sta_frames.append(c_sta.copy())
                k_f, k_r = exp.starch.rates(t)
                rec['m_starch'].append(float(np.mean(c_sta[nwj: nwj + n_cells])))
                rec['k_f'].append(k_f);  rec['k_r'].append(k_r)
            ratios.append(j_s / q if q > 0 else np.nan)

            if step % log_every == 0 or step == 1:
                self._log(f"  step {step:4d}  t={t:7.3f} d  j_w={rec['j_w'][-1]:.3e}  "
                          f"j_s={j_s*1e12:9.2f} pmol/d  export/prod={ratios[-1]:.4f}  "
                          f"coupling iters={n_iter}{'' if ok else ' (unconverged)'}")

            if ts.sink_fraction is not None and mass0 > 0 \
                    and rec['cum_out'][-1] >= ts.sink_fraction * mass0:
                stop_reason = f"≥{ts.sink_fraction*100:.0f}% of initial mass exported"
                break
            if ts.equil_tol is not None and dmax < ts.equil_tol:
                stop_reason = f"equilibrium (max|Δc| < {ts.equil_tol:g})"
                break
            if ts.balance_tol is not None and step >= n_win and q > 0:
                r = float(np.mean(ratios[-n_win:]))
                if abs(r - 1.0) < ts.balance_tol:
                    stop_reason = (f"steady state (|⟨export/production⟩_{ts.balance_window:g} d"
                                   f" − 1| < {ts.balance_tol})")
                    break

        self._log(f"  Stopped after {len(frames)-1} step(s): {stop_reason}")

        extras = {k: np.asarray(v) for k, v in rec.items()}
        cum_prod, cum_exp = extras['cum_prod'], extras['cum_export']
        with np.errstate(divide='ignore', invalid='ignore'):
            extras['ratio_inst'] = np.where(extras['q_prod'] > 0,
                                            extras['j_s'] / extras['q_prod'], np.nan)
            extras['ratio_cum'] = np.where(cum_prod > 0, cum_exp / cum_prod, np.nan)
        # Exact up to solver round-off: Δmass = production − outflow at all Dirichlet nodes.
        stored = np.asarray(masses) + np.asarray(sta_masses)
        extras['mass_balance'] = stored - stored[0] - (cum_prod - extras['cum_out'])
        if self.ref_area is not None:
            extras['ref_area'] = np.asarray(self.ref_area)
        extras.update(self._tissue_extras())
        extras.update(self._edge_extras(len(times) - 1, times[-1], c))

        polys, cell_ids, node_of_cell = self._cell_geometry()
        return SimResult(
            mode='dynamic', params=exp.describe(),
            concentration=np.asarray(frames, dtype=np.float32),
            concentration_starch=(np.asarray(sta_frames, dtype=np.float32)
                                  if sta_frames is not None else None),
            cell_polygons=polys, cell_ids=cell_ids, node_of_cell=node_of_cell,
            times=np.asarray(times), interior_mass=np.asarray(masses),
            frac_to_sink=extras['cum_out'], sink_label='to sink', sink_unit='pmol',
            sink_scale=1.0e12, stop_reason=stop_reason,
            water_flux_total=j_w0, extras=extras,
        )

    # ---- per-link snapshots (apo / membrane / PD): solute, water, Péclet -----
    def _edge_flux_scaffold(self):
        """Solute-carrying edges: matrix indices (E,2), segments (E,2,2), path code (E,)."""
        pos = self.mecha.network.graph.nodes
        air = self.st._air_nodes
        ij, seg, code = [], [], []
        for u, v, eattr in self.mecha.network.graph.edges(data=True):
            path = eattr.get('path')
            if path not in EDGE_PATHS:
                continue
            iu, iv = self.mecha.indice[u], self.mecha.indice[v]
            if iu == iv or iu in air or iv in air:
                continue
            ij.append((iu, iv))
            seg.append([pos[u].get('position', (0.0, 0.0)),
                        pos[v].get('position', (0.0, 0.0))])
            code.append(EDGE_PATHS.index(path))
        return (np.asarray(ij, dtype=int).reshape(-1, 2),
                np.asarray(seg, dtype=float).reshape(-1, 2, 2),
                np.asarray(code, dtype=np.int8))

    def _init_edge_record(self, dt: Optional[float] = None) -> None:
        """Snapshot links every ``exp.edge_snapshot_dt`` days (``dt`` = step)."""
        self._edge_ij, self._edge_seg, self._edge_code = self._edge_flux_scaffold()
        snap = self.exp.edge_snapshot_dt
        self._edge_stride = (1 if dt is None or not snap
                             else max(1, int(round(snap / dt))))
        self._edge_rec = {'t': [], 'step': [], 'solute': [], 'water': [], 'peclet': []}

    def _record_edges(self, step: int, t: float, c: np.ndarray) -> None:
        a = self.exp.anatomy
        i, j = self._edge_ij[:, 0], self._edge_ij[:, 1]
        T = self._spatial_operator()
        A = self.st.build_advection_matrix(a.i_mat, a.i_sce).tocsr()
        D = self.D_mat.tocsr()
        # T is antisymmetric per edge, so net u→v flux = T[v,u]c_u − T[u,v]c_v.
        solute = np.asarray(T[j, i]).ravel() * c[i] - np.asarray(T[i, j]).ravel() * c[j]
        carry = np.asarray(A[j, i]).ravel() - np.asarray(A[i, j]).ravel()   # (1−σ)·F, u→v
        d_e = np.asarray(D[i, j]).ravel()
        with np.errstate(divide='ignore', invalid='ignore'):
            peclet = np.where(d_e > 0.0, carry / d_e, np.nan)
        flux = {}
        for e in self.mecha.edge_flux_list[a.i_mat][a.i_sce]:
            key = (e['source'], e['target'])
            flux[key] = flux.get(key, 0.0) + e['flux']
        water = np.array([flux.get((u, v), 0.0) - flux.get((v, u), 0.0)
                          for u, v in self._edge_ij])
        rec = self._edge_rec
        rec['t'].append(float(t)); rec['step'].append(int(step))
        rec['solute'].append(solute.astype(np.float32))
        rec['water'].append(water.astype(np.float32))
        rec['peclet'].append(peclet.astype(np.float32))

    def _maybe_record_edges(self, step: int, t: float, c: np.ndarray) -> None:
        if step % self._edge_stride == 0:
            self._record_edges(step, t, c)

    def _edge_extras(self, step: int, t: float, c: np.ndarray) -> dict:
        """Close the record (always include the final state) and return extras."""
        rec = self._edge_rec
        if not rec['step'] or rec['step'][-1] != step:
            self._record_edges(step, t, c)
        return {
            'edge_segments': self._edge_seg,              # (E,2,2) µm, u→v
            'edge_path':     self._edge_code,             # (E,) index into EDGE_PATHS
            'edge_t':        np.asarray(rec['t']),        # (n_snap,) d
            'edge_solute':   np.asarray(rec['solute']),   # (n_snap,E) mol/d, u→v
            'edge_water':    np.asarray(rec['water']),    # (n_snap,E) cm³/d, u→v
            'edge_peclet':   np.asarray(rec['peclet']),   # (n_snap,E) signed (1−σ)F/D_e
        }

    def _tissue_extras(self) -> dict:
        """Tissue cell-id arrays so the anatomy map can be rendered offline."""
        return {
            'tissue_meso':        np.asarray(self.meso_cell_ids, dtype=int),
            'tissue_strasburger': np.asarray(self.strasburger_cell_ids, dtype=int),
            'tissue_phloem':      np.asarray(self.phloem_cell_ids, dtype=int),
            'tissue_xylem':       np.asarray(self.xylem_cell_ids, dtype=int),
            'tissue_airspace':    np.asarray(self.airspace_cell_ids, dtype=int),
        }

    def _total_transpiration_flux(self) -> float:
        """Net outward water flux [cm³/d] over all ``wall_air`` edges (wall→air)."""
        cm = self.mecha.network.cell_manager
        total = 0.0
        for u, v, eattr in self.mecha.network.graph.edges(data=True):
            if eattr.get('path') != 'wall_air':
                continue
            Q = eattr.get('Q')
            if Q is None:
                continue
            wall_is_u = cm.get_wall_by_node_id(u) is not None
            total += float(Q) if wall_is_u else -float(Q)
        return total

    def _cell_geometry(self):
        """Cell polygons (WKB) + full-network node index per cell id."""
        gdf = self.mecha.network._cells_gdf
        cell_ids = gdf['id_cell'].to_numpy()
        polys = np.array([g.wkb for g in gdf.geometry], dtype=object)
        node_of_cell = np.array(
            [self.mecha.indice[self.nwj + int(cid)] for cid in cell_ids])
        return polys, cell_ids, node_of_cell


# ══════════════════════════════════════════════════════════════════════════════
# Parallel driver
# ══════════════════════════════════════════════════════════════════════════════
def _run_one(exp: Experiment) -> SimResult:
    """Picklable worker: build the simulation from the cached anatomy and run it."""
    return NeedleSimulation(exp, get_geometry_base(exp.anatomy.seed), verbose=False).run()


def _pin_blas_single_thread():
    """Pool initializer: one BLAS thread per worker (avoids N×K oversubscription)."""
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[var] = '1'


def run_experiments(experiments, max_workers: Optional[int] = None):
    """Run independent experiments (process pool; inline for one job/worker)."""
    from concurrent.futures import ProcessPoolExecutor

    n_jobs = len(experiments)
    if n_jobs == 0:
        return []
    workers = (min(os.cpu_count() or 1, n_jobs) if max_workers is None
               else max(1, int(max_workers)))
    if workers == 1 or n_jobs == 1:
        return [_run_one(e) for e in experiments]
    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_pin_blas_single_thread) as pool:
        return list(pool.map(_run_one, experiments))


# ══════════════════════════════════════════════════════════════════════════════
# Demo: impulse redistribution at two PD diffusivities
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    from post_sim import mass_balance_plot, concentration_gif

    base = Experiment(
        anatomy=Anatomy(seed=42),
        bc=DirichletBC(psi_xyl=-200.0, air='psi', psi_air=-1.0e4),
        dynamics=TimeStepping(dt=0.1, max_steps=40, c0_meso=50.0e-6, sink_fraction=0.9),
        coupling=JFNK(),
    )
    experiments = [
        base.replace(label='dyn_lowPD',  solute=Solute(d_pd=1e-1)),
        base.replace(label='dyn_highPD', solute=Solute(d_pd=5e-1)),
    ]
    for res in run_experiments(experiments, max_workers=2):
        print(res.summary())
        stem = os.path.join(_OUT_DIR, f"needle_{res.params['label']}")
        res.save(stem + '.npz')
        mass_balance_plot(res, stem + '_massbalance.png')
        concentration_gif(res, stem + '_concentration.gif')
