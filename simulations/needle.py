"""
needle.py
=========
Design
------
``GeometryBase``
    Owns the single, expensive, *stochastic* ``NeedleAnatomy`` draw.  Built once
    and shared by every simulation that uses the same seed; per-worker cached so
    parallel processes rebuild it deterministically instead of pickling the live
    (unpicklable) network object.

``SimParams`` / ``SteadyParams`` / ``DynamicParams``
    Immutable configuration.

``NeedleSimulation`` (abstract) → ``NeedleSteadySimulation`` /
``NeedleDynamicSimulation``
    Build a private ``Mecha`` scaffold from the shared geometry, apply the
    experiment-specific parameters, and ``run()`` to produce a ``SimResult``.

``SimResult``
    Plain-array container (fully picklable, ``float32`` frames) returned by
    ``run()``.  Holds the raw data only; allows for GIF rendering as a *post-processing*
    step.
    
Units: lengths µm/cm, volumes cm³, conc mol/cm³ (=10³ M), rates mol/d, time d.
"""

from __future__ import annotations

import os
import sys
import copy
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import scipy.sparse as sp
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
from openalea.mecha.utils.coupled_solver import coupled_water_solute_solve
from openalea.mecha.calibration.forward_model import rh_to_water_potential

# Gas constant used by MECHA for van't Hoff Ψ_os = −R·T·c  [hPa cm³ mol⁻¹ K⁻¹].
_R_HPA = 8.314e4


# ══════════════════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class SimParams:
    """Configuration common to steady and dynamic needle simulations."""

    # ---- Water boundary conditions (Dirichlet) -------------------------------
    psi_xyl: float = -200.0        # hPa  xylem water potential (supply)
    air_mode: str = 'psi'          # 'psi' | 'rh'
    psi_atm: float = -1.0e4        # hPa  air-space water potential (air_mode='psi')
    rh_air: float = 0.99           # -    air-space RH in (0, 1]  (air_mode='rh')

    # ---- Solute diffusivities / membrane reflection (PER-INSTANCE) -----------
    d_pd: float = 5.0e-1           # cm²/d  plasmodesmatal diffusivity
    d_apo: float = 0.1             # cm²/d  apoplastic wall diffusivity
    d_mem: float = 1.0e-6          # cm²/d  passive transmembrane diffusivity
    sigma_sucrose: float = 0.7     # -      membrane reflection coefficient

    # ---- Physical constant ---------------------------------------------------
    t_kelvin: float = 298.15       # K   (van't Hoff Ψ_os = −R·T·c)

    # ---- Transport operator / solver (shared; safe defaults) -----------------
    ops: str = 'T'                 # 'D' | 'T'  (T = advection + diffusion)
    scheme: str = 'sg'             # 'upwind' | 'sg' (Scharfetter–Gummel, Pe-stable)
    method: str = 'jfnk'           # 'picard' | 'jfnk'
    jfnk_maxiter: int = 25
    jfnk_inner: int = 120
    jfnk_line_search: str = 'armijo'
    jfnk_rdiff: float = 1.0e-3

    # ---- Sink concentration (shared) -----------------------------------------
    c_sink: float = 0.0            # mol/cm³  Dirichlet sink (phloem + xylem)

    # ---- Indices -------------------------------------------------------------
    h_idx: int = 0
    i_mat: int = 0
    i_sce: int = 1

    # ---- Anatomy -------------------------------------------------------------
    seed: Optional[int] = None     # None → random; set an int for reproducibility

    # ---- Bookkeeping ---------------------------------------------------------
    label: str = 'needle'          # used for output filenames

    def diffusion_params(self) -> dict:
        """Build the SoluteTransport diffusion_params dict from this config."""
        return dict(
            apo_wall=self.d_apo,
            membrane=self.d_mem,
            plasmodesmata=self.d_pd,
            sigma={cg: self.sigma_sucrose for cg in range(1, 20)},
        )

    # Each params child names the simulation class that runs it (resolved lazily
    # by name since the sim classes are defined later).  This is the invariant
    # backbone: a new series adds a child + names its sim here, and _run_one /
    # run_experiments never change.
    _simulation_class_name: str = field(default='NeedleSimulation',
                                        init=False, repr=False, compare=False)

    def simulation_class(self) -> type:
        cls = globals().get(self._simulation_class_name)
        if cls is None:
            raise ValueError(
                f"Unknown simulation class {self._simulation_class_name!r} for "
                f"{type(self).__name__}.")
        return cls


@dataclass(frozen=True)
class SteadyParams(SimParams):
    """Steady-state (fixed-input) sucrose transport."""

    _simulation_class_name: str = field(default='NeedleSteadySimulation',
                                        init=False, repr=False, compare=False)
    c_meso: float = 50.0e-6        # mol/cm³  mesophyll concentration
    # True (default): mesophyll held at c_meso as a Dirichlet BC (constant source).
    # False: c_meso seeds the initial field only; mesophyll is free to redistribute.
    meso_as_source: bool = True
    couple_tol: float = 10.0       # hPa   convergence tol on max|Δψ_total|
    couple_maxiter: int = 25
    continuation: Optional[tuple] = None   # homotopy λ schedule (None → single stage)


@dataclass(frozen=True)
class DynamicParams(SimParams):
    """Transient (time-series) sucrose transport."""

    _simulation_class_name: str = field(default='NeedleDynamicSimulation',
                                        init=False, repr=False, compare=False)
    c_pulse: float = 50.0e-6       # mol/cm³  initial mesophyll concentration
    # False (default): c_pulse is a free initial impulse; mesophyll is released at t>0.
    # True: mesophyll is held at c_pulse as a Dirichlet BC (constant source term).
    meso_as_source: bool = False
    dt: float = 1.0e-1             # d   time step (implicit Euler)
    theta: float = 1.0             # 1.0 implicit Euler | 0.5 Crank–Nicolson
    max_steps: int = 200
    equil_tol_frac: float = 1.0e-6  # equilibrium when max|Δc| < this × c_pulse
    sink_fraction: float = 0.90    # stop once ≥ this fraction has left the interior
    couple_maxiter: int = 12       # inner hydraulic-coupling iterations per step

    # ---- Starch (immobile, reaction-coupled second solute) -------------------
    include_starch: bool = False   # activate the coupled multisolute pathway
    k_starch_syn: float = 1.0      # 1/d  sugar → starch synthesis rate
    k_starch_deg: float = 0.5      # 1/d  starch → sugar remobilization rate
    c_starch0: float = 0.0         # mol/cm³  initial starch (cells) at t=0


# ══════════════════════════════════════════════════════════════════════════════
# Shared geometry (expensive; built once, reused by many simulations)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class GeometryBase:
    """The single stochastic anatomy draw shared across experiments.

    ``NeedleAnatomy`` is expensive and stochastic; with a fixed ``seed`` it is
    fully deterministic, so parallel workers can rebuild it from the seed rather
    than pickling the live (unpicklable) network object.
    """

    seed: Optional[int]
    anatomy: NeedleAnatomy

    @classmethod
    def build(cls, seed: Optional[int] = None) -> "GeometryBase":
        anatomy = NeedleAnatomy(seed=seed) if seed is not None else NeedleAnatomy()
        anatomy.export_to_adjencymatrix()
        return cls(seed=seed, anatomy=anatomy)


# Per-process cache so a parallel worker builds the anatomy exactly once and
# reuses it for every experiment it is assigned (keyed by seed).
_BASE_CACHE: dict = {}


def get_geometry_base(seed: Optional[int]) -> GeometryBase:
    """Return a cached ``GeometryBase`` for ``seed`` (build on first use)."""
    if seed not in _BASE_CACHE:
        _BASE_CACHE[seed] = GeometryBase.build(seed)
    return _BASE_CACHE[seed]


# ══════════════════════════════════════════════════════════════════════════════
# Result container (raw data only; GIF rendering is a post-processing step)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class SimResult:
    """Plain-array, picklable result of a single simulation run.

    Frames are stored as ``float32`` to halve the memory / disk footprint; the
    concentration and Péclet arrays are 1-D (steady) or 2-D ``(n_steps, N)``
    (dynamic).  GIFs are NOT stored here — render them on demand from this
    object via ``concentration_gif`` / ``peclet_gif``.
    """

    mode: str                                  # 'steady' | 'dynamic'
    params: dict                               # asdict(SimParams) — picklable
    concentration: np.ndarray                  # float32 (N,) or (n_steps, N)

    # Per-bond local Péclet, aligned with ``pe_segments`` / ``pe_D``.
    peclet: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    pe_segments: np.ndarray = field(default_factory=lambda: np.empty((0, 2, 2)))

    # Cell-polygon geometry for rendering without a live Mecha (WKB or None).
    cell_polygons: Optional[object] = None
    cell_ids: Optional[np.ndarray] = None
    node_of_cell: Optional[np.ndarray] = None  # full-network index per cell id

    # ---- Dynamic-only --------------------------------------------------------
    times: Optional[np.ndarray] = None
    interior_mass: Optional[np.ndarray] = None
    frac_to_sink: Optional[np.ndarray] = None
    stop_reason: Optional[str] = None

    # Immobile starch field (dynamic multisolute run), same shape as
    # ``concentration``; None when include_starch is False.
    concentration_starch: Optional[np.ndarray] = None

    # Total transpiration water flux [cm³/d] out through the evaporating
    # (wall_air) walls at the pulse-free baseline hydraulic solve.
    water_flux_total: Optional[float] = None

    # ---- Steady-only ---------------------------------------------------------
    q_load: Optional[float] = None
    n_iterations: Optional[int] = None
    converged: Optional[bool] = None

    # ---- Open-ended, series-specific outputs --------------------------------
    # First-class extensible slot: any array a series produces that has no named
    # field goes here by key (e.g. 'j_w', 'j_s', 'm_starch').  Pickled with the
    # object AND round-tripped by save/load (each key stored as 'extra__<key>'),
    # so future series add outputs WITHOUT editing SimResult.
    extras: dict = field(default_factory=dict)

    # ---- Convenience read-only views over extras (established diurnal keys) --
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
        """Return a series-specific output array by name (None/default if absent)."""
        return self.extras.get(name, default)

    # ─────────────────────────────────────────────────────────────────────────
    def save(self, path: str) -> str:
        """Persist to a compressed ``.npz`` (params stored as a JSON sidecar)."""
        import json
        arrays = dict(
            mode=self.mode,
            concentration=self.concentration,
            peclet=self.peclet,
            pe_segments=self.pe_segments,
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
        # Series-specific extras: one array per key under an 'extra__' prefix.
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
            mode = str(z['mode'])
            is_dyn = mode == 'dynamic'
            times = z['times'] if z['times'].size else None
            res = cls(
                mode=mode,
                params={},
                concentration=z['concentration'],
                peclet=z['peclet'],
                pe_segments=z['pe_segments'],
                cell_polygons=(z['cell_polygons'] if z['cell_polygons'].size else None),
                cell_ids=(z['cell_ids'] if z['cell_ids'].size else None),
                node_of_cell=(z['node_of_cell'] if z['node_of_cell'].size else None),
                times=times,
                interior_mass=(z['interior_mass'] if z['interior_mass'].size else None),
                frac_to_sink=(z['frac_to_sink'] if z['frac_to_sink'].size else None),
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

    # ─────────────────────────────────────────────────────────────────────────
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
            out.update(
                q_load=self.q_load,
                n_iterations=self.n_iterations,
                converged=self.converged,
            )
        return out

    # ---- Rendering helpers (post-processing) --------------------------------
    def _cell_gdf(self):
        """Rebuild a GeoDataFrame of cell polygons from stored geometry."""
        import geopandas as gpd
        from shapely import wkb
        if self.cell_polygons is None:
            raise ValueError("No stored cell geometry; render with a live Mecha "
                             "via NeedleSimulation.concentration_gif instead.")
        geoms = [wkb.loads(b) for b in self.cell_polygons]
        return gpd.GeoDataFrame({'id_cell': self.cell_ids}, geometry=geoms)

    def _field_at(self, step: Optional[int]) -> np.ndarray:
        """Return the concentration field (1-D) for a step (dynamic) or steady."""
        if self.concentration.ndim == 1:
            return self.concentration
        return self.concentration[-1 if step is None else step]

    def concentration_gif(self, path: str, max_frames: int = 150, fps: int = 12,
                          cmap: str = 'plasma') -> str:
        """Render the transient concentration field as an animated GIF.

        Post-processing step: heavy render memory is only used here, one
        simulation at a time, decoupled from the (parallel) march.
        """
        if self.mode != 'dynamic':
            raise ValueError("concentration_gif is only defined for dynamic runs.")
        return _render_cell_field_gif(
            self._cell_gdf(), self.concentration, self.node_of_cell,
            self.times, self.frac_to_sink, path,
            max_frames=max_frames, fps=fps, cmap=cmap,
            title='Needle sucrose transport — impulse release',
            cbar_label='c (µM sucrose)', scale=1e6,
        )

    def peclet_gif(self, path: str, max_frames: int = 150, fps: int = 12) -> str:
        """Render the per-bond local Péclet field as an animated GIF."""
        if self.mode != 'dynamic':
            raise ValueError("peclet_gif is only defined for dynamic runs.")
        return _render_peclet_gif(
            self.pe_segments, self.peclet, self.times, path,
            max_frames=max_frames, fps=fps,
        )

    def mass_balance_plot(self, path: str) -> str:
        """Interior mass / fraction-to-sink vs. time (dynamic only)."""
        if self.mode != 'dynamic':
            raise ValueError("mass_balance_plot is only defined for dynamic runs.")
        return _render_mass_balance(self.times, self.interior_mass,
                                    self.frac_to_sink, path)


# ══════════════════════════════════════════════════════════════════════════════
# Simulation base class
# ══════════════════════════════════════════════════════════════════════════════
class NeedleSimulation:
    """Abstract base: build a private MECHA scaffold from shared geometry.

    Subclasses implement ``_build_st`` (the SoluteTransport configuration that
    encodes the simulation TYPE) and ``run`` (the stopping criterion).
    """

    mode = 'abstract'

    def __init__(self, params: SimParams,
                 geometry: Optional[GeometryBase] = None,
                 verbose: bool = True):
        if type(self) is NeedleSimulation:
            raise TypeError("NeedleSimulation is abstract; use a subclass.")
        self.params = params
        self.verbose = verbose
        self.geometry = geometry or get_geometry_base(params.seed)

        self._build_network()
        self._classify_tissue()
        self._resolve_air_bc()
        self._build_st()          # subclass: sets self.st (+ self.D_mat)
        self._build_bc()

    # ---- logging -------------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # ---- shared setup --------------------------------------------------------
    def _build_network(self) -> None:
        p = self.params
        self._log(f"=== Building MECHA network (label={p.label!r}) ===")
        network = NetworkBuilder(self.geometry.anatomy)
        network.populate_from_network()

        data = InData.needle_defaults()
        # Scenario I_SCE with the osmotic operator ON (s_factor = σ). Seed from a
        # FULL default scenario so the structural osmotic-profile fields are valid
        # numbers (not NaN), then zero the osmotic MAGNITUDES so the only osmotic
        # term is the dynamic van't Hoff sucrose contribution.
        s1 = copy.deepcopy(BoundaryData().scenarios[0])
        s1['s_factor']           = p.sigma_sucrose
        data.boundary.add_scenario(s1)

        self.mecha = Mecha(data, network=network)
        self.nwj     = self.mecha.network.n_wall_junction
        self.n_cells = self.mecha.network.n_cells
        self.n_total = self.nwj + self.n_cells

        self.mecha.psi_xyl[1, p.i_mat, p.i_sce]   = p.psi_xyl
        self.mecha.psi_sieve[1, p.i_mat, p.i_sce] = np.nan
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
        self.meso_cell_ids       = meso
        self.strasburger_cell_ids = strasburger
        self.phloem_cell_ids     = phloem
        self.xylem_cell_ids      = xylem
        self.airspace_cell_ids   = airspace
        self.phloem_loading_ids  = strasburger + phloem
        self._log(f"  mesophyll={len(meso)}  phloem-loading={len(self.phloem_loading_ids)}"
                  f"  xylem={len(xylem)}  air={len(airspace)}")

    def _resolve_air_bc(self) -> None:
        p = self.params
        if p.air_mode == 'rh':
            self.psi_air = rh_to_water_potential(p.rh_air, p.t_kelvin)
        elif p.air_mode == 'psi':
            self.psi_air = p.psi_atm
        else:
            raise ValueError(f"air_mode must be 'psi' or 'rh', got {p.air_mode!r}.")
        self.mecha.set_air_wall_bc(self.psi_air)
        self._log(f"  Air BC Ψ_air={self.psi_air:.1f} hPa ({p.air_mode})")

    def _build_bc(self) -> None:
        """Dirichlet BC dict: sink at phloem-loading + xylem, plus anchors for
        any disconnected component so the implicit solve is non-singular."""
        p = self.params
        nwj = self.nwj
        D = self.D_mat
        row_norms = np.array(np.abs(D).sum(axis=1)).ravel()
        isolated = list(np.where(row_norms == 0)[0])
        n_comp, labels = csgraph.connected_components(D, directed=False,
                                                      connection='weak')
        anchored = set([nwj + c for c in self.phloem_loading_ids]
                       + [nwj + c for c in self.xylem_cell_ids]
                       + [nwj + c for c in self.meso_cell_ids])
        iso_set = set(isolated)
        extra = []
        for comp in range(n_comp):
            members = np.where(labels == comp)[0]
            if any(m in anchored for m in members):
                continue
            non_iso = [m for m in members if m not in iso_set]
            if non_iso:
                extra.append(non_iso[0])

        bc = {}
        for nid in isolated:
            bc[nid] = p.c_sink
        for nid in extra:
            bc[nid] = p.c_sink
        for cid in self.phloem_loading_ids:
            bc[nwj + cid] = p.c_sink
        for cid in self.xylem_cell_ids:
            bc[nwj + cid] = p.c_sink
        self._apply_source_bc(bc)         # subclass adds its source term (if any)
        self.bc = bc

        # Node volumes + interior mask (mass budget over non-Dirichlet nodes).
        self.node_vols = self.st._compute_node_volumes(p.i_mat)
        mask = np.ones(self.n_total, dtype=bool)
        for nid in bc:
            if 0 <= nid < self.n_total:
                mask[nid] = False
        self.interior_mask = mask
        self._log(f"  Dirichlet sink/anchor nodes: {len(bc)}")

    def _apply_source_bc(self, bc: dict) -> None:
        """Hook: add mesophyll Dirichlet source when meso_as_source=True."""
        pass

    # ---- per-bond Péclet scaffold (geometry-only; A refreshed per step) ------
    def _peclet_scaffold(self):
        D_csr = self.D_mat.tocsr()
        pos = self.mecha.network.graph.nodes
        segments, edge_ij, d_edges = [], [], []
        for u, v in self.mecha.network.graph.edges():
            iu, iv = self.mecha.indice[u], self.mecha.indice[v]
            d_edge = abs(float(D_csr[iv, iu]))
            if d_edge <= 0.0:
                continue
            pu = pos[u].get('position', (0.0, 0.0))
            pv = pos[v].get('position', (0.0, 0.0))
            segments.append([pu, pv])
            edge_ij.append((iv, iu))
            d_edges.append(d_edge)
        seg = np.array(segments) if segments else np.empty((0, 2, 2))
        return seg, edge_ij, np.array(d_edges) if d_edges else np.empty(0)

    def _total_transpiration_flux(self) -> float:
        """Total water flux [cm³/d] leaving through the evaporating walls.

        Sums the signed edge flux ``Q`` (stored by ``compute_edge_flows``) over
        every ``wall_air`` edge, oriented wall→air, so a positive value is net
        outward transpiration.  ``water_flux`` must have been solved first.
        Zero for anatomies without ``wall_air`` edges (e.g. roots).
        """
        cm = self.mecha.network.cell_manager
        total = 0.0
        for u, v, eattr in self.mecha.network.graph.edges(data=True):
            if eattr.get('path') != 'wall_air':
                continue
            Q = eattr.get('Q')
            if Q is None:
                continue
            # Q is stored with the u→v convention; reorient to wall→air so the
            # sum is unambiguously outward transpiration.
            wall_is_u = cm.get_wall_by_node_id(u) is not None
            total += float(Q) if wall_is_u else -float(Q)
        return total

    def _cell_geometry(self):
        """Serialize cell polygons (WKB) + full-network node index per cell id,
        so a SimResult can render without the live Mecha object."""
        gdf = self.mecha.network._cells_gdf
        cell_ids = gdf['id_cell'].to_numpy()
        polys = np.array([g.wkb for g in gdf.geometry], dtype=object)
        node_of_cell = np.array(
            [self.mecha.indice[self.nwj + int(cid)] for cid in cell_ids])
        return polys, cell_ids, node_of_cell

    # ---- abstract ------------------------------------------------------------
    def _build_st(self) -> None:
        raise NotImplementedError

    def run(self) -> SimResult:
        raise NotImplementedError

    # ---- live-object rendering (optional; SimResult can also self-render) ----
    def concentration_gif(self, result: SimResult, path: str, **kw) -> str:
        gdf = self.mecha.network._cells_gdf.copy()
        node_of_cell = np.array(
            [self.mecha.indice[self.nwj + int(c)] for c in gdf['id_cell']])
        return _render_cell_field_gif(
            gdf, result.concentration, node_of_cell, result.times,
            result.frac_to_sink, path,
            title='Needle sucrose transport — impulse release',
            cbar_label='c (µM sucrose)', scale=1e6, **kw)


# ══════════════════════════════════════════════════════════════════════════════
# Steady-state simulation
# ══════════════════════════════════════════════════════════════════════════════
class NeedleSteadySimulation(NeedleSimulation):
    """Fixed-input steady-state transport: mesophyll held at ``c_meso``, sink at
    ``c_sink``; solved as a coupled fixed point (Δψ_total < tol)."""

    mode = 'steady'

    def _build_st(self) -> None:
        self._log("=== SoluteTransport (steady, no capacitance) ===")
        self.st = SoluteTransport(self.mecha, self.params.diffusion_params(),
                                  mode='full')
        self.D_mat = self.st.build_diffusion_matrix(self.params.h_idx,
                                                    self.params.i_mat)

    def _apply_source_bc(self, bc: dict) -> None:
        if self.params.meso_as_source:
            for cid in self.meso_cell_ids:
                bc[self.nwj + cid] = self.params.c_meso

    def run(self) -> SimResult:
        p = self.params
        self._log("=== Coupled steady solve ===")
        c, iters, converged = coupled_water_solute_solve(
            self.mecha, self.st, T=p.t_kelvin,
            boundary_conditions=self.bc,
            i_scenario=p.i_sce, i_maturity=p.i_mat, h=p.h_idx,
            tol=p.couple_tol, max_iter=p.couple_maxiter,
            operators=p.ops, scheme=p.scheme, method=p.method,
            jfnk_maxiter=p.jfnk_maxiter, jfnk_inner_maxiter=p.jfnk_inner,
            jfnk_line_search=p.jfnk_line_search, jfnk_rdiff=p.jfnk_rdiff,
            continuation_steps=p.continuation, verbose=self.verbose,
        )
        c = np.asarray(c, dtype=float)

        # Loading rate into the phloem complex (diffusive + advective).
        q_load = None
        if self.phloem_loading_ids:
            idx = self.nwj + np.array(self.phloem_loading_ids)
            A = (self.st.build_transport_operator(p.h_idx, p.i_mat, p.i_sce,
                                                  scheme='sg') - self.D_mat
                 if p.scheme == 'sg'
                 else self.st.build_advection_matrix(p.i_mat, p.i_sce))
            q_load = float(np.sum((self.D_mat + A).dot(c)[idx]))

        seg, edge_ij, d_edges = self._peclet_scaffold()
        peclet = self._peclet_from_flow(seg, edge_ij, d_edges)
        polys, cell_ids, node_of_cell = self._cell_geometry()

        self._log(f"  converged={converged}  iters={iters}  "
                  f"Q_load={None if q_load is None else q_load*1e12:.2f} pmol/d")
        return SimResult(
            mode='steady', params=asdict(p),
            concentration=c.astype(np.float32),
            peclet=peclet.astype(np.float32), pe_segments=seg,
            cell_polygons=polys, cell_ids=cell_ids, node_of_cell=node_of_cell,
            q_load=q_load, n_iterations=int(iters), converged=bool(converged),
        )

    def _peclet_from_flow(self, seg, edge_ij, d_edges):
        if not edge_ij:
            return np.empty(0)
        A = self.st.build_advection_matrix(self.params.i_mat, self.params.i_sce).tocsr()
        a_edge = np.abs(np.array([A[iv, iu] for (iv, iu) in edge_ij]))
        return a_edge / d_edges


# ══════════════════════════════════════════════════════════════════════════════
# Dynamic (transient) simulation
# ══════════════════════════════════════════════════════════════════════════════
class NeedleDynamicSimulation(NeedleSimulation):
    """Transient transport: an initial mesophyll impulse ``c_pulse`` redistributes
    while the phloem-loading complex and xylem act as a fixed sink; marched with
    implicit Euler and full water–solute coupling at every step."""

    mode = 'dynamic'

    def _build_st(self) -> None:
        self._log("=== SoluteTransport (dynamic, capacitance C/dt) ===")
        cap = {'dt': self.params.dt}
        geom = SoluteGeometry(self.mecha)
        # Sugar: the mobile primary solute (existing behaviour).
        self.st = SoluteTransport(self.mecha, self.params.diffusion_params(),
                                  cap, mode='full', geometry=geom)
        self.D_mat = self.st.build_diffusion_matrix(self.params.h_idx,
                                                    self.params.i_mat)

        # Optional immobile starch, coupled to sugar via a reversible reaction.
        # Species order: [sugar, starch]; K[s,r] = production of s from r [1/d].
        #   d(sugar)/dt  = -k_syn·sugar + k_deg·starch
        #   d(starch)/dt = +k_syn·sugar - k_deg·starch
        self.mst = None
        if self.params.include_starch:
            self._log("=== + immobile starch (coupled multisolute) ===")
            st_starch = SoluteTransport(
                self.mecha,
                {'apo_wall': 0.0, 'plasmodesmata': 0.0, 'membrane': 0.0},
                cap, mode='full', mobile=False, geometry=geom)
            k_syn, k_deg = self.params.k_starch_syn, self.params.k_starch_deg
            K = np.array([[-k_syn,  k_deg],
                          [ k_syn, -k_deg]])
            self.mst = MultiSoluteTransport(geom, [self.st, st_starch], K)

    def _apply_source_bc(self, bc: dict) -> None:
        # meso_as_source=True: mesophyll held at c_pulse throughout the march.
        if self.params.meso_as_source:
            for cid in self.meso_cell_ids:
                bc[self.nwj + cid] = self.params.c_pulse

    def run(self) -> SimResult:
        p = self.params
        nwj, n_cells = self.nwj, self.n_cells
        manager = self.mecha.network.cell_manager
        equil_tol = p.equil_tol_frac * p.c_pulse

        # Baseline Ψ_os from a pulse-FREE solve (so van't Hoff is added, not doubled).
        self.mecha.water_flux(h=p.h_idx, verbose=False)
        psi_os_baseline = np.array(
            [c.psi_os if c.psi_os is not None else 0.0 for c in manager])

        # Total transpiration flux at the (sucrose-free) baseline: driven purely
        # by the air/xylem water-potential BCs, so independent of d_pd.
        water_flux_total = self._total_transpiration_flux()

        # Initial impulse field + Dirichlet values at t=0.
        c = np.zeros(self.n_total)
        for cid in self.meso_cell_ids:
            c[nwj + cid] = p.c_pulse
        for nid, val in self.bc.items():
            if 0 <= nid < self.n_total:
                c[nid] = val

        # Immobile starch field (cells only): no transport, no Dirichlet BCs, so
        # it is free to accumulate/deplete via the sugar↔starch reaction alone.
        use_starch = self.mst is not None
        c_starch = np.zeros(self.n_total)
        if use_starch and p.c_starch0 != 0.0:
            for cid in range(n_cells):
                c_starch[nwj + cid] = p.c_starch0
        starch_frames = [c_starch.copy()] if use_starch else None

        mass0 = float(np.sum(self.node_vols[self.interior_mask]
                             * c[self.interior_mask]))

        # sink_stop: absolute mass threshold that triggers the sink_fraction stop.
        # Impulse: fraction of finite initial load; source: fraction of c_pulse × free volume.
        if p.meso_as_source:
            mass_ref = float(np.sum(self.node_vols[self.interior_mask])) * p.c_pulse
            mass_ref = mass_ref if mass_ref > 0 else 1.0
        else:
            mass_ref = mass0 if mass0 > 0 else 1.0
        sink_stop = p.sink_fraction * mass_ref
        mass_to_sink = 0.0   # cumulative absorbed mass (mol, both modes)

        seg, edge_ij, d_edges = self._peclet_scaffold()
        rhs0 = np.zeros(self.st._matrix_size)

        times   = [0.0]
        frames  = [c.copy()]
        masses  = [mass0]
        fracs   = [0.0]
        pe_all  = [np.zeros(len(edge_ij))]

        self._log(f"=== Coupled time march (dt={p.dt} d, θ={p.theta}, "
                  f"ops='{p.ops}', scheme='{p.scheme}') ===")
        stop_reason = f"reached max_steps={p.max_steps}"
        for step in range(1, p.max_steps + 1):
            # (a)+(b) refresh Ψ_os from current c and re-solve hydraulics.
            manager.set_osmotic_from_concentration(
                c[nwj: nwj + n_cells], nwj, p.t_kelvin, psi_os_baseline)
            self.mecha.water_flux(h=p.h_idx, use_stored_psi_os=True, verbose=False)

            # (c) one implicit-Euler transport step on the fresh flow.  When
            # starch is active, solve sugar+starch as one fully-coupled block;
            # starch is osmotically inactive so it never enters the Ψ_os update.
            if use_starch:
                c_new, c_starch = self.mst.solve_coupled(
                    h=p.h_idx, i_maturity=p.i_mat, i_scenario=p.i_sce,
                    c_prev=[c, c_starch],
                    boundary_conditions=[self.bc, None],
                    theta=p.theta, operators=p.ops, scheme=p.scheme)
            else:
                c_new = self.st.solve(
                    h=p.h_idx, i_maturity=p.i_mat, i_scenario=p.i_sce,
                    rhs=rhs0.copy(), boundary_conditions=self.bc,
                    c_prev=c, theta=p.theta, operators=p.ops, scheme=p.scheme)
            dmax = float(np.max(np.abs(c_new - c)))
            mass_prev = float(np.sum(self.node_vols[self.interior_mask]
                                     * c[self.interior_mask]))
            c = c_new

            mass = float(np.sum(self.node_vols[self.interior_mask]
                                * c[self.interior_mask]))
            # Net interior mass lost this step = absorbed by sink (both modes).
            mass_to_sink += mass_prev - mass

            if edge_ij:
                A = self.st.build_advection_matrix(p.i_mat, p.i_sce).tocsr()
                a_edge = np.abs(np.array([A[iv, iu] for (iv, iu) in edge_ij]))
                pe_step = a_edge / d_edges
            else:
                pe_step = np.zeros(0)

            times.append(step * p.dt)
            frames.append(c.copy())
            masses.append(mass)
            fracs.append(mass_to_sink)
            pe_all.append(pe_step)
            if use_starch:
                starch_frames.append(c_starch.copy())

            if step % 20 == 0 or step == 1:
                self._log(f"  step {step:4d}  t={step*p.dt:6.3f} d  "
                          f"max|Δc|={dmax:.3e}  to-sink={mass_to_sink*1e12:.4f} pmol")
            if mass_to_sink >= sink_stop:
                stop_reason = f"≥{p.sink_fraction*100:.0f}% of ref mass transported to sink"
                break
            if dmax < equil_tol:
                stop_reason = "equilibrium reached (max|Δc| < equil_tol)"
                break

        polys, cell_ids, node_of_cell = self._cell_geometry()
        self._log(f"  Stopped after {len(frames)-1} step(s): {stop_reason}")
        return SimResult(
            mode='dynamic', params=asdict(p),
            concentration=np.asarray(frames, dtype=np.float32),
            peclet=np.asarray(pe_all, dtype=np.float32), pe_segments=seg,
            cell_polygons=polys, cell_ids=cell_ids, node_of_cell=node_of_cell,
            times=np.asarray(times), interior_mass=np.asarray(masses),
            frac_to_sink=np.asarray(fracs), stop_reason=stop_reason,
            water_flux_total=water_flux_total,
            concentration_starch=(np.asarray(starch_frames, dtype=np.float32)
                                  if use_starch else None),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Parallel driver
# ══════════════════════════════════════════════════════════════════════════════
def _run_one(params: SimParams) -> SimResult:
    """Top-level (picklable) worker: build sim from cached geometry and run it.

    The params object names its own simulation class (``simulation_class()``),
    so this dispatcher never changes when a new series is added.
    """
    geometry = get_geometry_base(params.seed)
    sim = params.simulation_class()(params, geometry, verbose=False)
    return sim.run()


def _pin_blas_single_thread():
    """Pool initializer: force each worker's BLAS to one thread.

    The osmo-hydraulic coupling calls scipy ``spsolve``, whose BLAS backend is
    multithreaded; with N worker processes each spawning K BLAS threads the
    cores are oversubscribed (N×K threads).  Pinning to one thread per process
    lets each worker cleanly own one core — the embarrassingly-parallel regime.
    """
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[var] = '1'


def run_experiments(param_list, max_workers: Optional[int] = None):
    """Run independent simulations and return SimResults, one per param object.

    Experiments are fully independent and share the same anatomy (via ``seed``),
    so they run embarrassingly parallel across a process pool.  Each worker pays
    the anatomy build cost once (cached per process) and pins its BLAS to a
    single thread to avoid core oversubscription.

    Worker count is chosen adaptively from the job count unless ``max_workers``
    is given: for a single job, or when the pool would not help, it runs inline;
    otherwise it uses ``min(cpu_count, len(param_list))`` pinned processes.
    """
    from concurrent.futures import ProcessPoolExecutor

    n_jobs = len(param_list)
    if n_jobs == 0:
        return []

    if max_workers is None:
        cores = os.cpu_count() or 1
        workers = min(cores, n_jobs)
    else:
        workers = max(1, int(max_workers))

    # Inline (no pool) when parallelism cannot pay: 1 job or 1 worker.
    if workers == 1 or n_jobs == 1:
        return [_run_one(p) for p in param_list]

    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_pin_blas_single_thread) as pool:
        return list(pool.map(_run_one, param_list))


# ══════════════════════════════════════════════════════════════════════════════
# Rendering helpers (post-processing — heavy memory, used one run at a time)
# ══════════════════════════════════════════════════════════════════════════════
def _import_mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _render_cell_field_gif(gdf, concentration, node_of_cell, times, frac_to_sink,
                           path, max_frames=150, fps=12, cmap='plasma',
                           title='', cbar_label='', scale=1.0):
    plt = _import_mpl()
    from matplotlib.animation import FuncAnimation, PillowWriter

    frames = concentration if concentration.ndim == 2 else concentration[None, :]
    n = frames.shape[0]
    stride = max(1, n // max_frames)
    idx = list(range(0, n, stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    vmax = float(np.nanpercentile(frames[-1] * scale, 99)) or 1.0

    gdf = gdf.copy()
    fig, ax = plt.subplots(figsize=(9, 7))

    def _draw(k):
        fi = idx[k]
        ax.clear()
        gdf['value'] = [float(frames[fi][n_i]) * scale for n_i in node_of_cell]
        gdf.plot(ax=ax, column='value', cmap=cmap, edgecolor='black',
                 linewidth=0.3, vmin=0.0, vmax=vmax,
                 missing_kwds={'color': 'lightgray'})
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('x (µm)'); ax.set_ylabel('y (µm)')
        t = '' if times is None else f't = {times[fi]:.3f} d'
        s = ('' if frac_to_sink is None
             else f'   |   to sink: {frac_to_sink[fi]*1e12:.4f} pmol')
        ax.set_title(f'{title}\n{t}{s}')
        return ax.collections

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.0, vmax))
    fig.colorbar(sm, ax=ax, label=cbar_label)
    anim = FuncAnimation(fig, _draw, frames=len(idx), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


def _render_peclet_gif(pe_segments, peclet, times, path, max_frames=150, fps=12):
    plt = _import_mpl()
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import LineCollection

    frames = peclet if peclet.ndim == 2 else peclet[None, :]
    n = frames.shape[0]
    stride = max(1, n // max_frames)
    idx = list(range(0, n, stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)

    finite = frames[np.isfinite(frames) & (frames > 0)]
    vmin = float(np.percentile(finite, 1)) if finite.size else 1e-3
    vmax = float(np.percentile(finite, 99)) if finite.size else 1.0
    xs = pe_segments[:, :, 0]; ys = pe_segments[:, :, 1]

    fig, ax = plt.subplots(figsize=(9, 7))
    norm = plt.matplotlib.colors.LogNorm(vmin=max(vmin, 1e-6), vmax=max(vmax, 1e-3))

    def _draw(k):
        fi = idx[k]
        ax.clear()
        lc = LineCollection(pe_segments, cmap='coolwarm', norm=norm)
        lc.set_array(np.clip(frames[fi], 1e-6, None))
        lc.set_linewidth(0.8)
        ax.add_collection(lc)
        ax.set_xlim(xs.min(), xs.max()); ax.set_ylim(ys.min(), ys.max())
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('x (µm)'); ax.set_ylabel('y (µm)')
        t = '' if times is None else f't = {times[fi]:.3f} d'
        ax.set_title(f'Per-bond local Péclet |A_edge|/D_edge\n{t}')
        return [lc]

    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    fig.colorbar(sm, ax=ax, label='local Péclet (log)')
    anim = FuncAnimation(fig, _draw, frames=len(idx), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


def _render_mass_balance(times, interior_mass, frac_to_sink, path):
    plt = _import_mpl()
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(times, np.asarray(interior_mass) * 1e12, 'b-', label='interior mass')
    ax1.set_xlabel('t (d)'); ax1.set_ylabel('interior solute mass (pmol)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.plot(times, np.asarray(frac_to_sink) * 1e12, 'r--', label='to sink')
    ax2.set_ylabel('cumulative mass to sink (pmol)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    fig.suptitle('Solute mass balance')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Diurnal modulation helpers
# ══════════════════════════════════════════════════════════════════════════════
def f_transpiration(t, period: float = 1.0, phase: float = 0.0,
                    baseline: float = 0.1):
    """Transpirational driver f(t) ∈ [baseline, 1]: smooth cosine, peaks at noon.

    ``baseline + (1-baseline)·(1-cos)/2``: a night floor of ``baseline`` (stomata
    never fully close), rising to 1 at solar noon.  Never zero, so transpiration
    does not shut down completely overnight.
    """
    wave = 0.5 * (1.0 - np.cos(2.0 * np.pi * (t / period) + phase))
    return baseline + (1.0 - baseline) * wave


def g_photosynthesis(t, period: float = 1.0, phase: float = 0.0):
    """Photosynthetic sucrose source g(t) ∈ [0,1]: daytime-gated, zero at night.

    Carbon fixation genuinely stops in the dark, so g keeps the rectified shape
    ``max(0, -cos)`` — in phase with the noon peak of f_transpiration.
    """
    return np.maximum(0.0, -np.cos(2.0 * np.pi * (t / period) + phase))


# Sucrose plasmodesmatal diffusivity: 0.1 × 5.2e-6 cm²/s × 86400 s/d → cm²/d.
_D_PD_SUCROSE = 0.1 * 5.2e-6 * 86400.0


# ══════════════════════════════════════════════════════════════════════════════
# Diurnal parameters
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class DiurnalParams(DynamicParams):
    """Transient needle transport with diurnal transpiration & photosynthesis.

    Set ``starch=True`` to activate the reversible sucrose ↔ starch chemistry;
    ``k_starch_syn`` (k_f) and ``k_starch_deg`` (k_r) control the rates.
    """

    _simulation_class_name: str = field(default='NeedleDiurnalSimulation',
                                        init=False, repr=False, compare=False)
    d_pd: float = _D_PD_SUCROSE   # cm²/d  sucrose PD diffusivity
    psi_xyl: float = -200.0       # hPa
    psi_atm: float = -1.0e3       # hPa  peak (midday) air potential
    c_pulse: float = 50.0e-6      # mol/cm³  reference mesophyll sucrose level

    # Photosynthetic sucrose injection: each step adds c_photo·g(t) to the
    # mesophyll (free interior nodes), so sucrose can accumulate above c_pulse.
    c_photo: float = 5.0e-6       # mol/cm³  per-step sucrose increment at g=1
    meso_as_source: bool = False

    period: float = 1.0           # d   diurnal period
    f_phase: float = 0.0          # rad phase of transpiration driver
    g_phase: float = 0.0          # rad phase of photosynthesis source
    f_baseline: float = 0.1       # -   night transpiration floor (fraction of peak)

    dt: float = 1.0 / 24.0        # d   hourly steps
    max_steps: int = 48            # two full days
    sink_fraction: float = 1.0e9  # disable drain-out stop
    equil_tol_frac: float = -1.0  # disable equilibrium stop

    # ---- Starch interconversion (active when starch=True) -------------------
    starch: bool = False          # activate sucrose ↔ starch coupling
    k_starch_syn: float = 2.0     # 1/d  sucrose → starch synthesis rate (k_f)
    k_starch_deg: float = 1.0     # 1/d  starch → sucrose remobilisation (k_r)
    c_starch0: float = 0.0        # mol/cm³  initial starch concentration

    label: str = 'diurnal'

    def __post_init__(self):
        # Keep include_starch in sync with the starch flag so _build_st works.
        if self.starch and not self.include_starch:
            object.__setattr__(self, 'include_starch', True)
            object.__setattr__(self, 'k_starch_syn', self.k_starch_syn)
            object.__setattr__(self, 'k_starch_deg', self.k_starch_deg)
            object.__setattr__(self, 'c_starch0', self.c_starch0)


# ══════════════════════════════════════════════════════════════════════════════
# Diurnal simulation  (starch=False → sucrose only; starch=True → coupled)
# ══════════════════════════════════════════════════════════════════════════════
class NeedleDiurnalSimulation(NeedleDynamicSimulation):
    """Diurnally forced needle transport.

    At every implicit-Euler step: re-sets the air BC to PSI_AIR·f(t), holds the
    mesophyll at c_meso·g(t), refreshes hydraulics, and advances solute transport.
    When ``params.starch=True`` the coupled sucrose↔starch multisolute pathway is
    used (via ``self.mst``); otherwise only sucrose is transported.
    Records j_w, j_s, and (when starch active) mean interior starch concentration.
    """

    mode = 'dynamic'

    def _sink_nodes(self) -> np.ndarray:
        ids = list(self.phloem_loading_ids) + list(self.xylem_cell_ids)
        return self.nwj + np.array(sorted(set(ids)), dtype=int)

    def run(self) -> SimResult:
        p = self.params
        nwj, n_cells = self.nwj, self.n_cells
        manager = self.mecha.network.cell_manager
        use_starch = p.starch

        self.mecha.water_flux(h=p.h_idx, verbose=False)
        psi_os_baseline = np.array(
            [c.psi_os if c.psi_os is not None else 0.0 for c in manager])

        meso_nodes = np.array([nwj + cid for cid in self.meso_cell_ids], dtype=int)
        sink_nodes = self._sink_nodes()

        c = np.zeros(self.n_total)
        for nid, val in self.bc.items():
            if 0 <= nid < self.n_total:
                c[nid] = val

        c_sta = np.zeros(self.n_total)
        if use_starch and p.c_starch0 != 0.0:
            for cid in range(n_cells):
                c_sta[nwj + cid] = p.c_starch0

        seg, edge_ij, d_edges = self._peclet_scaffold()
        rhs0 = np.zeros(self.st._matrix_size)

        times     = [0.0]
        frames    = [c.copy()]
        sta_frames = [c_sta.copy()] if use_starch else None
        masses    = [float(np.sum(self.node_vols[self.interior_mask]
                                  * c[self.interior_mask]))]
        j_w_ts    = [0.0]
        j_s_ts    = [0.0]
        m_sta_ts  = [float(np.mean(c_sta[nwj: nwj + n_cells]))] if use_starch else None
        f_ts      = [float(f_transpiration(0.0, p.period, p.f_phase, p.f_baseline))]
        g_ts      = [float(g_photosynthesis(0.0, p.period, p.g_phase))]
        pe_all    = [np.zeros(len(edge_ij))]

        starch_tag = (f" k_f={p.k_starch_syn:.2f} k_r={p.k_starch_deg:.2f}"
                      if use_starch else "")
        self._log(f"=== Diurnal march{starch_tag} (dt={p.dt:.4f} d,"
                  f" {p.max_steps} steps, ops='{p.ops}', scheme='{p.scheme}') ===")
        stop_reason = f"reached max_steps={p.max_steps}"
        for step in range(1, p.max_steps + 1):
            t = step * p.dt
            f_t = float(f_transpiration(t, p.period, p.f_phase, p.f_baseline))
            g_t = float(g_photosynthesis(t, p.period, p.g_phase))

            # Diurnal air BC (never fully off — floors at p.f_baseline).
            self.mecha.set_air_wall_bc(self.psi_air * f_t)
            # Photosynthetic injection: ADD c_photo·g(t) to the mesophyll each
            # step (free interior nodes, no Dirichlet pin), so sucrose can
            # accumulate above c_pulse when production outpaces export.
            bc_step = self.bc
            c[meso_nodes] += p.c_photo * g_t

            manager.set_osmotic_from_concentration(
                c[nwj: nwj + n_cells], nwj, p.t_kelvin, psi_os_baseline)
            self.mecha.water_flux(h=p.h_idx, use_stored_psi_os=True, verbose=False)
            j_w = self._total_transpiration_flux()

            if use_starch:
                c, c_sta = self.mst.solve_coupled(
                    h=p.h_idx, i_maturity=p.i_mat, i_scenario=p.i_sce,
                    c_prev=[c, c_sta], boundary_conditions=[bc_step, None],
                    theta=p.theta, operators=p.ops, scheme=p.scheme)
            else:
                c = self.st.solve(
                    h=p.h_idx, i_maturity=p.i_mat, i_scenario=p.i_sce,
                    rhs=rhs0.copy(), boundary_conditions=bc_step,
                    c_prev=c, theta=p.theta, operators=p.ops, scheme=p.scheme)

            T = self.st.spatial_operator(
                h=p.h_idx, i_maturity=p.i_mat, i_scenario=p.i_sce,
                operators=p.ops, scheme=p.scheme).tocsr()
            j_s = float(np.sum(T.dot(c)[sink_nodes]))

            if edge_ij:
                A = self.st.build_advection_matrix(p.i_mat, p.i_sce).tocsr()
                a_edge = np.abs(np.array([A[iv, iu] for (iv, iu) in edge_ij]))
                pe_step = a_edge / d_edges
            else:
                pe_step = np.zeros(0)

            mass = float(np.sum(self.node_vols[self.interior_mask]
                                * c[self.interior_mask]))
            times.append(t);  frames.append(c.copy());  masses.append(mass)
            j_w_ts.append(j_w);  j_s_ts.append(j_s)
            f_ts.append(f_t);  g_ts.append(g_t);  pe_all.append(pe_step)
            if use_starch:
                sta_frames.append(c_sta.copy())
                m_sta_ts.append(float(np.mean(c_sta[nwj: nwj + n_cells])))

            if step % 6 == 0 or step == 1:
                starch_info = (f"  m_sta={m_sta_ts[-1]*1e6:.2f} µM"
                               if use_starch else "")
                self._log(f"  step {step:3d}  t={t:5.3f} d  f={f_t:.2f} g={g_t:.2f}"
                          f"  j_w={j_w:.3e}  j_s={j_s*1e12:.3f} pmol/d{starch_info}")

        polys, cell_ids, node_of_cell = self._cell_geometry()
        self._log(f"  Finished {len(frames)-1} step(s): {stop_reason}")

        extras = {
            'j_w':     np.asarray(j_w_ts),
            'j_s':     np.asarray(j_s_ts),
            'f_drive': np.asarray(f_ts),
            'g_drive': np.asarray(g_ts),
        }
        if use_starch:
            extras['m_starch'] = np.asarray(m_sta_ts)

        return SimResult(
            mode='dynamic', params=asdict(p),
            concentration=np.asarray(frames, dtype=np.float32),
            concentration_starch=(np.asarray(sta_frames, dtype=np.float32)
                                  if use_starch else None),
            peclet=np.asarray(pe_all, dtype=np.float32), pe_segments=seg,
            cell_polygons=polys, cell_ids=cell_ids, node_of_cell=node_of_cell,
            times=np.asarray(times), interior_mass=np.asarray(masses),
            frac_to_sink=np.asarray(j_s_ts), stop_reason=stop_reason,
            water_flux_total=(j_w_ts[-1] if j_w_ts else None),
            extras=extras,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Diurnal flux plot
# ══════════════════════════════════════════════════════════════════════════════
def diurnal_flux_plot(res: SimResult, path: str) -> str:
    """Time evolution of a diurnal run: j_w, j_s, j_s/j_w (+ starch if present).

    Reads the series from ``res.extras`` (via ``get_extra``); a starch panel is
    added automatically when the run produced an ``m_starch`` series.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    t   = np.asarray(res.times)
    j_w = np.asarray(res.get_extra('j_w'))
    j_s = np.asarray(res.get_extra('j_s'))
    m_s = res.get_extra('m_starch')
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(np.abs(j_w) > 0, j_s / j_w, np.nan)

    period  = float(res.params.get('period', 1.0))
    g_phase = float(res.params.get('g_phase', 0.0))

    n_panels = 4 if m_s is not None else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 3 * n_panels), sharex=True)
    axes = np.atleast_1d(axes)

    def _night_bands(ax):
        # Night = dark period (no photosynthesis); f no longer reaches zero.
        tt = np.linspace(t[0], t[-1], 2000)
        night = g_photosynthesis(tt, period, g_phase) <= 0.0
        ax.fill_between(tt, 0, 1, where=night, transform=ax.get_xaxis_transform(),
                        color='0.85', zorder=0, step='mid')

    ax_jw, ax_js, ax_rat = axes[0], axes[1], axes[-1]

    _night_bands(ax_jw)
    ax_jw.plot(t, j_w, 'b-o', ms=3, lw=1.2)
    ax_jw.set_ylabel('j_w  (cm³/d)', color='b')
    ax_jw.tick_params(axis='y', labelcolor='b')
    ax_jw.set_title('Transpirational water flux j_w')

    _night_bands(ax_js)
    ax_js.plot(t, j_s * 1e12, 'r-o', ms=3, lw=1.2)
    ax_js.set_ylabel('j_s  (pmol/d)', color='r')
    ax_js.tick_params(axis='y', labelcolor='r')
    ax_js.set_title('Transported solute flux j_s (to sink)')

    if m_s is not None:
        ax_st = axes[2]
        _night_bands(ax_st)
        ax_st.plot(t, np.asarray(m_s) * 1e6, 'g-o', ms=3, lw=1.2)
        ax_st.set_ylabel('mean starch  (µM)', color='g')
        ax_st.tick_params(axis='y', labelcolor='g')
        ax_st.set_title('Interior starch concentration (cell mean)')

    _night_bands(ax_rat)
    ax_rat.plot(t, ratio * 1e3, 'k-o', ms=3, lw=1.2)
    ax_rat.set_ylabel('j_s / j_w  (mM)')
    ax_rat.set_xlabel('t  (d)')
    ax_rat.set_title('Export ratio j_s / j_w')

    fig.suptitle('Diurnal needle transport: water, solute, and their quotient')
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Demo / smoke test
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    SEED = 42
    experiments = [
        DynamicParams(seed=SEED, label='dyn_lowPD',  d_pd=1e-1, max_steps=40),
        DynamicParams(seed=SEED, label='dyn_highPD', d_pd=5e-1, max_steps=40),
    ]
    results = run_experiments(experiments, max_workers=2)
    for res in results:
        print(res.summary())
        stem = os.path.join(_OUT_DIR, f"needle_{res.params['label']}")
        res.save(stem + '.npz')
        res.mass_balance_plot(stem + '_massbalance.png')
        res.concentration_gif(stem + '_concentration.gif')
