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
from openalea.mecha.utils.solute_transport import SoluteTransport
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
    sigma_sucrose: float = 0.6     # -      membrane reflection coefficient

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


@dataclass(frozen=True)
class SteadyParams(SimParams):
    """Steady-state (fixed-input) sucrose transport."""

    c_meso: float = 50.0e-6        # mol/cm³  mesophyll Dirichlet SOURCE
    couple_tol: float = 10.0       # hPa   convergence tol on max|Δψ_total|
    couple_maxiter: int = 25
    continuation: Optional[tuple] = None   # homotopy λ schedule (None → single stage)


@dataclass(frozen=True)
class DynamicParams(SimParams):
    """Transient (time-series) sucrose transport."""

    c_pulse: float = 50.0e-6       # mol/cm³  initial mesophyll impulse
    dt: float = 1.0e-1             # d   time step (implicit Euler)
    theta: float = 1.0             # 1.0 implicit Euler | 0.5 Crank–Nicolson
    max_steps: int = 200
    equil_tol_frac: float = 1.0e-6  # equilibrium when max|Δc| < this × c_pulse
    sink_fraction: float = 0.90    # stop once ≥ this fraction has left the interior
    couple_maxiter: int = 12       # inner hydraulic-coupling iterations per step


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

    # ---- Steady-only ---------------------------------------------------------
    q_load: Optional[float] = None
    n_iterations: Optional[int] = None
    converged: Optional[bool] = None

    # ─────────────────────────────────────────────────────────────────────────
    def save(self, path: str) -> str:
        """Persist to a compressed ``.npz`` (params stored as a JSON sidecar)."""
        import json
        np.savez_compressed(
            path,
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
        )
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
                frac_to_sink=(None if self.frac_to_sink is None
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
        """Hook: steady adds a mesophyll Dirichlet source; dynamic does not."""
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
        self.st = SoluteTransport(self.mecha, self.params.diffusion_params(),
                                  cap, mode='full')
        self.D_mat = self.st.build_diffusion_matrix(self.params.h_idx,
                                                    self.params.i_mat)

    # Dynamic has NO mesophyll Dirichlet source — the pulse is a free initial
    # condition, so _apply_source_bc stays a no-op (inherited).

    def run(self) -> SimResult:
        p = self.params
        nwj, n_cells = self.nwj, self.n_cells
        manager = self.mecha.network.cell_manager
        equil_tol = p.equil_tol_frac * p.c_pulse

        # Baseline Ψ_os from a pulse-FREE solve (so van't Hoff is added, not doubled).
        self.mecha.water_flux(h=p.h_idx, verbose=False)
        psi_os_baseline = np.array(
            [c.psi_os if c.psi_os is not None else 0.0 for c in manager])

        # Initial impulse field + Dirichlet values at t=0.
        c = np.zeros(self.n_total)
        for cid in self.meso_cell_ids:
            c[nwj + cid] = p.c_pulse
        for nid, val in self.bc.items():
            if 0 <= nid < self.n_total:
                c[nid] = val

        mass0 = float(np.sum(self.node_vols[self.interior_mask]
                             * c[self.interior_mask]))
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

            # (c) one implicit-Euler transport step on the fresh flow.
            c_new = self.st.solve(
                h=p.h_idx, i_maturity=p.i_mat, i_scenario=p.i_sce,
                rhs=rhs0.copy(), boundary_conditions=self.bc,
                c_prev=c, theta=p.theta, operators=p.ops, scheme=p.scheme)
            dmax = float(np.max(np.abs(c_new - c)))
            c = c_new

            mass = float(np.sum(self.node_vols[self.interior_mask]
                                * c[self.interior_mask]))
            frac = 1.0 - mass / mass0 if mass0 > 0 else 0.0

            if edge_ij:
                A = self.st.build_advection_matrix(p.i_mat, p.i_sce).tocsr()
                a_edge = np.abs(np.array([A[iv, iu] for (iv, iu) in edge_ij]))
                pe_step = a_edge / d_edges
            else:
                pe_step = np.zeros(0)

            times.append(step * p.dt)
            frames.append(c.copy())
            masses.append(mass)
            fracs.append(frac)
            pe_all.append(pe_step)

            if step % 20 == 0 or step == 1:
                self._log(f"  step {step:4d}  t={step*p.dt:6.3f} d  "
                          f"max|Δc|={dmax:.3e}  to-sink={frac*100:5.1f}%")
            if frac >= p.sink_fraction:
                stop_reason = f"≥{p.sink_fraction*100:.0f}% transported to sink"
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
        )


# ══════════════════════════════════════════════════════════════════════════════
# Parallel driver
# ══════════════════════════════════════════════════════════════════════════════
def _run_one(params: SimParams) -> SimResult:
    """Top-level (picklable) worker: build sim from cached geometry and run it."""
    geometry = get_geometry_base(params.seed)
    if isinstance(params, SteadyParams):
        sim = NeedleSteadySimulation(params, geometry, verbose=False)
    elif isinstance(params, DynamicParams):
        sim = NeedleDynamicSimulation(params, geometry, verbose=False)
    else:
        raise TypeError(f"Unsupported params type: {type(params)!r}")
    return sim.run()


def run_experiments(param_list, max_workers: Optional[int] = None):
    """Run several simulations in parallel (process pool) and return SimResults.

    All experiments should share the same ``seed`` so every worker rebuilds the
    identical anatomy (deterministic) and caches it once.  Each worker pays the
    anatomy build cost a single time, then runs its assigned experiments cheaply.
    """
    from concurrent.futures import ProcessPoolExecutor
    if max_workers == 1 or len(param_list) == 1:
        return [_run_one(p) for p in param_list]
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
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
             else f'   |   to sink: {frac_to_sink[fi]*100:.1f}%')
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
    ax2.plot(times, np.asarray(frac_to_sink) * 100, 'r--', label='to sink')
    ax2.set_ylabel('fraction to sink (%)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    fig.suptitle('Solute mass balance')
    fig.tight_layout()
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
