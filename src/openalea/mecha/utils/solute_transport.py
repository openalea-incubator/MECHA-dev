import copy
import warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse import coo_matrix

from openalea.mecha.utils.hydraulic_solver import HydraulicMatrixBuilder


def _cgroup_canonical(cg: int) -> int:
    """Map Cellset-style cgroup aliases to canonical MECHA codes."""
    return {19: 13, 20: 13, 21: 16, 23: 11, 26: 12}.get(cg, cg)


class SoluteTransport:
    """
    Convection-diffusion solute transport on a MECHA hydraulic network.

    Governing equation (per node i):
        C_i dc_i/dt = (T c)_i + s_i

    where
      C  — diagonal capacitance (node volumes × tissue storage fraction)
      T  — spatial transport operator  T = D + A − R
           D = diffusion matrix  (negative diagonal / positive off-diagonal)
           A = advection matrix  (upwind, same sign convention)
           R = first-order reaction (degradation) matrix, diag(k_deg × V_i);
               only present when reaction_params['k_deg'] > 0
           (T c)_i = net transport flux INTO node i
      s  — external source  (s > 0 → production at i)

    Crank-Nicolson θ-method:
        [C/dt − θ T] c_new = [C/dt + (1−θ) T] c_old + s

        θ = 0   Explicit Euler  1st-order; may oscillate when Pe > 2 
        θ = 0.5   Crank-Nicolson  2nd-order; may oscillate when Pe > 2 
        θ = 1.0   Implicit Euler  1st-order; oscillation-free at any Pe

    Steady state (C = 0, no 'dt' specified):
        T c = −s   →   T c = rhs   with  rhs = −s
        (rhs[i] < 0 for a production source; rhs[i] > 0 for a sink)
        Dirichlet BCs override this convention at fixed-concentration nodes.

    Parameters
    ----------
    mecha : Mecha
        Solved Mecha instance; water_flux() must have been called.
    diffusion_params : dict
        {
            'apo_wall'      : float  cm²/d  apoplastic cell-wall diffusivity
                              → replaces hormones.diff1_pw1 in the builder
            'plasmodesmata' : float  cm²/d  PD diffusivity
                              → replaces hormones.diff1_pd1
            'membrane'      : float  cm²/d  passive transmembrane diffusivity
                              (new; HydraulicMatrixBuilder does not build this)
            'sigma'         : dict {cgroup_int: float}
                              membrane reflection coefficient (0–1; default 0)
        }
    capacitance_params : dict, optional
        {
            'dt'    : float  time step (days); None → steady state
            'C_wall': float  apoplast storage fraction (0–1; default 1.0)
            'C_cell': float  symplast storage fraction (0–1; default 1.0)
        }
    reaction_params : dict, optional
        {
            'k_deg' : float  1/d  first-order degradation rate constant
                      (mol degraded / mol-day); default 0.0 (no reaction)
        }
    mode : str
        'full'  n_total × n_total
        'apo'   n_wall_junction × n_wall_junction
        'sym'   n_cells × n_cells
    """

    def __init__(self, mecha, diffusion_params: dict,
                 capacitance_params: dict = None, mode: str = 'full',
                 reaction_params: dict = None):
        if mode not in ('full', 'apo', 'sym'):
            raise ValueError(f"mode must be 'full', 'apo', or 'sym'; got '{mode}'")

        self.mecha              = mecha
        self.network            = mecha.network
        self.diffusion_params   = diffusion_params
        self.capacitance_params = capacitance_params
        self.reaction_params    = reaction_params
        self.mode               = mode

        self.D_wall = float(diffusion_params.get('apo_wall',      0.0))
        self.D_pd   = float(diffusion_params.get('plasmodesmata', 0.0))
        self.D_mem  = float(diffusion_params.get('membrane',      0.0))
        self.sigma  = diffusion_params.get('sigma', {})
        self.k_deg  = float(reaction_params.get('k_deg', 0.0)) if reaction_params else 0.0

        self.n_total         = mecha.network.graph.number_of_nodes()
        self.n_wall_junction = mecha.network.n_wall_junction
        self.n_cells         = mecha.network.n_cells
        self.n_walls         = mecha.network.n_walls

        self._matrix_size = {
            'full': self.n_total,
            'apo':  self.n_wall_junction,
            'sym':  self.n_cells,
        }[mode]

        self.position = mecha.position
        self.indice   = mecha.indice

        self._D_cache: dict = {}   # (h, i_maturity) → csr_matrix
        self._T_cache: dict = {}   # (scheme, mode, h, i_mat, i_scen, fp) → csr_matrix
        self._A_cache: dict = {}   # (mode, i_mat, i_scen, fp) → csr_matrix

    # ------------------------------------------------------------------
    # Diffusion operator D
    # ------------------------------------------------------------------

    def build_diffusion_matrix(self, h: int, i_maturity: int) -> sp.csr_matrix:
        """
        Diffusion matrix via HydraulicMatrixBuilder with custom constants.

        The builder's geometry code (temp, temp_factor from wall conductivity
        ratios) is reused unchanged. Only diff1_pw1 and diff1_pd1 are swapped
        for the values from diffusion_params, so tissue-specific barrier
        effects are inherited from the hydraulic configuration.

        Passive membrane diffusion (D_mem) is appended separately.
        Result is cached per (h, i_maturity).
        """
        cache_key = (h, i_maturity)
        if cache_key in self._D_cache:
            return self._D_cache[cache_key]

        mecha   = self.mecha
        network = self.network

        fake_hormones               = copy.copy(mecha.hormones)
        fake_hormones.diff1_pw1     = self.D_wall
        fake_hormones.diff1_pd1     = self.D_pd
        fake_hormones.carrier_elems = []   # no carrier transport
        fake_hormones.sym_zombie0   = []   # BCs handled in solve()

        fake_boundary        = copy.copy(mecha.boundary)
        fake_boundary.c_flag = True

        fake_general         = copy.copy(mecha.general)
        fake_general.c_flag  = True

        saved_apo_wall = getattr(network, 'apo_wall_zombies0', [])
        saved_apo_j    = getattr(network, 'apo_j_zombies0',    [])
        network.apo_wall_zombies0 = []
        network.apo_j_zombies0    = []

        try:
            builder = HydraulicMatrixBuilder(
                network=network,
                geometry=mecha.geometry,
                boundary=fake_boundary,
                hydraulic=mecha.hydraulic,
                hormones=fake_hormones,
                general=fake_general,
                geo_props=mecha.geo_props,
                position=mecha.position,
                indice=mecha.indice,
            )
            _, matrix_C, _, _, _, _, _, _ = builder.build(
                h=h,
                i_maturity=i_maturity,
                hydraulic_conductivities=mecha.hydraulic_conductivities,
                boundary=fake_boundary,
                psi_xyl=mecha.psi_xyl,
                psi_sieve=mecha.psi_sieve,
                distributed_flow_xyl=mecha.distributed_flow_xyl,
                distributed_flow_sieve=mecha.distributed_flow_sieve,
            )
        finally:
            network.apo_wall_zombies0 = saved_apo_wall
            network.apo_j_zombies0    = saved_apo_j

        D = matrix_C.tocsr()

        if self.D_mem > 0.0 and self.mode != 'apo':
            D = D + self._build_membrane_diffusion(i_maturity)

        nwj = self.n_wall_junction
        if self.mode == 'apo':
            # _fill_membrane adds wall→xylem-cell entries (diff1_pw1-scaled) for
            # cgroup 13/19/20 nodes when barrier>0.  Slicing to [:nwj,:nwj] drops
            # those off-diagonal columns but keeps the diagonal penalties → net
            # outflow → broken mass conservation.  Restore row-sum=0 by adding
            # the dropped column sums back onto the diagonal.
            cross_sum = np.array(D[:nwj, nwj:].sum(axis=1)).ravel()
            D = D[:nwj, :nwj] + sp.diags(cross_sum, format='csr')
        elif self.mode == 'sym':
            cross_sum = np.array(D[nwj:, :nwj].sum(axis=1)).ravel()
            D = D[nwj:, nwj:] + sp.diags(cross_sum, format='csr')

        self._D_cache[cache_key] = D
        return D

    def _build_membrane_diffusion(self, i_maturity: int) -> sp.csr_matrix:
        """
        Passive transmembrane diffusion (wall ↔ cell).

        DF = D_mem × L × h × 1e-4 / dist   [cm³/d]
          L    wall length (µm), h axial height (µm), dist cell-to-wall (µm)
          1e-4 converts (µm²/µm = µm) to cm: 1 µm = 1e-4 cm
        """
        height = float(self.mecha.geometry.maturity_stages[i_maturity].get('height'))
        cm = self.mecha.network.cell_manager
        nwj    = self.n_wall_junction
        n      = self.n_total
        rows, cols, data = [], [], []

        for node, edges in self.network.graph.adjacency():
            i = self.indice[node]
            
            if i >= nwj:
                continue
            for neighbor, eattr in edges.items():
                j = self.indice[neighbor]
                mb = cm.get_membrane_by_edge(i, j)

                if j <= i or eattr.get('path') != 'membrane':
                    continue
                L  = eattr.get('length', 1.0)
                d  = eattr.get('dist',   1.0)
                DF = self.D_mem * L * height * 1e-4 / d
                eattr['membrane_diffusivity'] = DF
                if mb is not None:
                    mb.diffusion_coeff  = self.D_mem
                
                for r, c, v in ((i, i, -DF), (i, j, DF), (j, j, -DF), (j, i, DF)):
                    rows.append(r); cols.append(c); data.append(v)

        return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

    # ------------------------------------------------------------------
    # Advection operator A  (upwind, same sign convention as D)
    # ------------------------------------------------------------------

    def build_advection_matrix(self, i_maturity: int,
                                  i_scenario: int) -> sp.csr_matrix:
        """
        Upwind advection matrix from MECHA water flow.

        Source: mecha.edge_flux_list[i_maturity][i_scenario]
        {'source': i, 'target': j, 'flux': F}  with F [cm³/d]:
          F > 0  flow from source to target (source = upstream)
          F < 0  flow from target to source (target = upstream)

        First-order upwind.  Stable only while the edge Peclet number
        Pe = |F·f| / D_edge stays ≲ 2; at high Pe the steady operator D + A 
        becomes ill-conditioned and can return non-physical concentrations.  
        For those regimes use the Scharfetter–Gummel operator via 
        build_transport_operator(scheme='sg') / solve(scheme='sg'),
        which discretizes D and A together per edge.

        Assembly (negative diagonal / positive off-diagonal, matching D):
          F > 0:  A[src,src] −= F·f,  A[tgt,src] += F·f
          F < 0:  A[tgt,tgt] += F·f,  A[src,tgt] −= F·f
        where f = 1 − σ (σ = reflection coeff. on membranes).

        Cached per (mode, i_maturity, i_scenario) plus a fingerprint of the
        physical drives (edge fluxes, and cell psi_os when a membrane reflection
        coefficient is active).  Unlike the geometry-only D, A depends on the
        water flow, which coupled_solver refreshes every iteration; the
        fingerprint detects that change and rebuilds, while a static drive hits
        the cache — mirroring the Scharfetter–Gummel operator cache.
        """
        cache_key = (self.mode, i_maturity, i_scenario,
                     self._operator_fingerprint(i_maturity, i_scenario))
        if cache_key in self._A_cache:
            return self._A_cache[cache_key]

        nwj  = self.n_wall_junction
        n    = self.n_total
        rows, cols, data = [], [], []

        for edge in self.mecha.edge_flux_list[i_maturity][i_scenario]:
            src = edge['source']
            tgt = edge['target']
            F   = edge['flux']
            if F == 0.0:
                continue

            src_apo = src < nwj
            tgt_apo = tgt < nwj
            is_apo  = src_apo and tgt_apo
            is_pd   = (not src_apo) and (not tgt_apo)
            is_mem  = not (is_apo or is_pd)

            if self.mode == 'apo' and not is_apo:
                continue
            if self.mode == 'sym' and not is_pd:
                continue

            if is_mem:
                cell_node = tgt if not tgt_apo else src
                cg     = _cgroup_canonical(
                    self.network.graph.nodes[cell_node].get('cgroup', 4))
                factor = 1.0 - float(self.sigma.get(cg, 0.0))
            else:
                factor = 1.0

            if F > 0.0:   # src is upstream
                for r, c, v in ((src, src, -F * factor),
                                (tgt, src,  F * factor)):
                    rows.append(r); cols.append(c); data.append(v)
            else:          # tgt is upstream (F < 0)
                for r, c, v in ((tgt, tgt,  F * factor),
                                (src, tgt, -F * factor)):
                    rows.append(r); cols.append(c); data.append(v)

        A = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        A = A + self._build_osmotic_advection(i_maturity)

        if self.mode == 'apo':
            A = A[:nwj, :nwj].tocsr()
        elif self.mode == 'sym':
            A = A[nwj:, nwj:].tocsr()

        # Keep only the latest matrix per structural key (evict stale
        # fingerprints) so the cache stays bounded under dynamic coupling.
        struct = cache_key[:-1]
        for k in [k for k in self._A_cache if k[:-1] == struct]:
            del self._A_cache[k]
        self._A_cache[cache_key] = A
        return A

    @staticmethod
    def _bernoulli(x: float) -> float:
        """Bernoulli function B(x) = x / (e^x − 1), scalar, singularity-safe.

        B(0)=1 (Taylor), B(x) → 0 as x → +∞ and B(x) → −x as x → −∞, so the
        Scharfetter–Gummel edge weights reduce to central differencing (B→1) at
        Pe→0 and to pure upwind at high |Pe|.
        """
        if abs(x) < 1e-10:
            return 1.0 - x / 2.0 + x * x / 12.0
        # Guard the exp overflow at large |x|: B(x) → −x as x → −∞ and
        # B(x) → 0 as x → +∞, so we can return the limits directly instead of
        # evaluating expm1 (which overflows for x ≳ 709 and underflows below).
        if x < -700.0:
            return -x
        if x > 700.0:
            return 0.0
        return x / np.expm1(x)

    def _operator_fingerprint(self, i_maturity: int, i_scenario: int) -> int:
        """Hash of the physical drives the transport operator depends on.

        The advective operator is set by the per-edge water fluxes; when any
        membrane reflection coefficient is non-zero the operator additionally
        depends on the cell/wall osmotic potentials (Kedem–Katchalsky term).
        Both are refreshed by mecha.water_flux(), so a fingerprint over them
        detects the change that dynamic coupling introduces and invalidates the
        cached operator. Geometry and diffusivities are constant per instance
        and are covered by the rest of the cache key.
        """
        flux = np.array(
            [e['flux'] for e in self.mecha.edge_flux_list[i_maturity][i_scenario]],
            dtype=float,
        )
        fp = hash(flux.tobytes())
        if any(float(v) != 0.0 for v in self.sigma.values()):
            os_vals = np.array(
                [(c.psi_os if getattr(c, 'psi_os', None) is not None else 0.0)
                 for c in self.network.cell_manager],
                dtype=float,
            )
            fp ^= hash(os_vals.tobytes())
        return fp

    def build_transport_operator(self, h: int, i_maturity: int,
                                 i_scenario: int,
                                 scheme: str = 'upwind') -> sp.csr_matrix:
        """
        Build the FULL steady transport operator T (advection + diffusion) in a
        single per-edge pass.

        Per edge (i, j) — i<j in full-network indices — with
          D_e  = diffusive conductance  [cm³/d]  (≥ 0; off-diagonal weight of D)
          F    = advective VOLUME flux i→j  [cm³/d]  (pressure + osmotic),
                 scaled by the membrane carry factor f = 1 − σ,
        the edge flux INTO node i is
          upwind :  D_e (c_j − c_i)  +  [ f·F>0 ? −f·F c_i : −f·F c_j ]   (donor)
          sg     :  D_e [ B(P) c_j − B(−P) c_i ],   P = f·F / D_e
        with B the Bernoulli function.  Both are assembled antisymmetrically so
        each row sums to zero (mass-conserving).  ``scheme='upwind'`` reproduces
        ``build_diffusion_matrix + build_advection_matrix`` exactly; ``'sg'``
        gives the exponentially-fitted, non-oscillatory operator.

        Parameters
        ----------
        h, i_maturity, i_scenario : int
        scheme : {'upwind', 'sg'}

        Returns
        -------
        csr_matrix  sliced to the active mode ('full'/'apo'/'sym').
        """
        if scheme not in ('upwind', 'sg'):
            raise ValueError(f"scheme must be 'upwind' or 'sg'; got {scheme!r}")

        # Cache the assembled operator keyed on (scheme, mode, indices) AND a
        # fingerprint of the physical inputs the operator depends on — the
        # per-edge water fluxes and, when osmotic membrane transport is active,
        # the cell/wall osmotic potentials. Geometry-only quantities (mesh,
        # diffusivities) never change for a given instance, so they need not be
        # fingerprinted. This keeps the cache correct under dynamic coupling
        # (coupled_solver re-solves water_flux every iteration, which rewrites
        # edge_flux_list and updates psi_os): a changed drive yields a different
        # fingerprint and forces a rebuild, while a static drive (e.g. a fixed
        # transport run) hits the cache on every step.
        cache_key = (scheme, self.mode, h, i_maturity, i_scenario,
                     self._operator_fingerprint(i_maturity, i_scenario))
        if cache_key in self._T_cache:
            return self._T_cache[cache_key]

        nwj = self.n_wall_junction
        n   = self.n_total
        cm  = self.network.cell_manager
        graph = self.network.graph

        # Diffusion off-diagonals D_e per node pair (full network).  Each simple
        # graph edge is a single physical link, so D[i,j] is that edge's D_e.
        D_full = self._diffusion_full(h, i_maturity)

        # ── Per-edge advective volume flux F_ij (i<j), pressure-driven ──────────
        # edge_flux_list holds one entry per active edge with a signed flux and a
        # source→target orientation; re-key to canonical (min,max) with a sign.
        flux_ij: dict = {}
        for edge in self.mecha.edge_flux_list[i_maturity][i_scenario]:
            src, tgt, F = edge['source'], edge['target'], edge['flux']
            if F == 0.0:
                continue
            a, b = (src, tgt) if src < tgt else (tgt, src)
            s = 1.0 if src < tgt else -1.0
            flux_ij[(a, b)] = flux_ij.get((a, b), 0.0) + s * F

        def _os(obj):
            v = getattr(obj, 'psi_os', None)
            return float(v) if (v is not None and not np.isnan(float(v))) else 0.0

        rows, cols, data = [], [], []

        def _emit(i, j, De, Ff):
            """Emit the edge operator (into T) for edge (i,j) with diffusion De
            and advective volume flux Ff from i→j.  Uses list.extend (mutation)
            so the enclosing rows/cols/data lists are updated in place."""
            if De <= 0.0 and Ff == 0.0:
                return
            if scheme == 'sg' and De > 0.0:
                P  = Ff / De                   # local Peclet number
                Bp = self._bernoulli(P)        # B(P)
                Bm = Bp + P                    # B(−P) = B(P) + P
                w_ij = De * Bp                 # coeff of c_j in flux into i
                w_ii = De * Bm                 # coeff of −c_i in flux into i
                # flux into i = w_ij c_j − w_ii c_i ; flux into j = −(that)
                rows.extend([i, i, j, j])
                cols.extend([j, i, i, j])
                data.extend([w_ij, -w_ii, w_ii, -w_ij])
            else:
                # upwind (also the De=0 fallback): diffusion + donor-cell advection
                if De > 0.0:
                    rows.extend([i, i, j, j])
                    cols.extend([j, i, i, j])
                    data.extend([De, -De, De, -De])
                if Ff > 0.0:                    # i upstream
                    rows.extend([i, j]); cols.extend([i, i]); data.extend([-Ff, Ff])
                elif Ff < 0.0:                  # j upstream
                    rows.extend([j, i]); cols.extend([j, j]); data.extend([Ff, -Ff])

        # ── Single per-edge pass (recycles the graph.edges iteration used by
        #    visualize('velocity')): each physical edge once, with its path,
        #    conductance and flux. ────────────────────────────────────────────
        seen = set()
        for u, v, eattr in graph.edges(data=True):
            i = self.indice[u]
            j = self.indice[v]
            if i == j:
                continue
            if i > j:
                i, j = j, i
            if (i, j) in seen:
                continue
            seen.add((i, j))

            i_apo = i < nwj
            j_apo = j < nwj
            is_apo = i_apo and j_apo
            is_pd  = (not i_apo) and (not j_apo)
            is_mem = not (is_apo or is_pd)

            # Mode filtering identical to build_advection_matrix / diffusion.
            if self.mode == 'apo' and not is_apo:
                continue
            if self.mode == 'sym' and not is_pd:
                continue

            De = float(D_full[i, j])

            # Advective volume flux i→j = pressure flux + osmotic membrane flux,
            # scaled by the membrane carry factor f = 1 − σ.
            F = flux_ij.get((i, j), 0.0)
            if is_mem:
                cell_node = v if not j_apo else u   # the cell end of the membrane
                cg = _cgroup_canonical(graph.nodes[cell_node].get('cgroup', 4))
                sig = float(self.sigma.get(cg, 0.0))
                factor = 1.0 - sig
                # Osmotic Kedem–Katchalsky flux F_os = K σ (ψ_os,wall − ψ_os,cell),
                # oriented wall→cell.  Rewrite in the canonical i→j orientation.
                if sig != 0.0:
                    K = eattr.get('K', 0.0)
                    if K:
                        wall_id = i if i_apo else j
                        cell_id = j if i_apo else i
                        wall_obj = cm.get_wall_by_node_id(wall_id)
                        cell_obj = cm.get_by_node_id(cell_id)
                        if wall_obj is not None and cell_obj is not None:
                            F_os_wc = K * sig * (_os(wall_obj) - _os(cell_obj))
                            # F_os_wc is wall→cell; convert to i→j.
                            F += F_os_wc if (wall_id == i) else -F_os_wc
            else:
                factor = 1.0

            _emit(i, j, De, F * factor)

        T = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

        # Slice to the active mode.  build_diffusion_matrix applies a boundary
        # cross-sum diagonal correction when slicing to 'apo'/'sym' (to restore
        # row-sum-0 after dropping cross-mode columns); replicate it here so the
        # sliced T stays mass-conserving.
        if self.mode == 'apo':
            cross = np.asarray(T[:nwj, nwj:].sum(axis=1)).ravel()
            T = (T[:nwj, :nwj] + sp.diags(cross, format='csr')).tocsr()
        elif self.mode == 'sym':
            cross = np.asarray(T[nwj:, :nwj].sum(axis=1)).ravel()
            T = (T[nwj:, nwj:] + sp.diags(cross, format='csr')).tocsr()

        # Keep only the latest operator per structural key: under dynamic
        # coupling every iteration produces a new fingerprint, so drop stale
        # entries that share the same (scheme, mode, indices) prefix to bound
        # the cache to one matrix per configuration.
        struct = cache_key[:-1]
        for k in [k for k in self._T_cache if k[:-1] == struct]:
            del self._T_cache[k]
        self._T_cache[cache_key] = T
        return T

    def _diffusion_full(self, h: int, i_maturity: int) -> sp.csr_matrix:
        """Return the FULL (n_total × n_total) diffusion matrix regardless of
        mode.  build_diffusion_matrix slices to the active mode; the per-edge
        transport-operator build needs the full off-diagonals to read per-edge
        conductances D_e for every edge type.  Cached separately.
        """
        cache_key = ('full', h, i_maturity)
        if cache_key in self._D_cache:
            return self._D_cache[cache_key]
        saved_mode = self.mode
        self.mode = 'full'
        try:
            self._D_cache.pop((h, i_maturity), None)   # avoid returning a slice
            D_full = self.build_diffusion_matrix(h, i_maturity)
        finally:
            self.mode = saved_mode
            self._D_cache.pop((h, i_maturity), None)   # clear the full-mode entry
        self._D_cache[cache_key] = D_full
        return D_full

    def _build_osmotic_advection(self, i_maturity: int) -> sp.csr_matrix:
        """
        Osmotic contribution to advective solute transport at membrane edges.

        edge_flux_list captures only the pressure-driven volume flux
            F_p = K × (ψ_p,wall − ψ_p,cell).
        The true Kedem-Katchalsky volume flux is
            J_v = K × (Δψ_p + σ Δψ_os)
        so the missing term is
            F_os = K × σ × (ψ_os,wall − ψ_os,cell).
        The advective solute flux couples with factor (1−σ), identical to the
        pressure-driven path, giving an additional upwind contribution
            ΔJ_s = (1−σ) × F_os × c_upstream.

        osmotic potentials are read from wall_obj.psi_os / cell_obj.psi_os,
        which are set by mecha.initialize_scenarios().  If both are zero (or
        None) for every membrane edge the returned matrix is all-zero, so
        calling this when no osmotic scenario has been set is safe.
        """
        nwj = self.n_wall_junction
        n   = self.n_total
        rows, cols, data = [], [], []
        cm = self.network.cell_manager

        def _os(obj):
            v = getattr(obj, 'psi_os', None)
            return float(v) if (v is not None and not np.isnan(float(v))) else 0.0

        for node, edges in self.network.graph.adjacency():
            i = self.indice[node]
            if i >= nwj:
                continue  # only start from wall nodes to avoid double-counting

            for neighbor, eattr in edges.items():
                j = self.indice[neighbor]
                if j <= i or eattr.get('path') != 'membrane':
                    continue

                K = eattr.get('K', 0.0)
                if not K:
                    continue

                cg  = _cgroup_canonical(
                    self.network.graph.nodes[neighbor].get('cgroup', 4))
                sig = float(self.sigma.get(cg, 0.0))
                if sig == 0.0:
                    continue

                wall_obj = cm.get_wall_by_node_id(i)
                cell_obj = cm.get_by_node_id(j)
                if wall_obj is None or cell_obj is None:
                    continue

                F_os = K * sig * (_os(wall_obj) - _os(cell_obj))
                if F_os == 0.0:
                    continue

                factor = 1.0 - sig   # Kedem-Katchalsky advection coupling

                if F_os > 0.0:       # osmotic drive: wall → cell (wall is upstream)
                    for r, c, v in ((i, i, -F_os * factor),
                                    (j, i,  F_os * factor)):
                        rows.append(r); cols.append(c); data.append(v)
                else:                # osmotic drive: cell → wall (cell is upstream)
                    for r, c, v in ((j, j,  F_os * factor),
                                    (i, j, -F_os * factor)):
                        rows.append(r); cols.append(c); data.append(v)

        return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

    # ------------------------------------------------------------------
    # Capacitance  C  (diagonal)
    # ------------------------------------------------------------------

    def _compute_node_volumes(self, i_maturity: int) -> np.ndarray:
        """Node volumes (cm³) for all n_total nodes."""
        height    = float(self.mecha.geometry.maturity_stages[i_maturity].get('height'))
        thickness = self.mecha.geometry.thickness
        network   = self.network
        vols      = np.zeros(self.n_total)

        for wid in range(network.n_walls):
            L = network.wall_lengths.get(wid, 0.0)
            vols[wid] = L * height * thickness * 1e-12

        for j in range(network.n_walls, network.n_wall_junction):
            L = network.wall_lengths.get(j, 0.0)
            vols[j] = height * thickness * L / 2.0 * 1e-12

        for cid in range(network.n_cells):
            vols[network.n_wall_junction + cid] = (
                network.cell_areas[cid] * height * 1e-12)

        # Prevent zero-volume nodes from creating singular rows in Cm − A.
        # Nodes with L=0 (degenerate walls) carry c_prev unchanged at eps→0.
        return np.maximum(vols, 1e-30)

    def build_capacitance(self, i_maturity: int) -> sp.csr_matrix:
        """
        Diagonal storage matrix C/dt for Crank-Nicolson.

        Returns a zero matrix when 'dt' is absent (steady-state mode).

        Diagonal entry i:
            (C/dt)_i = storage_fraction_i × V_i / dt   [cm³/d]
        so that [C/dt − θ T] has consistent units with T [cm³/d × mol/cm³ = mol/d].
        """
        n   = self._matrix_size
        nwj = self.n_wall_junction

        if (self.capacitance_params is None
                or self.capacitance_params.get('dt') is None):
            return sp.diags(np.zeros(n), format='csr')

        dt  = float(self.capacitance_params['dt'])
        C_w = float(self.capacitance_params.get('C_wall', 1.0))
        C_c = float(self.capacitance_params.get('C_cell', 1.0))

        vols     = self._compute_node_volumes(i_maturity)
        cap_diag = np.zeros(n)

        if self.mode == 'full':
            cap_diag[:nwj] = C_w * vols[:nwj] / dt
            cap_diag[nwj:] = C_c * vols[nwj:] / dt
        elif self.mode == 'apo':
            cap_diag[:] = C_w * vols[:nwj] / dt
        else:
            cap_diag[:] = C_c * vols[nwj:] / dt

        return sp.diags(cap_diag, format='csr')

    # ------------------------------------------------------------------
    # Reaction (first-order degradation) matrix R
    # ------------------------------------------------------------------

    def build_reaction_matrix(self, i_maturity: int) -> sp.csr_matrix:
        """
        Diagonal first-order degradation matrix.

        Diagonal entry i:
            R_i = k_deg × V_i   [cm³/d]
        so that subtracting R from T adds a −k_deg·c_i sink term to
        C dc/dt = (T c) + s, consistent with the units of D and A.
        """
        n   = self._matrix_size
        nwj = self.n_wall_junction

        if self.k_deg <= 0.0:
            return sp.diags(np.zeros(n), format='csr')

        vols = self._compute_node_volumes(i_maturity)
        react_diag = np.zeros(n)

        if self.mode == 'full':
            react_diag[:] = self.k_deg * vols
        elif self.mode == 'apo':
            react_diag[:] = self.k_deg * vols[:nwj]
        else:
            react_diag[:] = self.k_deg * vols[nwj:]

        return sp.diags(react_diag, format='csr')

    # ------------------------------------------------------------------
    # Peclet number diagnostic
    # ------------------------------------------------------------------

    def _warn_peclet(self, D: sp.csr_matrix, A: sp.csr_matrix) -> None:
        """
        Warn when mesh Peclet number Pe = |A_diag| / |D_diag| > 2.

        At Pe > 2 the Crank-Nicolson scheme (θ=0.5) may produce spurious
        spatial oscillations even though the solution remains bounded.
        CN is unconditionally stable for pure diffusion but NOT for
        advection-diffusion at high Pe. Use theta=1.0 (implicit Euler)
        or reduce dt to restore oscillation-free behavior.
        """
        D_diag = np.abs(D.diagonal())
        A_diag = np.abs(A.diagonal())
        mask   = D_diag > 0.0
        if not mask.any():
            return
        max_pe = (A_diag[mask] / D_diag[mask]).max()
        if max_pe > 2.0:
            warnings.warn(
                f"Max mesh Peclet number ≈ {max_pe:.2g} > 2.  "
                "Crank-Nicolson may produce spurious oscillations.  "
                "Consider theta=1.0 (implicit Euler) or a smaller time step.",
                UserWarning,
                stacklevel=3,
            )

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self, h: int, i_maturity: int, i_scenario: int,
              rhs: np.ndarray,
              boundary_conditions: dict = None,
              c_prev: np.ndarray = None,
              theta: float = 0.5,
              operators: str = 'T',
              scheme: str = 'upwind') -> np.ndarray:
        """
        Solve the transport equation for nodal concentrations.

        Steady-state (c_prev=None or no 'dt'):
            T c = rhs
            rhs = 0 for source-free nodes; rhs[i] = c0 at Dirichlet BC nodes
            (after row override).  For an internal production source Q > 0:
            rhs[i] = −Q  (see sign convention in class docstring).

        Dynamic Crank-Nicolson:
            [C/dt − θ T] c_new = [C/dt + (1−θ) T] c_old + rhs

        Parameters
        ----------
        h : int
            Hydraulic index (selects diffusion matrix cache entry).
        i_maturity, i_scenario : int
        rhs : ndarray, shape (n_size,)
            Right-hand side vector.  For pure Dirichlet BCs pass zeros and
            supply concentrations via boundary_conditions.
        boundary_conditions : dict {node_id: c_value}, optional
            Dirichlet BCs.  node_id in full network indices (0..n_total-1);
            auto-shifted to matrix index in 'sym' mode.
        c_prev : ndarray, shape (n_size,), optional
            Previous-step concentrations (required for CN stepping).
        theta : float
            θ ∈ [0, 1].  Default 0.5 (Crank-Nicolson).
        operators : str
            Which spatial operators to include in T:
              'T'  (default)  T = D + A  full advection-diffusion
              'D'             T = D      diffusion only
              'A'             T = A      advection only
        scheme : str
            Spatial discretization of the FULL advection-diffusion operator:
              'upwind' (default)  first-order upwind advection + diffusion
              'sg'                Scharfetter–Gummel exponentially-fitted flux.
            SG couples D and A per edge and CANNOT be written as D + A_sg, so it
            is only meaningful for operators='T': the whole operator is built in
            one per-edge pass by build_transport_operator.  Requesting scheme='sg'
            with operators='D' or 'A' raises ValueError.

        Returns
        -------
        ndarray, shape (n_size,)
            Concentrations.  Order: walls 0..n_walls-1,
            junctions n_walls..n_wall_junction-1, cells n_wall_junction+.
            In 'sym' mode: cell_id 0..n_cells-1.
        """
        if operators not in ('D', 'A', 'T'):
            raise ValueError(f"operators must be 'D', 'A', or 'T'; got '{operators!r}'")
        if scheme not in ('upwind', 'sg'):
            raise ValueError(f"scheme must be 'upwind' or 'sg'; got {scheme!r}")
        if scheme == 'sg' and operators != 'T':
            raise ValueError(
                "scheme='sg' requires operators='T': the Scharfetter–Gummel "
                "operator fuses diffusion and advection per edge and cannot be "
                "split into a diffusion-only or advection-only piece."
            )

        n    = self._matrix_size
        zero = sp.csr_matrix((n, n))

        if scheme == 'sg':
            # Full SG operator built (and cached) in one per-edge pass, with D
            # and A on the same footing.  SG is exponentially fitted and stable
            # at any Peclet number, so it needs neither the separate D/A
            # matrices nor the Peclet oscillation warning below.
            T = self.build_transport_operator(h, i_maturity, i_scenario, scheme='sg')
        else:
            D = self.build_diffusion_matrix(h, i_maturity) if operators in ('D', 'T') else zero
            A = (self.build_advection_matrix(i_maturity, i_scenario)
                 if operators in ('A', 'T') else zero)
            R = self.build_reaction_matrix(i_maturity)
            T = D + A - R

        Cm      = self.build_capacitance(i_maturity)
        rhs_eff = np.asarray(rhs, dtype=float).copy()

        is_dynamic = (
            c_prev is not None
            and self.capacitance_params is not None
            and self.capacitance_params.get('dt') is not None
        )

        if is_dynamic:
            if scheme != 'sg' and theta < 1.0:   # IE (θ=1) & SG are Pe-stable
                self._warn_peclet(D, A)
            # [C/dt − θ T] c_new = [C/dt + (1−θ) T] c_old + rhs
            lhs     = Cm - theta * T
            rhs_eff = (Cm + (1.0 - theta) * T).dot(c_prev) + rhs_eff
        else:
            lhs = T

        # Dirichlet BCs: row-override so c[idx] = c_val exactly
        if boundary_conditions:
            nwj = self.n_wall_junction
            lhs = lhs.tolil()
            for node_id, c_val in boundary_conditions.items():
                idx = node_id - nwj if self.mode == 'sym' else node_id
                if 0 <= idx < self._matrix_size:
                    lhs[idx, :] = 0.0
                    lhs[idx, idx] = 1.0
                    rhs_eff[idx] = float(c_val)
            lhs = lhs.tocsr()

        return spla.spsolve(lhs, rhs_eff)
