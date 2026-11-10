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
                              FALLBACK reflection coefficient (0–1) used only
                              when the per-edge membrane/PD σ set by mecha
                              (mb.sigma / pd.sigma, already baked into
                              edge_flux_list) is unavailable; default 0
            'sigma_casp'    : optional Casparian-band reflection coefficient (0-1)
                              for the advective carry factor f = 1 - sigma on
                              the band's wall-junction edges: a float (both
                              bands) or {'endo': s, 'exo': s} (a missing key
                              falls back to the hydraulic graph's 'sigma_wall',
                              i.e. HydraulicData.sigma_casp_*). Overrides the
                              graph value; applies ONLY to edges carrying
                              'casp' (never to other apoplastic edges). sigma is
                              solute-specific (e.g. NaCl vs mannitol). The
                              osmotic volume flux K*sigma*d_pi is already in the
                              edge fluxes (compute_edge_flows), so no extra
                              osmotic coupling is added for walls.
            'membrane_pd'   : optional. Structure-aware membrane diffusive
                              permeability [cm/d] (MECHA units cm, d):
                              ``dict(lipid=P_l, aqp=P_a)`` or a scalar (= lipid
                              only; 1 m/s = 8.64e6 cm/d). When given it REPLACES
                              the uniform 'membrane' diffusivity:
                                P = P_l + P_a * f_aqp   (f_aqp = the cell's
                                    HydraulicData aquaporin factor, default 1)
                              in series with a suberin lamella if the hydraulic
                              lamella model is on:
                                P_lam = D_wall (kw_lam/kw) / s,  s in cm
                                P_eff = 1 / (1/P + 1/P_lam)
                              conductance = P_eff * (height + dist) * length * 1e-8
                              [cm³/d] (the hydraulic membrane area). Intercellular
                              (air-space) cells get zero membrane diffusion.
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
        self.sigma_casp_override = diffusion_params.get('sigma_casp', None)
        mpd = diffusion_params.get('membrane_pd')
        if mpd is None:
            self.membrane_pd = None
        elif isinstance(mpd, dict):
            self.membrane_pd = (float(mpd.get('lipid', 0.0)), float(mpd.get('aqp', 0.0)))
        else:
            self.membrane_pd = (float(mpd), 0.0)
        self._last_solve = None   # (h, i_maturity, i_scenario, scheme, operators) of the last solve()
        self.k_deg  = float(reaction_params.get('k_deg', 0.0)) if reaction_params else 0.0

        # Advective solute exchange with the domain boundary (soil contact,
        # xylem/phloem axial BCs, elongation growth withdrawal). See
        # _build_boundary_advection. Set False to reproduce the older,
        # non-conservative behaviour.
        self.conserve_boundary = bool(diffusion_params.get('conserve_boundary', True))
        # Concentration of the water the boundary ADDS to the domain. Scalar,
        # or {node_id: value}; defaults to clean water.
        self.c_boundary = diffusion_params.get('c_boundary', 0.0)

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

        # Mesophyll air-space / sub-stomatal chamber cells carry a VAPOUR DOF
        # (coupled only by wall_air / air_link edges whose edge_flux_list fluxes
        # are vapour volume fluxes). Dissolved solute must never enter them, so
        # exclude any edge with an air-cell endpoint from D/A/T. Empty for roots.
        self._air_nodes: frozenset = self._collect_air_nodes()

    def _collect_air_nodes(self) -> frozenset:
        """Full-network node ids of mesophyll air spaces (vapour-only cells).

        Mirrors NetworkBuilder.is_wall_air_cell (cell_type == "air space" and
        protect_topology); falls back to the node attributes directly when that
        predicate is unavailable. Returns an empty set for anatomies without air
        spaces (e.g. roots), making the air-cell exclusion a no-op there.
        """
        network = self.network
        nwj     = self.n_wall_junction
        graph   = network.graph
        is_air  = getattr(network, 'is_wall_air_cell', None)
        nodes   = set()
        for cid in range(self.n_cells):
            node_id = nwj + cid
            if node_id not in graph:
                continue
            flagged = False
            if callable(is_air):
                try:
                    flagged = bool(is_air(node_id))
                except Exception:
                    flagged = False
            if not flagged:
                nd = graph.nodes[node_id]
                ct = str(nd.get('cell_type', '')).strip().lower()
                flagged = ct == 'air space' and bool(nd.get('protect_topology', False))
            if flagged:
                nodes.add(self.indice[node_id])
        return frozenset(nodes)

    # ------------------------------------------------------------------
    # Reflection coefficient σ
    # ------------------------------------------------------------------
    def _edge_sigma(self, i: int, j: int, is_mem: bool, is_pd: bool,
                    cell_node: int) -> float:
        """Per-edge reflection coefficient σ, consistent with mecha_class.

        mecha.compute_edge_flows bakes the Kedem–Katchalsky osmotic term with
        the per-edge membrane/PD σ (``mb.sigma`` / ``pd.sigma``) into
        edge_flux_list.  The solute carry factor (1−σ) must use the SAME σ, so
        read it from the same membrane/PD object.  Fall back to the per-cgroup
        ``diffusion_params['sigma']`` dict only when the object is unavailable.
        """
        cm = self.network.cell_manager
        if is_mem:
            mb = cm.get_membrane_by_edge(i, j)
            if mb is not None and mb.sigma is not None:
                return float(mb.sigma)
        elif is_pd:
            pd = cm.get_plasmodesmata_by_edge(i, j)
            if pd is not None and pd.sigma is not None:
                return float(pd.sigma)
        else:
            return 0.0   # walls/junctions: no reflection
        cg = _cgroup_canonical(
            self.network.graph.nodes[cell_node].get('cgroup', 4))
        return float(self.sigma.get(cg, 0.0))

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
            _, matrix_C, _, _, _, _, _, _, _ = builder.build(
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

        if self.membrane_pd is not None and self.mode != 'apo':
            D = D + self._build_membrane_diffusion_pd(h, i_maturity)
        elif self.D_mem > 0.0 and self.mode != 'apo':
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
                # No transmembrane solute diffusion into a vapour air space.
                if j in self._air_nodes:
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

    def _build_membrane_diffusion_pd(self, h: int, i_maturity: int) -> sp.csr_matrix:
        """Structure-aware passive transmembrane diffusion (``membrane_pd`` mode).

        Per membrane edge (wall i, cell j):
            P     = P_lipid + P_aqp * f_aqp(cell)            [cm/d]
            P_lam = D_wall * (kw_lam / kw) / (s * 1e-4)       [cm/d]  (lamella only)
            P_eff = 1 / (1/P + 1/P_lam)
            DF    = P_eff * (height + dist) * length * 1e-8   [cm3/d]
        Air-space (intercellular) cells: DF = 0. ``kw_lam`` / ``s`` are those the
        hydraulic builder used for the membrane series term (edge attributes
        ``kw_lam`` / ``s_lam``, only present with ``suberin_model='lamella'``).
        """
        P_l, P_a = self.membrane_pd
        height = float(self.mecha.geometry.maturity_stages[i_maturity].get('height'))
        hyd = self.mecha.hydraulic
        kw = float(hyd.get_kw_value(h))
        cm = self.mecha.network.cell_manager
        air_nodes = {c.node_id for c in cm.intercellular}
        nwj, n = self.n_wall_junction, self.n_total
        rows, cols, data = [], [], []

        for node, edges in self.network.graph.adjacency():
            i = self.indice[node]
            if i >= nwj:
                continue
            for neighbor, eattr in edges.items():
                j = self.indice[neighbor]
                if j <= i or eattr.get('path') != 'membrane':
                    continue
                cell = cm.get_by_node_id(neighbor)
                if neighbor in air_nodes:
                    P_eff = 0.0
                else:
                    f = hyd.get_aqp_factor(cell.cell_id, cell.cell_type) if cell is not None else None
                    P = P_l + P_a * (1.0 if f is None else f)
                    kw_lam = eattr.get('kw_lam')
                    s_lam = eattr.get('s_lam')
                    if P <= 0.0:
                        P_eff = 0.0
                    elif kw_lam is not None and s_lam:
                        P_lam = self.D_wall * (kw_lam / kw) / (s_lam * 1e-4) if kw > 0 else 0.0
                        P_eff = 0.0 if P_lam <= 0.0 else 1.0 / (1.0 / P + 1.0 / P_lam)
                    else:
                        P_eff = P
                DF = P_eff * (height + eattr.get('dist', 1.0)) * eattr.get('length', 1.0) * 1e-8
                eattr['membrane_diffusivity'] = DF
                if DF == 0.0:
                    continue
                for r, c, v in ((i, i, -DF), (i, j, DF), (j, j, -DF), (j, i, DF)):
                    rows.append(r); cols.append(c); data.append(v)

        return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

    def _wall_edge_sigma(self, u, v) -> float:
        """Reflection coefficient of apoplastic edge (u, v): non-zero only for the
        wall-junction edges of a Casparian band (edge attribute ``casp``). The
        ``diffusion_params['sigma_casp']`` override (float, or {'endo': s, 'exo': s};
        a missing key falls back to the graph) beats the graph's ``sigma_wall``."""
        ea = self.network.graph.get_edge_data(u, v)
        band = ea.get('casp') if ea else None
        if band is None:
            return 0.0
        ov = self.sigma_casp_override
        if ov is not None:
            if isinstance(ov, dict):
                if band in ov:
                    return float(ov[band])
            else:
                return float(ov)
        return float(ea.get('sigma_wall', 0.0))

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
        F is the FULL Kedem–Katchalsky volume flux K(Δψ_p + σΔψ_os) already
        assembled by mecha.compute_edge_flows, so the osmotic membrane term is
        NOT re-added here.

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

            # wall_air / air_link edges carry a VAPOUR volume flux; the dissolved
            # solute must not be advected into the air space. Skip any edge with
            # an air-cell endpoint.
            if src in self._air_nodes or tgt in self._air_nodes:
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
                factor = 1.0 - self._edge_sigma(src, tgt, is_mem, is_pd, cell_node)
            elif is_apo:
                factor = 1.0 - self._wall_edge_sigma(src, tgt)
            else: # need to check for pd
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
        if self.conserve_boundary:
            A = A + self._build_boundary_advection(i_maturity, i_scenario)

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

    def _boundary_flux(self, i_maturity: int, i_scenario: int) -> np.ndarray:
        """Net water flux each node exchanges with the domain boundary [cm³/d].

        The edge list only describes water moving *between* nodes.  Water also
        crosses the domain boundary — soil uptake at the wetted walls, the
        xylem and phloem axial boundary conditions, and the elongation growth
        term that withdraws water from every living cell.  None of that appears
        as a graph edge, so it shows up as a residual: at steady state the
        signed edge fluxes at a node must be balanced by its boundary exchange.

        Returns ``S`` with ``S[n] > 0`` where the boundary ADDS water to node
        ``n`` and ``S[n] < 0`` where it REMOVES water.
        """
        S = np.zeros(self.n_total)
        for edge in self.mecha.edge_flux_list[i_maturity][i_scenario]:
            F = edge['flux']
            if F == 0.0:
                continue
            # F > 0 flows source -> target.
            S[edge['source']] += F
            S[edge['target']] -= F
        return S

    def _build_boundary_advection(self, i_maturity: int,
                                  i_scenario: int) -> sp.csr_matrix:
        """Advective solute exchange with the domain boundary.

        Without this term the operator conserves solute only along edges: at a
        node the boundary drains, solute keeps arriving by advection and is
        never carried out with the water, so the concentration climbs above the
        source value (measured up to 1.8× a c=1 Dirichlet source on the
        partially-wetted Arabidopsis section, where cell elongation withdraws
        water from every epidermis and cortex cell).

        Water LEAVING carries the local concentration out — a negative
        diagonal.  Water ENTERING carries ``self.c_boundary``; that is a
        constant, so it belongs in the right-hand side rather than the
        operator, and is only represented here when it is zero (the default:
        soil and phloem water carrying none of the tracer).
        """
        S = self._boundary_flux(i_maturity, i_scenario)
        outflow = np.minimum(S, 0.0)          # ≤ 0, keeps the sign convention
        idx = np.nonzero(outflow)[0]
        return coo_matrix(
            (outflow[idx], (idx, idx)), shape=(self.n_total, self.n_total)
        ).tocsr()

    def boundary_flux_rhs(self, i_maturity: int, i_scenario: int) -> np.ndarray:
        """Right-hand side contribution of boundary water that carries solute.

        Zero unless ``c_boundary`` is non-zero.  Pair it with ``solve(rhs=...)``
        when the incoming soil or phloem water is not clean.
        """
        S = self._boundary_flux(i_maturity, i_scenario)
        inflow = np.maximum(S, 0.0)
        if isinstance(self.c_boundary, dict):
            c_bc = np.array([float(self.c_boundary.get(n, 0.0))
                             for n in range(self.n_total)])
        else:
            c_bc = float(self.c_boundary)
        return inflow * c_bc

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
        graph = self.network.graph

        # Diffusion off-diagonals D_e per node pair (full network).  Each simple
        # graph edge is a single physical link, so D[i,j] is that edge's D_e.
        D_full = self._diffusion_full(h, i_maturity)

        # ── Per-edge advective volume flux F_ij (i<j) ──────────────────────────
        # edge_flux_list holds one entry per active edge with a signed flux and a
        # source→target orientation; re-key to canonical (min,max) with a sign.
        # F is the FULL Kedem–Katchalsky flux K(Δψ_p + σΔψ_os) from mecha.
        flux_ij: dict = {}
        for edge in self.mecha.edge_flux_list[i_maturity][i_scenario]:
            src, tgt, F = edge['source'], edge['target'], edge['flux']
            if F == 0.0:
                continue
            a, b = (src, tgt) if src < tgt else (tgt, src)
            s = 1.0 if src < tgt else -1.0
            flux_ij[(a, b)] = flux_ij.get((a, b), 0.0) + s * F

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

            # Air spaces (sub-stomatal chambers) hold a vapour DOF; their
            # wall_air / air_link edges carry vapour, not dissolved solute.
            if i in self._air_nodes or j in self._air_nodes:
                continue

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

            # Advective volume flux i→j is the FULL Kedem–Katchalsky flux
            # K(Δψ_p + σΔψ_os) already in edge_flux_list; it is only scaled by
            # the membrane carry factor f = 1 − σ.
            F = flux_ij.get((i, j), 0.0)
            if is_mem:
                cell_node = v if not j_apo else u   # the cell end of the membrane
                factor = 1.0 - self._edge_sigma(i, j, is_mem, is_pd, cell_node)
            elif is_apo:
                factor = 1.0 - self._wall_edge_sigma(u, v)   # wall reflection (default 0)
            else: # need to check for pd
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

    # ------------------------------------------------------------------
    # Capacitance C and reaction R — both are diagonal, volume-weighted,
    # mode-sliced matrices; only their per-node coefficient and gating
    # condition differ, so that scaffolding lives in one helper.
    # ------------------------------------------------------------------

    def _build_node_diagonal(self, i_maturity: int, coeff_wall: float,
                              coeff_cell: float, gate: bool) -> sp.csr_matrix:
        """
        Diagonal matrix  diag_i = coeff_wall × V_i  (wall/junction nodes)
                          diag_i = coeff_cell × V_i  (cell nodes)
        sliced to the active mode ('full'/'apo'/'sym').  Returns an all-zero
        matrix when `gate` is False, without touching node volumes.
        """
        n = self._matrix_size
        if not gate:
            return sp.diags(np.zeros(n), format='csr')

        nwj  = self.n_wall_junction
        vols = self._compute_node_volumes(i_maturity)
        diag = np.zeros(n)

        if self.mode == 'full':
            diag[:nwj] = coeff_wall * vols[:nwj]
            diag[nwj:] = coeff_cell * vols[nwj:]
        elif self.mode == 'apo':
            diag[:] = coeff_wall * vols[:nwj]
        else:
            diag[:] = coeff_cell * vols[nwj:]

        return sp.diags(diag, format='csr')

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
        gate = (self.capacitance_params is not None
                and self.capacitance_params.get('dt') is not None)
        if not gate:
            return self._build_node_diagonal(i_maturity, 0.0, 0.0, gate=False)

        dt  = float(self.capacitance_params['dt'])
        C_w = float(self.capacitance_params.get('C_wall', 1.0))
        C_c = float(self.capacitance_params.get('C_cell', 1.0))

        return self._build_node_diagonal(i_maturity, C_w / dt, C_c / dt, gate=True)

    def build_reaction_matrix(self, i_maturity: int) -> sp.csr_matrix:
        """
        Diagonal first-order degradation matrix.

        Diagonal entry i:
            R_i = k_deg × V_i   [cm³/d]
        so that subtracting R from T adds a −k_deg·c_i sink term to
        C dc/dt = (T c) + s, consistent with the units of D and A.
        Degradation rate is uniform across wall and cell nodes (unlike
        capacitance, which distinguishes C_wall/C_cell storage fractions).
        """
        return self._build_node_diagonal(i_maturity, self.k_deg, self.k_deg,
                                          gate=self.k_deg > 0.0)

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
              'T'  (default)  T = D + A - R  full advection-diffusion
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
        self._last_solve = (h, i_maturity, i_scenario, scheme, operators)

        R = self.build_reaction_matrix(i_maturity)

        if scheme == 'sg':
            # Full SG operator built (and cached) in one per-edge pass, with D
            # and A on the same footing.  SG is exponentially fitted and stable
            # at any Peclet number, so it needs neither the separate D/A
            # matrices nor the Peclet oscillation warning below.
            T = self.build_transport_operator(h, i_maturity, i_scenario, scheme='sg')
            T -= R 
        else:
            D = self.build_diffusion_matrix(h, i_maturity) if operators in ('D', 'T') else zero
            A = (self.build_advection_matrix(i_maturity, i_scenario)
                 if operators in ('A', 'T') else zero)
            
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

    # ------------------------------------------------------------------
    # Boundary flux diagnostic
    # ------------------------------------------------------------------

    def boundary_flux(self, c: np.ndarray, nodes, h: int = None,
                      i_maturity: int = None, i_scenario: int = None,
                      operators: str = None, scheme: str = None,
                      per_node: bool = False):
        """Net solute flux leaving the domain through the Dirichlet *nodes*.

        Computed from the UNMODIFIED assembled spatial operator T = D + A - R (as
        in :meth:`solve`, i.e. before the Dirichlet row overrides, which only exist
        on a local copy inside ``solve``).  Since (T c)_i is the net flux INTO node
        i from its neighbours, node i must hand exactly that amount to the external
        reservoir holding it at c_i:

            flux = sum_{i in nodes} (T c)_i      [mol/d when c is mol/cm3, since
                                                  T is in cm3/d]

        Sign: POSITIVE = solute leaves the domain through *nodes* (e.g. a c=0 sink);
        NEGATIVE = solute enters through them (e.g. a c=1 source).  For a
        conservative operator (no reaction, steady state) the sums over inlet and
        outlet node sets are equal and opposite.  (This is -(A_bc c) with A_bc the
        flux operator from the boundary node into the domain.)  Capacitance is
        ignored (steady / quasi-steady diagnostic).

        Parameters
        ----------
        c : ndarray, concentrations in the active mode's indexing (as returned by
            :meth:`solve`).
        nodes : iterable of full-network node ids (auto-shifted in 'sym' mode).
        h, i_maturity, i_scenario, scheme, operators : default to those of the
            last ``solve`` call ('T' if none; operators 'T' | 'D' | 'A' as in
            :meth:`solve`).
        per_node : return the per-node array instead of the sum.
        """
        if self._last_solve is None and (h is None or i_maturity is None or i_scenario is None):
            raise ValueError("pass h, i_maturity, i_scenario (no previous solve())")
        lh, lm, ls, lsch, lop = self._last_solve if self._last_solve else (None,) * 5
        operators = (lop or 'T') if operators is None else operators
        if operators not in ('D', 'A', 'T'):
            raise ValueError(f"operators must be 'D', 'A', or 'T'; got {operators!r}")
        h = lh if h is None else h
        i_maturity = lm if i_maturity is None else i_maturity
        i_scenario = ls if i_scenario is None else i_scenario
        scheme = (lsch or 'upwind') if scheme is None else scheme
        if scheme == 'sg' and operators != 'T':
            raise ValueError("scheme='sg' requires operators='T'")

        R = self.build_reaction_matrix(i_maturity)
        if scheme == 'sg':
            T = self.build_transport_operator(h, i_maturity, i_scenario, scheme='sg') - R
        else:
            n = self._matrix_size
            zero = sp.csr_matrix((n, n))
            D = self.build_diffusion_matrix(h, i_maturity) if operators in ('D', 'T') else zero
            A = (self.build_advection_matrix(i_maturity, i_scenario)
                 if operators in ('A', 'T') else zero)
            T = D + A - R

        nwj = self.n_wall_junction
        idx = [(nd - nwj if self.mode == 'sym' else nd) for nd in nodes]
        idx = np.array([k for k in idx if 0 <= k < self._matrix_size], dtype=int)
        Tc = np.asarray(T.dot(np.asarray(c, dtype=float))).ravel()
        return Tc[idx] if per_node else float(Tc[idx].sum())
