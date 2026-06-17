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
      T  — spatial transport operator  T = D + A
           D = diffusion matrix  (negative diagonal / positive off-diagonal)
           A = advection matrix  (upwind, same sign convention)
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
        Barrier geometry (Casparian strip, exodermis suberisation) is encoded
        in the hydraulic temp_factor = kw_barrier/kw and carried through
        unchanged to the diffusion matrix via HydraulicMatrixBuilder.
    capacitance_params : dict, optional
        {
            'dt'    : float  time step (days); None → steady state
            'C_wall': float  apoplast storage fraction (0–1; default 1.0)
            'C_cell': float  symplast storage fraction (0–1; default 1.0)
        }
    mode : str
        'full'  n_total × n_total
        'apo'   n_wall_junction × n_wall_junction
        'sym'   n_cells × n_cells
    """

    def __init__(self, mecha, diffusion_params: dict,
                 capacitance_params: dict = None, mode: str = 'full'):
        if mode not in ('full', 'apo', 'sym'):
            raise ValueError(f"mode must be 'full', 'apo', or 'sym'; got '{mode}'")

        self.mecha              = mecha
        self.network            = mecha.network
        self.diffusion_params   = diffusion_params
        self.capacitance_params = capacitance_params
        self.mode               = mode

        self.D_wall = float(diffusion_params.get('apo_wall',      0.0))
        self.D_pd   = float(diffusion_params.get('plasmodesmata', 0.0))
        self.D_mem  = float(diffusion_params.get('membrane',      0.0))
        self.sigma  = diffusion_params.get('sigma', {})

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

        Assembly (negative diagonal / positive off-diagonal, matching D):
          F > 0:  A[src,src] −= F·f,   A[tgt,src] += F·f
          F < 0:  A[tgt,tgt] += F·f,   A[src,tgt] −= F·f
        where  f = 1 − σ  (σ = reflection coefficient for membrane edges).
        """
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

        if self.mode == 'apo':
            return A[:nwj, :nwj]
        if self.mode == 'sym':
            return A[nwj:, nwj:]
        return A

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
    # Peclet number diagnostic
    # ------------------------------------------------------------------

    def _warn_peclet(self, D: sp.csr_matrix, A: sp.csr_matrix) -> None:
        """
        Warn when mesh Peclet number Pe = |A_diag| / |D_diag| > 2.

        At Pe > 2 the Crank-Nicolson scheme (θ=0.5) may produce spurious
        spatial oscillations even though the solution remains bounded.
        CN is unconditionally stable for pure diffusion but NOT for
        advection-diffusion at high Pe.  Use theta=1.0 (implicit Euler)
        or reduce dt to restore oscillation-free behaviour.
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
              operators: str = 'T') -> np.ndarray:
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

        Returns
        -------
        ndarray, shape (n_size,)
            Concentrations.  Order: walls 0..n_walls-1,
            junctions n_walls..n_wall_junction-1, cells n_wall_junction+.
            In 'sym' mode: cell_id 0..n_cells-1.
        """
        if operators not in ('D', 'A', 'T'):
            raise ValueError(f"operators must be 'D', 'A', or 'T'; got '{operators!r}'")

        n    = self._matrix_size
        zero = sp.csr_matrix((n, n))

        D = self.build_diffusion_matrix(h, i_maturity) if operators in ('D', 'T') else zero
        A = self.build_advection_matrix(i_maturity, i_scenario) if operators in ('A', 'T') else zero
        T = D + A

        Cm      = self.build_capacitance(i_maturity)
        rhs_eff = np.asarray(rhs, dtype=float).copy()

        is_dynamic = (
            c_prev is not None
            and self.capacitance_params is not None
            and self.capacitance_params.get('dt') is not None
        )

        if is_dynamic:
            if theta < 1.0:   # IE (θ=1) is oscillation-free at any Pe
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
