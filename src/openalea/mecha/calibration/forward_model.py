"""Forward model for the needle transpiration calibration.

Wraps a single MECHA hydraulic solve into a pure function of the unknown
transport coefficients ``theta`` and the boundary conditions
``(psi_xyl, RH_airspace)``, returning the total transpiration flux that leaves
the cross-section through the mesophyll air spaces.

Boundary conditions
-------------------
* **Xylem water potential** ``psi_xyl`` [hPa] — Dirichlet BC at the xylem cells,
  applied through the existing ``pressure_xyl_prox`` scenario
  field (``Mecha.psi_xyl``).
* **Air-space relative humidity** ``RH_airspace`` [0-1] — Dirichlet BC at the
  mesophyll substomatal air spaces.
  The relative humidity is converted to a liquid-equivalent water potential
  via the Kelvin equation and pinned on those nodes with the same large-conductance 
  penalty technique MECHA already uses for the xylem/phloem Dirichlet BCs.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from openalea.mecha.mecha_class import Mecha
from openalea.mecha.utils.data_loader import InData


# ── Physical constants (fixed room-temperature reference) ──────────────────────
#: Universal gas constant in MECHA's pressure units [hPa cm³ mol⁻¹ K⁻¹].
#: R = 8.314462618 J mol⁻¹ K⁻¹ = 8.314462618e4 hPa cm³ mol⁻¹ K⁻¹
#: (1 J = 1 Pa m³ = 1e-2 hPa · 1e6 cm³ = 1e4 hPa cm³).
R_GAS_HPA_CM3 = 8.314462618e4
#: Molar mass of water [g mol⁻¹].
M_W = 18.01528
#: Fixed room temperature [K] (25 °C).
T_ROOM = 298.15

#: Order of the calibrated transport coefficients in the ``theta`` vector.
PARAM_KEYS: Tuple[str, ...] = ("kw", "kmb", "kaqp", "kpl")


def p_sat_room_temperature(T: float = T_ROOM) -> float:
    """Saturation vapour pressure of water at room temperature.

    Uses the Arden Buck (1981) equation, which is accurate to <0.1 % over the
    physiological range.

    At ``T = 298.15 K`` (25 °C) this returns ``31.69 hPa``

    Parameters
    ----------
    T : float, optional
        Temperature [K].  Defaults to :data:`T_ROOM` (25 °C).

    Returns
    -------
    float
        Saturation vapour pressure [hPa].
    """
    Tc = T - 273.15  # °C
    # Arden Buck equation → kPa, then ×10 → hPa.
    p_kpa = 0.61121 * math.exp((18.678 - Tc / 234.5) * (Tc / (257.14 + Tc)))
    return p_kpa * 10.0


def rh_to_water_potential(rh: float, T: float = T_ROOM) -> float:
    """Convert relative humidity to a liquid-equivalent water potential.

    The air-space water vapour is assumed in local thermodynamic equilibrium

    Parameters
    ----------
    rh : float
        Relative humidity in ``(0, 1]``.
    T : float, optional
        Temperature [K].  Defaults to :data:`T_ROOM`.

    Returns
    -------
    float
        Liquid-equivalent water potential [hPa] (``≤ 0``).
    """
    if not (0.0 < rh <= 1.0):
        raise ValueError(f"RH must be in (0, 1], got {rh}.")
    if rh == 1.0:
        return 0.0
    return (R_GAS_HPA_CM3 * T / M_W) * math.log(rh)


class ForwardModel:
    """Stateless-per-call MECHA transpiration forward model.

    The (expensive) network topology is built **once** at construction and
    reused for every parameter evaluation; only the cheap
    :class:`InData` physics container is rebuilt per call.

    Parameters
    ----------
    network : NetworkBuilder
        A GRANAP-populated network (``populate_from_network`` already called).
    base_input_factory : callable, optional
        Zero-argument callable returning a fresh :class:`InData` configured for
        the needle (defaults to :meth:`InData.needle_defaults`). A new instance
        is produced for every evaluation so parameter overrides never leak
        between calls.
    T : float, optional
        Temperature [K] used for ``p_sat`` and the Kelvin conversion.  Defaults
        to :data:`T_ROOM` (25 °C).
    i_maturity : int, optional
        Maturity-stage index to solve (default 0).

    Notes
    -----
    The external boundary conditions are deliberately kept as *per-call*
    arguments (``psi_xyl``, ``rh_airspace``) rather than baked into the model,
    so the same forward model can reproduce measurements taken under different
    experimental conditions.
    """

    def __init__(
        self,
        network,
        base_input_factory=None,
        T: float = T_ROOM,
        i_maturity: int = 0,
    ) -> None:
        self.network = network
        self.T = float(T)
        self.i_maturity = int(i_maturity)
        self.p_sat = p_sat_room_temperature(self.T)
        if base_input_factory is None:
            self._make_input = InData.needle_defaults
        else:
            self._make_input = base_input_factory

        # Cache the air-space (substomatal) node ids once. These are the
        # mesophyll "air space" cells flagged protect_topology upstream by
        # GRANAP and connected to walls through path='wall_air' edges.
        self._air_space_nodes: List[int] = self._collect_air_space_nodes()
        # Cache the evaporating wall nodes: the wall side of every wall_air
        # edge. These are the transpirationally active surfaces where the
        # air relative humidity fixes the (liquid-equivalent) water potential.
        self._evaporating_wall_nodes: List[int] = self._collect_evaporating_wall_nodes()
        # Cache the transpiring surface area [cm^2] represented by those walls.
        self._transpiring_area_cm2: float = self._compute_transpiring_area()

    # ── transpiring surface area ──────────────────────────────────────────────
    def _segment_height_um(self) -> float:
        """Needle-segment depth (out-of-plane length) of the solved slice [µm].

        MECHA's 2-D cross-section represents a slice of finite depth ``height``; 
        wall-air surfaces are given the area ``wall_length * height``
        in :meth:`HydraulicMatrixBuilder._fill_wall_air`.
        """
        data = self._make_input()
        return float(data.geometry.maturity_stages[self.i_maturity]["height"])

    def _compute_transpiring_area(self) -> float:
        """Total evaporating wall-air interface area [cm^2].

        Mirrors MECHA's own wall-air surface convention
        (:meth:`HydraulicMatrixBuilder._fill_wall_air`):
        """
        wl = self.network.wall_lengths
        height_um = self._segment_height_um()
        total_len_um = 0.0
        for node in self._evaporating_wall_nodes:
            try:
                total_len_um += float(wl[node])
            except (KeyError, TypeError):
                continue
        return total_len_um * height_um * 1.0e-8

    @property
    def transpiring_area_cm2(self) -> float:
        """Total evaporating wall-air interface area [cm^2] (cached)."""
        return self._transpiring_area_cm2

    # ── air-space node discovery ──────────────────────────────────────────────
    def _collect_air_space_nodes(self) -> List[int]:
        """Return graph node ids of the mesophyll substomatal air spaces.

        Detection mirrors :meth:`NetworkBuilder.is_wall_air_cell`: a cell node
        whose ``cell_type`` is ``"air space"`` and whose ``protect_topology``
        flag is set.  Falls back to "any node with a ``wall_air`` edge" if the
        node attributes are unavailable.
        """
        g = self.network.graph
        nodes: List[int] = []

        is_air = getattr(self.network, "is_wall_air_cell", None)
        nwj = self.network.n_wall_junction
        n_cells = self.network.n_cells

        for cid in range(n_cells):
            node_id = nwj + cid
            if node_id not in g:
                continue
            flagged = False
            if callable(is_air):
                try:
                    flagged = bool(is_air(node_id))
                except Exception:
                    flagged = False
            if not flagged:
                # Attribute fallback (identical predicate to is_wall_air_cell).
                nd = g.nodes[node_id]
                ct = str(nd.get("cell_type", "")).strip().lower()
                flagged = ct == "air space" and bool(nd.get("protect_topology", False))
            if flagged:
                nodes.append(node_id)

        # Last-resort fallback: infer from wall_air edges if no node was flagged.
        if not nodes:
            for u, v, eattr in g.edges(data=True):
                if eattr.get("path") == "wall_air":
                    for cand in (u, v):
                        if cand >= nwj and cand not in nodes:
                            nodes.append(cand)
        return nodes

    @property
    def air_space_nodes(self) -> List[int]:
        """Graph node ids of the substomatal air spaces (cached)."""
        return list(self._air_space_nodes)

    def _collect_evaporating_wall_nodes(self) -> List[int]:
        """Return the wall node ids on the wall ↔ air-space interfaces.

        These are the transpirationally active evaporating surfaces. The wall
        node degree of freedom is a water potential [hPa], so the
        air-humidity boundary condition is imposed here by pinning each such
        wall to the RH-equivalent (Kelvin) liquid water potential.

        The wall side of a ``wall_air`` edge is the endpoint that owns a
        :class:`HydraulicWall` in the cell manager (mirrors the orientation
        logic in :meth:`HydraulicMatrixBuilder._fill_wall_air`).
        """
        g = self.network.graph
        cm = self.network.cell_manager
        walls: List[int] = []
        seen = set()
        for u, v, eattr in g.edges(data=True):
            if eattr.get("path") != "wall_air":
                continue
            wall = cm.get_wall_by_node_id(u) or cm.get_wall_by_node_id(v)
            wall_node = None
            if wall is not None:
                wall_node = wall.node_id
            else:
                # Fallback: the wall is the node below n_wall_junction.
                nwj = self.network.n_wall_junction
                wall_node = u if u < nwj else (v if v < nwj else None)
            if wall_node is not None and wall_node not in seen:
                seen.add(wall_node)
                walls.append(wall_node)
        return walls

    @property
    def evaporating_wall_nodes(self) -> List[int]:
        """Graph node ids of the evaporating (wall ↔ air) wall surfaces."""
        return list(self._evaporating_wall_nodes)

    # ── boundary-condition assembly ───────────────────────────────────────────
    def _build_input(self, theta: Dict[str, float]) -> InData:
        """Create a fresh needle :class:`InData` with ``theta`` applied.

        Only the hydraulic transport coefficients listed in :data:`PARAM_KEYS`
        are overridden.  ``p_sat`` / ``psi_ref`` / temperature are pushed onto
        the hydraulic container so the wall-air Kelvin coupling uses the fixed
        room-temperature reference.
        """
        data = self._make_input()
        hyd = data.hydraulic

        if "kw" in theta:
            hyd.kw = [float(theta["kw"])]
        if "kmb" in theta:
            hyd.kmb = float(theta["kmb"])
        if "kaqp" in theta:
            # kaqp is a list of per-scenario dicts; scale the base ``value``.
            for entry in hyd.kaqp:
                entry["value"] = float(theta["kaqp"])
        if "kpl" in theta:
            for entry in hyd.kpl:
                entry["value"] = float(theta["kpl"])

        # Fixed room-temperature vapour parameters for the wall-air coupling.
        hyd.p_sat = self.p_sat
        hyd.psi_ref_wall_air = float(getattr(hyd, "psi_ref_wall_air", 0.0))
        hyd.M_w = M_W
        hyd.R_gas = R_GAS_HPA_CM3
        hyd.T = self.T
        return data

    def _apply_air_space_rh(self, mecha: Mecha, rh: float) -> float:
        """Record the RH-derived Dirichlet value for the evaporating wall nodes.

        The transpirationally active surfaces are the **wall nodes** on the
        ``wall_air`` interfaces. Their degree of freedom is a water potential
        [hPa], so the air relative humidity enters as an exact Dirichlet
        condition via the Kelvin equation (:func:`rh_to_water_potential`):

        This method only stores the target nodes and potential; the actual
        row-elimination is performed inside :meth:`_clean_solve`.

        Returns the imposed evaporating-surface water potential [hPa].
        """
        psi_air = rh_to_water_potential(rh, self.T)
        mecha._calib_air_bc = {
            "nodes": list(self._evaporating_wall_nodes),
            "psi_air": float(psi_air),
        }
        return psi_air

    # ── main entry point ──────────────────────────────────────────────────────
    def transpiration_flux(
        self,
        theta: Dict[str, float],
        psi_xyl: float,
        rh_airspace: float = 1.0,
        output: str = "k_r",
        verbose: bool = False,
    ) -> float:
        """Solve MECHA once and return the transpiration response.

        Parameters
        ----------
        theta : dict
            Transport coefficients to test.  Keys are a subset of
            :data:`PARAM_KEYS`; missing keys keep their needle-default value.
        psi_xyl : float
            Xylem water potential Dirichlet BC [hPa] (supply side).
        rh_airspace : float, optional
            Relative humidity of the substomatal air spaces (sink side),
            in ``(0, 1]``.  Defaults to ``1.0`` (saturated reference).
        output : {"k_r", "Q"}, optional
            Quantity to return (default ``"k_r"``):

            * ``"k_r"`` — **radial conductance** of the whole cross-section
              [cm hPa⁻¹ d⁻¹], normalized by the transpiring wall-air surface
              area and by the water-potential drop across it, exactly as MECHA's
              :meth:`Mecha.standard_water_flow` defines ``kr_tot``::

                  k_r = Q / (|psi_xyl - psi_air| * A_transp)

            * ``"Q"`` — the raw total transpiration flux [cm³ d⁻¹] leaving the
              cross-section through the air spaces.
        verbose : bool, optional
            Print a one-line diagnostic of the boundary conditions.

        Returns
        -------
        float
            ``k_r`` [cm hPa⁻¹ d⁻¹] (default) or ``Q``
            [cm³ d⁻¹].
        """
        if output not in ("k_r", "Q"):
            raise ValueError(f"output must be 'k_r' or 'Q', got {output!r}.")

        data = self._build_input(theta)
        mecha = Mecha(data, network=self.network)

        # Xylem supply-side Dirichlet BC. Stored on the standard psi_xyl array
        # (proximal index 1) for the maturity/scenario being solved.
        mecha.psi_xyl[1, self.i_maturity, :] = float(psi_xyl)
        # No phloem BC in the transpiration configuration.
        mecha.psi_sieve[1, self.i_maturity, :] = np.nan

        # Air relative-humidity Dirichlet BC on the evaporating wall surfaces.
        psi_air = self._apply_air_space_rh(mecha, rh_airspace)
        if verbose:
            print(
                f"[forward_model] psi_xyl={psi_xyl:.1f} hPa  RH={rh_airspace:.3f}"
                f"  -> psi_air={psi_air:.1f} hPa  p_sat={self.p_sat:.2f} hPa"
                f"  ({len(self._evaporating_wall_nodes)} evaporating wall nodes,"
                f" {len(self._air_space_nodes)} air spaces,"
                f" A_transp={self._transpiring_area_cm2:.4e} cm^2)"
            )

        solution, W_orig = self._clean_solve(mecha)
        q_out = self._transpiration_flux_from_bc(mecha, solution, W_orig)
        if output == "Q":
            return q_out

        # Radial conductance: normalise by the driving potential drop and the
        # upstream-derived transpiring area (MECHA kr_tot convention).
        dpsi = abs(float(psi_xyl) - psi_air)
        area = self._transpiring_area_cm2
        if dpsi == 0.0 or area == 0.0:
            return 0.0
        return q_out / dpsi / area

    # ── clean single solve (bypasses solve_W post-processing) ─────────────────
    def _clean_solve(self, mecha: Mecha):
        """Assemble and solve the hydraulic system in one well-posed pass.

        Uses :meth:`Mecha.build_matrices` directly to get the physical matrix
        (which includes the xylem penalty BC via
        ``_apply_xylo_phloem_boundary``) and the persistent wall-air Kelvin
        source ``rhs_wa``, then adds the evaporating-wall RH penalty BC before
        a single solve. Returns the solution vector and a copy of the
        pre-evap-BC matrix (kept for API compatibility).

        Bypasses :meth:`Mecha.solve_W` intentionally: that method calls
        ``remove_xyl_phloem_BC`` which strips the xylem anchor back out of the
        stored matrix, corrupting any subsequent re-solve.
        """
        self.network.cell_manager.reset_hydraulic_properties()
        (matrix_W, _matrix_C, _rhs_C, _rhs_p, _rhs_x, _rhs_s,
         rhs, rhs_wa, _Kmb) = mecha.build_matrices(h=0, i_maturity=self.i_maturity)
        rhs = np.array(rhs, dtype=float) + np.array(rhs_wa, dtype=float)
        psi_xyl_val = float(mecha.psi_xyl[1, self.i_maturity, 0])
        if not np.isnan(psi_xyl_val):
            k_xyl_val = (mecha.hydraulic.k_xyl
                         if not isinstance(mecha.hydraulic.k_xyl, list)
                         else mecha.hydraulic.k_xyl[0])
            for c in mecha.network.cell_manager.xylem:
                idx_xyl = mecha.indice[c.node_id]
                rhs[idx_xyl, 0] = -float(k_xyl_val) * psi_xyl_val
        # Keep a copy of the assembled matrix before the evap-wall penalty is
        # added; used for API compatibility in _transpiration_flux_from_bc.
        W_orig = matrix_W.tocsr().copy()
        matrix_W = matrix_W.tolil()

        # ── Evaporating-wall RH BC (penalty method) ───────────────────────────
        bc = getattr(mecha, "_calib_air_bc", None)
        if bc and bc["nodes"]:
            psi_air_val = bc["psi_air"]
            # Choose penalty 1e3× the largest physical diagonal entry.
            # W_orig diagonal gives the physical scale; exclude the xylem penalty
            # rows (already much larger) to avoid inflating the estimate.
            max_phys_diag = float(np.abs(W_orig.diagonal()).max())
            k_air = max(1e3 * max_phys_diag, 1.0)
            W_lil = matrix_W.tolil()
            rhs_vec = rhs.ravel()
            for node in bc["nodes"]:
                idx = mecha.indice[node]
                W_lil[idx, idx] -= k_air
                rhs_vec[idx] -= k_air * psi_air_val
            matrix_W = W_lil.tocsr()
            rhs = rhs_vec.reshape(-1, 1)
            # Store penalty strength for flux computation.
            bc["k_air"] = k_air
        solution, _ = mecha.solve(
            matrix=matrix_W, rhs=rhs, sparse_matrix=mecha.general.sparse_matrix
        )
        # Store so downstream MECHA utilities (edge flows, visualisation) work.
        mecha.results.append({
            "maturity stage": self.i_maturity, "scenario": "transpiration",
            "solution": solution, "matrix_W": matrix_W, "rhs": rhs,
        })
        mecha.compute_edge_flows(solution, i_maturity=self.i_maturity)
        return solution, W_orig

    # ── flux extraction ───────────────────────────────────────────────────────
    def _transpiration_flux_from_bc(
        self,
        mecha: Mecha,
        solution: np.ndarray,
        W_orig,  # kept for API compatibility; not used in penalty formulation
    ) -> float:
        """Transpiration flux from the evaporating-wall penalty residual.

        Each evaporating-wall node is constrained by a penalty conductance
        ``k_air`` that holds its potential at ``psi_air``. The water
        extracted by the penalty at each node is


            Q_k = k_air(psi_k - psi_air),

        which equals the net liquid-side inflow in steady state. Summing over
        all evaporating walls gives the total transpiration rate
        (positive = water leaving the liquid phase).
        """
        bc = getattr(mecha, "_calib_air_bc", None)
        if not bc or not bc["nodes"]:
            return 0.0
        sol = np.asarray(solution, dtype=float).ravel()
        k_air = bc["k_air"]
        psi_air = bc["psi_air"]
        # Penalty-BC residual: the water extracted by the penalty term at each
        # evaporating wall equals k_air*(ψ_solved − ψ_air). Summed over all
        # evaporating walls this is the total transpiration flux (positive outward).
        q_out = 0.0
        for node in bc["nodes"]:
            idx = mecha.indice[node]
            q_out += k_air * (float(sol[idx]) - psi_air)
        return float(q_out)


def run_mecha_transpiration(
    theta: Dict[str, float],
    psi_xyl: float,
    network,
    rh_airspace: float = 1.0,
    output: str = "k_r",
    T: float = T_ROOM,
    i_maturity: int = 0,
    verbose: bool = False,
) -> float:
    """One-shot convenience wrapper around :class:`ForwardModel`.

    Prefer constructing a :class:`ForwardModel` once and reusing it when many
    evaluations are needed (calibration / sensitivity loops), because it caches
    the air-space node discovery.

    Parameters
    ----------
    theta : dict
        Transport coefficients (subset of :data:`PARAM_KEYS`).
    psi_xyl : float
        Xylem water-potential Dirichlet BC [hPa].
    network : NetworkBuilder
        GRANAP-populated network.
    rh_airspace : float, optional
        Air-space relative humidity in ``(0, 1]`` (default 1.0).
    output : {"k_r", "Q"}, optional
        ``"k_r"`` radial conductance [cm hPa⁻¹ d⁻¹] (default) or ``"Q"`` total
        flux [cm³ d⁻¹]. See :meth:`ForwardModel.transpiration_flux`.
    T : float, optional
        Temperature [K] (default 25 °C).
    i_maturity : int, optional
        Maturity-stage index (default 0).
    verbose : bool, optional
        Verbose solve output.

    Returns
    -------
    float
        Radial conductance ``k_r`` [cm hPa⁻¹ d⁻¹] (default) or total flux ``Q``
        [cm³ d⁻¹].
    """
    fm = ForwardModel(network, T=T, i_maturity=i_maturity)
    return fm.transpiration_flux(
        theta, psi_xyl, rh_airspace, output=output, verbose=verbose
    )
