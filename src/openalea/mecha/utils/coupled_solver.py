"""
coupled_solver.py

Iterative coupling between MECHA's hydraulic solver and solute-transport solver.

The coupling loop:
  1. Solve water flow (mecha.water_flux) with current osmotic potentials.
  2. Solve solute transport (SoluteTransport.solve) to obtain concentrations.
  3. Convert concentrations to osmotic potentials via van't Hoff and write them
     onto HydraulicCell objects (HydraulicCellManager.set_osmotic_from_concentration).
  4. Re-solve water flow with the updated cell osmotic potentials.
  5. Repeat until max |Δψ_total| < tol.

Convergence criterion
---------------------
Convergence is measured on the total water potential ψ_total = ψ_p − σ·ψ_os,
not on ψ_os alone.  This is the physically motivated choice because ψ_total is
the driving force for water flow; converging on ψ_os alone would be either
over-conservative (large Δψ_os but small Δψ_total) or insufficient (ψ_os
plateaued while ψ_p still drifts).

Known limitation — isosmotic cancellation
-----------------------------------------
The criterion has a blind spot: if ψ_p and σ·ψ_os drift in exact opposition
(Δψ_p ≈ −Δ(σ·ψ_os) at every cell), their sum Δψ_total ≈ 0 even though neither
component has equilibrated.  This is not a pathological edge case — it is the
normal isosmotic-flow regime where turgor and osmotic potential co-vary to keep
ψ_total nearly constant while both individually change.
In such cases the solver reports converged=True at a state that is
not a genuine fixed point.

The optional component_tol parameter guards against this: when set, after
the Δψ_total criterion fires the solver additionally checks max|Δψ_p| and
max|Δψ_os| separately and emits a UserWarning if either exceeds component_tol.
This does not prevent convergence from being reported — it is a diagnostic
that alerts the caller to inspect the component potentials.

Cells where psi_total / psi_p / psi_os is None (air spaces, isolated nodes)
contribute nan to the snapshots and are excluded from nanmax.

Architecture notes
------------------
- Scenario 0 has no osmotic term (built by solve_W / HydraulicMatrixBuilder
  without rhs_o) and is therefore not coupled here. i_scenario must be >= 1.
- Wall-node osmotic potentials (soil / xylem external boundaries) are always
  taken from the scenario dict; only cell-node psi_os values are updated from
  concentrations.
- The use_stored_psi_os=True flag on water_flux / initialize_scenarios preserves
  concentration-derived psi_os values across the reset_hydraulic_properties()
  call that occurs at the start of each solve_W.
- The baseline psi_os (captured from the scenario dict after the initial
  water_flux) represents unknown background solutes. The dynamic van't Hoff
  term from the transport solve is added on top rather than replacing it.
- An optional under-relaxation factor (relaxation ∈ (0, 1]) damps the
  concentration feedback (c_used = ω·c_new + (1−ω)·c_prev). The bare Picard
  iteration (ω=1) can limit-cycle or diverge under operators='T' with strong
  osmotic drives; ω < 1 stabilises it at the cost of more iterations.  Dirichlet
  BC values are re-imposed exactly after relaxation so they are never damped.

Solver choice — Picard vs JFNK
------------------------------
Two outer solvers are available via ``method``:

- ``method='picard'`` (default): the classic fixed-point iteration described
  above, optionally under-relaxed. Cheap per iteration but only converges when
  the fixed-point map g(c) is a contraction.

- ``method='jfnk'``: Jacobian-Free Newton–Krylov. Instead of iterating the map
  it drives the residual F(c) = g(c) − c to zero with Newton's method, using
  ``scipy.optimize.newton_krylov``. Newton converges quadratically
  near the root and — crucially — does NOT require g to be a contraction, so it
  reaches the fixed point where Picard blows up. Each residual evaluation
  performs one full transport + hydraulic solve; a line search globalizes the
  step (part of the scipy method). The continuation schedule still wraps the JFNK solve to
  ramp the osmotic drive from an easy value up to full strength.
"""

import warnings

import numpy as np

try:
    from scipy.optimize import newton_krylov
    try:
        from scipy.optimize import NoConvergence          # SciPy >= 1.15
    except ImportError:                                    # older SciPy
        from scipy.optimize.nonlin import NoConvergence
    _HAVE_NEWTON_KRYLOV = True
except Exception:  # pragma: no cover - SciPy always present in practice
    _HAVE_NEWTON_KRYLOV = False


def _snapshot_psi_total(manager) -> np.ndarray:
    """Return psi_total for all cells; nan where psi_total is None."""
    return np.array([
        cell.psi_total if cell.psi_total is not None else np.nan
        for cell in manager
    ])


def _snapshot_psi_p(manager) -> np.ndarray:
    """Return pressure potential (psi_p) for all cells; nan where None."""
    return np.array([
        cell.psi_p if cell.psi_p is not None else np.nan
        for cell in manager
    ])


def _snapshot_psi_os(manager) -> np.ndarray:
    """Return osmotic potential (psi_os) for all cells; nan where None."""
    return np.array([
        cell.psi_os if cell.psi_os is not None else np.nan
        for cell in manager
    ])


def coupled_water_solute_solve(
    mecha,
    st,
    T: float,
    boundary_conditions: dict,
    rhs: np.ndarray = None,
    i_scenario: int = 1,
    i_maturity: int = 0,
    h: int = 0,
    tol: float = 10.0,
    max_iter: int = 20,
    operators: str = 'T',
    relaxation: float = 1.0,
    component_tol: float = None,
    scheme: str = 'upwind',
    time_stepping: bool = False,
    theta: float = 1.0,
    method: str = 'picard',
    jfnk_f_tol: float = None,
    jfnk_maxiter: int = None,
    jfnk_inner_maxiter: int = 30,
    jfnk_line_search: str = 'armijo',
    jfnk_rdiff: float = 1e-3,
    continuation_steps=None,
    verbose: bool = False,
) -> tuple:
    """Iteratively solve water flow and solute transport until convergence.

    Parameters
    ----------
    mecha : Mecha
        Fully initialised Mecha instance. ``water_flux()`` should not have been
        called yet — this function calls it on the first iteration.
    st : SoluteTransport
        Configured SoluteTransport instance.  Must use mode='sym' or 'full'.
        Should correspond to the same h / i_maturity / i_scenario.
    T : float
        Temperature [K].  R = 8.314e4 [hPa cm³ mol⁻¹ K⁻¹] is used internally.
    boundary_conditions : dict {node_id: c_value}
        Dirichlet BCs for the transport solver.
    rhs : ndarray, optional
        Transport RHS vector. Defaults to zeros (pure BC-driven steady state).
    i_scenario : int
        Scenario index to couple (must be >= 1).
    i_maturity : int
        Maturity stage index.
    h : int
        Hydraulic configuration index.
    tol : float
        Convergence tolerance on max |Δψ_total| [hPa]. Default 10 hPa.
        Cells with psi_total=None are excluded from the maximum.
    max_iter : int
        Maximum outer iterations.
    operators : str
        Which spatial operators to pass to SoluteTransport.solve:
          'T' (default)  full advection-diffusion (D + A)
          'D'            diffusion only — stable for stiff osmotic flows;
                         use this when osmotic-driven advection dominates and
                         makes the transport matrix ill-conditioned.
          'A'            advection only
    component_tol : float or None
        If not None, a secondary diagnostic tolerance [hPa] checked *after* the
        Δψ_total criterion fires (converged=True). If max|Δψ_p| or max|Δψ_os|
        exceeds component_tol a UserWarning is emitted describing the
        isosmotic-cancellation situation: ψ_p and σ·ψ_os drifted in opposition
        so their sum appeared converged while each individually had not
        equilibrated. Default None (no component check).
    relaxation : float
        Under-relaxation factor ω ∈ (0, 1] applied to the concentration field
        that drives the osmotic feedback. Default 1.0 (no relaxation — pure
        Picard iteration). Each outer iteration mixes the freshly solved
        concentration with the previous one:

            c_used = ω · c_new + (1 − ω) · c_prev

        Values ω < 1 damp the solute → osmotic → water feedback and can break
        the limit cycle / divergence that the bare Picard scheme exhibits under
        operators='T' with strong osmotic drives (e.g. stiff Casparian
        gradients). Typical stabilising values are ω ≈ 0.3–0.7. The Dirichlet
        BC nodes are re-imposed exactly after relaxation so boundary values are
        never damped. The convergence check is performed on the relaxed state.
    scheme : str
        Advection discretization forwarded to SoluteTransport.solve:
          'upwind' (default)  first-order upwind (stable only for Pe ≲ 2)
          'sg'                Scharfetter–Gummel exponentially-fitted flux;
                              unconditionally non-oscillatory at any Pe. This
                              is the recommended way to make operators='T'
                              usable under strong osmotic membrane fluxes that
                              otherwise make the upwind steady operator ill-posed.
    time_stepping : bool
        When True the transport solve is advanced as an implicit-Euler pseudo-
        time step instead of a direct steady solve: each outer iteration passes
        the previous concentration field as c_prev to SoluteTransport.solve and
        marches [C/dt − θT] c_new = C/dt c_old + rhs. Implicit Euler (theta=1)
        is oscillation-free at ANY Peclet number (unlike the direct steady solve
        of D + A), so this is suggestion (1): march to steady state under
        advection instead of solving it in one shot. Requires
        st.capacitance_params to define 'dt'. The outer coupling convergence
        criterion (Δψ_total) is unchanged: the loop terminates when both the
        pseudo-time march and the osmotic feedback have settled.
    theta : float
        Time-integration parameter used only when time_stepping=True. Default
        1.0 (implicit Euler, unconditionally non-oscillatory). 0.5 gives
        Crank–Nicolson (2nd order but may oscillate at Pe > 2; 
        probably irrelevant if SG is used).
    method : str
        Outer solver for the coupled fixed point:
          'picard'  (default)  plain fixed-point iteration c ← g(c) with optional
                    under-relaxation ω. Cheap per step but only converges when
                    g is a contraction. Under strong osmotic feedback the map is
                    repelling (spectral radius > 1) and Picard diverges.
          'jfnk'    Jacobian-Free Newton–Krylov: drive the residual
                    F(c) = g(c) − c to zero with scipy.optimize.newton_krylov.
                    The Jacobian action is approximated by finite differences and
                    the Newton system is solved with GMRES, so no matrix is ever
                    assembled. Converges without requiring g to be a contraction,
                    reaching the fixed point where Picard blows up. ``relaxation``
                    is ignored (Newton globalizes with a line search instead).
    jfnk_f_tol : float or None
        Absolute residual tolerance ‖F(c)‖ for newton_krylov. When None a value
        is derived from ``tol`` and the R·T scaling so that a converged residual
        corresponds roughly to Δψ_total ≈ tol. Ignored unless method='jfnk'.
    jfnk_maxiter : int or None
        Maximum Newton iterations for newton_krylov. When None defaults to
        ``max_iter``. Ignored unless method='jfnk'.
    jfnk_inner_maxiter : int
        Maximum inner Krylov (GMRES) iterations per Newton step. Default 30.
        Ignored unless method='jfnk'.
    jfnk_line_search : str or None
        Globalization strategy passed to newton_krylov ('armijo' default,
        'wolfe', or None to disable). Ignored unless method='jfnk'.
    jfnk_rdiff : float
        Relative step size for the finite-difference Jacobian-vector products
        (newton_krylov's ``rdiff``).  The coupling map g is a noisy black box
        (each evaluation is a full transport + hydraulic solve), so the default
        sqrt(eps) probe is too small and yields a near-zero Jacobian that stalls
        GMRES.  Default 1e-3 gives a clean directional derivative for this map.
        Ignored unless method='jfnk'.
    continuation_steps : sequence of float or None
        Homotopy schedule on the concentration-derived osmotic drive.  Each
        entry λ ∈ (0, 1] scales the dynamic van't Hoff term (Ψ_os = baseline
        − λ·R·T·c) for one continuation STAGE; stages run in order, each solved
        to ``tol`` by the inner loop and warm-started from the previous stage's
        concentration field. This ramps a hard, divergent full-strength problem
        (λ=1) up from an easy one (small λ), which stabilizes the cold start.
        Example: [0.25, 0.5, 1.0].  None or [1.0] (default) → no continuation
        (single stage at full strength). ``max_iter`` applies PER stage.
    verbose : bool
        Print per-iteration diagnostics when True. Includes Δψ_p and Δψ_os
        alongside Δψ_total when component_tol is set, so cancellation is visible
        in the per-iteration log even before the warning fires.

    Returns
    -------
    c : ndarray
        Final concentration field used for the osmotic update. With
        relaxation < 1 this is the relaxed (mixed) field, not the raw transport
        solution.
    n_iter : int
        Number of outer iterations performed (Newton iterations for JFNK, as
        reported by the solver callback; Picard iterations otherwise).
    converged : bool
        True when the solver reached its tolerance. For Picard this is
        |Δψ_total|_max < tol; for JFNK it is ‖F(c)‖ < jfnk_f_tol. See the
        component_tol note above for the isosmotic-cancellation caveat.

    Raises
    ------
    ValueError
        If i_scenario < 1, st.mode is not 'sym' or 'full', relaxation is not
        in (0, 1], or method is not 'picard'/'jfnk'.
    ImportError
        If method='jfnk' but scipy.optimize.newton_krylov is unavailable.
    """
    if i_scenario < 1:
        raise ValueError(
            "i_scenario must be >= 1: scenario 0 has no osmotic term and "
            "cannot be coupled via this solver."
        )
    if st.mode not in ('sym', 'full'):
        raise ValueError(
            f"coupled solver requires st.mode='sym' or 'full'; got {st.mode!r}. "
            "Apoplastic-only transport has no cell concentrations to couple."
        )
    if scheme not in ('upwind', 'sg'):
        raise ValueError(f"scheme must be 'upwind' or 'sg'; got {scheme!r}.")
    if time_stepping and (st.capacitance_params is None
                          or st.capacitance_params.get('dt') is None):
        raise ValueError(
            "time_stepping=True requires st.capacitance_params with a 'dt' "
            "entry so SoluteTransport.solve can build the storage matrix C/dt. "
            "Pass e.g. SoluteTransport(..., capacitance_params={'dt': <days>})."
        )
    if not (0.0 < relaxation <= 1.0):
        raise ValueError(
            f"relaxation must be in (0, 1]; got {relaxation!r}. "
            "Use 1.0 for no under-relaxation (pure Picard)."
        )
    if method not in ('picard', 'jfnk'):
        raise ValueError(
            f"method must be 'picard' or 'jfnk'; got {method!r}."
        )
    if method == 'jfnk' and not _HAVE_NEWTON_KRYLOV:
        raise ImportError(
            "method='jfnk' requires scipy.optimize.newton_krylov, which could "
            "not be imported. Install/upgrade SciPy or use method='picard'."
        )
    # Continuation schedule: normalise to a non-empty ascending list ending at 1.
    if continuation_steps is None or len(continuation_steps) == 0:
        lambda_schedule = [1.0]
    else:
        lambda_schedule = [float(x) for x in continuation_steps]
        if any(not (0.0 < x <= 1.0) for x in lambda_schedule):
            raise ValueError(
                f"continuation_steps must all lie in (0, 1]; got {continuation_steps!r}."
            )

    nwj     = mecha.network.n_wall_junction
    n_cells = mecha.network.n_cells
    manager = mecha.network.cell_manager

    if rhs is None:
        rhs = np.zeros(st._matrix_size)

    # ── Iteration 0: initial hydraulic solve from scenario dict ──────────────
    mecha.water_flux(h=h)

    # Baseline osmotic potentials represent solutes not tracked by the transport
    # solver (unknown background contributors). The dynamic solve adds on top.
    psi_os_baseline = np.array([
        cell.psi_os if cell.psi_os is not None else 0.0
        for cell in manager
    ])

    psi_total_prev = _snapshot_psi_total(manager)
    if component_tol is not None:
        psi_p_prev  = _snapshot_psi_p(manager)
        psi_os_prev = _snapshot_psi_os(manager)

    # Index map for re-imposing Dirichlet BCs exactly after relaxation / mixing.
    # In 'sym' mode the transport vector is cell-indexed (node_id - nwj);
    # in 'full' mode it spans the whole network (node_id directly).
    bc_idx_val = []
    for node_id, c_val in (boundary_conditions or {}).items():
        idx = node_id - nwj if st.mode == 'sym' else node_id
        if 0 <= idx < st._matrix_size:
            bc_idx_val.append((idx, float(c_val)))

    def _reimpose_bc(vec):
        """Restore exact Dirichlet values after relaxation / a Newton step."""
        for idx, c_val in bc_idx_val:
            vec[idx] = c_val
        return vec

    def _cells_of(vec):
        """Cell-indexed slice of a transport-sized vector (sym: identity)."""
        if st.mode == 'sym':
            return vec
        return vec[nwj: nwj + n_cells]

    c_prev_ts = [None]    # boxed so the closure below can update it in place

    def _apply_step(c_field, lam):
        """One coupling half-step: set Ψ_os from c_field, re-solve hydraulics,
        solve transport, and return the fresh concentration g(c_field).

        This is the fixed-point map g used by BOTH Picard and JFNK. It sets the
        cell osmotic potentials from the *input* field (van't Hoff, scaled by the
        continuation λ), re-solves the water flux with those potentials
        (use_stored_psi_os preserves them across reset_hydraulic_properties), and
        then solves the transport operator on the updated flow to produce the new
        concentration field.
        """
        c_cells = _cells_of(c_field)
        manager.set_osmotic_from_concentration(
            c_cells * lam if lam != 1.0 else c_cells, nwj, T, psi_os_baseline)
        mecha.water_flux(h=h, use_stored_psi_os=True)
        c_new = st.solve(
            h=h,
            i_maturity=i_maturity,
            i_scenario=i_scenario,
            rhs=rhs.copy(),
            boundary_conditions=boundary_conditions,
            operators=operators,
            scheme=scheme,
            c_prev=c_prev_ts[0] if time_stepping else None,
            theta=theta,
        )
        if time_stepping:
            c_prev_ts[0] = c_new.copy()
        return c_new

    # Boundary cell mask: cell_ids that carry a Dirichlet BC (pinned in Newton).
    bc_cell_ids = set()
    for node_id in (boundary_conditions or {}):
        cid = node_id - nwj
        if 0 <= cid < n_cells:
            bc_cell_ids.add(cid)
    bc_cell_val = {}
    for node_id, c_val in (boundary_conditions or {}).items():
        cid = node_id - nwj
        if 0 <= cid < n_cells:
            bc_cell_val[cid] = float(c_val)

    # Last full transport field — the reduced (cell-only) Newton map slots the
    # candidate cell concentrations into this field, runs one coupling step, and
    # reads back the new cell concentrations. Walls/junctions are slaved to the
    # transport solve, so they need not be Newton unknowns.
    full_field = [None]

    def _reimpose_bc_cells(q):
        for cid, c_val in bc_cell_val.items():
            q[cid] = c_val
        return q

    def _apply_step_cells(q_cells, lam):
        """Reduced coupling map on CELL concentrations only.

        Given a cell-concentration vector q (length n_cells), set Ψ_os from it,
        re-solve hydraulics + transport, and return the new cell-concentration
        vector g(q). This is the physically minimal fixed point: the osmotic
        feedback flows solely through the cell nodes, so the ~11k wall/junction
        nodes need not be Newton unknowns. Doing so shrinks the Jacobian ~6×
        and removes the non-participating wall rows that otherwise dominate the
        residual norm and stall GMRES.
        """
        # Build the full field to drive the osmotic update: cell slice = q.
        if st.mode == 'sym':
            c_field = q_cells
        else:
            base = full_field[0]
            c_field = (base.copy() if base is not None
                       else np.zeros(st._matrix_size))
            c_field[nwj: nwj + n_cells] = q_cells
        c_new = _apply_step(c_field, lam)
        full_field[0] = c_new.copy()
        return _cells_of(c_new).copy()

    # Residual scale for JFNK: ‖F‖ in concentration units. A converged Δψ_total
    # of ~tol hPa corresponds to Δc ≈ tol / (σ·R·T). Use R·T (σ≤1) as a
    # conservative scale so f_tol is not looser than the requested ψ tolerance.
    RT = 8.314e4 * T
    default_f_tol = tol / RT if RT > 0 else tol

    c_used    = None      # field actually driving the osmotic update
    converged = False
    total_iters = 0
    n_stages = len(lambda_schedule)

    # ── Continuation stages: ramp λ on the dynamic osmotic drive ─────────────
    # Ψ_os = baseline − λ·R·T·c. Early low-λ stages are converge faster than the
    # harder high-λ stages; a single stage λ=1 is the no-continuation default.
    for i_stage, lam in enumerate(lambda_schedule):
        stage_note = (f" [stage {i_stage + 1}/{n_stages}, λ={lam:.3g}]"
                      if n_stages > 1 else "")
        stage_converged = False

        # Warm start: first stage cold-starts from the initial transport solve;
        # later stages reuse the previous stage's converged field.
        if c_used is None:
            c_used = st.solve(
                h=h, i_maturity=i_maturity, i_scenario=i_scenario,
                rhs=rhs.copy(), boundary_conditions=boundary_conditions,
                operators=operators, scheme=scheme,
                c_prev=c_prev_ts[0] if time_stepping else None, theta=theta,
            )
            if time_stepping:
                c_prev_ts[0] = c_used.copy()
        # Seed the full-field buffer so the reduced cell map has proper wall
        # values to slot the candidate cell concentrations into.
        full_field[0] = np.asarray(c_used).copy()

        if method == 'jfnk':
            # ── Jacobian-Free Newton–Krylov (reduced, cell-only unknown) ──────
            # Newton drives the residual F(q) = g(q) − q to zero, where q is the
            # CELL-concentration vector (length n_cells) — NOT the full field.
            # Three design choices make this matrix-free solve tractable:
            #
            #  (1) Reduced unknown. The osmotic feedback flows solely through the
            #      cell nodes (Ψ_os = −R·T·q). The ~11k wall/junction nodes are
            #      slaved to the transport solve and carry no independent
            #      feedback, so including them only inflates the residual norm
            #      with non-participating rows and stalls GMRES. Solving on the
            #      cells alone shrinks the Jacobian ~6× and conditions it far
            #      better.
            #  (2) State scaling. Concentrations are O(1e-5) mol cm⁻³ while the
            #      van't Hoff coupling multiplies them by R·T ≈ 2.5e7 to form
            #      Ψ_os. newton_krylov sizes its finite-difference probe from
            #      ‖x‖, so on the raw O(1e-5) field the probe is ~1e-13 and, after
            #      R·T amplification, changes Ψ_os below hydraulic-solver noise →
            #      the Jacobian looks like zero. We solve for x = q / c_scale
            #      (O(1)) and unscale inside the residual.
            #  (3) Residual scaling. F is returned in scaled units so ‖F‖ and
            #      f_tol are dimensionless and comparable to x.
            #
            # Dirichlet cells are pinned (residual ≡ 0) so Newton never moves them.
            q_used = _cells_of(c_used).copy()
            bc_mag = max((abs(v) for v in bc_cell_val.values()), default=0.0)
            c_scale = bc_mag if bc_mag > 0 else float(
                np.max(np.abs(q_used)) or 1.0)
            if c_scale <= 0:
                c_scale = 1.0

            f_tol = jfnk_f_tol if jfnk_f_tol is not None else default_f_tol
            # Interpret f_tol in scaled units (‖F/c_scale‖).
            f_tol_scaled = max(f_tol / c_scale, 1e-12)
            n_newton = jfnk_maxiter if jfnk_maxiter is not None else max_iter
            eval_count = [0]

            def _residual(x_flat):
                eval_count[0] += 1
                q_in = _reimpose_bc_cells((x_flat * c_scale).copy())
                g = _apply_step_cells(q_in, lam)
                F = (g - q_in) / c_scale        # scaled cell residual
                for cid in bc_cell_ids:
                    F[cid] = 0.0                # Dirichlet cells pinned to 0
                if verbose:
                    print(
                        f"[coupled_solver]{stage_note} JFNK residual eval "
                        f"{eval_count[0]:3d}: ‖F‖={np.linalg.norm(F):.4e}  "
                        f"q_max={np.max(np.abs(q_in)):.4e}"
                    )
                return F

            iter_box = [0]

            def _callback(x_cur, f_cur):
                iter_box[0] += 1
                if verbose:
                    print(
                        f"[coupled_solver]{stage_note} Newton iter "
                        f"{iter_box[0]:3d}: ‖F‖={np.linalg.norm(f_cur):.4e}"
                    )

            x0 = _reimpose_bc_cells(q_used.copy()) / c_scale
            try:
                x_sol = newton_krylov(
                    _residual, x0,
                    f_tol=f_tol_scaled,
                    maxiter=n_newton,
                    inner_maxiter=jfnk_inner_maxiter,
                    line_search=jfnk_line_search,
                    rdiff=jfnk_rdiff,
                    callback=_callback,
                    verbose=False,
                )
                stage_converged = True
            except NoConvergence as exc:
                # newton_krylov raises with the best iterate in exc.args[0].
                x_sol = np.asarray(exc.args[0]).ravel()
                stage_converged = False
                if verbose:
                    print(f"[coupled_solver]{stage_note} JFNK did NOT converge "
                          f"(‖F‖ still > f_tol={f_tol_scaled:.2e}).")

            # Realize the accepted cell solution: run one final coupling step so
            # the model state (hydraulics, full transport field) is consistent.
            q_final = _reimpose_bc_cells(np.asarray(x_sol).ravel() * c_scale)
            _apply_step_cells(q_final, lam)
            c_used = full_field[0]
            total_iters += iter_box[0]

        else:
            # ── Plain Picard (optionally under-relaxed) ──────────────────────
            for _ in range(max_iter):
                total_iters += 1
                c = _apply_step(c_used, lam)
                if relaxation < 1.0:
                    c_used = relaxation * c + (1.0 - relaxation) * c_used
                    c_used = _reimpose_bc(c_used)
                else:
                    c_used = c

                psi_total_curr = _snapshot_psi_total(manager)
                delta = float(np.nanmax(np.abs(psi_total_curr - psi_total_prev)))
                psi_total_prev = psi_total_curr

                if component_tol is not None:
                    psi_p_curr   = _snapshot_psi_p(manager)
                    psi_os_curr  = _snapshot_psi_os(manager)
                    delta_p  = float(np.nanmax(np.abs(psi_p_curr  - psi_p_prev)))
                    delta_os = float(np.nanmax(np.abs(psi_os_curr - psi_os_prev)))
                    psi_p_prev  = psi_p_curr
                    psi_os_prev = psi_os_curr
                else:
                    delta_p = delta_os = float('nan')

                if verbose:
                    c_max = float(np.max(np.abs(_cells_of(c_used))))
                    comp_note = (
                        f"  Δψ_p={delta_p:8.2f}  Δψ_os={delta_os:8.2f}"
                        if component_tol is not None else ""
                    )
                    print(
                        f"[coupled_solver] iter {total_iters:3d}{stage_note}: "
                        f"Δψ_total_max = {delta:8.2f} hPa  "
                        f"c_max = {c_max:.4e}{comp_note}"
                    )

                if delta < tol:
                    stage_converged = True
                    if component_tol is not None:
                        bad_p  = np.isfinite(delta_p)  and delta_p  > component_tol
                        bad_os = np.isfinite(delta_os) and delta_os > component_tol
                        if bad_p or bad_os:
                            warnings.warn(
                                f"[coupled_solver] Isosmotic cancellation detected at "
                                f"iter {total_iters}: Δψ_total={delta:.2f} hPa < tol "
                                f"({tol} hPa) but Δψ_p={delta_p:.2f} hPa and "
                                f"Δψ_os={delta_os:.2f} hPa (component_tol={component_tol} hPa). "
                                f"ψ_p and σ·ψ_os may be co-varying in opposition "
                                f"(isosmotic flow regime); the reported fixed point "
                                f"may not represent true equilibration of both fields.",
                                UserWarning,
                                stacklevel=2,
                            )
                    break

        # Full convergence is only meaningful at the final (λ=1) stage; earlier
        # stages just need to be close enough to warm-start the next.
        if i_stage == n_stages - 1:
            converged = stage_converged

    if verbose:
        status = "converged" if converged else "did NOT converge"
        notes = [f"method={method}"]
        if method == 'picard' and relaxation < 1.0:
            notes.append(f"relaxation ω={relaxation:.2f}")
        if n_stages > 1:
            notes.append(f"{n_stages}-stage continuation")
        note = f" ({', '.join(notes)})"
        print(f"[coupled_solver] {status} after {total_iters} iteration(s){note}.")

    # Return the field that actually drove the final osmotic update so the
    # caller's concentrations are consistent with the converged psi_os.
    return c_used, total_iters, converged
