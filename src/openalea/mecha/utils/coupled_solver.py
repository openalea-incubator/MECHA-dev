"""Iterative coupling of MECHA water flow and solute transport.

Cell Ψ_os = baseline − R·T·c (van't Hoff) feeds back into the hydraulic solve.
Outer solvers: Picard (optionally under-relaxed) or JFNK (newton_krylov).

See docs/CoupledSolver.md for the full description and usage examples.
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


def _jfnk_fixed_point(g, q0, c_scale, bc_cell_val, f_tol, maxiter,
                      inner_maxiter, line_search, rdiff,
                      verbose=False, note=""):
    """Solve q = g(q) with newton_krylov on x = q / c_scale; BC cells pinned.

    Returns (q, n_newton, converged).
    """
    def _pin(q):
        for cid, c_val in bc_cell_val.items():
            q[cid] = c_val
        return q

    eval_count = [0]

    def _residual(x_flat):
        eval_count[0] += 1
        q_in = _pin((x_flat * c_scale).copy())
        F = (g(q_in) - q_in) / c_scale
        for cid in bc_cell_val:
            F[cid] = 0.0
        if verbose:
            print(f"[coupled_solver]{note} JFNK residual eval {eval_count[0]:3d}: "
                  f"‖F‖={np.linalg.norm(F):.4e}  q_max={np.max(np.abs(q_in)):.4e}")
        return F

    iter_box = [0]

    def _callback(x_cur, f_cur):
        iter_box[0] += 1
        if verbose:
            print(f"[coupled_solver]{note} Newton iter {iter_box[0]:3d}: "
                  f"‖F‖={np.linalg.norm(f_cur):.4e}")

    f_tol_scaled = max(f_tol / c_scale, 1e-12)
    x0 = _pin(np.asarray(q0, dtype=float).copy()) / c_scale
    try:
        x_sol = newton_krylov(
            _residual, x0, f_tol=f_tol_scaled, maxiter=maxiter,
            inner_maxiter=inner_maxiter, line_search=line_search,
            rdiff=rdiff, callback=_callback, verbose=False)
        converged = True
    except NoConvergence as exc:
        # newton_krylov raises with the best iterate in exc.args[0].
        x_sol = np.asarray(exc.args[0]).ravel()
        converged = False
        if verbose:
            print(f"[coupled_solver]{note} JFNK did NOT converge "
                  f"(‖F‖ still > f_tol={f_tol_scaled:.2e}).")
    q = _pin(np.asarray(x_sol, dtype=float).ravel() * c_scale)
    return q, iter_box[0], converged


def coupled_implicit_step(
    mecha,
    st,
    c_prev: np.ndarray,
    T: float,
    psi_os_baseline: np.ndarray,
    boundary_conditions: dict = None,
    rhs: np.ndarray = None,
    i_scenario: int = 1,
    i_maturity: int = 0,
    h: int = 0,
    theta: float = 1.0,
    operators: str = 'T',
    scheme: str = 'sg',
    method: str = 'picard',
    tol: float = 1.0,
    max_iter: int = 12,
    relaxation: float = 1.0,
    jfnk_f_tol: float = None,
    jfnk_maxiter: int = None,
    jfnk_inner_maxiter: int = 30,
    jfnk_line_search: str = 'armijo',
    jfnk_rdiff: float = 1e-3,
    transport_step=None,
    verbose: bool = False,
) -> tuple:
    """Advance one time step dt from fixed ``c_prev`` with implicit coupling.

    Returns (c_new, n_iter, converged). See docs/CoupledSolver.md.
    """
    if method not in ('picard', 'jfnk'):
        raise ValueError(f"method must be 'picard' or 'jfnk'; got {method!r}.")
    if method == 'jfnk' and not _HAVE_NEWTON_KRYLOV:
        raise ImportError("method='jfnk' requires scipy.optimize.newton_krylov.")
    if not (0.0 < relaxation <= 1.0):
        raise ValueError(f"relaxation must be in (0, 1]; got {relaxation!r}.")

    nwj = mecha.network.n_wall_junction
    n_cells = mecha.network.n_cells
    manager = mecha.network.cell_manager
    RT = 8.314e4 * T
    sym = st.mode == 'sym'
    if rhs is None:
        rhs = np.zeros(st._matrix_size)

    def _cells_of(vec):
        return vec if sym else vec[nwj: nwj + n_cells]

    if transport_step is None:
        def transport_step():
            return st.solve(
                h=h, i_maturity=i_maturity, i_scenario=i_scenario,
                rhs=rhs.copy(), boundary_conditions=boundary_conditions,
                c_prev=c_prev, theta=theta, operators=operators, scheme=scheme)

    last = [None]

    def g(q_cells):
        manager.set_osmotic_from_concentration(q_cells, nwj, T, psi_os_baseline)
        mecha.water_flux(h=h, use_stored_psi_os=True, verbose=False)
        last[0] = transport_step()
        return _cells_of(last[0]).copy()

    bc_cell_val = {}
    for node_id, c_val in (boundary_conditions or {}).items():
        cid = node_id - nwj
        if 0 <= cid < n_cells:
            bc_cell_val[cid] = float(c_val)

    q = _cells_of(np.asarray(c_prev, dtype=float)).copy()

    if method == 'picard':
        converged = False
        n_iter = 0
        for n_iter in range(1, max_iter + 1):
            q_new = g(q)
            delta = RT * float(np.max(np.abs(q_new - q))) if q.size else 0.0
            if verbose:
                print(f"[coupled_step] Picard {n_iter:3d}: Δψ_os_max = {delta:.3e} hPa")
            q = q_new if relaxation == 1.0 else relaxation * q_new + (1.0 - relaxation) * q
            if delta < tol:
                converged = True
                break
        if relaxation < 1.0:
            g(q)
        return last[0], n_iter, converged

    bc_mag = max((abs(v) for v in bc_cell_val.values()), default=0.0)
    c_scale = max(bc_mag, float(np.max(np.abs(q))) if q.size else 0.0, 10.0 * tol / RT)
    f_tol = jfnk_f_tol if jfnk_f_tol is not None else tol / RT
    q, n_iter, converged = _jfnk_fixed_point(
        g, q, c_scale, bc_cell_val, f_tol,
        maxiter=jfnk_maxiter if jfnk_maxiter is not None else max_iter,
        inner_maxiter=jfnk_inner_maxiter, line_search=jfnk_line_search,
        rdiff=jfnk_rdiff, verbose=verbose, note=" [step]")
    g(q)
    return last[0], n_iter, converged


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
    """Solve the coupled water–solute steady state (or pseudo-time march).

    Picard converges on max|Δψ_total| < tol [hPa]; JFNK on ‖F‖ < jfnk_f_tol.
    Returns (c, n_iter, converged). See docs/CoupledSolver.md for parameters.
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

    # Background Ψ_os from untracked solutes; the van't Hoff term is added on top.
    psi_os_baseline = np.array([
        cell.psi_os if cell.psi_os is not None else 0.0
        for cell in manager
    ])

    psi_total_prev = _snapshot_psi_total(manager)
    if component_tol is not None:
        psi_p_prev  = _snapshot_psi_p(manager)
        psi_os_prev = _snapshot_psi_os(manager)

    # 'sym' vectors are cell-indexed (node_id - nwj); 'full' use node_id.
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
        """Fixed-point map g: Ψ_os from c_field → water_flux → transport."""
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

    # Dirichlet cells (pinned in Newton).
    bc_cell_val = {}
    for node_id, c_val in (boundary_conditions or {}).items():
        cid = node_id - nwj
        if 0 <= cid < n_cells:
            bc_cell_val[cid] = float(c_val)

    # Last full transport field; wall/junction values are slaved to transport.
    full_field = [None]

    def _apply_step_cells(q_cells, lam):
        """Reduced map g(q) on cell concentrations only (Newton unknown)."""
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

    # Δψ ≈ σ·R·T·Δc with σ ≤ 1, so tol / (R·T) is a conservative f_tol.
    RT = 8.314e4 * T
    default_f_tol = tol / RT if RT > 0 else tol

    c_used    = None      # field actually driving the osmotic update
    converged = False
    total_iters = 0
    n_stages = len(lambda_schedule)

    # ── Continuation stages: Ψ_os = baseline − λ·R·T·c ───────────────────────
    for i_stage, lam in enumerate(lambda_schedule):
        stage_note = (f" [stage {i_stage + 1}/{n_stages}, λ={lam:.3g}]"
                      if n_stages > 1 else "")
        stage_converged = False

        # Later stages warm-start from the previous stage.
        if c_used is None:
            c_used = st.solve(
                h=h, i_maturity=i_maturity, i_scenario=i_scenario,
                rhs=rhs.copy(), boundary_conditions=boundary_conditions,
                operators=operators, scheme=scheme,
                c_prev=c_prev_ts[0] if time_stepping else None, theta=theta,
            )
            if time_stepping:
                c_prev_ts[0] = c_used.copy()
        full_field[0] = np.asarray(c_used).copy()

        if method == 'jfnk':
            # ── JFNK on cell concentrations, scaled to O(1) ───────────────────
            q_used = _cells_of(c_used).copy()
            bc_mag = max((abs(v) for v in bc_cell_val.values()), default=0.0)
            c_scale = bc_mag if bc_mag > 0 else float(
                np.max(np.abs(q_used)) or 1.0)
            if c_scale <= 0:
                c_scale = 1.0

            f_tol = jfnk_f_tol if jfnk_f_tol is not None else default_f_tol
            q_final, n_newton_done, stage_converged = _jfnk_fixed_point(
                lambda q: _apply_step_cells(q, lam), q_used, c_scale,
                bc_cell_val, f_tol,
                maxiter=jfnk_maxiter if jfnk_maxiter is not None else max_iter,
                inner_maxiter=jfnk_inner_maxiter,
                line_search=jfnk_line_search, rdiff=jfnk_rdiff,
                verbose=verbose, note=stage_note)

            # Final evaluation leaves mecha/st consistent with q_final.
            _apply_step_cells(q_final, lam)
            c_used = full_field[0]
            total_iters += n_newton_done

        else:
            # ── Picard, optionally under-relaxed ─────────────────────────────
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

        # Only the final stage determines convergence.
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

    return c_used, total_iters, converged
