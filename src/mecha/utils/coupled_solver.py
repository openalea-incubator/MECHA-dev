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
normal isosmotic-flow regime (e.g. phloem loading) where turgor and osmotic
potential co-vary to keep ψ_total nearly constant while both individually
change.  In such cases the solver reports converged=True at a state that is
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
  without rhs_o) and is therefore not coupled here.  i_scenario must be >= 1.
- Wall-node osmotic potentials (soil / xylem external boundaries) are always
  taken from the scenario dict; only cell-node psi_os values are updated from
  concentrations.
- The use_stored_psi_os=True flag on water_flux / initialize_scenarios preserves
  concentration-derived psi_os values across the reset_hydraulic_properties()
  call that occurs at the start of each solve_W.
- The baseline psi_os (captured from the scenario dict after the initial
  water_flux) represents unknown background solutes.  The dynamic van't Hoff
  term from the transport solve is added on top rather than replacing it.
- An optional under-relaxation factor (relaxation ∈ (0, 1]) damps the
  concentration feedback (c_used = ω·c_new + (1−ω)·c_prev).  The bare Picard
  iteration (ω=1) can limit-cycle or diverge under operators='T' with strong
  osmotic drives; ω < 1 stabilises it at the cost of more iterations.  Dirichlet
  BC values are re-imposed exactly after relaxation so they are never damped.
"""

import warnings

import numpy as np


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
    verbose: bool = False,
) -> tuple:
    """Iteratively solve water flow and solute transport until convergence.

    Parameters
    ----------
    mecha : Mecha
        Fully initialised Mecha instance.  water_flux() should not have been
        called yet — this function calls it on the first iteration.
    st : SoluteTransport
        Configured SoluteTransport instance.  Must use mode='sym' or 'full'.
        Should correspond to the same h / i_maturity / i_scenario.
    T : float
        Temperature [K].  R = 8.314e4 hPa cm³ mol⁻¹ K⁻¹ is used internally.
    boundary_conditions : dict {node_id: c_value}
        Dirichlet BCs for the transport solver.
    rhs : ndarray, optional
        Transport RHS vector.  Defaults to zeros (pure BC-driven steady state).
    i_scenario : int
        Scenario index to couple (must be >= 1).
    i_maturity : int
        Maturity stage index.
    h : int
        Hydraulic configuration index.
    tol : float
        Convergence tolerance on max |Δψ_total| [hPa].  Default 10 hPa.
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
        Δψ_total criterion fires (converged=True).  If max|Δψ_p| or max|Δψ_os|
        exceeds component_tol a UserWarning is emitted describing the
        isosmotic-cancellation situation: ψ_p and σ·ψ_os drifted in opposition
        so their sum appeared converged while each individually had not
        equilibrated.  Default None (no component check).
    relaxation : float
        Under-relaxation factor ω ∈ (0, 1] applied to the concentration field
        that drives the osmotic feedback.  Default 1.0 (no relaxation — pure
        Picard iteration).  Each outer iteration mixes the freshly solved
        concentration with the previous one:

            c_used = ω · c_new + (1 − ω) · c_prev

        Values ω < 1 damp the solute → osmotic → water feedback and can break
        the limit cycle / divergence that the bare Picard scheme exhibits under
        operators='T' with strong osmotic drives (e.g. stiff Casparian
        gradients).  Typical stabilising values are ω ≈ 0.3–0.7.  The Dirichlet
        BC nodes are re-imposed exactly after relaxation so boundary values are
        never damped.  The convergence check is performed on the relaxed state.
    verbose : bool
        Print per-iteration diagnostics when True.  Includes Δψ_p and Δψ_os
        alongside Δψ_total when component_tol is set, so cancellation is visible
        in the per-iteration log even before the warning fires.

    Returns
    -------
    c : ndarray
        Final concentration field used for the osmotic update.  With
        relaxation < 1 this is the relaxed (mixed) field, not the raw transport
        solution.
    n_iter : int
        Number of outer iterations performed.
    converged : bool
        True when |Δψ_total|_max < tol before max_iter was reached.  See the
        component_tol note above for the isosmotic-cancellation caveat.

    Raises
    ------
    ValueError
        If i_scenario < 1, st.mode is not 'sym' or 'full', or relaxation is not
        in (0, 1].
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
    if not (0.0 < relaxation <= 1.0):
        raise ValueError(
            f"relaxation must be in (0, 1]; got {relaxation!r}. "
            "Use 1.0 for no under-relaxation (pure Picard)."
        )

    nwj     = mecha.network.n_wall_junction
    n_cells = mecha.network.n_cells
    manager = mecha.network.cell_manager

    if rhs is None:
        rhs = np.zeros(st._matrix_size)

    # ── Iteration 0: initial hydraulic solve from scenario dict ──────────────
    mecha.water_flux(h=h)

    # Baseline osmotic potentials represent solutes not tracked by the transport
    # solver (unknown background contributors).  The dynamic solve adds on top.
    psi_os_baseline = np.array([
        cell.psi_os if cell.psi_os is not None else 0.0
        for cell in manager
    ])

    psi_total_prev = _snapshot_psi_total(manager)
    if component_tol is not None:
        psi_p_prev  = _snapshot_psi_p(manager)
        psi_os_prev = _snapshot_psi_os(manager)

    # Index map for re-imposing Dirichlet BCs exactly after relaxation.
    # In 'sym' mode the transport vector is cell-indexed (node_id - nwj);
    # in 'full' mode it spans the whole network (node_id directly).
    if relaxation < 1.0:
        bc_idx_val = []
        for node_id, c_val in (boundary_conditions or {}).items():
            idx = node_id - nwj if st.mode == 'sym' else node_id
            if 0 <= idx < st._matrix_size:
                bc_idx_val.append((idx, float(c_val)))

    c         = None      # raw transport solution
    c_used    = None      # relaxed field actually driving the osmotic update
    converged = False

    for n_iter in range(max_iter):

        # ── Transport solve ───────────────────────────────────────────────────
        c = st.solve(
            h=h,
            i_maturity=i_maturity,
            i_scenario=i_scenario,
            rhs=rhs.copy(),
            boundary_conditions=boundary_conditions,
            operators=operators,
        )

        # ── Under-relaxation on the coupled field ─────────────────────────────
        # Mix with the previous relaxed field to damp the feedback.  On the
        # first iteration there is no previous field, so use the raw solution.
        if relaxation < 1.0 and c_used is not None:
            c_used = relaxation * c + (1.0 - relaxation) * c_used
            # Re-impose Dirichlet BCs exactly: boundary values must not be damped.
            for idx, c_val in bc_idx_val:
                c_used[idx] = c_val
        else:
            c_used = c.copy()

        # ── Extract cell-indexed concentrations ───────────────────────────────
        if st.mode == 'sym':
            c_cells = c_used                   # (n_cells,), indexed by cell_id
        else:                                  # 'full'
            c_cells = c_used[nwj: nwj + n_cells]

        # ── Update cell osmotic potentials from concentrations ────────────────
        manager.set_osmotic_from_concentration(c_cells, nwj, T, psi_os_baseline)

        # ── Re-solve hydraulics with updated osmotic potentials ───────────────
        # Must precede the convergence check so psi_total reflects the current
        # coupled state.  use_stored_psi_os=True preserves the concentration-
        # derived psi_os values across reset_hydraulic_properties() in solve_W.
        mecha.water_flux(h=h, use_stored_psi_os=True)

        # ── Convergence check on Δψ_total ─────────────────────────────────────
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
            c_max = float(np.max(np.abs(c_cells))) if len(c_cells) else 0.0
            comp_note = (
                f"  Δψ_p={delta_p:8.2f}  Δψ_os={delta_os:8.2f}"
                if component_tol is not None else ""
            )
            print(
                f"[coupled_solver] iter {n_iter + 1:3d}: "
                f"Δψ_total_max = {delta:8.2f} hPa  "
                f"c_max = {c_max:.4e}{comp_note}"
            )

        if delta < tol:
            converged = True
            # ── Isosmotic-cancellation guard ──────────────────────────────────
            # Warn if ψ_p and ψ_os are still drifting in opposition (their sum
            # Δψ_total appeared small but neither component has equilibrated).
            if component_tol is not None:
                bad_p  = np.isfinite(delta_p)  and delta_p  > component_tol
                bad_os = np.isfinite(delta_os) and delta_os > component_tol
                if bad_p or bad_os:
                    warnings.warn(
                        f"[coupled_solver] Isosmotic cancellation detected at "
                        f"iter {n_iter + 1}: Δψ_total={delta:.2f} hPa < tol "
                        f"({tol} hPa) but Δψ_p={delta_p:.2f} hPa and "
                        f"Δψ_os={delta_os:.2f} hPa (component_tol={component_tol} hPa). "
                        f"ψ_p and σ·ψ_os may be co-varying in opposition "
                        f"(isosmotic flow regime); the reported fixed point "
                        f"may not represent true equilibration of both fields.",
                        UserWarning,
                        stacklevel=2,
                    )
            break

    if verbose:
        status = "converged" if converged else "did NOT converge"
        relax_note = "" if relaxation >= 1.0 else f" (relaxation ω={relaxation:.2f})"
        print(f"[coupled_solver] {status} after {n_iter + 1} iteration(s){relax_note}.")

    # Return the relaxed field that actually drove the final osmotic update so
    # the caller's concentrations are consistent with the converged psi_os.
    return (c_used if c_used is not None else c), n_iter + 1, converged
