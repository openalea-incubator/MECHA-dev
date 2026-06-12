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
not on ψ_os alone.  This is the physically correct choice: ψ_total is the
driving force for water flow, and changes in ψ_p and ψ_os can partially cancel.
Converging on ψ_os alone would be either over-conservative (large Δψ_os but
small Δψ_total) or insufficient (ψ_os plateaued while ψ_p still drifting).

Cells where psi_total is None (air spaces, isolated nodes) contribute nan to
the snapshot and are excluded from nanmax — they do not affect convergence.

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
"""

import numpy as np


def _snapshot_psi_total(manager) -> np.ndarray:
    """Return psi_total for all cells; nan where psi_total is None."""
    return np.array([
        cell.psi_total if cell.psi_total is not None else np.nan
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
    verbose : bool
        Print per-iteration diagnostics when True.

    Returns
    -------
    c : ndarray
        Final concentration field from SoluteTransport.solve().
    n_iter : int
        Number of outer iterations performed.
    converged : bool
        True when |Δψ_total|_max < tol before max_iter was reached.

    Raises
    ------
    ValueError
        If i_scenario < 1 or st.mode is not 'sym' or 'full'.
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

    c         = None
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

        # ── Extract cell-indexed concentrations ───────────────────────────────
        if st.mode == 'sym':
            c_cells = c                        # (n_cells,), indexed by cell_id
        else:                                  # 'full'
            c_cells = c[nwj: nwj + n_cells]

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

        if verbose:
            c_max = float(np.max(np.abs(c_cells))) if len(c_cells) else 0.0
            print(
                f"[coupled_solver] iter {n_iter + 1:3d}: "
                f"Δψ_total_max = {delta:8.2f} hPa  "
                f"c_max = {c_max:.4e}"
            )

        if delta < tol:
            converged = True
            break

    if verbose:
        status = "converged" if converged else "did NOT converge"
        print(f"[coupled_solver] {status} after {n_iter + 1} iteration(s).")

    return c, n_iter + 1, converged
