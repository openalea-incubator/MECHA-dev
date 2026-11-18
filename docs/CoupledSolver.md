# Coupled water–solute solver

Module: [src/openalea/mecha/utils/coupled_solver.py](../src/openalea/mecha/utils/coupled_solver.py)

The module couples MECHA's hydraulic solver (`Mecha.water_flux`) with the
solute-transport solver (`SoluteTransport.solve`). Cell solute concentrations
change the cell osmotic potentials (van't Hoff), which change the water flow,
which in turn changes advective solute transport.

It provides two public functions:

| Function | Purpose |
|---|---|
| `coupled_water_solute_solve` | Coupled **steady state** (or a pseudo-time march to it). |
| `coupled_implicit_step` | **One physical time step** `dt` with implicit coupling, for transient runs. |

---

## 1. The coupling map

Both functions solve a fixed point $c = g(c)$ on the cell concentrations. One
evaluation of $g$ does:

1. Write cell osmotic potentials from the concentration field
   (`HydraulicCellManager.set_osmotic_from_concentration`):

   $$\Psi_{os} = \Psi_{os}^{\text{baseline}} - \lambda \, R T \, c, \qquad R = 8.314\times10^{4}\ \text{hPa cm}^3\,\text{mol}^{-1}\,\text{K}^{-1}$$

2. Re-solve water flow: `mecha.water_flux(h=h, use_stored_psi_os=True)`.
3. Solve solute transport on the updated flow field to obtain the new $c$.

Notes:

- **Baseline.** $\Psi_{os}^{\text{baseline}}$ represents untracked background
  solutes. `coupled_water_solute_solve` captures it from the scenario after an
  initial `water_flux`; for `coupled_implicit_step` the caller supplies it.
  The van't Hoff term is added on top, not substituted.
- **Only cell nodes are updated.** Wall-node osmotic potentials (soil / xylem
  boundaries) always come from the scenario definition.
- **`use_stored_psi_os=True`** keeps the concentration-derived $\Psi_{os}$
  across the `reset_hydraulic_properties()` call at the start of each `solve_W`.
- **Scenario 0 cannot be coupled** (it has no osmotic term); `i_scenario >= 1`.
- **Transport mode** must be `'sym'` or `'full'`; apoplastic-only transport has
  no cell concentrations.
- $\lambda$ is the continuation factor (1 unless `continuation_steps` is used).

---

## 2. Outer solvers (`method`)

### Picard (`method='picard'`, default)

Plain fixed-point iteration with optional under-relaxation:

$$c_{\text{used}} = \omega\, c_{\text{new}} + (1-\omega)\, c_{\text{prev}}, \qquad \omega = \texttt{relaxation} \in (0, 1]$$

- Cheap per iteration, but converges only if $g$ is a contraction.
- Under `operators='T'` with strong osmotic drives the bare iteration
  ($\omega = 1$) can limit-cycle or diverge; $\omega \approx 0.3$–$0.7$ usually
  stabilises it at the cost of more iterations.
- Dirichlet BC values are re-imposed exactly after relaxation.

### JFNK (`method='jfnk'`)

Jacobian-Free Newton–Krylov via `scipy.optimize.newton_krylov` on the residual
$F(q) = g(q) - q$. Jacobian–vector products are finite differences; the Newton
system is solved with GMRES. Converges where Picard diverges (no contraction
requirement); `relaxation` is ignored, a line search globalises the step.

Implementation choices that make it work:

1. **Reduced unknown.** $q$ is the *cell* concentration vector only. The
   osmotic feedback acts solely through cells; wall/junction nodes are slaved
   to the transport solve. This shrinks the Jacobian (~6×) and avoids
   non-participating rows stalling GMRES.
2. **State scaling.** Concentrations are $O(10^{-5})$ mol cm⁻³ while
   $RT \approx 2.5\times10^{7}$. Newton solves for $x = q / c_{\text{scale}}$
   ($O(1)$) so the finite-difference probe produces a measurable change in
   $\Psi_{os}$.
3. **Residual scaling.** $F$ is returned in scaled units so `f_tol` is
   dimensionless.
4. **`jfnk_rdiff = 1e-3`.** $g$ is a noisy black box (each call is a full
   hydraulic + transport solve); the default $\sqrt{\varepsilon}$ probe is too
   small and yields a near-zero Jacobian.
5. Dirichlet cells are pinned (residual $\equiv 0$).
6. After convergence, one final evaluation of $g$ leaves `mecha` and `st`
   consistent with the returned field.

Default residual tolerance: $f_{tol} = \texttt{tol} / (RT)$ (conservative since
$\Delta\psi \approx \sigma R T\, \Delta c$ with $\sigma \le 1$).

---

## 3. Convergence criterion (Picard steady solve)

`coupled_water_solute_solve` stops when

$$\max_{\text{cells}} |\Delta \psi_{\text{total}}| < \texttt{tol}, \qquad \psi_{\text{total}} = \psi_p - \sigma\,\psi_{os}$$

$\psi_{\text{total}}$ drives the water flow, so it is the physically relevant
quantity. Cells with `psi_total is None` (air spaces, isolated nodes) are
ignored.

**Isosmotic cancellation.** If $\psi_p$ and $\sigma\psi_{os}$ drift in
opposition, $\Delta\psi_{\text{total}} \approx 0$ although neither has
equilibrated (normal in the isosmotic-flow regime). Set `component_tol` to get
a `UserWarning` when $\max|\Delta\psi_p|$ or $\max|\Delta\psi_{os}|$ still
exceeds it at convergence. This is diagnostic only; `converged` stays `True`.
With `verbose=True` and `component_tol` set, both components are printed every
iteration.

`coupled_implicit_step` with Picard instead stops when
$RT \cdot \max|\Delta q_{\text{cells}}| < \texttt{tol}$.

---

## 4. `coupled_water_solute_solve`

```python
c, n_iter, converged = coupled_water_solute_solve(
    mecha, st, T, boundary_conditions, rhs=None,
    i_scenario=1, i_maturity=0, h=0,
    tol=10.0, max_iter=20,
    operators='T', relaxation=1.0, component_tol=None,
    scheme='upwind', time_stepping=False, theta=1.0,
    method='picard',
    jfnk_f_tol=None, jfnk_maxiter=None, jfnk_inner_maxiter=30,
    jfnk_line_search='armijo', jfnk_rdiff=1e-3,
    continuation_steps=None, verbose=False)
```

`mecha.water_flux()` is called internally first; it need not be called before.

| Parameter | Meaning |
|---|---|
| `mecha` | Initialised `Mecha` instance. |
| `st` | `SoluteTransport` (mode `'sym'` or `'full'`) for the same `h` / `i_maturity` / `i_scenario`. |
| `T` | Temperature [K]. |
| `boundary_conditions` | `{node_id: c}` Dirichlet BCs for transport. |
| `rhs` | Transport source vector (default zeros). |
| `tol` | [hPa] Tolerance on $\max\lvert\Delta\psi_{\text{total}}\rvert$ (JFNK: sets default `f_tol`). |
| `max_iter` | Max outer iterations **per continuation stage**. |
| `operators` | `'T'` advection–diffusion, `'D'` diffusion only (robust for stiff osmotic flows), `'A'` advection only. |
| `scheme` | `'upwind'` (stable for Pe ≲ 2) or `'sg'` (Scharfetter–Gummel, non-oscillatory at any Pe; recommended with `'T'`). |
| `relaxation` | Picard under-relaxation $\omega$. |
| `component_tol` | [hPa] Isosmotic-cancellation diagnostic (see §3). |
| `time_stepping` | March to steady state with implicit pseudo-time steps (`c_prev` = previous iterate). Requires `st.capacitance_params['dt']`. |
| `theta` | θ-method for `time_stepping` (1 implicit Euler, 0.5 Crank–Nicolson). |
| `method` | `'picard'` or `'jfnk'`. |
| `jfnk_*` | Newton tolerance, Newton iterations (default `max_iter`), GMRES iterations, line search (`'armijo'`, `'wolfe'`, `None`), FD step. |
| `continuation_steps` | Homotopy schedule on $\lambda$, e.g. `[0.25, 0.5, 1.0]`. Each stage warm-starts the next; only the last stage sets `converged`. |
| `verbose` | Per-iteration log. |

Returns:

- `c` — field that drove the final osmotic update (the relaxed field if $\omega<1$).
- `n_iter` — total Picard or Newton iterations over all stages.
- `converged` — tolerance reached in the final stage.

Raises `ValueError` for `i_scenario < 1`, invalid `st.mode`, `scheme`,
`relaxation`, `method`, `continuation_steps`, or missing `dt` with
`time_stepping=True`; `ImportError` if JFNK is requested without
`newton_krylov`.

### Minimal example

```python
from openalea.mecha.utils.coupled_solver import coupled_water_solute_solve

c, n, ok = coupled_water_solute_solve(
    mecha, st, T=293.15, boundary_conditions=bc,
    i_scenario=1, operators='T', scheme='sg',
    method='jfnk', continuation_steps=[0.25, 0.5, 1.0], verbose=True)
```

---

## 5. `coupled_implicit_step`

```python
c_new, n_iter, converged = coupled_implicit_step(
    mecha, st, c_prev, T, psi_os_baseline,
    boundary_conditions=None, rhs=None,
    i_scenario=1, i_maturity=0, h=0,
    theta=1.0, operators='T', scheme='sg',
    method='picard', tol=1.0, max_iter=12, relaxation=1.0,
    jfnk_f_tol=None, jfnk_maxiter=None, jfnk_inner_maxiter=30,
    jfnk_line_search='armijo', jfnk_rdiff=1e-3,
    transport_step=None, verbose=False)
```

Solves $c^{n+1} = g(c^{n+1})$ where each evaluation of $g$ takes one θ-step of
transport from the **fixed** $c^n$ = `c_prev`. Unlike
`coupled_water_solute_solve(time_stepping=True)`, $c^n$ is never advanced inside
the iteration, so the result is a genuine time step of length `dt`
(`st.capacitance_params['dt']`).

| Parameter | Meaning |
|---|---|
| `c_prev` | Field at $t^n$ in the transport-vector layout of `st`; also the initial guess. |
| `psi_os_baseline` | Background $\Psi_{os}$, shape `(n_cells,)`. |
| `method` | `'picard'` or `'jfnk'`. `method='picard', max_iter=1` reproduces explicit (lagged) coupling. |
| `tol` | [hPa] Picard: $RT\max\lvert\Delta q\rvert < $ `tol`; JFNK default `f_tol = tol/(RT)`. |
| `transport_step` | Optional callable `() -> c_new` replacing the default `st.solve(c_prev=c_prev, ...)`, e.g. a multi-solute block solve. It must always step from $t^n$. |

Returns `c_new` (hydraulic state in `mecha` consistent with it to within
`tol`), `n_iter`, `converged`.

---

## 6. Examples in the repository: `simulations/needle.py`

[simulations/needle.py](../simulations/needle.py) uses both functions behind
small strategy dataclasses selected through `Experiment(coupling=...)`:

- [`NoCoupling`](../simulations/needle.py#L373) — lagged coupling: one Picard
  pass (`max_iter=1, tol=np.inf`) per time step; for steady runs a single
  uncoupled `water_flux` + `st.solve`.
- [`Picard`](../simulations/needle.py#L391) — `tol`, `max_iter`, `relax`
  forwarded as `method='picard'`.
- [`JFNK`](../simulations/needle.py#L409) — forwards `jfnk_maxiter`,
  `jfnk_inner_maxiter`, `jfnk_line_search`, `jfnk_rdiff`.

The glue methods on the simulation object:

- [`_implicit_step`](../simulations/needle.py#L835) wraps
  `coupled_implicit_step` for transient runs (`TimeStepping` dynamics).
- [`_coupled_steady`](../simulations/needle.py#L844) wraps
  `coupled_water_solute_solve` for `SteadyState` dynamics and passes
  `SteadyState.continuation` as `continuation_steps`.
- [`_step`](../simulations/needle.py#L860) shows the `transport_step` hook:
  with starch enabled, sucrose and starch are advanced as one block via
  `MultiSoluteTransport.solve_coupled`, and the starch field is captured by
  closure.
- [`_run_transient`](../simulations/needle.py#L901) shows how
  `psi_os_baseline` is built: run `water_flux` once with no sucrose and read
  `cell.psi_os` from the cell manager.

Switching solver in an experiment:

```python
from needle import Experiment, Anatomy, DirichletBC, TimeStepping, JFNK, Picard

base = Experiment(
    anatomy=Anatomy(seed=42),
    bc=DirichletBC(psi_xyl=-200.0, air='psi', psi_air=-1.0e4),
    dynamics=TimeStepping(dt=0.1, max_steps=40, c0_meso=50.0e-6),
    coupling=JFNK(),
)
picard = base.replace(coupling=Picard(relax=0.5))
```

Other scripts built on `needle.py` (e.g.
[simulations/diurnal.py](../simulations/diurnal.py),
[simulations/flux_budget.py](../simulations/flux_budget.py)) inherit the same
coupling options.

---

## 7. Practical guidance

- Start with `operators='T', scheme='sg'`. Fall back to `operators='D'` if
  advection makes transport ill-conditioned.
- If Picard oscillates or diverges, first try `relaxation≈0.5`, then
  `method='jfnk'`.
- For cold starts with strong osmotic drive, add
  `continuation_steps=[0.25, 0.5, 1.0]`.
- If JFNK stalls with GMRES making no progress, increase `jfnk_rdiff` or
  `jfnk_inner_maxiter`.
- Use `component_tol` when the flow is near-isosmotic to check that both
  $\psi_p$ and $\psi_{os}$ have settled.
