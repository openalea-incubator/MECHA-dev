import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
nb['metadata'] = {
    "kernelspec": {"display_name": "mecha", "language": "python", "name": "python3"},
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.13.14",
    },
}

cells = []

def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))

def code(src):
    cells.append(nbf.v4.new_code_cell(src))

# ---------------------------------------------------------------------------
md(r"""# Tutorial 04 – Capacitance, reaction, and both together

`SoluteTransport` assembles the transport equation

$$ C_i\,\frac{dc_i}{dt} = (T c)_i + s_i \, ,\qquad T = D + A - R $$

as two independent diagonal pieces — a **capacitance** matrix $C$ (`build_capacitance`)
and a **reaction** matrix $R$ (`build_reaction_matrix`) — glued onto the same spatial
operator $T = D + A$. They are built by the same node-volume/mode-slicing machinery
(`_build_node_diagonal`) but play very different roles in the equation:

* $C$ sits on the **left-hand side**, multiplying the time derivative. It only sets
  *how fast* the system moves.
* $R$ is subtracted **into $T$** as a distributed sink. It changes *what the system
  moves towards*.

This tutorial isolates each effect in turn and then recombines them, using a
epidermis → xylem transport scenario.
We set a constant solute source at the epidermis (Dirichlet BC, $c=1$), 
diffusing/advecting through the wall–membrane–plasmodesmata network
towards the xylem, with first-order degradation along the way.

1. **Capacitance alone** ($k_\text{deg}=0$) – same equilibrium, different rate.
2. **Reaction alone** (steady state, no capacitance) – degradation reshapes the
   equilibrium itself into a decaying profile.
3. **Both together** (dynamic Crank–Nicolson) – the transient from Part 1 relaxes
   onto the reaction-shaped steady state from Part 2.

---""")

# ---------------------------------------------------------------------------
md(r"""## Setup

We build the hydraulic network from `tutorials/inputs/Arabidopsis.xml`
(Pascut et al. 2021 Arabidopsis root cross-section) and solve `water_flux()` once —
`SoluteTransport` reuses its wall/membrane/plasmodesmata conductances for diffusion
and its solved edge fluxes for advection.

The source/target tissues are the epidermis and the pericycle, addressed through
`cell_manager.epidermis` / `.pericycle` and shifted by `n_wall_junction` into
full-network node indices. Diffusivities and the degradation constant come straight
from `mecha.hormones` (MECHA's built-in first-hormone defaults — no separate hormones
file is needed for this network).

We deliberately select the *elongation zone* maturity stage, and the reason is
structural, not stylistic. MECHA's default maturity stage carries `barrier=1` (`En_cs`, an
endodermal Casparian strip), which sets the endodermis radial-wall conductance to
~1e-16 — effectively closing the apoplastic route across the endodermis and forcing
everything through the symplast. That is realistic for a mature root, but it also
spans ~16 orders of magnitude in wall conductance in the same matrix, which makes
the assembled operator numerically ill-conditioned for a plain sparse solve. We
follow the same choice and select `barrier=0` ("No apo barrier", elongation zone) so
the apoplastic path stays open end-to-end and the linear systems below stay
well-conditioned.

Two more setup notes:

* `im_scale` rescales the raw image-polygon coordinates into µm; `Arabidopsis.xml`
  needs `im_scale = 0.42` (its default of 1000.0 is calibrated for a different
  dataset and silently distorts every wall length, cell area and volume otherwise).
* We deliberately leave `pressure_xyl_prox` at its default (a mild pull) rather than
  imposing a strong transpiration BC. A strong xylem pressure sink pulls *water* out
  of the network at the xylem node, but `SoluteTransport` has no matching solute
  sink there — solute keeps arriving by advection with nowhere to leave, so it
  accumulates without bound (concentrations above the source's `c=1` are then
  *expected*, not a bug). That is a real, separate effect worth its own tutorial;
  here we want a clean, mass-conserving comparison, so we avoid triggering it.""")

code(r"""import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Make the in-repo `openalea.mecha` package importable without installing it.
TUTORIAL_DIR = os.path.dirname(os.path.abspath("tutorial_04_capacitance_reaction.ipynb"))
sys.path.insert(0, os.path.join(TUTORIAL_DIR, "..", "src"))

from openalea.mecha.legacy import Mecha, InData, SoluteTransport, ScenarioBuilder

ARABIDOPSIS = os.path.join(TUTORIAL_DIR, "inputs", "Arabidopsis.xml")
OUT_DIR     = os.path.join(TUTORIAL_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

data = InData(cellset_file=ARABIDOPSIS)
data.geometry.set_maturity_stages([0], height=[200.0])   # elongation zone, no Casparian strip
data.geometry.im_scale = 0.42                            # Arabidopsis.xml's image-to-µm scale
mecha = Mecha(data)
mecha.water_flux()

cm  = mecha.network.cell_manager
nwj = mecha.network.n_wall_junction
n_total = mecha.network.graph.number_of_nodes()

source_cells = list(cm.epidermis)
target_cells = list(cm.pericycle)
source_node_ids = [nwj + c.cell_id for c in source_cells]
target_node_ids = [nwj + c.cell_id for c in target_cells]

print(f"Network: {nwj} wall/junction nodes, {mecha.network.n_cells} cells "
      f"({n_total} nodes total)")
print(f"Epidermis (source): {len(source_cells)} cells")
print(f"Pericycle (target): {len(target_cells)} cells")""")

code(r"""# Diffusivities and degradation constant: MECHA's default first-hormone
# parameters, read off mecha.hormones.
diffusion_params = {
    'apo_wall':      mecha.hormones.diff_pw1,
    'plasmodesmata': mecha.hormones.diff_pd1,
    'membrane':      mecha.hormones.diff_mb1,
}
K_DEG_DEFAULT = mecha.hormones.degrad1

print("diffusion_params:", diffusion_params)
print(f"degrad1 (default reaction rate): {K_DEG_DEFAULT} /d")

boundary_conditions = {node_id: 1.0 for node_id in source_node_ids}
rhs = np.zeros(n_total)

DT_DAY = 0.02   # 28.8 min per step
THETA  = 1.0    # implicit Euler — unconditionally stable at any Peclet number""")

# ---------------------------------------------------------------------------
md(r"""## Part 1 — Capacitance alone

Set $k_\text{deg}=0$: $T = D + A$ is exactly mass-conserving, so its rows sum to
zero and the constant vector $\mathbf{1}$ is in its null space ($T\mathbf{1}=0$).
With a single Dirichlet source and no sink anywhere else, the *only* steady state
compatible with that source is the trivial uniform one: eventually **the whole
network saturates to $c=1$**, whatever $C$ is.

$C$ does not touch that fixed point — it only rescales the local relaxation
time $\tau_i \approx C_i / |T_{ii}|$. We check this by running the same dynamics
twice, changing only the storage fraction:

* **unit storage** — `C_wall = C_cell = 1.0` (the entire node volume stores solute)
* **physiological storage** — `C_wall/C_cell = mecha.boundary.water_fraction_apo/sym`
  (only the water-filled fraction of each node can hold a dissolved solute)

A *smaller* $C$ means the same flux produces a *larger* $dc/dt = (Tc)/C$, so the
physiological case should reach the c=1 plateau faster, not slower.""")

code(r"""STORAGE_CASES = {
    "unit storage (C=1)": dict(
        C_wall=1.0, C_cell=1.0, color="#4393c3"),
    "physiological storage (water fraction)": dict(
        C_wall=mecha.boundary.water_fraction_apo,
        C_cell=mecha.boundary.water_fraction_sym,
        color="#d6604d"),
}

N_STEPS_1 = 60   # 1.2 d

part1_traces = {}
for label, cfg in STORAGE_CASES.items():
    capacitance_params = {'dt': DT_DAY, 'C_wall': cfg['C_wall'], 'C_cell': cfg['C_cell']}
    st = SoluteTransport(mecha, diffusion_params, capacitance_params,
                         mode='full', reaction_params={'k_deg': 0.0})

    c = np.zeros(n_total)
    t_h    = [0.0]
    c_peri = [float(np.mean(c[target_node_ids]))]
    for step in range(1, N_STEPS_1 + 1):
        c = st.solve(h=0, i_maturity=0, i_scenario=0, rhs=rhs,
                     boundary_conditions=boundary_conditions,
                     c_prev=c, theta=THETA, operators='T')
        t_h.append(step * DT_DAY * 24.0)
        c_peri.append(float(np.mean(c[target_node_ids])))
    part1_traces[label] = (np.array(t_h), np.array(c_peri), cfg['color'])
    print(f"{label:<40s} pericycle c after {t_h[-1]:.2f} h: {c_peri[-1]:.4f}")""")

code(r"""fig, ax = plt.subplots(figsize=(7, 5))
for label, (t_h, c_peri, color) in part1_traces.items():
    ax.plot(t_h, c_peri, '-o', ms=3, color=color, label=label)
ax.axhline(1.0, color='k', ls=':', lw=1, label='equilibrium (c = 1 everywhere)')
ax.set_xlabel('time (h)')
ax.set_ylabel('mean pericycle concentration (a.u.)')
ax.set_title('Capacitance alone ($k_\\mathrm{deg}=0$): same equilibrium, different rate')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "part1_capacitance.png"), dpi=150)
plt.show()""")

# ---------------------------------------------------------------------------
md(r"""## Part 2 — Reaction alone

Now switch off capacitance (`capacitance_params=None`, i.e. `dt` absent) and switch
on degradation. Without a time derivative, `solve()` goes straight to the steady
branch $T c = 0$ (subject to the epidermis Dirichlet override), with
$T = D + A - R$.

Adding $R = \mathrm{diag}(k_\text{deg} V_i)$ breaks the null-space property from
Part 1: $R\,\mathbf{1} \ne 0$ whenever $k_\text{deg}>0$, so the flat $c=1$ solution
no longer solves $Tc=0$. The steady state becomes a genuine **spatial gradient** —
a balance between diffusive/advective supply from the epidermis and distributed
degradation everywhere else, decaying over a penetration length that shrinks as
$k_\text{deg}$ grows. This is the conceptual role of $R$: it is what turns a flat
plateau into a shaped, distance-dependent gradient — the network analogue of the
classical reaction–diffusion penetration depth $\lambda \sim \sqrt{D_\text{eff}/k_\text{deg}}$.

We sweep $k_\text{deg}$ from 0 (recovering Part 1's flat profile as a sanity check)
up past `mecha.hormones.degrad1` (MECHA's default).""")

code(r"""def radial_profile(mecha, c, cx, cy):
    # (r, c) pairs for every positioned node, in full-network indexing.
    rs, cs = [], []
    for nd, d in mecha.network.graph.nodes(data=True):
        if 'position' not in d:
            continue
        idx = mecha.indice[nd]
        rs.append(np.hypot(d['position'][0] - cx, d['position'][1] - cy))
        cs.append(float(c[idx]))
    return np.array(rs), np.array(cs)

cx, cy = ScenarioBuilder.root_center(mecha)
K_DEG_VALUES = [0.0, 0.25 * K_DEG_DEFAULT, K_DEG_DEFAULT, 3.0 * K_DEG_DEFAULT]
colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(K_DEG_VALUES)))

fig, ax = plt.subplots(figsize=(7, 5))
peri_means = {}
for k_deg, color in zip(K_DEG_VALUES, colors):
    st = SoluteTransport(mecha, diffusion_params, capacitance_params=None,
                         mode='full', reaction_params={'k_deg': k_deg})
    c_steady = st.solve(h=0, i_maturity=0, i_scenario=0, rhs=rhs,
                        boundary_conditions=boundary_conditions,
                        operators='T')
    r, cvals = radial_profile(mecha, c_steady, cx, cy)
    order = np.argsort(r)
    ax.plot(r[order], cvals[order], '.', ms=3, color=color,
           label=f"k_deg = {k_deg:.1f} /d")
    peri_means[k_deg] = float(np.mean(c_steady[target_node_ids]))

ax.set_xlabel('distance from root centre (µm)')
ax.set_ylabel('steady-state concentration (a.u.)')
ax.set_title('Reaction alone: degradation shortens the penetration depth')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "part2_reaction.png"), dpi=150)
plt.show()

for k_deg, m in peri_means.items():
    print(f"k_deg = {k_deg:>7.2f} /d   mean pericycle c = {m:.4f}")""")

md(r"""Note the $k_\text{deg}=0$ curve: a flat line at $c=1$ everywhere — exactly
the equilibrium Part 1's dynamics were relaxing towards. Reaction alone recovers
that limit as a special case; what it *adds* is the decaying shape at $k_\text{deg}>0$,
and a strictly lower, distance-dependent pericycle concentration as degradation
strengthens.""")

# ---------------------------------------------------------------------------
md(r"""## Part 3 — Both together

This is the full transport equation: dynamic Crank–Nicolson stepping (here
$\theta=1$, implicit Euler) with **both** the physiological capacitance from Part 1 and the
default degradation rate from Part 2 active in the same `T = D + A - R` operator.

The prediction from Parts 1–2: the *shape* of the approach to steady state is set
by $C$ (Part 1's relaxation-rate story), but the *plateau* it relaxes onto is no
longer $c=1$ — it is the reaction-shaped steady pericycle concentration computed
in Part 2 for `k_deg = K_DEG_DEFAULT`. We overlay that value as a horizontal
reference line and check the dynamic run lands on it.""")

code(r"""capacitance_params_full = {
    'dt':     DT_DAY,
    'C_wall': mecha.boundary.water_fraction_apo,
    'C_cell': mecha.boundary.water_fraction_sym,
}
reaction_params_full = {'k_deg': K_DEG_DEFAULT}

st_full = SoluteTransport(mecha, diffusion_params, capacitance_params_full,
                          mode='full', reaction_params=reaction_params_full)

N_STEPS_3 = 60   # 1.2 d, same window as Part 1
c = np.zeros(n_total)
t_h_full    = [0.0]
c_peri_full = [float(np.mean(c[target_node_ids]))]
for step in range(1, N_STEPS_3 + 1):
    c = st_full.solve(h=0, i_maturity=0, i_scenario=0, rhs=rhs,
                      boundary_conditions=boundary_conditions,
                      c_prev=c, theta=THETA, operators='T')
    t_h_full.append(step * DT_DAY * 24.0)
    c_peri_full.append(float(np.mean(c[target_node_ids])))

steady_ref = peri_means[K_DEG_DEFAULT]
print(f"Dynamic run, pericycle c after {t_h_full[-1]:.2f} h: {c_peri_full[-1]:.4f}")
print(f"Part 2 steady state (same k_deg):                    {steady_ref:.4f}")""")

code(r"""fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(t_h_full, c_peri_full, '-', color="#2166ac", lw=2,
        label='dynamic (capacitance + reaction)')
ax.axhline(steady_ref, color='k', ls='--', lw=1.5,
          label=f'Part 2 steady state (k_deg={K_DEG_DEFAULT:.0f}/d): {steady_ref:.4f}')
ax.axhline(1.0, color='0.6', ls=':', lw=1,
          label='Part 1 equilibrium (k_deg=0): 1.0')
ax.set_xlabel('time (h)')
ax.set_ylabel('mean pericycle concentration (a.u.)')
ax.set_title('Capacitance + reaction: the transient relaxes onto the\nreaction-shaped steady state')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "part3_both.png"), dpi=150)
plt.show()""")

# ---------------------------------------------------------------------------
md(r"""## Summary

| Term | Where it enters | What it controls | Isolated in |
|------|-----------------|-------------------|-------------|
| Capacitance $C$ | left-hand side, multiplies $dc/dt$ | **rate** of relaxation; leaves the fixed point unchanged | Part 1 |
| Reaction $R$ | subtracted into $T$ | **shape/value** of the steady state (breaks the uniform null space) | Part 2 |
| Both | $[C/dt - \theta T]\,c_\text{new} = [C/dt + (1-\theta) T]\,c_\text{old} + s$ | dynamic approach onto the reaction-shaped steady state | Part 3 |

Both matrices are assembled by the same `_build_node_diagonal` helper in
`solute_transport.py` (node volumes → diagonal → mode-sliced), differing only in
their per-node coefficient ($C_\text{wall,cell}/dt$ vs. $k_\text{deg}$) and gating
condition (`dt is not None` vs. `k_deg > 0`) — the shared plumbing reflects that
both are *diagonal, volume-weighted* terms, while the split into two public
methods reflects that they play distinct, independently-toggleable roles in the
same transport equation.

Going further: to inspect the spatial field rather than a single tissue's mean, collect
one `{cell_id: c[nwj + cell_id]}` dict per step and pass the list, with the matching
times, to `export_cell_concentration_timeseries(mecha, cc_by_step, times_h, prefix=...)`
(`openalea.mecha.legacy.utils.paraview_export`), which writes a full-network ParaView time
series (`.pvd` + one `.vtk` per step).""")

nb['cells'] = cells

out_path = os.path.join("tutorials", "tutorial_04_capacitance_reaction.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("wrote", out_path, "with", len(cells), "cells")
