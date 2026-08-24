"""
needle_dynamic.py
=================
Transient (time-series) sucrose transport in a conifer needle cross-section.

GRANAP NeedleAnatomy, physics from InData.needle_defaults(), barrier=1;
xylem water potential PSI_XYL as supply and an air BC — a water potential or an
RH via the Kelvin equation — on the evaporating wall_air nodes.

Solute — INITIAL-IMPULSE dynamic transport (this is the difference from the
steady script):
  * t = 0 : an impulse of sucrose C_PULSE is placed in the mesophyll cells; the
            rest of the network is solute-free.  The mesophyll is NOT held at a
            fixed concentration — it is released and free to redistribute.
  * t > 0 : the full transport operator is marched in time with implicit-Euler. 
  * Sink  : the phloem-loading complex (Strasburger + sieve) and the xylem are
            held at c = 0 Dirichlet, so solute leaving the mesophyll accumulates
            at / is removed by the sink.

Water–solute coupling — FULL two-way coupling every time step, exactly as the
steady script (needle_steady.py) but marched in time.  At each implicit-Euler
step the cell osmotic potentials are refreshed from the current concentration
field via van't Hoff (Ψ_os = baseline − R·T·c), the hydraulics are re-solved
with those Ψ_os (Kedem–Katchalsky), and the fresh water flow is fed back into
the advective transport operator before the step is taken.  So as the sucrose
pulse redistributes, the osmotic drive it exerts on the water flow evolves with
it — the transport operator is NO longer frozen at the t=0 state.

Stopping — the march ends when EITHER
  * equilibrium is reached : max|Δc| per step < EQUIL_TOL, or
  * ≥ SINK_FRACTION (90 %) of the initial solute mass has left the interior
    (been transported to the sink).

Output (~/simulations/outputs/):
  needle_dynamic_concentration.gif   – every time step of the cell c field
  needle_dynamic_massbalance.png     – interior mass / fraction-to-sink vs time
  needle_dynamic_results.txt         – key numbers

Units: lengths µm/cm, volumes cm³, conc mol/cm³ (=10³ M), rates mol/d, time d.
"""

import os
import sys
import copy
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR    = os.path.join(_SCRIPT_DIR, 'outputs')
os.makedirs(_OUT_DIR, exist_ok=True)

_MECHA_SRC  = os.path.join(_SCRIPT_DIR, '..', '..', 'MECHA-dev', 'src')
_GRANAP_SRC = os.path.join(_SCRIPT_DIR, '..', '..', 'GRANAP-dev', 'src')
for p in (_MECHA_SRC, _GRANAP_SRC):
    sys.path.insert(0, os.path.abspath(p))

from openalea.granap.needle_class import NeedleAnatomy
from openalea.mecha.mecha_class import Mecha
from openalea.mecha.utils.data_loader import InData, BoundaryData
from openalea.mecha.utils.network_builder import NetworkBuilder
from openalea.mecha.utils.solute_transport import SoluteTransport
from openalea.mecha.calibration.forward_model import rh_to_water_potential


# ── Parameters ────────────────────────────────────────────────────────────────

# ---- Water BCs (Dirichlet) — identical to needle_steady.py -------------------
PSI_XYL  = -200.0   # hPa  xylem water potential (supply side)
AIR_MODE = 'psi'    # 'psi' (default) | 'rh'
PSI_ATM  = -1e4     # hPa  air-space water potential (AIR_MODE='psi')
RH_AIR   = 0.99     #  -   air-space relative humidity in (0, 1] (AIR_MODE='rh')

# ---- Solute impulse + transport ----------------------------------------------
C_PULSE  = 50e-6    # mol/cm³  initial mesophyll sucrose impulse (= 50 mM)
C_SINK   = 0.0      # mol/cm³  phloem-loading + xylem Dirichlet sink

# --- Solute transport coefficients  ----------
D_PD  = 5e-1         # cm²/d  plasmodesmatal diffusivity (symplastic loading route)
D_APO = 0.1          # cm²/d  effective apoplastic wall diffusivity
D_MEM = 1e-6         # cm²/d  passive transmembrane diffusivity
SIGMA_SUCROSE = 0.6  # membrane reflection coefficient

T_KELVIN = 298.15   # K  (25 °C) — van't Hoff Ψ_os = −R·T·c for the water solve

# ---- Time integration --------------------------------------------------------
DT           = 1e-1   # d   time step (implicit Euler, θ=1 → oscillation-free)
THETA        = 1.0    # 1.0 implicit Euler | 0.5 Crank-Nicolson
MAX_STEPS    = 300    # hard cap on the number of steps
# Equilibrium: per-step change below a small fraction of the initial pulse.
EQUIL_TOL    = 1e-6 * C_PULSE   # mol/cm³  max|Δc| per step below this ⇒ steady
SINK_FRACTION = 0.90  # stop once ≥90 % of the initial mass has left the interior

# ---- Transport operator regime -----------------------------------------------
# 'T' (advection + diffusion) is the physically coupled regime: the osmotic drive
# from the current sucrose field feeds the water flow, which advects the solute.
# Scharfetter–Gummel + implicit Euler is oscillation-free at any Peclet, so the
# advective coupling is stable in this transient (see Section 7).  'D'
# (diffusion-only) is flow-independent and drops the advective feedback.
OPS    = 'T'          # 'D' | 'T'
SCHEME = 'sg'         # 'upwind' | 'sg' (sg requires OPS='T')

# ---- Indices -----------------------------------------------------------------
H_IDX = 0
I_MAT = 0
I_SCE = 1


# ── Plot helper (cell-polygon choropleth) ─────────────────────────────────────
def _cell_values(mecha_obj, sol_arr):
    """Map a full-network array to a per-cell value column on the cell GDF."""
    gdf = mecha_obj.network._cells_gdf.copy()
    nwj_loc = mecha_obj.network.n_wall_junction
    idx_map = mecha_obj.indice

    def _val(cid):
        try:
            return float(sol_arr[idx_map[nwj_loc + int(cid)]])
        except (KeyError, IndexError, TypeError):
            return float('nan')

    gdf['value'] = gdf['id_cell'].apply(_val)
    return gdf


# ── 1. Build GRANAP needle ────────────────────────────────────────────────────
print("=== 1. Building GRANAP NeedleAnatomy ===")
needle = NeedleAnatomy()
needle.export_to_adjencymatrix()
print(f"  Cells generated: {len(needle._cells_gdf)}")


# ── 2. Build MECHA network + needle-default physics ──────────────────────────
print("\n=== 2. Building MECHA network (needle defaults, barrier=1) ===")
network = NetworkBuilder(needle)
network.populate_from_network()

data = InData.needle_defaults()

# Scenario 1 (I_SCE) with the osmotic operator ON (s_factor = σ).
# Osmotic magnitudes zeroed so the only osmotic term is the dynamic sucrose one.
_s1 = copy.deepcopy(BoundaryData().scenarios[0])
_s1['s_factor']           = SIGMA_SUCROSE
_s1['s_hetero']           = 0
_s1['os_hetero']          = 0
_s1['os_cortex']          = 0.0
_s1['osmotic_sieve']      = 0.0
_s1['osmotic_xyl']        = 0.0
_s1['osmotic_endo']       = 0.0
_s1['osmotic_left_soil']  = 0.0
_s1['osmotic_right_soil'] = 0.0
_s1['psi_soil_left']      = 0.0
data.boundary.add_scenario(_s1)

mecha = Mecha(data, network=network)
nwj     = mecha.network.n_wall_junction
n_cells = mecha.network.n_cells
N_TOTAL = nwj + n_cells
print(f"  n_wall_junction={nwj}  n_cells={n_cells}")

mecha.psi_xyl[1, I_MAT, I_SCE]   = PSI_XYL
mecha.psi_sieve[1, I_MAT, I_SCE] = np.nan

if AIR_MODE == 'rh':
    psi_air = rh_to_water_potential(RH_AIR, T_KELVIN)
    print(f"  Air BC: RH={RH_AIR:.3f} → Ψ_air={psi_air:.1f} hPa (Kelvin, T={T_KELVIN:.2f} K)")
elif AIR_MODE == 'psi':
    psi_air = PSI_ATM
    print(f"  Air BC: Ψ_air={psi_air:.1f} hPa (water potential)")
else:
    raise ValueError(f"AIR_MODE must be 'psi' or 'rh', got {AIR_MODE!r}.")


# ── 3. Identify tissue cell indices ──────────────────────────────────────────
print("\n=== 3. Identifying tissue cell indices ===")

meso_cell_ids        = []
strasburger_cell_ids = []
phloem_cell_ids      = []
xylem_cell_ids       = []
airspace_cell_ids    = []

for nd, d in mecha.network.graph.nodes(data=True):
    idx = mecha.indice[nd]
    if idx < nwj:
        continue
    cell_id = idx - nwj
    cg      = int(d.get('cgroup', -1))
    ct      = str(d.get('cell_type', ''))

    if mecha.network.is_wall_air_cell(nd):
        airspace_cell_ids.append(cell_id)
    elif cg == 4 and ct == 'mesophyll':
        meso_cell_ids.append(cell_id)
    elif cg == 12 and ct == 'Strasburger cell':
        strasburger_cell_ids.append(cell_id)
    elif cg == 11:
        phloem_cell_ids.append(cell_id)
    elif cg == 13:
        xylem_cell_ids.append(cell_id)

if not meso_cell_ids:
    for nd, d in mecha.network.graph.nodes(data=True):
        idx = mecha.indice[nd]
        if idx >= nwj and int(d.get('cgroup', -1)) == 4 and not mecha.network.is_wall_air_cell(nd):
            meso_cell_ids.append(idx - nwj)

phloem_loading_ids = strasburger_cell_ids + phloem_cell_ids

# Air/transpiration Dirichlet BC on the evaporating wall nodes (needle pathway).
mecha.set_air_wall_bc(psi_air)

print(f"  Mesophyll cells (impulse)          : {len(meso_cell_ids)}")
print(f"  Phloem loading complex (sink)      : {len(phloem_loading_ids)}")
print(f"  Xylem cells (sink)                 : {len(xylem_cell_ids)}")
print(f"  Air spaces (protect_topology)      : {len(airspace_cell_ids)}")


# ── 4. Initial water flow solve (osmosis from the initial pulse) ─────────────
# The march is FULLY COUPLED: at every step Ψ_os is refreshed from the current
# concentration and the hydraulics are re-solved (see Section 7).  This section
# only establishes the t=0 flow and captures the osmotic BASELINE (Ψ_os of any
# non-dynamic background solutes), onto which the van't Hoff term −R·T·c is added
# each step.  The baseline MUST be read from a solve BEFORE the pulse is imposed.
print(f"\n=== 4. Initial water flow solve  PSI_XYL={PSI_XYL:.0f}  Ψ_air={psi_air:.0f} hPa ===")

manager = mecha.network.cell_manager

# Baseline Ψ_os (background solutes not tracked by the transport solve). Read it
# from the pulse-free hydraulic solve so the dynamic term is added, not doubled.
mecha.water_flux(h=H_IDX, verbose=False)
psi_os_baseline = np.array([c.psi_os if c.psi_os is not None else 0.0 for c in manager])

# t = 0 osmotic state from the initial pulse, then the first hydraulic solve.
c0_cells = np.zeros(n_cells)
for cid in meso_cell_ids:
    c0_cells[cid] = C_PULSE
manager.set_osmotic_from_concentration(c0_cells, nwj, T_KELVIN, psi_os_baseline)
mecha.water_flux(h=H_IDX, use_stored_psi_os=True, verbose=False)
print("  Initial hydraulic solve done (osmotic state from the t=0 pulse).")


# ── 5. Build the transport operator + capacitance (rebuilt each step) ────────
# The SoluteTransport instance is reused across the march; its advective operator
# is rebuilt from the current water flow on every st.solve() call (Section 7), so
# the operator tracks the evolving osmotic drive rather than being frozen at t=0.
print("\n=== 5. Building SoluteTransport operator (full) ===")
DP_FULL = dict(apo_wall=D_APO, membrane=D_MEM, plasmodesmata=D_PD,
               sigma={cg: SIGMA_SUCROSE for cg in range(1, 20)})
cap = {'dt': DT}                     # capacitance ON → dynamic stepping
st  = SoluteTransport(mecha, DP_FULL, cap, mode='full')

D_mat = st.build_diffusion_matrix(H_IDX, I_MAT)
D_row_norms       = np.array(np.abs(D_mat).sum(axis=1)).ravel()
isolated_node_ids = list(np.where(D_row_norms == 0)[0])
n_comp, comp_labels = csgraph.connected_components(D_mat, directed=False, connection='weak')
print(f"  Zero-row nodes: {len(isolated_node_ids)}   connected components: {n_comp}")

# Anchor any component with no Dirichlet node so the implicit solve is non-singular.
anchored = set([nwj + cid for cid in phloem_loading_ids]
               + [nwj + cid for cid in xylem_cell_ids])
isolated_set = set(isolated_node_ids)
extra_anchor_ids = []
for comp_id in range(n_comp):
    members = np.where(comp_labels == comp_id)[0]
    if any(m in anchored for m in members):
        continue
    non_iso = [m for m in members if m not in isolated_set]
    if non_iso:
        extra_anchor_ids.append(non_iso[0])

# Dirichlet BCs held constant over the whole march (sink + anchors).
bc = {}
for nid in isolated_node_ids:
    bc[nid] = C_SINK
for nid in extra_anchor_ids:
    bc[nid] = C_SINK
for cid in phloem_loading_ids:
    bc[nwj + cid] = C_SINK
for cid in xylem_cell_ids:
    bc[nwj + cid] = C_SINK
print(f"  Dirichlet sink/anchor nodes: {len(bc)}")

# Node volumes (cm³) for the interior-mass budget.
node_vols = st._compute_node_volumes(I_MAT)
# "Interior" = every node NOT pinned by a Dirichlet BC (the sink removes mass).
interior_mask = np.ones(N_TOTAL, dtype=bool)
for nid in bc:
    if 0 <= nid < N_TOTAL:
        interior_mask[nid] = False


# ── 6. Initial condition (impulse) ───────────────────────────────────────────
c = np.zeros(N_TOTAL)
for cid in meso_cell_ids:
    c[nwj + cid] = C_PULSE
# Apply the Dirichlet values at t=0 too.
for nid, val in bc.items():
    if 0 <= nid < N_TOTAL:
        c[nid] = val

mass0 = float(np.sum(node_vols[interior_mask] * c[interior_mask]))
print(f"\n=== 6. Initial impulse ===")
print(f"  Initial interior solute mass: {mass0*1e12:.4f} pmol "
      f"(C_PULSE={C_PULSE*1e6:.0f} µM in {len(meso_cell_ids)} mesophyll cells)")


# ── 7. Coupled time march ─────────────────────────────────────────────────────
# Each implicit-Euler step is a full water–solute coupling step (as in
# needle_steady.py's _apply_step, but advancing real time):
#   (a) refresh Ψ_os = baseline − R·T·c from the CURRENT cell concentrations,
#   (b) re-solve the hydraulics with those Ψ_os (use_stored_psi_os=True),
#   (c) take one implicit-Euler transport step on the freshly updated flow.
# Implicit Euler (θ=1) is oscillation-free at any Peclet, so advective coupling
# (operators='T', scheme='sg') is stable here — unlike the direct steady solve,
# which is why needle_steady.py falls back to diffusion-only for its fixed point.
print(f"\n=== 7. Coupled time march  (dt={DT} d, θ={THETA}, ops='{OPS}', scheme='{SCHEME}') ===")

rhs0 = np.zeros(st._matrix_size)

times          = [0.0]
frames         = [c.copy()]
interior_mass  = [mass0]
frac_to_sink   = [0.0]

stop_reason = f"reached MAX_STEPS={MAX_STEPS}"
for step in range(1, MAX_STEPS + 1):
    # (a)+(b) Two-way coupling: current c → Ψ_os → re-solve water flow. The fresh
    # edge fluxes / cell Ψ_os make st.solve rebuild its advective operator below.
    manager.set_osmotic_from_concentration(
        c[nwj: nwj + n_cells], nwj, T_KELVIN, psi_os_baseline)
    mecha.water_flux(h=H_IDX, use_stored_psi_os=True, verbose=False)

    # (c) One implicit-Euler transport step on the updated flow.
    c_new = st.solve(
        h=H_IDX, i_maturity=I_MAT, i_scenario=I_SCE,
        rhs=rhs0.copy(), boundary_conditions=bc,
        c_prev=c, theta=THETA, operators=OPS, scheme=SCHEME,
    )
    dmax = float(np.max(np.abs(c_new - c)))
    c = c_new

    mass = float(np.sum(node_vols[interior_mask] * c[interior_mask]))
    frac = 1.0 - mass / mass0 if mass0 > 0 else 0.0

    times.append(step * DT)
    frames.append(c.copy())
    interior_mass.append(mass)
    frac_to_sink.append(frac)

    if step % 20 == 0 or step == 1:
        print(f"  step {step:4d}  t={step*DT:6.3f} d  max|Δc|={dmax:.3e}  "
              f"interior mass={mass*1e12:8.4f} pmol  to-sink={frac*100:5.1f}%")

    if frac >= SINK_FRACTION:
        stop_reason = f"≥{SINK_FRACTION*100:.0f}% of solute transported to sink"
        break
    if dmax < EQUIL_TOL:
        stop_reason = "equilibrium reached (max|Δc| < EQUIL_TOL)"
        break

n_frames = len(frames)
print(f"  Stopped after {n_frames-1} step(s): {stop_reason}")
print(f"  Final: t={times[-1]:.1f} d  interior mass={interior_mass[-1]*1e12:.4f} pmol  "
      f"to-sink={frac_to_sink[-1]*100:.1f}%")


# ── 8. Animated GIF over all time steps ──────────────────────────────────────
# Subsample to at most GIF_MAX_FRAMES evenly-spaced steps (always keep the last)
# so the GIF stays a reasonable size while still showcasing the whole transient.
print("\n=== 8. Rendering GIF ===")
GIF_MAX_FRAMES = 150
stride = max(1, n_frames // GIF_MAX_FRAMES)
gif_idx = list(range(0, n_frames, stride))
if gif_idx[-1] != n_frames - 1:
    gif_idx.append(n_frames - 1)
vmax = C_PULSE * 1e6

fig, ax = plt.subplots(figsize=(9, 7))

def _draw(k):
    frame_idx = gif_idx[k]
    ax.clear()
    gdf = _cell_values(mecha, frames[frame_idx] * 1e6)
    gdf.plot(ax=ax, column='value', cmap='plasma', edgecolor='black',
             linewidth=0.3, vmin=0.0, vmax=vmax,
             missing_kwds={'color': 'lightgray'})
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x (µm)')
    ax.set_ylabel('y (µm)')
    ax.set_title(
        f'Needle sucrose transport — impulse release\n'
        f't = {times[frame_idx]:.3f} d   |   to sink: {frac_to_sink[frame_idx]*100:.1f}%'
    )
    return ax.collections

_sm = plt.cm.ScalarMappable(cmap='plasma',
                            norm=plt.Normalize(vmin=0.0, vmax=vmax))
fig.colorbar(_sm, ax=ax, label='c (µM sucrose)')

anim = FuncAnimation(fig, _draw, frames=len(gif_idx), blit=False)
gif_path = os.path.join(_OUT_DIR, 'needle_dynamic_concentration.gif')
anim.save(gif_path, writer=PillowWriter(fps=12))
plt.close(fig)
print(f"  Saved → {gif_path}  ({len(gif_idx)} frames of {n_frames} steps)")


# ── 9. Mass-balance plot ─────────────────────────────────────────────────────
print("\n=== 9. Mass-balance plot ===")
fig2, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(times, np.array(interior_mass) * 1e12, 'C0-', label='interior mass')
ax1.set_xlabel('time (d)')
ax1.set_ylabel('interior solute mass (pmol)', color='C0')
ax1.tick_params(axis='y', labelcolor='C0')
ax2 = ax1.twinx()
ax2.plot(times, np.array(frac_to_sink) * 100, 'C3-', label='fraction to sink')
ax2.axhline(SINK_FRACTION * 100, color='C3', ls='--', lw=0.8)
ax2.set_ylabel('fraction transported to sink (%)', color='C3')
ax2.tick_params(axis='y', labelcolor='C3')
ax1.set_title('Impulse redistribution — interior mass vs fraction reaching the sink')
fig2.tight_layout()
mb_path = os.path.join(_OUT_DIR, 'needle_dynamic_massbalance.png')
fig2.savefig(mb_path, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"  Saved → {mb_path}")


# ── 10. Results summary ───────────────────────────────────────────────────────
results_path = os.path.join(_OUT_DIR, 'needle_dynamic_results.txt')
with open(results_path, 'w') as f:
    f.write("Conifer needle — transient (impulse) sucrose transport\n")
    f.write("=" * 60 + "\n\n")
    f.write("Geometry: GRANAP NeedleAnatomy, physics from InData.needle_defaults()\n")
    f.write("  barrier=1 (Casparian strip at endodermis radial walls)\n\n")
    f.write("Water boundary conditions:\n")
    f.write(f"  PSI_XYL = {PSI_XYL:.1f} hPa (Dirichlet at xylem cells)\n")
    if AIR_MODE == 'rh':
        f.write(f"  Air BC  = RH {RH_AIR:.3f} → Ψ_air = {psi_air:.1f} hPa (Kelvin)\n\n")
    else:
        f.write(f"  Air BC  = Ψ_air {psi_air:.1f} hPa (water potential)\n\n")
    f.write("Solute:\n")
    f.write(f"  Initial impulse C_PULSE = {C_PULSE*1e6:.1f} µM in "
            f"{len(meso_cell_ids)} mesophyll cells\n")
    f.write(f"  Sink (Dirichlet c=0): phloem loading ({len(phloem_loading_ids)}) "
            f"+ xylem ({len(xylem_cell_ids)})\n")
    f.write(f"  D_PD={D_PD:.2e}  D_APO={D_APO:.2e}  D_MEM={D_MEM:.2e} cm²/d  "
            f"σ={SIGMA_SUCROSE}\n\n")
    f.write("Time integration (fully coupled water–solute march):\n")
    f.write(f"  dt={DT} d  θ={THETA}  operators='{OPS}'  scheme='{SCHEME}'\n")
    f.write("  Ψ_os = baseline − R·T·c refreshed each step; hydraulics re-solved\n")
    f.write("  every step so the osmotic drive tracks the evolving sucrose field.\n")
    f.write(f"  equilibrium tol={EQUIL_TOL:.1e}  sink fraction target={SINK_FRACTION*100:.0f}%\n")
    f.write(f"  steps taken={n_frames-1}  stop: {stop_reason}\n\n")
    f.write("Results:\n")
    f.write(f"  initial interior mass = {mass0*1e12:.4f} pmol\n")
    f.write(f"  final   interior mass = {interior_mass[-1]*1e12:.4f} pmol\n")
    f.write(f"  final time            = {times[-1]:.3f} d\n")
    f.write(f"  fraction to sink      = {frac_to_sink[-1]*100:.1f}%\n")

print(f"\n  Results written to: {results_path}")
print("\n=== DONE ===")
print(f"Outputs in: {_OUT_DIR}")
