"""
needle_steady.py
================
Fixed-input steady-state sucrose transport in a conifer needle cross-section.

Geometry : GRANAP NeedleAnatomy (Pinus-like), physics from InData.needle_defaults()
           (barrier=1 — Casparian strip at endodermis radial walls).

Water BC : Dirichlet – xylem water potential PSI_XYL (supply) and an air BC at the
           substomatal air spaces.  The air BC is chosen either by a water potential
           (AIR_MODE='psi', default) or by a relative humidity (AIR_MODE='rh'), which
           is converted to a liquid-equivalent water potential via the Kelvin equation
           (see openalea.mecha.calibration.forward_model.rh_to_water_potential),
           exactly as in calibrate_template.py.

Solute   : full operator (apoplastic + transmembrane + plasmodesmatal), steady-state,
           two-way coupled with the water flow (van't Hoff Ψ_os → Kedem–Katchalsky).
             c_meso   = C_MESO   (Dirichlet at mesophyll cells)
             c_phloem = 0        (Dirichlet sink at phloem loading complex)
             c_xylem  = 0        (Dirichlet at xylem cells)

Outputs (~/simulations/outputs/):
  needle_steady_anatomy.png / _initial_waterpotential.png / _initial_concentration.png
  needle_steady_concentration.png / _concentration_gradient.png
  needle_steady_waterflow.png / _solute_velocity.png / _results.txt

Units: lengths µm/cm, volumes cm³, fluxes cm³/d, conc mol/cm³ (=10³ M), rates mol/d.
"""

import os
import sys
import copy
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
from openalea.mecha.utils.coupled_solver import coupled_water_solute_solve
from openalea.mecha.utils.visu import visualize
from openalea.mecha.utils.network_export import _plot_edge_vector_property
from openalea.mecha.calibration.forward_model import rh_to_water_potential


def _plot_concentration_polygon(mecha_obj, sol_arr, label, title, save_path,
                                cmap='plasma', vmin=0.0, vmax=None):
    """Polygon choropleth of *sol_arr* (indexed by the full network) on the cell GDF."""
    gdf = mecha_obj.network._cells_gdf.copy()
    nwj_loc = mecha_obj.network.n_wall_junction
    idx_map = mecha_obj.indice

    def _val(cid):
        try:
            return float(sol_arr[idx_map[nwj_loc + int(cid)]])
        except (KeyError, IndexError, TypeError):
            return float('nan')

    gdf['value'] = gdf['id_cell'].apply(_val)
    finite = gdf['value'].dropna()
    if vmax is None:
        vmax = float(np.nanpercentile(finite, 99)) if len(finite) > 0 else 1.0

    fig, ax = plt.subplots(figsize=(10, 8))
    gdf.plot(ax=ax, column='value', cmap=cmap, edgecolor='black', linewidth=0.3,
             vmin=vmin, vmax=vmax, legend=True,
             legend_kwds={'label': label, 'orientation': 'vertical'},
             missing_kwds={'color': 'lightgray'})
    ax.set_aspect('equal', 'box')
    ax.set_title(title)
    ax.set_xlabel('x (µm)')
    ax.set_ylabel('y (µm)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ── Parameters ────────────────────────────────────────────────────────────────

# ---- Water BCs (Dirichlet) --------------------------------------------------
PSI_XYL = -200.0   # hPa  xylem water potential (supply side)

# Air boundary condition at the substomatal air spaces.  Choose one of:
#   AIR_MODE = 'psi' → pin the air spaces at PSI_ATM directly (water potential).
#   AIR_MODE = 'rh'  → convert RH_AIR to a liquid-equivalent water potential via
#                      the Kelvin equation (as in calibrate_template.py).
AIR_MODE = 'psi'   # 'psi' (default) | 'rh'
PSI_ATM  = -1e4     # hPa  air-space water potential (used when AIR_MODE='psi')
RH_AIR   = 0.99    # -    air-space relative humidity in (0, 1] (used when AIR_MODE='rh')

# ---- Sucrose concentrations (fixed inputs) -----------------------------------
C_MESO   = 50e-6   # mol/cm³  (= 50 mM; typical mesophyll sucrose)
C_PHLOEM = 0.0     # mol/cm³  (reference sink; we solve the Δc)

# ---- Solute diffusivities / membrane reflection -----------------------------
D_PD  = 1e-4       # cm²/d  plasmodesmatal diffusivity for sucrose
D_APO = 0.1        # cm²/d  effective apoplastic wall diffusivity
D_MEM = 1e-6       # cm²/d  passive transmembrane diffusivity
SIGMA_SUCROSE = 0.9   # membrane reflection coefficient (Kedem–Katchalsky)

# ---- Temperature (van't Hoff Ψ_os = −R·T·c) + coupling controls --------------
T_KELVIN = 298.15  # K  (25 °C)
COUPLE_TOL     = 10.0   # hPa   convergence tol on max|Δψ_total|
COUPLE_MAXITER = 30     # outer iterations
COUPLE_RELAX   = 0.5    # under-relaxation (Picard only)
# The needle osmotic drive (Ψ_os up to ~-1240 hPa at c_meso) makes the advective
# coupling map g(c) a strong REPELLER — plain Picard diverges and even the JFNK
# finite-difference Jacobian is unreliable at operators='T'.  operators='D'
# (diffusion-only transport) makes the steady c field flow-INDEPENDENT, so the
# c → Ψ_os → water-flux → advection loop reaches a genuine fixed point in ~2
# iterations while STILL exercising the full van't Hoff / Kedem–Katchalsky
# coupling.  The advective carry the osmotic flux would add is only ~(1−σ)=10%
# and is quantified separately in Section 8b.
COUPLE_OPS     = 'D'    # 'D' diffusion-only (stable) | 'T' advection + diffusion
COUPLE_SCHEME  = 'upwind'  # 'upwind' | 'sg' (Scharfetter–Gummel; requires OPS='T')
COUPLE_TIMESTEP = False
COUPLE_THETA    = 1.0
COUPLE_DT       = 1e-2

# ---- Outer solver: Jacobian-Free Newton–Krylov (JFNK) ------------------------
COUPLE_METHOD        = 'jfnk'   # 'jfnk' | 'picard'
COUPLE_JFNK_MAXITER  = 25
COUPLE_JFNK_INNER    = 120
COUPLE_JFNK_FTOL     = None
COUPLE_JFNK_LS       = 'armijo'
COUPLE_JFNK_RDIFF    = 1e-3
COUPLE_CONTINUATION  = None   # single stage at λ=1 (D is stable, no homotopy needed)

# ---- Indices ----------------------------------------------------------------
# The coupled solver needs i_scenario >= 1 (scenario 0 carries no osmotic term).
H_IDX = 0
I_MAT = 0
I_SCE = 1


# ── 1. Build GRANAP needle ────────────────────────────────────────────────────

print("=== 1. Building GRANAP NeedleAnatomy ===")

needle = NeedleAnatomy()
needle.export_to_adjencymatrix()
print(f"  Cells generated: {len(needle._cells_gdf)}")

fig_a, ax_a = plt.subplots(figsize=(7, 5))
needle.plot_cells(show=False, ax=ax_a, title="Needle cross-section (GRANAP)")
fig_a.savefig(os.path.join(_OUT_DIR, 'needle_steady_anatomy.png'), dpi=150, bbox_inches='tight')
plt.close(fig_a)


# ── 2. Build MECHA network + needle-default physics ──────────────────────────

print("\n=== 2. Building MECHA network (needle defaults, barrier=1) ===")
network = NetworkBuilder(needle)
network.populate_from_network()

data = InData.needle_defaults()

# Build scenario 1 (I_SCE) with the osmotic operator ON: s_factor = σ activates the
# reflection-coefficient term so the concentration-derived Ψ_os = −R·T·c drives the
# transmembrane water flux.
#
# IMPORTANT: needle_defaults() clears every scenario-0 key to NaN (except
# psi_soil_left). The hydraulic assembly (initialize_scenarios) reads the soil/xylem
# osmotic-PROFILE fields (psi_soil_right, osmotic_symmetry/shape/diffusivity_*, …) to
# build the RHS, so leaving them NaN poisons the entire rhs with NaN and every solve
# returns NaN → 0.  The coupled JFNK residual g(c) then evaluates on a broken solve
# and cannot converge.  We therefore seed scenario 1 from a FULL default scenario
# (BoundaryData()), which populates all structural profile fields with valid numbers,
# then zero the osmotic MAGNITUDES so the ONLY osmotic contribution is the dynamic
# sucrose term from the transport solve.
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
print(f"  n_wall_junction={nwj}  n_cells={n_cells}")

# Xylem supply-side Dirichlet BC; no phloem BC.
mecha.psi_xyl[1, I_MAT, I_SCE]   = PSI_XYL
mecha.psi_sieve[1, I_MAT, I_SCE] = np.nan

# Resolve the air-space water potential from the chosen mode.
if AIR_MODE == 'rh':
    psi_air = rh_to_water_potential(RH_AIR, T_KELVIN)
    print(f"  Air BC: RH={RH_AIR:.3f} → Ψ_air={psi_air:.1f} hPa (Kelvin, T={T_KELVIN:.2f} K)")
elif AIR_MODE == 'psi':
    psi_air = PSI_ATM
    print(f"  Air BC: Ψ_air={psi_air:.1f} hPa (water potential)")
else:
    raise ValueError(f"AIR_MODE must be 'psi' or 'rh', got {AIR_MODE!r}.")


# ── 3. Identify tissue cell indices ──────────────────────────────────────────
#
# Tissue lists drive the solute Dirichlet BCs (mesophyll source, phloem-loading
# sink, xylem) and the Δc_load metric.  The substomatal air spaces are now found
# directly from the GRANAP protect_topology flag via network.is_wall_air_cell()
# (they carry wall_air edges), replacing the old guard-cell adjacency heuristic.

print("\n=== 3. Identifying tissue cell indices ===")

meso_cell_ids              = []
endo_cell_ids              = []
transfusion_parenchyma_ids = []
transfusion_tracheid_ids   = []
strasburger_cell_ids       = []
phloem_cell_ids            = []
xylem_cell_ids             = []
airspace_cell_ids          = []

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
    elif ct == 'transfusion parenchyma':
        transfusion_parenchyma_ids.append(cell_id)
    elif ct == 'transfusion tracheid':
        transfusion_tracheid_ids.append(cell_id)
    elif cg == 3:
        endo_cell_ids.append(cell_id)
    elif cg == 5:
        transfusion_parenchyma_ids.append(cell_id)
    elif cg == 12 and ct == 'Strasburger cell':
        strasburger_cell_ids.append(cell_id)
    elif cg == 11:
        phloem_cell_ids.append(cell_id)
    elif cg == 13:
        xylem_cell_ids.append(cell_id)

# Fallback: cgroup-4 cells as mesophyll when no explicit label found.
if not meso_cell_ids:
    for nd, d in mecha.network.graph.nodes(data=True):
        idx = mecha.indice[nd]
        if idx >= nwj and int(d.get('cgroup', -1)) == 4 and not mecha.network.is_wall_air_cell(nd):
            meso_cell_ids.append(idx - nwj)

phloem_loading_ids   = strasburger_cell_ids + phloem_cell_ids

# Impose the air/transpiration Dirichlet BC on the EVAPORATING WALL NODES — the
# wall side of every wall_air edge — via Mecha.set_air_wall_bc.  These are the
# transpirationally active surfaces in the GRANAP needle (the air-space cells
# themselves own no ordinary walls, so the classic hormones.contact border-wall
# route never reaches them).  A large-conductance penalty pins them to psi_air,
# exactly as the calibration forward model does.  Inert for root anatomies.
evaporating_wall_nodes = mecha.set_air_wall_bc(psi_air)

print(f"  Mesophyll cells                    : {len(meso_cell_ids)}")
print(f"  Endodermis cells       (cgroup  3) : {len(endo_cell_ids)}")
print(f"  Transfusion parenchyma (living)    : {len(transfusion_parenchyma_ids)}")
print(f"  Transfusion tracheids  (dead)      : {len(transfusion_tracheid_ids)}")
print(f"  Strasburger cells      (cgroup 12) : {len(strasburger_cell_ids)}")
print(f"  Phloem sieve cells     (cgroup 11) : {len(phloem_cell_ids)}")
print(f"  Xylem cells            (cgroup 13) : {len(xylem_cell_ids)}")
print(f"  Air spaces (protect_topology)      : {len(airspace_cell_ids)}")
print(f"  Evaporating wall nodes (air BC)    : {len(evaporating_wall_nodes)}")


# ── 4. Preliminary hydraulic solve (osmosis OFF — baseline BC map) ───────────

print(f"\n=== 4. Preliminary hydraulic solve  PSI_XYL={PSI_XYL:.0f}  Ψ_air={psi_air:.0f} hPa ===")

mecha.water_flux(h=H_IDX, verbose=False)
print("  Baseline (osmosis OFF) hydraulic solve done.")

# BC map: xylem cells fixed at PSI_XYL (source); air-space cells shown at psi_air
# since they equilibrate to the evaporating-wall BC imposed on their wall_air walls.
psi_bc_arr = np.full(nwj + n_cells, np.nan)
for cid in xylem_cell_ids:
    psi_bc_arr[nwj + cid] = PSI_XYL
for cid in airspace_cell_ids:
    psi_bc_arr[nwj + cid] = psi_air

_plot_concentration_polygon(
    mecha, psi_bc_arr,
    label='Ψ (hPa, BC only — gray = free nodes)',
    title=('Initial water potential — Dirichlet BCs\n'
           f'xylem {PSI_XYL:.0f} hPa  |  evaporating walls {psi_air:.0f} hPa  |  gray: free'),
    save_path=os.path.join(_OUT_DIR, 'needle_steady_initial_waterpotential.png'),
    cmap='RdYlBu', vmin=min(psi_air, PSI_XYL), vmax=max(psi_air, PSI_XYL),
)


# ── 6. SoluteTransport operator ───────────────────────────────────────────────

print("\n=== 6. Building SoluteTransport operator (full, steady-state) ===")

DP_FULL = dict(apo_wall=D_APO, membrane=D_MEM, plasmodesmata=D_PD,
               sigma={cg: SIGMA_SUCROSE for cg in range(1, 20)})
_cap = {'dt': COUPLE_DT} if COUPLE_TIMESTEP else None
st = SoluteTransport(mecha, DP_FULL, _cap, mode='full')

D_mat = st.build_diffusion_matrix(H_IDX, I_MAT)
D_diag_max = float(np.abs(D_mat.diagonal()).max())
N_TOTAL = nwj + n_cells
print(f"  Diffusion matrix shape: {D_mat.shape}  |D_diag|_max = {D_diag_max:.3e}")

# Anchor disconnected components so the transport system is well-posed.
D_row_norms       = np.array(np.abs(D_mat).sum(axis=1)).ravel()
isolated_node_ids = list(np.where(D_row_norms == 0)[0])
n_comp, comp_labels = csgraph.connected_components(D_mat, directed=False, connection='weak')
print(f"  Zero-row nodes: {len(isolated_node_ids)}   connected components: {n_comp}")

anchored_full_ids = set([nwj + cid for cid in phloem_loading_ids]
                        + [nwj + cid for cid in meso_cell_ids]
                        + [nwj + cid for cid in xylem_cell_ids])
isolated_set = set(isolated_node_ids)

extra_anchor_full_ids = []
for comp_id in range(n_comp):
    members = np.where(comp_labels == comp_id)[0]
    if any(m in anchored_full_ids for m in members):
        continue
    non_iso = [m for m in members if m not in isolated_set]
    if non_iso:
        extra_anchor_full_ids.append(non_iso[0])
print(f"  Extra Dirichlet anchors needed: {len(extra_anchor_full_ids)}")


# ── 7. Coupled water–solute solve ─────────────────────────────────────────────
#
# Dirichlet BCs: c_meso = C_MESO, c_phloem = C_PHLOEM = 0, c_xylem = 0.
# On each outer iteration the transport solve yields c, the cell osmotic potentials
# are set from c via van't Hoff (Ψ_os = −R·T·c), the hydraulics are re-solved with
# those Ψ_os (Kedem–Katchalsky), and the new water flow feeds back into advection.

print(f"\n=== 7. Coupled transport solve ===")
print(f"  c_meso={C_MESO*1e6:.1f} µM ({len(meso_cell_ids)} cells)  "
      f"c_phloem={C_PHLOEM*1e6:.1f} µM ({len(phloem_loading_ids)} cells)")

bc = {}
for nid in isolated_node_ids:
    bc[nid] = 0.0
for nid in extra_anchor_full_ids:
    bc[nid] = 0.0
for cid in phloem_loading_ids:
    bc[nwj + cid] = C_PHLOEM
for cid in meso_cell_ids:
    bc[nwj + cid] = C_MESO
for cid in xylem_cell_ids:
    bc[nwj + cid] = 0.0

rhs = np.zeros(N_TOTAL)


def _advection_equiv(scheme):
    """Advection-equivalent matrix (T_sg − D for SG so D + A reconstructs T_sg)."""
    if scheme == 'sg':
        return st.build_transport_operator(H_IDX, I_MAT, I_SCE, scheme='sg') - D_mat
    return st.build_advection_matrix(I_MAT, I_SCE)


def _solve_transport_and_loading(A_operator, include_advection_in_solve):
    """Solve steady transport for a given advection matrix; return loading metrics."""
    import scipy.sparse.linalg as _spla
    T_solve = (D_mat + A_operator) if include_advection_in_solve else D_mat
    EPS_ = 1e-10 * D_diag_max
    T_ = (T_solve + EPS_ * sp.eye(N_TOTAL, format='csr')).tolil()
    rhs_ = np.zeros(N_TOTAL)
    for node_id, c_val in bc.items():
        if 0 <= node_id < N_TOTAL:
            T_[node_id, :] = 0.0
            T_[node_id, node_id] = 1.0
            rhs_[node_id] = float(c_val)
    c_ = _spla.spsolve(T_.tocsr(), rhs_)
    idx = nwj + np.array(phloem_loading_ids) if phloem_loading_ids else None
    Q_diff = float(np.sum(D_mat.dot(c_)[idx]))      if idx is not None else np.nan
    Q_adv  = float(np.sum(A_operator.dot(c_)[idx])) if idx is not None else np.nan
    c_tr = (float(np.mean(c_[nwj + np.array(transfusion_parenchyma_ids)]))
            if transfusion_parenchyma_ids else np.nan)
    c_st = (float(np.mean(c_[nwj + np.array(strasburger_cell_ids)]))
            if strasburger_cell_ids else np.nan)
    dc_ = c_tr - (c_st if not np.isnan(c_st) else C_PHLOEM)
    return c_, Q_diff, Q_adv, dc_


# Osmosis-OFF loading baseline (captured while psi_os = 0).  The Scharfetter–Gummel
# operator is non-oscillatory at any mesh Peclet, so no Pe diagnostic is needed.
A_mat0_sig = _advection_equiv(COUPLE_SCHEME)
_use_adv = (COUPLE_OPS != 'D')
(c_baseline, Qdiff_baseline, Qadv_baseline,
 dc_load_baseline) = _solve_transport_and_loading(A_mat0_sig, _use_adv)
Q_load_baseline = Qdiff_baseline + Qadv_baseline

print(f"  Ψ_os(c_meso) = {-8.314e4 * T_KELVIN * C_MESO:.1f} hPa  "
      f"(operators='{COUPLE_OPS}', scheme='{COUPLE_SCHEME}', method={COUPLE_METHOD})")

c_sol, n_couple_iter, couple_converged = coupled_water_solute_solve(
    mecha, st,
    T=T_KELVIN,
    boundary_conditions=bc,
    rhs=rhs,
    i_scenario=I_SCE,
    i_maturity=I_MAT,
    h=H_IDX,
    tol=COUPLE_TOL,
    max_iter=COUPLE_MAXITER,
    operators=COUPLE_OPS,
    relaxation=COUPLE_RELAX,
    component_tol=COUPLE_TOL,
    scheme=COUPLE_SCHEME,
    time_stepping=COUPLE_TIMESTEP,
    theta=COUPLE_THETA,
    method=COUPLE_METHOD,
    jfnk_f_tol=COUPLE_JFNK_FTOL,
    jfnk_maxiter=COUPLE_JFNK_MAXITER,
    jfnk_inner_maxiter=COUPLE_JFNK_INNER,
    jfnk_line_search=COUPLE_JFNK_LS,
    jfnk_rdiff=COUPLE_JFNK_RDIFF,
    continuation_steps=COUPLE_CONTINUATION,
    verbose=True,
)
print(f"  Coupled solve: {n_couple_iter} iteration(s), converged={couple_converged}")

# Refresh matrices from the converged coupled state.
D_mat  = st.build_diffusion_matrix(H_IDX, I_MAT)
A_mat  = _advection_equiv(COUPLE_SCHEME)

if np.any(~np.isfinite(c_sol)):
    print("  WARNING: transport solve returned NaN/Inf — system may be singular")
else:
    print(f"  Solve OK.  Cell c range: [{c_sol[nwj:].min()*1e6:.2f}, {c_sol[nwj:].max()*1e6:.2f}] µM"
          f"   Wall c range: [{c_sol[:nwj].min()*1e6:.2f}, {c_sol[:nwj].max()*1e6:.2f}] µM")

# Verify Ψ_os = −R·T·c on non-BC cells.
manager   = mecha.network.cell_manager
_expected = -8.314e4 * T_KELVIN * c_sol[nwj: nwj + n_cells]
_stored   = np.array([c.psi_os if c.psi_os is not None else np.nan for c in manager])
_finite   = np.isfinite(_stored) & np.isfinite(_expected)
_max_err  = float(np.max(np.abs(_stored[_finite] - _expected[_finite]))) if np.any(_finite) else float('nan')
_psi_p    = np.array([c.psi_p     if c.psi_p     is not None else np.nan for c in manager])
_psi_t    = np.array([c.psi_total if c.psi_total is not None else np.nan for c in manager])
print(f"  Ψ_os check: max|Ψ_os − (−R·T·c)| = {_max_err:.3e} hPa  "
      f"range [{np.nanmin(_stored):.1f}, {np.nanmax(_stored):.1f}] hPa")


# ── 7b. Initial concentration field (BC map) ──────────────────────────────────

print("\n=== 7b. Plotting initial concentration field (BC state) ===")

c_bc_arr = np.full(N_TOTAL, np.nan)
for cid in meso_cell_ids:
    c_bc_arr[nwj + cid] = C_MESO * 1e6
for cid in phloem_loading_ids:
    c_bc_arr[nwj + cid] = C_PHLOEM * 1e6
for cid in xylem_cell_ids:
    c_bc_arr[nwj + cid] = 0.0

_plot_concentration_polygon(
    mecha, c_bc_arr,
    label='c (µM sucrose, BC only — gray = free nodes)',
    title=('Initial concentration field — Dirichlet BCs\n'
           f'mesophyll {C_MESO*1e6:.0f} µM  |  phloem 0 µM  |  gray: free'),
    save_path=os.path.join(_OUT_DIR, 'needle_steady_initial_concentration.png'),
    cmap='plasma', vmin=0.0, vmax=C_MESO * 1e6,
)


# ── 8. Output variables ───────────────────────────────────────────────────────

print("\n=== 8. Computing output variables ===")

T_orig = D_mat + A_mat
Tc     = T_orig.dot(c_sol)
Q_load = float(np.sum(Tc[nwj + np.array(phloem_loading_ids)])) if phloem_loading_ids else np.nan

c_meso_mean   = float(np.mean(c_sol[nwj + np.array(meso_cell_ids)]))              if meso_cell_ids              else np.nan
c_endo_mean   = float(np.mean(c_sol[nwj + np.array(endo_cell_ids)]))              if endo_cell_ids              else np.nan
c_transf_mean = float(np.mean(c_sol[nwj + np.array(transfusion_parenchyma_ids)])) if transfusion_parenchyma_ids else np.nan
c_strasb      = float(np.mean(c_sol[nwj + np.array(strasburger_cell_ids)]))       if strasburger_cell_ids       else np.nan

# Δc_load: driving gradient across the last step into the phloem loading complex.
dc_load = c_transf_mean - (c_strasb if not np.isnan(c_strasb) else C_PHLOEM)

print(f"  c_meso={c_meso_mean*1e6:.2f} µM  c_endo={c_endo_mean*1e6:.2f} µM  "
      f"c_transf={c_transf_mean*1e6:.2f} µM  c_strasb={c_strasb*1e6:.2f} µM")
print(f"  Δc_load (transfusion → Strasburger) = {dc_load*1e6:.2f} µM")
print(f"  Q_load (full operator)              = {Q_load*1e12:.4f} pmol/d")


# ── 8b. Significance of the osmotic term on sugar loading ─────────────────────

print("\n=== 8b. Significance of the osmotic term on sugar loading (ON vs OFF) ===")

(c_osmON, Qdiff_on, Qadv_on, dc_load_on) = _solve_transport_and_loading(A_mat, _use_adv)
Q_load_on = Qdiff_on + Qadv_on


def _rel_pct(new, ref):
    return (new - ref) / abs(ref) * 100.0 if (ref not in (0, None) and abs(ref) > 0) else float('nan')


dQ_abs, dQ_pct   = Q_load_on - Q_load_baseline,   _rel_pct(Q_load_on, Q_load_baseline)
ddc_abs, ddc_pct = dc_load_on - dc_load_baseline, _rel_pct(dc_load_on, dc_load_baseline)

print(f"  {'':24s}{'OFF':>15s}{'ON':>15s}{'Δ (ON−OFF)':>15s}{'rel.':>9s}")
print(f"  {'Q_load  total [pmol/d]':24s}{Q_load_baseline*1e12:15.4f}{Q_load_on*1e12:15.4f}"
      f"{dQ_abs*1e12:15.4f}{dQ_pct:8.1f}%")
print(f"  {'Δc_load [µM]':24s}{dc_load_baseline*1e6:15.4f}{dc_load_on*1e6:15.4f}"
      f"{ddc_abs*1e6:15.4f}{ddc_pct:8.1f}%")

adv_share = (abs(Qadv_on) / (abs(Qdiff_on) + abs(Qadv_on)) * 100.0
             if (abs(Qdiff_on) + abs(Qadv_on)) > 0 else float('nan'))
_sig = "SIGNIFICANT" if (np.isfinite(dQ_pct) and abs(dQ_pct) >= 5.0) else "minor (<5%)"
print(f"  → Osmotic-term effect on sugar loading: {_sig} "
      f"({dQ_pct:+.1f}% on Q_load, {ddc_pct:+.1f}% on Δc_load; advective share {adv_share:.1f}%).")


# ── 9. Concentration field plot ───────────────────────────────────────────────

print("\n=== 9. Plotting concentration field ===")

_plot_concentration_polygon(
    mecha, c_sol * 1e6,
    label='c (µM sucrose)',
    title=(f'Steady-state sucrose field  (barrier=1, Casparian strip)\n'
           f'c_meso = {C_MESO*1e6:.0f} µM  |  ΔΨ = {PSI_XYL - psi_air:.0f} hPa  |  '
           f'Δc_load = {dc_load*1e6:.1f} µM'),
    save_path=os.path.join(_OUT_DIR, 'needle_steady_concentration.png'),
    cmap='plasma', vmin=0.0, vmax=C_MESO * 1e6,
)


# ── 9b. Concentration gradient magnitude plot ─────────────────────────────────

print("\n=== 9b. Plotting concentration gradient magnitude ===")

_gdf_xy      = mecha.network._cells_gdf.copy()
_centroid_of = {}
for _, _row in _gdf_xy.iterrows():
    _centroid_of[nwj + int(_row['id_cell'])] = (_row.geometry.centroid.x, _row.geometry.centroid.y)

_cell_adj = defaultdict(set)
for _wnd in mecha.network.graph.nodes():
    if mecha.indice.get(_wnd, nwj) >= nwj:
        continue
    _cnbs = [_n for _n in mecha.network.graph.neighbors(_wnd) if _n in _centroid_of]
    for _ii, _a in enumerate(_cnbs):
        for _b in _cnbs[_ii + 1:]:
            _cell_adj[_a].add(_b)
            _cell_adj[_b].add(_a)
for _u, _v in mecha.network.graph.edges():
    if _u in _centroid_of and _v in _centroid_of:
        _cell_adj[_u].add(_v)
        _cell_adj[_v].add(_u)

_grad_arr = np.full(N_TOTAL, np.nan)
for _nd, (_xi, _yi) in _centroid_of.items():
    _idx_i = mecha.indice.get(_nd)
    if _idx_i is None or np.isnan(c_sol[_idx_i]):
        continue
    _ci = float(c_sol[_idx_i])
    _dr, _dc = [], []
    for _nb in _cell_adj[_nd]:
        _idx_j = mecha.indice.get(_nb)
        if _idx_j is None or np.isnan(c_sol[_idx_j]):
            continue
        _xj, _yj = _centroid_of[_nb]
        _dr.append([_xj - _xi, _yj - _yi])
        _dc.append(float(c_sol[_idx_j]) - _ci)
    if len(_dr) >= 2:
        _g, *_ = np.linalg.lstsq(np.array(_dr), np.array(_dc), rcond=None)
        _grad_arr[_idx_i] = float(np.linalg.norm(_g))
    elif len(_dr) == 1:
        _d = np.linalg.norm(_dr[0])
        _grad_arr[_idx_i] = abs(_dc[0]) / _d if _d > 0 else 0.0
    else:
        _grad_arr[_idx_i] = 0.0

print(f"  max |∇c| = {float(np.nanmax(_grad_arr)) * 1e6:.4f} µM/µm")

_plot_concentration_polygon(
    mecha, _grad_arr * 1e6,
    label='|∇c| (µM µm⁻¹)',
    title=('Concentration gradient magnitude  |∇c|\n'
           f'c_meso = {C_MESO*1e6:.0f} µM  |  barrier=1 (Casparian strip)'),
    save_path=os.path.join(_OUT_DIR, 'needle_steady_concentration_gradient.png'),
    cmap='hot_r', vmin=0.0,
)


# ── 10. Velocity network plot ─────────────────────────────────────────────────

print("\n=== 10. Plotting velocity network ===")
visualize(mecha, visu_type='velocity', maturity_idx=I_MAT,
          save_path=os.path.join(_OUT_DIR, 'needle_steady_waterflow.png'))


# ── 11. Solute flux network plot ──────────────────────────────────────────────

print("\n=== 11. Plotting solute flux network ===")
T_full = (D_mat + A_mat).tocsr()
for u, v, eattr in mecha.network.graph.edges(data=True):
    iu, iv = mecha.indice[u], mecha.indice[v]
    K_sol = abs(float(T_full[iv, iu]))
    J = K_sol * (c_sol[iu] - c_sol[iv]) if K_sol > 0 else 0.0
    eattr['solute_flux'] = abs(J)
    eattr['Q'] = J

_plot_edge_vector_property(
    mecha, prop_name='solute_flux', unit='mol d⁻¹', maturity_idx=I_MAT,
    save_path=os.path.join(_OUT_DIR, 'needle_steady_solute_velocity.png'))


# ── 12. Write results summary ─────────────────────────────────────────────────

results_path = os.path.join(_OUT_DIR, 'needle_steady_results.txt')
with open(results_path, 'w') as f:
    f.write("Conifer needle — symplastic sucrose transport with Casparian strip\n")
    f.write("=" * 70 + "\n\n")

    f.write("Geometry: GRANAP NeedleAnatomy, physics from InData.needle_defaults()\n")
    f.write("  barrier=1 (Casparian strip at endodermis radial walls)\n\n")

    f.write("Cell counts:\n")
    f.write(f"  Mesophyll                      : {len(meso_cell_ids)}\n")
    f.write(f"  Endodermis (cgroup 3)          : {len(endo_cell_ids)}\n")
    f.write(f"  Transfusion parenchyma (living): {len(transfusion_parenchyma_ids)}\n")
    f.write(f"  Transfusion tracheids  (dead)  : {len(transfusion_tracheid_ids)}\n")
    f.write(f"  Strasburger cells (cgroup 12)  : {len(strasburger_cell_ids)}\n")
    f.write(f"  Phloem sieve cells (cgroup 11) : {len(phloem_cell_ids)}\n")
    f.write(f"  Xylem (cgroup 13)              : {len(xylem_cell_ids)}\n\n")

    f.write("Boundary conditions:\n")
    f.write(f"  PSI_XYL  = {PSI_XYL:.1f} hPa  (Dirichlet at xylem cells)\n")
    if AIR_MODE == 'rh':
        f.write(f"  Air BC   = RH {RH_AIR:.3f} → Ψ_air = {psi_air:.1f} hPa (Kelvin, T={T_KELVIN:.2f} K)\n")
    else:
        f.write(f"  Air BC   = Ψ_air {psi_air:.1f} hPa (water potential)\n")
    f.write(f"  C_MESO   = {C_MESO*1e6:.1f} µM   (mesophyll sucrose, Dirichlet)\n")
    f.write(f"  C_PHLOEM = {C_PHLOEM*1e6:.1f} µM   (phloem loading complex, reference sink)\n\n")

    f.write("Transport parameters (full operator):\n")
    f.write(f"  D_PD      = {D_PD:.2e} cm²/d\n")
    f.write(f"  D_APO     = {D_APO:.2e} cm²/d\n")
    f.write(f"  D_MEM     = {D_MEM:.2e} cm²/d\n")
    f.write(f"  σ_sucrose = {SIGMA_SUCROSE}\n\n")

    f.write("Water–solute coupling:\n")
    f.write(f"  scenario={I_SCE} (osmotic ON; s_factor=σ)  T={T_KELVIN:.2f} K → Ψ_os=−R·T·c\n")
    f.write(f"  operators='{COUPLE_OPS}'  scheme='{COUPLE_SCHEME}'  method={COUPLE_METHOD}  tol={COUPLE_TOL} hPa\n")
    f.write(f"  iterations={n_couple_iter}  converged={couple_converged}\n")
    f.write(f"  max|Ψ_os − (−R·T·c)| = {_max_err:.3e} hPa\n")
    f.write(f"  Ψ_os range = [{np.nanmin(_stored):.1f}, {np.nanmax(_stored):.1f}] hPa\n")
    f.write(f"  Ψ_p  range = [{np.nanmin(_psi_p):.1f}, {np.nanmax(_psi_p):.1f}] hPa\n")
    f.write(f"  Ψ_total range = [{np.nanmin(_psi_t):.1f}, {np.nanmax(_psi_t):.1f}] hPa\n\n")

    f.write("Concentration profile (tissue means):\n")
    f.write(f"  c_meso        = {c_meso_mean*1e6:10.3f} µM   (imposed: {C_MESO*1e6:.1f} µM)\n")
    f.write(f"  c_endodermis  = {c_endo_mean*1e6:10.3f} µM\n")
    f.write(f"  c_transfusion = {c_transf_mean*1e6:10.3f} µM\n")
    f.write(f"  c_strasburger = {c_strasb*1e6:10.3f} µM\n\n")

    f.write("Output variables:\n")
    f.write(f"  Q_load (full operator)            = {Q_load*1e12:.4f} pmol/d\n")
    f.write(f"  Δc_load (transfusion→Strasburger) = {dc_load*1e6:.2f} µM\n\n")

    f.write("Significance of the osmotic term (osmosis ON vs OFF):\n")
    f.write(f"  {'':24s}{'OFF':>13s}{'ON':>13s}{'Δ':>13s}{'rel.':>9s}\n")
    f.write(f"  {'Q_load total [pmol/d]':24s}{Q_load_baseline*1e12:13.4f}{Q_load_on*1e12:13.4f}"
            f"{dQ_abs*1e12:13.4f}{dQ_pct:8.1f}%\n")
    f.write(f"  {'Δc_load [µM]':24s}{dc_load_baseline*1e6:13.4f}{dc_load_on*1e6:13.4f}"
            f"{ddc_abs*1e6:13.4f}{ddc_pct:8.1f}%\n")
    f.write(f"  → Osmotic term is {_sig} for sugar loading "
            f"({dQ_pct:+.1f}% on Q_load, {ddc_pct:+.1f}% on Δc_load).\n")

print(f"\n  Results written to: {results_path}")
print("\n=== DONE ===")
print(f"Outputs in: {_OUT_DIR}")
