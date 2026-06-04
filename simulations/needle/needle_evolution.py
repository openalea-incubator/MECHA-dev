"""
needle_evolution.py
===================
Diurnal-cycle dynamic simulation of water-potential and sucrose transport
in a conifer needle cross-section.

SETUP
-----
Geometry : same as needle_steady.py (GRANAP NeedleAnatomy, barrier=1).

Forcing   : sinusoidal diurnal cycle with t=0 at dawn.
              PSI_XYL(t) = PSI_MEAN − PSI_AMP · sin(2π t / T_DAY)
                           (peak at dawn, trough at midday)
              C_MESO(t)  = C_MEAN  + C_AMP  · sin(2π t / T_DAY)
                           (trough at dawn, peak at midday — photosynthesis)

Water     : quasi-steady hydraulic solve at each timestep (MECHA Doussan).
            Capacitance is not included for the water sub-problem.

Solute    : backward-Euler implicit time integration with per-cell volumetric
            capacitance V_i [cm³].  For mesophyll and transfusion parenchyma
            the capacitance is scaled by a user-adjustable multiplier
            (C_CAP_MESO, C_CAP_TRANSF) to represent different symplastic
            buffering fractions.

            Equation for each free node i at step n+1:
              (V_i / Δt) · (c_i^{n+1} − c_i^n)
              + Σ_j T_ij · c_j^{n+1} = 0
            → (T + diag(V/Δt)) · c^{n+1} = diag(V/Δt) · c^n  + rhs_BC

OUTPUTS
-------
  ~/simulations/needle/outputs/
    needle_evo_bc_waterpotential.png   – initial-cycle water-potential Dirichlet BC map
    needle_evo_bc_concentration.png   – initial-cycle concentration Dirichlet BC map
    needle_evo_timeseries.png          – Q_load, c_transf, forcing over time
    snapshots/
      needle_evo_psi_{nn}.png          – water-potential choropleth snapshots
      needle_evo_c_{nn}.png            – concentration choropleth snapshots
      needle_evo_gradc_{nn}.png        – |∇c| magnitude snapshots (µM/µm)
    needle_evo_concentration_grid.png  – all concentration snapshots in one figure
    needle_evo_results.txt             – run summary

UNITS
-----
  Lengths µm / cm   Volumes cm³   Fluxes cm³/d
  Conc mol/cm³      Rates mol/d   Time d
"""

import os
import sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse import csgraph
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR    = os.path.join(_SCRIPT_DIR, 'outputs')
_SNAP_DIR   = os.path.join(_OUT_DIR, 'snapshots')
os.makedirs(_SNAP_DIR, exist_ok=True)

_MECHA_SRC  = os.path.join(_SCRIPT_DIR, '..', '..', 'MECHA-dev', 'src')
_GRANAP_SRC = os.path.join(_SCRIPT_DIR, '..', '..', 'GRANAP-dev', 'src')
for _p in (_MECHA_SRC, _GRANAP_SRC):
    sys.path.insert(0, os.path.abspath(_p))

from granap.needle_class import NeedleAnatomy
from mecha.mecha_class import Mecha
from mecha.utils.data_loader import InData
from mecha.utils.network_builder import NetworkBuilder
from mecha.solute_transport import SoluteTransport


# ── Parameters ─────────────────────────────────────────────────────────────────

# ---- Diurnal forcing ---------------------------------------------------------
# t = 0 is dawn.  PSI_XYL troughs at midday; C_MESO peaks at midday.
T_DAY        = 1.0      # d  period of diurnal oscillation
PSI_MEAN     = 1000.0   # hPa  mean xylem water potential
PSI_AMP      =  400.0   # hPa  amplitude of diurnal ψ swing
PSI_ATM      =    0.0   # hPa  atmospheric reference (stomata sink, constant)
C_MEAN       =  50e-6   # mol/cm³  mean mesophyll sucrose (~50 mM)
C_AMP        =  20e-6   # mol/cm³  amplitude (~±20 mM diurnal swing)
C_PHLOEM     =    0.0   # mol/cm³  phloem loading sink (reference)

# ---- Capacitance (solute, per cell) ------------------------------------------
# Dimensionless multiplier on cell volume V_cell [cm³].
# 1.0 = full cell volume participates in buffering; decrease for tight cytoplasm,
# increase to represent vacuolar loading/release contributions.
C_CAP_MESO   = 1.0   # mesophyll
C_CAP_TRANSF = 2.0   # transfusion parenchyma (larger vacuole fraction assumed)

# ---- Transport diffusivities -------------------------------------------------
D_PD          = 1e-4   # cm²/d  plasmodesmatal
D_APO         = 0.1    # cm²/d  apoplastic wall
D_MEM         = 1e-6   # cm²/d  passive transmembrane
SIGMA_SUCROSE = 0.9   # Staverman reflection coefficient

# ---- Time integration --------------------------------------------------------
T_SIM       = 3.0    # d   total simulation (3 diurnal cycles; spin-up + output)
DT          = 1/48   # d   timestep (30 min)
N_SNAP      = 8      # snapshots per diurnal cycle (every 3 h in the last cycle)

# ---- Indices -----------------------------------------------------------------
H_IDX = 0
I_MAT = 0
I_SCE = 0
X_CONTACT = 1e10   # µm  (disables position-based outer BC)

CGROUP_NAMES = {
    1: 'epidermis', 2: 'cortex', 3: 'endodermis',
    4: 'mesophyll / airspace', 5: 'transfusion parenchyma',
    9: 'hypodermis', 10: 'proto-phloem', 11: 'phloem sieve',
    12: 'Strasburger cell', 13: 'xylem', 14: 'cambium',
    15: 'parenchyma', 16: 'airspace', 19: 'meta-xylem', 20: 'proto-xylem',
}


# ── Helper: polygon choropleth ─────────────────────────────────────────────────

def _plot_polygon(mecha_obj, sol_arr, label, title, save_path,
                  cmap='plasma', vmin=0.0, vmax=None, ax=None):
    """Choropleth of sol_arr (indexed by full network) on the cell GDF.

    If *ax* is given the figure is drawn into it (no save/close); otherwise a
    new figure is created and saved to *save_path*.
    """
    gdf     = mecha_obj.network._cells_gdf.copy()
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
        vmax = float(np.nanpercentile(finite, 99)) if len(finite) else 1.0

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(9, 7))

    gdf.plot(ax=ax, column='value', cmap=cmap, edgecolor='black', linewidth=0.3,
             vmin=vmin, vmax=vmax, legend=True,
             legend_kwds={'label': label, 'orientation': 'vertical'},
             missing_kwds={'color': 'lightgray'})
    ax.set_aspect('equal', 'box')
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('x (µm)', fontsize=8)
    ax.set_ylabel('y (µm)', fontsize=8)

    if own_fig:
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved → {save_path}")


# ── 1. Build anatomy + MECHA network ─────────────

print("=== Building anatomy and MECHA network ===")
needle = NeedleAnatomy()
needle.update_params("stomata",               "n_files",             8)
needle.update_params("resin_duct",            "n_files",             2)
needle.update_params("inter_cellular_spaces", "n_files",             8)
needle.update_params("transfusion_tissue",    "transfusion_type",   True)
needle.update_params("central_cylinder",      "shape",     "half_ellipse")
_ = needle.export_to_adjencymatrix()
cells_gdf = needle._cells_gdf

network = NetworkBuilder(needle)
network.populate_from_network()

default_input = InData()
default_input.geometry.set_maturity_stages([1], [200.0])
default_input.hydraulic.xcontactrange = [X_CONTACT]

mecha   = Mecha(default_input, network=network)
nwj     = mecha.network.n_wall_junction
n_cells = mecha.network.n_cells
N_TOTAL = nwj + n_cells

h_cm = float(mecha.geometry.maturity_stages[I_MAT].get('height')) * 1e-4

mecha.psi_xyl[1, I_MAT, I_SCE]   = PSI_MEAN   # updated at each step
mecha.psi_sieve[1, I_MAT, I_SCE] = np.nan

print(f"  n_wall_junction={nwj}  n_cells={n_cells}  N_TOTAL={N_TOTAL}")


# ── 2. Identify tissue cell indices  ─────────────────────────────────

print("=== Identifying tissue cells ===")

cgroup_of_cell             = {}
meso_cell_ids              = []
endo_cell_ids              = []
transfusion_parenchyma_ids = []
transfusion_tracheid_ids   = []
strasburger_cell_ids       = []
phloem_cell_ids            = []
xylem_cell_ids             = []
outer_cell_ids             = []
airspace_cell_ids          = []

for nd, d in mecha.network.graph.nodes(data=True):
    idx = mecha.indice[nd]
    if idx < nwj:
        continue
    cell_id = idx - nwj
    cg = int(d.get('cgroup', -1))
    ct = str(d.get('cell_type', ''))
    cgroup_of_cell[cell_id] = cg

    if   cg == 4 and ct == 'mesophyll':        meso_cell_ids.append(cell_id)
    elif ct == 'transfusion parenchyma':        transfusion_parenchyma_ids.append(cell_id)
    elif ct == 'transfusion tracheid':          transfusion_tracheid_ids.append(cell_id)
    elif cg == 3:                               endo_cell_ids.append(cell_id)
    elif cg == 5:                               transfusion_parenchyma_ids.append(cell_id)
    elif cg == 12 and ct == 'Strasburger cell': strasburger_cell_ids.append(cell_id)
    elif cg == 11:                              phloem_cell_ids.append(cell_id)
    elif cg == 13:                              xylem_cell_ids.append(cell_id)
    elif cg in (1, 9):                          outer_cell_ids.append(cell_id)
    elif cg == 16 or 'air' in ct.lower():       airspace_cell_ids.append(cell_id)
    elif cg == 4:                               airspace_cell_ids.append(cell_id)

if not meso_cell_ids:
    for nd, d in mecha.network.graph.nodes(data=True):
        idx = mecha.indice[nd]
        if idx >= nwj and int(d.get('cgroup', -1)) == 4:
            meso_cell_ids.append(idx - nwj)

transfusion_cell_ids = transfusion_parenchyma_ids + transfusion_tracheid_ids
phloem_loading_ids   = strasburger_cell_ids + phloem_cell_ids

guard_node_ids = {nd for nd, d in mecha.network.graph.nodes(data=True)
                  if mecha.indice[nd] >= nwj and str(d.get('cell_type','')) == 'guard cell'}

stomatal_airspace_ids = []
for nd, d in mecha.network.graph.nodes(data=True):
    idx = mecha.indice[nd]
    if idx < nwj or str(d.get('cell_type','')) not in ('air space', 'pore'):
        continue
    if any(nb in guard_node_ids for nb in mecha.network.graph.neighbors(nd)):
        stomatal_airspace_ids.append(idx - nwj)

mecha.hormones.contact = stomatal_airspace_ids
mecha.boundary.scenarios[I_SCE]['psi_soil_left'] = PSI_ATM

print(f"  mesophyll={len(meso_cell_ids)}  endo={len(endo_cell_ids)}"
      f"  transf_paren={len(transfusion_parenchyma_ids)}"
      f"  transf_trach={len(transfusion_tracheid_ids)}"
      f"  phloem_load={len(phloem_loading_ids)}"
      f"  stomatal_AS={len(stomatal_airspace_ids)}")


# ── 3. Build solute transport operator (once) ──────────────────────────────────

print("=== Building SoluteTransport operator ===")

DP = dict(apo_wall=D_APO, membrane=D_MEM, plasmodesmata=D_PD,
          sigma={cg: SIGMA_SUCROSE for cg in range(1, 20)})
st    = SoluteTransport(mecha, DP, None, mode='full')
D_mat = st.build_diffusion_matrix(H_IDX, I_MAT)

D_row_norms     = np.array(np.abs(D_mat).sum(axis=1)).ravel()
isolated_ids    = list(np.where(D_row_norms == 0)[0])
D_diag_max      = float(np.abs(D_mat.diagonal()).max())
EPS             = 1e-10 * D_diag_max

n_comp, comp_labels = csgraph.connected_components(D_mat, directed=False, connection='weak')
anchored_ids = set(
    [nwj + c for c in phloem_loading_ids]
    + [nwj + c for c in meso_cell_ids]
    + [nwj + c for c in xylem_cell_ids])
isolated_set = set(isolated_ids)

extra_anchor_ids = []
for comp_id in range(n_comp):
    members = np.where(comp_labels == comp_id)[0]
    if any(m in anchored_ids for m in members):
        continue
    non_iso = [m for m in members if m not in isolated_set]
    if non_iso:
        extra_anchor_ids.append(non_iso[0])

print(f"  D_mat {D_mat.shape}  components={n_comp}"
      f"  isolated={len(isolated_ids)}  extra_anchors={len(extra_anchor_ids)}")


# ── 4. Cell volumes and capacitance diagonal ───────────────────────────────────

print("=== Computing cell volumes and capacitance vector ===")

meso_set  = set(meso_cell_ids)
transf_set = set(transfusion_parenchyma_ids)

# Wall nodes: negligible capacitance (walls equilibrate fast; avoids singularity).
# Use mecha.network._cells_gdf so id_cell is guaranteed consistent with mecha.indice.
V_diag = np.full(N_TOTAL, EPS)

for _, row in mecha.network._cells_gdf.iterrows():
    cid      = int(row['id_cell'])
    area_cm2 = max(float(row.geometry.area), 0.0) * 1e-8   # µm² → cm²
    V_cell   = max(area_cm2 * h_cm, EPS)                    # cm³; floor at EPS
    if cid in meso_set:
        V_diag[nwj + cid] = C_CAP_MESO   * V_cell
    elif cid in transf_set:
        V_diag[nwj + cid] = C_CAP_TRANSF * V_cell
    else:
        V_diag[nwj + cid] = V_cell

print(f"  V_cell range (all cells): "
      f"[{V_diag[nwj:].min():.3e}, {V_diag[nwj:].max():.3e}] cm³")

# Estimate characteristic solute time-constant τ = V_meso / K_PD.
# If τ >> T_DAY, diurnal oscillations will be strongly attenuated; reduce
# C_CAP_MESO / C_CAP_TRANSF or increase D_PD to bring τ closer to T_DAY.
_V_meso_mean = float(np.mean(V_diag[nwj + np.array(meso_cell_ids)])) if meso_cell_ids else 0
_K_PD_est    = D_PD * 1e-8 / (1e-3) * 5   # D × A_pd/L × ~5 PDs per interface
_tau_est     = _V_meso_mean / _K_PD_est if _K_PD_est > 0 else float('inf')
print(f"  Estimated solute τ ≈ {_tau_est:.1f} d  (T_DAY = {T_DAY} d)"
      f"{'  ← τ >> T_DAY: oscillations will be attenuated' if _tau_est > 2*T_DAY else ''}")


# ── 5. Diurnal forcing functions ───────────────────────────────────────────────

def psi_xyl_t(t):
    """Xylem water potential [hPa]: peaks at dawn (t=0), troughs at midday."""
    return PSI_MEAN - PSI_AMP * np.sin(2.0 * np.pi * t / T_DAY)


def c_meso_t(t):
    """Mesophyll sucrose [mol/cm³]: trough at dawn, peak at midday."""
    return C_MEAN + C_AMP * np.sin(2.0 * np.pi * t / T_DAY)


# ── 6. Hydraulic and transport helpers ─────────────────────────────────────────

def _solve_hydraulics(psi_val):
    """Set xylem BC, call solve_W, return (sol_W, fluxes, A_mat)."""
    mecha.psi_xyl[1, I_MAT, I_SCE] = psi_val
    sol_W_raw, _, mat_W, _, _ = mecha.solve_W(h=H_IDX, i_maturity=I_MAT)
    sol_W = np.array(sol_W_raw).ravel()

    coo = mat_W.tocoo()
    fluxes = []
    for i, j, v in zip(coo.row, coo.col, coo.data):
        if i < j and v > 0:
            pi, pj = float(sol_W[i]), float(sol_W[j])
            if not (np.isnan(pi) or np.isnan(pj)):
                fluxes.append({'source': int(i), 'target': int(j),
                               'flux': v * (pi - pj)})

    mecha.edge_flux_list[I_MAT][I_SCE] = fluxes
    A_mat = st.build_advection_matrix(I_MAT, I_SCE)
    return sol_W, fluxes, A_mat


def _build_rhs_bc(c_meso_val):
    """Dirichlet BC dict for the current timestep."""
    bc = {}
    for nid in isolated_ids:
        bc[nid] = 0.0
    for nid in extra_anchor_ids:
        bc[nid] = 0.0
    for cid in meso_cell_ids:
        bc[nwj + cid] = c_meso_val
    for cid in phloem_loading_ids:
        bc[nwj + cid] = C_PHLOEM
    for cid in xylem_cell_ids:
        bc[nwj + cid] = 0.0
    return bc


def _apply_bc(T_mat, rhs, bc):
    """In-place Dirichlet substitution on a LIL-converted copy of T_mat."""
    T_lil = T_mat.tolil()
    for node_id, c_val in bc.items():
        if 0 <= node_id < N_TOTAL:
            T_lil[node_id, :]          = 0.0
            T_lil[node_id, node_id]    = 1.0
            rhs[node_id]               = float(c_val)
    return T_lil.tocsr(), rhs


def _compute_grad_mag(c_vec, centroid_of, cell_adj):
    """Estimate |∇c| [mol cm⁻³ µm⁻¹] at each cell via least-squares over neighbours.

    centroid_of : dict  node_id → (x µm, y µm)
    cell_adj    : dict  node_id → set of adjacent cell node_ids
    Returns a full-network array (NaN at wall nodes).
    """
    grad_arr = np.full(N_TOTAL, np.nan)
    for nd, (xi, yi) in centroid_of.items():
        idx_i = mecha.indice.get(nd)
        if idx_i is None:
            continue
        ci = float(c_vec[idx_i])
        if not np.isfinite(ci):
            continue
        dr, dc = [], []
        for nb in cell_adj[nd]:
            idx_j = mecha.indice.get(nb)
            if idx_j is None:
                continue
            cj = float(c_vec[idx_j])
            if not np.isfinite(cj):
                continue
            xj, yj = centroid_of[nb]
            dr.append([xj - xi, yj - yi])
            dc.append(cj - ci)
        if len(dr) >= 2:
            g, *_ = np.linalg.lstsq(np.array(dr), np.array(dc), rcond=None)
            grad_arr[idx_i] = float(np.linalg.norm(g))
        elif len(dr) == 1:
            d = np.linalg.norm(dr[0])
            grad_arr[idx_i] = abs(dc[0]) / d if d > 0 else 0.0
        else:
            grad_arr[idx_i] = 0.0
    return grad_arr


# ── 7. Initial condition (steady-state at dawn forcing) ───────────────────────

print("\n=== Initialising from steady-state at dawn forcing ===")

sol_W0, fluxes0, A_mat0 = _solve_hydraulics(psi_xyl_t(0.0))
T0      = D_mat + A_mat0 + EPS * sp.eye(N_TOTAL, format='csr')
rhs0    = np.zeros(N_TOTAL)
T0_bc, rhs0_bc = _apply_bc(T0, rhs0, _build_rhs_bc(c_meso_t(0.0)))
c_state = spla.spsolve(T0_bc, rhs0_bc)

bad = ~np.isfinite(c_state)
if bad.any():
    print(f"  WARNING: initial solve returned {bad.sum()} non-finite values — zeroing")
    c_state[bad] = 0.0
else:
    print(f"  c_state range: [{c_state[nwj:].min()*1e6:.2f}, "
          f"{c_state[nwj:].max()*1e6:.2f}] µM")

# ── 7b. BC map figures (initial cycle) ────────────────────────────────────────
#
# Water-potential Dirichlet BC map: xylem fixed at PSI_MEAN (dawn), stomatal
# air spaces at PSI_ATM.  All other nodes are free (shown in gray).
print("\n=== Saving initial-cycle BC map figures ===")

psi_bc = np.full(N_TOTAL, np.nan)
for cid in xylem_cell_ids:
    psi_bc[nwj + cid] = psi_xyl_t(0.0)
for cid in stomatal_airspace_ids:
    psi_bc[nwj + cid] = PSI_ATM

_plot_polygon(
    mecha, psi_bc,
    label='Ψ (hPa, BC only — gray = free)',
    title=(f'Water-potential Dirichlet BCs (dawn, t=0)\n'
           f'xylem = {psi_xyl_t(0.0):.0f} hPa  |  stomata = {PSI_ATM:.0f} hPa  |  gray = free'),
    save_path=os.path.join(_OUT_DIR, 'needle_evo_bc_waterpotential.png'),
    cmap='RdYlBu', vmin=PSI_ATM, vmax=PSI_MEAN + PSI_AMP,
)

# Concentration Dirichlet BC map: mesophyll at C_MEAN (dawn value), phloem/xylem at 0.
c_bc = np.full(N_TOTAL, np.nan)
for cid in meso_cell_ids:
    c_bc[nwj + cid] = c_meso_t(0.0) * 1e6
for cid in phloem_loading_ids:
    c_bc[nwj + cid] = C_PHLOEM * 1e6
for cid in xylem_cell_ids:
    c_bc[nwj + cid] = 0.0

_plot_polygon(
    mecha, c_bc,
    label='c (µM, BC only — gray = free)',
    title=(f'Concentration Dirichlet BCs (dawn, t=0)\n'
           f'mesophyll = {c_meso_t(0.0)*1e6:.0f} µM  |  phloem/xylem = 0 µM  |  gray = free'),
    save_path=os.path.join(_OUT_DIR, 'needle_evo_bc_concentration.png'),
    cmap='plasma', vmin=0.0, vmax=(C_MEAN + C_AMP) * 1e6,
)

# ── 7c. Precompute gradient structures (used in snapshot section) ─────────────

_centroid_of = {}
for _, _row in mecha.network._cells_gdf.iterrows():
    _cid = int(_row['id_cell'])
    _centroid_of[nwj + _cid] = (_row.geometry.centroid.x, _row.geometry.centroid.y)

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

print(f"  Gradient structures ready: {len(_centroid_of)} cell centroids")


# ── 8. Time loop ───────────────────────────────────────────────────────────────

N_STEPS = int(round(T_SIM / DT))
t_last_start = T_SIM - T_DAY                 # start of last diurnal cycle
snap_times   = np.linspace(t_last_start, T_SIM, N_SNAP, endpoint=False)
snap_steps   = set(int(round(ts / DT)) for ts in snap_times)

# Time-series records
ts_t       = []
ts_psi_xyl = []
ts_c_meso  = []
ts_Q_load  = []
ts_c_transf = []
ts_c_endo  = []

snap_records = []   # (t, sol_W, c_state_copy, snap_idx)
snap_idx     = 0

print(f"\n=== Time loop: {N_STEPS} steps × Δt={DT*24:.1f} h  "
      f"({T_SIM:.1f} d total) ===")

for step in range(N_STEPS):
    t_new = (step + 1) * DT

    # ── hydraulics (quasi-steady at t_new) ──
    sol_W, fluxes, A_mat = _solve_hydraulics(psi_xyl_t(t_new))

    # ── solute: backward-Euler implicit step ──
    T_phys   = D_mat + A_mat + EPS * sp.eye(N_TOTAL, format='csr')
    V_inv_dt = V_diag / DT
    T_dyn    = T_phys + sp.diags(V_inv_dt, format='csr')
    rhs_dyn  = V_inv_dt * c_state

    c_meso_val = c_meso_t(t_new)
    T_dyn_bc, rhs_dyn_bc = _apply_bc(T_dyn, rhs_dyn, _build_rhs_bc(c_meso_val))
    c_new = spla.spsolve(T_dyn_bc, rhs_dyn_bc)

    _bad = ~np.isfinite(c_new)
    if _bad.any():
        print(f"  WARNING: {_bad.sum()} non-finite values at step {step} "
              f"(t={t_new:.4f} d) — zeroing and continuing")
        c_new[_bad] = 0.0

    c_state = c_new

    # ── record time series (last diurnal cycle only) ──
    if t_new >= t_last_start:
        T_orig = D_mat + A_mat
        Tc     = T_orig.dot(c_state)
        _ph    = nwj + np.array(phloem_loading_ids)
        _tr    = nwj + np.array(transfusion_parenchyma_ids)
        _en    = nwj + np.array(endo_cell_ids)
        Q_load     = float(np.sum(Tc[_ph]))          if phloem_loading_ids          else np.nan
        c_transf_m = float(np.nanmean(c_state[_tr])) if transfusion_parenchyma_ids  else np.nan
        c_endo_m   = float(np.nanmean(c_state[_en])) if endo_cell_ids               else np.nan
        # replace any remaining inf with nan so statistics stay meaningful
        if not np.isfinite(Q_load):     Q_load     = np.nan
        if not np.isfinite(c_transf_m): c_transf_m = np.nan
        if not np.isfinite(c_endo_m):   c_endo_m   = np.nan

        ts_t.append(t_new)
        ts_psi_xyl.append(psi_xyl_t(t_new))
        ts_c_meso.append(c_meso_val)
        ts_Q_load.append(Q_load)
        ts_c_transf.append(c_transf_m)
        ts_c_endo.append(c_endo_m)

    # ── snapshots ──
    if step in snap_steps:
        snap_records.append((t_new, sol_W.copy(), c_state.copy(), snap_idx))
        snap_idx += 1
        print(f"  snapshot {snap_idx}/{N_SNAP}  t={t_new*24:.1f} h  "
              f"ψ_xyl={psi_xyl_t(t_new):.0f} hPa  "
              f"c_meso={c_meso_val*1e6:.1f} µM")


# ── 9. Save snapshot figures ───────────────────────────────────────────────────

print("\n=== Saving snapshot figures ===")

C_MESO_MAX = (C_MEAN + C_AMP) * 1e6   # µM  colormap ceiling for concentration

for (t_snap, sol_W_snap, c_snap, sidx) in snap_records:
    t_h    = t_snap * 24.0
    t_rel  = (t_snap - t_last_start) * 24.0   # hours into last cycle
    label  = f"t = {t_rel:.1f} h  (ψ_xyl={psi_xyl_t(t_snap):.0f} hPa)"

    # Water potential
    psi_arr = np.full(N_TOTAL, np.nan)
    psi_arr[:] = sol_W_snap
    _plot_polygon(
        mecha, psi_arr,
        label='Ψ (hPa)',
        title=f'Water potential — {label}',
        save_path=os.path.join(_SNAP_DIR, f'needle_evo_psi_{sidx:02d}.png'),
        cmap='RdYlBu', vmin=PSI_ATM, vmax=PSI_MEAN + PSI_AMP,
    )

    # Concentration
    _plot_polygon(
        mecha, c_snap * 1e6,
        label='c (µM sucrose)',
        title=f'Sucrose concentration — {label}',
        save_path=os.path.join(_SNAP_DIR, f'needle_evo_c_{sidx:02d}.png'),
        cmap='plasma', vmin=0.0, vmax=C_MESO_MAX,
    )

    # Concentration gradient magnitude |∇c|
    _grad = _compute_grad_mag(c_snap, _centroid_of, _cell_adj)
    _plot_polygon(
        mecha, _grad * 1e6,
        label='|∇c| (µM µm⁻¹)',
        title=f'Concentration gradient |∇c| — {label}',
        save_path=os.path.join(_SNAP_DIR, f'needle_evo_gradc_{sidx:02d}.png'),
        cmap='hot_r', vmin=0.0,
    )


# ── 10. Concentration overview grid ───────────────────────────────────────────

print("\n=== Saving concentration overview grid ===")

n_cols = 4
n_rows = int(np.ceil(N_SNAP / n_cols))
fig_g, axes_g = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
axes_flat = axes_g.ravel()

for idx, (t_snap, sol_W_snap, c_snap, sidx) in enumerate(snap_records):
    ax = axes_flat[idx]
    t_rel = (t_snap - t_last_start) * 24.0
    _plot_polygon(
        mecha, c_snap * 1e6,
        label='c (µM)',
        title=f't = {t_rel:.1f} h',
        save_path=None, ax=ax,
        cmap='plasma', vmin=0.0, vmax=C_MESO_MAX,
    )

for ax in axes_flat[N_SNAP:]:
    ax.set_visible(False)

fig_g.suptitle('Diurnal sucrose evolution — one full cycle', fontsize=13)
plt.tight_layout()
grid_path = os.path.join(_OUT_DIR, 'needle_evo_concentration_grid.png')
fig_g.savefig(grid_path, dpi=120, bbox_inches='tight')
plt.close(fig_g)
print(f"  Saved → {grid_path}")


# ── 11. Time-series figure ─────────────────────────────────────────────────────

print("\n=== Saving time-series figure ===")

ts_t_h = [(t - t_last_start) * 24 for t in ts_t]

fig_ts, axes_ts = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# Panel 1: forcing functions
ax1 = axes_ts[0]
ax1b = ax1.twinx()
ax1.plot(ts_t_h, ts_psi_xyl, color='steelblue', label='ψ_xyl (hPa)')
ax1b.plot(ts_t_h, [c * 1e6 for c in ts_c_meso], color='forestgreen',
          linestyle='--', label='c_meso (µM)')
ax1.set_ylabel('ψ_xyl (hPa)', color='steelblue')
ax1b.set_ylabel('c_meso (µM)', color='forestgreen')
ax1.set_title('Diurnal forcing')
lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labs1 + labs2, loc='upper right', fontsize=8)

# Panel 2: tissue-mean concentrations
ax2 = axes_ts[1]
ax2.plot(ts_t_h, [c * 1e6 for c in ts_c_transf], color='darkorange',
         label='c_transf_paren (µM)')
ax2.plot(ts_t_h, [c * 1e6 for c in ts_c_endo],   color='purple',
         label='c_endo (µM)')
ax2.set_ylabel('Concentration (µM)')
ax2.set_title('Tissue-mean sucrose')
ax2.legend(loc='upper right', fontsize=8)

# Panel 3: phloem loading rate
ax3 = axes_ts[2]
ax3.plot(ts_t_h, [q * 1e12 for q in ts_Q_load], color='crimson',
         label='Q_load (pmol/d)')
ax3.axhline(0, color='gray', linewidth=0.5)
ax3.set_ylabel('Q_load (pmol/d)')
ax3.set_xlabel('Time in last cycle (h)')
ax3.set_title('Phloem loading rate')
ax3.legend(loc='upper right', fontsize=8)

plt.tight_layout()
ts_path = os.path.join(_OUT_DIR, 'needle_evo_timeseries.png')
fig_ts.savefig(ts_path, dpi=150, bbox_inches='tight')
plt.close(fig_ts)
print(f"  Saved → {ts_path}")


# ── 12. Results summary ────────────────────────────────────────────────────────

results_path = os.path.join(_OUT_DIR, 'needle_evo_results.txt')
with open(results_path, 'w') as _f:
    _f.write("Conifer needle — diurnal dynamic sucrose transport\n")
    _f.write("=" * 60 + "\n\n")

    _f.write("Simulation parameters:\n")
    _f.write(f"  T_SIM        = {T_SIM:.1f} d  ({int(T_SIM)} diurnal cycles)\n")
    _f.write(f"  DT           = {DT*24:.1f} h  ({N_STEPS} steps)\n")
    _f.write(f"  N_SNAP       = {N_SNAP} per last cycle\n\n")

    _f.write("Diurnal forcing:\n")
    _f.write(f"  PSI_MEAN={PSI_MEAN:.0f}  PSI_AMP={PSI_AMP:.0f}  [hPa]\n")
    _f.write(f"  C_MEAN={C_MEAN*1e6:.1f}  C_AMP={C_AMP*1e6:.1f}  [µM]\n\n")

    _f.write("Capacitance multipliers:\n")
    _f.write(f"  C_CAP_MESO   = {C_CAP_MESO}\n")
    _f.write(f"  C_CAP_TRANSF = {C_CAP_TRANSF}\n\n")

    if ts_Q_load:
        def _fmt(arr, scale=1.0):
            a = np.array(arr, dtype=float)
            a = a[np.isfinite(a)]
            if len(a) == 0:
                return 'mean=n/a  amp=n/a'
            return (f"mean={np.mean(a)*scale:.4g}  "
                    f"amp={0.5*(a.max()-a.min())*scale:.4g}  "
                    f"[range {a.min()*scale:.4g}…{a.max()*scale:.4g}]")

        _f.write("Last-cycle statistics (non-finite values excluded):\n")
        _f.write(f"  Q_load   (pmol/d): {_fmt(ts_Q_load,  1e12)}\n")
        _f.write(f"  c_transf (µM):     {_fmt(ts_c_transf, 1e6)}\n")
        _f.write(f"  c_endo   (µM):     {_fmt(ts_c_endo,   1e6)}\n")

print(f"\n  Results written → {results_path}")
print("\n=== DONE ===")
print(f"Outputs in: {_OUT_DIR}")
