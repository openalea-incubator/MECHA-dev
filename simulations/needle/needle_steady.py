"""
needle_steady.py
================
Fixed-input steady-state sucrose transport in a conifer needle cross-section.

SETUP
-----
Geometry : GRANAP NeedleAnatomy (Pinus-like, barrier=1 — Casparian strip at endodermis
           radial walls; tangential walls permeable to transmembrane flow)

Water BC  : Dirichlet – fixed water potential at xylem cells (PSI_XYL, high) and at
            substomatal air spaces adjacent to guard cells (PSI_ATM, reference = 0 hPa).
            Outer-boundary soil contact is disabled (xcontactrange = 1e10) so that only
            the two explicit Dirichlet nodes drive the flow.
            Expected flow path:
              xylem → (apo) transfusion tracheids → (CS barrier) endodermis → mesophyll
              → substomatal chambers → stomata
            With barrier=1 (CS), apoplastic conductance at endo-endo walls is reduced
            to kw_barrier_casparian ≈ 1e-16 cm/d/hPa, forcing water through the
            transmembrane (Lp) path across endodermal cells.

Solute transport (full operator, steady-state):
  - Mode   : 'full'  (apoplastic + transmembrane + plasmodesmatal)
  - Inputs : c_meso = C_MESO   (Dirichlet BC at mesophyll cells)
             c_xylem = 0       (Dirichlet BC at xylem cells; xylem sap sucrose ≈ 0)
  - Sink   : c_phloem = 0      (Dirichlet BC at phloem loading complex)
  - Other  : all remaining nodes solved freely (walls + cells)

OUTPUT VARIABLES
----------------
  Q_load   [mol/d]  net sucrose loading rate into phloem complex (Strasburger + sieve)
  Δc_load  [µM]     concentration gradient driving passive loading:
                    mean(c_transfusion) − mean(c_strasburger)

OUTPUTS
-------
  ~/simulations/needle/outputs/
    needle_steady_anatomy.png              – cross-section with tissue labels
    needle_steady_initial_waterpotential.png – water potential Dirichlet BC map (xylem/stomata/free)
    needle_steady_initial_concentration.png  – Dirichlet BC map (source/sink/free)
    needle_steady_concentration.png          – steady-state c field (cell nodes)
    needle_steady_concentration_gradient.png – |∇c| magnitude per cell (µM/µm)
    needle_steady_waterflow.png              – water velocity arrows on the network
    needle_steady_solute_velocity.png        – solute flux arrows on the network
    needle_steady_results.txt                – key numbers

UNITS
-----
  Lengths  µm / cm     Volumes  cm³     Fluxes  cm³/d
  Conc     mol/cm³  (= 10³ M)   Rates  mol/d
"""

import os
import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csgraph
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OUT_DIR    = os.path.join(_SCRIPT_DIR, 'outputs')
os.makedirs(_OUT_DIR, exist_ok=True)

_MECHA_SRC  = os.path.join(_SCRIPT_DIR, '..', '..', 'MECHA-dev', 'src')
_GRANAP_SRC = os.path.join(_SCRIPT_DIR, '..', '..', 'GRANAP-dev', 'src')
for p in (_MECHA_SRC, _GRANAP_SRC):
    sys.path.insert(0, os.path.abspath(p))

from granap.needle_class import NeedleAnatomy
from mecha.mecha_class import Mecha
from mecha.utils.data_loader import InData
from mecha.utils.network_builder import NetworkBuilder
from mecha.solute_transport import SoluteTransport
from mecha.utils.scenario_builder import ScenarioBuilder
from mecha.utils.visu import visualize
from mecha.utils.network_export import _plot_edge_vector_property


def _plot_concentration_polygon(mecha_obj, sol_arr, label, title, save_path,
                                cmap='plasma', vmin=0.0, vmax=None):
    """Polygon choropleth of sol_arr values on the cell GDF.

    Modified version of _visualize_water_potential + plot_water_potential_map:
    accepts any array indexed by the full network (sol_arr[node_id]) and saves
    to *save_path* instead of calling plt.show().
    """
    gdf = mecha_obj.network._cells_gdf.copy()
    nwj_loc = mecha_obj.network.n_wall_junction
    idx_map = mecha_obj.indice  # node_id → solution index (identity for GRANAP path)

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
# Xylem water potential: positive turgor representing root supply.
# Substomatal chamber potential: 0 hPa (atmospheric reference), driving net
# outflow from xylem → mesophyll → stomata.
PSI_XYL = 1000.0   # hPa  xylem turgor (supply side)
PSI_ATM  =    0.0  # hPa  atmospheric reference at stomatal air spaces (sink)

# Outer-boundary soil contact is fully disabled (x_contact = 1e10 µm → no
# border wall satisfies x >= x_contact in the needle cross-section).
X_CONTACT = 1e10   # µm  (effectively ∞; disables the soil/atm outer BC)

# ---- Sucrose concentrations (fixed inputs) -----------------------------------
C_MESO   = 50e-6   # mol/cm³  (= 50 mM; typical mesophyll sucrose, Reidel 2009)
C_PHLOEM = 0.0     # mol/cm³  (reference; actual phloem sap ~400 mM but we solve Δc)

# ---- Solute diffusivities ---------------------------------------------------
D_PD     = 1e-4    # cm²/d  plasmodesmatal diffusivity for sucrose (~1.2 µm²/s)

# Apoplastic diffusivity for sucrose in cell walls
# D_water ≈ 500 µm²/s = 43 cm²/d; with wall porosity/tortuosity ε/τ ≈ 0.002:
D_APO    = 0.1     # cm²/d  (~effective; apoplastic wall diffusivity)
D_MEM    = 1e-6    # cm²/d  passive transmembrane diffusivity (sucrose; <<D_PD, minor)

# ---- Membrane reflection coefficient (Kedem–Katchalsky) ----------------------
# σ = 0 → solute moves freely with water   σ = 1 → solute perfectly reflected
# Literature range for sucrose at plant plasma membranes: 0.85–0.97
# (Steudle 1993, Tyree & Zimmermann 2002, Nobel 2009)
SIGMA_SUCROSE = 0.9

# ---- Indices ----------------------------------------------------------------
H_IDX  = 0    # hydraulic parameter set index
I_MAT  = 0    # maturity stage index (only one stage here)
I_SCE  = 0    # scenario index

# cgroup → readable tissue name
CGROUP_NAMES = {
    1:  'epidermis',
    2:  'cortex',
    3:  'endodermis',
    4:  'mesophyll / airspace',
    5:  'transfusion parenchyma',
    9:  'hypodermis',
    10: 'proto-phloem',
    11: 'phloem sieve',
    12: 'Strasburger cell',
    13: 'xylem',
    14: 'cambium',
    15: 'parenchyma',
    16: 'airspace',
    19: 'meta-xylem',
    20: 'proto-xylem',
}

# ── 1. Build GRANAP needle ────────────────────────────────────────────────────

print("=== 1. Building GRANAP NeedleAnatomy ===")
needle = NeedleAnatomy()

needle.update_params("stomata",               "n_files",             8)
needle.update_params("resin_duct",            "n_files",             2)
needle.update_params("inter_cellular_spaces", "n_files",             8)
needle.update_params("transfusion_tissue", "transfusion_type", True)
needle.update_params("central_cylinder",      "shape",     "half_ellipse")

_ = needle.export_to_adjencymatrix()
cells_gdf = needle._cells_gdf
print(f"  Cells generated: {len(cells_gdf)}")

cell_types = cells_gdf['type'].value_counts()
print(f"  Cell type counts:\n{cell_types.to_string()}")

# Anatomy figure
fig_a, ax_a = plt.subplots(figsize=(7, 5))
needle.plot_cells(show=False, ax=ax_a, title="Needle cross-section (GRANAP)")
fig_a.savefig(os.path.join(_OUT_DIR, 'needle_steady_anatomy.png'), dpi=150, bbox_inches='tight')
plt.close(fig_a)
print("  Anatomy figure saved.")


# ── 2. Build MECHA network (barrier=1 — Casparian strip) ─────────────────────
#
# barrier=1: kw_endo_endo = kw_barrier_casparian ≈ 1e-16 cm/d/hPa (CS radial walls).
#            kw_endo_peri, kw_endo_cortex = kw (tangential walls remain permeable).
# This is the physiologically relevant state: "Casparian strip has evolved" means
# the radial walls of endodermal cells carry the casparian band, blocking apoplastic
# flow laterally while transmembrane (Lp) flow through the tangential walls still
# operates.  Full suberisation of tangential walls (barrier=3) additionally blocks
# these; it tends to make the hydraulic matrix poorly conditioned in MECHA because
# the only remaining path is the low-conductance membrane, so barrier=1 is used here.

print("\n=== 2. Building MECHA network (barrier=1 — Casparian strip) ===")
network = NetworkBuilder(needle)
network.populate_from_network()

default_input = InData()
default_input.geometry.set_maturity_stages([1], [200.0])
default_input.hydraulic.xcontactrange = [X_CONTACT]   # disable outer-boundary soil contact

mecha = Mecha(default_input, network=network)

nwj     = mecha.network.n_wall_junction
n_cells = mecha.network.n_cells
print(f"  barrier=1: kw_endo_endo = kw_barrier_casparian (Casparian strip active)")
print(f"  n_wall_junction={nwj}  n_cells={n_cells}")

# Dirichlet potential BC at xylem; phloem BC disabled
mecha.psi_xyl[1, I_MAT, I_SCE]   = PSI_XYL
mecha.psi_sieve[1, I_MAT, I_SCE] = np.nan


# ── 3. Identify tissue cell indices ──────────────────────────────────────────

print("\n=== 3. Identifying tissue cell indices ===")

cgroup_of_cell = {}   # cell_id → cgroup (for all cells)

meso_cell_ids              = []
endo_cell_ids              = []
transfusion_parenchyma_ids = []   # living; symplastic sucrose pathway to phloem
transfusion_tracheid_ids   = []   # dead water-conductors; no symplastic sucrose
transfusion_cell_ids       = []   # combined (kept for backward-compat with Pe calc)
strasburger_cell_ids       = []
phloem_cell_ids            = []
xylem_cell_ids             = []

outer_cell_ids    = []   # epidermis / hypodermis (outer needle boundary)
airspace_cell_ids = []   # gas-filled lacunae — no liquid sucrose

for nd, d in mecha.network.graph.nodes(data=True):
    idx = mecha.indice[nd]
    if idx < nwj:
        continue
    cell_id = idx - nwj
    cg      = int(d.get('cgroup', -1))
    ct      = str(d.get('cell_type', ''))
    cgroup_of_cell[cell_id] = cg

    if cg == 4 and ct == 'mesophyll':
        meso_cell_ids.append(cell_id)
    elif ct == 'transfusion parenchyma':
        transfusion_parenchyma_ids.append(cell_id)
    elif ct == 'transfusion tracheid':
        transfusion_tracheid_ids.append(cell_id)
    elif cg == 3:
        endo_cell_ids.append(cell_id)
    elif cg == 5:                              # fallback for non-subdivided anatomy
        transfusion_parenchyma_ids.append(cell_id)
    elif cg == 12 and ct == 'Strasburger cell':
        strasburger_cell_ids.append(cell_id)
    elif cg == 11:
        phloem_cell_ids.append(cell_id)
    elif cg == 13:
        xylem_cell_ids.append(cell_id)
    elif cg in (1, 9):                        # epidermis, hypodermis
        outer_cell_ids.append(cell_id)
    elif cg == 16 or 'air' in ct.lower():     # lacunae / intercellular air spaces
        airspace_cell_ids.append(cell_id)
    elif cg == 4:                             # cgroup-4 non-mesophyll = air space
        airspace_cell_ids.append(cell_id)

transfusion_cell_ids = transfusion_parenchyma_ids + transfusion_tracheid_ids

# Fallback: cgroup-4 mesophyll cells when no mesophyll label found
if not meso_cell_ids:
    for nd, d in mecha.network.graph.nodes(data=True):
        idx = mecha.indice[nd]
        if idx < nwj:
            continue
        if int(d.get('cgroup', -1)) == 4:
            meso_cell_ids.append(idx - nwj)

phloem_loading_ids = strasburger_cell_ids + phloem_cell_ids

# Identify guard cells and the substomatal air spaces adjacent to them.
# Guard cells → cgroup 12, cell_type "guard cell".
# Substomatal chambers → cell_type "air space" or "pore", adjacent to a guard cell.
# These receive the atmospheric Dirichlet BC (PSI_ATM) via hormones.contact.
guard_node_ids = set()
for nd, d in mecha.network.graph.nodes(data=True):
    if mecha.indice[nd] >= nwj and str(d.get('cell_type', '')) == 'guard cell':
        guard_node_ids.add(nd)

stomatal_airspace_ids = []
for nd, d in mecha.network.graph.nodes(data=True):
    idx = mecha.indice[nd]
    if idx < nwj:
        continue
    ct = str(d.get('cell_type', ''))
    if ct not in ('air space', 'pore'):
        continue
    for neighbor in mecha.network.graph.neighbors(nd):
        if neighbor in guard_node_ids:
            stomatal_airspace_ids.append(idx - nwj)
            break

# Wire substomatal chambers to atmosphere and cut off general outer contact
mecha.hormones.contact = stomatal_airspace_ids
mecha.boundary.scenarios[I_SCE]['psi_soil_left'] = PSI_ATM

print(f"  Mesophyll cells                    : {len(meso_cell_ids)}")
print(f"  Endodermis cells       (cgroup  3) : {len(endo_cell_ids)}")
print(f"  Transfusion parenchyma (living)    : {len(transfusion_parenchyma_ids)}")
print(f"  Transfusion tracheids  (dead)      : {len(transfusion_tracheid_ids)}")
print(f"  Strasburger cells      (cgroup 12) : {len(strasburger_cell_ids)}")
print(f"  Phloem sieve cells     (cgroup 11) : {len(phloem_cell_ids)}")
print(f"  Xylem cells            (cgroup 13) : {len(xylem_cell_ids)}")
print(f"  Phloem loading complex (11+12)     : {len(phloem_loading_ids)}")
print(f"  Outer boundary cells (epi/hypo)    : {len(outer_cell_ids)}")
print(f"  Air-space / lacuna cells           : {len(airspace_cell_ids)}")
print(f"  Guard cells            (cgroup 12) : {len(guard_node_ids)}")
print(f"  Stomatal air spaces (atm BC)       : {len(stomatal_airspace_ids)}")


# ── 4. Solve hydraulics (Dirichlet: xylem = PSI_XYL, stomata = PSI_ATM) ──────

print(f"\n=== 4. Hydraulic solve  PSI_XYL = {PSI_XYL:.0f} hPa  PSI_ATM = {PSI_ATM:.0f} hPa ===")

h_cm = float(mecha.geometry.maturity_stages[I_MAT].get('height')) * 1e-4
print(f"  Section height: {h_cm*1e4:.0f} µm = {h_cm:.4f} cm")
print(f"  Stomatal air spaces wired to atmosphere: {len(stomatal_airspace_ids)} cells")


def _solve_and_build_fluxes():
    """Dirichlet xylem + stomatal BCs → solve hydraulics → return (sol_arr, fluxes)."""
    sol_W, _verify, mat_W, _kmb, _rhs_s = mecha.solve_W(h=H_IDX, i_maturity=I_MAT)
    sol_arr = np.array(sol_W).ravel()
    coo = mat_W.tocoo()
    fluxes = []
    for i, j, v in zip(coo.row, coo.col, coo.data):
        if i < j and v > 0:
            pi, pj = float(sol_arr[i]), float(sol_arr[j])
            if not (np.isnan(pi) or np.isnan(pj)):
                f = v * (pi - pj)
                fluxes.append({'source': int(i), 'target': int(j), 'flux': f})
    return sol_arr, fluxes


sol_W, fluxes = _solve_and_build_fluxes()
mecha.edge_flux_list[I_MAT][I_SCE] = fluxes
print(f"  Hydraulic solve done: {len(fluxes)} edge fluxes")

# Pressure at xylem nodes
xyl_psis = [float(sol_W[mecha.indice[nd]])
            for nd, d in mecha.network.graph.nodes(data=True)
            if int(d.get('cgroup', -1)) == 13 and mecha.indice[nd] >= nwj
            and not np.isnan(sol_W[mecha.indice[nd]])]
if xyl_psis:
    print(f"  Xylem pressure: mean={np.mean(xyl_psis):.2f} hPa  "
          f"(range {min(xyl_psis):.1f}–{max(xyl_psis):.1f})")


# ── 4b. Initial water potential field (BC map) ───────────────────────────────
#
# Shows which cells carry Dirichlet BCs for water potential and which are
# solved freely.  Xylem cells are fixed at PSI_XYL; substomatal air spaces at
# PSI_ATM; all remaining cell nodes are free (shown in gray).

print("\n=== 4b. Plotting initial water potential field (BC state) ===")

psi_bc_arr = np.full(nwj + n_cells, np.nan)
for cid in xylem_cell_ids:
    psi_bc_arr[nwj + cid] = PSI_XYL
for cid in stomatal_airspace_ids:
    psi_bc_arr[nwj + cid] = PSI_ATM

_plot_concentration_polygon(
    mecha, psi_bc_arr,
    label='Ψ (hPa, BC only — gray = free nodes)',
    title=(
        'Initial water potential — Dirichlet BCs\n'
        f'Source: xylem {PSI_XYL:.0f} hPa  |  '
        f'Sink: stomatal air spaces {PSI_ATM:.0f} hPa  |  Gray: free'
    ),
    save_path=os.path.join(_OUT_DIR, 'needle_steady_initial_waterpotential.png'),
    cmap='RdYlBu', vmin=PSI_ATM, vmax=PSI_XYL,
)


# ── 5. Water flow connectivity check ─────────────────────────────────────────
#
# Classify every edge in edge_flux_list by the tissue pair it connects.
# 'apo' nodes have index < nwj; cell nodes >= nwj → cgroup from cgroup_of_cell.
# Edge types: apo-apo (wall diffusion), mem (wall↔cell), pd (cell↔cell via PD).
# The sign convention: flux recorded with F > 0 meaning net flow source→target;
# we track net signed flux between tissue groups to reveal the flow direction.

print("\n=== 5. Water flow connectivity check ===")

# Aggregate signed flux between tissue-group pairs (positive = flow in that direction)
flux_apo_by_pair  = defaultdict(float)   # apoplastic (apo-apo) tissue pairs
flux_mem_by_type  = defaultdict(float)   # transmembrane: tissue direction
flux_pd_by_pair   = defaultdict(float)   # symplastic PD: tissue-cell pairs

def _cgroup_label(idx):
    """Return tissue label for a node index (wall nodes grouped as 'apoplast')."""
    if idx < nwj:
        return 'apoplast'
    cid = idx - nwj
    cg  = cgroup_of_cell.get(cid, -1)
    return CGROUP_NAMES.get(cg, f'cgroup_{cg}')

for edge in fluxes:
    src, tgt, F = edge['source'], edge['target'], edge['flux']
    src_apo = src < nwj
    tgt_apo = tgt < nwj

    sl = _cgroup_label(src)
    tl = _cgroup_label(tgt)

    if src_apo and tgt_apo:
        # Apoplastic wall-to-wall flow
        key = (sl, tl) if sl <= tl else (tl, sl)
        flux_apo_by_pair[key] += abs(F)
    elif not src_apo and not tgt_apo:
        # Symplastic PD flow: F > 0 means flow from src-cell to tgt-cell
        key = (sl, tl) if F > 0 else (tl, sl)
        flux_pd_by_pair[key] += abs(F)
    else:
        # Transmembrane.  F is computed as K × (P_src − P_tgt):
        #   F > 0 → actual flow from src to tgt
        #   F < 0 → actual flow from tgt to src
        # One of src/tgt is an apo wall, the other is a cell.
        if src_apo and not tgt_apo:
            # src = wall, tgt = cell
            cell_label = tl
            direction  = 'apo→cell' if F > 0 else 'cell→apo'
        else:
            # src = cell, tgt = wall
            cell_label = sl
            direction  = 'cell→apo' if F > 0 else 'apo→cell'
        flux_mem_by_type[(cell_label, direction)] += abs(F)

# Net water flux per cell type (sum of all edges entering that cell type)
net_flux_per_cg = defaultdict(float)
for edge in fluxes:
    src, tgt, F = edge['source'], edge['target'], edge['flux']
    # Positive flow from src to tgt
    net_flux_per_cg[_cgroup_label(tgt)] += F
    net_flux_per_cg[_cgroup_label(src)] -= F

print("\n  Net water flux per tissue type (+ = net inflow, − = net outflow):")
ordered_tissues = ['xylem', 'transfusion parenchyma', 'endodermis',
                   'mesophyll / airspace', 'Strasburger cell',
                   'phloem sieve', 'epidermis', 'apoplast']
for tissue in ordered_tissues + [t for t in sorted(net_flux_per_cg) if t not in ordered_tissues]:
    q = net_flux_per_cg.get(tissue, 0.0)
    if abs(q) > 1e-15:
        print(f"    {tissue:30s}  {q:+.3e} cm³/d")

# Top PD-mediated symplastic fluxes (reveals symplastic water path)
print("\n  Top PD water fluxes between cell types:")
pd_sorted = sorted(flux_pd_by_pair.items(), key=lambda x: -x[1])[:8]
for (a, b), q in pd_sorted:
    if q > 1e-15:
        print(f"    {a:25s} → {b:25s}   {q:.3e} cm³/d")

# Top transmembrane fluxes (reveals Casparian strip waterpath)
print("\n  Top transmembrane fluxes (mem edges):")
mem_sorted = sorted(flux_mem_by_type.items(), key=lambda x: -x[1])[:8]
for (tissue, direction), q in mem_sorted:
    if q > 1e-15:
        print(f"    {tissue:25s}  {direction:12s}   {q:.3e} cm³/d")


# ── 6. SoluteTransport setup and connectivity ─────────────────────────────────

print("\n=== 6. Building SoluteTransport operator (full, steady-state) ===")

DP_FULL = dict(apo_wall=D_APO, membrane=D_MEM, plasmodesmata=D_PD,
               sigma={cg: SIGMA_SUCROSE for cg in range(1, 20)})
st = SoluteTransport(mecha, DP_FULL, None, mode='full')

D_mat = st.build_diffusion_matrix(H_IDX, I_MAT)
D_diag = np.abs(D_mat.diagonal())
D_diag_max = float(D_diag.max())
N_TOTAL = nwj + n_cells   # full system size (wall nodes + cell nodes)
print(f"  Diffusion matrix shape: {D_mat.shape}  |D_diag|_max = {D_diag_max:.3e}")

# Connectivity across the full (wall + cell) graph
D_row_norms     = np.array(np.abs(D_mat).sum(axis=1)).ravel()
isolated_node_ids = list(np.where(D_row_norms == 0)[0])   # full-network indices
print(f"  Zero-row nodes (no connections): {len(isolated_node_ids)}")

n_comp, comp_labels = csgraph.connected_components(D_mat, directed=False, connection='weak')
print(f"  Full transport graph: {n_comp} connected components")

# Nodes already anchored by Dirichlet BCs (full-network indices)
anchored_full_ids = set(
    [nwj + cid for cid in phloem_loading_ids]
    + [nwj + cid for cid in meso_cell_ids]
    + [nwj + cid for cid in xylem_cell_ids])
isolated_set = set(isolated_node_ids)

# For each component not containing an anchored node: add one extra anchor
extra_anchor_full_ids = []
for comp_id in range(n_comp):
    members = np.where(comp_labels == comp_id)[0]
    if any(m in anchored_full_ids for m in members):
        continue
    if all(m in isolated_set for m in members):
        continue
    non_iso = [m for m in members if m not in isolated_set]
    if non_iso:
        extra_anchor_full_ids.append(non_iso[0])

print(f"  Extra Dirichlet anchors needed: {len(extra_anchor_full_ids)}")

if phloem_loading_ids:
    ph_norms = D_row_norms[nwj + np.array(phloem_loading_ids)]
    print(f"  Phloem loading complex: {int(np.sum(ph_norms > 0))} / "
          f"{len(phloem_loading_ids)} cells have connections")


# ── 7. Solve solute transport (fixed c_meso + c_phloem) ──────────────────────
#
# Dirichlet BCs:
#   c_meso   = C_MESO  (mesophyll sucrose fixed by photosynthesis/osmosis)
#   c_phloem = C_PHLOEM = 0  (reference; actual gradient = c_meso − c_phloem)
# RHS = 0 everywhere (concentrations driven by BCs, no explicit source needed).

print(f"\n=== 7. Transport solve ===")
print(f"  c_meso   = {C_MESO*1e6:.1f} µM   (Dirichlet at {len(meso_cell_ids)} mesophyll cells)")
print(f"  c_phloem = {C_PHLOEM*1e6:.1f} µM   (reference Dirichlet at phloem loading complex)")

bc = {}
# Anchor isolated / disconnected nodes at 0 (full-network indices)
for nid in isolated_node_ids:
    bc[nid] = 0.0
for nid in extra_anchor_full_ids:
    bc[nid] = 0.0
# Dirichlet BCs at cell nodes (full-network index = nwj + cell_id)
for cid in phloem_loading_ids:
    bc[nwj + cid] = C_PHLOEM
for cid in meso_cell_ids:
    bc[nwj + cid] = C_MESO
for cid in xylem_cell_ids:
    bc[nwj + cid] = 0.0
# Outer boundary: epidermis/hypodermis and gas-filled lacunae anchored at 0.
# Prevents advection accumulation in the outer apoplast (no explicit stomatal sink).
#for cid in outer_cell_ids:
#    bc[nwj + cid] = 0.0
#for cid in airspace_cell_ids:
#    bc[nwj + cid] = 0.0

rhs = np.zeros(N_TOTAL)

A_mat = st.build_advection_matrix(I_MAT, I_SCE)
A_diag_max = float(np.abs(A_mat.diagonal()).max())
Pe_mesh_global = A_diag_max / D_diag_max if D_diag_max > 0 else 0.0
print(f"  Global mesh Peclet number (full) = {Pe_mesh_global:.4f}")

# Regularise T = D + A with small ε to handle structural null space
EPS = 1e-10 * D_diag_max
T_mat = D_mat + A_mat + EPS * sp.eye(N_TOTAL, format='csr')

T_lil = T_mat.tolil()
for node_id, c_val in bc.items():
    if 0 <= node_id < N_TOTAL:
        T_lil[node_id, :] = 0.0
        T_lil[node_id, node_id] = 1.0
        rhs[node_id] = float(c_val)
T_csr = T_lil.tocsr()

import scipy.sparse.linalg as spla
c_sol = spla.spsolve(T_csr, rhs)

if np.any(np.isnan(c_sol)) or np.any(np.isinf(c_sol)):
    print("  WARNING: transport solve returned NaN/Inf — system may be singular")
else:
    c_cells = c_sol[nwj:]
    c_walls = c_sol[:nwj]
    print(f"  Solve OK.  Cell c range:  [{c_cells.min()*1e6:.2f}, {c_cells.max()*1e6:.2f}] µM")
    print(f"             Wall c range:  [{c_walls.min()*1e6:.2f}, {c_walls.max()*1e6:.2f}] µM")


# ── 7b. Initial concentration field (BC map) ──────────────────────────────────
#
# Shows which cells carry Dirichlet BCs and which are solved freely.
# Useful to verify that source / sink placement matches the anatomy.

print("\n=== 7b. Plotting initial concentration field (BC state) ===")

# Build a full-network array with BC values (µM) and NaN for free nodes.
# Uses the same polygon choropleth as the steady-state plot (modified
# version of visualize(visu_type='water_potential')).
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
    title=(
        'Initial concentration field — Dirichlet BCs\n'
        f'Source: mesophyll {C_MESO*1e6:.0f} µM  |  '
        f'Sink: phloem loading 0 µM  |  Gray: free'
    ),
    save_path=os.path.join(_OUT_DIR, 'needle_steady_initial_concentration.png'),
    cmap='plasma', vmin=0.0, vmax=C_MESO * 1e6,
)


# ── 8. Output variables ───────────────────────────────────────────────────────

print("\n=== 8. Computing output variables ===")

# --- Q_load: net sucrose flux into phloem loading complex --------------------
# T_orig (before Dirichlet substitution) × c gives the physical flux at each node.
# For phloem cells held at c=0, this equals the total sucrose arriving from neighbors.
T_orig = D_mat + A_mat
Tc     = T_orig.dot(c_sol)
Q_load = (float(np.sum(Tc[nwj + np.array(phloem_loading_ids)]))
          if phloem_loading_ids else np.nan)

# --- Tissue-mean concentrations (c_sol indexed by full network: nwj + cell_id) ---
c_meso_mean   = float(np.mean(c_sol[nwj + np.array(meso_cell_ids)]))               if meso_cell_ids               else np.nan
c_endo_mean   = float(np.mean(c_sol[nwj + np.array(endo_cell_ids)]))               if endo_cell_ids               else np.nan
c_transf_mean = float(np.mean(c_sol[nwj + np.array(transfusion_parenchyma_ids)]))  if transfusion_parenchyma_ids  else np.nan
c_strasb      = float(np.mean(c_sol[nwj + np.array(strasburger_cell_ids)]))        if strasburger_cell_ids        else np.nan
c_phl_ss      = float(np.mean(c_sol[nwj + np.array(phloem_cell_ids)]))             if phloem_cell_ids             else np.nan

# --- Δc_load: concentration gradient at the loading site ----------------------
# The phloem loading complex (Strasburger + sieve) is pinned to C_PHLOEM = 0.
# The meaningful gradient is from the LAST upstream tissue (transfusion parenchyma)
# to the phloem loading complex — this is the driving force for passive loading.
# Δc at loading interface: c_transfusion → c_strasburger (= 0 by BC)
dc_load = c_transf_mean - (c_strasb if not np.isnan(c_strasb) else C_PHLOEM)

# Print results
print(f"\n  ── Concentration profile (tissue means) ──")
print(f"  c_meso          = {c_meso_mean*1e6:8.2f} µM   (input: {C_MESO*1e6:.1f} µM)")
print(f"  c_endo          = {c_endo_mean*1e6:8.2f} µM")
print(f"  c_transf paren  = {c_transf_mean*1e6:8.2f} µM")
print(f"  c_strasburger   = {c_strasb*1e6:8.2f} µM   (BC sink = 0)")
print(f"  c_phloem (BC)   = {C_PHLOEM*1e6:8.2f} µM")

print(f"\n  ── Loading site ──")
print(f"  Δc_load (transfusion → Strasburger) = {dc_load*1e6:.2f} µM")
print(f"    (last upstream step into phloem loading complex)")
print(f"  Q_load (full operator)              = {Q_load*1e12:.4f} pmol/d")


# ── 9. Concentration field plot ───────────────────────────────────────────────
#
# Modified version of visualize(visu_type='water_potential'): polygon choropleth
# of the steady-state sucrose field on the actual cell geometry.

print("\n=== 9. Plotting concentration field ===")

_plot_concentration_polygon(
    mecha, c_sol * 1e6,
    label='c (µM sucrose)',
    title=(
        f'Steady-state sucrose field  (barrier=1, Casparian strip)\n'
        f'c_meso = {C_MESO*1e6:.0f} µM  |  ΔΨ = {PSI_XYL - PSI_ATM:.0f} hPa  |  '
        f'Δc_load = {dc_load*1e6:.1f} µM'
    ),
    save_path=os.path.join(_OUT_DIR, 'needle_steady_concentration.png'),
    cmap='plasma', vmin=0.0, vmax=C_MESO * 1e6,
)


# ── 9b. Concentration gradient magnitude plot ─────────────────────────────────
#
# For each cell, |∇c| is estimated from the concentration differences to
# anatomically adjacent cells (those sharing a wall node in the graph), using
# a 2-component least-squares fit:  Δc_ij ≈ ∇c · (r_j − r_i).
# Units: c in mol/cm³ (displayed as µM ×1e6), coordinates in µm → µM µm⁻¹.

print("\n=== 9b. Plotting concentration gradient magnitude ===")

# Cell centroid lookup: graph node_id → (x µm, y µm)
_gdf_xy      = mecha.network._cells_gdf.copy()
_centroid_of = {}
for _, _row in _gdf_xy.iterrows():
    _cid = int(_row['id_cell'])
    _centroid_of[nwj + _cid] = (_row.geometry.centroid.x,
                                 _row.geometry.centroid.y)

# Cell adjacency: two cells are adjacent when they share a wall/junction node
_cell_adj = defaultdict(set)
for _wnd in mecha.network.graph.nodes():
    if mecha.indice.get(_wnd, nwj) >= nwj:
        continue                      # skip cell nodes, only process wall nodes
    _cnbs = [_n for _n in mecha.network.graph.neighbors(_wnd)
             if _n in _centroid_of]
    for _ii, _a in enumerate(_cnbs):
        for _b in _cnbs[_ii + 1:]:
            _cell_adj[_a].add(_b)
            _cell_adj[_b].add(_a)
# also capture direct cell-cell (plasmodesmatal) edges
for _u, _v in mecha.network.graph.edges():
    if _u in _centroid_of and _v in _centroid_of:
        _cell_adj[_u].add(_v)
        _cell_adj[_v].add(_u)

# Gradient magnitude at each cell via least-squares over neighbours
_grad_arr = np.full(N_TOTAL, np.nan)
for _nd, (_xi, _yi) in _centroid_of.items():
    _idx_i = mecha.indice.get(_nd)
    if _idx_i is None:
        continue
    _ci = float(c_sol[_idx_i])
    if np.isnan(_ci):
        continue
    _dr, _dc = [], []
    for _nb in _cell_adj[_nd]:
        _idx_j = mecha.indice.get(_nb)
        if _idx_j is None:
            continue
        _cj = float(c_sol[_idx_j])
        if np.isnan(_cj):
            continue
        _xj, _yj = _centroid_of[_nb]
        _dr.append([_xj - _xi, _yj - _yi])
        _dc.append(_cj - _ci)
    if len(_dr) >= 2:
        _g, *_ = np.linalg.lstsq(np.array(_dr), np.array(_dc), rcond=None)
        _grad_arr[_idx_i] = float(np.linalg.norm(_g))
    elif len(_dr) == 1:
        _d = np.linalg.norm(_dr[0])
        _grad_arr[_idx_i] = abs(_dc[0]) / _d if _d > 0 else 0.0
    else:
        _grad_arr[_idx_i] = 0.0

_n_grad = int(np.sum(np.isfinite(_grad_arr)))
_grad_max = float(np.nanmax(_grad_arr)) * 1e6
print(f"  Gradient computed for {_n_grad} cells; max |∇c| = {_grad_max:.4f} µM/µm")

_plot_concentration_polygon(
    mecha, _grad_arr * 1e6,
    label='|∇c| (µM µm⁻¹)',
    title=(
        'Concentration gradient magnitude  |∇c|\n'
        f'c_meso = {C_MESO*1e6:.0f} µM  |  barrier=1 (Casparian strip)'
    ),
    save_path=os.path.join(_OUT_DIR, 'needle_steady_concentration_gradient.png'),
    cmap='hot_r', vmin=0.0,
)


# ── 10. Velocity network plot ─────────────────────────────────────────────────
#
# Arrow magnitude ∝ |v| = |Q/A| per edge, coloured by path type
# (wall / membrane / plasmodesmata).  Uses the built-in velocity visualizer;
# Q and velocity are written on graph edges by solve_W → compute_edge_flows.

print("\n=== 10. Plotting velocity network ===")

visualize(
    mecha, visu_type='velocity',
    maturity_idx=I_MAT,
    save_path=os.path.join(_OUT_DIR, 'needle_steady_waterflow.png'),
)
print("  Velocity network figure saved.")


# ── 11. Solute flux network plot ──────────────────────────────────────────────
#
# Analogous to the water velocity plot: arrows on each graph edge show the net
# solute flux  J_ij = T[j,i] * (c_i - c_j)  where T = D_mat + A_mat.
# Magnitude is |J|; direction sign is taken from the signed value.

print("\n=== 11. Plotting solute flux network ===")

T_full = (D_mat + A_mat).tocsr()
for u, v, eattr in mecha.network.graph.edges(data=True):
    iu, iv = mecha.indice[u], mecha.indice[v]
    K_sol = abs(float(T_full[iv, iu]))
    J = K_sol * (c_sol[iu] - c_sol[iv]) if K_sol > 0 else 0.0
    eattr['solute_flux'] = abs(J)
    eattr['Q'] = J

_plot_edge_vector_property(
    mecha, prop_name='solute_flux', unit='mol d⁻¹',
    maturity_idx=I_MAT,
    save_path=os.path.join(_OUT_DIR, 'needle_steady_solute_velocity.png'),
)
print("  Solute flux network figure saved.")


# ── 12. Write results summary ─────────────────────────────────────────────────

results_path = os.path.join(_OUT_DIR, 'needle_steady_results.txt')
with open(results_path, 'w') as f:
    f.write("Conifer needle — symplastic sucrose transport with Casparian strip\n")
    f.write("=" * 70 + "\n\n")

    f.write("Geometry: GRANAP NeedleAnatomy (default Pinus-like params)\n")
    f.write(f"  Barrier configuration : barrier=1 (Casparian strip at endo radial walls)\n")
    f.write(f"    kw_endo_endo = kw_barrier_casparian (≈ 1e-16 cm/d/hPa)\n")
    f.write(f"    kw_endo tangential walls = kw (normal; membrane path open)\n")
    f.write(f"  Note: barrier=3 (full suberisation) additionally closes tangential walls;\n")
    f.write(f"        it is numerically challenging in MECHA and is the more extreme state.\n\n")

    f.write("Cell counts:\n")
    f.write(f"  Mesophyll                     : {len(meso_cell_ids)}\n")
    f.write(f"  Endodermis (cgroup 3)         : {len(endo_cell_ids)}\n")
    f.write(f"  Transfusion parenchyma (living): {len(transfusion_parenchyma_ids)}\n")
    f.write(f"  Transfusion tracheids  (dead) : {len(transfusion_tracheid_ids)}\n")
    f.write(f"  Strasburger cells (cgroup 12) : {len(strasburger_cell_ids)}\n")
    f.write(f"  Phloem sieve cells (cgroup 11): {len(phloem_cell_ids)}\n")
    f.write(f"  Xylem (cgroup 13)             : {len(xylem_cell_ids)}\n\n")

    f.write("Fixed inputs:\n")
    f.write(f"  C_MESO   = {C_MESO*1e6:.1f} µM   (mesophyll sucrose, Dirichlet BC)\n")
    f.write(f"  C_PHLOEM = {C_PHLOEM*1e6:.1f} µM   (phloem loading complex, reference sink)\n")
    f.write(f"  PSI_XYL  = {PSI_XYL:.1f} hPa  (Dirichlet BC at xylem cells)\n")
    f.write(f"  PSI_ATM  = {PSI_ATM:.1f} hPa  (Dirichlet BC at substomatal air spaces)\n")
    f.write(f"  X_CONTACT= {X_CONTACT:.0e} µm  (outer-boundary soil contact disabled)\n\n")

    f.write("Transport parameters (full operator):\n")
    f.write(f"  D_PD          = {D_PD:.2e} cm²/d  (plasmodesmatal diffusivity for sucrose)\n")
    f.write(f"  D_APO         = {D_APO:.2e} cm²/d  (apoplastic wall diffusivity)\n")
    f.write(f"  D_MEM         = {D_MEM:.2e} cm²/d  (passive transmembrane diffusivity)\n")
    f.write(f"  σ_sucrose     = {SIGMA_SUCROSE}       (membrane reflection coefficient; 10% of water flow carries sucrose)\n\n")

    f.write("Transpirational flow path (from hydraulic solve):\n")
    f.write(f"  Dirichlet BCs: xylem = {PSI_XYL:.0f} hPa → stomatal air spaces = {PSI_ATM:.0f} hPa\n")
    f.write(f"  Expected: xylem → transfusion parenchyma → endodermis (CS barrier)\n")
    f.write(f"            → mesophyll → substomatal chambers\n")
    f.write(f"  Net water flux per tissue (positive = net inflow, negative = net outflow):\n")
    ordered_tissues_txt = ['xylem', 'transfusion parenchyma', 'endodermis',
                           'mesophyll / airspace', 'Strasburger cell',
                           'phloem sieve', 'epidermis', 'apoplast']
    for t in ordered_tissues_txt:
        q = net_flux_per_cg.get(t, 0.0)
        f.write(f"    {t:30s}: {q:+.3e} cm³/d\n")
    f.write(f"\n  Top PD (symplastic) water fluxes:\n")
    for (a, b), q in pd_sorted[:5]:
        if q > 1e-15:
            f.write(f"    {a:25s} → {b:25s}  {q:.3e} cm³/d\n")
    f.write(f"\n  Top transmembrane water fluxes:\n")
    for (tissue, direction), q in mem_sorted[:5]:
        if q > 1e-15:
            f.write(f"    {tissue:25s}  ({direction:12s})  {q:.3e} cm³/d\n")

    f.write("\nConcentration profile (tissue means):\n")
    f.write(f"  c_meso          = {c_meso_mean*1e6:10.3f} µM   (imposed: {C_MESO*1e6:.1f} µM)\n")
    f.write(f"  c_endodermis    = {c_endo_mean*1e6:10.3f} µM\n")
    f.write(f"  c_transfusion   = {c_transf_mean*1e6:10.3f} µM\n")
    f.write(f"  c_strasburger   = {c_strasb*1e6:10.3f} µM\n")
    f.write(f"  c_phloem (BC)   = {C_PHLOEM*1e6:10.3f} µM   (reference sink)\n\n")

    f.write("Output variables:\n")
    f.write(f"  Q_load (full operator)       = {Q_load*1e12:.4f} pmol/d\n")
    f.write(f"  Δc_load (transfusion→Strasburger) = {dc_load*1e6:.2f} µM\n")
    f.write(f"    (driving force across the last step into the phloem loading complex)\n")
    f.write(f"  Global mesh Pe (full)        = {Pe_mesh_global:.4f}\n")

print(f"\n  Results written to: {results_path}")
print("\n=== DONE ===")
print(f"Outputs in: {_OUT_DIR}")
