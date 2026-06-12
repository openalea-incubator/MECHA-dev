"""
test_osmo_couple.py

Tests for the concentration–osmotic coupling implementation.

Five tests:

  test_set_osmotic_from_concentration
      Unit test: HydraulicCellManager.set_osmotic_from_concentration writes
      Ψ_os = −RT·c correctly onto every cell object.

  test_set_osmotic_with_baseline
      Unit test: set_osmotic_from_concentration with psi_os_baseline adds the
      dynamic van't Hoff term on top of the baseline (three cases).

  test_psi_os_preserved_through_water_flux
      Integration test: water_flux(use_stored_psi_os=True) preserves
      concentration-derived psi_os values through the internal
      reset_hydraulic_properties() call and does not overwrite them with
      scenario-dict values in initialize_scenarios.

  test_coupled_solver_convergence
      Integration test: coupled_water_solute_solve returns a valid,
      non-negative concentration field and reaches convergence.
      Convergence is on max |Δψ_total| (total water potential).

  test_coupled_solver_updates_osmotic
      Verifies that after coupling, psi_os = baseline + dynamic term and
      psi_total at mesophyll cells has shifted from the uncoupled reference.

Geometry
--------
GRANAP NeedleAnatomy with transfusion_type=False (undifferentiated transfusion
tissue, cgroup 4).  The anatomy is expensive to generate and is cached at
module level; a fresh NetworkBuilder + Mecha is constructed for each test.

Scenario layout (2 scenarios)
------------------------------
  Scenario 0 : default hydraulic only (no osmotic; s_factor=0).
  Scenario 1 : with osmotic (os_cortex=-4800 hPa, osmotic_sieve=-10000 hPa,
               s_factor=SIGMA).

The coupled solver requires i_scenario >= 1 (scenario 0 has no osmotic term).

Temperature and van't Hoff
--------------------------
  T = 298.15 K (25 °C).  R = 8.314e4 hPa cm³ mol⁻¹ K⁻¹ is used internally.
  Osmotic potential: Ψ_os [hPa] = −R·T × c [mol/cm³]
  50 mM sucrose = 50e-6 mol/cm³  →  Ψ_os ≈ −1239 hPa at 25 °C
"""

import copy
import os
import sys

import numpy as np
import scipy.sparse.csgraph as csgraph
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from granap.needle_class import NeedleAnatomy
from mecha.mecha_class import Mecha
from mecha.utils.data_loader import InData
from mecha.utils.network_builder import NetworkBuilder
from mecha.solute_transport import SoluteTransport
from mecha.utils.coupled_solver import coupled_water_solute_solve

# ── Physical constants ────────────────────────────────────────────────────────
T_25C  = 298.15              # K  (25 °C)  passed directly
_RT_25C = 8.314e4 * T_25C  # hPa·cm³/mol  used only in test assertions
C_MESO = 50e-6               # mol/cm³  mesophyll sucrose source (≈ 50 mM)
SIGMA  = 0.9                 # reflection coefficient

# Transport diffusivities
D_PD  = 1e-4   # cm²/d  plasmodesmata
D_APO = 0.1    # cm²/d  apoplastic wall
D_MEM = 1e-6   # cm²/d  membrane (max at σ=0); effective = D_MEM * (1-σ)

_OUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')

# ── Module-level anatomy cache ────────────────────────────────────────────────

_ANATOMY: NeedleAnatomy = None


def _get_anatomy() -> NeedleAnatomy:
    """Build and cache NeedleAnatomy (transfusion_type=False, ~5 s)."""
    global _ANATOMY
    if _ANATOMY is None:
        print('\n[fixture] Building NeedleAnatomy (transfusion_type=False)...')
        needle = NeedleAnatomy()
        needle.update_params("transfusion_tissue", "transfusion_type", False)
        needle.export_to_adjencymatrix()
        _ANATOMY = needle
        n = len(needle._cells_gdf)
        types = needle._cells_gdf['type'].value_counts().to_dict()
        print(f'  total cells: {n}')
        print(f'  type counts: {types}')
    return _ANATOMY


# ── Mecha builder ─────────────────────────────────────────────────────────────

def _build_mecha() -> Mecha:
    """
    Build a fresh Mecha from the cached anatomy.

    Two scenarios:
      0 – pure hydraulic (no osmotic, s_factor=0)
      1 – with osmotic (os_cortex=-4800 hPa, osmotic_sieve=-10000 hPa, s_factor=SIGMA)
    """
    needle = _get_anatomy()
    network = NetworkBuilder(needle)
    network.populate_from_network()

    data = InData()
    data.geometry.set_maturity_stages([1], [200.0])

    # Scenario 0: no osmotic driving force
    data.boundary.scenarios[0]['os_cortex']    = 0.0
    data.boundary.scenarios[0]['osmotic_sieve'] = 0.0
    data.boundary.scenarios[0]['s_factor']      = 0.0

    # Scenario 1: full osmotic
    s1 = copy.deepcopy(data.boundary.scenarios[0])
    s1['os_cortex']     = -4800.0
    s1['osmotic_sieve'] = -10000.0
    s1['s_factor']      = SIGMA
    s1['os_hetero']     = 0
    data.boundary.add_scenario(s1)

    return Mecha(data, network=network)


# ── Cell classification ───────────────────────────────────────────────────────

def _classify_cells(mecha: Mecha) -> dict:
    """Return {role: [cell_id, ...]} for transport BC assignment."""
    nwj = mecha.network.n_wall_junction
    meso, phloem, xylem = [], [], []
    for nd, d in mecha.network.graph.nodes(data=True):
        idx = mecha.indice[nd]
        if idx < nwj:
            continue
        cg = int(d.get('cgroup', -1))
        ct = str(d.get('cell_type', ''))
        cid = idx - nwj
        if cg == 4 and ct == 'mesophyll':
            meso.append(cid)
        elif cg == 11:
            phloem.append(cid)
        elif cg in (13, 19, 20):
            xylem.append(cid)
    return {'mesophyll': meso, 'phloem': phloem, 'xylem': xylem}


def _build_transport_bc(mecha: Mecha, st: SoluteTransport, cells: dict) -> dict:
    """
    Dirichlet BC dict for sym-mode transport.

    Mesophyll → C_MESO (source).
    Phloem and xylem → 0 (sinks).
    One node per unanchored connected component → 0, to prevent a singular matrix.
    Keys are full network node_ids (nwj + cell_id); SoluteTransport.solve shifts
    by nwj automatically in sym mode.
    """
    nwj = mecha.network.n_wall_junction
    bc: dict = {}
    for cid in cells['mesophyll']:
        bc[nwj + cid] = C_MESO
    for cid in cells['phloem']:
        bc[nwj + cid] = 0.0
    for cid in cells['xylem']:
        bc[nwj + cid] = 0.0

    # Find any connected component that has no Dirichlet anchor and pin its first
    # non-isolated member to 0 
    D_mat      = st.build_diffusion_matrix(0, 0)
    row_norms  = np.asarray(np.abs(D_mat).sum(axis=1)).ravel()
    isolated   = set(int(i) for i in np.where(row_norms == 0)[0])
    anchored   = {nid - nwj for nid in bc}   # cell_id set

    n_comp, labels = csgraph.connected_components(D_mat, directed=False,
                                                   connection='weak')
    for comp_id in range(n_comp):
        members = np.where(labels == comp_id)[0]
        if any(m in anchored for m in members):
            continue   # already anchored
        non_iso = [m for m in members if m not in isolated]
        if non_iso:
            bc[nwj + non_iso[0]] = 0.0
        else:
            # fully isolated nodes: pin every one
            for m in members:
                bc[nwj + m] = 0.0

    return bc


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_set_osmotic_from_concentration():
    """
    set_osmotic_from_concentration writes Ψ_os = −RT·c on all cells.

    Checks:
      - every cell with cell_id < n_cells gets psi_os = -RT * c_uniform
      - all psi_os values are negative (c > 0)
      - the van't Hoff value is correct to within numerical noise
    """
    print('\n[test_set_osmotic_from_concentration]')
    mecha   = _build_mecha()
    manager = mecha.network.cell_manager
    n_cells = mecha.network.n_cells
    nwj     = mecha.network.n_wall_junction

    c_uniform = np.full(n_cells, C_MESO)

    manager.set_osmotic_from_concentration(c_uniform, nwj, T_25C)

    expected = -_RT_25C * C_MESO
    n_set = 0
    for cell in manager:
        assert cell.psi_os is not None, \
            f'cell_id={cell.cell_id} has psi_os=None after set_osmotic_from_concentration'
        assert abs(cell.psi_os - expected) < 1e-6, \
            f'cell_id={cell.cell_id}: psi_os={cell.psi_os:.4f} != expected {expected:.4f}'
        assert cell.psi_os < 0, \
            f'cell_id={cell.cell_id}: psi_os should be negative, got {cell.psi_os}'
        n_set += 1

    assert n_set == n_cells, f'Expected {n_cells} cells set, got {n_set}'

    psi_val = -_RT_25C * C_MESO
    print(f'  n_cells={n_cells}  Ψ_os = {psi_val:.1f} hPa for C_MESO={C_MESO*1e6:.0f} mM')
    print('  PASSED: all cells have correct psi_os')


def test_set_osmotic_with_baseline():
    """
    set_osmotic_from_concentration with psi_os_baseline sums baseline and dynamic terms.

    Three cases:
      1. psi_os_baseline=None  →  Ψ_os = −RT·c          
      2. uniform baseline      →  Ψ_os = baseline + (−RT·c)
      3. heterogeneous baseline→  Ψ_os = baseline[i] + (−RT·c) per cell

    This is the unit test for the feature that allows unknown background solutes
    (represented by the baseline osmotic potential set from the scenario dict) to
    coexist with dynamically solved solutes.
    """
    print('\n[test_set_osmotic_with_baseline]')
    mecha   = _build_mecha()
    manager = mecha.network.cell_manager
    n_cells = mecha.network.n_cells
    nwj     = mecha.network.n_wall_junction

    c_uniform = np.full(n_cells, C_MESO)
    dynamic   = -_RT_25C * C_MESO       # ≈ −1239 hPa at 25 °C, 50 mM

    # ── case 1: no baseline ───────────────────────────────────────────────────
    manager.set_osmotic_from_concentration(c_uniform, nwj, T_25C)
    for cell in manager:
        assert abs(cell.psi_os - dynamic) < 1e-6, \
            f'case 1 cell_id={cell.cell_id}: expected {dynamic:.4f}, got {cell.psi_os:.4f}'

    # ── case 2: uniform baseline ──────────────────────────────────────────────
    baseline   = np.full(n_cells, -2000.0)
    expected_2 = -2000.0 + dynamic
    manager.set_osmotic_from_concentration(c_uniform, nwj, T_25C, baseline)
    for cell in manager:
        assert abs(cell.psi_os - expected_2) < 1e-6, \
            (f'case 2 cell_id={cell.cell_id}: '
             f'expected baseline+dynamic={expected_2:.4f}, got {cell.psi_os:.4f}')

    # ── case 3: heterogeneous baseline ────────────────────────────────────────
    rng        = np.random.default_rng(42)
    baseline_h = rng.uniform(-5000.0, -500.0, n_cells)
    manager.set_osmotic_from_concentration(c_uniform, nwj, T_25C, baseline_h)
    for cell in manager:
        cid      = cell.cell_id
        expected = float(baseline_h[cid]) + dynamic
        assert abs(cell.psi_os - expected) < 1e-6, \
            (f'case 3 cell_id={cid}: expected {expected:.4f}, got {cell.psi_os:.4f}')

    print(f'  dynamic term: {dynamic:.1f} hPa  (C_MESO={C_MESO*1e6:.0f} mM)')
    print('  PASSED: additive baseline respected in all three cases')


def test_psi_os_preserved_through_water_flux():
    """
    water_flux(use_stored_psi_os=True) preserves concentration-derived psi_os.

    Sequence:
      1. water_flux() — initial solve; cells get psi_os from scenario dict.
      2. set_osmotic_from_concentration(c_uniform) — overwrite with RT·c values.
      3. water_flux(use_stored_psi_os=True) — re-solve; must NOT overwrite psi_os.

    Checks:
      - After step 3, every cell still has psi_os == −RT · c_uniform.
      - The scenario-dict values (−4800 hPa for cortex cells in scenario 1) must
        NOT have been restored by initialize_scenarios.

    Scenario 1 prescribes os_cortex=-4800 hPa; the uniform concentration gives
    Ψ_os ≈ -1239 hPa — a clear difference used to verify preservation.
    """
    print('\n[test_psi_os_preserved_through_water_flux]')
    mecha   = _build_mecha()
    manager = mecha.network.cell_manager
    n_cells = mecha.network.n_cells
    nwj     = mecha.network.n_wall_junction

    # Step 1: initial hydraulic solve (assigns psi_os from scenario dict)
    print('  Step 1: initial water_flux() ...')
    mecha.water_flux()

    # Snapshot psi_os after scenario-dict solve
    psi_os_from_scenario = {
        cell.cell_id: cell.psi_os
        for cell in manager
        if cell.psi_os is not None
    }
    n_scenario_set = len(psi_os_from_scenario)
    print(f'  Cells with psi_os after scenario solve: {n_scenario_set}')
    assert n_scenario_set > 0, 'No cells have psi_os after water_flux()'

    # Verify that at least some cells have the scenario-prescribed value
    # (os_cortex=-4800 in scenario 1)
    scenario_vals = list(psi_os_from_scenario.values())
    assert any(abs(v - (-4800.0)) < 50.0 for v in scenario_vals), \
        'Expected some cells to have psi_os ≈ -4800 hPa from scenario 1'

    # Step 2: overwrite psi_os with uniform concentration-derived values
    c_uniform = np.full(n_cells, C_MESO)
    manager.set_osmotic_from_concentration(c_uniform, nwj, T_25C)

    expected = -_RT_25C * C_MESO   # ≈ -1239 hPa at 25°C, 50 mM
    print(f'  Step 2: set_osmotic_from_concentration → Ψ_os = {expected:.1f} hPa')

    for cell in manager:
        assert abs(cell.psi_os - expected) < 1e-6, \
            f'cell_id={cell.cell_id} not updated: psi_os={cell.psi_os:.2f}'

    # Step 3: re-solve with stored psi_os; the flag must prevent overwriting
    print('  Step 3: water_flux(use_stored_psi_os=True) ...')
    mecha.water_flux(use_stored_psi_os=True)

    n_overwritten = 0
    for cell in manager:
        if cell.psi_os is None:
            # A cell that had psi_os set but now has None was wiped — failure.
            assert False, \
                f'cell_id={cell.cell_id}: psi_os was reset to None by water_flux'
        if abs(cell.psi_os - expected) > 1.0:   # 1 hPa tolerance
            n_overwritten += 1

    assert n_overwritten == 0, (
        f'{n_overwritten} cells had psi_os overwritten by water_flux '
        f'(use_stored_psi_os=True should prevent this). '
        f'Expected {expected:.1f} hPa, found deviations > 1 hPa.'
    )
    print(f'  PASSED: all {n_cells} cells kept psi_os = {expected:.1f} hPa')


def test_coupled_solver_convergence():
    """
    coupled_water_solute_solve returns a valid concentration field and converges.

    operators='D' (diffusion only) is used here because the needle anatomy with
    Casparian-strip osmotic driving forces (~4800 hPa) makes _build_osmotic_advection
    produce very stiff advection terms that render operators='T' ill-conditioned.
    Diffusion-only transport is bounded, well-conditioned, and concentrations are
    independent of flow, so the coupling loop converges in 2 iterations:
      iter 1 → c from diffusion BCs → update psi_os → water_flux → psi_total_1
      iter 2 → same c (flow unchanged) → same psi_os → water_flux → psi_total_2
            → Δψ_total = 0 < tol → done.
    This tests the coupling MECHANISM (psi_os update → hydraulics change) without
    requiring a stable full advection-diffusion solve.

    Checks:
      - No exception raised.
      - Returned concentration array has no NaN at BC-anchored cells.
      - Minimum concentration >= −1e-10 (no unphysical negatives).
      - n_iter >= 1 (at least one transport solve performed).
      - converged == True within max_iter=5.

    Transport (sym mode):
      Source: mesophyll cells at C_MESO = 50 mM.
      Sink  : phloem and xylem cells at 0.
      Unanchored connected components pinned at 0.
    """
    print('\n[test_coupled_solver_convergence]')
    mecha   = _build_mecha()
    n_cells = mecha.network.n_cells
    nwj     = mecha.network.n_wall_junction

    cells = _classify_cells(mecha)
    print(f'  mesophyll={len(cells["mesophyll"])}  '
          f'phloem={len(cells["phloem"])}  '
          f'xylem={len(cells["xylem"])}')
    assert len(cells['mesophyll']) > 0, 'No mesophyll cells found'
    assert len(cells['phloem'])    > 0, 'No phloem cells found'

    D_MEM_EFF = D_MEM * (1.0 - SIGMA)
    dp = dict(apo_wall=D_APO, plasmodesmata=D_PD, membrane=D_MEM_EFF,
              sigma={cg: SIGMA for cg in range(1, 20)})
    st = SoluteTransport(mecha, dp, capacitance_params=None, mode='sym')

    bc = _build_transport_bc(mecha, st, cells)
    print(f'  Dirichlet BCs: {len(bc)} nodes '
          f'(mesophyll={len(cells["mesophyll"])}, '
          f'sink+isolated={len(bc)-len(cells["mesophyll"])})')

    rhs = np.zeros(n_cells)

    print(f'  T={T_25C:.2f} K  '
          f'C_MESO={C_MESO*1e6:.0f} mM  '
          f'→ Ψ_os(source) = {-_RT_25C*C_MESO:.1f} hPa')

    c, n_iter, converged = coupled_water_solute_solve(
        mecha, st,
        T=T_25C,
        boundary_conditions=bc,
        rhs=rhs,
        i_scenario=1,
        i_maturity=0,
        h=0,
        tol=10.0,
        max_iter=5,
        operators='D',
        verbose=True,
    )

    print(f'  n_iter={n_iter}  converged={converged}')
    print(f'  c range: [{c.min()*1e6:.2f}, {c.max()*1e6:.2f}] µM')

    assert c is not None, 'coupled_water_solute_solve returned None'
    assert c.shape == (n_cells,), \
        f'Expected concentration shape ({n_cells},), got {c.shape}'
    assert n_iter >= 1, 'No iterations performed'
    assert converged, \
        f'Solver did not converge in {n_iter} iterations with operators="D" (tol=10 hPa). ' \
        f'Diffusion-only transport is flow-independent, so iter 2 must give Δψ_total=0.'

    # No unphysical negatives at interior cells (small numerical noise tolerated)
    c_min = float(c.min())
    assert c_min >= -1e-10, \
        f'Unphysical negative concentration: c_min = {c_min:.3e} mol/cm³'

    # Source cells should be close to their Dirichlet BC value
    meso_ids = cells['mesophyll']
    c_meso_mean = float(c[meso_ids].mean()) if meso_ids else float('nan')
    assert abs(c_meso_mean - C_MESO) / C_MESO < 1e-6, \
        f'Mesophyll concentration deviated from Dirichlet BC: ' \
        f'{c_meso_mean*1e6:.3f} µM vs expected {C_MESO*1e6:.0f} µM'

    # Phloem sink cells should be at 0
    phloem_ids = cells['phloem']
    c_phloem_max = float(c[phloem_ids].max()) if phloem_ids else 0.0
    assert c_phloem_max < 1e-10, \
        f'Phloem Dirichlet BC violated: max c_phloem = {c_phloem_max:.3e}'

    print('  PASSED')


def test_coupled_solver_updates_osmotic():
    """
    The coupling adds the dynamic van't Hoff term on top of the scenario baseline.

    After the coupled solve, mesophyll cells must have
        Ψ_os ≈ Ψ_os_baseline + (−RT · C_MESO)
              ≈ −4800 + (−1239) ≈ −6039 hPa
    where Ψ_os_baseline (−4800 hPa) comes from the scenario dict (os_cortex) and
    represents unknown background solutes, and −RT · C_MESO is the dynamic van't
    Hoff contribution from the transport solver.

    The difference |Ψ_os_coupled − Ψ_os_scenario| for mesophyll cells must
    exceed 1000 hPa — a clear signal that the dynamic term was applied.  It must
    also be within 100 hPa of the additive expectation (baseline + dynamic).
    """
    print('\n[test_coupled_solver_updates_osmotic]')
    mecha   = _build_mecha()
    manager = mecha.network.cell_manager
    n_cells = mecha.network.n_cells
    nwj     = mecha.network.n_wall_junction

    cells = _classify_cells(mecha)

    D_MEM_EFF = D_MEM * (1.0 - SIGMA)
    dp = dict(apo_wall=D_APO, plasmodesmata=D_PD, membrane=D_MEM_EFF,
              sigma={cg: SIGMA for cg in range(1, 20)})
    st = SoluteTransport(mecha, dp, capacitance_params=None, mode='sym')
    bc = _build_transport_bc(mecha, st, cells)

    # ── Reference: psi_os and psi_total after a plain water_flux ─────────────
    mecha.water_flux()
    psi_os_ref    = {cell.cell_id: (cell.psi_os    if cell.psi_os    is not None else 0.0)
                     for cell in manager}
    psi_total_ref = {cell.cell_id: cell.psi_total
                     for cell in manager
                     if cell.psi_total is not None}

    # Scenario 1 should have assigned os_cortex=-4800 to mesophyll-like cells.
    # Verify reference contains that value.
    meso_ids = cells['mesophyll']
    ref_meso_vals = [psi_os_ref[cid] for cid in meso_ids if cid in psi_os_ref]
    print(f'  Reference psi_os at mesophyll cells: '
          f'mean={np.mean(ref_meso_vals):.1f} hPa  '
          f'(scenario os_cortex=-4800 hPa)')
    assert any(abs(v - (-4800.0)) < 100.0 for v in ref_meso_vals), \
        ('Expected mesophyll cells to have psi_os ≈ −4800 hPa from scenario 1 '
         '(initialize_scenarios with os_hetero=0, os_cortex=-4800).')

    # ── Rebuild for coupled solve (fresh mecha to reset all state) ────────────
    mecha2  = _build_mecha()
    manager2 = mecha2.network.cell_manager
    st2 = SoluteTransport(mecha2, dp, capacitance_params=None, mode='sym')
    bc2 = _build_transport_bc(mecha2, st2, cells)

    c, n_iter, converged = coupled_water_solute_solve(
        mecha2, st2,
        T=T_25C,
        boundary_conditions=bc2,
        rhs=np.zeros(n_cells),
        i_scenario=1,
        tol=10.0,
        max_iter=5,
        operators='D',
        verbose=False,
    )
    print(f'  Coupled solve: n_iter={n_iter}  converged={converged}')

    # ── Check that mesophyll psi_os = baseline + van't Hoff dynamic term ────────
    # The baseline is the scenario-dict value captured before the coupled loop.
    # The dynamic term is -RT * C_MESO for mesophyll cells (Dirichlet at C_MESO).
    baseline_meso        = float(np.mean(ref_meso_vals))   # ≈ -4800 hPa
    dynamic_term         = -_RT_25C * C_MESO               # ≈ -1239 hPa
    expected_meso_psi_os = baseline_meso + dynamic_term    # ≈ -6039 hPa

    psi_os_coupled_meso = [
        cell.psi_os for cell in manager2
        if cell.cell_id in meso_ids and cell.psi_os is not None
    ]
    assert len(psi_os_coupled_meso) > 0, \
        'No mesophyll cells have psi_os after coupled solve'

    mean_coupled = float(np.mean(psi_os_coupled_meso))
    print(f'  Coupled psi_os at mesophyll: mean={mean_coupled:.1f} hPa  '
          f'(expected ≈ {expected_meso_psi_os:.1f} hPa = '
          f'{baseline_meso:.1f} baseline + {dynamic_term:.1f} dynamic)')

    # The coupled value must differ from the raw scenario baseline — the dynamic
    # term (~-1239 hPa) must have been added on top.
    delta_from_scenario = abs(mean_coupled - baseline_meso)
    assert delta_from_scenario > 1000.0, (
        f'Coupled psi_os at mesophyll ({mean_coupled:.1f} hPa) is too close to '
        f'the scenario baseline ({baseline_meso:.1f} hPa): '
        f'Δ = {delta_from_scenario:.1f} hPa (expected > 1000 hPa). '
        f'The dynamic van\'t Hoff term may not have been applied.'
    )

    # The coupled value must equal baseline + dynamic within tolerance.
    delta_from_additive = abs(mean_coupled - expected_meso_psi_os)
    assert delta_from_additive < 100.0, (
        f'Coupled psi_os at mesophyll ({mean_coupled:.1f} hPa) deviates from '
        f'additive expectation ({expected_meso_psi_os:.1f} hPa = '
        f'{baseline_meso:.1f} + {dynamic_term:.1f}) by '
        f'{delta_from_additive:.1f} hPa (tolerance 100 hPa).'
    )

    print(f'  Δ from scenario baseline: {delta_from_scenario:.1f} hPa  '
          f'(threshold > 1000 hPa ✓)')
    print(f'  Δ from additive expectation: {delta_from_additive:.1f} hPa  '
          f'(tolerance 100 hPa ✓)')

    # ── Check that psi_total shifted — the convergence criterion is on Δψ_total ─
    # If coupling had no effect on psi_total the new convergence criterion would
    # converge trivially without actually coupling anything.
    psi_total_coupled_meso = [
        cell.psi_total for cell in manager2
        if cell.cell_id in meso_ids and cell.psi_total is not None
    ]
    assert len(psi_total_coupled_meso) > 0, \
        'No mesophyll cells have psi_total after coupled solve'

    ref_psi_total_meso = [psi_total_ref[cid] for cid in meso_ids
                          if cid in psi_total_ref]
    assert len(ref_psi_total_meso) > 0, \
        'No mesophyll cells have psi_total in reference solve'

    mean_psi_total_coupled = float(np.mean(psi_total_coupled_meso))
    mean_psi_total_ref     = float(np.mean(ref_psi_total_meso))
    delta_psi_total        = abs(mean_psi_total_coupled - mean_psi_total_ref)

    assert delta_psi_total > 100.0, (
        f'psi_total at mesophyll did not shift after coupling: '
        f'ref={mean_psi_total_ref:.1f} hPa, coupled={mean_psi_total_coupled:.1f} hPa, '
        f'Δ={delta_psi_total:.1f} hPa (expected > 100 hPa). '
        f'The Δψ_total convergence criterion may not be exercised.'
    )
    print(f'  psi_total shift at mesophyll: {delta_psi_total:.1f} hPa  '
          f'(ref={mean_psi_total_ref:.1f}, coupled={mean_psi_total_coupled:.1f}) ✓')
    print('  PASSED')


# ── Standalone execution ──────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs(_OUT_DIR, exist_ok=True)

    test_set_osmotic_from_concentration()
    print('\n✓ test_set_osmotic_from_concentration PASSED')

    test_set_osmotic_with_baseline()
    print('\n✓ test_set_osmotic_with_baseline PASSED')

    test_psi_os_preserved_through_water_flux()
    print('\n✓ test_psi_os_preserved_through_water_flux PASSED')

    test_coupled_solver_convergence()
    print('\n✓ test_coupled_solver_convergence PASSED')

    test_coupled_solver_updates_osmotic()
    print('\n✓ test_coupled_solver_updates_osmotic PASSED')

    print('\n=== All tests PASSED ===')
