"""MECHA tutorial: water uptake in a tomato root, across day/night and control/stress
scenarios.

This script builds a tomato (Solanum lycopersicum) root anatomy, wraps it in a
hydraulic network, and solves for radial water flow under several boundary
conditions. It ends by plotting flow and water-potential profiles.

If you have little coding experience, you only need to touch two places:

    1. build_tomato_root()  -- anatomical parameters (cell sizes, layer counts,
                                aerenchyma, xylem geometry, ...).
    2. base_scenario()      -- osmotic and pressure boundary values (soil,
                                tissues, xylem), in hPa.

Everything below those two functions (running scenarios, solving, plotting)
does not need to change to explore different anatomies or osmotic conditions.
"""

from openalea.mecha import Mecha, InData, NetworkBuilder, visualize
from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData
import matplotlib.pyplot as plt
import numpy as np


SEED = 0  # random seed for cell generation (same seed -> same anatomy)


# ---------------------------------------------------------------------------
# 1. TOMATO ROOT ANATOMY (edit me)
# ---------------------------------------------------------------------------
def build_tomato_root(seed: int = SEED) -> RootAnatomy:
    """Build a tomato primary (young, non-secondary-growth) root anatomy.

    Change any of the numbers below to explore a different tomato anatomy.
    Each `set_value(tissue, parameter, value)` call tunes one anatomical
    feature; the tissue/parameter names are fixed, only the values are meant
    to be edited.
    """
    tomato = OrganInputData.for_dicot_root()  # dicot root preset (tomato is a dicot)

    # -- Aerenchyma (air channels in the cortex) -----------------------------
    # 0.0 = no aerenchyma. Increase towards 1.0 to simulate aerenchyma
    # formation (e.g. under flooding/stress).
    tomato.set_value("aerenchyma", "aerenchyma_proportion", 0.0)

    # -- Stele size + radial parenchyma gradient -----------------------------
    tomato.set_value("stele", "thickness", 0.116)             # overall stele radius
    tomato.set_value("stele", "cell_diameter", 0.01)           # cell size at stele edge
    tomato.set_value("stele", "cell_diameter_center", 0.030)   # cell size at stele center

    # -- Star xylem (number of arms, radial extent, vessel size gradient) ---
    tomato.set_value("xylem", "n_vascular_peak", 2)            # number of xylem star arms
    tomato.set_value("xylem", "radius_valley_side", 0.015)
    tomato.set_value("xylem", "radius_peak_side", 0.058)
    tomato.set_value("xylem", "arc_peak_side", 0.008)
    tomato.set_value("xylem", "arc_valley_side", 0.015)
    tomato.set_value("xylem", "vessel_diameter", 0.0166)       # metaxylem vessel size
    tomato.set_value("xylem", "vessel_diameter_min", 0.01)
    tomato.set_value("xylem", "pith_radius", 0.0)
    tomato.set_value("xylem", "gradient_inflection", 0.8)
    tomato.set_value("xylem", "gradient_steepness", 5)
    tomato.set_value("xylem", "allow_ellipse", True)

    # -- Primary phloem (between the star arms, near the cambium) -----------
    tomato.set_value("phloem", "sieve_diameter", 0.005)
    tomato.set_value("phloem", "relative_distance", 0.8)
    tomato.set_value("phloem", "cluster_height", 0.013)
    tomato.set_value("phloem", "cluster_width", 0.035)

    # -- Primary cambium ring (valleys first, maturing outward) -------------
    tomato.set_value("cambium", "radius_valley_side", 0.017)
    tomato.set_value("cambium", "radius_peak_side", 0.059)
    tomato.set_value("cambium", "visible_distance", 0.02)
    tomato.set_value("cambium", "arc_peak_side", 0.013)
    tomato.set_value("cambium", "arc_valley_side", 0.016)
    tomato.set_value("cambium", "cell_diameter", 0.005)
    tomato.set_value("cambium", "cell_width", 0.01)

    # -- Endodermis / pericycle -----------------------------------------------
    tomato.set_value("endodermis", "cell_diameter", 0.015)
    tomato.set_value("endodermis", "cell_width", 0.019)
    tomato.set_value("pericycle", "cell_diameter", 0.0141)
    tomato.set_value("pericycle", "cell_width", 0.0086)

    # -- Cortex layers (inner / main / outer) --------------------------------
    tomato.params.append({
        "name": "inner_cortex",
        "cell_diameter": 0.015,
        "cell_width": 0.014,
        "n_layers": 1,
        "shift": 0.5,
        "order": 3.5,
    })
    tomato.set_value("cortex", "cell_diameter", 0.034)
    tomato.set_value("cortex", "cell_width", 0.03)
    tomato.set_value("cortex", "n_layers", 1)
    tomato.params.append({
        "name": "outer_cortex",
        "cell_diameter": 0.0225,
        "cell_width": 0.0225,
        "n_layers": 2,
        "shift": 0.5,
        "order": 4.5,
    })

    # -- Epidermis ------------------------------------------------------------
    tomato.set_value("epidermis", "cell_diameter", 0.027)
    tomato.set_value("epidermis", "cell_width", 0.0255)

    # -- Intercellular spaces across the cortex tissues ----------------------
    tomato.set_value("inter_cellular_spaces", "smoothness", 0.05)
    tomato.set_value("inter_cellular_spaces", "tissue",
                      ["inner_cortex", "cortex", "outer_cortex"])

    # Build the anatomy once, after all configuration is in place.
    root = RootAnatomy(tomato, seed=seed)
    root.remove_layer("exodermis")  # tomato has no separate exodermis layer
    root.generate_cells()

    # Merge the inner/outer cortex tags into a single "cortex" tag.
    root.retag_cells("inner_cortex", "cortex")
    root.retag_cells("outer_cortex", "cortex")

    # Prepare the anatomy for MECHA (builds the cell-to-cell connectivity graph)
    root.export_to_adjencymatrix()
    return root


# ---------------------------------------------------------------------------
# 2. OSMOTIC / PRESSURE BOUNDARY VALUES (edit me)
# ---------------------------------------------------------------------------
def base_scenario(input_data: InData) -> dict:
    """Populate and return the base ("control night") boundary scenario.

    All values are in hPa (1 MPa = 1E4 hPa). More negative = drier / lower
    water potential. Edit these to explore different soil or plant water
    status conditions; the scenario functions below derive day/night and
    control/stress variants from this base by copying and tweaking it.
    """
    scenario = input_data.boundary.scenarios[0]
    scenario['osmotic_left_soil'] = -1.5E3    # soil osmotic potential, left side
    scenario['osmotic_right_soil'] = -1.5E3   # soil osmotic potential, right side
    scenario['osmotic_epi'] = -6.8E3          # epidermis osmotic potential
    scenario['osmotic_exo'] = -7.55E3         # exodermis osmotic potential
    scenario['osmotic_endo'] = -4.80E3        # endodermis osmotic potential
    scenario['osmotic_peri'] = -5.5E3         # pericycle osmotic potential
    scenario['osmotic_stele'] = -5.5E3        # stele osmotic potential
    scenario['osmotic_xyl'] = -1.40E3         # xylem osmotic potential
    scenario['pressure_xyl_prox'] = -5.0E3    # xylem pressure at the proximal end
    scenario['osmotic_sieve'] = -1.0E4        # phloem sieve tube osmotic potential
    scenario['os_hetero'] = 4                 # use per-tissue osmotic values (epi/exo/endo/peri/stele)
    scenario['os_cortex'] = -8.3E3            # cortex osmotic potential
    return scenario


def run_waterpump() -> None:
    """Run standard water pump scenarios and visualize radial profiles and flows.

    This function tests control night/day and stress night/day scenarios for different
    maturity stages of the root, and generates visual plots of flow and water potential
    profiles.
    """
    root = build_tomato_root()

    # prepare default inputs for mecha
    default_input = InData()

    # control night no suberin
    default_input.geometry.set_maturity_stages([6, 9])
    base_scenario(default_input)

    # control night no suberin
    new_scenario = default_input.boundary.scenarios[0].copy()
    new_scenario['osmotic_exo'] = (new_scenario['osmotic_epi'] + new_scenario['os_cortex']) / 2
    new_scenario['osmotic_peri'] = -5E3
    new_scenario['osmotic_stele'] = -5E3
    new_scenario['osmotic_endo'] = (new_scenario['osmotic_peri'] + new_scenario['os_cortex']) / 2
    new_scenario['pressure_xyl_prox'] = -2.5E3
    default_input.boundary.add_scenario(new_scenario)
    # control day no suberin
    new_scenario = default_input.boundary.scenarios[0].copy()
    new_scenario['osmotic_epi'] = -7.5E3
    new_scenario['os_cortex'] = -6.8E3
    new_scenario['osmotic_exo'] = (new_scenario['osmotic_epi'] + new_scenario['os_cortex']) / 2
    new_scenario['osmotic_peri'] = -6E3
    new_scenario['osmotic_stele'] = -6E3
    new_scenario['osmotic_endo'] = (new_scenario['osmotic_peri'] + new_scenario['os_cortex']) / 2
    new_scenario['osmotic_xyl'] = -1.4E3
    new_scenario['pressure_xyl_prox'] = -4.7E3
    default_input.boundary.add_scenario(new_scenario)

    # stress night with suberin
    new_scenario = default_input.boundary.scenarios[0].copy()
    new_scenario['osmotic_left_soil'] = -7.5E3
    new_scenario['osmotic_right_soil'] = -7.5E3
    new_scenario['osmotic_epi'] = -8.0E3
    new_scenario['os_cortex'] = -7.6E3
    new_scenario['osmotic_exo'] = (new_scenario['osmotic_epi'] + new_scenario['os_cortex']) / 2
    new_scenario['osmotic_peri'] = -4E3
    new_scenario['osmotic_stele'] = -4E3
    new_scenario['osmotic_endo'] = (new_scenario['osmotic_peri'] + new_scenario['os_cortex']) / 2
    new_scenario['osmotic_xyl'] = -3.7E3
    new_scenario['pressure_xyl_prox'] = -2.5E3
    default_input.boundary.add_scenario(new_scenario)

    # stress day with suberin
    new_scenario = default_input.boundary.scenarios[3].copy()
    new_scenario['osmotic_epi'] = -8.5E3
    new_scenario['os_cortex'] = -7.5E3
    new_scenario['osmotic_exo'] = (new_scenario['osmotic_epi'] + new_scenario['os_cortex']) / 2
    new_scenario['osmotic_peri'] = -5E3
    new_scenario['osmotic_stele'] = -5E3
    new_scenario['osmotic_endo'] = (new_scenario['osmotic_peri'] + new_scenario['os_cortex']) / 2
    new_scenario['osmotic_xyl'] = -3.6E3
    new_scenario['pressure_xyl_prox'] = -8.0E3
    default_input.boundary.add_scenario(new_scenario)

    network = NetworkBuilder(root)
    network.populate_from_network()

    m = Mecha(default_input, network=network)

    # 3. Solve
    print("--- Solving Hydraulic System ---")
    print(f"Number of scenarios to solve: {m.boundary.n_scenarios}")

    m.water_flux()

    print(f'Number of solutions in results: {len(m.results)}')

    _, ax = plt.subplots(2, 2, figsize=(10, 10))
    visualize(m, visu_type="flow", maturity_idx=1, scenario_idx=1, ax=ax[0, 0], show_plot=False)
    ax[0, 0].set_title("Flow - Mat xSub - Scenario night", color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    visualize(m, visu_type="flow", maturity_idx=1, scenario_idx=2, ax=ax[0, 1], show_plot=False)
    ax[0, 1].set_title("Flow - Mat xSub - Scenario day", color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    visualize(m, visu_type="flow", maturity_idx=0, scenario_idx=3, ax=ax[1, 0], show_plot=False)
    ax[1, 0].set_title("Flow - Mat Sub - Scenario night", color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    visualize(m, visu_type="flow", maturity_idx=0, scenario_idx=4, ax=ax[1, 1], show_plot=False)
    ax[1, 1].set_title("Flow - Mat Sub - Scenario day", color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    plt.tight_layout()
    plt.show()

    visualize(m, visu_type="paraview", prefix='outputs/waterpump')

    # add dash grey line y = 0
    _, ax = plt.subplots(1, 4, figsize=(15, 6))
    visualize(m, visu_type='psi_profile', maturity_idx=1, scenario_idx=1, ax=ax[0], show_plot=False)
    ax[0].set_title("Psi Profile - Mat xSub - Scenario night", color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    visualize(m, visu_type='psi_profile', maturity_idx=1, scenario_idx=2, ax=ax[1], show_plot=False)
    ax[1].set_title("Psi Profile - Mat xSub - Scenario day", color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    visualize(m, visu_type='psi_profile', maturity_idx=0, scenario_idx=3, ax=ax[2], show_plot=False)
    ax[2].set_title("Psi Profile - Mat Sub - Scenario night", color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    visualize(m, visu_type='psi_profile', maturity_idx=0, scenario_idx=4, ax=ax[3], show_plot=False)
    ax[3].set_title("Psi Profile - Mat Sub - Scenario day", color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    # add dash grey line y = 0
    ax[0].axhline(y=0, color='gray', linestyle=':')
    ax[1].axhline(y=0, color='gray', linestyle=':')
    ax[2].axhline(y=0, color='gray', linestyle=':')
    ax[3].axhline(y=0, color='gray', linestyle=':')
    # share the same x and y axis
    ax[1].sharex(ax[0])
    ax[2].sharex(ax[0])
    ax[3].sharex(ax[0])
    # max and min from both axes
    y_min = min(ax[0].get_ylim()[0], ax[1].get_ylim()[0], ax[2].get_ylim()[0], ax[3].get_ylim()[0])
    y_max = max(ax[0].get_ylim()[1], ax[1].get_ylim()[1], ax[2].get_ylim()[1], ax[3].get_ylim()[1])
    ax[0].set_ylim(y_min, y_max)
    ax[1].set_ylim(y_min, y_max)
    ax[2].set_ylim(y_min, y_max)
    ax[3].set_ylim(y_min, y_max)
    plt.tight_layout()
    plt.show()


def run_stress_osmotic_sweep() -> None:
    """Run 1D parameter sweep for osmotic_peri and osmotic_stele (set equal) under stress.

    This function sweeps osmotic_peri (and sets osmotic_stele equal to it) from -0.1 MPa to -0.8 MPa
    (which correspond to -1E3 hPa to -8E3 hPa in model units) for both the stress night (scenario 3)
    and stress day (scenario 4) conditions. It determines if and when positive water uptake (total flow > 0)
    is triggered for both maturity stages (stage 0: suberized, stage 1: non-suberized) and plots the
    results as 1D line plots.
    """
    root = build_tomato_root()

    # prepare default inputs for mecha
    default_input = InData()
    default_input.geometry.set_maturity_stages([6, 9])

    # Setup the base scenario (Scenario 0)
    base_scenario(default_input)

    # Define sweep values (-0.1 MPa to -0.8 MPa, i.e., -1000 to -8000 hPa)
    peri_vals = np.linspace(-1E3, -8E3, 11)

    scen_map_night = np.zeros((len(peri_vals)), dtype=int)
    scen_map_day = np.zeros((len(peri_vals)), dtype=int)

    # Populate the boundary scenarios with the parameter sweep grid
    for i, peri in enumerate(peri_vals):
        # stress night with suberin (derived from Scenario 3)
        night_scen = default_input.boundary.scenarios[0].copy()
        night_scen['osmotic_left_soil'] = -7.5E3
        night_scen['osmotic_right_soil'] = -7.5E3
        night_scen['osmotic_epi'] = -8.0E3
        night_scen['os_cortex'] = -7.6E3
        night_scen['osmotic_exo'] = (night_scen['osmotic_epi'] + night_scen['os_cortex']) / 2
        night_scen['osmotic_peri'] = peri
        night_scen['osmotic_stele'] = peri
        night_scen['osmotic_endo'] = (peri + night_scen['os_cortex']) / 2
        night_scen['osmotic_xyl'] = -3.7E3
        night_scen['pressure_xyl_prox'] = -2.5E3

        default_input.boundary.add_scenario(night_scen)
        scen_map_night[i] = default_input.boundary.n_scenarios - 1

        # stress day with suberin (derived from Scenario 4)
        day_scen = default_input.boundary.scenarios[0].copy()
        day_scen['osmotic_left_soil'] = -7.5E3
        day_scen['osmotic_right_soil'] = -7.5E3
        day_scen['osmotic_epi'] = -8.5E3
        day_scen['os_cortex'] = -7.5E3
        day_scen['osmotic_exo'] = (day_scen['osmotic_epi'] + day_scen['os_cortex']) / 2
        day_scen['osmotic_peri'] = peri
        day_scen['osmotic_stele'] = peri
        day_scen['osmotic_endo'] = (peri + day_scen['os_cortex']) / 2
        day_scen['osmotic_xyl'] = -3.6E3
        day_scen['pressure_xyl_prox'] = -8.0E3

        default_input.boundary.add_scenario(day_scen)
        scen_map_day[i] = default_input.boundary.n_scenarios - 1

    network = NetworkBuilder(root)
    network.populate_from_network()

    m = Mecha(default_input, network=network)

    print("--- Solving Sweep Scenarios ---")
    print(f"Number of scenarios to solve: {m.boundary.n_scenarios}")
    # Solves all scenarios for the maturity stages
    m.water_flux(verbose=False)

    flow_grid_night_mat0 = np.zeros((len(peri_vals)))
    flow_grid_night_mat1 = np.zeros((len(peri_vals)))
    flow_grid_day_mat0 = np.zeros((len(peri_vals)))
    flow_grid_day_mat1 = np.zeros((len(peri_vals)))

    for i in range(len(peri_vals)):
        scen_night = scen_map_night[i]
        scen_day = scen_map_day[i]

        flow_grid_night_mat0[i] = m.total_flow[0][scen_night]
        flow_grid_night_mat1[i] = m.total_flow[1][scen_night]

        flow_grid_day_mat0[i] = m.total_flow[0][scen_day]
        flow_grid_day_mat1[i] = m.total_flow[1][scen_day]

    # Convert hPa to MPa for printing and plotting
    peri_m = peri_vals / 1E4

    print("\n=== Parameter Sweep Results: Night Stress ===")
    for mat_idx, label in [(0, "Mat Sub (Maturity Stage 0)"), (1, "Mat xSub (Maturity Stage 1)")]:
        flow_grid = flow_grid_night_mat0 if mat_idx == 0 else flow_grid_night_mat1
        pos_indices = np.where(flow_grid > 0)
        if len(pos_indices[0]) > 0:
            print(f"Positive water uptake triggered for {label} under Night Stress!")
            peri_pos = peri_m[pos_indices[0]]
            print(f"  Osmotic Peri range: {np.max(peri_pos):.2f} to {np.min(peri_pos):.2f} MPa (more negative)")
        else:
            print(f"No combinations triggered positive water uptake for {label} under Night Stress.")

    print("\n=== Parameter Sweep Results: Day Stress ===")
    for mat_idx, label in [(0, "Mat Sub (Maturity Stage 0)"), (1, "Mat xSub (Maturity Stage 1)")]:
        flow_grid = flow_grid_day_mat0 if mat_idx == 0 else flow_grid_day_mat1
        pos_indices = np.where(flow_grid > 0)
        if len(pos_indices[0]) > 0:
            print(f"Positive water uptake triggered for {label} under Day Stress!")
            peri_pos = peri_m[pos_indices[0]]
            print(f"  Osmotic Peri range: {np.max(peri_pos):.2f} to {np.min(peri_pos):.2f} MPa (more negative)")
        else:
            print(f"No combinations triggered positive water uptake for {label} under Day Stress.")

    # Plot results as line plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Night Stress Plot
    axes[0].plot(peri_m, flow_grid_night_mat0, 'o-', label="Mat Sub (Stage 0)", color='#E66101', linewidth=2)
    axes[0].plot(peri_m, flow_grid_night_mat1, 's-', label="Mat xSub (Stage 1)", color='#5E3C99', linewidth=2)
    axes[0].axhline(y=0.0, color='gray', linestyle='--', linewidth=1.5, label="Zero Flow")
    axes[0].set_title("Night Stress Condition Sweep", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Pericycle & Stele Osmotic Potential (MPa)", fontsize=10)
    axes[0].set_ylabel("Total Flow (cm³ d⁻¹)", fontsize=10)
    axes[0].legend(frameon=True, facecolor='white', edgecolor='none')
    axes[0].grid(True, linestyle=':', alpha=0.5)

    # Day Stress Plot
    axes[1].plot(peri_m, flow_grid_day_mat0, 'o-', label="Mat Sub (Stage 0)", color='#E66101', linewidth=2)
    axes[1].plot(peri_m, flow_grid_day_mat1, 's-', label="Mat xSub (Stage 1)", color='#5E3C99', linewidth=2)
    axes[1].axhline(y=0.0, color='gray', linestyle='--', linewidth=1.5, label="Zero Flow")
    axes[1].set_title("Day Stress Condition Sweep", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Pericycle & Stele Osmotic Potential (MPa)", fontsize=10)
    axes[1].set_ylabel("Total Flow (cm³ d⁻¹)", fontsize=10)
    axes[1].legend(frameon=True, facecolor='white', edgecolor='none')
    axes[1].grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_waterpump()
    # run_stress_osmotic_sweep()
