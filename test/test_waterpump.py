from mecha.mecha_class import Mecha
from mecha.utils.data_loader import InData
from mecha.utils.network_builder import NetworkBuilder
from granap.root_class import RootAnatomy
from mecha.utils.visu import visualize
from mecha.utils.network_export import export_to_graphml
import matplotlib.pyplot as plt
import numpy as np


def test_waterpump() -> None:
    """Run standard water pump scenarios and visualize radial profiles and flows.

    This function tests control night/day and stress night/day scenarios for different
    maturity stages of the root, and generates visual plots of flow and water potential profiles.
    """
    root = RootAnatomy()
    root.update_params("aerenchyma", "aerenchyma_proportion", 0.0)    
    
    # Generated a root anatomy without aerenchyma
    # prepare granap data structure for mecha
    root.export_to_adjencymatrix()

    # prepare default inputs for mecha
    default_input = InData()
    
    # control night no suberin
    default_input.geometry.set_maturity_stages([6,9])

    default_input.boundary.scenarios[0]['osmotic_left_soil'] = -1.5E3
    default_input.boundary.scenarios[0]['osmotic_right_soil']= -1.5E3
    default_input.boundary.scenarios[0]['osmotic_epi'] = -6.8E3
    default_input.boundary.scenarios[0]['osmotic_exo'] = -7.55E-3
    default_input.boundary.scenarios[0]['osmotic_endo'] = -4.80E3
    default_input.boundary.scenarios[0]['osmotic_peri'] = -5.5E3
    default_input.boundary.scenarios[0]['osmotic_stele']= -5.5E3
    default_input.boundary.scenarios[0]['osmotic_xyl'] = -1.40E3
    default_input.boundary.scenarios[0]['pressure_xyl_prox'] = -5.0E3
    default_input.boundary.scenarios[0]['osmotic_sieve'] = -1.0E4
    default_input.boundary.scenarios[0]['os_hetero'] = 4
    default_input.boundary.scenarios[0]['os_cortex'] = -8.3E3

    # control night no suberin
    new_scenario = default_input.boundary.scenarios[0].copy()
    new_scenario['osmotic_exo'] = (new_scenario['osmotic_epi']+new_scenario['os_cortex'])/2
    new_scenario['osmotic_peri'] = -5E3
    new_scenario['osmotic_stele']= -5E3
    new_scenario['osmotic_endo'] = (new_scenario['osmotic_peri']+new_scenario['os_cortex'])/2
    new_scenario['pressure_xyl_prox'] = -2.5E3
    default_input.boundary.add_scenario(new_scenario)
    # control day no suberin
    new_scenario = default_input.boundary.scenarios[0].copy()
    new_scenario['osmotic_epi'] = -7.5E3
    new_scenario['os_cortex'] = -6.8E3
    new_scenario['osmotic_exo'] = (new_scenario['osmotic_epi']+new_scenario['os_cortex'])/2
    new_scenario['osmotic_peri'] = -6E3
    new_scenario['osmotic_stele']= -6E3
    new_scenario['osmotic_endo'] = (new_scenario['osmotic_peri']+new_scenario['os_cortex'])/2
    new_scenario['osmotic_xyl'] = -1.4E3
    new_scenario['pressure_xyl_prox'] = -4.7E3
    default_input.boundary.add_scenario(new_scenario)

    # stress night with suberin
    new_scenario = default_input.boundary.scenarios[0].copy()
    new_scenario['osmotic_left_soil'] = -7.5E3
    new_scenario['osmotic_right_soil'] = -7.5E3
    new_scenario['osmotic_epi'] = -8.0E3
    new_scenario['os_cortex'] = -7.6E3
    new_scenario['osmotic_exo'] = (new_scenario['osmotic_epi']+new_scenario['os_cortex'])/2
    new_scenario['osmotic_peri'] = -4E3
    new_scenario['osmotic_stele']= -4E3
    new_scenario['osmotic_endo'] = (new_scenario['osmotic_peri']+new_scenario['os_cortex'])/2
    new_scenario['osmotic_xyl'] = -3.7E3
    new_scenario['pressure_xyl_prox'] = -2.5E3
    default_input.boundary.add_scenario(new_scenario)

    # stress day with suberin
    new_scenario = default_input.boundary.scenarios[3].copy()
    new_scenario['osmotic_epi'] = -8.5E3
    new_scenario['os_cortex'] = -7.5E3
    new_scenario['osmotic_exo'] = (new_scenario['osmotic_epi']+new_scenario['os_cortex'])/2
    new_scenario['osmotic_peri'] = -5E3
    new_scenario['osmotic_stele']= -5E3
    new_scenario['osmotic_endo'] = (new_scenario['osmotic_peri']+new_scenario['os_cortex'])/2
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
    visualize(m, visu_type="flow", maturity_idx= 1, scenario_idx=1, ax=ax[0,0], show_plot=False)
    ax[0,0].set_title("Flow - Mat xSub - Scenario night", color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    visualize(m, visu_type="flow", maturity_idx= 1, scenario_idx=2, ax=ax[0,1], show_plot=False)
    ax[0,1].set_title("Flow - Mat xSub - Scenario day", color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    visualize(m, visu_type="flow", maturity_idx= 0, scenario_idx=3, ax=ax[1,0], show_plot=False)
    ax[1,0].set_title("Flow - Mat Sub - Scenario night", color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    visualize(m, visu_type="flow", maturity_idx= 0, scenario_idx=4, ax=ax[1,1], show_plot=False)
    ax[1,1].set_title("Flow - Mat Sub - Scenario day", color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    plt.tight_layout()
    plt.show()

    visualize(m, visu_type="paraview", prefix='outputs/waterpump')

    # add dash grey line y = 0
    _, ax = plt.subplots(1, 4, figsize=(15, 6))
    visualize(m, visu_type='psi_profile', maturity_idx= 1, scenario_idx=1, ax=ax[0], show_plot=False)
    ax[0].set_title("Psi Profile - Mat xSub - Scenario night", color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    visualize(m, visu_type='psi_profile', maturity_idx= 1, scenario_idx=2, ax=ax[1], show_plot=False)
    ax[1].set_title("Psi Profile - Mat xSub - Scenario day", color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    visualize(m, visu_type='psi_profile', maturity_idx= 0, scenario_idx=3, ax=ax[2], show_plot=False)
    ax[2].set_title("Psi Profile - Mat Sub - Scenario night", color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    visualize(m, visu_type='psi_profile', maturity_idx= 0, scenario_idx=4, ax=ax[3], show_plot=False)
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
    #max and min from both axes
    y_min = min(ax[0].get_ylim()[0], ax[1].get_ylim()[0], ax[2].get_ylim()[0], ax[3].get_ylim()[0])
    y_max = max(ax[0].get_ylim()[1], ax[1].get_ylim()[1], ax[2].get_ylim()[1], ax[3].get_ylim()[1])
    ax[0].set_ylim(y_min, y_max)
    ax[1].set_ylim(y_min, y_max)
    ax[2].set_ylim(y_min, y_max)
    ax[3].set_ylim(y_min, y_max)
    plt.tight_layout()
    plt.show()


def test_stress_osmotic_sweep() -> None:
    """Run 1D parameter sweep for osmotic_peri and osmotic_stele (set equal) under stress.

    This function sweeps osmotic_peri (and sets osmotic_stele equal to it) from -0.5 MPa to -1.5 MPa
    (which correspond to -5E3 hPa to -15E3 hPa in model units) for both the stress night (scenario 3)
    and stress day (scenario 4) conditions. It determines if and when positive water uptake (total flow > 0)
    is triggered for both maturity stages (stage 0: suberized, stage 1: non-suberized) and plots the
    results as 1D line plots.
    """
    root = RootAnatomy()
    root.update_params("aerenchyma", "aerenchyma_proportion", 0.0)    
    root.export_to_adjencymatrix()

    # prepare default inputs for mecha
    default_input = InData()
    default_input.geometry.set_maturity_stages([6,9])

    # Setup the base scenario (Scenario 0)
    default_input.boundary.scenarios[0]['osmotic_left_soil'] = -1.5E3
    default_input.boundary.scenarios[0]['osmotic_right_soil']= -1.5E3
    default_input.boundary.scenarios[0]['osmotic_epi'] = -6.8E3
    default_input.boundary.scenarios[0]['osmotic_exo'] = -7.55E-3
    default_input.boundary.scenarios[0]['osmotic_endo'] = -4.80E3
    default_input.boundary.scenarios[0]['osmotic_peri'] = -5.5E3
    default_input.boundary.scenarios[0]['osmotic_stele']= -5.5E3
    default_input.boundary.scenarios[0]['osmotic_xyl'] = -1.40E3
    default_input.boundary.scenarios[0]['pressure_xyl_prox'] = -5.0E3
    default_input.boundary.scenarios[0]['osmotic_sieve'] = -1.0E4
    default_input.boundary.scenarios[0]['os_hetero'] = 4
    default_input.boundary.scenarios[0]['os_cortex'] = -8.3E3

    # Define sweep values (-0.5 MPa to -1.5 MPa, i.e., -5000 to -15000 hPa)
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
    test_waterpump()
    # test_stress_osmotic_sweep()