
from openalea.mecha.utils import paraview_export
from openalea.mecha.mecha_class import Mecha
from openalea.mecha.utils.data_loader import InData
from openalea.mecha.utils.network_builder import NetworkBuilder
from openalea.granap.root_class import RootAnatomy
from openalea.mecha.utils.visu import visualize
from openalea.mecha.utils.network_export import export_to_graphml
import matplotlib.pyplot as plt


def test_osmotic():
    root = RootAnatomy()
    root.update_params("aerenchyma", "aerenchyma_proportion", 0.0)    
    
    # Generated a root anatomy without aerenchyma
    # prepare granap data structure for mecha
    root.export_to_adjencymatrix()

    # prepare default inputs for mecha
    default_input = InData()
    default_input.geometry.set_maturity_stages([1,8])
    
    # Use the new helper to set osmotic scenarios (0, 1, 2, 3)
    # The first scenario (0) is kept as base, then 1, 2, 3 are added.
    default_input.boundary.set_os_hetero_scenarios([0, 1, 2, 3])
    
    network = NetworkBuilder(root)
    network.populate_from_network()
    
    m = Mecha(default_input, network=network)
    
    # 3. Solve
    print("--- Solving Hydraulic System ---")
    print(f"Number of scenarios to solve: {m.boundary.n_scenarios}")

    m.water_flux()

    print(f'Number of solutions in results: {len(m.results)}')

    _, ax = plt.subplots(1, 2, figsize=(15, 6))
    visualize(m, visu_type="flow", maturity_idx= 0, scenario_idx="standard water flow", ax=ax[0], show_plot=False)
    ax[0].set_title("Flow - Mat 0 - Std Water Flow", color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    visualize(m, visu_type="flow", maturity_idx= 1, scenario_idx=1, ax=ax[1], show_plot=False)
    ax[1].set_title("Flow - Mat 1 - Scenario 1", color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    plt.tight_layout()
    plt.show()

    visualize(m, visu_type="flow_pathway", maturity_idx=0, scenario_idx=3, show_plot=True)
    visualize(m, visu_type="flow_pathway", maturity_idx=1, scenario_idx=3, show_plot=True)

    _, ax = plt.subplots(1, 2, figsize=(15, 6))
    visualize(m, visu_type="velocity", maturity_idx=0, scenario_idx="standard water flow", ax=ax[0], show_plot=False)
    ax[0].set_title("Velocity - Mat 0 - Std Water Flow", color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    visualize(m, visu_type="velocity", maturity_idx=1, scenario_idx=2, ax=ax[1], show_plot=False)
    ax[1].set_title("Velocity - Mat 1 - Scenario 2", color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    plt.tight_layout()
    plt.show()

    # visualize(m, visu_type="paraview", prefix='outputs/my_sim')

    # two subplots to compare osmotic scenarios
    _, ax = plt.subplots(1, 5, figsize=(20, 6))
    visualize(m, visu_type='psi_profile', maturity_idx= 0, scenario_idx="standard water flow", ax=ax[0], show_plot=False)
    ax[0].set_title("Psi Profile - Mat 0 - Std Water Flow", color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    visualize(m, visu_type='psi_profile', maturity_idx= 0, scenario_idx=1, ax=ax[1], show_plot=False)
    ax[1].set_title("Psi Profile - Mat 0 - Scenario 1", color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    visualize(m, visu_type='psi_profile', maturity_idx= 0, scenario_idx=2, ax=ax[2], show_plot=False)
    ax[2].set_title("Psi Profile - Mat 0 - Scenario 2", color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    visualize(m, visu_type='psi_profile', maturity_idx= 0, scenario_idx=3, ax=ax[3], show_plot=False)
    ax[3].set_title("Psi Profile - Mat 0 - Scenario 3", color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    visualize(m, visu_type='psi_profile', maturity_idx= 0, scenario_idx=4, ax=ax[4], show_plot=False)
    ax[4].set_title("Psi Profile - Mat 0 - Scenario 4", color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    # share the same x and y axis
    ax[1].sharex(ax[0])
    ax[2].sharex(ax[0])
    ax[3].sharex(ax[0])
    ax[4].sharex(ax[0])
    #max and min from both axes
    y_min = min(ax[0].get_ylim()[0], ax[1].get_ylim()[0], ax[2].get_ylim()[0], ax[3].get_ylim()[0], ax[4].get_ylim()[0])
    y_max = max(ax[0].get_ylim()[1], ax[1].get_ylim()[1], ax[2].get_ylim()[1], ax[3].get_ylim()[1], ax[4].get_ylim()[1])
    ax[0].set_ylim(y_min, y_max)
    ax[1].set_ylim(y_min, y_max)
    ax[2].set_ylim(y_min, y_max)
    ax[3].set_ylim(y_min, y_max)
    ax[4].set_ylim(y_min, y_max)
    plt.tight_layout()
    plt.show()


test_osmotic()
