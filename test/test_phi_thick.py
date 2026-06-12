from openalea.mecha.mecha_class import Mecha
from openalea.mecha.utils.data_loader import InData
from openalea.mecha.utils.network_builder import NetworkBuilder
from openalea.granap.root_class import RootAnatomy
from openalea.mecha.utils.visu import visualize
import matplotlib.pyplot as plt



def test_phi_thick():
    root = RootAnatomy()
    root.update_params("aerenchyma", "aerenchyma_proportion", 0.0)    
    
    # Generated a root anatomy without aerenchyma
    # prepare granap data structure for mecha
    root.export_to_adjencymatrix()

    # prepare default inputs for mecha
    default_input = InData()
    default_input.geometry.set_maturity_stages([1,8])

    network = NetworkBuilder(root)
    network.n_phi_layers = 2
    network.phi_type = 3
    network.populate_from_network()
    
    m = Mecha(default_input, network=network)
    
    # 3. Solve
    print("--- Solving Hydraulic System ---")
    print(f"Number of scenarios to solve: {m.boundary.n_scenarios}")

    m.water_flux()

    _, ax = plt.subplots(1, 2, figsize=(15, 6))
    visualize(m, visu_type="flow", ax=ax[0], maturity_idx= 0, scenario_idx="standard water flow", show_plot=False)
    visualize(m, visu_type="flow", ax=ax[1], maturity_idx= 1, scenario_idx="standard water flow", show_plot=False)
    # add Main title for the figure
    plt.suptitle("$\Phi$ Thickening Type III", color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    # Add title for the figure
    ax[0].set_title("Flow - Maturity $En_{cs}$ - Std Water Flow", color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    ax[1].set_title("Flow - Maturity $En_{sub} & Ex_{sub}$ - Std Water Flow", color='black', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    test_phi_thick()
