from mecha.utils.network_export import export_to_graphml
from mecha.mecha_class import Mecha
from mecha.utils.data_loader import InData
from mecha.utils.network_builder import NetworkBuilder
from granap.root_class import RootAnatomy
from mecha.utils.visu import visualize
import matplotlib.pyplot as plt
import numpy as np

def test_all_visu_types():
    # 1. Setup root anatomy
    print("--- Setting up Root Anatomy ---")
    root = RootAnatomy()
    # Add some complexity: aerenchyma and intercellular spaces
    root.update_params("aerenchyma", "aerenchyma_proportion", 0.0)
    root.update_params("inter_cellular_spaces", "tissue", "cortex")
    root.export_to_adjencymatrix()

    # 2. Setup MECHA
    print("--- Setting up MECHA ---")
    default_input = InData()
    # We'll use two maturity stages
    default_input.geometry.set_maturity_stages([1,3])
    
    network = NetworkBuilder(root)
    network.populate_from_network()
    
    m = Mecha(default_input, network=network)
    
    # 3. Solve
    print("--- Solving Hydraulic System ---")
    # Standard solve (maturity 1)
    m.solve_W(h=0, i_maturity=1)

    # 4. Test Visualizations
    # We use plt.show() which might be interactive. 
    # For a "test" script, we might want to just check if they run without error.
    # But since the user asked to "make" the test, I'll include all types.
    
    print("\n--- Testing Visualization Types ---")
    
    visu_types = [
        "paraview",
        "polygon",
        "network",
        "water_potential",
        "conductance",
        "psi_profile",
        "flow_pathway",
        "flow"
    ]
    
    for vt in visu_types:
        print(f"Testing '{vt}'...")
        try:
            # We use maturity_idx=1 (Mature zone)
            # scenario_idx='standard water flow' is the default
            if vt == 'paraview':
                visualize(m, visu_type=vt, maturity_idx=1, prefix='outputs/my_sim', extrude_z=50.0)
            elif vt == 'network':
                visualize(m, visu_type=vt, maturity_idx=1)
                visualize(m, visu_type=vt, prop_name='psi')
                export_to_graphml(m, "outputs/my_sim.graphml")
            else:
                visualize(m, visu_type=vt, maturity_idx=1)
        except Exception as e:
            print(f"  FAILED '{vt}': {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_all_visu_types()
