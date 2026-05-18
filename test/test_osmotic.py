
from mecha.mecha_class import Mecha
from mecha.utils.data_loader import InData
from mecha.utils.network_builder import NetworkBuilder
from granap.root_class import RootAnatomy
from mecha.utils.visu import visualize
from mecha.utils.network_export import export_to_graphml


def test_osmotic():
    root = RootAnatomy()
    root.update_params("aerenchyma", "aerenchyma_proportion", 0.0)    
    
    # Generated a root anatomy without aerenchyma
    # prepare granap data structure for mecha
    root.export_to_adjencymatrix()

    # prepare default inputs for mecha
    default_input = InData()
    default_input.geometry.set_maturity_stages([3])
    
    # Use the new helper to set osmotic scenarios (0, 1, 2, 3)
    # The first scenario (0) is kept as base, then 1, 2, 3 are added.
    default_input.boundary.set_os_hetero_scenarios([1])
    
    network = NetworkBuilder(root)
    network.populate_from_network()
    
    m = Mecha(default_input, network=network)
    
    # 3. Solve
    print("--- Solving Hydraulic System ---")
    print(f"Number of scenarios to solve: {m.boundary.n_scenarios}")

    m.water_flux()
    
    # Note on Q values: If you observe NaN fluxes, it is usually because the osmotic boundary 
    # conditions (like osmotic_sieve) were not fully defined for a scenario. 
    # The solver now includes robust nan_to_num(0.0) checks to ensure that 
    # incomplete scenarios still produce valid (hydrostatic-only) fluxes.

    visualize(m, visu_type='paraview', 
        prefix='outputs/osmotic',
        extrude_z= 50)

    export_to_graphml(m, "outputs/osmotic.graphml")

    visualize(m, visu_type='osmotic_profile')


test_osmotic()