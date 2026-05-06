
from granap import NeedleAnatomy
from mecha.utils.data_loader import InData
from mecha.mecha_class import Mecha
from mecha.utils.network_builder import NetworkBuilder
from mecha.utils.visu import visualize


needle_anatomy = NeedleAnatomy()
_ = needle_anatomy.export_to_adjencymatrix()

default_input = InData()
default_input.geometry.set_maturity_stages([1])


ganache_network = NetworkBuilder(needle_anatomy)
ganache_network.populate_from_network()

  
needle_transpiration = Mecha(default_input, network=ganache_network)
# to implement
# solve_mecha(mode="transpiration") should do the solve_W but with transpiration condition. 
# this means:  
# 1. Set boundary water potential (in air space similarly to xylem water potential but lower water potential) 
# 2. no boundary condition assign to the epidermis/is_border=True edges/walls
needle_transpiration.solve_mecha(mode="transpiration") 

visualize(needle_transpiration, plot_type="water potential")

# to implement
# it will export the geometry and the water potential to a vtp file
# for paraview
# should add an option to save the flow Q as well.
# should add an option to save the pressure P as well.
# should add an option to save the osmotic potential as well.
# should add an option to save K_computed on edges.
visualize(needle_transpiration, plot_type="paraview", save_filename="transpiration_paraview.vtp")
