
from granap.needle_class import NeedleAnatomy
from mecha.utils.data_loader import InData
from mecha.mecha_class import Mecha
from mecha.utils.network_builder import NetworkBuilder
from mecha.utils.visu import visualize


needle_anatomy = NeedleAnatomy()
_ = needle_anatomy.export_to_adjencymatrix()

default_input = InData()
# new barrier to implement
default_input.geometry.set_maturity_stages([10]) # needle barriers = Endodermis with Casparian strip, epidermis and hypodermis lignified

# update the default_input
default_input.hydraulic.xcontactrange = [5.0E8] # should target air space instead of epidermis

default_input.boundary.scenarios[0]['psi_left_soil'] = -15E3 # psi air
default_input.boundary.scenarios[0]['psi_right_soil']= -15E3 # psi air
default_input.boundary.scenarios[0]['psi_air'] = -15E3
default_input.boundary.scenarios[0]['osmotic_epi'] = -6.8E3
default_input.boundary.scenarios[0]['osmotic_exo'] = -7.55E-3
default_input.boundary.scenarios[0]['osmotic_endo'] = -4.80E3
default_input.boundary.scenarios[0]['osmotic_peri'] = -5.5E3  # osmotic of transfusion parenchyma
default_input.boundary.scenarios[0]['osmotic_stele']= -5.5E3
default_input.boundary.scenarios[0]['osmotic_xyl'] = -1.40E3
default_input.boundary.scenarios[0]['pressure_xyl_prox'] = -5.0E3
default_input.boundary.scenarios[0]['osmotic_sieve'] = -1.0E4
default_input.boundary.scenarios[0]['os_hetero'] = 4
default_input.boundary.scenarios[0]['os_cortex'] = -8.3E3  # osmotic of mesophyll

# control night no suberin
new_scenario = default_input.boundary.scenarios[0].copy()
new_scenario['osmotic_exo'] = (new_scenario['osmotic_epi']+new_scenario['os_cortex'])/2
new_scenario['osmotic_peri'] = -5E3
new_scenario['osmotic_stele']= -5E3
new_scenario['osmotic_endo'] = (new_scenario['osmotic_peri']+new_scenario['os_cortex'])/2
new_scenario['pressure_xyl_prox'] = -2.5E3
default_input.boundary.add_scenario(new_scenario)

needle_network = NetworkBuilder(needle_anatomy)
needle_network.populate_from_network()

needle_transpiration = Mecha(default_input, network=needle_network)
# to implement 
 
# 1. Set boundary water potential (in air space similarly to xylem water potential but lower water potential) 
# 2. no boundary condition assign to the epidermis/is_border=True edges/walls

# 3. Solve
print("--- Solving Hydraulic System ---")
print(f"Number of scenarios to solve: {needle_transpiration.boundary.n_scenarios}")

needle_transpiration.solve_W() # currently wrong but with water_flux it has an error
needle_transpiration.water_flux()
# ERROR:
# mecha_class.py", line 1640, in standard_transmembrane_fractions
#    row = int(self.network.rank_to_row[rank])
# ValueError: cannot convert float NaN to integer


# wrong direction of the flow
visualize(needle_transpiration, maturity_idx= 0, scenario_idx=1, visu_type="flow")

visualize(needle_transpiration, visu_type="paraview", prefix='outputs/needle')
