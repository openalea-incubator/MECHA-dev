
from py_compile import main
import numpy as np 
from numpy import genfromtxt #Load data from a text file, with missing values handled as specified.
from numpy.random import *  # for random sampling
import scipy.linalg as slin #Linear algebra functions
import pylab #Found in the package pyqt
from pylab import *  # for plotting
import networkx as nx
import sys, os
import re
from pylab import *  # for plotting
import argparse # for command-line argument parsing


from mecha.utils.data_loader import *
from mecha.utils.network_builder import *
from mecha.utils.prepare_paraview import prepare_geometrical_properties
from mecha.hydraulic_solver import HydraulicMatrixBuilder
from granap.network_base import AbstractNetwork

class Mecha:
    """Main class of the library, encodes a hydraulic anatomy to solve.
    """

    def __init__(self, all_input: Optional[InData] = None, network: Optional['NetworkBuilder'] = None):
        """Initialize the Mecha class.

        Parameters
        ----------
        all_input : InData, optional
            Input data containing all configurations.
        network : NetworkBuilder, optional
            Pre-populated NetworkBuilder (e.g. from GRANAP Organ).
        """
        if all_input is not None:
            self.all_input = all_input
        else:
            self.all_input = None

        if self.all_input is not None:
            self.boundary = self.all_input.boundary
            self.general = self.all_input.general
            self.geometry = self.all_input.geometry
            self.hormones = self.all_input.hormones
            self.hydraulic = self.all_input.hydraulic
            self.cellset_data = self.all_input.cellset_data
        else:
            self.boundary = None
            self.general = None
            self.geometry = None
            self.hormones = None
            self.hydraulic = None
            self.cellset_data = None

        self.results = []
        self.standardized_results = []
        self.hydraulic_conductivities = {}

        if network is None:
            self.network = NetworkBuilder()
            if self.all_input is not None:
                self._build_anatomy()
        else:
            self.network = network
            self.network.populate_from_network()
            self._initialize_from_network()

        if self.all_input is not None:
            self._count_surrounding_cells()
            self._set_hydraulics()

    
    @property
    def _details(self, verbose: bool = True):
        """Sums up the characteristics of the anatomy.

        Returns
        -------
        str
            A printable description of the anatomy.

        """

        description = "=== Mecha Configuration ===\n\n"

        if self.all_input is not None:
            description += self.all_input.info(verbose = False)

        if verbose:
            print(description)
        else:
            return description

    def _build_anatomy(self):
        """Build the anatomical network."""
        self.network.build_network(self.general, self.geometry, self.cellset_data)
        self.position=nx.get_node_attributes(self.network.graph,'position') #Updates nodes XY positions (micrometers)
        self.indice=nx.get_node_attributes(self.network.graph,'indice') #Node indices (walls, junctions and cells)
        self.geo_props = prepare_geometrical_properties(self.general, self.network, self.position, self.indice)
        if self.general.apo_contagion==2:
            self._initialize_apo_j_zombies0()
        
    def _initialize_from_network(self):
        """Initialize Mecha when using an external network (e.g. GRANAP Organ)."""
        self.position = nx.get_node_attributes(self.network.graph, 'position')
        self.indice = nx.get_node_attributes(self.network.graph, 'indice')

        # prepare_geometrical_properties requires general
        if self.general is not None and self.hormones is not None:
            self.geo_props = prepare_geometrical_properties(
                self.general, self.network,
                self.position, self.indice
            )
        else:
            self.geo_props = None

    
    def _set_hydraulics(self) -> None:
        """Set up hydraulic properties and solution arrays."""
        # Initialize dimensions
        n_maturity = self.geometry.n_maturity
        n_scenarios = self.boundary.n_scenarios
        r_discret = self._get_r_discret()

        # Initialize solution arrays
        
        self._initialize_flow_arrays(n_maturity, n_scenarios) # Q_tot, kr_tot
        self._initialize_tropism_arrays(n_maturity, n_scenarios) # hydrotropism, hydropatterning
        self._initialize_osmotic_arrays(n_maturity, n_scenarios)# os, s, elong

        self._initialize_stf_arrays(r_discret, n_maturity)
        self._initialize_layer_arrays(r_discret, n_maturity, n_scenarios)
        self._initialize_pressure_arrays(r_discret, n_maturity, n_scenarios)
        self.edge_flux_list = [[[] for _ in range(n_scenarios)] for _ in range(n_maturity)]

        # Set initial conditions for each maturity stage
        self._set_maturity_initial_conditions()

    def _get_r_discret(self) -> int:
        """Get the radial discretization value."""
        if self.network.r_discret[0] is not None:
            return int(self.network.r_discret[0])
        return 10  # Default value

    def _initialize_xylem_arrays(self, n_maturity: int, n_scenarios: int) -> None:
        """Initialize xylem-related arrays."""
        self.psi_xyl = np.empty((2,n_maturity, n_scenarios))
        self.psi_xyl[:] = np.nan

        self.dpsi_xyl = np.empty((n_maturity, n_scenarios))
        self.dpsi_xyl[:] = np.nan

        self.i_equil_xyl = np.nan  # Index of the equilibrium root xylem pressure scenario

        # Initialize with one extra row for total flow
        self.distributed_flow_xyl = np.empty((2, len(self.network.xylem_cells) + 1, n_scenarios))
        self.distributed_flow_xyl[:] = np.nan
        
        for i in range(n_scenarios):
            self.psi_xyl[1,:, i] = self.boundary.scenarios[i].get("pressure_xyl_prox")
            self.psi_xyl[0,:, i] = self.boundary.scenarios[i].get("pressure_xyl_dist")
            self.dpsi_xyl[:, i] = self.boundary.scenarios[i].get("delta_p_xyl")

            # Set initial xylem flow rate if available
            if self.boundary.scenarios[i].get('flowrate_prox'):
                flowrate_prox = self.boundary.scenarios[i].get("flowrate_prox")
                flowrate_dist = self.boundary.scenarios[i].get("flowrate_dist")
                if flowrate_prox is not None:
                    self.distributed_flow_xyl[1, 0, i] = float(flowrate_prox)
                if flowrate_dist is not None:
                    self.distributed_flow_xyl[0, 0, i] = float(flowrate_dist)


    def _initialize_phloem_arrays(self, n_maturity: int, n_scenarios: int) -> None:
        """Initialize phloem-related arrays."""
        self.psi_sieve = np.empty((2,n_maturity, n_scenarios))
        self.psi_sieve[:] = np.nan

        self.dpsi_sieve = np.empty((n_maturity, n_scenarios))
        self.dpsi_sieve[:] = np.nan

        self.i_equil_sieve = np.nan  # Index of the equilibrium root phloem pressure scenario

        # Initialize with one extra row for total flow
        self.distributed_flow_sieve = np.empty((2, self.network.n_sieve + 1, n_scenarios))
        self.distributed_flow_sieve[:] = np.nan

        for i_scenario in range(n_scenarios):
            self.psi_sieve[1, :, i_scenario] = self.boundary.scenarios[i_scenario].get("pressure_sieve_prox")
            self.psi_sieve[0, :, i_scenario] = self.boundary.scenarios[i_scenario].get("pressure_sieve_dist")
            self.dpsi_sieve[:, i_scenario] = self.boundary.scenarios[i_scenario].get("delta_p_sieve")

            # Set initial phloem flow rate if available
            if self.boundary.scenarios[i_scenario].get('flowrate_prox'):
                flowrate_prox = self.boundary.scenarios[i_scenario].get("flowrate_prox")
                flowrate_dist = self.boundary.scenarios[i_scenario].get("flowrate_dist")
                if flowrate_prox is not None:
                    self.distributed_flow_sieve[1, 0, i_scenario] = float(flowrate_prox)
                if flowrate_dist is not None:
                    self.distributed_flow_sieve[0, 0, i_scenario] = float(flowrate_dist)

    def _initialize_osmotic_arrays(self,n_maturity: int, n_scenarios: int) -> None:
        """Initialize osmotic-related arrays."""
        self.os_sieve = np.empty((n_maturity, n_scenarios))
        self.os_cortex = np.empty((n_maturity, n_scenarios))
        self.os_hetero = np.empty((n_maturity, n_scenarios))
        self.s_factor = np.empty((n_maturity, n_scenarios))
        self.s_hetero = np.empty((n_maturity, n_scenarios))
        self.elong_cell = np.empty((n_maturity, n_scenarios))
        self.elong_cell_side_diff = np.empty((n_maturity, n_scenarios))

        self.boundary.get_osmotic_potentials()
        self.boundary.get_reflection_coefficients()

    def _initialize_layer_arrays(self, r_discret: int, n_maturity: int, n_scenarios: int) -> None:
        """Initialize layer-based arrays."""
        self.uptake_layer_plus = np.zeros((r_discret, n_maturity, n_scenarios))
        self.uptake_layer_minus = np.zeros((r_discret, n_maturity, n_scenarios))
        self.flow_xyl_layer = np.zeros((r_discret, n_maturity, n_scenarios))
        self.flow_sieve_layer = np.zeros((r_discret, n_maturity, n_scenarios))
        self.flow_elong_layer = np.zeros((r_discret, n_maturity, n_scenarios))

    def _initialize_stf_arrays(self, r_discret: int, n_maturity: int) -> None:
        """Initialize STF (Specific Tissue Function) arrays."""
        self.stf_mb = np.zeros((self.network.n_membrane, n_maturity))
        self.stf_cell_plus = np.zeros((self.network.n_cells, n_maturity))
        self.stf_cell_minus = np.zeros((self.network.n_cells, n_maturity))
        self.stf_layer_plus = np.zeros((r_discret, n_maturity))
        self.stf_layer_minus = np.zeros((r_discret, n_maturity))

    def _initialize_pressure_arrays(self, r_discret: int, n_maturity: int, n_scenarios: int) -> None:
        """Initialize pressure and osmotic arrays."""
        self.psi_cell_layer = np.zeros((r_discret, n_maturity, n_scenarios))
        self.psi_wall_layer = np.zeros((r_discret, n_maturity, n_scenarios))
        self.os_cell_layer = np.zeros((r_discret, n_maturity, n_scenarios))
        self.n_os_cell_layer = np.zeros((r_discret, n_maturity, n_scenarios))
        self.os_wall_layer = np.zeros((r_discret, n_maturity, n_scenarios))
        self.n_os_wall_layer = np.zeros((r_discret, n_maturity, n_scenarios))
        self.n_wall_layer = np.zeros((r_discret, n_maturity, n_scenarios))

    def _initialize_flow_arrays(self, n_maturity: int, n_scenarios: int) -> None:
        """Initialize flow and conductivity arrays."""
        self.total_flow = np.zeros((n_maturity, n_scenarios))
        self.kr_tot = np.zeros((n_maturity, 1))

    def _initialize_tropism_arrays(self, n_maturity: int, n_scenarios: int) -> None:
        """Initialize tropism arrays."""
        self.hydropatterning = np.empty((n_maturity, n_scenarios))
        self.hydropatterning[:] = np.nan
        self.hydrotropism = np.empty((n_maturity, n_scenarios))
        self.hydrotropism[:] = np.nan

    def _initialize_apo_j_zombies0(self, use_thick: bool = True) -> None:
        """Initialize Apo_j_Zombies0 / Apo_j_cc arrays."""
        if use_thick:
            for j in range(self.network.n_walls, self.network.n_wall_junction):
                j_idx = j - self.network.n_walls
                for cid in self.network.junction_wall_cell[j_idx]:
                    if isnan(cid):
                        continue
                    cell_index = int(cid - self.network.n_wall_junction)
                    if cell_index in self.hormones.apo_zombie0:
                        cc = self.hormones.apo_cc[self.hormones.apo_zombie0.index(cell_index)]
                        if j not in self.network.apo_j_zombies0:
                            self.network.apo_j_zombies0.append(j)
                            self.network.apo_j_cc.append(cc)

    def _set_maturity_initial_conditions(self) -> None:
        """Set initial conditions for each maturity stage."""

        n_scenarios = len(self.boundary.scenarios)
        n_maturity = len(self.geometry.maturity_stages)
        self._initialize_xylem_arrays(n_maturity, n_scenarios) # psi_xyl, dpsi_xyl, flow_xyl
        self._initialize_phloem_arrays(n_maturity, n_scenarios) # psi_sieve, dpsi_sieve, flow_sieve

        for i_maturity, maturity in enumerate(self.geometry.maturity_stages):

            barrier = int(maturity.get("barrier"))
            height = float(maturity.get("height"))
            self._set_hydraulic_conductivities(i_maturity, barrier, height)

            if self.boundary.scenarios[0].get('flow_xyl_prox') is not None:
                for i_scenario, _ in enumerate(self.boundary.scenarios):
                    self._handle_xylem_flow_conditions(i_maturity, i_scenario)
                    self._handle_phloem_flow_conditions(i_maturity, i_scenario)
            

    def _handle_xylem_flow_conditions(self, i_maturity: int, i_scenario: int) -> None:
        """Handle xylem flow conditions."""
        if np.isnan(self.distributed_flow_xyl[1, 0, i_scenario]):
            return

        if np.isnan(self.psi_xyl[1, i_maturity, i_scenario]) and np.isnan(self.dpsi_xyl[i_maturity, i_scenario]):
            self._distribute_xylem_flow()
            if self.distributed_flow_xyl[1, 0, i_scenario] == 0.0:
                self.i_equil_xyl = 0
        else:
            print('Error: Cannot have both pressure and flow BC at xylem boundary')

    def _distribute_xylem_flow(self) -> None:
        """Distribute xylem flow proportionally to xylem cross-section area."""
        flow = self.distributed_flow_xyl[1, 0, 0]
        for i in range(self.network.n_xylem):
            self.distributed_flow_xyl[1, i+1, 0] = flow * self.network.xylem_area_ratio[i]

    def _handle_phloem_flow_conditions(self, i_maturity: int, i_scenario: int) -> None:
        """Handle phloem flow conditions."""
        if np.isnan(self.distributed_flow_sieve[1, 0, 0]):
            return

        if np.isnan(self.psi_sieve[1, i_maturity, i_scenario]) and np.isnan(self.dpsi_sieve[i_maturity, i_scenario]):
            self._distribute_phloem_flow()
            if self.distributed_flow_sieve[1, 0, 0] == 0.0:
                self.i_equil_sieve = 0
        else:
            print('Error: Cannot have both pressure and flow BC at phloem boundary')

    def _distribute_phloem_flow(self) -> None:
        """Distribute phloem flow proportionally to phloem cross-section area."""
        flow = self.distributed_flow_sieve[1, 0, 0]

        # Distribute flow
        for i in range(self.network.n_protosieve):
            self.distributed_flow_sieve[1, i+1, 0] = flow * self.network.phloem_area_ratio[i]

    def _set_hydraulic_conductivities(self, i_maturity: int, barrier: int, height: float) -> None:
        """Set cell wall hydraulic conductivity and plasmodesmatal conductance."""
        hydraulic = self.hydraulic

        # Loop through hydraulic scenarios (default is 1)
        for h in range(hydraulic.n_hydraulics):
            # Cell wall hydraulic conductivity
            kw =  hydraulic.get_kw_value(h)
            kw_barrier_casp, kw_barrier_sub = hydraulic.get_kw_barrier_values(h)

            # Set wall conductivities based on barrier type
            kw_config = hydraulic.get_wall_conductivities(barrier, kw, kw_barrier_casp, kw_barrier_sub)

            # Plasmodesmatal hydraulic conductance
            kpl_config = hydraulic.get_plasmodesmatal_conductance(h)

            # Contribution of aquaporins to membrane hydraulic conductivity
            kaqp_config = hydraulic.get_aquaporin_contributions(h)

            # Calculate parameter a for cortex
            a_cortex, b_cortex = self._calculate_cortex_parameters(height = height, kaqp_cortex = kaqp_config['kaqp_cortex'], hydraulic = hydraulic)

            # Store values in a dictionary
            self._set_hydraulic_conductivities_dict(h, i_maturity, barrier, height, kw_config, kpl_config, kaqp_config, a_cortex, b_cortex)


    def _calculate_cortex_parameters(self, height: float, kaqp_cortex: float, hydraulic: HydraulicData) -> tuple:
        """Calculate parameter a for cortex."""
        if hydraulic.ratio_cortex == 1:  # Uniform AQP activity in all cortex membranes
            a_cortex = 0.0  # (1/hPa/d)
            b_cortex = kaqp_cortex  # (cm/hPa/d)
        else:
            # Calculate total surface and other parameters
            tot_surf_cortex=0.0 #Total membrane exchange surface in cortical cells (square centimeters)
            temp=0.0 #Term for summation (cm3)
            for cell_group in self.network.cellset['cell_to_wall']: #Loop on cells. network.cellset['cell_to_wall'] contains cell wall groups info (one group by cell)
                cell_id = int(cell_group.getparent().get("id")) #Cell ID number
                for r in cell_group: #Loop for wall elements around the cell
                    wall_id= int(r.get("id")) #Cell wall ID
                    if self.network.graph.nodes[self.network.n_wall_junction + cell_id]['cgroup']==4: #Cortex
                        dist_cell=sqrt(square(self.position[wall_id][0]-self.position[self.network.n_wall_junction+cell_id][0])+square(self.position[wall_id][1]-self.position[self.network.n_wall_junction+cell_id][1])) #distance between wall node and cell node (micrometers)
                        surf=(height+dist_cell)*self.network.wall_lengths[wall_id]*1.0E-08 #(square centimeters)
                        temp+=surf*1.0E-04*(self.network.distance_center_grav[wall_id]+(hydraulic.ratio_cortex*self.network.distance_max_cortex-self.network.distance_min_cortex)/(1-hydraulic.ratio_cortex))
                        tot_surf_cortex+=surf
            a_cortex=kaqp_cortex*tot_surf_cortex/temp  #(1/hPa/d)
            b_cortex=a_cortex*1.0E-04*(hydraulic.ratio_cortex*self.network.distance_max_cortex-self.network.distance_min_cortex)/(1-hydraulic.ratio_cortex) #(cm/hPa/d)

        return a_cortex, b_cortex

    def _set_hydraulic_conductivities_dict(self, h: int, i_maturity: int, barrier: int, height: float, kw_config: np.ndarray, kpl_config: np.ndarray, kaqp_config: np.ndarray, a_cortex: float, b_cortex: float) -> None:
        """Set hydraulic conductivities in a dictionary."""
        self.hydraulic_conductivities[h, i_maturity, barrier] = {
            "kw": kw_config,
            "kpl": kpl_config,
            "kaqp": kaqp_config,
            "a_cortex": a_cortex,
            "b_cortex": b_cortex,
            "height": height,
        }



    def calculate_axial_conductance(self, i_maturity: int) -> tuple[np.ndarray, float]:
        """
        Calculates the axial conductances for a specific hydraulic scenario and maturity stage.
        
        Parameters:
        i_maturity: int, maturity stage index

        Returns:
        - K_axial: Array of axial conductances
        - K_xyl_spec: Specific axial conductance
        """

        hydraulic = self.hydraulic
        network = self.network

        barrier = self.geometry.maturity_stages[i_maturity].get('barrier')
        height = self.geometry.maturity_stages[i_maturity].get('height')


        #Axial conductances
        K_axial=np.zeros((network.n_cells + network.n_walls + network.n_junctions,1)) #Vector of apoplastic and plasmodesmatal axial conductances
        if barrier>0: 
            if hydraulic.axial_conductance_source==2:
                for K_xyl in hydraulic.k_xyl_elems:
                    cellnumber=int(K_xyl.get("id"))
                    K_axial[cellnumber+network.n_wall_junction]=float(K_xyl.get("value"))
                K_xyl_spec=sum(K_axial)*height/1.0E04
                for K_sieve in hydraulic.k_sieve_elems:
                    cellnumber=int(K_sieve.get("id"))
                    K_axial[cellnumber+network.n_wall_junction]=float(K_sieve.get("value"))
            else: #K_xyl_spec calculated from Poiseuille law (cm^3/hPa/d)
                for cid in network.xylem_cells:
                    K_axial[cid]=network.cell_areas[cid-network.n_wall_junction]**2/(8*3.141592*height*1.0E-05/3600/24)*1.0E-12 #(micron^4/micron)->(cm^3) & (1.0E-3 Pa.s)->(1.0E-05/3600/24 hPa.d) 
                K_xyl_spec=sum(K_axial)*height/1.0E04
                for cid in network.sieve_cells:
                    K_axial[cid]=network.cell_areas[cid-network.n_wall_junction]**2/(8*3.141592*height*1.0E-05/3600/24)*1.0E-12 #(micron^4/micron)->(cm^3) & (1.0E-3 Pa.s)->(1.0E-05/3600/24 hPa.d) 
        else: # barrier=0
            if hydraulic.axial_conductance_source==2:
                for K_sieve in hydraulic.k_sieve_elems:
                    cellnumber=int(K_sieve.get("id"))
                    if cellnumber+network.n_wall_junction in network.listprotosieve:
                        K_axial[cellnumber+network.n_wall_junction]=float(K_sieve.get("value"))
            else: #Calculated from Poiseuille law (cm^3/hPa/d)
                for cid in network.listprotosieve:
                    K_axial[cid]=network.cell_areas[cid-network.n_wall_junction]**2/(8*math.pi*height*1.0E-05/3600/24)*1.0E-12 #(micron^4/micron)->(cm^3) & (1.0E-3 Pa.s)->(1.0E-05/3600/24 hPa.d)

        return K_axial, K_xyl_spec

    def _count_surrounding_cells(self):
        """
        Count the number of surrounding cells of a given node.
        Uses geometry data when available, falls back to graph data from GRANAP.
        """
        n_walls = self.network.n_walls
        n_wall_junction = self.network.n_wall_junction

        # Get passage / intercellular cell IDs from geometry or graph
        if self.geometry is not None:
            passage_cell_ids = np.array(self.geometry.passage_cell_ids)
            intercellular_ids = np.array(self.geometry.intercellular_ids)
            xylem_pieces = self.geometry.xylem_pieces
        else:
            # GRANAP mode: look in graph for 'passage' and 'intercellular' cell_types
            passage_cell_ids = np.array(self.network.passage_cells)
            intercellular_ids = np.array(self.network.intercellular_cells)
            xylem_pieces = False

        # Ensure we have self.list_ghostwalls
        if not hasattr(self, 'list_ghostwalls'):
            self.list_ghostwalls = []

        for node, edges in self.network.graph.adjacency():
            i = self.indice[node]
            count_endo, count_stele_overall, count_exo = 0, 0, 0
            count_epi, count_cortex, count_passage = 0, 0, 0
            count_xyl = 0
            count_interC = 0
            if i < n_walls:
                for neighboor, eattr in edges.items():
                    if eattr['path'] == 'membrane':
                        neighbor_cell_id = self.indice[neighboor] - n_wall_junction
                        if len(intercellular_ids) > 0 and any(intercellular_ids == neighbor_cell_id):
                            count_interC += 1
                            if count_interC == 2 and i not in self.list_ghostwalls:
                                self.list_ghostwalls.append(i)
                        elif self.network.graph.nodes[neighboor]['cgroup'] in [13, 19, 20]:
                            count_xyl += 1
                            if (count_xyl == 2 and xylem_pieces) and i not in self.list_ghostwalls:
                                self.list_ghostwalls.append(i)
                        elif len(passage_cell_ids) > 0 and any(passage_cell_ids == neighbor_cell_id):
                            count_passage += 1
                        elif self.network.graph.nodes[neighboor]['cgroup'] == 3:  # Endodermis
                            count_endo += 1
                        elif self.network.graph.nodes[neighboor]['cgroup'] > 4:  # Pericycle or stele
                            count_stele_overall += 1
                        elif self.network.graph.nodes[neighboor]['cgroup'] == 4:  # Cortex
                            count_cortex += 1
                        elif self.network.graph.nodes[neighboor]['cgroup'] == 1:  # Exodermis
                            count_exo += 1
                        elif self.network.graph.nodes[neighboor]['cgroup'] == 2:  # Epidermis
                            count_epi += 1
            # Store counts as node attributes
            self.network.graph.nodes[node]['count_endo'] = count_endo
            self.network.graph.nodes[node]['count_stele_overall'] = count_stele_overall
            self.network.graph.nodes[node]['count_exo'] = count_exo
            self.network.graph.nodes[node]['count_epi'] = count_epi
            self.network.graph.nodes[node]['count_cortex'] = count_cortex
            self.network.graph.nodes[node]['count_passage'] = count_passage
            self.network.graph.nodes[node]['count_xyl'] = count_xyl
            self.network.graph.nodes[node]['count_interC'] = count_interC
    
    def build_matrices(self, h, i_maturity):
        """
        Builds the Doussan matrix (matrix_W) and convection/diffusion matrices for a specific
        hydraulic scenario (h) and maturity stage using HydraulicMatrixBuilder.
        """
        builder = HydraulicMatrixBuilder(
            network=self.network, geometry=self.geometry,
            boundary=self.boundary,
            hydraulic=self.hydraulic, hormones=self.hormones,
            general=self.general, geo_props=self.geo_props,
            position=self.position, indice=self.indice
        )
        return builder.build(
            h=h, i_maturity=i_maturity,
            hydraulic_conductivities=self.hydraulic_conductivities,
            boundary=self.boundary,
            psi_xyl=self.psi_xyl, psi_sieve=self.psi_sieve,
            distributed_flow_xyl=self.distributed_flow_xyl,
            distributed_flow_sieve=self.distributed_flow_sieve
        )


    @staticmethod
    def solve(matrix: np.ndarray, rhs: np.ndarray, sparse_matrix: int) -> tuple: 
        """Solve the system.

        Solve the system based on the provided configurations and network.
        """

        print("Solving system")
        t00 = time.perf_counter()  #Lasts about 35 sec in the 24mm root

        if sparse_matrix==1:
            matrix=matrix.tocsr()
            solution = spsolve(matrix,rhs)
            verification_1=np.allclose(np.dot(matrix,solution),rhs)
        else:
            solution = np.linalg.solve(matrix,rhs) #Solving the equation to get potentials inside the network
            verification_1=np.allclose(np.dot(matrix,solution),rhs)
        
        t01 = time.perf_counter()
        print(t01-t00, "seconds process time to solve system")
        return solution, verification_1

    def solve_all_W(self, h: int=0) -> tuple: 
        """Solve the hydraulic system for all maturity stages."""

        for i_maturity in range(self.geometry.n_maturity):
            self.solve_W(h = h, i_maturity = i_maturity)

    def compute_conductivities(self, h: int=0) -> tuple: 
        """Compute conductivities for all maturity stages."""

        self.root_hydraulic_properties = self.hydraulic.conductivities
        self.solve_all_W(h = h)
        idx = 0
        for i_maturity, maturity_stage in enumerate(self.geometry.maturity_stages):
            _, kx = self.calculate_axial_conductance(i_maturity = i_maturity)
            self.root_hydraulic_properties.append({'barrier': int(maturity_stage['barrier']), 
                            'height': float(maturity_stage['height']),
                            'kr': float(self.kr_tot[idx][0]),
                            'Kx': float(kx)})
            idx += 1 

    def elongation_BC(self, i_scenario: int, i_maturity: int) -> np.ndarray:
        """Calculate elongation boundary condition vector."""
        rhs_e = np.zeros((self.network.graph.number_of_nodes(), 1))
        
        # Unpack needed properties
        elong_cell = self.elong_cell[i_maturity][i_scenario]
        elong_side = self.elong_cell_side_diff[i_maturity][i_scenario]
        thickness = self.geometry.thickness
        x_rel = self.geo_props['x_rel']
        
        barrier = int(self.geometry.maturity_stages[i_maturity].get("barrier"))

        if barrier == 0:  # No elongation from the Casparian strip on
            for wall_id in range(self.network.n_walls):
                rhs_e[wall_id][0] = self.network.wall_lengths[wall_id] * thickness/2 * 1.0E-08 * \
                                    (elong_cell + (x_rel[wall_id] - 0.5) * elong_side) * \
                                    self.boundary.water_fraction_apo
            
            for cid in range(self.network.n_cells):
                node_idx = self.network.n_wall_junction + cid
                if self.network.cell_areas[cid] > self.network.cell_perimeters[cid] * thickness/2:
                    rhs_e[node_idx][0] = (self.network.cell_areas[cid] - self.network.cell_perimeters[cid] * thickness/2) * \
                                         1.0E-8 * (elong_cell + (x_rel[node_idx] - 0.5) * elong_side) * \
                                         self.boundary.water_fraction_sym
                else:
                    rhs_e[node_idx][0] = 0.0
                    
        return rhs_e
        
    def initialize_scenarios(self, i_scenario: int, i_maturity: int, Kmb: np.ndarray) -> tuple:
        """Initialize vectors and matrices for a specific scenario."""
        
        # Initialize vectors
        n_nodes = self.network.graph.number_of_nodes()
        rhs = np.zeros((n_nodes, 1))
        rhs_x = np.zeros((n_nodes, 1))
        rhs_p = np.zeros((n_nodes, 1))
        rhs_o = np.zeros((n_nodes, 1))
        
        # Osmotic potentials and reflection coefficients
        os_membranes = np.zeros((self.network.n_membrane, 2))
        s_membranes = np.zeros((self.network.n_membrane, 1))
        os_walls = np.zeros((self.network.n_walls, 1))
        os_cells = np.zeros((self.network.n_cells, 1))
        
        # Scenario parameters
        s_hetero = int(self.boundary.scenarios[i_scenario].get("s_hetero"))
        s_factor = float(self.boundary.scenarios[i_scenario].get("s_factor"))
        self.s_hetero[i_maturity][i_scenario] = s_hetero
        self.s_factor[i_maturity][i_scenario] = s_factor
        
        # Elongation parameters
        self.elong_cell[i_maturity][i_scenario] = float(self.boundary.scenarios[i_scenario].get("elongation_midpoint_rate"))
        self.elong_cell_side_diff[i_maturity][i_scenario] = float(self.boundary.scenarios[i_scenario].get("elongation_side_rate_difference"))
        
        # Reflection coefficients setup
        if s_hetero == 0:
            s_vals = {k: s_factor * 1.0 for k in ['epi', 'exo_epi', 'exo_cortex', 'cortex', 'endo_cortex', 
                                                 'endo_peri', 'peri', 'stele', 'comp', 'sieve']}
        elif s_hetero == 1:
            s_vals = {k: s_factor * 1.0 for k in ['epi', 'exo_epi', 'exo_cortex', 'cortex', 'endo_cortex']}
            s_vals.update({k: s_factor * 0.5 for k in ['endo_peri', 'peri', 'stele', 'comp', 'sieve']})
        elif s_hetero == 2:
            s_vals = {k: s_factor * 0.5 for k in ['epi', 'exo_epi', 'exo_cortex', 'cortex', 'endo_cortex']}
            s_vals.update({k: s_factor * 1.0 for k in ['endo_peri', 'peri', 'stele', 'comp', 'sieve']})
            
        # Osmotic potentials setup
        os_hetero = int(self.boundary.scenarios[i_scenario].get("os_hetero"))
        os_cortex = float(self.boundary.scenarios[i_scenario].get("os_cortex"))
        os_sieve = float(self.boundary.scenarios[i_scenario].get("osmotic_sieve"))
        self.os_hetero[i_maturity][i_scenario] = os_hetero
        self.os_cortex[i_maturity][i_scenario] = os_cortex
        self.os_sieve[i_maturity][i_scenario] = os_sieve
        
        # Determine specific osmotic values based on os_hetero

        vals = {}
        if os_hetero == 0:
            base = os_cortex
            vals = {k: base for k in ['epi', 'exo', 'endo', 'peri', 'stele'] + [f'c{i}' for i in range(1,9)]}
            vals['comp'] = (os_sieve + os_cortex)/2
        elif os_hetero == 1:
             vals = {'epi': -5000, 'exo': -5700, 'endo': -6200, 'peri': -5000, 'stele': -7400}
             vals.update({f'c{i}': v for i, v in enumerate([-6400, -7100, -7800, -8500, -9000, -9300, -9000, -8500], 1)})
             vals['comp'] = (os_sieve - 7400)/2
        elif os_hetero == 2:
             vals = {'epi': -11200, 'exo': -11500, 'endo': -10500, 'peri': -9200, 'stele': -12100}
             vals.update({f'c{i}': v for i, v in enumerate([-11800, -12100, -12400, -12700, -12850, -12950, -12850, -12700], 1)})
             vals['comp'] = (os_sieve - 12100)/2
        elif os_hetero == 3:
             base = os_cortex
             vals = {k: base for k in ['epi', 'exo'] + [f'c{i}' for i in range(1,9)]}
             vals.update({'endo': (base - 5000.0)/2.0, 'peri': -5000.0, 'stele': -5000.0})
             vals['comp'] = (os_sieve - 5000.0)/2
             
        # Extract props
        x_rel = self.geo_props['x_rel']
        r_rel = self.geo_props['r_rel']
        L_diff = self.geo_props['L_diff']
        passage_cell_ids = np.array(self.geometry.passage_cell_ids)
        
        # Calculate local soil/xyl osmotic potentials
        # Note: we calculate them during loop or pre-calculate? loop is easier for node dependence
        
        jmb = 0
        barrier = int(self.geometry.maturity_stages[i_maturity].get("barrier"))
        
        # Loop over network to fill Os_membranes and s_membranes and rhs_o
        # First calculate u (velocity) if c_flag
        u = np.zeros((2, 1))
        # Note: c_flag iterative optimization handles updating u and Os_membranes.
        # Here we just do the basic initialization/one-pass logic.
        
        scenario_data = self.boundary.scenarios[i_scenario]
        
        for node, edges in self.network.graph.adjacency():
            i = self.indice[node]
            if i < self.network.n_walls:
                # Calculate Os_soil_local and Os_xyl_local
                os_soil_local = 0.0
                os_xyl_local = 0.0
                
                if scenario_data['osmotic_symmetry_soil'] == 2:
                     if scenario_data['osmotic_diffusivity_soil'] == 0:
                         os_soil_local=float(scenario_data['osmotic_left_soil']+(scenario_data['osmotic_right_soil']-scenario_data['osmotic_left_soil'])*abs(r_rel[i])**scenario_data['osmotic_shape_soil'])
                     else:
                         if r_rel[i]>=0:
                             os_soil_local=scenario_data['osmotic_left_soil']*np.exp(u[0][0]*abs(r_rel[i])*L_diff[0]/scenario_data['osmotic_diffusivity_soil'])
                elif scenario_data['osmotic_symmetry_soil'] == 1:
                     os_soil_local=float(scenario_data['osmotic_left_soil']*(1-x_rel[i])+scenario_data['osmotic_right_soil']*x_rel[i])
                     
                if scenario_data['osmotic_symmetry_xyl'] == 2:
                     if scenario_data['osmotic_diffusivity_xyl'] == 0:
                         os_xyl_local=float(scenario_data['osmotic_endo']+(scenario_data['osmotic_xyl']-scenario_data['osmotic_endo'])*(1-abs(r_rel[i]))**scenario_data['osmotic_shape_xyl'])
                     else:
                         if r_rel[i]<0:
                             os_xyl_local=scenario_data['osmotic_xyl']*np.exp(-u[1][0]*abs(r_rel[i])*L_diff[1]/scenario_data['osmotic_diffusivity_xyl'])
                elif scenario_data['osmotic_symmetry_xyl'] == 1:
                     os_xyl_local=float((scenario_data['osmotic_xyl']+scenario_data['osmotic_endo'])/2)

                # unpack count neighbors
                count_epi = self.network.graph.nodes[node]['count_epi']
                count_endo = self.network.graph.nodes[node]['count_endo']
                count_stele_overall = self.network.graph.nodes[node]['count_stele_overall']
                count_passage = self.network.graph.nodes[node]['count_passage']
                count_cortex = self.network.graph.nodes[node]['count_cortex']

                # Second loop for membrane connections
                for neighboor, eattr in edges.items():
                    j = self.indice[neighboor]
                    if j > i and eattr['path'] == "membrane":
                        rank = int(self.network.cell_ranks[int(j - self.network.n_wall_junction)])
                        row = int(self.network.rank_to_row[rank])
                        
                        # Determine os_membranes and s_membranes
                       
                        cell_os = 0.0
                        wall_os = os_soil_local
                        sig = 0.0
                        
                        cgroup = self.network.graph.nodes[neighboor]['cgroup']
                        
                        if rank == 1: # Exodermis
                            cell_os = vals['exo']
                            sig = s_vals['exo_epi'] if count_epi == 1 else s_vals['exo_cortex']
                            wall_os = os_soil_local
                        elif rank == 2: # Epidermis
                            cell_os = vals['epi']
                            sig = s_vals['epi']
                            wall_os = os_soil_local
                        elif rank == 3: # Endodermis
                            cell_os = vals['endo']
                            sig = s_vals['endo_cortex']
                            if count_stele_overall > 0 and count_endo > 0:
                                sig = s_vals['endo_peri']
                                wall_os = os_xyl_local if barrier > 0 else os_soil_local
                            elif count_stele_overall == 0 and count_cortex > 0:
                                wall_os = os_soil_local
                            else: # Between endodermal cells
                                wall_os = os_xyl_local if barrier > 0 else os_soil_local
                                sig = s_vals['endo_peri']
                        elif 40 <= rank < 50: # Cortex
                            if int(j - self.network.n_wall_junction) in self.geometry.intercellular_ids:
                                cell_os = os_soil_local
                                wall_os = os_soil_local
                                sig = 0
                            else:
                                # Row mapping for cortex layers
                                c_idx = row - (self.network.row_outer_cortex - 8) # simplistic mapping
                                # Using explicit row check from main.py is safer but simplified here
                                c_key = f'c{max(1, min(8, 8 - (self.network.row_outer_cortex - row)))}'
                                cell_os = vals.get(c_key, vals.get('c1', 1))
                                sig = s_vals['cortex']
                                wall_os = os_soil_local
                        elif cgroup == 5: # Stele
                            cell_os = vals['stele']
                            sig = s_vals['stele']
                            wall_os = os_xyl_local if barrier > 0 else os_soil_local
                        elif rank == 16: # Pericycle
                            cell_os = vals['peri']
                            sig = s_vals['peri']
                            wall_os = os_xyl_local if barrier > 0 else os_soil_local
                        elif cgroup in [11, 23]: # Phloem
                            cell_os = os_sieve if not np.isnan(os_sieve) else vals['stele']
                            sig = s_vals['sieve']
                            wall_os = os_xyl_local if barrier > 0 else os_soil_local
                        elif cgroup in [12, 26]: # Companion
                            cell_os = vals['comp']
                            sig = s_vals['comp']
                            wall_os = os_xyl_local if barrier > 0 else os_soil_local
                        elif cgroup in [13, 19, 20]: # Xylem
                            if barrier == 0:
                                cell_os = vals['stele']
                                sig = s_vals['stele']
                                wall_os = os_soil_local
                            else:
                                cell_os = os_xyl_local
                                sig = 0.0
                                wall_os = os_xyl_local
                        
                        os_membranes[jmb][0] = wall_os
                        os_membranes[jmb][1] = cell_os
                        s_membranes[jmb] = sig
                        os_walls[i] = wall_os
                        os_cells[int(j - self.network.n_wall_junction)] = cell_os
                        
                        K = Kmb[jmb][0]
                        rhs_o[i] += K * sig * (wall_os - cell_os)
                        rhs_o[j] += K * sig * (cell_os - wall_os)
                        
                        jmb += 1

        # Calculate rhs_x (Xylem BC)
        if barrier > 0:
            psi_x = self.psi_xyl[1][i_maturity][i_scenario]
            flow_x = self.distributed_flow_xyl[1][0][i_scenario] if not np.isnan(self.distributed_flow_xyl[1][0][i_scenario]) else np.nan
            
            if not np.isnan(psi_x):
                for cid in self.network.xylem_cells:
                    rhs_x[cid][0] = -self.hydraulic.k_xyl
            elif not np.isnan(flow_x):
                 for i, cid in enumerate(self.network.xylem_cells):
                     rhs_x[cid][0] = self.distributed_flow_xyl[1][i+1][i_scenario]

        # Calculate rhs_p (Phloem BC)
        psi_p = self.psi_sieve[1][i_maturity][i_scenario]
        flow_p = self.distributed_flow_sieve[1][0][i_scenario] if not np.isnan(self.distributed_flow_sieve[1][0][i_scenario]) else np.nan
        
        target_sieve = self.network.protosieve_list if barrier == 0 else self.network.sieve_cells
        k_sieve = self.hydraulic.k_sieve
        
        if not np.isnan(psi_p):
            for cid in target_sieve:
                rhs_p[cid][0] = -k_sieve
        elif not np.isnan(flow_p):
            for i, cid in enumerate(target_sieve):
                 rhs_p[cid][0] = self.distributed_flow_sieve[1][i+1][i_scenario]

        return rhs, rhs_x, rhs_p, rhs_o

    def water_flux(self, h: int=0) -> tuple: 
        """Solve the hydraulic system for all maturity stages."""

        for i_maturity in range(self.geometry.n_maturity):
            solution, _, matrix_W, Kmb, rhs_s = self.solve_W(h = h, i_maturity = i_maturity)
            # Calculate standard transmembrane fractions
            self.standard_transmembrane_fractions(solution, i_maturity, Kmb)

            barrier = int(self.geometry.maturity_stages[i_maturity].get("barrier"))
            height_val = float(self.geometry.maturity_stages[i_maturity].get("height"))
            
            x_rel = self.geo_props['x_rel']

            for i_scenario in range(1,self.boundary.n_scenarios):
                rhs, rhs_x, rhs_p, rhs_o = self.initialize_scenarios(i_scenario, i_maturity, Kmb) # set and reset matrices for each scenario
                 
                # Elongation BC
                rhs_e = np.zeros_like(rhs)
                if barrier==0:
                    rhs_e = self.elongation_BC(i_scenario, i_maturity)
                    
                # Adding up all BCs
                rhs += rhs_e
                rhs += rhs_o
                
                # Soil BC
                # boundary.scenarios[count]['psi_soil_left']*(1-x_rel)+boundary.scenarios[count]['psi_soil_right']*x_rel
                psi_soil_left = self.boundary.scenarios[i_scenario].get('psi_soil_left', 0.0)
                psi_soil_right = self.boundary.scenarios[i_scenario].get('psi_soil_right', 0.0)
                
                psi_soil_profile = psi_soil_left * (1 - x_rel) + psi_soil_right * x_rel
                
                rhs += np.multiply(rhs_s, psi_soil_profile)

                # Xylem BC
                psi_xyl_val = self.psi_xyl[1][i_maturity][i_scenario]
                flow_xyl_val = self.distributed_flow_xyl[1][0][i_scenario]

                if barrier > 0:
                     if not np.isnan(psi_xyl_val): # Pressure BC
                         for cid in self.network.xylem_cells:
                             matrix_W[cid][cid] -= self.hydraulic.k_xyl
                         rhs += rhs_x * psi_xyl_val
                     elif not np.isnan(flow_xyl_val): # Flow BC
                         rhs += rhs_x

                # Phloem BC
                psi_sieve_val = self.psi_sieve[1][i_maturity][i_scenario]
                flow_sieve_val = self.distributed_flow_sieve[1][0][i_scenario]
                
                if barrier == 0:
                    if not np.isnan(psi_sieve_val):
                         for cid in self.network.protosieve_list:
                             matrix_W[cid][cid] -= self.hydraulic.k_sieve
                         rhs += rhs_p * psi_sieve_val
                    elif not np.isnan(flow_sieve_val):
                         rhs += rhs_p
                elif barrier > 0:
                    if not np.isnan(psi_sieve_val):
                         for cid in self.network.sieve_cells:
                             matrix_W[cid][cid] -= self.hydraulic.k_sieve
                         rhs += rhs_p * psi_sieve_val
                    elif not np.isnan(flow_sieve_val):
                         rhs += rhs_p
                
                # Solve Doussan equation, results in soln matrix 
                solution, _ = self.solve(matrix=matrix_W, rhs=rhs, sparse_matrix=self.general.sparse_matrix)
                self.results.append({'maturity stage': i_maturity, 'scenario': i_scenario, 
                                     'solution': solution, 'matrix_W': matrix_W, 'Kmb': Kmb, 'rhs': rhs})

                # Removing Xylem and phloem BC terms
                self.remove_xyl_phloem_BC(matrix_W, i_maturity, i_scenario)

                # Calculate interface fluxes 
                self._calculate_interface_flows(i_maturity, solution, rhs, rhs_s, i_scenario)

                # Calcul of fluxes between nodes and creation of the edge_flux_list
                self._calculate_edge_fluxes(i_maturity, i_scenario, matrix_W, solution)
    
        return solution, matrix_W


    def standard_solute_flux(self, h: int=0, i_maturity: int=0, i_scenario: int=0) -> tuple: 
        """Calculate standard solute flux."""

    
                # Resets matrix_C and rhs_C to geometrical factor values

                # build C matricies and rhs_C

                # solve system

                # calculate solute flux




    def solve_W(self, h: int=0, i_maturity: int=0) -> tuple: 
        """Solve the hydraulic system.

        Solve the hydraulic system based on the provided configurations and network.
        """
        # Unpack hydraulic properties for this scenario
        maturity_stages = self.geometry.maturity_stages
        barrier = int(maturity_stages[i_maturity].get("barrier"))
        height = float(maturity_stages[i_maturity].get("height"))

        # Build matrices
        matrix_W, matrix_C, rhs_C, rhs_p, rhs_x, rhs_s, rhs, Kmb =\
            self.build_matrices(h = h, i_maturity = i_maturity)
        # Solve system
        solution, verification_1 = self.solve(matrix = matrix_W, rhs = rhs, sparse_matrix = self.general.sparse_matrix)

        # Calculate standard water flow
        if barrier==0:
            self.standard_water_flow(matrix_W, rhs_s, rhs_p, solution, height, i_maturity)
        else:
            self.standard_water_flow(matrix_W, rhs_s, rhs_x, solution, height, i_maturity)

        self.standardized_results.append(solution)
        return solution, verification_1, matrix_W, Kmb, rhs_s

    
    def remove_xyl_phloem_BC(self, matrix_W: np.ndarray, i_maturity: int, i_scenario: int = 0) -> np.ndarray :

        barrier = int(self.geometry.maturity_stages[i_maturity].get("barrier"))
        #Removing xylem and phloem BC terms
        if barrier==0:
            if not isnan(self.psi_sieve[1][i_maturity][i_scenario]):
                for cid in self.network.protosieve_list:
                    matrix_W[cid][cid] += self.hydraulic.k_sieve
        else:
            if not isnan(self.psi_xyl[1][i_maturity][i_scenario]): #Pressure xylem BC
                for cid in self.network.xylem_cells:
                    matrix_W[cid][cid] += self.hydraulic.k_xyl
            else:
                pass
        return matrix_W

    def _calculate_interface_flows(self, i_maturity: int, solution: np.ndarray, rhs: np.ndarray, rhs_s: np.ndarray, i_scenario: int = 0) -> tuple:
        """
        Calculate flow rates for soil, xylem or phloem interfaces.
        """
        q_xyl = []
        q_sieve = []
        q_soil=[]

        barrier = int(self.geometry.maturity_stages[i_maturity].get("barrier"))

        for ind in self.network.border_walls:
            q_soil.append(rhs_s[ind]*(solution[ind]-self.boundary.scenarios[i_scenario]['psi_soil_left'])) #(cm^3/d) Positive for water flowing into the root
        for ind in self.network.border_junction:
            q_soil.append(rhs_s[ind]*(solution[ind]-self.boundary.scenarios[i_scenario]['psi_soil_left'])) #(cm^3/d) Positive for water flowing into the root

        if barrier > 0:
            if not np.isnan(self.psi_xyl[1][i_maturity][i_scenario]):
                for cid in self.network.xylem_cells:
                    Q = rhs[cid][0] * (solution[cid][0] - self.psi_xyl[1][i_maturity][i_scenario])
                    q_xyl.append(Q)
                    rank = int(self.network.cell_ranks[cid - self.network.n_wall_junction])
                    row = int(self.network.rank_to_row[rank])
                    self.flow_xyl_layer[row][i_maturity][i_scenario] += Q
            elif not np.isnan(self.distributed_flow_xyl[1][0][i_scenario]):
                 for cid in self.network.xylem_cells:
                    Q = -rhs[cid][0]
                    q_xyl.append(Q)
                    rank = int(self.network.cell_ranks[cid - self.network.n_wall_junction])
                    row = int(self.network.rank_to_row[rank])
                    self.flow_xyl_layer[row][i_maturity][i_scenario] += Q
            else:
                 print("Error: Scenario >0 should have xylem pressure boundary conditions, or flow xylem should be defined")

        elif barrier == 0:
            if not np.isnan(self.psi_sieve[1][i_maturity][i_scenario]):
                for cid in self.network.protosieve_list:
                    Q = rhs[cid][0] * (solution[cid][0] - self.psi_sieve[1][i_maturity][i_scenario])
                    q_sieve.append(Q)
                    rank = int(self.network.cell_ranks[cid - self.network.n_wall_junction])
                    row = int(self.network.rank_to_row[rank])
                    self.flow_sieve_layer[row][i_maturity][i_scenario] += Q
            elif not np.isnan(self.distributed_flow_sieve[1][0][i_scenario]):
                for cid in self.network.protosieve_list:
                     Q = -rhs[cid][0]
                     q_sieve.append(Q)
                     rank = int(self.network.cell_ranks[cid - self.network.n_wall_junction])
                     row = int(self.network.rank_to_row[rank])
                     self.flow_sieve_layer[row][i_maturity][i_scenario] += Q
            else:
                 print("Error: Scenario >0 should have phloem pressure boundary conditions, or flow phloem should be defined")

        return q_soil, q_xyl, q_sieve

    def _calculate_edge_fluxes(self, i_maturity: int, i_scenario: int, matrix_W: np.ndarray, solution: np.ndarray) -> None:
        """Calculate fluxes between nodes and store in edge_flux_list."""
        
        fluxes = []
        # Iterate over non-zero elements of matrix_W
        # matrix_W[i, j] is -EquivConductance[i, j] for i != j
        
        # If matrix is sparse:
        if hasattr(matrix_W, "tocoo"):
             cx = matrix_W.tocoo()
             for i, j, v in zip(cx.row, cx.col, cx.data):
                 if i < j: # One way
                     k_e = -v 
                     if k_e > 0: 
                         f = k_e * (solution[i][0] - solution[j][0])
                         fluxes.append({'source': i, 'target': j, 'flux': f})
        else:
             # Dense matrix
             rows, cols = matrix_W.shape
             for i in range(rows):
                 for j in range(i+1, cols):
                     v = matrix_W[i, j]
                     if v != 0:
                         k_e = -v
                         if k_e > 0:
                             f = k_e * (solution[i][0] - solution[j][0])
                             fluxes.append({'source': i, 'target': j, 'flux': f})

        self.edge_flux_list[i_maturity][i_scenario] = fluxes

    def standard_water_flow(self, matrix_W: np.ndarray, rhs_s: np.ndarray, rhs: np.ndarray, solution: np.ndarray, height: float, i_maturity: int) -> None:
        """
        Calculate the flow rates at interfaces.

        rhs is iether for xylem or phloem
        xylem is triggered by barrier>0
        phloem is triggered by barrier==0
        """
        barrier = int(self.geometry.maturity_stages[i_maturity].get("barrier"))
        matrix_W = self.remove_xyl_phloem_BC(matrix_W, i_maturity)
        q_soil, q_xyl, q_sieve = self._calculate_interface_flows(i_maturity, solution, rhs, rhs_s)
            
        self.total_flow[i_maturity][0]=sum(q_soil) #Total flow rate at root surface
        if barrier>0:
            if not isnan(self.psi_xyl[1][i_maturity][0]):
                self.kr_tot[i_maturity][0]=self.total_flow[i_maturity][0]/(self.boundary.scenarios[0]['psi_soil_left']-self.psi_xyl[1][i_maturity][0])/self.network.perimeter/height/1.0E-04
            else:
                print('Error: Scenario 0 should have xylem pressure boundary conditions, except for the elongation zone')
        elif barrier==0:
            if not isnan(self.psi_sieve[1][i_maturity][0]):
                self.kr_tot[i_maturity][0]=self.total_flow[i_maturity][0]/(self.boundary.scenarios[0]['psi_soil_left']-self.psi_sieve[1][i_maturity][0])/self.network.perimeter/height/1.0E-04
            else:
                print('Error: Scenario 0 should have phloem pressure boundary conditions in the elongation zone')

        if barrier>0 and isnan(self.psi_xyl[1][i_maturity][0]):
            self.psi_xyl[1][i_maturity][0]=0.0
            for cid in self.network.xylem_cells:
                self.psi_xyl[1][i_maturity][0]+=solution[cid][0]/len(self.network.xylem_cells) #Average of xylem water pressures
        elif barrier==0 and isnan(self.psi_sieve[1][i_maturity][0]):
            self.psi_sieve[1][i_maturity][0]=0.0
            for cid in self.network.protosieve_list:
                self.psi_sieve[1][i_maturity][0]+=solution[cid][0]/self.network.n_protosieve #Average of protophloem water pressures

    #Calculation of standard transmembrane fractions
    def standard_transmembrane_fractions(self, solution, i_maturity, Kmb):
        """
        Calculates the standard transmembrane fractions for a given maturity stage.

        Parameters
        ----------
        i_maturity : int
            Maturity stage index.

        Returns
        -------
        None
        """

        #Calculation of standard transmembrane fractions
        jmb=0 #Index for membrane conductance vector
        passage_cell_ids = np.array(self.geometry.passage_cell_ids)
        barrier = int(self.geometry.maturity_stages[i_maturity].get("barrier"))
            
        for node, edges in self.network.graph.adjacency() : #adjacency_iter returns an iterator of (node, adjacency dict) tuples for all nodes. This is the fastest way to look at every edge. For directed graphs, only outgoing adjacencies are included.
            i = self.indice[node] #Node ID number
            if i<self.network.n_walls: #wall ID 
                psi = solution[i][0]               
                for neighboor, eattr in edges.items(): #Loop on connections (edges)
                    j = self.indice[neighboor] #Neighbouring node ID number
                    path = eattr['path'] #eattr is the edge attribute (i.e. connection type)
                    if path == "membrane": #Membrane connection
                        psi_neigh = solution[j][0] #Neighbouring node water potential
                        K=Kmb[jmb][0]
                        jmb+=1
                        #Flow densities calculation
                        #Macroscopic distributed parameter for transmembrane flow
                        #Discretization based on cell layers and apoplasmic barriers
                        rank = int(self.network.cell_ranks[j-self.network.n_wall_junction])
                        row = int(self.network.rank_to_row[rank])
                        if rank == 1 and self.network.graph.nodes[node]['count_epi'] > 0: #Outer exodermis
                            row += 1
                        if rank == 3 and self.network.graph.nodes[node]['count_cortex'] > 0: #Outer endodermis
                            if any(passage_cell_ids==np.array(j-self.network.n_wall_junction)) and barrier==2:
                                row += 2
                            else:
                                row += 3
                        elif rank == 3 and self.network.graph.nodes[node]['count_stele_overall'] > 0: #Inner endodermis
                            if any(passage_cell_ids==np.array(j-self.network.n_wall_junction)) and barrier==2:
                                row += 1
                                
                        flow = K * (psi - psi_neigh) #Note that this is only valid because we are in the scenario 0 with no osmotic potentials
                        if ((j-self.network.n_wall_junction not in self.geometry.intercellular_ids) and (j not in self.network.xylem_cells)) or barrier==0: #Not part of STF if crosses an intercellular space "membrane" or mature xylem "membrane" (that is no membrane though still labelled like one)
                            if flow > 0 :
                                self.uptake_layer_plus[row][i_maturity][0] += flow #grouping membrane flow rates in cell layers
                            else:
                                self.uptake_layer_minus[row][i_maturity][0] += flow
                            if flow/self.total_flow[i_maturity][0] > 0 :
                                self.stf_layer_plus[row][i_maturity] += flow/self.total_flow[i_maturity][0] #Cell standard transmembrane fraction (positive)
                                self.stf_cell_plus[j-self.network.n_wall_junction][i_maturity] += flow/self.total_flow[i_maturity][0] #Cell standard transmembrane fraction (positive)
                                #STFmb[jmb-1][iMaturity] = Flow/Q_tot[iMaturity][0]
                            else:
                                self.stf_layer_minus[row][i_maturity] += flow/self.total_flow[i_maturity][0] #Cell standard transmembrane fraction (negative)
                                self.stf_cell_minus[j-self.network.n_wall_junction][i_maturity] += flow/self.total_flow[i_maturity][0] #Cell standard transmembrane fraction (negative)
                                #STFmb[jmb-1][iMaturity] = Flow/Q_tot[iMaturity][0]
                            self.stf_mb[jmb-1][i_maturity] = flow/self.total_flow[i_maturity][0]

    

    def info(self) -> str:
        """Prints the descrition of the problem.

        Returns
        -------
        str
            Description of the problem.

        """
        return self._details
    

