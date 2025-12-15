
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


from src.utils.data_loader import *
from src.utils.network_builder import *

class Mecha:
    """Main class of the library, encodes a hydraulic anatomy to solve.
    """

    def __init__(self, all_input: Optional[InData] = None):
        """Initialize the Mecha class.

        Parameters
        ----------
        all_input : InData, optional
            Input data containing all configurations.
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

        self.solution = None
        self.network = NetworkBuilder()

        if self.all_input is not None:
            self._build_anatomy()
            self._set_hydraulics()

    
    @property
    def _details(self):
        """Sums up the characteristics of the anatomy.

        Returns
        -------
        str
            A printable description of the anatomy.

        """

        description = "=== Mecha Configuration ===\n\n"

        if self.all_input is not None:
            description += self.all_input.info()
        else:

        return description

    def _build_anatomy(self):
        """Build the anatomical network."""
        self.network.build_network(self.general, self.geometry, self.cellset_data)
        self.position=nx.get_node_attributes(self.network.graph,'position') #Updates nodes XY positions (micrometers)
        self.indice=nx.get_node_attributes(self.network.graph,'indice') #Node indices (walls, junctions and cells)
        
    def _set_hydraulics(self) -> None:
        """Set up hydraulic properties and solution arrays."""
        # Initialize dimensions
        n_maturity = self.geometry.n_maturity
        n_scenarios = self.boundary.n_scenarios
        pile_up = self.geometry.pile_up
        r_discret = self._get_r_discret()

        # Initialize solution arrays
        self._initialize_xylem_arrays(n_maturity, n_scenarios)
        self._initialize_phloem_arrays(n_maturity, n_scenarios)
        self._initialize_osmotic_arrays(n_scenarios)

        self._initialize_arrays(pile_up, r_discret, n_maturity, n_scenarios)
        
        self._initialize_stf_arrays(r_discret, n_maturity)
        self._initialize_pressure_arrays(r_discret, n_maturity, n_scenarios)
        self._initialize_flow_arrays(n_maturity, n_scenarios)
        self._initialize_tropism_arrays(n_maturity, n_scenarios)

        # Set initial conditions for each maturity stage
        self._set_maturity_initial_conditions()

    def _get_r_discret(self) -> int:
        """Get the radial discretization value."""
        if hasattr(self.network, 'r_discret') and self.network.r_discret:
            return int(self.network.r_discret[0])
        return 10  # Default value

    def _initialize_xylem_arrays(self, n_maturity: int, n_scenarios: int) -> None:
        """Initialize xylem-related arrays."""
        self.Psi_xyl = np.empty((n_maturity, n_scenarios))
        self.Psi_xyl[:] = np.nan

        self.dPsi_xyl = np.empty((n_maturity, n_scenarios))
        self.dPsi_xyl[:] = np.nan

        self.iEquil_xyl = np.nan  # Index of the equilibrium root xylem pressure scenario

        # Initialize with one extra row for total flow
        self.Flow_xyl = np.empty((len(self.network.xylem_cells) + 1, n_scenarios))
        self.Flow_xyl[:] = np.nan

        # Set initial xylem flow rate if available
        if self.boundary.bc_xyl_elems and len(self.boundary.bc_xyl_elems) > 0:
            flowrate = self.boundary.bc_xyl_elems[0].get("flowrate")
            if flowrate is not None:
                self.Flow_xyl[0, 0] = float(flowrate)

    def _initialize_phloem_arrays(self, n_maturity: int, n_scenarios: int) -> None:
        """Initialize phloem-related arrays."""
        self.Psi_sieve = np.empty((n_maturity, n_scenarios))
        self.Psi_sieve[:] = np.nan

        self.dPsi_sieve = np.empty((n_maturity, n_scenarios))
        self.dPsi_sieve[:] = np.nan

        self.iEquil_sieve = np.nan  # Index of the equilibrium root phloem pressure scenario

        # Initialize with one extra row for total flow
        self.Flow_sieve = np.empty((self.network.n_sieve + 1, n_scenarios))
        self.Flow_sieve[:] = np.nan

        # Set initial phloem flow rate if available
        if self.boundary.bc_sieve_elems and len(self.boundary.bc_sieve_elems) > 0:
            flowrate = self.boundary.bc_sieve_elems[0].get("flowrate")
            if flowrate is not None:
                self.Flow_sieve[0, 0] = float(flowrate)

    def _initialize_osmotic_arrays(self, n_scenarios: int) -> None:
        """Initialize osmotic-related arrays."""
        self.Os_sieve = np.zeros((1, n_scenarios))
        self.Os_cortex = np.zeros((1, n_scenarios))
        self.Os_hetero = np.zeros((1, n_scenarios))
        self.s_factor = np.zeros((1, n_scenarios))
        self.s_hetero = np.zeros((1, n_scenarios))
        self.Elong_cell = np.zeros((1, n_scenarios))
        self.Elong_cell_side_diff = np.zeros((1, n_scenarios))

    def _initialize_layer_arrays(self, r_discret: int, n_maturity: int, n_scenarios: int) -> None:
        """Initialize layer-based arrays."""
        self.UptakeLayer_plus = np.zeros((r_discret, n_maturity, n_scenarios))
        self.UptakeLayer_minus = np.zeros((r_discret, n_maturity, n_scenarios))
        self.Q_xyl_layer = np.zeros((r_discret, n_maturity, n_scenarios))
        self.Q_sieve_layer = np.zeros((r_discret, n_maturity, n_scenarios))
        self.Q_elong_layer = np.zeros((r_discret, n_maturity, n_scenarios))

    def _initialize_stf_arrays(self, r_discret: int, n_maturity: int) -> None:
        """Initialize STF (Specific Tissue Function) arrays."""
        self.STFmb = np.zeros((self.network.n_membrane, n_maturity))
        self.STFcell_plus = np.zeros((self.network.n_cells, n_maturity))
        self.STFcell_minus = np.zeros((self.network.n_cells, n_maturity))
        self.STFlayer_plus = np.zeros((r_discret, n_maturity))
        self.STFlayer_minus = np.zeros((r_discret, n_maturity))

    def _initialize_pressure_arrays(self, r_discret: int, n_maturity: int, n_scenarios: int) -> None:
        """Initialize pressure and osmotic arrays."""
        self.PsiCellLayer = np.zeros((r_discret, n_maturity, n_scenarios))
        self.PsiWallLayer = np.zeros((r_discret, n_maturity, n_scenarios))
        self.OsCellLayer = np.zeros((r_discret, n_maturity, n_scenarios))
        self.nOsCellLayer = np.zeros((r_discret, n_maturity, n_scenarios))
        self.OsWallLayer = np.zeros((r_discret, n_maturity, n_scenarios))
        self.nOsWallLayer = np.zeros((r_discret, n_maturity, n_scenarios))
        self.NWallLayer = np.zeros((r_discret, n_maturity, n_scenarios))

    def _initialize_flow_arrays(self, n_maturity: int, n_scenarios: int) -> None:
        """Initialize flow and conductivity arrays."""
        self.Q_tot = np.zeros((n_maturity, n_scenarios))
        self.kr_tot = np.zeros((n_maturity, 1))

    def _initialize_tropism_arrays(self, n_maturity: int, n_scenarios: int) -> None:
        """Initialize tropism arrays."""
        self.Hydropatterning = np.empty((n_maturity, n_scenarios))
        self.Hydropatterning[:] = np.nan
        self.Hydrotropism = np.empty((n_maturity, n_scenarios))
        self.Hydrotropism[:] = np.nan

    def _set_maturity_initial_conditions(self) -> None:
        """Set initial conditions for each maturity stage."""
        iMaturity = 0

        for Maturity in self.geometry.maturity_elems:

            # Set xylem initial conditions
            self._set_xylem_initial_conditions(iMaturity)

            # Set phloem initial conditions
            self._set_phloem_initial_conditions(iMaturity)
            
            # Set cell wall hydraulic conductivity and plasmodesmatal conductance
            barrier = int(Maturity.get("Barrier"))
            height = int(Maturity.get("height"))
            self._set_hydraulic_conductivities(iMaturity, barrier, height)

            self._fill_doussan_mx(barrier)

            self._solve_doussan()

            iMaturity += 1

    def _set_xylem_initial_conditions(self, iMaturity: int) -> None:
        """Set initial conditions for xylem."""
        if not self.boundary.bc_xyl_elems:
            return

        # Set xylem pressure potential
        pressure = self.boundary.bc_xyl_elems[0].get("pressure")
        if pressure is not None:
            self.Psi_xyl[iMaturity, 0] = float(pressure)

        # Set xylem pressure potential change
        deltaP = self.boundary.bc_xyl_elems[0].get("deltaP")
        if deltaP is not None:
            self.dPsi_xyl[iMaturity, 0] = float(deltaP)

        # Handle xylem flow conditions
        self._handle_xylem_flow_conditions(iMaturity)

    def _handle_xylem_flow_conditions(self, iMaturity: int) -> None:
        """Handle xylem flow conditions."""
        if np.isnan(self.Flow_xyl[0, 0]):
            return

        if np.isnan(self.Psi_xyl[iMaturity, 0]) and np.isnan(self.dPsi_xyl[iMaturity, 0]):
            self._distribute_xylem_flow()
            if self.Flow_xyl[0, 0] == 0.0:
                self.iEquil_xyl = 0
        else:
            print('Error: Cannot have both pressure and flow BC at xylem boundary')

    def _distribute_xylem_flow(self) -> None:
        """Distribute xylem flow proportionally to xylem cross-section area."""
        tot_flow = self.Flow_xyl[0, 0]
        sum_area = 0.0

        # Calculate total area
        for cid in self.network.xylem_cells:
            area = self.network.cell_areas[cid - self.network.n_wall_junction]
            sum_area += area

        # Distribute flow
        i = 1
        for cid in self.network.xylem_cells:
            area = self.network.cell_areas[cid - self.network.n_wall_junction]
            self.Flow_xyl[i, 0] = tot_flow * (area / sum_area)
            i += 1

    def _set_phloem_initial_conditions(self, iMaturity: int) -> None:
        """Set initial conditions for phloem."""
        if not self.boundary.bc_sieve_elems:
            return

        # Set phloem pressure potential
        pressure = self.boundary.bc_sieve_elems[0].get("pressure")
        if pressure is not None:
            self.Psi_sieve[iMaturity, 0] = float(pressure)

        # Set phloem pressure potential change
        deltaP = self.boundary.bc_sieve_elems[0].get("deltaP")
        if deltaP is not None:
            self.dPsi_sieve[iMaturity, 0] = float(deltaP)

        # Handle phloem flow conditions
        self._handle_phloem_flow_conditions(iMaturity)

    def _handle_phloem_flow_conditions(self, iMaturity: int) -> None:
        """Handle phloem flow conditions."""
        if np.isnan(self.Flow_sieve[0, 0]):
            return

        if np.isnan(self.Psi_sieve[iMaturity, 0]) and np.isnan(self.dPsi_sieve[iMaturity, 0]):
            self._distribute_phloem_flow()
            if self.Flow_sieve[0, 0] == 0.0:
                self.iEquil_sieve = 0
        else:
            print('Error: Cannot have both pressure and flow BC at phloem boundary')

    def _distribute_phloem_flow(self) -> None:
        """Distribute phloem flow proportionally to phloem cross-section area."""
        tot_flow = self.Flow_sieve[0, 0]
        sum_area = 0.0

        # Calculate total area
        for cid in self.network.protosieve_list:
            area = self.network.cell_areas[cid - self.network.n_wall_junction]
            sum_area += area

        # Distribute flow
        i = 1
        for cid in self.network.protosieve_list:
            area = self.network.cell_areas[cid - self.network.n_wall_junction]
            self.Flow_sieve[i, 0] = tot_flow * (area / sum_area)
            i += 1

    def _set_hydraulic_conductivities(self, iMaturity: int, barrier: int, height: int) -> None:
        """Set cell wall hydraulic conductivity and plasmodesmatal conductance."""
        hydraulic = self.hydraulic

        # Loop through hydraulic scenarios
        for h in range(hydraulic.n_hydraulics):
            # Cell wall hydraulic conductivity
            kw = self._get_kw_value(h, hydraulic)
            kw_barrier_casp = self._get_kw_barrier_value(h, hydraulic, "Casp")
            kw_barrier_sub = self._get_kw_barrier_value(h, hydraulic, "Sub")

            # Set wall conductivities based on barrier type
            kw_endo_endo, kw_exo_exo, kw_cortex_cortex, kw_endo_peri, kw_endo_cortex, kw_passage = \
                self._get_wall_conductivities(Barrier, kw, kw_barrier_casp, kw_barrier_sub)

            # Plasmodesmatal hydraulic conductance
            Kpl = self._get_plasmodesmatal_conductance(h, hydraulic)

            # Contribution of aquaporins to membrane hydraulic conductivity
            kaqp, kaqp_stele, kaqp_endo, kaqp_exo, kaqp_epi, kaqp_cortex = \
                self._get_aquaporin_contributions(h, hydraulic)

            # Calculate parameter a for cortex
            a_cortex, b_cortex = self._calculate_cortex_parameters(height, kaqp_cortex, hydraulic)

            # Store or use these values as needed
            # For example, you might want to store them in a dictionary or use them directly
            # This part depends on how you plan to use these values in your calculations

    def _get_kw_value(self, h: int, hydraulic: HydraulicData) -> float:
        """Get the kw value based on the scenario index."""
        if hydraulic.n_kw == hydraulic.n_hydraulics:
            return float(hydraulic.kw_elems[h].get("value"))
        elif hydraulic.n_kw == 1:
            return float(hydraulic.kw_elems[0].get("value"))
        else:
            return float(hydraulic.kw_elems[int(h/(hydraulic.n_kaqp*hydraulic.n_kpl))%hydraulic.n_kw].get("value"))

    def _get_kw_barrier_value(self, h: int, hydraulic: HydraulicData, type:str) -> float:
        """Get the kw_barrier value based on the scenario index."""
        if hydraulic.n_kw_barrier == hydraulic.n_hydraulics:
            return float(hydraulic.kw_barrier_elems[h].get(type))
        elif hydraulic.n_kw_barrier == 1:
            return float(hydraulic.kw_barrier_elems[0].get(type))
        else:
            return float(hydraulic.kw_barrier_elems[int(h/(hydraulic.n_kaqp*hydraulic.n_kpl*hydraulic.n_kw))%hydraulic.n_kw_barrier].get(type))

    def _get_wall_conductivities(self, barrier: int, kw: float, kw_barrier_casparian: float, kw_barrier_suberin: float) -> tuple:
        """Get wall conductivities based on barrier type."""
        if barrier == 0:  # No Casparian strip
            kw_endo_endo = kw
            kw_puncture = kw
            kw_exo_exo = kw
            kw_exo_epi=kw
            kw_cortex_cortex = kw
            kw_endo_peri = kw
            kw_endo_cortex = kw
            kw_passage = kw
        elif barrier == 1:  # Endodermis radial walls
            kw_endo_endo = kw_barrier_casparian
            kw_exo_exo = kw
            kw_exo_epi=kw
            kw_cortex_cortex = kw
            kw_endo_peri = kw
            kw_endo_cortex = kw
            kw_passage = kw
        elif barrier == 2:  # Endodermis with passage cells
            kw_endo_endo = kw_barrier_casparian
            kw_exo_exo = kw
            kw_cortex_cortex = kw
            kw_endo_peri = kw_barrier_suberin
            kw_endo_cortex = kw_barrier_suberin
            kw_passage = kw
        elif barrier == 3:  # Endodermis full
            kw_endo_endo = kw_barrier_casparian
            kw_exo_exo = kw
            kw_cortex_cortex = kw
            kw_endo_peri = kw_barrier_suberin
            kw_endo_cortex = kw_barrier_suberin
            kw_passage = kw_barrier_suberin
        elif barrier == 4:  # Endodermis full and exodermis radial walls
            kw_endo_endo = kw_barrier_casparian
            kw_exo_exo = kw_barrier_casparian
            kw_cortex_cortex = kw
            kw_endo_peri = kw_barrier_suberin
            kw_endo_cortex = kw_barrier_suberin
            kw_passage = kw_barrier_suberin
        elif Barrier==5: # Endodermal & exodermal Casparian strips
            kw_endo_endo=kw_barrier_casparian
            kw_exo_exo=kw_barrier_casparian #(cm^2/hPa/d) hydraulic conductivity of the suberised walls between exodermis cells
            kw_exo_epi=kw
            kw_exo_cortex=kw
            kw_cortex_cortex=kw
            kw_endo_peri=kw #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
            kw_endo_cortex=kw #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
            kw_passage=kw #(cm^2/hPa/d) hydraulic conductivity of passage cells tangential walls
        elif Barrier==6: #Exodermis full and endodermis radial walls
            kw_endo_endo=kw_barrier_casparian
            kw_exo_exo=kw_barrier_casparian #(cm^2/hPa/d) hydraulic conductivity of the suberised walls between exodermis cells
            kw_exo_epi=kw_barrier_suberin
            kw_exo_cortex=kw_barrier_suberin
            kw_cortex_cortex=kw
            kw_endo_peri=kw #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
            kw_endo_cortex=kw #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
            kw_passage=kw #(cm^2/hPa/d) hydraulic conductivity of passage cells tangential walls
        elif Barrier==7: #Exodermis radial walls
            kw_endo_endo=kw
            kw_exo_exo=kw_barrier_casparian #(cm^2/hPa/d) hydraulic conductivity of the suberised walls between exodermis cells
            kw_exo_epi=kw
            kw_exo_cortex=kw
            kw_cortex_cortex=kw
            kw_endo_peri=kw #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
            kw_endo_cortex=kw #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
            kw_passage=kw #(cm^2/hPa/d) hydraulic conductivity of passage cells tangential walls
        elif Barrier==8: #Exodermis full suberized and endodermis full suberized
            kw_endo_endo=kw_barrier_casparian
            kw_exo_exo=kw_barrier_casparian #(cm^2/hPa/d) hydraulic conductivity of the suberised walls between exodermis cells
            kw_exo_epi=kw_barrier_suberin
            kw_exo_cortex=kw_barrier_suberin
            kw_cortex_cortex=kw
            kw_endo_peri=kw_barrier_suberin #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
            kw_endo_cortex=kw_barrier_suberin #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
            kw_passage=kw #(cm^2/hPa/d) hydraulic conductivity of passage cells tangential walls
        elif Barrier==9: #Lignin Cap
            kw_endo_endo=kw_barrier_casparian
            kw_exo_exo=kw_barrier_casparian #(cm^2/hPa/d) hydraulic conductivity of the suberised walls between exodermis cells
            kw_exo_epi=kw_barrier_suberin
            kw_exo_cortex=kw
            kw_cortex_cortex=kw
            kw_endo_peri=kw #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
            kw_endo_cortex=kw #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
            kw_passage=kw #(cm^2/hPa/d) hydraulic conductivity of passage cells tangential walls


        return kw_endo_endo, kw_exo_exo, kw_cortex_cortex, kw_endo_peri, kw_endo_cortex, kw_passage

    def _get_plasmodesmatal_conductance(self, h: int, hydraulic: HydraulicData) -> float:
        """Get plasmodesmatal hydraulic conductance."""
        if hydraulic.n_kpl == hydraulic.n_hydraulics:
            iPD = h
        elif hydraulic.n_kpl == 1:
            iPD = 0
        else:
            iPD = int(h/hydraulic.n_kaqp)%hydraulic.n_kpl

        return float(hydraulic.kpl_elems[iPD].get("value"))

    def _get_aquaporin_contributions(self, h: int, hydraulic: HydraulicData) -> tuple:
        """Get aquaporin contributions to membrane hydraulic conductivity."""
        if hydraulic.n_kaqp == hydraulic.n_hydraulics:
            iAQP = h
        elif hydraulic.n_kaqp == 1:
            iAQP = 0
        else:
            iAQP = h%hydraulic.n_kaqp

        kaqp = float(hydraulic.kaqp_elems[iAQP].get("value"))
        kaqp_stele = kaqp * float(hydraulic.kaqp_elems[iAQP].get("stele_factor"))
        kaqp_endo = kaqp * float(hydraulic.kaqp_elems[iAQP].get("endo_factor"))
        kaqp_exo = kaqp * float(hydraulic.kaqp_elems[iAQP].get("exo_factor"))
        kaqp_epi = kaqp * float(hydraulic.kaqp_elems[iAQP].get("epi_factor"))
        kaqp_cortex = kaqp * float(hydraulic.kaqp_elems[iAQP].get("cortex_factor"))

        return kaqp, kaqp_stele, kaqp_endo, kaqp_exo, kaqp_epi, kaqp_cortex

    def _calculate_cortex_parameters(self, height: int, kaqp_cortex: float, hydraulic: HydraulicData) -> tuple:
        """Calculate parameter a for cortex."""
        if self.hydraulic.ratio_cortex == 1:  # Uniform AQP activity in all cortex membranes
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
                        temp+=surf*1.0E-04*(self.network.distance_center_grav[wall_id]+(self.hydraulic.ratio_cortex*self.network.distance_max_cortex-self.network.distance_min_cortex)/(1-self.hydraulic.ratio_cortex))
                        tot_surf_cortex+=surf
            a_cortex=kaqp_cortex*tot_surf_cortex/temp  #(1/hPa/d)
            b_cortex=a_cortex*1.0E-04*(self.hydraulic.ratio_cortex*self.network.distance_max_cortex-self.network.distance_min_cortex)/(1-self.hydraulic.ratio_cortex) #(cm/hPa/d)

        return a_cortex, b_cortex

    def _solve_doussan(self):
        """Solve the hydraulic system.

        Solve the hydraulic system based on the provided configurations and network.
        """
        self.solution_W = np.linalg.solve(self.matrix_W, self.rhs) #Solving the equation to get potentials inside the network

        self.verification_1 = np.allclose(np.dot(self.matrix_W,self.solution_W),self.rhs) #Verification that computation was correct
    

    def info(self):
        """Prints the descrition of the problem.

        Returns
        -------
        None

        """
        print(self._details)
    

