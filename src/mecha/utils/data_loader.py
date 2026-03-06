#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#       mecha.utils.loader
#
#       File author(s):
#           Dilhan Ozturk, Adrien Heymans
#
#       File contributor(s):
#
#       File maintainer(s):
#           Valentin Couvreur
#
#       Copyright © by UCLouvain
#       Distributed under the LGPL License..
#       See accompanying file LICENSE.txt or copy at
#           https://www.gnu.org/licenses/lgpl-3.0.en.html
#
# -----------------------------------------------------------------------
import os
import xml.etree.ElementTree as ET
from lxml import etree 
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class BoundaryData:
    """Boundary conditions file configuration

    This class loads and manages boundary condition scenarios from an XML file.
    If no file is provided, default values are used.

    Attributes
    ----------
    bc_file : str
        Path to the boundary condition XML file.
    psi_soil_elems : list, optional
        List of soil pressure elements from the XML file.
    bc_xyl_elems : list, optional
        List of xylem boundary condition elements from the XML file.
    bc_sieve_elems : list, optional
        List of sieve boundary condition elements from the XML file.
    psi_cell_elems : list, optional
        List of cell pressure elements from the XML file.
    elong_cell_elems : list, optional
        List of cell elongation elements from the XML file.
    n_scenarios : int, optional
        Number of boundary condition scenarios.
    water_fraction_apo : float, optional
        Relative volumetric fraction of water in the apoplast.
    water_fraction_sym : float, optional
        Relative volumetric fraction of water in the symplast.
    c_flag : bool, optional
        Flag indicating whether to calculate solute stationary fluxes.
    path_scenarios : str, optional
        Path for scenario outputs.
    scenarios : list, optional
        List of dictionaries, each representing a boundary condition scenario.
    osmotic_diffusivity_soil : float, optional
        Osmotic diffusivity in the soil.
    osmotic_diffusivity_xyl : float, optional
        Osmotic diffusivity in the xylem.

    Methods
    -------
    _load_boundary_conditions()
        Load boundary condition scenarios from the XML file.

    """
    # File path
    bc_file: Optional[str] = None

    # Boundary condition elements (raw from XML)
    psi_soil_elems: List[Any] = field(default_factory=list)
    bc_xyl_elems: List[Any] = field(default_factory=list)
    bc_sieve_elems: List[Any] = field(default_factory=list)
    psi_cell_elems: List[Any] = field(default_factory=list)
    elong_cell_elems: List[Any] = field(default_factory=list)
    
    # Counts
    n_scenarios: int = 1
    
    # Water fraction parameters
    water_fraction_apo: float = 0.69  # Relative volumetric fraction of water in the apoplast
    water_fraction_sym: float = 0.7  # Relative volumetric fraction of water in the symplast
    
    # Solute transport flag
    c_flag: bool = False  # Do we calculate solute stationary fluxes?
    
    # Path for scenario outputs
    path_scenarios: str = ""
    
    # Boundary condition scenarios
    scenarios: List[Dict[str, Any]] = field(default_factory=list)
    osmotic_potentials: List[Dict[str,Any]] = field(default_factory=list)
    reflection_coefficients: List[Dict[str,Any]] = field(default_factory=list)
    
    # Arrays for boundary conditions
    
    # Diffusivity parameters
    osmotic_diffusivity_soil: float = 0.0
    osmotic_diffusivity_xyl: float = 0.0

    def __post_init__(self):
        """Post-initialization method to load boundary conditions."""
        if self.bc_file is not None:
            self._load_boundary_conditions()
        else:
            self._set_default_scenario()

    def _load_boundary_conditions(self):
        """Load boundary condition scenarios from the XML file.

        This method parses the XML file to extract boundary condition elements,
        water fractions, diffusivity parameters, and scenario details.
        """
        root = etree.parse(self.bc_file).getroot()

        # Parse different boundary condition elements
        self.psi_soil_elems = root.xpath('Psi_soil_range/Psi_soil')
        self.bc_xyl_elems = root.xpath('BC_xyl_range/BC_xyl')
        self.bc_sieve_elems = root.xpath('BC_sieve_range/BC_sieve')
        self.psi_cell_elems = root.xpath('Psi_cell_range/Psi_cell')
        self.elong_cell_elems = root.xpath('Elong_cell_range/Elong_cell')
        
        # Extract water fractions
        water_fractions = root.xpath('Water_fractions')[0]
        self.water_fraction_apo = float(water_fractions.get("Apoplast"))
        self.water_fraction_sym = float(water_fractions.get("Symplast"))
        
        # Extract path for scenarios
        path_scenarios_elem = root.xpath('path_scenarios/Output')
        self.path_scenarios = path_scenarios_elem[0].get("path") if path_scenarios_elem else ""

        # Extract diffusivity parameters
        osmotic_diff_soil_elem = root.xpath('Psi_soil_range/osmotic_diffusivity')
        self.osmotic_diffusivity_soil = float(osmotic_diff_soil_elem[0].get("value")) if osmotic_diff_soil_elem else 0.0
        
        osmotic_diff_xyl_elem = root.xpath('BC_xyl_range/osmotic_diffusivity')
        self.osmotic_diffusivity_xyl = float(osmotic_diff_xyl_elem[0].get("value")) if osmotic_diff_xyl_elem else 0.0

        # Set number of scenarios
        self.n_scenarios = len(self.psi_soil_elems)

        # Check if we need to calculate solute stationary fluxes
        if self.osmotic_diffusivity_xyl != 0 and self.osmotic_diffusivity_soil != 0:
            self.c_flag = True
            print('Calculation of analytical solution for radial solute transport in cell walls')

        # Load boundary condition scenarios
        for count in range(self.n_scenarios):            

            # Create scenario dictionary
            scenario = {
                'psi_soil_left': float(self.psi_soil_elems[count].get("pressure_left")),
                'psi_soil_right': float(self.psi_soil_elems[count].get("pressure_right")),
                'osmotic_left_soil': float(self.psi_soil_elems[count].get("osmotic_left")),
                'osmotic_right_soil': float(self.psi_soil_elems[count].get("osmotic_right")),
                'osmotic_symmetry_soil': float(self.psi_soil_elems[count].get("osmotic_symmetry")),
                'osmotic_shape_soil': float(self.psi_soil_elems[count].get("osmotic_shape")), #1 for linear, >1 for outer slope flat, <1 for inner slope flat
                'osmotic_diffusivity_soil': self.osmotic_diffusivity_soil,
                'osmotic_xyl': float(self.bc_xyl_elems[count].get("osmotic_xyl")) if self.bc_xyl_elems[count].get("osmotic_xyl") else 0.0,
                'osmotic_endo': float(self.bc_xyl_elems[count].get("osmotic_endo")) if self.bc_xyl_elems[count].get("osmotic_endo") else 0.0,
                'osmotic_symmetry_xyl': float(self.bc_xyl_elems[count].get("osmotic_symmetry")) if self.bc_xyl_elems[count].get("osmotic_symmetry") else 1.0,
                'osmotic_shape_xyl': float(self.bc_xyl_elems[count].get("osmotic_shape")) if self.bc_xyl_elems[count].get("osmotic_shape") else 1.0,
                'osmotic_diffusivity_xyl': self.osmotic_diffusivity_xyl,
                'pressure_xyl_prox': float(self.bc_xyl_elems[count].get("pressure_prox")) if self.bc_xyl_elems[count].get("pressure_prox") or self.bc_xyl_elems[count].get("pressure") else np.nan,
                'pressure_xyl_dist': float(self.bc_xyl_elems[count].get("pressure_dist")) if self.bc_xyl_elems[count].get("pressure_dist") else np.nan,
                'flow_xyl_prox': float(self.bc_xyl_elems[count].get("flowrate_prox")) if self.bc_xyl_elems[count].get("flowrate_prox") or self.bc_xyl_elems[count].get("flowrate") else np.nan,
                'flow_xyl_dist': float(self.bc_xyl_elems[count].get("flowrate_dist")) if self.bc_xyl_elems[count].get("flowrate_dist") else np.nan,
                'delta_p_xyl': float(self.bc_xyl_elems[count].get("deltaP")) if self.bc_xyl_elems[count].get("deltaP") else np.nan,
                'pressure_sieve_prox': float(self.bc_sieve_elems[count].get("pressure_prox")) if self.bc_sieve_elems[count].get("pressure_prox") or self.bc_sieve_elems[count].get("pressure") else np.nan,
                'pressure_sieve_dist': float(self.bc_sieve_elems[count].get("pressure_dist")) if self.bc_sieve_elems[count].get("pressure_dist") else np.nan,
                'flow_sieve_prox': float(self.bc_sieve_elems[count].get("flowrate_prox")) if self.bc_sieve_elems[count].get("flowrate_prox") or self.bc_sieve_elems[count].get("flowrate") else np.nan,
                'flow_sieve_dist': float(self.bc_sieve_elems[count].get("flowrate_dist")) if self.bc_sieve_elems[count].get("flowrate_dist") else np.nan,
                'delta_p_sieve': float(self.bc_sieve_elems[count].get("deltaP")) if self.bc_sieve_elems[count].get("deltaP") else np.nan,
                'osmotic_sieve': float(self.bc_sieve_elems[count].get("osmotic")) if self.bc_sieve_elems[count].get("osmotic") else np.nan,
                's_hetero': int(self.psi_cell_elems[count].get("s_hetero")),
                's_factor': float(self.psi_cell_elems[count].get("s_factor")),
                'os_hetero': int(self.psi_cell_elems[count].get("Os_hetero")),
                'os_cortex': float(self.psi_cell_elems[count].get("Os_cortex")),
                'elongation_midpoint_rate': float(self.elong_cell_elems[count].get("midpoint_rate")),
                'elongation_side_rate_difference': float(self.elong_cell_elems[count].get("side_rate_difference")),
            }

            self.scenarios.append(scenario)
            
    def _set_default_scenario(self):
        """Set default boundary condition scenario if no file is provided."""
        print('Setting default boundary conditions...')
        scenario = {
            'psi_soil_left': 0.0,
            'psi_soil_right': 0.0,
            'osmotic_left_soil': 0.0,
            'osmotic_right_soil': 0.0,
            'osmotic_symmetry_soil': 1.0,
            'osmotic_shape_soil': 1.0,
            'osmotic_diffusivity_soil': 0.0,
            'osmotic_xyl': 0.0,
            'osmotic_endo': 0.0,
            'osmotic_symmetry_xyl': 1.0,
            'osmotic_shape_xyl': 1.0,
            'osmotic_diffusivity_xyl': 0.0,
            'pressure_xyl_prox': -5.0E3,
            'pressure_xyl_dist': np.nan,
            'flow_xyl_prox': np.nan,
            'flow_xyl_dist': np.nan,
            'delta_p_xyl_prox': np.nan,
            'pressure_sieve_prox': 1.1E4,
            'pressure_sieve_dist': np.nan,
            'flow_sieve_prox': np.nan,
            'flow_sieve_dist': np.nan,
            'delta_p_sieve': np.nan,
            'osmotic_sieve': np.nan,
            's_hetero': 0,
            's_factor': 1.0,
            'os_hetero': 0,
            'os_cortex': 0.0,
            'elongation_midpoint_rate': 2.8,
            'elongation_side_rate_difference': 0.0,
        }
        self.scenarios.append(scenario)

    def add_scenario(self, scenario):
        """Add a scenario to the list of scenarios."""
        self.scenarios.append(scenario)
        self.n_scenarios = len(self.scenarios)

    def get_reflection_coefficients(self):

        for i_scenario in range(1,self.n_scenarios):
            #Reflection coefficients of membranes (undimensional)
            s_hetero=int(self.scenarios[i_scenario].get("s_hetero")) #0:Uniform, 1: non-uniform, stele twice more permeable to solute, 2: non-uniform, cortex twice more permeable to solute
            s_factor=float(self.scenarios[i_scenario].get("s_factor")) #(undimensional [0 -> 1]) multiplies all sigma values
            elong_cell=float(self.scenarios[i_scenario].get("elongation_midpoint_rate")) #Cell elongation rate (cm/d)
            elong_cell_side_diff=float(self.scenarios[i_scenario].get("elongation_side_rate_difference")) #Difference between cell elongation rates on the sides of the root in the EZ (cm/d)
            if s_hetero==0:
                s_epi=s_factor*1.0
                s_exo_epi=s_factor*1.0
                s_exo_cortex=s_factor*1.0
                s_cortex=s_factor*1.0
                s_endo_cortex=s_factor*1.0
                s_endo_peri=s_factor*1.0
                s_peri=s_factor*1.0
                s_stele=s_factor*1.0
                s_comp=s_factor*1.0
                s_sieve=s_factor*1.0
            elif s_hetero==1:
                s_epi=s_factor*1.0
                s_exo_epi=s_factor*1.0
                s_exo_cortex=s_factor*1.0
                s_cortex=s_factor*1.0
                s_endo_cortex=s_factor*1.0
                s_endo_peri=s_factor*0.5
                s_peri=s_factor*0.5
                s_stele=s_factor*0.5
                s_comp=s_factor*0.5
                s_sieve=s_factor*0.5
            elif s_hetero==2:
                s_epi=s_factor*0.5
                s_exo_epi=s_factor*0.5
                s_exo_cortex=s_factor*0.5
                s_cortex=s_factor*0.5
                s_endo_cortex=s_factor*0.5
                s_endo_peri=s_factor*1.0
                s_peri=s_factor*1.0
                s_stele=s_factor*1.0
                s_comp=s_factor*1.0
                s_sieve=s_factor*1.0

            self.reflection_coefficients.append({
                's_epi': s_epi,
                's_exo_epi': s_exo_epi,
                's_exo_cortex': s_exo_cortex,
                's_cortex': s_cortex,
                's_endo_cortex': s_endo_cortex,
                's_endo_peri': s_endo_peri,
                's_peri': s_peri,
                's_stele': s_stele,
                's_comp': s_comp,
                's_sieve': s_sieve,
            })
    
    

    def get_osmotic_potentials(self):
        
        for i_scenario in range(1,self.n_scenarios):
            #Osmotic potentials (hPa)
            Os_hetero=int(self.scenarios[i_scenario].get("os_hetero")) #0:Uniform, 1: non-uniform no KNO3 treatment, 2: non-uniform with KNO3 treatment to help guttation
            Os_cortex=float(self.scenarios[i_scenario].get("os_cortex")) # Cortical cell osmotic potential (hPa)
            Os_sieve=float(self.scenarios[i_scenario].get("osmotic_sieve"))
            if Os_hetero==0:
                #Os_apo=-3000 #-0.3 MPa (Enns et al., 2000) applied stress
                #-0.80 MPa (Enns et al., 2000) concentration of cortical cells, no KNO3
                Os_epi=float(Os_cortex)
                Os_exo=float(Os_cortex)
                Os_c1=float(Os_cortex)
                Os_c2=float(Os_cortex)
                Os_c3=float(Os_cortex)
                Os_c4=float(Os_cortex)
                Os_c5=float(Os_cortex)
                Os_c6=float(Os_cortex)
                Os_c7=float(Os_cortex)
                Os_c8=float(Os_cortex)
                Os_endo=float(Os_cortex)
                Os_peri=float(Os_cortex)
                Os_stele=float(Os_cortex)
                Os_comp=(float(Os_sieve)+Os_cortex)/2 #Average phloem and parenchyma
                #Os_sieve=float(Os_cortex[i_maturity][count])
            elif Os_hetero==1:
                Os_epi=-5000 #(Rygol et al. 1993) #float(Os_cortex[i_maturity][count]) #-0.80 MPa (Enns et al., 2000) concentration of cortical cells, no KNO3
                Os_exo=-5700 #(Rygol et al. 1993) #float(Os_cortex[i_maturity][count]) #-0.80 MPa (Enns et al., 2000) concentration of cortical cells, no KNO3
                Os_c1=-6400 #(Rygol et al. 1993)
                Os_c2=-7100 #(Rygol et al. 1993)
                Os_c3=-7800 #(Rygol et al. 1993)
                Os_c4=-8500 #(Rygol et al. 1993)
                Os_c5=-9000 #(Rygol et al. 1993)
                Os_c6=-9300 #(Rygol et al. 1993)
                Os_c7=-9000 #(Rygol et al. 1993)
                Os_c8=-8500 #(Rygol et al. 1993)
                Os_endo=-6200 #-0.62 MPa (Enns et al., 2000) concentration of endodermis cells, no KNO3
                Os_peri=-5000 #-0.50 MPa (Enns et al., 2000) concentration of pericycle cells, no KNO3
                Os_stele=-7400 #-0.74 MPa (Enns et al., 2000) concentration of xylem parenchyma cells, no KNO3
                Os_comp=(float(Os_sieve)-7400)/2 #Average phloem and parenchyma
                #Os_sieve=-14200 #-1.42 MPa (Pritchard, 1996) in barley phloem
            elif Os_hetero==2:
                Os_epi=-11200 #(Rygol et al. 1993) #float(Os_cortex[i_maturity][count]) #-1.26 MPa (Enns et al., 2000) concentration of cortical cells, with KNO3
                Os_exo=-11500 #(Rygol et al. 1993) #float(Os_cortex[i_maturity][count]) #-1.26 MPa (Enns et al., 2000) concentration of cortical cells, with KNO3
                Os_c1=-11800 #(Rygol et al. 1993)
                Os_c2=-12100 #(Rygol et al. 1993)
                Os_c3=-12400 #(Rygol et al. 1993)
                Os_c4=-12700 #(Rygol et al. 1993)
                Os_c5=-12850 #(Rygol et al. 1993)
                Os_c6=-12950 #(Rygol et al. 1993)
                Os_c7=-12850 #(Rygol et al. 1993)
                Os_c8=-12700 #(Rygol et al. 1993)
                Os_endo=-10500 #-1.05 MPa (Enns et al., 2000) concentration of endodermis cells, with KNO3
                Os_peri=-9200 #-0.92 MPa (Enns et al., 2000) concentration of pericycle cells, with KNO3
                Os_stele=-12100 #-1.21 MPa (Enns et al., 2000) concentration of xylem parenchyma cells, with KNO3
                Os_comp=(float(Os_sieve)-12100)/2 #Average of phloem and parenchyma
            #Os_sieve=-14200 #-1.42 MPa (Pritchard, 1996) in barley phloem
            elif Os_hetero==3:
                Os_epi=float(Os_cortex)
                Os_exo=float(Os_cortex)
                Os_c1=float(Os_cortex)
                Os_c2=float(Os_cortex)
                Os_c3=float(Os_cortex)
                Os_c4=float(Os_cortex)
                Os_c5=float(Os_cortex)
                Os_c6=float(Os_cortex)
                Os_c7=float(Os_cortex)
                Os_c8=float(Os_cortex)
                Os_endo=float((Os_cortex-5000.0)/2.0)
                Os_peri=-5000.0 #Simple case with no stele pushing water out
                Os_stele=-5000.0
                Os_comp=(float(Os_sieve)-5000.0)/2 #Average phloem and parenchyma
                #Os_sieve=-5000.0
            self.osmotic_potentials.append({
                'Os_epi': Os_epi,
                'Os_exo': Os_exo,
                'Os_c1': Os_c1,
                'Os_c2': Os_c2,
                'Os_c3': Os_c3,
                'Os_c4': Os_c4,
                'Os_c5': Os_c5,
                'Os_c6': Os_c6,
                'Os_c7': Os_c7,
                'Os_c8': Os_c8,
                'Os_endo': Os_endo,
                'Os_peri': Os_peri,
                'Os_stele': Os_stele,
                'Os_comp': Os_comp,
                'Os_sieve': Os_sieve,
            })


@dataclass
class GeneralData:
    """General file configuration.

    This class loads and manages general configuration parameters from an XML file.
    It includes display options, analysis flags, and display parameters for visualization
    and simulation purposes.

    Attributes
    ----------
    general_file : str
        Path to the general configuration XML file.
    paraview : int, optional
        Flag to enable/disable ParaView visualization (default is 0).
    paraview_wf : int, optional
        Flag to enable/disable wall flux visualization in ParaView (default is 0).
    paraview_mf : int, optional
        Flag to enable/disable membrane flux visualization in ParaView (default is 0).
    paraview_pf : int, optional
        Flag to enable/disable plasmodesmata flux visualization in ParaView (default is 0).
    paraview_wp : int, optional
        Flag to enable/disable wall potential visualization in ParaView (default is 0).
    paraview_cp : int, optional
        Flag to enable/disable cell potential visualization in ParaView (default is 0).
    sym_contagion : int, optional
        Flag to enable/disable symplastic contagion analysis (default is 0).
    apo_contagion : int, optional
        Flag to enable/disable apoplastic contagion analysis (default is 0).
    par_track : int, optional
        Flag to enable/disable particle tracking (default is 0).
    color_threshold : float, optional
        Threshold for color mapping in visualizations (default is 1.0).
    thickness_disp : float, optional
        Wall thickness for display purposes (default is 0.0).
    thickness_junction_disp : float, optional
        Junction thickness for display purposes (default is 0.0).
    radius_plasmodesm_disp : float, optional
        Plasmodesmata radius for display purposes (default is 0.0).

    Methods
    -------
    _load_general()
        Load general configuration parameters from the XML file.
    """
    # File paths
    general_file: Optional[str] = None

    # Display options - These fields will automatically get their default values
    paraview: int = 0
    paraview_wf: int = 0 # Wall flux
    paraview_mf: int = 0 # Membrane flux
    paraview_pf: int = 0 # Plasmodesmata flux
    paraview_wp: int = 0 # Wall potential
    paraview_cp: int = 0 # Cell potential
    paraview_uniwalls: int = 1 # UniX walls

    # Analysis options
    sparse_matrix: int = 0 # Sparse matrix

    # Analysis flags
    sym_contagion: int = 0 # Symplastic contagion
    apo_contagion: int = 0 # Apoplastic contagion
    par_track: int = 0

    # Display parameters
    color_threshold: float = 1.0 # Threshold for color mapping
    thickness_disp: float = 1.5 # Wall thickness display
    thickness_junction_disp: float = 2.5 # Junction thickness display
    radius_plasmodesm_disp: float = 4.0 # Plasmodesmata radius display

    # Output path
    output_path: str = "out/"
    # Operating system
    os: str=os.name

    def __post_init__(self):
        """Post-initialization method to load general configuration parameters."""
        if self.general_file is not None:
            self._load_general()
        else:
            self._set_default_values()

    def _load_general(self):
        """Load general configuration parameters from the XML file.

        This method parses the XML file to extract display options, analysis flags,
        and display parameters for visualization and simulation.
        """
        root = etree.parse(self.general_file).getroot()

        self.paraview = int(root.xpath('Paraview')[0].get("value"))
        self.paraview_wf = int(root.xpath('Paraview')[0].get("WallFlux"))
        self.paraview_mf = int(root.xpath('Paraview')[0].get("MembraneFlux"))
        self.paraview_pf = int(root.xpath('Paraview')[0].get("PlasmodesmataFlux"))
        self.paraview_wp = int(root.xpath('Paraview')[0].get("WallPot"))
        self.paraview_cp = int(root.xpath('Paraview')[0].get("CellPot"))
        self.paraview_uniwalls = int(root.xpath('UniXwalls')[0].get("value"))

        self.par_track = int(root.xpath('ParTrack')[0].get("value"))
        self.sym_contagion = int(root.xpath('Sym_Contagion')[0].get("value"))
        self.apo_contagion = int(root.xpath('Apo_Contagion')[0].get("value"))
        self.sparse_matrix = int(root.xpath('sparse')[0].get("value"))

        self.color_threshold = float(root.xpath('color_threshold')[0].get("value"))
        self.thickness_disp = float(root.xpath('thickness_disp')[0].get("value"))
        self.thickness_junction_disp = float(root.xpath('thicknessJunction_disp')[0].get("value"))
        self.radius_plasmodesm_disp = float(root.xpath('radiusPlasmodesm_disp')[0].get("value"))

    def _set_default_values(self):
        """Set default values if no file is provided."""
        # Default values are already set in the class definition


@dataclass
class GeometryData:
    """Geometry file configuration.

    This class loads and manages geometry configuration parameters from an XML file.
    It includes plant-specific parameters, maturity stages, passage cells, intercellular spaces,
    and other geometric properties relevant to the simulation.

    Attributes
    ----------
    geometry_file : str
        Path to the geometry configuration XML file.
    plant_name : str, optional
        Name of the plant (default is "").
    im_scale : float, optional
        Image scale factor (default is 1000.0).
    maturity_elems : List[Dict[str, int]], optional
        List of raw maturity elements from the XML file (default is an empty list).
    maturity_stages : List[Dict[str, int]], optional
        List of maturity stages, each represented as a dictionary with 'barrier' and 'height' keys (default is an empty list).
    n_maturity : int, optional
        Number of maturity stages (default is 0).
    passage_cell_ids : List[int], optional
        List of passage cell IDs (default is an empty list).
    intercellular_ids : List[int], optional
        List of intercellular space IDs (default is an empty list).
    interc_perims : List[float], optional
        List of intercellular perimeters (default is [0.0, 0.0, 0.0, 0.0, 0.0]).
    k_interc : float, optional
        Intercellular permeability coefficient (default is 0.0).
    cell_per_layer : numpy.ndarray, optional
        Number of cells per layer for cortex and stele (default is a 2x1 array of zeros).
    diffusion_length : numpy.ndarray, optional
        Diffusion length for cortex and stele (default is a 2x1 array of zeros).
    thickness : float, optional
        Thickness of the cell walls in microns (default is 0.0).
    pd_section : float, optional
        Plasmodesmata section area in square microns (default is 0.0).
    xylem_pieces : bool, optional
        Flag indicating whether xylem is modeled as separate pieces (default is False).

    Methods
    -------
    _load_geometry()
        Load geometry configuration parameters from the XML file.
    """
    # File paths
    geometry_file: Optional[str] = None

    # Parsed configuration
    plant_name: str = ""
    im_scale: float = 1000.0
   
    # Maturity stages
    maturity_elems: List[Dict[str, int]] = field(default_factory=list) # List of maturity dicts
    maturity_stages: List[Dict[str, int]] = field(default_factory=list) # List of maturity dicts
    n_maturity: int = 0

    # Passage cells and aerenchyma
    passage_cell_ids: List[int] = field(default_factory=list) # not used
    intercellular_ids: List[int] = field(default_factory=list)

    # Intercellular perimeters
    interc_perims: List[float] = field(default_factory=lambda: [0.0]*5)
    k_interc: float = 0.0

    # Cell layers
    cell_per_layer: np.ndarray = field(default_factory=lambda: np.zeros((2, 1)))
    diffusion_length: np.ndarray = field(default_factory=lambda: np.zeros((2, 1))) # not used

    # Geometry parameters
    thickness: float = 1.5
    pd_section: float = 7.47E-5
    xylem_pieces: bool = False

    # Additional parameters
    print_layer: int = 0
    xwalls: int = 1
    pile_up: int = 0
    interc_perim_search: int = 0

    def __post_init__(self):
        """Post-initialization method to load geometry configuration parameters."""
        if self.geometry_file is not None:
            self._load_geometry()
        else:
            self._set_default_values()

    def _load_geometry(self):
        """Load geometry configuration parameters from the XML file.

        This method parses the XML file to extract plant name, image scale, maturity stages,
        passage cells, intercellular spaces, and other geometric properties.
        """
        root = etree.parse(self.geometry_file).getroot()
        
        self.plant_name = root.xpath('Plant')[0].get("value")
        self.im_scale = float(root.xpath('im_scale')[0].get("value"))
        
        # Parse maturity stages
        self.maturity_elems = root.xpath('Maturityrange/Maturity')
        for mat in self.maturity_elems:
            self.maturity_stages.append({
                'barrier': int(mat.get("Barrier")),
                'height': float(mat.get("height"))
            })
        
        self.n_maturity = len(self.maturity_stages)
        # Parse passage cells
        passage_elems = root.xpath('passage_cell_range/passage_cell')
        self.passage_cell_ids = [int(pc.get("id")) for pc in passage_elems]
        
        # Parse aerenchyma (intercellular spaces)
        aerenchyma_elems = root.xpath('aerenchyma_range/aerenchyma')
        self.intercellular_ids = [
            int(aer.get("id")) for aer in aerenchyma_elems 
            if int(aer.get("id")) > 0 and not int(aer.get("id"))>9E5
        ]
        
        # Intercellular perimeters
        for i in range(1, 5):
            self.interc_perims[i-1] = float(
                root.xpath(f'InterC_perim{i}')[0].get("value")
            )
        self.k_interc = float(root.xpath('kInterC')[0].get("value"))
        
        # Cell layers
        cell_layer_elem = root.xpath('cell_per_layer')[0]
        self.cell_per_layer[0][0] = float(cell_layer_elem.get("cortex"))
        self.cell_per_layer[1][0] = float(cell_layer_elem.get("stele"))
        
        diff_length_elem = root.xpath('diffusion_length')[0]
        self.diffusion_length[0][0] = float(diff_length_elem.get("cortex"))
        self.diffusion_length[1][0] = float(diff_length_elem.get("stele"))
        
        self.thickness = float(root.xpath('thickness')[0].get("value")) # in microns
        self.pd_section = float(root.xpath('PD_section')[0].get("value")) # in microns^2
        self.xylem_pieces = int(root.xpath('Xylem_pieces')[0].get("flag")) == 1

    def _set_default_values(self):
        """Set default values if no file is provided."""
        # Default values are already set in the class definition
        # Add default maturity stages
        self.maturity_stages = [
            {'barrier': int(1), 'height': float(200.0), 'nlayers': int(1)}
        ]
        self.n_maturity = len(self.maturity_stages)
        # Set default passage cell ID
        self.passage_cell_ids = [-1]

    def add_maturity_stage(self, barrier: list[int], height: list[float] = [200.0]):
        if len(barrier) != len(height):
            print("barrier and height must have the same length")
            height = [height[0]] * len(barrier)

        for i, b in enumerate(barrier):
            self.maturity_stages.append({'barrier': b, 'height': height[i], 'nlayers': int(1)})
        self.n_maturity = len(self.maturity_stages)

    def set_maturity_stages(self, barrier: list[int], height: list[float] = [200.0]):
        if len(barrier) != len(height):
            print("barrier and height must have the same length")
            height = [height[0]] * len(barrier)
            
        self.maturity_stages = []
        for i, b in enumerate(barrier):
            self.maturity_stages.append({'barrier': b, 'height': height[i], 'nlayers': int(1)})
        self.n_maturity = len(self.maturity_stages)

    def add_passage_cell(self, cid: int):
        self.passage_cell_ids.append(cid)

    def add_aer_space(self, cid: int):
        self.intercellular_ids.append(cid)

    def get_barrier(self, i: int) -> int:
        return self.maturity_stages[i]['barrier']

    def get_height(self, i: int) -> float:
        return self.maturity_stages[i]['height']

@dataclass
class HormonesData:
    """Hormones file configuration.

    This class loads and manages hormone configuration parameters from an XML file.
    It includes hormone movement parameters, active transport carriers, symplastic and apoplastic contagion,
    and contact range information.

    Attributes
    ----------
    hormone_file : str
        Path to the hormone configuration XML file.
    degrad1 : float, optional
        Degradation constant for hormone 1 (default is 0.0).
    diff_pd1 : float, optional
        Diffusivity of hormone 1 through plasmodesmata (default is 0.0).
    diff_pw1 : float, optional
        Diffusivity of hormone 1 through cell walls (default is 0.0).
    d2o1 : bool, optional
        Flag indicating whether hormone 1 is D2O (deuterium oxide) labeled (default is False).
    carrier_elems : List[Any], optional
        List of active transport carrier elements from the XML file (default is an empty list).
    sym_zombie0 : List[int], optional
        List of source cell IDs for symplastic contagion (default is an empty list).
    sym_cc : List[float], optional
        List of concentrations for symplastic contagion sources (default is an empty list).
    sym_target : List[int], optional
        List of target cell IDs for symplastic contagion (default is an empty list).
    sym_immune : List[int], optional
        List of immune cell IDs for symplastic contagion (default is an empty list).
    apo_zombie0 : List[int], optional
        List of source cell IDs for apoplastic contagion (default is an empty list).
    apo_cc : List[float], optional
        List of concentrations for apoplastic contagion sources (default is an empty list).
    apo_target : List[int], optional
        List of target cell IDs for apoplastic contagion (default is an empty list).
    apo_immune : List[int], optional
        List of immune cell IDs for apoplastic contagion (default is an empty list).
    contact : List[int], optional
        List of cell IDs in the contact range (default is an empty list).

    Methods
    -------
    _load_hormones()
        Load hormone and carrier configuration parameters from the XML file.
    """
    # File paths
    hormone_file: Optional[str] = None

    
    # Hormone movement parameters
    degrad1: float = 48.0
    diff_pd1: float = 0.0035
    diff_pw1: float = 0.0035
    d2o1: bool = False

    # Active transport carriers - Use field(default_factory=list) for mutable defaults
    carrier_elems: List[Any] = field(default_factory=list)

    # Symplastic contagion
    sym_zombie0: List[int] = field(default_factory=lambda: [-1])
    sym_cc: List[float] = field(default_factory=lambda: [1.0])
    sym_target: List[int] = field(default_factory=lambda: [-1, -1])
    sym_immune: List[int] = field(default_factory=lambda: [-1])

    # Apoplastic contagion
    apo_zombie0: List[int] = field(default_factory=lambda: [-1])
    apo_cc: List[float] = field(default_factory=lambda: [1.0])
    apo_target: List[int] = field(default_factory=lambda: [-1, -1])
    apo_immune: List[int] = field(default_factory=lambda: [-1])

    # Contact range
    contact: List[int] = field(default_factory=lambda: [-1])

    def __post_init__(self):
        """Post-initialization method to load hormone configuration parameters."""
        if self.hormone_file is not None:
            self._load_hormones()
        else:
            self._set_default_values()

    def _load_hormones(self):
        """Load hormone and carrier configuration parameters from the XML file.

        This method parses the XML file to extract hormone movement parameters,
        active transport carriers, symplastic and apoplastic contagion details,
        and contact range information.
        """
        root = etree.parse(self.hormone_file).getroot()

        # Hormone movement parameters
        self.degrad1 = float(root.xpath('Hormone_movement/Degradation_constant_H1')[0].get("value"))
        self.diff_pd1 = float(root.xpath('Hormone_movement/Diffusivity_PD_H1')[0].get("value"))
        self.diff_pw1 = float(root.xpath('Hormone_movement/Diffusivity_PW_H1')[0].get("value"))
        self.d2o1 = int(root.xpath('Hormone_movement/H1_D2O')[0].get("flag")) == 1

        # Parse active transport carriers
        self.carrier_elems = root.xpath('Hormone_active_transport/carrier_range/carrier')

        # Parse symplastic contagion
        sym_source_elems = root.xpath('Sym_Contagion/source_range/source')
        self.sym_zombie0 = [int(source.get("id")) for source in sym_source_elems]
        self.sym_cc = [float(source.get("concentration")) for source in sym_source_elems]

        sym_target_elems = root.xpath('Sym_Contagion/target_range/target')
        self.sym_target = [int(target.get("id")) for target in sym_target_elems]

        sym_immune_elems = root.xpath('Sym_Contagion/immune_range/immune')
        self.sym_immune = [int(immune.get("id")) for immune in sym_immune_elems]
        # Parse apoplastic contagion
        apo_source_elems = root.xpath('Apo_Contagion/source_range/source')
        self.apo_zombie0 = [int(source.get("id")) for source in apo_source_elems]
        self.apo_cc = [float(source.get("concentration")) for source in apo_source_elems]

        apo_target_elems = root.xpath('Apo_Contagion/target_range/target')
        self.apo_target = [int(target.get("id")) for target in apo_target_elems]
        
        apo_immune_elems = root.xpath('Apo_Contagion/immune_range/immune')
        self.apo_immune = [int(immune.get("id")) for immune in apo_immune_elems]

        # Parse contact range
        contact_elems = root.xpath('Contactrange/Contact')
        self.contact = [int(contact.get("id")) for contact in contact_elems]
    
    def _set_default_values(self):
        """Set default values if no file is provided."""
        self.carrier_elems = [{'tissue': '-1', 'constant': '7.9E-11', 'direction': '-1'}]


@dataclass
class HydraulicData:
    """Hydraulic file configuration.

    This class loads and manages hydraulic configuration parameters from an XML file.
    It includes hydraulic parameter elements, counts, single-value parameters,
    plasmodesmata (PD) height parameters for different tissue interfaces,
    conductance parameters, and processed parameter arrays.

    Attributes
    ----------
    hydraulics_file : str
        Path to the hydraulic configuration XML file.
    kw_elems : List[Any], optional
        List of raw cell wall hydraulic conductivity elements from the XML file (default is an empty list).
    kw_barrier_elems : List[Any], optional
        List of raw cell wall barrier hydraulic conductivity elements from the XML file (default is an empty list).
    kaqp_elems : List[Any], optional
        List of raw aquaporin hydraulic conductivity elements from the XML file (default is an empty list).
    kpl_elems : List[Any], optional
        List of raw plasmodesmata hydraulic conductivity elements from the XML file (default is an empty list).
    xcontactrange : List[Any], optional
        List of raw xylem contact range elements from the XML file (default is an empty list).
    path_hydraulics : List[Any], optional
        List of output paths for hydraulic scenarios (default is an empty list).
    n_kw : int, optional
        Number of cell wall hydraulic conductivity elements (default is 0).
    n_kw_barrier : int, optional
        Number of cell wall barrier hydraulic conductivity elements (default is 0).
    n_kaqp : int, optional
        Number of aquaporin hydraulic conductivity elements (default is 0).
    n_kpl : int, optional
        Number of plasmodesmata hydraulic conductivity elements (default is 0).
    n_xcontact : int, optional
        Number of xylem contact range elements (default is 0).
    n_hydraulics : int, optional
        Number of hydraulic scenarios (default is 1).
    kmb : float, optional
        Membrane hydraulic conductivity (default is 0.0).
    ratio_cortex : float, optional
        Ratio related to cortex hydraulic properties (default is 0.0).
    fplxheight : float, optional
        Default plasmodesmata height (default is 0.0).
    fplxheight_epi_exo : float, optional
        Plasmodesmata height for epidermis-exodermis interface (default is 0.0).
    fplxheight_outer_cortex : float, optional
        Plasmodesmata height for outer cortex interface (default is 0.0).
    fplxheight_cortex_cortex : float, optional
        Plasmodesmata height for cortex-cortex interface (default is 0.0).
    fplxheight_cortex_endo : float, optional
        Plasmodesmata height for cortex-endodermis interface (default is 0.0).
    fplxheight_endo_endo : float, optional
        Plasmodesmata height for endodermis-endodermis interface (default is 0.0).
    fplxheight_endo_peri : float, optional
        Plasmodesmata height for endodermis-pericycle interface (default is 0.0).
    fplxheight_peri_peri : float, optional
        Plasmodesmata height for pericycle-pericycle interface (default is 0.0).
    fplxheight_peri_stele : float, optional
        Plasmodesmata height for pericycle-stele interface (default is 0.0).
    fplxheight_stele_stele : float, optional
        Plasmodesmata height for stele-stele interface (default is 0.0).
    fplxheight_stele_comp : float, optional
        Plasmodesmata height for stele-companion cell interface (default is 0.0).
    fplxheight_peri_comp : float, optional
        Plasmodesmata height for pericycle-companion cell interface (default is 0.0).
    fplxheight_comp_comp : float, optional
        Plasmodesmata height for companion cell-companion cell interface (default is 0.0).
    fplxheight_comp_sieve : float, optional
        Plasmodesmata height for companion cell-sieve tube interface (default is 0.0).
    fplxheight_peri_sieve : float, optional
        Plasmodesmata height for pericycle-sieve tube interface (default is 0.0).
    fplxheight_stele_sieve : float, optional
        Plasmodesmata height for stele-sieve tube interface (default is 0.0).
    
    k_sieve : float, optional
        Sieve tube hydraulic conductance (default is 0.0).

    k_xyl : float, optional
        Xylem vessel axial hydraulic conductance (default is 0.0).
    kw : List[float], optional
        Processed list of cell wall hydraulic conductivity values (default is an empty list).
    kw_barrier : List[float], optional
        Processed list of cell wall barrier hydraulic conductivity values (default is an empty list).

    Methods
    -------
    _load_hydraulics()
        Load hydraulic configuration parameters from the XML file.
    """
    # File paths
    hydraulics_file: Optional[str] = None

    # Hydraulic parameter elements (raw from XML)
    kw_elems: List[Any] = field(default_factory=list)
    kw_barrier_elems: List[Any] = field(default_factory=list)
    kaqp_elems: List[Any] = field(default_factory=list)
    kpl_elems: List[Any] = field(default_factory=list)
    xcontactrange: List[Any] = field(default_factory=lambda: [0])
    path_hydraulics: List[Any] = field(default_factory=list)

    # Counts
    n_kw: int = 1
    n_kw_barrier: int = 1
    n_kaqp: int = 1
    n_kpl: int = 1
    n_xcontact: int = 1
    n_hydraulics: int = 1

    # Single-value parameters
    kmb: float = 3.0E-5
    ratio_cortex: float = 1.0
    
    # PD height (Fplxheight) parameters for different tissue interfaces
    fplxheight: float = 8.0E5
    fplxheight_epi_exo: float = 1.08E6
    fplxheight_outer_cortex: float = 2.28E6
    fplxheight_cortex_cortex: float = 8.6E5
    fplxheight_cortex_endo: float = 8.8E5
    fplxheight_endo_endo: float = 6.4E5
    fplxheight_endo_peri: float = 9.6E5
    fplxheight_peri_peri: float = 7.0E5 # not use
    fplxheight_peri_stele: float = 1.08E6
    fplxheight_stele_stele: float = 6.4E5
    fplxheight_stele_comp: float = 9.8E5
    fplxheight_peri_comp: float = 7.0E5
    fplxheight_comp_comp: float = 6.8E5
    fplxheight_comp_sieve: float = 1.76E6
    fplxheight_peri_sieve: float = 7.2E5
    fplxheight_stele_sieve: float = 9.0E5
    
    # Conductance parameters
    axial_conductance_source: int = 1
    k_sieve_elems: List[Any] = field(default_factory=list)
    k_xyl_elems: List[Any] = field(default_factory=list)
    k_sieve: float = 1.0E-6  # Sieve tube hydraulic conductance
    K_axial: Optional[np.ndarray] = None
    k_xyl: float = 1.0E-6   # Xylem vessel axial hydraulic conductance
    K_xyl_spec: float = 1.0E-6   # Xylem vessel axial hydraulic conductance ## remove?

    # Root conductivities
    conductivities: List[Dict[str, Any]] = field(default_factory=list)

    # Matrices for Doussan calculations
    matrix_W: Optional[np.ndarray] = None
    matrix_C: Optional[np.ndarray] = None
    matrix_ApoC: Optional[np.ndarray] = None
    matrix_SymC: Optional[np.ndarray] = None
    rhs_C: Optional[np.ndarray] = None
    rhs_ApoC: Optional[np.ndarray] = None
    rhs_SymC: Optional[np.ndarray] = None
    rhs: Optional[np.ndarray] = None
    rhs_s: Optional[np.ndarray] = None
    rhs_x: Optional[np.ndarray] = None
    rhs_p: Optional[np.ndarray] = None
    
    # Processed parameter arrays
    kw: List[float] = field(default_factory=lambda: [0.00024])
    kw_barrier: List[float] = field(default_factory=lambda: [1.00E-16])
    kaqp: List[Dict[str, float]] = field(default_factory=lambda: [{'value': 0.000430, 'cortex_factor': 1.0, 'endo_factor': 1.0, 'epi_factor': 1.0, 'exo_factor': 1.0, 'stele_factor': 1.0}])
    kpl: List[Dict[str, float]] = field(default_factory=lambda: [{'value': 5.3E-12, 'phloem_companion_cell_factor': 1.0, 'pericycle_phloem_pole_factor': 1.0, 'phloem_sieve_tube_factor': 1.0, 'cortex_factor': 1.0}])

    def __post_init__(self):
        """Post-initialization method to load hydraulic configuration parameters."""
        if self.hydraulics_file is not None:
            self._load_hydraulics()
        else:
            self._set_default_values()

    def _load_hydraulics(self):
        """Load hydraulic configuration parameters from the XML file.

        This method parses the XML file to extract hydraulic parameter elements,
        single-value parameters, plasmodesmata height parameters, conductance parameters,
        contact range, and output paths. It also processes parameter arrays.
        """
        root = etree.parse(self.hydraulics_file).getroot()

        # Parse different hydraulic parameter sets
        self.kw_elems = root.xpath('kwrange/kw')
        self.kw_barrier_elems = root.xpath('kw_barrier_range/kw_barrier')
        self.kaqp_elems = root.xpath('kAQPrange/kAQP')
        self.kpl_elems = root.xpath('Kplrange/Kpl')

        self.n_kw = len(self.kw_elems)
        self.n_kw_barrier = len(self.kw_barrier_elems)
        self.n_kaqp = len(self.kaqp_elems)
        self.n_kpl = len(self.kpl_elems)

        # Extract single-value parameters
        self.kmb = float(root.xpath('km')[0].get("value"))
        self.ratio_cortex = float(root.xpath('ratio_cortex')[0].get("value"))
        
        # PD height parameters
        self.fplxheight = float(root.xpath('Fplxheight')[0].get("value"))
        self.fplxheight_epi_exo = float(root.xpath('Fplxheight_epi_exo')[0].get("value"))
        self.fplxheight_outer_cortex = float(root.xpath('Fplxheight_outer_cortex')[0].get("value"))
        self.fplxheight_cortex_cortex = float(root.xpath('Fplxheight_cortex_cortex')[0].get("value"))
        self.fplxheight_cortex_endo = float(root.xpath('Fplxheight_cortex_endo')[0].get("value"))
        self.fplxheight_endo_endo = float(root.xpath('Fplxheight_endo_endo')[0].get("value"))
        self.fplxheight_endo_peri = float(root.xpath('Fplxheight_endo_peri')[0].get("value"))
        self.fplxheight_peri_peri = float(root.xpath('Fplxheight_peri_peri')[0].get("value"))
        self.fplxheight_peri_stele = float(root.xpath('Fplxheight_peri_stele')[0].get("value"))
        self.fplxheight_stele_stele = float(root.xpath('Fplxheight_stele_stele')[0].get("value"))
        self.fplxheight_stele_comp = float(root.xpath('Fplxheight_stele_comp')[0].get("value"))
        self.fplxheight_peri_comp = float(root.xpath('Fplxheight_peri_comp')[0].get("value"))
        self.fplxheight_comp_comp = float(root.xpath('Fplxheight_comp_comp')[0].get("value"))
        self.fplxheight_comp_sieve = float(root.xpath('Fplxheight_comp_sieve')[0].get("value"))
        self.fplxheight_peri_sieve = float(root.xpath('Fplxheight_peri_sieve')[0].get("value"))
        self.fplxheight_stele_sieve = float(root.xpath('Fplxheight_stele_sieve')[0].get("value"))
        
        # Conductance parameters
        # 1: Poiseuille law (based on cross-section area); 2: Prescribed here below (for all sieve tubes, and vessel per vessel)
        self.axial_conductance_source = int(root.xpath('Kax_source')[0].get("value")) if root.xpath('Kax_source') else 1
        self.k_sieve_elems = root.xpath('K_sieve_range/K_sieve')
        self.k_xyl_elems = root.xpath('K_xyl_range/K_xyl')
        self.k_sieve = [float(k_sieve.get("value")) for k_sieve in self.k_sieve_elems] if self.k_sieve_elems else [0.0]
        self.k_xyl = [float(k_xyl.get("value")) for k_xyl in self.k_xyl_elems] if self.k_xyl_elems else [0.0]
        
        # Contact range
        self.xcontactrange = root.xpath('Xcontactrange/Xcontact')
        self.n_xcontact = len(self.xcontactrange)
        
        # Output paths
        self.path_hydraulics = root.xpath('path_hydraulics/Output')
        self.n_hydraulics = len(self.path_hydraulics) if self.path_hydraulics else 1
        
        # Process parameter arrays
        self.kw = [float(kw.get("value")) for kw in self.kw_elems] if self.kw_elems else [0.00024]
        self.kw_barrier = [float(kw_barrier.get("value")) for kw_barrier in self.kw_barrier_elems] if self.kw_barrier_elems else [1.00E-16]

        self.kaqp = []
        for kaqp_elem in self.kaqp_elems if self.kaqp_elems else [{'value': 0.000430, 'cortex_factor': 1.0, 'endo_factor': 1.0, 'epi_factor': 1.0, 'exo_factor': 1.0, 'stele_factor': 1.0}]:
            kaqp_dict = {'value': float(kaqp_elem.get("value"))}
            kaqp_dict['cortex_factor'] = float(kaqp_elem.get("cortex_factor"))
            kaqp_dict['endo_factor'] = float(kaqp_elem.get("endo_factor"))
            kaqp_dict['epi_factor'] = float(kaqp_elem.get("epi_factor"))
            kaqp_dict['exo_factor'] = float(kaqp_elem.get("exo_factor"))
            kaqp_dict['stele_factor'] = float(kaqp_elem.get("stele_factor"))
            self.kaqp.append(kaqp_dict)

        self.kpl = []
        for kpl_elem in self.kpl_elems if self.kpl_elems else [{'value': 5.3E-12, 'PCC_factor': 1.0, 'PPP_factor': 1.0, 'cortex_factor': 1.0}]:
            kpl_dict = {'value': float(kpl_elem.get("value"))}
            kpl_dict['phloem_companion_cell_factor'] = float(kpl_elem.get("PCC_factor")) # 
            kpl_dict['pericycle_phloem_pole_factor'] = float(kpl_elem.get("PPP_factor")) # 
            kpl_dict['phloem_sieve_tube_factor'] = float(kpl_elem.get("PST_factor")) # 
            kpl_dict['cortex_factor'] = float(kpl_elem.get("cortex_factor"))
            kpl_dict['endo_in_factor'] = float(kpl_elem.get("endo_in_factor"))
            kpl_dict['endo_out_factor'] = float(kpl_elem.get("endo_out_factor"))

            self.kpl.append(kpl_dict)

    def _set_default_values(self):
        """Set default values if no file is provided."""
        self.kw_barrier_elems = [{'value': 1.00E-16, 'Casp': 1.00E-16, 'Sub': 1.00E-16, 'Sub_in': 1.00E-16, 'Sub_out': 1.00E-16}]
        self.kw_elems = [{'value': 0.00024}]

    def get_kw_value(self, h: int) -> float:
        """Get the kw value based on the scenario index."""
        if self.n_kw == self.n_hydraulics:
            return self.kw[h]
        elif self.n_kw == 1:
            return self.kw[0]
        else:
            return self.kw[int(h/(self.n_kaqp*self.n_kpl))%self.n_kw]

    def get_kw_barrier_values(self, h: int) -> Tuple[float, List[float]]:
        """Get the kw_barrier values based on the scenario index."""
        if self.n_kw_barrier == self.n_hydraulics:
            kw_barrier_casparian = float(self.kw_barrier_elems[h].get("Casp"))
            kw_barrier_suberin = float(self.kw_barrier_elems[h].get("Sub"))
            kw_barrier_suberin_in = float(self.kw_barrier_elems[h].get("Sub_in"))
            kw_barrier_suberin_out = float(self.kw_barrier_elems[h].get("Sub_out"))
        elif self.n_kw_barrier == 1:
            kw_barrier_casparian = float(self.kw_barrier_elems[0].get("Casp"))
            kw_barrier_suberin = float(self.kw_barrier_elems[0].get("Sub"))
            kw_barrier_suberin_in = float(self.kw_barrier_elems[0].get("Sub_in"))
            kw_barrier_suberin_out = float(self.kw_barrier_elems[0].get("Sub_out"))
        else:
            index = int(h/(self.n_kaqp*self.n_kpl*self.n_kw))%self.n_kw_barrier
            kw_barrier_casparian = float(self.kw_barrier_elems[index].get("Casp"))
            kw_barrier_suberin = float(self.kw_barrier_elems[index].get("Sub"))
            kw_barrier_suberin_in = float(self.kw_barrier_elems[index].get("Sub_in"))
            kw_barrier_suberin_out = float(self.kw_barrier_elems[index].get("Sub_out"))

        # Use the general 'suberin' value if specific ones are missing
        if kw_barrier_suberin_in is None:
            kw_barrier_suberin_in = float(kw_barrier_suberin) if kw_barrier_suberin is not None else 1E-16
        if kw_barrier_suberin_out is None:
            kw_barrier_suberin_out = float(kw_barrier_suberin) if kw_barrier_suberin is not None else 1E-16

        kw_barrier_suberin_all = [float(kw_barrier_suberin_in), float(kw_barrier_suberin_out)]

        return kw_barrier_casparian, kw_barrier_suberin_all

    def get_wall_conductivities(self, barrier: int, kw: float, kw_barrier_casparian: float, kw_barrier_suberin: List[float]) -> Dict[str, float]:
        """Get wall conductivities based on barrier type."""
        barrier_configs = {
            0: {  # No Casparian strip
                'kw_endo_endo': kw,
                'kw_puncture': kw,
                'kw_exo_exo': kw,
                'kw_exo_epi': kw,
                'kw_exo_cortex': kw,
                'kw_cortex_cortex': kw,
                'kw_endo_peri': kw,
                'kw_endo_cortex': kw,
                'kw_passage': kw
            },
            1: {  # Endodermis radial walls
                'kw_endo_endo': kw_barrier_casparian,
                'kw_exo_exo': kw,
                'kw_exo_epi': kw,
                'kw_exo_cortex': kw,
                'kw_cortex_cortex': kw,
                'kw_endo_peri': kw,
                'kw_endo_cortex': kw,
                'kw_passage': kw
            },
            2: {  # Endodermis with passage cells
                'kw_endo_endo': kw_barrier_casparian,
                'kw_exo_exo': kw,
                'kw_exo_epi': kw,
                'kw_exo_cortex': kw,
                'kw_cortex_cortex': kw,
                'kw_endo_peri': kw_barrier_suberin[0],
                'kw_endo_cortex': kw_barrier_suberin[1],
                'kw_passage': kw
            },
            3: {  # Endodermis full
                'kw_endo_endo': kw_barrier_casparian,
                'kw_exo_exo': kw,
                'kw_exo_epi': kw,
                'kw_exo_cortex': kw,
                'kw_cortex_cortex': kw,
                'kw_endo_peri': kw_barrier_suberin[0],
                'kw_endo_cortex': kw_barrier_suberin[1],
                'kw_passage': kw_barrier_suberin[0]
            },
            4: {  # Endodermis full and exodermis radial walls
                'kw_endo_endo': kw_barrier_casparian,
                'kw_exo_exo': kw_barrier_casparian,
                'kw_exo_epi': kw,
                'kw_exo_cortex': kw,
                'kw_cortex_cortex': kw,
                'kw_endo_peri': kw_barrier_suberin[0],
                'kw_endo_cortex': kw_barrier_suberin[1],
                'kw_passage': kw_barrier_suberin[0]
            },
            5: {  # Endodermal & exodermal Casparian strips
                'kw_endo_endo': kw_barrier_casparian,
                'kw_exo_exo': kw_barrier_casparian,
                'kw_exo_epi': kw,
                'kw_exo_cortex': kw,
                'kw_cortex_cortex': kw,
                'kw_endo_peri': kw,
                'kw_endo_cortex': kw,
                'kw_passage': kw
            },
            6: {  # Exodermis full and endodermis radial walls
                'kw_endo_endo': kw_barrier_casparian,
                'kw_exo_exo': kw_barrier_casparian,
                'kw_exo_epi': kw_barrier_suberin[1],
                'kw_exo_cortex': kw_barrier_suberin[0],
                'kw_cortex_cortex': kw,
                'kw_endo_peri': kw,
                'kw_endo_cortex': kw,
                'kw_passage': kw
            },
            7: {  # Exodermis radial walls
                'kw_endo_endo': kw,
                'kw_exo_exo': kw_barrier_casparian,
                'kw_exo_epi': kw,
                'kw_exo_cortex': kw,
                'kw_cortex_cortex': kw,
                'kw_endo_peri': kw,
                'kw_endo_cortex': kw,
                'kw_passage': kw
            },
            8: {  # Exodermis full suberized and endodermis full suberized
                'kw_endo_endo': kw_barrier_casparian,
                'kw_exo_exo': kw_barrier_casparian,
                'kw_exo_epi': kw_barrier_suberin[1],
                'kw_exo_cortex': kw_barrier_suberin[0],
                'kw_cortex_cortex': kw,
                'kw_endo_peri': kw_barrier_suberin[0],
                'kw_endo_cortex': kw_barrier_suberin[1],
                'kw_passage': kw
            },
            9: {  # Lignin Cap
                'kw_endo_endo': kw_barrier_casparian,
                'kw_exo_exo': kw_barrier_casparian,
                'kw_exo_epi': kw_barrier_suberin[1],
                'kw_exo_cortex': kw,
                'kw_cortex_cortex': kw,
                'kw_endo_peri': kw,
                'kw_endo_cortex': kw,
                'kw_passage': kw
            }
        }

        # Get the configuration for the specified barrier
        config = barrier_configs.get(barrier, barrier_configs[0])
        return config

    def get_plasmodesmatal_conductance(self, h: int) -> Dict[str, float]:
        """Get plasmodesmata conductance."""
        if self.n_kpl == self.n_hydraulics:
            iPD = h
        elif self.n_kpl == 1:
            iPD = 0
        else:
            iPD = int(h/self.n_kaqp)%self.n_kpl

        kpl = float(self.kpl[iPD].get("value"))

        # float() argument must be a string or a real number, not 'NoneType'
        if self.kpl[iPD].get("stele_factor") is not None:
            stele_factor = float(self.kpl[iPD].get("stele_factor"))
        else:
            stele_factor = 1.0

        if self.kpl[iPD].get("endo_in_factor") is not None:
            endo_in_factor = float(self.kpl[iPD].get("endo_in_factor"))
        else:
            endo_in_factor = 1.0

        if self.kpl[iPD].get("endo_out_factor") is not None:
            endo_out_factor = float(self.kpl[iPD].get("endo_out_factor"))
        else:
            endo_out_factor = 1.0

        if self.kpl[iPD].get("exo_factor") is not None:
            exo_factor = float(self.kpl[iPD].get("exo_factor"))
        else:
            exo_factor = 1.0

        if self.kpl[iPD].get("epi_factor") is not None:
            epi_factor = float(self.kpl[iPD].get("epi_factor"))
        else:
            epi_factor = 1.0

        if self.kpl[iPD].get("cortex_factor") is not None:
            cortex_factor = float(self.kpl[iPD].get("cortex_factor"))
        else:
            cortex_factor = 1.0

        if self.kpl[iPD].get('phloem_companion_cell_factor') is not None:
            phloem_companion_cell_factor = float(self.kpl[iPD].get('phloem_companion_cell_factor'))
        else:
            phloem_companion_cell_factor = 1.0

        if self.kpl[iPD].get('phloem_pericycle_pole_factor') is not None:
            phloem_pericycle_pole_factor = float(self.kpl[iPD].get('phloem_pericycle_pole_factor'))
        else:
            phloem_pericycle_pole_factor = 1.0

        if self.kpl[iPD].get('phloem_sieve_tube_factor') is not None:
            phloem_sieve_tube_factor = float(self.kpl[iPD].get('phloem_sieve_tube_factor'))
        else:
            phloem_sieve_tube_factor = 1.0

        config = {
            'kpl': kpl,
            'kpl_stele': kpl * stele_factor,
            'kpl_endo_in': kpl * endo_in_factor,
            'kpl_endo_out': kpl * endo_out_factor,
            'kpl_exo': kpl * exo_factor,
            'kpl_epi': kpl * epi_factor,
            'kpl_cortex': kpl * cortex_factor,
            'phloem_companion_cell_factor': phloem_companion_cell_factor, # PCC
            'phloem_pericycle_pole_factor': phloem_pericycle_pole_factor, # PPP
            'phloem_sieve_tube_factor': phloem_sieve_tube_factor, # PST
            'cortex_factor':cortex_factor
        }
        return config

    def get_aquaporin_contributions(self, h: int) -> Dict[str, float]:
        """Get aquaporin contributions to membrane hydraulic conductivity."""
        if self.n_kaqp == self.n_hydraulics:
            iAQP = h
        elif self.n_kaqp == 1:
            iAQP = 0
        else:
            iAQP = h%self.n_kaqp

        kaqp = float(self.kaqp[iAQP].get("value"))
        config = {
            'kaqp': kaqp,
            'kaqp_stele': kaqp * float(self.kaqp[iAQP].get("stele_factor")),
            'kaqp_endo': kaqp * float(self.kaqp[iAQP].get("endo_factor")),
            'kaqp_exo': kaqp * float(self.kaqp[iAQP].get("exo_factor")),
            'kaqp_epi': kaqp * float(self.kaqp[iAQP].get("epi_factor")),
            'kaqp_cortex': kaqp * float(self.kaqp[iAQP].get("cortex_factor"))
        }
        return config


        
def parse_cellset(cellset_file: str) -> Dict[str, Any]:
    """
    Parse the cell set XML file to extract wall and cell information
        
    Parameters
    ----------
    cellset_file : str
        Path to the cellset XML file
            
    Returns
    -------
    Dict containing:
        - points: wall point coordinates
        - walls: wall connectivity
        - cells: cell definitions
        - cell_to_wall: mapping of cells to walls
    """
    tree = etree.parse(cellset_file)
    root = tree.getroot()
        
    return {
        'root': root,
        'points': root.xpath('walls/wall/points'),
        'walls': root.xpath('cells/cell/walls/wall'),
        'cells': root.xpath('cells/cell'),
        'cell_to_wall': root.xpath('cells/cell/walls')
    }



@dataclass
class InData:
    """Master configuration dataclass.

    This class encapsulates all individual configuration loaders
    (`BoundaryData`, `GeneralData`, `GeometryData`, `HormonesData`, `HydraulicData`)
    and provides a unified interface for loading and accessing configurations.

    Attributes
    ----------
    boundary_config : BoundaryData
        Boundary conditions configuration.
    general_config : GeneralData
        General configuration.
    geometry_config : GeometryData
        Geometry configuration.
    hormones_config : HormonesData
        Hormones configuration.
    hydraulic_config : HydraulicData
        Hydraulic configuration.
    cellset_data : Dict[str, Any]
        Parsed cellset data from the XML file.
    """

    # File paths
    boundary_file: Optional[str] = None
    general_file: Optional[str] = None
    geometry_file: Optional[str] = None
    hormones_file: Optional[str] = None
    hydraulics_file: Optional[str] = None
    cellset_file: Optional[str] = None

    # Sub-configurations
    boundary: BoundaryData = field(init=False)
    general: GeneralData = field(init=False)
    geometry: GeometryData = field(init=False)
    hormones: HormonesData = field(init=False)
    hydraulic: HydraulicData = field(init=False)
    cellset_data: Dict[str, Any] = field(init=False)

    def __post_init__(self):
        """Post-initialization method to load all configurations."""
        self._load_all_configs()

    def _load_all_configs(self):
        """Load all configurations from their respective XML files."""

        # Initialize and load sub-configurations
        self.boundary = BoundaryData(bc_file=self.boundary_file)
        self.general = GeneralData(general_file=self.general_file)
        self.geometry = GeometryData(geometry_file=self.geometry_file)
        self.hormones = HormonesData(hormone_file=self.hormones_file)
        self.hydraulic = HydraulicData(hydraulics_file=self.hydraulics_file)
        self.cellset_data = parse_cellset(cellset_file= self.cellset_file) if self.cellset_file is not None else {}

    def info(self, verbose: bool = True) -> None:
        """
        Display a summary of all inputs and configurations.

        Parameters
        ----------
        verbose : bool, optional
            If True, display detailed information about each configuration.
            If False, display only a brief summary. Default is False.
        """
        description = "\n=== InData Configuration Summary ===\n"

        # Display file paths
        description += "\nFile Paths:\n"
        description += f"  Boundary file: {self.boundary_file if self.boundary_file else 'Using defaults'}\n"
        description += f"  General file: {self.general_file if self.general_file else 'Using defaults'}\n"
        description += f"  Geometry file: {self.geometry_file if self.geometry_file else 'Using defaults'}\n"
        description += f"  Hormones file: {self.hormones_file if self.hormones_file else 'Using defaults'}\n"
        description += f"  Hydraulics file: {self.hydraulics_file if self.hydraulics_file else 'Using defaults'}\n"
        description += f"  Cellset file: {self.cellset_file}\n"

        # Display boundary configuration
        description += "\nBoundary Configuration:\n"
        description += f"  Number of scenarios: {self.boundary.n_scenarios}\n"
        description += f"  Solute transport flag: {self.boundary.c_flag}\n"

        if verbose and self.boundary.scenarios:
            description += "  Scenarios:\n"
            for i, scenario in enumerate(self.boundary.scenarios):
                description += f"    Scenario {i+1}:\n"
                for key, value in scenario.items():
                    description += f"      {key}: {value}\n"

        # Display general configuration
        description += "\nGeneral Configuration:\n"
        description += f"  Symplastic contagion: {self.general.sym_contagion}\n"
        description += f"  Apoplastic contagion: {self.general.apo_contagion}\n"
        description += f"  Particle tracking: {self.general.par_track}\n"

        # Display geometry configuration
        description += "\nGeometry Configuration:\n"
        description += f"  Plant name: {self.geometry.plant_name}\n"
        description += f"  Image scale: {self.geometry.im_scale}\n"
        description += f"  Number of maturity stages: {self.geometry.n_maturity}\n"
        description += f"  Passage cell IDs: {self.geometry.passage_cell_ids}\n"
        description += f"  Cell wall thickness: {self.geometry.thickness} µm \n"
        description += f"  Plasmodesmata section: {self.geometry.pd_section} µm²\n"

        if verbose and self.geometry.maturity_stages:
            description += "  Maturity stages:\n"
            for i, stage in enumerate(self.geometry.maturity_stages):
                description +=f"    Stage {i+1}:\n"
                for key, value in stage.items():
                    description +=f"      {key}: {value}\n"

        # Display hormones configuration
        description += "\nHormones Configuration:\n"
        description += f"  Degradation constant: {self.hormones.degrad1}\n"
        description += f"  PD diffusivity: {self.hormones.diff_pd1}\n"
        description += f"  PW diffusivity: {self.hormones.diff_pw1}\n"
        description += f"  D2O flag: {self.hormones.d2o1}\n"
        description += f"  Number of carriers: {len(self.hormones.carrier_elems)}\n"

        if verbose and self.hormones.carrier_elems:
            description +="  Carriers:\n"
            for i, carrier in enumerate(self.hormones.carrier_elems):
                description +=f"    Carrier {i+1}:\n"
                for key, value in carrier.items():
                    description +=f"      {key}: {value}\n"

        # Display hydraulic configuration
        description += "\nHydraulic Configuration:"
        description += f"  Number of scenarios: {self.hydraulic.n_hydraulics}"
        description += f"  Membrane conductivity: {self.hydraulic.kmb}"
        description += f"  kAQP: {self.hydraulic.kaqp}"
        description += f"  Plasmodesmata conductance: {self.hydraulic.kpl}"
        description += f"  Cell wall conductivity: {self.hydraulic.kw}"
        description += f"  Xylem conductance: {self.hydraulic.k_xyl}"

        # Display cellset data
        description += "\nCellset Data:\n"
        description += f"  Number of cells: {len(self.cellset_data['cells'])}"
        description += f"  Number of walls: {len(self.cellset_data['walls'])}"
        description += f"  Number of points: {len(self.cellset_data['points'])}"
        description += "\n=== End of Configuration Summary ==="

        if verbose:
            print(description)
        else:
            return description






