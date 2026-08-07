#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#       mecha.utils.loader
#
#       File author(s):
#           Dilhan Ozturk, Adrien Heymans
#
#       File contributor(s):
#           Jonas Sonnenschein
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
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field

BARRIER_KEY = {
    0 : "No apo barrier",
    1 : "En_cs",
    2 : "En_sub + PC",
    3 : "En_sub",
    4 : "En_sub + Ex_cs",
    5 : "En_cs + Ex_cs",
    6 : "En_cs + En_sub",
    7 : "Ex_cs",
    8 : "En_sub + Ex_sub",
    9 : "Ex_Lcap"
}

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
            'osmotic_left_soil': -0.5E3,
            'osmotic_right_soil': -0.5E3,
            'osmotic_symmetry_soil': 1,
            'osmotic_shape_soil': 1.0,
            'osmotic_diffusivity_soil': 1,
            'osmotic_xyl': -1.80E3,
            'osmotic_endo': -4.80E3,
            'osmotic_symmetry_xyl': 2,
            'osmotic_shape_xyl': 1.0,
            'osmotic_diffusivity_xyl': 0.5,
            'pressure_xyl_prox': -5.0E3,
            'pressure_xyl_dist': np.nan,
            'flow_xyl_prox': np.nan,
            'flow_xyl_dist': np.nan,
            'delta_p_xyl_prox': np.nan,
            'pressure_sieve_prox': np.nan,
            'pressure_sieve_dist': np.nan,
            'flow_sieve_prox': np.nan,
            'flow_sieve_dist': np.nan,
            'delta_p_sieve': np.nan,
            'osmotic_sieve': -1.0E4,
            's_hetero': 0,
            's_factor': 1.0,
            'os_hetero': 0,
            'os_cortex': -4.8E3,
            'elongation_midpoint_rate': 2.8,
            'elongation_side_rate_difference': 0.0,
        }
        self.scenarios.append(scenario)

    def add_scenario(self, scenario):
        """Add a scenario to the list of scenarios."""
        self.scenarios.append(scenario)
        self.n_scenarios = len(self.scenarios)

    def set_os_hetero_scenarios(self, os_hetero_values: List[int], 
                                os_cortex_values: List[float]=None, 
                                osmotic_xyl_values: List[float]=None, 
                                osmotic_sieve_values: List[float]=None):
        """
        Quickly set new scenarios that change the osmotic potential (Os_hetero).
        Keeps the first scenario as a base and adds new scenarios for each value in os_hetero_values.
        
        Parameters
        ----------
        os_hetero_values : List[int]
            List of Os_hetero values to create scenarios for.
        """
        if not self.scenarios:
            self._set_default_scenario()
            
        base_scenario = self.scenarios[0].copy()
        self.scenarios = []
        self._set_default_scenario()
        for i, val in enumerate(os_hetero_values):
            new_scenario = base_scenario.copy()
            new_scenario['os_hetero'] = val
            if os_cortex_values is not None:
                new_scenario['os_cortex'] = os_cortex_values[i]
            else:
                new_scenario['os_cortex'] = base_scenario['os_cortex']
            if osmotic_xyl_values is not None:
                new_scenario['osmotic_xyl'] = osmotic_xyl_values[i]
            else:
                new_scenario['osmotic_xyl'] = base_scenario['osmotic_xyl']
            if osmotic_sieve_values is not None:
                new_scenario['osmotic_sieve'] = osmotic_sieve_values[i]
            else:
                new_scenario['osmotic_sieve'] = base_scenario['osmotic_sieve']
            self.add_scenario(new_scenario)


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
        Plasmodesmata section area in square microns (default is 7.47E-5 µm² or 45 nm radius).
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
    thickness: float = 1.5 # µm
    # 7.47E-5 µm²: historical value used in MECHA simulations (Couvreur et al.)
    # 1.79E-4 µm²: PD type I width = 22 nm, desmotubule diameter = 16 nm --> cytoplasmic sleeve thickness = 3 nm (Nicolas et al. 2017)
    # 8.16E-4 µm²: PD type II + spokes: 36 nm, desmotubule diameter = 16 nm --> sleeve thick=10 nm (Nicolas et al. 2017)
    # 1.92E-3 µm²: PD type II - spokes: 52 nm, desmotubule diameter = 16 nm --> sleeve thick=18 nm (Nicolas et al. 2017)   
    pd_section: float = 1.92E-3 # µm² 
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
                'height': float(mat.get("height")),
                'apo_barrier_type': BARRIER_KEY.get(int(mat.get("Barrier")), "NA")
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
            # from the key get the string for the apoplastic barrier
            apo_barrier_type = BARRIER_KEY.get(b) 
            self.maturity_stages.append({'barrier': b, 'height': height[i], 'nlayers': int(1), 'apo_barrier_type': apo_barrier_type})
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
    It includes hormone movement parameters, active transport carriers, symplastic and apoplastic transport,
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
    diff_mb1 : float, optional
        Diffusivity of hormone 1 across cell membranes (default is 0.0).
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
    diff_mb1: float = 0.0035
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
        diff_mb1_elem = root.xpath('Hormone_movement/Diffusivity_MB_H1')
        self.diff_mb1 = float(diff_mb1_elem[0].get("value")) if diff_mb1_elem else 0.0
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
        print(f'Contact range: {self.contact}')
    
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
    hydraulics_file : str, optional
        Path to the hydraulic configuration XML file (default is None).
    kw_elems : List[Any]
        List of raw cell wall hydraulic conductivity elements from the XML file.
    kw_septa_elems : List[Any]
        List of raw cell wall septa hydraulic conductivity elements from the XML file.
    kw_barrier_elems : List[Any]
        List of raw cell wall barrier hydraulic conductivity elements from the XML file.
    kaqp_elems : List[Any]
        List of raw aquaporin hydraulic conductivity elements from the XML file.
    kpl_elems : List[Any]
        List of raw plasmodesmata hydraulic conductivity elements from the XML file.
    xcontactrange : List[Any]
        List of raw xylem contact range elements from the XML file.
    path_hydraulics : List[Any]
        List of output paths for hydraulic scenarios.
    n_kw : int
        Number of cell wall hydraulic conductivity elements (default is 1).
    n_kw_septa : int
        Number of cell wall septa hydraulic conductivity elements (default is 1).
    n_kw_barrier : int
        Number of cell wall barrier hydraulic conductivity elements (default is 1).
    n_kaqp : int
        Number of aquaporin hydraulic conductivity elements (default is 1).
    n_kpl : int
        Number of plasmodesmata hydraulic conductivity elements (default is 1).
    n_xcontact : int
        Number of xylem contact range elements (default is 1).
    n_hydraulics : int
        Number of hydraulic scenarios (default is 1).
    kmb : float
        Membrane hydraulic conductivity (default is 3.0E-5).
    ratio_cortex : float
        Ratio related to cortex hydraulic properties (default is 1.0).
    fplxheight_map : Dict[Tuple[int, int], float]
        Dictionary mapping tissue interface ID pairs to plasmodesmata height values (number per unit height).
    interface_map : Dict[str, Any]
        Dictionary mapping XML tag names to lists of tissue interfaces.
    interface_kpl_factor_map : Dict[Tuple[int, int], Union[str, Tuple[str, str]]]
        Dictionary mapping tissue interfaces to specific plasmodesmata conductance configuration factor keys.
    axial_conductance_source : int
        Source type for axial conductance (1 for area-based Poiseuille law; 2 for prescribed values) (default is 1).
    k_sieve_elems : List[Any]
        List of raw sieve tube axial conductance elements from the XML file.
    k_xyl_elems : List[Any]
        List of raw xylem vessel axial conductance elements from the XML file.
    k_sieve : Union[float, List[float]]
        Sieve tube hydraulic conductance value(s) (default is 1.0E-6).
    K_axial : np.ndarray, optional
        Axial conductance matrix (default is None).
    k_xyl : Union[float, List[float]]
        Xylem vessel axial hydraulic conductance value(s) (default is 1.0E-6).
    K_xyl_spec : float
        Specific xylem vessel axial hydraulic conductance (default is 1.0E-6).
    conductivities : List[Dict[str, Any]]
        List of root conductivity results.
    kw : List[float]
        Processed list of cell wall hydraulic conductivity values.
    kw_barrier : List[float]
        Processed list of cell wall barrier hydraulic conductivity values.
    kw_septa : List[float]
        Processed list of cell wall septa hydraulic conductivity values.
    kaqp : List[Dict[str, float]]
        Processed list of aquaporin hydraulic conductivity parameter configurations.
    kpl : List[Dict[str, float]]
        Processed list of plasmodesmata hydraulic conductivity parameter configurations.

    Methods
    -------
    _load_hydraulics()
        Load hydraulic configuration parameters from the XML file.
    set_pd_interface(network)
        Define mapping of XML tags to tissue interface ID pairs using network info.
    get_kw_value(h)
        Get cell wall hydraulic conductivity for the scenario index h.
    get_kw_septa_value(h)
        Get cell wall septa hydraulic conductivity for the scenario index h.
    get_kw_barrier_values(h)
        Get casparian and suberin cell wall barrier conductivities for scenario h.
    get_wall_conductivities(barrier, h)
        Get dictionary of wall conductivities for a specific barrier type and scenario h.
    get_plasmodesmatal_conductance(h)
        Get plasmodesmata conductance configuration dict for scenario h.
    get_aquaporin_contributions(h)
        Get aquaporin contributions configuration dict for scenario h.
    """
    # File paths
    hydraulics_file: Optional[str] = None

    # Hydraulic parameter elements (raw from XML)
    kw_elems: List[Any] = field(default_factory=list)
    kw_barrier_elems: List[Any] = field(default_factory=list)
    kaqp_elems: List[Any] = field(default_factory=list)
    kpl_elems: List[Any] = field(default_factory=list)
    xcontactrange: List[Any] = field(default_factory=lambda: [-15E9])
    path_hydraulics: List[Any] = field(default_factory=list)

    # φ-thickening           
    kw_phi_thick: List[float] = field(default_factory=lambda: [1.0E-16])

    # Counts
    n_kw: int = 1
    n_kw_septa: int = 1
    n_kw_barrier: int = 1
    n_kaqp: int = 1
    n_kpl: int = 1
    n_xcontact: int = 1
    n_hydraulics: int = 1

    # Single-value parameters
    kmb: float = 3.0E-5
    ratio_cortex: float = 1.0
    
    # PD height (Fplxheight) parameters for different tissue interfaces
    fplxheight_map: Dict[Tuple[int, int], float] = field(default_factory=lambda: {
        # Symmetric interfaces 
        (0, 0): 8.0E5,
        (1, 1): 8.0E5,           # default (fallback) or hypodermis-hypodermis
        (2, 2): 8.0E5,           # default (fallback) or epi-epi
        (1, 2): 1.08E6,          # epi-exo/hypo
        (1, 4): 2.28E6,           # exo/hypo-cortex/mesophyll

        (4, 4): 8.6E5,           # cortex-cortex/mesophyll-mesophyll
        (3, 4): 8.8E5,           # cortex-endo/mesophyll-endo
        (3, 3): 6.4E5,           # endo-endo
        (3, 16): 9.6E5,          # endo-peri
        (3, 5): 9.6E5,           # endo-stele 

        (3, 17): 1.08E6,         # endo-transfusion parenchyma
        (3, 18): 0.0,            # endo-transfusion tracheid

        (5, 16): 1.08E6,         # stele-peri
        (5, 11): 9.0E5,          # stele-phloem
        (5, 23): 9.0E5,          # stele-protophloem
        (5, 12): 9.8E5,          # stele-comp
        (5, 13): 6.4E5,          # stele-xylem
        (5, 5): 6.4E5,           # stele-stele

        (5, 17): 6.4E5,         # stele-transfusion parenchyma
        (5, 18): 0.0,            # stele-transfusion tracheid
        
        (11, 12): 1.76E6,        # sieve-comp
        (11, 16): 7.2E5,         # sieve-peri
        (11, 13): 0.0,           # sieve-xylem
        (11, 11): 0.0,           # sieve-sieve
        (11, 23): 0.0,           # sieve-protophloem
        (11, 17): 6.4E5,         # sieve-transfusion parenchyma
        (11, 18): 0.0,           # sieve-transfusion tracheid

        (12, 13): 9.8E5,         # comp-xylem
        (12, 16): 7.0E5,         # comp-peri
        (12, 12): 6.8E5,         # comp-comp
        (12, 23): 6.8E5,         # comp-protophloem

        (12, 17): 1.08E6,        # Strasburger cell-transfusion parenchyma
        (12, 18): 0.0,           # Strasburger cell -transfusion tracheid

        (13, 16): 1.08E6,        # xylem-peri   
        (13, 13): 6.4E5,         # xylem-xylem
        (13, 23): 0.0,           # xylem-protophloem

        (13, 17): 1.08E6,        # xylem-transfusion parenchyma
        (13, 18): 1.76E6,        # xylem-transfusion tracheid

        (17, 17): 8.0e5,         # transfusion parenchyma-transfusion parenchyma
        (17, 18): 0.0,           # transfusion parenchyma-transfusion tracheid
        (18, 18): 0.0,           # transfusion tracheid - transfusion tracheid

        (23, 23): 0.0,
        # Add other mappings as needed
    })

    interface_map: Dict[str, Any] = field(default_factory=dict)

    # Maps a sorted (cgroup_i, cgroup_j) interface tuple to a kpl_config key (str)
    # or a pair of keys (Tuple[str, str]) whose harmonic mean is used.
    # Used by HydraulicMatrixBuilder._fill_plasmodesmata() to look up the
    # per-interface conductance factor without a long if/elif chain.
    # Fixes the bug where endo_in_factor / endo_out_factor were defined in
    # the XML but never consulted by the solver.
    interface_kpl_factor_map: Dict[Tuple[int, int], Union[str, Tuple[str, str]]] = field(
        default_factory=lambda: {
            # Same-tissue interfaces (single factor)
            (4, 4):   'cortex_factor',                # cortex–cortex
            (12, 12): 'phloem_companion_cell_factor',  # companion–companion
            (11, 11): 'phloem_sieve_tube_factor',      # sieve–sieve

            # Cross-tissue: inner endodermis (peri/stele side)
            (3, 5):   'endo_in_factor',                # endo–stele
            (3, 16):  'endo_in_factor', # endo–peri 

            # Cross-tissue: outer endodermis (cortex side)
            (3, 4):   'endo_out_factor',  # endo–cortex (harmonic mean)

            # Stele / pericycle / phloem interfaces (single factor)
            (5, 12):  'pericycle_phloem_pole_factor',  # stele–companion
            (12, 13): 'pericycle_phloem_pole_factor',  # companion–xylem
            (11, 12): 'phloem_companion_cell_factor',  # sieve–companion
            (11, 16): 'phloem_sieve_tube_factor',  # sieve–peri
            (11, 13): 'phloem_sieve_tube_factor',  # sieve–xylem
            (5, 11):  'phloem_sieve_tube_factor',  # stele–sieve
            (11, 11): 'phloem_phloem_tube_factor', # Inter phloem factor

            # Companion–pericycle: harmonic mean of the two pole factors
            (12, 16): 'pericycle_phloem_pole_factor',  # companion–peri
        }
    )

    # Maps a sorted (cgroup_i, cgroup_j) tissue-interface pair to the kw_config
    # key that _fill_wall() should use for that apoplastic wall edge.
    # Used by HydraulicMatrixBuilder._fill_wall() to replace the previous
    # if/elif chain over count_* node attributes.
    # The lookup falls back to plain `kw` for any interface not listed here.
    interface_kw_key_map: Dict[Tuple[int, int], str] = field(
        default_factory=lambda: {
            # ── Endodermis radial walls ─────────────────────────────────────
            (3, 3):  'kw_endo_endo',      # endo–endo (Casparian strip)
            (3, 4):  'kw_endo_cortex',    # endo–cortex (outer face, suberin)
            (3, 5):  'kw_endo_peri',      # endo–stele  (inner face, suberin)
            (3, 11): 'kw_endo_peri',      # endo–phloem sieve (inner face)
            (3, 12): 'kw_endo_peri',      # endo–companion   (inner face)
            (3, 13): 'kw_endo_peri',      # endo–xylem       (inner face)
            (3, 16): 'kw_endo_peri',      # endo–pericycle   (inner face)

            # ── Exodermis radial walls ──────────────────────────────────────
            (1, 1):  'kw_exo_exo',        # exo–exo  (Casparian strip)
            (1, 2):  'kw_exo_epi',        # exo–epidermis
            (1, 4):  'kw_exo_cortex',     # exo–cortex

            # ── Cortex tangential walls ─────────────────────────────────────
            # TODO: Implement the MSC (outer cortex, how many layers?)
            (4, 4):  'kw_cortex_cortex',  # cortex–cortex

            # ── Passage-cell and septa walls are handled as special cases ───
            # before the map lookup in _fill_wall() and are not listed here.
        }
    )

    # Conductance parameters
    axial_conductance_source: int = 1
    k_sieve_elems: List[Any] = field(default_factory=list)
    k_xyl_elems: List[Any] = field(default_factory=list)
    k_sieve: float = 1.0  # Penalty conductance for sieve-tube Dirichlet BC [cm³ hPa⁻¹ d⁻¹].
    K_axial: Optional[np.ndarray] = None
    # Penalty conductance for the xylem Dirichlet BC [cm³ hPa⁻¹ d⁻¹].
    # This is a *numerical* penalty not a physical conductance.
    # It must dominate the largest physical diagonal entry.
    k_xyl: float = 1.0
    K_xyl_spec: float = 1.0E-6   # Xylem vessel specific axial hydraulic conductance

    # Root conductivities
    conductivities: List[Dict[str, Any]] = field(default_factory=list)

    # Processed parameter arrays
    kw: List[float] = field(default_factory=lambda: [0.00024])
    kw_barrier: List[float] = field(default_factory=lambda: [1.00E-16])
    kw_septa: List[float] = field(default_factory=lambda: [0.00012])
    kaqp: List[Dict[str, float]] = field(default_factory=lambda: [{'value': 0.000430, 'cortex_factor': 1.0, 'endo_factor': 1.0, 'epi_factor': 1.0, 'exo_factor': 1.0, 'stele_factor': 1.0}])
    kpl: List[Dict[str, float]] = field(default_factory=lambda: [{'value': 5.3E-12, 'phloem_companion_cell_factor': 0.0, 'pericycle_phloem_pole_factor': 0.0, 'phloem_sieve_tube_factor': 0.0, 'cortex_factor': 1.0}])

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
        self.kw_septa_elems = root.xpath('kw_septa_range/kw_septa')
        self.kw_barrier_elems = root.xpath('kw_barrier_range/kw_barrier')
        self.kaqp_elems = root.xpath('kAQPrange/kAQP')
        self.kpl_elems = root.xpath('Kplrange/Kpl')

        self.n_kw = len(self.kw_elems)
        self.n_kw_septa = len(self.kw_septa_elems)
        self.n_kw_barrier = len(self.kw_barrier_elems)
        self.n_kaqp = len(self.kaqp_elems)
        self.n_kpl = len(self.kpl_elems)

        # φ-thickening configuration
        phi_elem = root.xpath('phi_thick')
        self.kw_phi_thick = [float(kw.get("value")) for kw in phi_elem] if phi_elem else [1E-16]

        # Extract single-value parameters
        self.kmb = float(root.xpath('km')[0].get("value"))
        self.ratio_cortex = float(root.xpath('ratio_cortex')[0].get("value"))
        
        # PD height parameters
        # Populate the map from XML
        for xml_tag, interfaces in self.interface_map.items():
            elem = root.xpath(xml_tag) # might not work
            if elem:
                value = float(elem[0].get("value"))
                for interface in interfaces:
                    self.fplxheight_map[interface] = value
           
        # Conductance parameters
        # 1: Poiseuille law (based on cross-section area); 2: Prescribed here below (for all sieve tubes, and vessel per vessel)
        self.axial_conductance_source = int(root.xpath('Kax_source')[0].get("value")) if root.xpath('Kax_source') else 1
        self.k_sieve_elems = root.xpath('K_sieve_range/K_sieve')
        self.k_xyl_elems = root.xpath('K_xyl_range/K_xyl')
        self.k_sieve = [float(k_sieve.get("value")) for k_sieve in self.k_sieve_elems] if self.k_sieve_elems else [1.0E-6]
        self.k_xyl = [float(k_xyl.get("value")) for k_xyl in self.k_xyl_elems] if self.k_xyl_elems else [1.0E-6]
        
        # Contact range
        self.xcontactrange = [float(xcontact.get("value")) for xcontact in root.xpath('Xcontactrange/Xcontact')]
        self.n_xcontact = len(self.xcontactrange)
        
        # Output paths
        self.path_hydraulics = root.xpath('path_hydraulics/Output')
        self.n_hydraulics = len(self.path_hydraulics) if self.path_hydraulics else 1
        
        # Process parameter arrays
        self.kw = [float(kw.get("value")) for kw in self.kw_elems] if self.kw_elems else [0.00024]
        self.kw_septa = [float(kw_septa.get("value")) for kw_septa in self.kw_septa_elems] if self.kw_septa_elems else [0.00012]
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
        for kpl_elem in self.kpl_elems if self.kpl_elems else [{'value': 5.3E-12, 'PCC_factor': 0.0, 'PPP_factor': 0.0, 'PST_factor':0.0, 'cortex_factor': 1.0}]:
            kpl_dict = {'value': float(kpl_elem.get("value"))}
            kpl_dict['phloem_companion_cell_factor'] = float(kpl_elem.get("PCC_factor")) # 
            kpl_dict['pericycle_phloem_pole_factor'] = float(kpl_elem.get("PPP_factor")) # 
            kpl_dict['phloem_sieve_tube_factor'] = float(kpl_elem.get("PST_factor", 0.0)) # 
            kpl_dict['cortex_factor'] = float(kpl_elem.get("cortex_factor", 1.0))
            kpl_dict['endo_in_factor'] = float(kpl_elem.get("endo_in_factor", 1.0))
            kpl_dict['endo_out_factor'] = float(kpl_elem.get("endo_out_factor", 1.0))

            self.kpl.append(kpl_dict)

    def _set_default_values(self):
        """Set default values if no file is provided."""
        self.kw_barrier_elems = [{'value': 1.00E-16, 'Casp': 1.00E-16, 'Sub': 1.00E-16, 'Sub_in': 1.00E-16, 'Sub_out': 1.00E-16, 'Lig': 1.00E-16}]
        self.kw_elems = [{'value': 0.00024}]
        self.kw_septa_elems = [{'value': 0.00012}]
    
    def set_pd_interface(self, network):
        self.interface_map = {
            'Fplxheight': [(1, 1)],
            'Fplxheight_epi_exo': [(1, 2)],
            'Fplxheight_outer_cortex': [
                (network.outercortex_connec_rank, 4),
                (4, network.outercortex_connec_rank)
            ],
            'Fplxheight_cortex_cortex': [(4, 4)],
            'Fplxheight_cortex_endo': [(3, 4)],
            'Fplxheight_endo_endo': [(3, 3)],
            'Fplxheight_endo_peri': [(3, 16)],
            'Fplxheight_peri_peri': [(16, 16)],
            'Fplxheight_peri_stele': [(5, 16), (13, 16)],
            'Fplxheight_stele_stele': [(5, 5), (5, 13), (13, 13)],
            'Fplxheight_stele_comp': [(5, 12), (12, 13)],
            'Fplxheight_peri_comp': [(12, 16)],
            'Fplxheight_comp_comp': [(12, 12)],
            'Fplxheight_comp_sieve': [(11, 12)],
            'Fplxheight_peri_sieve': [(11, 16)],
            'Fplxheight_stele_sieve': [(5, 11), (11, 13)],
        }

    def get_kw_value(self, h: int) -> float:
        """Get the kw value based on the scenario index."""
        if self.n_kw == self.n_hydraulics:
            return self.kw[h]
        elif self.n_kw == 1:
            return self.kw[0]
        else:
            return self.kw[int(h/(self.n_kaqp*self.n_kpl))%self.n_kw]

    def get_kw_septa_value(self, h: int) -> float:
        """Get the kw_septa value based on the scenario index."""
        if self.n_kw_septa == self.n_hydraulics:
            return self.kw_septa[h]
        elif self.n_kw_septa == 1:
            return self.kw_septa[0]
        else:
            return self.kw_septa[int(h/(self.n_kaqp*self.n_kpl))%self.n_kw_septa]

    def get_kw_barrier_values(self, h: int) -> Tuple[float, List[float]]:
        """Get the kw_barrier values based on the scenario index."""
        if h >= len(self.kw_barrier_elems):
            h = 0
        sub_value = self.kw_barrier_elems[h].get("Sub")
        if self.kw_barrier_elems[h].get("Sub_in") is None and self.kw_barrier_elems[h].get("Sub") is not None:
            sub_in_value = sub_value
            sub_out_value = sub_value
        else:
            sub_in_value = float(self.kw_barrier_elems[h].get("Sub_in"))
            sub_out_value = float(self.kw_barrier_elems[h].get("Sub_out"))
        if self.n_kw_barrier == self.n_hydraulics:
            kw_barrier_casparian = float(self.kw_barrier_elems[h].get("Casp"))
            kw_barrier_suberin = float(self.kw_barrier_elems[h].get("Sub"))
            kw_barrier_suberin_in = sub_in_value
            kw_barrier_suberin_out = sub_out_value
            kw_barrier_lignin = float(self.kw_barrier_elems[h].get("Lig"))
        elif self.n_kw_barrier == 1:
            kw_barrier_casparian = float(self.kw_barrier_elems[0].get("Casp"))
            kw_barrier_suberin = float(self.kw_barrier_elems[0].get("Sub"))
            kw_barrier_suberin_in = sub_in_value
            kw_barrier_suberin_out = sub_out_value
            kw_barrier_lignin = float(self.kw_barrier_elems[0].get("Lig"))
        else:
            index = int(h/(self.n_kaqp*self.n_kpl*self.n_kw))%self.n_kw_barrier
            kw_barrier_casparian = float(self.kw_barrier_elems[index].get("Casp"))
            kw_barrier_suberin = float(self.kw_barrier_elems[index].get("Sub"))
            kw_barrier_suberin_in = sub_in_value
            kw_barrier_suberin_out = sub_out_value
            kw_barrier_lignin = float(self.kw_barrier_elems[index].get("Lig"))

        # Use the general 'suberin' value if specific ones are missing
        if kw_barrier_suberin_in is None:
            kw_barrier_suberin_in = float(kw_barrier_suberin) if kw_barrier_suberin is not None else 1E-16
        if kw_barrier_suberin_out is None:
            kw_barrier_suberin_out = float(kw_barrier_suberin) if kw_barrier_suberin is not None else 1E-16
        if kw_barrier_lignin is None:
            kw_barrier_lignin = float(kw_barrier_lignin) if kw_barrier_lignin is not None else 1E-16

        kw_barrier_suberin_all = [float(kw_barrier_suberin_in), float(kw_barrier_suberin_out)]

        return kw_barrier_casparian, kw_barrier_suberin_all, kw_barrier_lignin

    def get_wall_conductivities(self, barrier: int, h: int) -> Dict[str, float]:
        """Get wall conductivities based on barrier type."""
        kw =  self.get_kw_value(h)
        kw_septa = self.get_kw_septa_value(h)
        kw_barrier_casparian, kw_barrier_suberin, kw_barrier_lignin = self.get_kw_barrier_values(h)
        barrier_configs = {
            0: {  # No Casparian strip
                'kw_endo_endo': kw,
                'kw_puncture': kw,
                'kw_exo_exo': kw,
                'kw_exo_epi': kw,
                'kw_exo_cortex': kw,
                'kw_septa': kw_septa,
                'kw_cortex_cortex': kw,
                'kw_endo_peri': kw,
                'kw_endo_cortex': kw,
                'kw_passage': kw,
                'kw_phi_thick': kw_barrier_lignin
            },
            1: {  # Endodermis radial walls
                'kw_endo_endo': kw_barrier_casparian,
                'kw_exo_exo': kw,
                'kw_exo_epi': kw,
                'kw_exo_cortex': kw,
                'kw_septa': kw_septa,
                'kw_cortex_cortex': kw,
                'kw_endo_peri': kw,
                'kw_endo_cortex': kw,
                'kw_passage': kw,
                'kw_phi_thick': kw_barrier_lignin
            },
            2: {  # Endodermis with passage cells
                'kw_endo_endo': kw_barrier_casparian,
                'kw_exo_exo': kw,
                'kw_exo_epi': kw,
                'kw_exo_cortex': kw,
                'kw_septa': kw_septa,
                'kw_cortex_cortex': kw,
                'kw_endo_peri': kw_barrier_suberin[0],
                'kw_endo_cortex': kw_barrier_suberin[1],
                'kw_passage': kw,
                'kw_phi_thick': kw_barrier_lignin
            },
            3: {  # Endodermis full
                'kw_endo_endo': kw_barrier_casparian,
                'kw_exo_exo': kw,
                'kw_exo_epi': kw,
                'kw_exo_cortex': kw,
                'kw_septa': kw_septa,
                'kw_cortex_cortex': kw,
                'kw_endo_peri': kw_barrier_suberin[0],
                'kw_endo_cortex': kw_barrier_suberin[1],
                'kw_passage': kw_barrier_suberin[0],
                'kw_phi_thick': kw_barrier_lignin
            },
            4: {  # Endodermis full and exodermis radial walls
                'kw_endo_endo': kw_barrier_casparian,
                'kw_exo_exo': kw_barrier_casparian,
                'kw_exo_epi': kw,
                'kw_exo_cortex': kw,
                'kw_septa': kw_septa,
                'kw_cortex_cortex': kw,
                'kw_endo_peri': kw_barrier_suberin[0],
                'kw_endo_cortex': kw_barrier_suberin[1],
                'kw_passage': kw_barrier_suberin[0],
                'kw_phi_thick': kw_barrier_lignin
            },
            5: {  # Endodermal & exodermal Casparian strips
                'kw_endo_endo': kw_barrier_casparian,
                'kw_exo_exo': kw_barrier_casparian,
                'kw_exo_epi': kw,
                'kw_exo_cortex': kw,
                'kw_septa': kw_septa,
                'kw_cortex_cortex': kw,
                'kw_endo_peri': kw,
                'kw_endo_cortex': kw,
                'kw_passage': kw,
                'kw_phi_thick': kw_barrier_lignin
            },
            6: {  # Exodermis full and endodermis radial walls
                'kw_endo_endo': kw_barrier_casparian,
                'kw_exo_exo': kw_barrier_casparian,
                'kw_exo_epi': kw_barrier_suberin[1],
                'kw_exo_cortex': kw_barrier_suberin[0],
                'kw_septa': kw_septa,
                'kw_cortex_cortex': kw,
                'kw_endo_peri': kw,
                'kw_endo_cortex': kw,
                'kw_passage': kw,
                'kw_phi_thick': kw_barrier_lignin
            },
            7: {  # Exodermis radial walls
                'kw_endo_endo': kw,
                'kw_exo_exo': kw_barrier_casparian,
                'kw_exo_epi': kw,
                'kw_exo_cortex': kw,
                'kw_septa': kw_septa,
                'kw_cortex_cortex': kw,
                'kw_endo_peri': kw,
                'kw_endo_cortex': kw,
                'kw_passage': kw,
                'kw_phi_thick': kw_barrier_lignin
            },
            8: {  # Exodermis full suberized and endodermis full suberized
                'kw_endo_endo': kw_barrier_casparian,
                'kw_exo_exo': kw_barrier_casparian,
                'kw_exo_epi': kw_barrier_suberin[1],
                'kw_exo_cortex': kw_barrier_suberin[0],
                'kw_septa': kw_septa,
                'kw_cortex_cortex': kw,
                'kw_endo_peri': kw_barrier_suberin[0],
                'kw_endo_cortex': kw_barrier_suberin[1],
                'kw_passage': kw,
                'kw_phi_thick': kw_barrier_lignin
            },
            9: {  # Lignin Cap
                'kw_endo_endo': kw_barrier_casparian,
                'kw_exo_exo': kw_barrier_casparian,
                'kw_exo_epi': kw_barrier_lignin,
                'kw_exo_cortex': kw,
                'kw_septa': kw_septa,
                'kw_cortex_cortex': kw,
                'kw_endo_peri': kw,
                'kw_endo_cortex': kw,
                'kw_passage': kw,
                'kw_phi_thick': kw_barrier_lignin
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
            phloem_companion_cell_factor = 0.0

        if self.kpl[iPD].get('phloem_pericycle_pole_factor') is not None:
            phloem_pericycle_pole_factor = float(self.kpl[iPD].get('phloem_pericycle_pole_factor'))
        else:
            phloem_pericycle_pole_factor = 0.0

        if self.kpl[iPD].get('phloem_sieve_tube_factor') is not None:
            phloem_sieve_tube_factor = float(self.kpl[iPD].get('phloem_sieve_tube_factor'))
        else:
            phloem_sieve_tube_factor = 0.0

        config = {
            'kpl': kpl,
            'stele': stele_factor,
            'endo_in': endo_in_factor,
            'endo_out': endo_out_factor,
            'exo': exo_factor,
            'epi': epi_factor,
            'cortex': cortex_factor,
            'phloem_companion_cell_factor': phloem_companion_cell_factor, # PCC
            'phloem_pericycle_pole_factor': phloem_pericycle_pole_factor, # PPP
            'phloem_sieve_tube_factor': phloem_sieve_tube_factor # PST
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

    all_points = []
    points_groups = [points for points in root.xpath('walls/wall/points')]
    for points in points_groups:
        all_points.extend(points)
    # center of the tissue in x direction
    x_center = np.mean([float(point.get('x')) for point in all_points])
    y_center = np.mean([float(point.get('y')) for point in all_points])

    def _recenter_walls(points_groups, x_center, y_center):
        for points in points_groups:
            for pt in points:
                pt.set('x', str(float(pt.get('x')) - x_center))
                pt.set('y', str(float(pt.get('y')) - y_center))
    _recenter_walls(points_groups, x_center, y_center)

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

    @classmethod
    def needle_defaults(cls) -> "InData":
        """Build an :class:`InData` pre-configured for a conifer needle.

        The needle *geometry / topology* is produced upstream by GRANAP
        (``NeedleAnatomy`` → ``NetworkBuilder.populate_from_network()``); this
        factory only sets the *physics parameters* that MECHA needs and that are
        otherwise keyed to maize-root calibrations.

        Design rules followed here
        --------------------------
        * Start from the built-in root defaults (``cls()`` fires every
          ``_set_default_values()``), then override only what differs for a
          needle.
        * A value is changed away from the root default **only when it can be
          cited**.  Every parameter that is *inherited* from the root
          calibration and still awaits needle-specific evidence is flagged with
          a ``# ROOT DEFAULT (needs needle evidence)`` comment so it can be
          revisited.

        Returns
        -------
        InData
            Configuration ready to pass to :class:`Mecha` together with a
            GRANAP-populated ``NetworkBuilder``.

        """
        data = cls()

        # ── Geometry / maturity ────────────────────────────────────────────
        # Needles possess a Casparian-strip-equivalent structure on the radial
        # walls of the endodermis surrounding the vascular (transfusion) tissue.
        # barrier = 1 activates the endodermal Casparian strip (En_cs): apoplastic
        # flow across endo–endo radial walls is blocked (kw_endo_endo →
        # kw_barrier_casparian), while the tangential endo–cortex / endo–peri
        # walls stay permeable to transmembrane flow.
        # Endodermal Casparian band in conifer needles:
        # Canny (1993), Liesche et al. (2011)
        data.geometry.set_maturity_stages([1], [200.0])

        # ROOT DEFAULT (needs needle evidence): double cell-wall thickness (µm).
        # Root value from Andème-Onzighi et al. (2002); conifer needle epidermal
        # and hypodermal walls are reported thicker, but no calibrated MECHA
        # value is available yet.
        # data.geometry.thickness = 1.5

        # ROOT DEFAULT (needs needle evidence): plasmodesmatal open cross-section
        # (µm²) No reported needle value is available yet
        # data.geometry.pd_section = 7.47E-5

        # ── Hydraulics ─────────────────────────────────────────────────────
        # Disable the outer-boundary soil contact: a needle cross-section has no
        # soil interface, so no border wall should carry the soil Dirichlet BC.
        # x_contact = 1e10 µm (effectively ∞) → no border wall satisfies
        # x >= x_contact.  (needle_steady.py: X_CONTACT = 1e10)
        data.hydraulic.xcontactrange = [1.0E10]
        data.hydraulic.n_xcontact = 1

        # ROOT DEFAULT (needs needle evidence): bulk cell-wall conductivity
        # (cm hPa⁻¹ d⁻¹).  Maize-root calibration.
        # data.hydraulic.kw = [2.4E-4]

        # Casparian-strip / suberised
        # barrier wall conductivity (cm hPa⁻¹ d⁻¹).  Retained so barrier=1 blocks
        # the endo–endo apoplastic path as intended.
        data.hydraulic.kw_barrier = [1.0E-16]     # High default for numerical stability, 
                                                  # exact unknown

        # ROOT DEFAULT (needs needle evidence): background (non-aquaporin)
        # membrane conductivity (cm hPa⁻¹ d⁻¹).  Maize-root cortex calibration.
        # no needle-specific evidence yet, so retain the root default for now.
        # data.hydraulic.kmb = 3.0E-5

        # ROOT DEFAULT (needs needle evidence): aquaporin membrane conductivity
        # and per-tissue factors. Uniform factors inherited from the root; the
        # needle transfusion parenchyma / mesophyll aquaporin activity is not yet
        # differentiated.
        # data.hydraulic.kaqp[0]['value'] = 4.3E-4

        # ROOT DEFAULT (needs needle evidence): plasmodesmatal conductance
        # (cm³ hPa⁻¹ d⁻¹).  Maize-root value.
        # data.hydraulic.kpl[0]['value'] = 5.3E-12

        # ROOT DEFAULT (needs needle evidence): axial conductances of the
        # conducting elements (cm³ hPa⁻¹ d⁻¹). Placeholder values, not needle
        # transfusion-tracheid specific.
        # data.hydraulic.k_xyl = 1.0E-6
        # data.hydraulic.k_sieve = 1.0E-6

        # ── Boundary conditions (hydraulic only) ───────────────────────────
        sc = data.boundary.scenarios[0]

        # Clear all scenario values to NaN 
        for key in sc.keys(): sc[key] = np.nan

        sc['psi_soil_left'] = 0.0

        return data


# Canny MJ. 1993. Transfusion tissue of pine needles as a site of retrieval
# of solutes from the transpiration stream. New Phytologist 123, 227–232

# Liesche J, Martens HJ, Schulz A. 2011. Symplasmic transport and
# phloem loading in gymnosperm leaves. Protoplasma 248, 181–190