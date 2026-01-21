#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#       mecha.utils.loader
#
#       File author(s):
#           Dilhan Ozturk
#
#       File contributor(s):
#           Adrien Heymans
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

"""
Network builder for MECHA
Constructs the hydraulic network graph from cell data
"""

import networkx as nx
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from lxml import etree
import geopandas as gpd

from src.utils.data_loader import GeneralData, GeometryData

class NetworkBuilder:
    """Builds the hydraulic network graph from XML cell data"""
    def __init__(self):

        self.cellset_data: Dict[str, Any] = {}

        self.graph: nx.Graph = nx.Graph()

        # Network dimensions
        self.n_walls: int = 0
        self.n_junctions: int = 0
        self.n_wall_junction: int = 0
        self.n_cells: int = 0
        self.n_total: int = 0
        self.n_membrane: int = 0
        self.n_membrane_from_epi: int = 0
        self.n_nodes: int = 0
        
        # Cell and wall properties
        self.cell_areas: Optional[np.ndarray] = None 
        self.cell_perimeters: Optional[np.ndarray] = None 
        self.cell_ranks: Optional[np.ndarray] = None 
        self.cell_groups: Optional[np.ndarray] = None 
        
        # Spatial data
        self.wall_lengths: Optional[np.ndarray] = None 
        self.distance_wall_cell: Optional[np.ndarray] = None 
        self.junction_lengths: Dict[Any, Any] = {}
        self.junction_positions: Dict[Any, Any] = {}
        
        # Border identification
        self.border_walls: List[int] = []
        self.border_aerenchyma: List[int] = []
        self.border_junction: List[int] = []
        self.border_link: Optional[np.ndarray] = None 
        
        # Special cells
        self.xylem_cells: List[int] = []
        self.sieve_cells: List[int] = []
        self.xylem_walls: List[int] = []
        self.proto_sieve_cells: List[int] = []
        self.intercellular_cells: List[int] = []
        self.passage_cells: List[int] = []
        self.xylem_80_percentile_distance: float = 0.0
        self.total_xylem_area: float = 0.0
        self.xylem_area: List[float] = []
        self.xylem_area_ratio: List[float] = []
        self.n_sieve: int = 0
        self.n_protosieve: int = 0
        self.total_phloem_area: float = 0.0
        self.phloem_area: List[float] = []
        self.phloem_area_ratio: List[float] = []
        
        # Connectivity
        self.n_cell_connections: Optional[np.ndarray] = None
        self.cell_connections: Optional[np.ndarray] = None 
        self.wall_to_cell: Optional[np.ndarray] = None 
        self.junction_to_wall: Dict[Any, Any] = {}
        self.n_junction_to_wall: Dict[Any, Any] = {} 

        # Cellset artifacts
        self.list_ghostwalls: List[int] = [] #"Fake walls" not to be displayed
        self.list_ghostjunctions: List[int] = [] #"Fake junctions" not to be displayed
        self.n_ghost_junction2wall: int = 0
        
        # Gravity center and geometry
        self.x_grav: float = 0.0
        self.y_grav: float = 0.0
        self.x_min: float = np.inf
        self.x_max: float = 0.0
        
        # Layer discretization
        self.layer_dist: Optional[np.ndarray] = None 
        self.n_layer: Optional[np.ndarray] = None 
        self.rank_to_row: Optional[np.ndarray] = None 
        self.r_discret: Optional[np.ndarray] = None 
        self.distance_from_center: Optional[np.ndarray] = None
        self.row_outer_cortex: Optional[np.ndarray] = None

        # Rank 
        self.stele_connec_rank: int = 0
        self.outercortex_connec_rank: int = 0
        
        # Lists for special cells
        self.xylem_distance: List[int] = []
        self.protosieve_list: List[int] = []

        # Distance computation
        self.distance_max_cortex: float = 0.0
        self.distance_min_cortex = np.inf
        self.distance_avg_epi: float = 0.0
        self.distance_center_grav: float = 0.0
        self.perimeter: float = 0.0

        # Cell surface computation
        self.len_outer_cortex: float = 0.0
        self.len_cortex_cortex: float = 0.0
        self.len_cortex_endo: float = 0.0
        self.cross_section_outer_cortex: float = 0.0
        self.cross_section_cortex_cortex: float = 0.0
        self.cross_section_cortex_endo: float = 0.0
        self.plasmodesmata_indice: List[int] = []

        # list of contagion parameters
        self.apo_wall_zombies0: List[int] = []
        self.apo_wall_cc: List[int] = []
        self.apo_wall_target: List[int] = []
        self.apo_wall_immune: List[int] = []

        self.apo_j_zombies0: List[int] = []
        self.apo_j_cc: List[int] = []


    def build_network(self, general: GeneralData, geometry: GeometryData, cellset_data, verbose: bool = False):
        """Main method to build network from XML data"""
        if cellset_data is None:
            raise ValueError("Cellset data is None")
        if general is None:
            raise ValueError("General data is None")
        if geometry is None:
            raise ValueError("Geometry data is None")
        
        self.cellset = cellset_data
        self.n_walls = len(self.cellset['points'])
        self.n_cells = len(self.cellset['cells'])

        if verbose:
            print('  Creating wall, junction and cell nodes...')
        self.create_wall_junction_nodes(geometry.im_scale)
        self.identify_border_walls_junctions()
        self.create_cell_nodes(geometry, general.apo_contagion)
        
        if verbose:
            print('  Creating membrane connections...')
        self.build_membrane_connections()
        self.compute_cell_properties()
        self.build_plasmodesmata_connections()
        self.build_wall_connections()
        self.compute_gravity_center()

        if verbose:
            print('  Ranking cells by tissue type and distance from root center...')
        self.rank_cells(geometry)
        self.compute_cell_surface(geometry)
        self.create_layer_discretization()
        self.compute_distance_from_center()
        self.n_nodes = self.graph.number_of_nodes()
        self._calculate_xylem_area()
        self._calculate_phloem_area()
    
    def create_wall_junction_nodes(self, im_scale: float, n_dec_position: int = 6):
        points = self.cellset['points']

        junction_ni = 0
        self.junction_to_wall = {}
        self.n_junction_to_wall = {}
        junction_list = {}
        for point_groups in points: #Loop on wall elements

            wall_id = int((point_groups.getparent().get)("id")) # wall_id records the current wall id number

            coords = []
            for point in point_groups:
                x = round(im_scale * float(point.get("x")), n_dec_position)
                y = round(im_scale * float(point.get("y")), n_dec_position)
                coords.append((x, y))
            # Store junction positions for this wall
            self.junction_positions[wall_id] = [
                coords[0][0], coords[0][1],  # First junction
                coords[-1][0], coords[-1][1]  # Last junction
            ]

            if len(coords) < 2: # Skip if there are not enough points to define a wall
                continue

            # Calculate wall length
            length = sum(
                np.hypot(coords[i+1][0]-coords[i][0], coords[i+1][1]-coords[i][1])
                for i in range(len(coords)-1)
            )

            # Find midpoint
            mid_x, mid_y = self._find_wall_midpoint(coords, length)

            # Track min/max for later interpolation
            self.x_min = min(self.x_min, mid_x)
            self.x_max = max(self.x_max, mid_x)

            # Add wall node
            self.graph.add_node(
                wall_id,
                indice=wall_id,
                type="apo",
                position=(round(mid_x, n_dec_position), round(mid_y, n_dec_position)),
                length=length
            )

            # Add junction node
            for coord in [coords[0], coords[-1]]: # First and last point as junctions
                pos_key = f"x{coord[0]}y{coord[1]}"

                if pos_key not in junction_list:
                    node_id = self.n_walls + junction_ni
                    self.graph.add_node(
                        node_id,
                        indice=node_id,
                        type="apo",
                        position=coord,
                        length=0
                    )
                    junction_list[pos_key] = node_id
                    self.junction_to_wall[node_id] = [wall_id]
                    self.n_junction_to_wall[node_id] = 1
                    junction_ni += 1 # New junction created 
                else:
                    junction_id = junction_list[pos_key]
                    self.junction_to_wall[junction_id].append(wall_id) # Several cell wall ID numbers can correspond to the same X Y coordinate where they meet
                    self.n_junction_to_wall[junction_id] += 1 # Count how many walls connect to this junction
        
        self.n_junctions = junction_ni
        self.n_wall_junction = self.n_walls + self.n_junctions

    def identify_border_walls_junctions(self):
        """Identify walls and junctions at the soil-root interface"""
        walls_loop = self.cellset['walls']
        cell_to_wall = self.cellset['cell_to_wall']
        self.wall_lengths = nx.get_node_attributes(self.graph,'length') #Nodes lengths (micrometers)

        # Initialize border tracking
        self.border_link = 2*np.ones((self.n_wall_junction, 1), dtype=int)

        # Count how many cells each wall is connected to
        for wall_elem in walls_loop:
            wall_id = int(wall_elem.get("id"))
            self.border_link[wall_id] -= 1
        
        for cell_group in cell_to_wall:
            cgroup = int(cell_group.getparent().get("group"))
            for wall_ref in cell_group:
                wall_id = int(wall_ref.get("id"))
                if wall_id < self.n_walls:
                    # Wall at soil interface (epidermis and single connection)
                    if self.border_link[wall_id] == 1 and cgroup == 2:
                        if wall_id not in self.border_walls:
                            self.border_walls.append(wall_id)
                    # Wall at aerenchyma surface
                    elif self.border_link[wall_id] == 1 and cgroup != 2:
                        if wall_id not in self.border_aerenchyma:
                            self.border_aerenchyma.append(wall_id)
        
        # Identify border junctions
        junction_id = 0
        for junction, wall in self.junction_to_wall.items():
            count=0
            length=0
            for wall_id in wall:
                if wall_id in self.border_walls:
                    count += 1
                    length += self.wall_lengths[wall_id] / 4.0
            if count == 2:
                self.border_junction.append(junction_id + self.n_walls)
                self.border_link[junction_id + self.n_walls] = 1  # Junction node at the interface with soil
                self.wall_lengths[junction_id + self.n_walls] = length
            else:
                self.border_link[junction_id + self.n_walls] = 0
            junction_id+=1
    
    def create_cell_nodes(self, geometry:GeometryData, contagion: Any = 0):
        """Create nodes for cells"""
        cell_to_wall = self.cellset['cell_to_wall']
        position = nx.get_node_attributes(self.graph,'position') #Nodes XY positions (micrometers)
        # Initialize tracking arrays
        self.intercellular_cells = list(geometry.intercellular_ids)
        self.passage_cells = list(geometry.passage_cell_ids)
        
        for cell_group in cell_to_wall:
            cell_id = int(cell_group.getparent().get("id"))
            cell_type = int(cell_group.getparent().get("group"))
            
            # Calculate cell center from wall position
            wall_positions = []
            for wall_ref in cell_group:
                wall_id = int(wall_ref.get("id"))
                if wall_id in position:
                    wall_positions.append(position[wall_id])
            
            if not wall_positions:
                continue
            
            center_x = np.mean([p[0] for p in wall_positions])
            center_y = np.mean([p[1] for p in wall_positions])
            
            node_id = self.n_walls + self.n_junctions + cell_id
            
            self.graph.add_node(
                node_id,
                indice=node_id,
                type="cell",
                position=(center_x, center_y),
                cgroup=cell_type
            )
            
            # Track special cell types
            if cell_type in [11, 23]:  # Phloem sieve
                self.sieve_cells.append(node_id)
            elif cell_type in [13, 19, 20]:  # Xylem
                self.xylem_cells.append(node_id)
                for cell in cell_group:
                    wall_id = int(cell.get("id"))
                    self.xylem_walls.append(wall_id)

            if contagion:
                for cell in cell_group:
                    wall_id = int(cell.get("id"))
                    cc_id = self.apo_cc[self.apo_zombie0.index(cell_id)]
                    if cell_id in self.apo_zombie0:
                        cc_id = self.apo_cc[self.apo_zombie0.index(cell_id)]
                    if wall_id not in self.apo_wall_zombies0:
                        self.apo_wall_zombies0.append(wall_id)
                        self.apo_wall_cc.append(cc_id)
                    if cell_id in self.apo_target and wall_id not in self.apo_wall_target:
                        self.apo_wall_target.append(wall_id)
                    if cell_id in self.apo_immune and wall_id not in self.apo_wall_immune:
                        self.apo_wall_immune.append(wall_id)
                
    def build_membrane_connections(self):
        """Build membrane connections between cells and walls"""
        cell_to_wall = self.cellset['cell_to_wall']
        self.distance_wall_cell = np.zeros((self.n_walls, 1))
        position = nx.get_node_attributes(self.graph,'position') #Nodes XY positions (micrometers)
        
        for cell_group in cell_to_wall:
            cell_id = int(cell_group.getparent().get("id"))
            cell_node_id = self.n_walls + self.n_junctions + cell_id
            
            for wall_ref in cell_group:
                wall_id = int(wall_ref.get("id"))
                
                if wall_id >= self.n_walls:
                    continue
                
                # Calculate distance and direction
                position_cell = position[cell_node_id]
                position_wall = position[wall_id]
                
                d_vec = np.array([position_wall[0] - position_cell[0], position_wall[1] - position_cell[1]])
                dist = np.linalg.norm(d_vec)
                
                if dist > 0:
                    d_vec = d_vec / dist
                
                self.distance_wall_cell[wall_id] += dist

                self.graph.add_edge(
                    cell_node_id,
                    wall_id,
                    path='membrane',
                    length=self.wall_lengths[wall_id],
                    dist=dist,
                    d_vec=d_vec
                )
                self.n_membrane += 1 

    def build_wall_connections(self):
        """Build connections between walls and junctions"""
        position = nx.get_node_attributes(self.graph,'position') #Nodes XY positions (micrometers)

        for junction_id, wall_ids in self.junction_to_wall.items():
            for wall_id in wall_ids:
                if wall_id >= self.n_walls:
                    continue
                
                # Calculate direction vector
                position_junction = position[junction_id]
                position_wall = position[wall_id]
                
                d_vec = np.array([position_wall[0] - position_junction[0], position_wall[1] - position_junction[1]])
                dist = np.linalg.norm(d_vec)
                
                if dist > 0:
                    d_vec = d_vec / dist
                
                self.graph.add_edge(
                    junction_id,
                    wall_id,
                    path = 'wall',
                    length = self.wall_lengths[wall_id] / 2,
                    lateral_distance = self.distance_wall_cell[int(wall_id)][0],
                    d_vec = d_vec,
                    distnode_wall_cell = dist
                )
    
    def build_plasmodesmata_connections(self, rank_number: int = 65):
        """Build plasmodesmata connections between cells"""
        # Big time saver function compared to previous version
        walls_list = self.cellset['walls']
        position = nx.get_node_attributes(self.graph,'position') #Nodes XY positions (micrometers)
        
        self.n_cell_connections = np.zeros((self.n_cells, 1), dtype=int)
        self.cell_connections = -np.ones((self.n_cells, rank_number), dtype=int)

        # Build wall-to-cells mapping
        wall_to_cells = {}
        for wall_elem in walls_list:
            wall_id = int(wall_elem.get("id"))
            cell_id = int(wall_elem.getparent().getparent().get("id"))
            
            if wall_id not in wall_to_cells:
                wall_to_cells[wall_id] = []
            wall_to_cells[wall_id].append(cell_id)
            
        # Connect cells that share walls
        for wall_id, cell_ids in wall_to_cells.items():
            if len(cell_ids) == 2:
                cell1_node = self.n_walls + self.n_junctions + cell_ids[0]
                cell2_node = self.n_walls + self.n_junctions + cell_ids[1]
                
                pos1 = position[cell1_node]
                pos2 = position[cell2_node]
                
                d_vec = np.array([pos2[0] - pos1[0], pos2[1] - pos1[1]])
                dist = np.linalg.norm(d_vec)
                
                if dist > 0:
                    d_vec = d_vec / dist
                
                self.graph.add_edge(
                    cell1_node,
                    cell2_node,
                    path='plasmodesmata',
                    length=self.wall_lengths[wall_id] if wall_id in self.wall_lengths else 0,
                    d_vec=d_vec
                )

                # Update cell connections
                idx_node1 = self.n_cell_connections[cell_ids[0]]
                self.cell_connections[cell_ids[0]][idx_node1] = cell_ids[1]
                self.n_cell_connections[cell_ids[0]] += 1

                idx_node2 = self.n_cell_connections[cell_ids[1]]
                self.cell_connections[cell_ids[1]][idx_node2] = cell_ids[0]
                self.n_cell_connections[cell_ids[1]] += 1

    def compute_cell_properties(self):
        """Compute cell areas and perimeters using shoelace formula"""
        cell_to_wall = self.cellset['cell_to_wall']
        self.cell_areas = np.zeros(self.n_cells)
        self.cell_perimeters = np.zeros(self.n_cells)
        position = nx.get_node_attributes(self.graph,'position') #Nodes XY positions (micrometers)
        
        for cell_group in cell_to_wall:
            cell_id = int(cell_group.getparent().get("id"))
            
            # Collect ordered wall positions for area calculation
            wall_ids = []
            for wall_ref in cell_group:
                wall_id = int(wall_ref.get("id"))
                wall_ids.append(wall_id)
                
                # Add to perimeter
                if wall_id in self.wall_lengths:
                    self.cell_perimeters[cell_id] += self.wall_lengths[wall_id]
            
            # Calculate area using wall centers and junctions
            # This assumes walls are ordered anti-clockwise around the cell center
            if len(wall_ids) >= 3:
                area = 0.0
            
                for i in range(len(wall_ids)):
                    wall_id_1 = wall_ids[i]
                    wall_id_2 = wall_ids[(i + 1) % len(wall_ids)]  # Next wall (wraps around)
                    
                    wall_pos_1 = position[wall_id_1]
                    wall_pos_2 = position[wall_id_2]
                    
                    # Find the junction closest to wall1
                    # junction_positions[wall_id] contains [x1, y1, x2, y2] for the two junctions
                    junction_pos = self.junction_positions[wall_id_2]
                    
                    dist1 = np.hypot(
                        wall_pos_1[0] - junction_pos[0],
                        wall_pos_1[1] - junction_pos[1]
                    )
                    dist2 = np.hypot(
                        wall_pos_1[0] - junction_pos[2],
                        wall_pos_1[1] - junction_pos[3]
                    )
                    
                    # Choose closest junction
                    if dist1 < dist2:
                        j = 0  # Use first junction (indices 0, 1)
                    else:
                        j = 2  # Use second junction (indices 2, 3)
                    
                    # Add two segments to the area calculation (shoelace formula)
                    # Segment 1: from wall1 center to chosen junction of wall2
                    area += (wall_pos_1[0] + junction_pos[0 + j]) * (wall_pos_1[1] - junction_pos[1 + j])
                    
                    # Segment 2: from junction to wall2 center
                    area += (junction_pos[0 + j] + wall_pos_2[0]) * (junction_pos[1 + j] - wall_pos_2[1])

            self.cell_areas[cell_id] = abs(area) / 2.0

    def compute_gravity_center(self):
        """Compute gravity center of endodermis cells"""
        position = nx.get_node_attributes(self.graph,'position') #Nodes XY positions (micrometers)
        x_sum = 0.0
        y_sum = 0.0
        count = 0
        
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('type') == 'cell':
                if self.graph.nodes[node].get('cgroup') == 3:  # Endodermis
                    pos = position[node]
                    x_sum += pos[0]
                    y_sum += pos[1]
                    count += 1
        
        if count > 0:
            self.x_grav = x_sum / count
            self.y_grav = y_sum / count
    
    def compute_cell_surface(self, geometry: GeometryData):
        """Calculate cell surfaces at tissue interfaces"""
        indice = nx.get_node_attributes(self.graph,'indice') #Node indices (walls, junctions and cells)
        
        # Initialize counters
        self.len_outer_cortex = 0
        self.len_cortex_cortex = 0
        self.len_cortex_endo = 0
        self.cross_section_outer_cortex = 0
        self.cross_section_cortex_cortex = 0
        self.cross_section_cortex_endo = 0
        self.plasmodesmata_indice = []
        
        for node, edges in self.graph.adjacency():
            i = indice[node]
            
            # Skip walls and junctions
            if i < self.n_walls + self.n_junctions:
                continue
                
            node_group = self.graph.nodes[i]['cgroup']
            
            # Handle specific cell groups (16, 21)
            if node_group in [16, 21]:
                for neighboor, eattr in edges.items():
                    if eattr['path'] == "plasmodesmata" and self.graph.nodes[indice[neighboor]]['cgroup'] in [11, 23]:
                        self.plasmodesmata_indice.append(i - (self.n_walls + self.n_junctions))
                continue
            
            # Handle outer cortex, cortex, and endodermis (not intercellular)
            if node_group not in [self.outercortex_connec_rank, 3, 4]:
                continue
            if i - (self.n_walls + self.n_junctions) in geometry.intercellular_ids:
                continue
                
            for neighboor, eattr in edges.items():
                if eattr['path'] != "plasmodesmata":
                    continue
                    
                j = indice[neighboor]
                j_group = self.graph.nodes[j]['cgroup']
                length = eattr['length']
                is_not_intercellular = j - (self.n_walls + self.n_junctions) not in geometry.intercellular_ids
                
                # Outer cortex - cortex
                if {node_group, j_group} == {self.outercortex_connec_rank, 4}:
                    self.len_outer_cortex += length
                    if is_not_intercellular:
                        self.cross_section_outer_cortex += length
                # Cortex - cortex
                elif node_group == j_group == 4:
                    self.len_cortex_cortex += length
                    if is_not_intercellular:
                        self.cross_section_cortex_cortex += length
                # Cortex - endodermis
                elif {node_group, j_group} == {3, 4}:
                    self.len_cortex_endo += length
                    if is_not_intercellular:
                        self.cross_section_cortex_endo += length


    def rank_cells(self, geometry: GeometryData):
        """
        Assign ranks to cells based on tissue type and connectivity.
        
        Ranking system:
        - 1: Exodermis
        - 2: Epidermis
        - 3: Endodermis
        - 4: Cortex (updated to 40-49 based on layer)
        - 5: Stele (updated to 50-61 based on layer)
        - 11: Phloem sieve tube
        - 12: Companion cell
        - 13: Xylem
        - 16: Pericycle
        - 40-44: Cortex layers from endodermis outward
        - 45-49: Cortex layers from exodermis inward
        - 50-60: Stele layers from pericycle inward
        - 61: Central stele
        """
        self.cell_ranks = np.zeros(self.n_cells, dtype=int)
        self.layer_dist = np.zeros(62)
        self.n_layer = np.zeros(62, dtype=int)

        position = nx.get_node_attributes(self.graph,'position') #Nodes XY positions (micrometers)
        
        # First pass: Basic cell type assignment
        for node_id in range(self.n_walls + self.n_junctions, 
                            self.n_walls + self.n_junctions + self.n_cells):
            cell_id = node_id - self.n_walls - self.n_junctions
            cgroup = self.graph.nodes[node_id].get('cgroup', 0)
            
            # Normalize cell groups to standard types
            if cgroup in [19, 20]:  # Proto- and Meta-xylem
                cgroup = 13
            elif cgroup == 21:  # Xylem pole pericycle
                cgroup = 16
            elif cgroup == 23:  # Phloem
                cgroup = 11
            elif cgroup == 26:  # Companion cell
                cgroup = 12
            
            self.cell_ranks[cell_id] = cgroup
            
            # Calculate distance from gravity center
            pos = position[node_id]
            dist = np.hypot(pos[0] - self.x_grav, pos[1] - self.y_grav)
            
            self.layer_dist[cgroup] += dist
            self.n_layer[cgroup] += 1
            
            # Track xylem distances
            if cgroup == 13:
                self.xylem_distance.append(dist)
        
        # Calculate xylem 80th percentile distance
        self.xylem_80_percentile_distance = np.percentile(self.xylem_distance, 80) if self.xylem_distance else 0
        
        # Determine connection ranks based on tissue presence
        if self.n_layer[16] == 0:  # No pericycle
            self.stele_connec_rank = 3  # Endodermis connects to stele
        else:
            self.stele_connec_rank = 16  # Pericycle connects to stele

        if self.n_layer[1] == 0:  # No exodermis
            self.outercortex_connec_rank = 2  # Epidermis connects to cortex
        else:
            self.outercortex_connec_rank = 1  # Exodermis connects to cortex
        
        # Second pass: Initial layer assignment
        for cell_id in range(self.n_cells):
            node_id = self.n_walls + self.n_junctions + cell_id
            celltype = self.cell_ranks[cell_id]
            pos = position[node_id]
            
            # Get connected cells
            connected_ranks = self._get_connected_cell_ranks(cell_id)
            
            if celltype == 4:  # Cortex
                if 3 in connected_ranks:  # Connected to endodermis
                    self.cell_ranks[cell_id] = 40
                    dist = np.hypot(pos[0] - self.x_grav, pos[1] - self.y_grav)
                    self.layer_dist[40] += dist
                    self.n_layer[40] += 1
                    
                    # Check for intercellular spaces
                    if self.cell_perimeters[cell_id] < geometry.interc_perims[0]:
                        geometry.intercellular_ids.append(cell_id)
                        
                elif self.outercortex_connec_rank in connected_ranks:  # Connected to outer layer
                    self.cell_ranks[cell_id] = 49
                    dist = np.hypot(pos[0] - self.x_grav, pos[1] - self.y_grav)
                    self.layer_dist[49] += dist
                    self.n_layer[49] += 1
                    
                    if self.cell_perimeters[cell_id] < geometry.interc_perims[4]:
                        geometry.intercellular_ids.append(cell_id)
            
            elif celltype in [5, 11, 12, 13]:  # Stele tissues
                if self.stele_connec_rank in connected_ranks:  # Connected to pericycle
                    self.cell_ranks[cell_id] = 50
                    dist = np.hypot(pos[0] - self.x_grav, pos[1] - self.y_grav)
                    self.layer_dist[50] += dist
                    self.n_layer[50] += 1
                    
                    # Track protophloem
                    cgroup = self.graph.nodes[node_id].get('cgroup', 0)
                    if cgroup in [11, 23]:
                        self.protosieve_list.append(node_id)
        
        # Iterative pass: Refine layer rankings
        for iteration in range(12):
            for cell_id in range(self.n_cells):
                node_id = self.n_walls + self.n_junctions + cell_id
                celltype = self.cell_ranks[cell_id]
                pos = position[node_id]
                
                connected_ranks = self._get_connected_cell_ranks(cell_id)
                
                # Cortex layers (up to 4 layers from each side)
                if celltype == 4 and iteration < 4:
                    # Inward from endodermis
                    if (40 + iteration) in connected_ranks:
                        self.cell_ranks[cell_id] = 41 + iteration
                        dist = np.hypot(pos[0] - self.x_grav, pos[1] - self.y_grav)
                        self.layer_dist[41 + iteration] += dist
                        self.n_layer[41 + iteration] += 1
                        
                        # Intercellular space detection
                        perimeter = self.cell_perimeters[cell_id]
                        if iteration == 0 and perimeter < geometry.interc_perims[1]:
                            geometry.intercellular_ids.append(cell_id)
                        elif iteration == 1 and perimeter < geometry.interc_perims[2]:
                            geometry.intercellular_ids.append(cell_id)
                        elif iteration == 2 and perimeter < geometry.interc_perims[3]:
                            geometry.intercellular_ids.append(cell_id)
                        elif iteration > 2 and perimeter < geometry.interc_perims[4]:
                            geometry.intercellular_ids.append(cell_id)
                    
                    # Outward from exodermis
                    elif (49 - iteration) in connected_ranks:
                        self.cell_ranks[cell_id] = 48 - iteration
                        dist = np.hypot(pos[0] - self.x_grav, pos[1] - self.y_grav)
                        self.layer_dist[48 - iteration] += dist
                        self.n_layer[48 - iteration] += 1
                        
                        if self.cell_perimeters[cell_id] < geometry.interc_perims[4]:
                            geometry.intercellular_ids.append(cell_id)
                
                # Stele layers (up to 10 layers from pericycle)
                elif celltype in [5, 11, 12, 13]:
                    if iteration < 10:
                        if (50 + iteration) in connected_ranks:
                            self.cell_ranks[cell_id] = 51 + iteration
                            dist = np.hypot(pos[0] - self.x_grav, pos[1] - self.y_grav)
                            self.layer_dist[51 + iteration] += dist
                            self.n_layer[51 + iteration] += 1
                    else:
                        # Central stele (beyond 10 layers)
                        self.cell_ranks[cell_id] = 61
                        dist = np.hypot(pos[0] - self.x_grav, pos[1] - self.y_grav)
                        self.layer_dist[61] += dist
                        self.n_layer[61] += 1
        
        # Calculate average layer distances
        for i in range(62):
            if self.n_layer[i] > 0:
                self.layer_dist[i] /= self.n_layer[i]
        
        # Store counts
        self.n_sieve = len(self.sieve_cells)
        self.n_protosieve = len(self.protosieve_list)

    def create_layer_discretization(self):
        """
        Create radial discretization for layer-wise hydraulic analysis.
        
        This maps cell ranks to computational rows and creates layer groups
        for radial hydraulic conductivity calculations.
        
        Layer structure (from center outward):
        - Stele layers (61 → 50)
        - Pericycle (16)
        - Endodermis (3) - 4 rows for inner/outer + passage cells
        - Cortex layers (40 → 49)
        - Exodermis (1) - 2 rows
        - Epidermis (2) - 1 row
        """
        # Short aliases
        n_layer    = self.n_layer
        layer_dist = self.layer_dist

        def weighted_avg(indices):
            """Weighted average of layer_dist over given indices, using n_layer as weights."""
            total_n = sum(n_layer[i] for i in indices)
            return sum(n_layer[i] * layer_dist[i] for i in indices) / total_n

        # --- Initialisation -------------------------------------------------------

        j = 0                      # Global row counter
        prev_j = 0                 # Row counter at the start of current tissue type
        rank_to_row = np.full(62, np.nan, dtype=float)

        distance_from_center = []  # will be converted to array at the end
        r_counts = []              # rows per tissue type (later r_discret = [total_rows, *r_counts])

        # --- Stele (ranks 61..50) -------------------------------------------------

        for i in range(61, 49, -1):
            if n_layer[i] > 0:
                rank_to_row[i] = j
                distance_from_center.append(layer_dist[i])
                j += 1

        # Finish tissue type: Stele
        r_counts.append(j - prev_j)
        prev_j = j

        # --- Pericycle (rank 16) --------------------------------------------------

        if n_layer[16] > 0:
            rank_to_row[16] = j
            distance_from_center.append(layer_dist[16])
            j += 1

            # Finish tissue type: Pericycle
            r_counts.append(j - prev_j)
            prev_j = j

        # --- Endodermis: 4 rows (inner/outer + passage cells) ---------------------

        rank_to_row[3] = j
        distance_from_center.extend([layer_dist[3]] * 4)
        j += 4

        # Finish tissue type: Endodermis
        r_counts.append(j - prev_j)
        prev_j = j

        # --- Cortex (starting reference layer i1 = 40) ----------------------------

        i1 = 40
        nLayer_ref    = n_layer[i1]
        Layer_dist_ref = layer_dist[i1]

        rank_to_row[i1] = j
        distance_from_center.append(layer_dist[i1])
        j += 1

        i1 = 41
        ratio_complete = 0.75

        while i1 < 50:  # Cortex
            if n_layer[i1] > ratio_complete * nLayer_ref:  # Likely complete layer
                rank_to_row[i1] = j
                distance_from_center.append(layer_dist[i1])
                j += 1
                nLayer_ref    = n_layer[i1]
                Layer_dist_ref = layer_dist[i1]
                i1 += 1

            elif n_layer[i1] > 0:  # Likely incomplete layer
                # Find next non-empty rank i2
                i2 = i1 + 1
                while i2 < 50 and n_layer[i2] == 0:
                    i2 += 1
                if i2 >= 50:
                    # No usable next layer in the cortex: just move on
                    i1 += 1
                    continue

                # Check if i1 + i2 together form (approximately) a full layer
                if n_layer[i1] + n_layer[i2] > ratio_complete * nLayer_ref:

                    # Case: i2 itself is a full layer
                    if n_layer[i2] > ratio_complete * nLayer_ref:

                        # Decide whether i1 joins i2 (new layer) or previous layer
                        d_i1_i2   = abs(layer_dist[i1] - layer_dist[i2])
                        d_i1_prev = abs(layer_dist[i1] - Layer_dist_ref)

                        if d_i1_i2 < d_i1_prev:
                            # i1 & i2 form a new layer together
                            rank_to_row[i1] = j
                            rank_to_row[i2] = j
                            avg_dist = weighted_avg([i1, i2])
                            distance_from_center.append(avg_dist)
                            j += 1
                        else:
                            # i1 joins previous layer; i2 is new layer alone
                            rank_to_row[i1] = j - 1
                            avg_prev = weighted_avg([i1])  # with previous reference
                            distance_from_center[j - 1] = (
                                (n_layer[i1] * layer_dist[i1] +
                                nLayer_ref * Layer_dist_ref) /
                                (n_layer[i1] + nLayer_ref)
                            )

                            rank_to_row[i2] = j
                            distance_from_center.append(layer_dist[i2])
                            j += 1

                    else:
                        # i1 and i2 together form a layer
                        rank_to_row[i1] = j
                        rank_to_row[i2] = j
                        avg_dist = weighted_avg([i1, i2])
                        distance_from_center.append(avg_dist)
                        j += 1

                    i1 = i2 + 1

                else:
                    # i1 + i2 are not a full layer; may need to consider i2+1 as well
                    if i2 + 1 < 50 and n_layer[i2 + 1] > 0:
                        d_i1_next = abs(layer_dist[i1] - layer_dist[i2 + 1])
                        d_i1_prev = abs(layer_dist[i1] - Layer_dist_ref)

                        if d_i1_next < d_i1_prev:
                            # i1 + i2 + i2+1 form a layer
                            rank_to_row[i1] = j
                            rank_to_row[i2] = j
                            rank_to_row[i2 + 1] = j
                            avg_dist = weighted_avg([i1, i2, i2 + 1])
                            distance_from_center.append(avg_dist)
                            j += 1

                        else:
                            # i1 joins previous; i2 (and maybe i2+1) make their own
                            rank_to_row[i1] = j  # correction in original: j replaces j-1

                            d_i2_next = abs(layer_dist[i2] - layer_dist[i2 + 1])
                            d_i2_prev = abs(layer_dist[i2] - Layer_dist_ref)

                            if d_i2_next < d_i2_prev:
                                # i2 + i2+1 form a layer;
                                # i1 merges with previous (update previous average)
                                avg_prev = (
                                    n_layer[i1] * layer_dist[i1] +
                                    nLayer_ref * Layer_dist_ref
                                ) / (n_layer[i1] + nLayer_ref)
                                distance_from_center[j] = avg_prev

                                rank_to_row[i2] = j
                                rank_to_row[i2 + 1] = j
                                avg_dist = weighted_avg([i2, i2 + 1])
                                distance_from_center.append(avg_dist)
                                j += 1

                            else:
                                # i2 merges with previous; i2+1 is a separate layer
                                rank_to_row[i2] = j - 1
                                avg_prev = (
                                    n_layer[i1] * layer_dist[i1] +
                                    n_layer[i2] * layer_dist[i2] +
                                    nLayer_ref * Layer_dist_ref
                                ) / (n_layer[i1] + n_layer[i2] + nLayer_ref)
                                distance_from_center[j] = avg_prev

                                rank_to_row[i2 + 1] = j
                                distance_from_center.append(layer_dist[i2 + 1])
                                j += 1

                    else:
                        # Only i1 + i2 available; merge them
                        rank_to_row[i1] = j
                        rank_to_row[i2] = j
                        avg_dist = weighted_avg([i1, i2])
                        distance_from_center.append(avg_dist)
                        j += 1

                    i1 = i2 + 2

            else:
                # n_layer[i1] == 0: likely no cortex layer here
                i1 += 1

        # Finish tissue type: Cortex
        r_counts.append(j - prev_j)
        prev_j = j

        # --- Exodermis (optional, rank 1) ----------------------------------------

        if n_layer[1] > 0:
            rank_to_row[1] = j
            distance_from_center.extend([layer_dist[1]] * 2)
            j += 2

            # Finish tissue type: Exodermis
            r_counts.append(j - prev_j)
            prev_j = j

        # --- Epidermis (rank 2) ---------------------------------------------------

        rank_to_row[2] = j
        distance_from_center.append(layer_dist[2])
        j += 1

        # Finish tissue type: Epidermis
        r_counts.append(j - prev_j)

        # --- Finalise outputs --------------------------------------------------
        total_rows = j
        r_discret = np.array([total_rows] + r_counts, dtype=int)
        distance_from_center = np.array(distance_from_center, dtype=float)

        # Outer cortex row: just before exodermis or epidermis
        row_outer_cortex = rank_to_row[1] - 1 if not np.isnan(rank_to_row[1]) else rank_to_row[2] - 1
        row_outer_cortex = int(row_outer_cortex)

        self.distance_from_center = distance_from_center
        self.r_discret = r_discret
        self.rank_to_row = rank_to_row
        self.row_outer_cortex = row_outer_cortex

    def compute_distance_from_center(self):
        """Compute distance"""
        cell_to_wall = self.cellset['cell_to_wall']
        position = nx.get_node_attributes(self.graph,'position') #Nodes XY positions (micrometers)

        self.distance_center_grav = np.zeros((self.n_walls,1))
        t = 0
        for cell_group in cell_to_wall:
            cell_id = int(cell_group.getparent().get("id")) #Cell ID number
            node_id = self.n_walls + self.n_junctions + cell_id
            for wall_ref in cell_group:
                wall_id = int(wall_ref.get("id"))
                self.distance_center_grav[wall_id] = np.sqrt(
                    (position[wall_id][0]-self.x_grav) ** 2 + (position[wall_id][1]-self.y_grav) ** 2
                    )
                if self.graph.nodes[node_id]['cgroup'] == 4: #Cortex
                    self.distance_max_cortex = max(
                        self.distance_max_cortex,
                        self.distance_center_grav[wall_id]
                        )
                    self.distance_min_cortex = min(
                        self.distance_min_cortex,
                        self.distance_center_grav[wall_id]
                        )
                elif self.graph.nodes[node_id]['cgroup']==2: #Epidermis
                    self.distance_avg_epi += self.distance_center_grav[wall_id]
                    t += 1.0
        self.distance_avg_epi /= t #Last step of averaging (note that we take both inner and outer membranes into account in the averaging)
        self.perimeter = 2*np.pi*self.distance_avg_epi[0]*1.0E-04 #(cm)

    def _get_connected_cell_ranks(self, cell_id):
        """Get ranks of all cells connected to this cell"""
        node_id = self.n_walls + self.n_junctions + cell_id
        connected_ranks = []
        
        for neighbor in self.graph.neighbors(node_id):
            # Check if neighbor is a cell (not wall or junction)
            if neighbor >= self.n_walls + self.n_junctions:
                neighbor_cell_id = neighbor - self.n_walls - self.n_junctions
                if neighbor_cell_id < self.n_cells:
                    connected_ranks.append(self.cell_ranks[neighbor_cell_id])
        
        return connected_ranks


    def _find_wall_midpoint(self, coords: List[Tuple], total_length: float) -> Tuple[float, float]:
        """Find the midpoint along a wall defined by coordinates"""
        target_length = total_length / 2.0
        cumulative = 0.0
        
        for i in range(len(coords) - 1):
            segment_length = np.hypot(
                coords[i+1][0] - coords[i][0],
                coords[i+1][1] - coords[i][1]
            )
            
            if cumulative + segment_length >= target_length:
                # Midpoint is in this segment
                remaining = target_length - cumulative
                t = remaining / segment_length if segment_length > 0 else 0
                
                mid_x = coords[i][0] + t * (coords[i+1][0] - coords[i][0])
                mid_y = coords[i][1] + t * (coords[i+1][1] - coords[i][1])
                return mid_x, mid_y
            
            cumulative += segment_length
        
        # Fallback to last point
        return coords[-1]

    def _calculate_xylem_area(self):
        # Calculate total area
        for cid in self.xylem_cells:
            area = self.cell_areas[cid - self.n_wall_junction]
            self.total_xylem_area += area
            self.xylem_area.append(area)
        self.xylem_area_ratio = self.xylem_area / self.total_xylem_area

    def _calculate_phloem_area(self):
        # Calculate total area
        for cid in self.protosieve_list:
            area = self.cell_areas[cid - self.n_wall_junction]
            self.total_phloem_area += area
            self.phloem_area.append(area)
        self.phloem_area_ratio = self.phloem_area / self.total_phloem_area

        
