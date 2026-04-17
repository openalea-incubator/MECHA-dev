#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#       mecha.utils.loader
#
#       File author(s):
#           Dilhan Ozturk, Adrien Heymans
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

from mecha.utils.data_loader import GeneralData, GeometryData
from mecha.utils.hydraulic_cell import HydraulicCellManager

from granap.network_base import AbstractNetwork
from shapely.geometry import Polygon

class NetworkBuilder(AbstractNetwork):
    """Builds the hydraulic network graph from XML cell data"""
    def __init__(self, source_network: AbstractNetwork = None):
        super().__init__()
        self._source_network = source_network

        self.cellset_data: Dict[str, Any] = {}
        self._is_populated: bool = False  # Guard against double population

        self.graph: nx.Graph = nx.Graph()

        # Network dimensions
        self.n_walls: int = 0
        self.n_junctions: int = 0
        self.n_wall_junction: int = 0
        self.n_cells: int = 0
        # self.n_total: int = 0
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
        self.wall_to_cells: Optional[np.ndarray] = None 
        self.junction_to_wall: Dict[Any, Any] = {}
        self.n_junction_to_wall: Dict[Any, Any] = {} 

        # Cellset artifacts
        self.list_ghostwalls: List[int] = [] #"Fake walls" not to be displayed
        self.list_ghostjunctions: List[int] = [] #"Fake junctions" not to be displayed
        self.n_ghost_junction2wall: int = 0
        
        # Gravity center and geometry
        self.x_grav: float = 0.0
        self.y_grav: float = 0.0
        self.x_min: float = 0.0
        self.x_max: float = 0.0
        
        # Layer discretization
        self.layer_dist: Optional[np.ndarray] = None 
        self.n_layer: Optional[np.ndarray] = None 
        self.rank_to_row: Optional[np.ndarray] = None 
        self.r_discret: Optional[np.ndarray] = None 
        self.distance_from_center: Optional[np.ndarray] = None
        self.row_outer_cortex: Optional[np.ndarray] = None

        # Relative positions for walls
        self.r_rel: Optional[np.ndarray] = None
        self.x_rel: Optional[np.ndarray] = None

        # Rank 
        self.stele_connec_rank: int = 3 # Default: Endodermis/passage cells
        self.outercortex_connec_rank: int = 1 # Default: Exodermis/hypodermis
        
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

        # Cell manager — populated after build_network / populate_from_network
        self.cell_manager: Optional[HydraulicCellManager] = None


    def _build_anatnetwork(self) -> None:
        """
        Implementation of AbstractNetwork._build_network.
        For NetworkBuilder, this is typically handled by build_network with arguments,
        or populate_from_network.
        """
        # If we are using valid input data, build_network should be called explicitly.
        # If we are populating from another network, this might not be needed.
        pass

    def build_network(self, general: GeneralData, geometry: GeometryData, cellset_data, verbose: bool = False, centroid_method: str = "shapely"):
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
        self.create_cell_nodes(geometry, general.apo_contagion, centroid_method=centroid_method)
        
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
        self.compute_cell_surface(geometry.intercellular_ids)
        self.create_layer_discretization()
        self.compute_distance_from_center()
        self.n_nodes = self.graph.number_of_nodes()
        self._calculate_xylem_area()
        self._calculate_phloem_area()
        self._build_cell_manager()
    
    def populate_from_network(self, type_mapper: Dict[str, int] = None) -> None:
        """
        Populate this NetworkBuilder from the stored source AbstractNetwork.

        Uses ``self._source_network`` (set during ``__init__``).
        Derives all MECHA-specific attributes from the graph structure
        without requiring XML cellset data.

        Parameters
        ----------
        type_mapper : Dict[str, int], optional
            Mapping from GRANAP cell type strings to MECHA cgroup integers.
        """
        if self._is_populated:
            return  # Already populated — skip to avoid double unit-scaling

        src = self._source_network
        if src is None:
            raise ValueError("No source_network was provided to NetworkBuilder.")

        # ------------------------------------------------------------------
        # Step 1: Copy graph & counts
        # ------------------------------------------------------------------
        self.graph = src.graph.copy()
        for _, data in self.graph.nodes(data=True):
            if 'position' in data:
                data['position'] = (data['position'][0] * 1000, data['position'][1] * 1000)
            if 'length' in data:
                data['length'] = data['length'] * 1000
            if 'area' in data:
                data['area'] = data['area'] * 1000**2
            if 'dist' in data:
                data['dist'] = data['dist'] * 1000

        for u, v, data in self.graph.edges(data=True):
            if 'length' in data:
                data['length'] = data['length'] * 1000
            if 'dist' in data:
                data['dist'] = data['dist'] * 1000
            if 'lateral_distance' in data:
                data['lateral_distance'] = data['lateral_distance'] * 1000
            if 'distnode_wall_cell' in data:
                data['distnode_wall_cell'] = data['distnode_wall_cell'] * 1000
                
        self.n_walls = src.n_walls
        self.n_junctions = src.n_junctions
        self.n_cells = src.n_cells
        self.n_wall_junction = self.n_walls + self.n_junctions
        self.n_nodes = self.graph.number_of_nodes()
        self.x_min = src._cells_gdf['x'].min()*1000
        self.x_max = src._cells_gdf['x'].max()*1000
        self.y_min = src._cells_gdf['y'].min()*1000
        self.y_max = src._cells_gdf['y'].max()*1000

        # ------------------------------------------------------------------
        # Step 2: Extract wall_lengths from graph node attributes
        # ------------------------------------------------------------------
        self.wall_lengths = {}
        for i in range(self.n_walls):
            node = self.graph.nodes[i]
            self.wall_lengths[i] = node.get('length', 0.0)

        # Also set junction wall_lengths to 0
        for i in range(self.n_walls, self.n_wall_junction):
            self.wall_lengths[i] = self.graph.nodes[i].get('length', 0.0)

        # Step 3: Map cell types via type_mapper → set cgroup
        if type_mapper is None:
            type_mapper = {
                'exodermis': 1,
                'epidermis': 2,
                'endodermis': 3,
                'passage': 3,
                'cortex': 4,
                'mesophyll':4,
                'stele': 5,
                'pith': 5,
                'parenchyma': 5,
                'vascular_parenchyma': 5,
                'phloem': 11,
                'companion': 12,               
                'cambium':12,
                'guard cell': 12,
                'Strasburger cell': 12,
                'xylem': 13,
                'protoxylem': 13,
                'metaxylem': 13,
                'pericycle': 16,
            }

        for i in range(self.n_wall_junction, self.n_wall_junction + self.n_cells):
            node = self.graph.nodes[i]
            cell_type_str = node.get('cell_type', '')
            cgroup = node.get('cgroup')

            # Map string cgroup or missing/invalid cgroup using cell_type_str
            try:
                cgroup = int(cgroup)
            except (ValueError, TypeError):
                cgroup = type_mapper.get(cell_type_str, 4)  # default cortex
            self.graph.nodes[i]['cgroup'] = cgroup

        # which outer layer is connected to the cortex/mesophyll
        # do we have a cgroup of 1? if not outercortex_connec_rank = 2, direct connection to the epidermis
        self.outercortex_connec_rank = 2
        for i in range(self.n_wall_junction, self.n_wall_junction + self.n_cells):
            if self.graph.nodes[i]['cgroup'] == 1:
                self.outercortex_connec_rank = 1
                break

        # Step 4: Rebuild junction_to_wall / n_junction_to_wall from edges mimicking create_wall_junction_nodes
        old_juncs = list(range(self.n_walls, self.n_wall_junction))
        wall_to_junc_coords = {}
        for wall_id in range(self.n_walls):
            junc_coords = []
            for neighbor in self.graph.neighbors(wall_id):
                if neighbor in old_juncs:
                    edge_data = self.graph.edges[wall_id, neighbor]
                    if edge_data.get('path') == 'wall':
                        junc_coords.append(self.graph.nodes[neighbor]['position'])
            wall_to_junc_coords[wall_id] = junc_coords
        
        # Remove old junctions and their edges
        self.graph.remove_nodes_from(old_juncs)

        self.junction_to_wall = {}
        self.n_junction_to_wall = {}
        self.junction_positions = {}
        junction_list = {}
        junction_ni = 0
        n_dec_position = 6

        for wall_id in range(self.n_walls):
            coords = wall_to_junc_coords.get(wall_id, [])
            if len(coords) < 2:
                continue
            
            # Store junction positions for this wall
            self.junction_positions[wall_id] = [
                coords[0][0], coords[0][1],  # First junction
                coords[1][0], coords[1][1]   # Last junction
            ]
            
            for coord in coords:
                # Use raw unrounded floats for the dedup key to mimic XML parsing path
                pos_key = "x" + str(coord[0]) + "y" + str(coord[1])
                
                if pos_key not in junction_list:
                    node_id = self.n_walls + junction_ni
                    self.graph.add_node(
                        node_id,
                        indice=node_id,
                        type="apo",
                        position=(round(coord[0], n_dec_position), round(coord[1], n_dec_position)),
                        length=0
                    )
                    junction_list[pos_key] = node_id
                    self.junction_to_wall[node_id] = [wall_id]
                    self.n_junction_to_wall[node_id] = 1
                    junction_ni += 1
                else:
                    junction_id = junction_list[pos_key]
                    self.junction_to_wall[junction_id].append(wall_id)
                    self.n_junction_to_wall[junction_id] += 1
                    
        self.n_junctions = junction_ni
        self.n_wall_junction = self.n_walls + self.n_junctions

        # Step 5: Compute cell_areas, cell_perimeters from source or graph
        self.cell_areas = np.zeros(self.n_cells)
        self.cell_perimeters = np.zeros(self.n_cells)
        self.cell_types = [''] * self.n_cells

        # Try to get areas from the source_network's all_cells if available
        if hasattr(src, 'all_cells') and hasattr(src.all_cells, 'cells'):
            cells_list = src.all_cells.cells
            for idx, cell in enumerate(cells_list):
                if idx < self.n_cells:
                    if hasattr(cell, 'area') and cell.area is not None:
                        self.cell_areas[idx] = cell.area*1000**2
                    if hasattr(cell, 'polygon') and cell.polygon is not None:
                        self.cell_perimeters[idx] = cell.polygon.length*1000
                    if hasattr(cell, 'type') and cell.type is not None:
                        self.cell_types[idx] = cell.type

        # Fallback: compute from membrane edges if areas are still zero
        if np.sum(self.cell_areas) == 0:
            for i in range(self.n_cells):
                node_id = self.n_wall_junction + i
                # Sum wall lengths around the cell for perimeter
                perimeter = 0.0
                for neighbor in self.graph.neighbors(node_id):
                    edge_data = self.graph.edges[node_id, neighbor]
                    if edge_data.get('path') == 'membrane':
                        perimeter += edge_data.get('length', 0.0)
                self.cell_perimeters[i] = perimeter
                # Rough area estimate: circle with this perimeter
                if perimeter > 0:
                    self.cell_areas[i] = (perimeter ** 2) / (4 * np.pi)

        # Step 6: Compute distance_wall_cell from graph positions
        position = nx.get_node_attributes(self.graph, 'position')
        # position = {k: (v[0]/1000, v[1]/1000) for k, v in position_raw.items()}
        self.distance_wall_cell = np.zeros((self.n_walls, 1))

        for i in range(self.n_walls):
            total_dist = 0.0
            n_membranes = 0
            for neighbor in self.graph.neighbors(i):
                edge_data = self.graph.edges[i, neighbor]
                if edge_data.get('path') == 'membrane':
                    total_dist += edge_data.get('dist', 0.0)
                    n_membranes += 1
            self.distance_wall_cell[i] = total_dist

        # Step 7: Identify border_walls (walls with only 1 membrane neighbor)
        self.border_walls = []
        self.border_aerenchyma = []
        self.border_link = 2 * np.ones((self.n_wall_junction, 1), dtype=int)

        for i in range(self.n_walls):
            membrane_cells = []
            for neighbor in self.graph.neighbors(i):
                edge_data = self.graph.edges[i, neighbor]
                if edge_data.get('path') == 'membrane':
                    membrane_cells.append(neighbor)

            if len(membrane_cells) == 1:
                cell_cgroup = self.graph.nodes[membrane_cells[0]].get('cgroup', 0)
                if cell_cgroup == 2:  # Epidermis → soil border
                    self.border_walls.append(i)
                else:  # Other single-membrane = aerenchyma border
                    self.border_aerenchyma.append(i)
                self.border_link[i] = 1
            else:
                self.border_link[i] = 0

        # Step 8: Identify border_junction from border walls
        self.border_junction = []
        for j_id, wall_ids in self.junction_to_wall.items():
            count = 0
            length = 0.0
            for wall_id in wall_ids:
                if wall_id in self.border_walls:
                    count += 1
                    length += self.wall_lengths.get(wall_id, 0.0) / 4.0
            if count == 2:
                self.border_junction.append(j_id)
                self.border_link[j_id] = 1
                self.wall_lengths[j_id] = length
            else:
                self.border_link[j_id] = 0

        # Count membranes
        self.n_membrane = 0
        self.n_membrane_from_epi = 0
        for u, v, data in self.graph.edges(data=True):
            if data.get('path') == 'membrane':
                self.n_membrane += 1

        # Step 9: Compute gravity center from endodermis cells
        self._compute_gravity_center_from_graph(position)

        # Step 10: Track xylem/phloem/passage/intercellular cells
        self.xylem_cells = []
        self.sieve_cells = []
        self.xylem_walls = []
        self.passage_cells = []
        self.intercellular_cells = []

        for i in range(self.n_cells):
            node_id = self.n_wall_junction + i
            cgroup = self.graph.nodes[node_id].get('cgroup', 0)
            cell_type_str = self.graph.nodes[node_id].get('cell_type', '')

            if cgroup in [13, 19, 20]:
                self.xylem_cells.append(node_id)
                # Find walls connected to this xylem cell
                for neighbor in self.graph.neighbors(node_id):
                    edge_data = self.graph.edges[node_id, neighbor]
                    if edge_data.get('path') == 'membrane' and neighbor < self.n_walls:
                        self.xylem_walls.append(neighbor)
            elif cgroup in [11, 23]:
                self.sieve_cells.append(node_id)
            elif cell_type_str == 'passage':
                self.passage_cells.append(i)
            elif cell_type_str == 'intercellular':
                self.intercellular_cells.append(i)
            elif cell_type_str == 'air space':
                self.intercellular_cells.append(i)
            elif cell_type_str == 'aerenchyma':
                self.intercellular_cells.append(i)

        self.compute_cell_surface(self.intercellular_cells)

        # Step 11: Rank cells from graph
        self._rank_cells_from_graph(position)

        # Step 12: Create layer discretization
        self.create_layer_discretization()

        # Step 13: Compute distance_center_grav / distance_from_center
        self._compute_distance_from_center_graph(position)

        # Step 14: Calculate xylem/phloem areas
        self._calculate_xylem_area()
        self._calculate_phloem_area()

        # Build cell connections for symplastic paths
        self._build_cell_connections_from_graph()
        self._build_cell_manager()
        
        # Build new wall connections to junctions
        self.build_wall_connections()

        self._is_populated = True

    # ------------------------------------------------------------------
    # Cell manager construction
    # ------------------------------------------------------------------

    def _build_cell_manager(self) -> None:
        """Construct ``self.cell_manager`` from the current graph state.

        Called automatically at the end of :meth:`build_network` and
        :meth:`populate_from_network`.  Safe to call multiple times — the
        manager is rebuilt from scratch each time.
        """
        manager = HydraulicCellManager()
        manager.sync_from_network(self)
        self.cell_manager = manager

    # ------------------------------------------------------------------
    # Graph-only helper methods (used by populate_from_network)
    # ------------------------------------------------------------------

    def _compute_gravity_center_from_graph(self, position: dict) -> None:
        """Compute gravity center from endodermis cell positions."""
        x_sum, y_sum, count = 0.0, 0.0, 0
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('type') == 'cell':
                if self.graph.nodes[node].get('cgroup') == 3:
                    pos = position.get(node)
                    if pos:
                        x_sum += pos[0]
                        y_sum += pos[1]
                        count += 1
        if count > 0:
            self.x_grav = x_sum / count
            self.y_grav = y_sum / count

    def _rank_cells_from_graph(self, position: dict) -> None:
        """
        Rank cells using cgroup and connectivity — graph-only version.
        Equivalent to rank_cells but without XML cellset dependency.
        """
        self.cell_ranks = np.zeros(self.n_cells, dtype=int)
        self.layer_dist = np.zeros(62)
        self.n_layer = np.zeros(62, dtype=int)
        self.xylem_distance = []

        # First pass: basic cell type assignment
        for cell_id in range(self.n_cells):
            node_id = self.n_wall_junction + cell_id
            cgroup = self.graph.nodes[node_id].get('cgroup', 0)

            # Normalize cell groups
            if cgroup in [19, 20]:
                cgroup = 13
            elif cgroup == 21:
                cgroup = 16
            elif cgroup == 23:
                cgroup = 11
            elif cgroup == 26:
                cgroup = 12

            self.cell_ranks[cell_id] = cgroup
            pos = position.get(node_id, (0, 0))
            dist = np.hypot(pos[0] - self.x_grav, pos[1] - self.y_grav)
            self.layer_dist[cgroup] += dist
            self.n_layer[cgroup] += 1

            if cgroup == 13:
                self.xylem_distance.append(dist)

        self.xylem_80_percentile_distance = (
            np.percentile(self.xylem_distance, 80) if self.xylem_distance else 0
        )

        # Determine connection ranks
        if self.n_layer[16] == 0:
            self.stele_connec_rank = 3
        else:
            self.stele_connec_rank = 16

        if self.n_layer[1] == 0:
            self.outercortex_connec_rank = 2
        else:
            self.outercortex_connec_rank = 1

        # Second pass: initial layer assignment
        for cell_id in range(self.n_cells):
            node_id = self.n_wall_junction + cell_id
            celltype = self.cell_ranks[cell_id]
            pos = position.get(node_id, (0, 0))
            connected_ranks = self._get_connected_cell_ranks(cell_id)

            if celltype == 4:  # Cortex
                if 3 in connected_ranks:
                    self.cell_ranks[cell_id] = 40
                    dist = np.hypot(pos[0] - self.x_grav, pos[1] - self.y_grav)
                    self.layer_dist[40] += dist
                    self.n_layer[40] += 1
                elif self.outercortex_connec_rank in connected_ranks:
                    self.cell_ranks[cell_id] = 49
                    dist = np.hypot(pos[0] - self.x_grav, pos[1] - self.y_grav)
                    self.layer_dist[49] += dist
                    self.n_layer[49] += 1

            elif celltype in [5, 11, 12, 13]:
                if self.stele_connec_rank in connected_ranks:
                    self.cell_ranks[cell_id] = 50
                    dist = np.hypot(pos[0] - self.x_grav, pos[1] - self.y_grav)
                    self.layer_dist[50] += dist
                    self.n_layer[50] += 1
                    cgroup = self.graph.nodes[node_id].get('cgroup', 0)
                    if cgroup in [11, 23]:
                        self.protosieve_list.append(node_id)

        # Iterative pass: refine layer rankings
        for iteration in range(12):
            for cell_id in range(self.n_cells):
                node_id = self.n_wall_junction + cell_id
                celltype = self.cell_ranks[cell_id]
                pos = position.get(node_id, (0, 0))
                connected_ranks = self._get_connected_cell_ranks(cell_id)

                if celltype == 4 and iteration < 4:
                    if (40 + iteration) in connected_ranks:
                        self.cell_ranks[cell_id] = 41 + iteration
                        dist = np.hypot(pos[0] - self.x_grav, pos[1] - self.y_grav)
                        self.layer_dist[41 + iteration] += dist
                        self.n_layer[41 + iteration] += 1
                    elif (49 - iteration) in connected_ranks:
                        self.cell_ranks[cell_id] = 48 - iteration
                        dist = np.hypot(pos[0] - self.x_grav, pos[1] - self.y_grav)
                        self.layer_dist[48 - iteration] += dist
                        self.n_layer[48 - iteration] += 1

                elif celltype in [5, 11, 12, 13]:
                    if iteration < 10:
                        if (50 + iteration) in connected_ranks:
                            self.cell_ranks[cell_id] = 51 + iteration
                            dist = np.hypot(pos[0] - self.x_grav, pos[1] - self.y_grav)
                            self.layer_dist[51 + iteration] += dist
                            self.n_layer[51 + iteration] += 1
                    else:
                        self.cell_ranks[cell_id] = 61
                        dist = np.hypot(pos[0] - self.x_grav, pos[1] - self.y_grav)
                        self.layer_dist[61] += dist
                        self.n_layer[61] += 1

        # Average distances
        for i in range(62):
            if self.n_layer[i] > 0:
                self.layer_dist[i] /= self.n_layer[i]

        self.n_sieve = len(self.sieve_cells)
        self.n_protosieve = len(self.protosieve_list)

    def _compute_distance_from_center_graph(self, position: dict) -> None:
        """Compute wall→gravity-center distances from graph positions."""
        self.distance_center_grav = np.zeros((self.n_walls, 1))
        self.distance_max_cortex = 0.0
        self.distance_min_cortex = np.inf
        self.distance_avg_epi = 0.0
        t = 0

        for i in range(self.n_walls):
            pos = position.get(i, (0, 0))
            d = np.sqrt((pos[0] - self.x_grav) ** 2 + (pos[1] - self.y_grav) ** 2)
            self.distance_center_grav[i] = d

            # Check which cells this wall is connected to via membrane
            for neighbor in self.graph.neighbors(i):
                edge_data = self.graph.edges[i, neighbor]
                if edge_data.get('path') == 'membrane':
                    cgroup = self.graph.nodes[neighbor].get('cgroup', 0)
                    if cgroup == 4:  # Cortex
                        self.distance_max_cortex = max(self.distance_max_cortex, d)
                        self.distance_min_cortex = min(self.distance_min_cortex, d)
                    elif cgroup == 2:  # Epidermis
                        self.distance_avg_epi += d
                        t += 1

        if t > 0:
            self.distance_avg_epi /= t
        self.perimeter = 2 * np.pi * self.distance_avg_epi * 1.0E-04  # (cm)

    def _build_cell_connections_from_graph(self, rank_number: int = 65) -> None:
        """Build cell-to-cell connection arrays from plasmodesmata edges."""
        self.n_cell_connections = np.zeros((self.n_cells, 1), dtype=int)
        self.cell_connections = -np.ones((self.n_cells, rank_number), dtype=int)

        for u, v, data in self.graph.edges(data=True):
            if data.get('path') == 'plasmodesmata':
                if u >= self.n_wall_junction and v >= self.n_wall_junction:
                    cell_id_u = u - self.n_wall_junction
                    cell_id_v = v - self.n_wall_junction
                    if cell_id_u < self.n_cells and cell_id_v < self.n_cells:
                        idx_u = self.n_cell_connections[cell_id_u][0]
                        if idx_u < rank_number:
                            self.cell_connections[cell_id_u][idx_u] = cell_id_v
                            self.n_cell_connections[cell_id_u] += 1

                        idx_v = self.n_cell_connections[cell_id_v][0]
                        if idx_v < rank_number:
                            self.cell_connections[cell_id_v][idx_v] = cell_id_u
                            self.n_cell_connections[cell_id_v] += 1


    def create_wall_junction_nodes(self, im_scale: float, n_dec_position: int = 6):
        points = self.cellset['points']

        junction_ni = 0
        self.junction_to_wall = {}
        self.n_junction_to_wall = {}
        junction_list = {}
        for point_groups in points: #Loop on wall elements

            wall_id = int((point_groups.getparent().get)("id")) # wall_id records the current wall id number

            coords     = []  # rounded, used for geometry
            coords_raw = []  # unrounded, used for dedup key (mirrors old MECHA str(x)+str(y) approach)
            for point in point_groups:
                x_raw = im_scale * float(point.get("x"))
                y_raw = im_scale * float(point.get("y"))
                coords_raw.append((x_raw, y_raw))
                coords.append((round(x_raw, n_dec_position), round(y_raw, n_dec_position)))

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

            # Add junction node — use raw (unrounded) floats for the dedup key so that
            # two walls whose endpoint was written from the same Python float get the
            # identical key after the XML round-trip.  This matches old MECHA's
            # `pos = "x" + str(x) + "y" + str(y)` pattern.
            for raw_coord, coord in zip([coords_raw[0], coords_raw[-1]],
                                        [coords[0],     coords[-1]]):
                pos_key = "x" + str(raw_coord[0]) + "y" + str(raw_coord[1])

                if pos_key not in junction_list:
                    node_id = self.n_walls + junction_ni
                    self.graph.add_node(
                        node_id,
                        indice=node_id,
                        type="apo",
                        position=coord,   # rounded position for graph geometry
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
    
    def create_cell_nodes(self, geometry:GeometryData, contagion: Any = 0, centroid_method: str = "shapely"):
        """Create nodes for cells"""
        cell_to_wall = self.cellset['cell_to_wall']
        position = nx.get_node_attributes(self.graph,'position') #Nodes XY positions (micrometers)
        # Initialize tracking arrays
        self.intercellular_cells = list(geometry.intercellular_ids)
        self.passage_cells = list(geometry.passage_cell_ids)
        
        poly_dict = {}
        if centroid_method == "shapely":
            from mecha.utils.visu import prep_section
            gdf = prep_section(self.cellset)
            poly_dict = {row["id_cell"]: row["geometry"] for _, row in gdf.iterrows()}
        
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
            
            if centroid_method == "shapely":
                poly = poly_dict.get(cell_id)
                if poly is not None and not poly.is_empty:
                    center_x = poly.centroid.x * geometry.im_scale
                    center_y = poly.centroid.y * geometry.im_scale
                else:
                    center_x = np.mean([p[0] for p in wall_positions])
                    center_y = np.mean([p[1] for p in wall_positions])
            else:
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
            
            if len(wall_ids) >= 3:
                poly_points = []
            
                for i in range(len(wall_ids)):
                    wall_id_1 = wall_ids[i]
                    wall_id_2 = wall_ids[(i + 1) % len(wall_ids)]
                    
                    wall_pos_1 = position[wall_id_1]
                    
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
                    j = 0 if dist1 < dist2 else 2
                    
                    # Append wall center and junction to the polygon
                    poly_points.append((wall_pos_1[0], wall_pos_1[1]))
                    poly_points.append((junction_pos[j], junction_pos[j + 1]))

                self.cell_areas[cell_id] = Polygon(poly_points).area

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
    
    def compute_cell_surface(self, intercellular_ids: List[int]):
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
            if i - (self.n_walls + self.n_junctions) in intercellular_ids:
                continue
                
            for neighboor, eattr in edges.items():
                if eattr['path'] != "plasmodesmata":
                    continue
                    
                j = indice[neighboor]
                j_group = self.graph.nodes[j]['cgroup']
                length = eattr['length']
                is_not_intercellular = j - (self.n_walls + self.n_junctions) not in intercellular_ids
                
                # Exodermis/hypodermis or epidermis if no exodermis outer layer - cortex
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
        # each element of the list is divided by the total area
        for xyl_area in self.xylem_area:
            self.xylem_area_ratio.append(xyl_area / self.total_xylem_area)

    def _calculate_phloem_area(self):
        # Calculate total area
        
        for cid in self.protosieve_list:
            area = self.cell_areas[cid - self.n_wall_junction]
            self.total_phloem_area += area
            self.phloem_area.append(area)
        # each element of the list is divided by the total area
        for phl_area in self.phloem_area:
            self.phloem_area_ratio.append(phl_area / self.total_phloem_area)

    def compute_relative_positions(self):
        """
        Compute radial (r_rel) and horizontal (x_rel) positions for each wall.
        r_rel: negative for stelar side, positive for cortical side.
        x_rel: relative bound based on overall x_min and x_max.
        """
        self.r_rel = np.full((self.n_walls, 1), np.nan)
        self.x_rel = np.full((self.n_wall_junction + self.n_cells, 1), np.nan)
        
        position = nx.get_node_attributes(self.graph, 'position')
        
        # Build mapping of wall to valid connected cells based on 'membrane' path
        wall_to_cell = [[] for _ in range(self.n_walls)]
        for u, v, data in self.graph.edges(data=True):
            if data.get('path') == 'membrane':
                if u < self.n_walls and v >= self.n_wall_junction:
                    wall_to_cell[u].append(v)
                elif v < self.n_walls and u >= self.n_wall_junction:
                    wall_to_cell[v].append(u)

        def get_row_from_cell_id(cid):
            if cid - self.n_wall_junction >= self.n_cells:
                return np.nan
            cr = self.cell_ranks[int(cid - self.n_wall_junction)]
            row = self.rank_to_row[int(cr.item() if hasattr(cr, "item") else cr)]
            return row.item() if hasattr(row, "item") else row

        def radial_position(row1, row2, cid2_exists):
            def dc_val(idx):
                v = self.distance_from_center[int(idx)]
                return v[0] if hasattr(v, "__len__") and len(v) == 1 else v

            d_endodermis = self.layer_dist[3]
            d_epidermis  = self.layer_dist[2]
            xylem80 = self.xylem_80_percentile_distance

            # Handle exception when row1 is nan
            if pd.isna(row1) if 'pd' in globals() else np.isnan(row1):
                return np.nan

            if row2 is not None and not np.isnan(row2) and row1 <= self.rank_to_row[3] and row2 <= self.rank_to_row[3]:
                # Stelar side, two cells
                rad_pos = -((dc_val(row1) + dc_val(row2)) / 2.0 - xylem80) / (d_endodermis - xylem80)
                return max(min(rad_pos, -0.00001), -1.0)
            elif (row2 is None or np.isnan(row2)) and row1 <= self.rank_to_row[3]:
                # Stelar side, single cell
                rad_pos = -(dc_val(row1) - xylem80) / (d_endodermis - xylem80)
                return max(min(rad_pos, -0.00001), -1.0)
            elif row2 is not None and not np.isnan(row2):
                # Cortical side, two cells
                rad_pos = (d_epidermis - (dc_val(row1) + dc_val(row2)) / 2.0) / (d_epidermis - d_endodermis)
                return min(max(rad_pos, 0.00001), 1.0)
            else:
                # Cortical side, single cell
                rad_pos = (d_epidermis - dc_val(row1)) / (d_epidermis - d_endodermis)
                return min(max(rad_pos, 0.00001), 1.0)

        for wall_id in range(self.n_walls):
            cells = wall_to_cell[wall_id]
            if len(cells) == 0:
                self.r_rel[wall_id] = np.nan
            else:
                cid1 = cells[0]
                cid2 = cells[1] if len(cells) > 1 else None
                row1 = get_row_from_cell_id(cid1)
                row2 = get_row_from_cell_id(cid2) if cid2 is not None else np.nan
                
                self.r_rel[wall_id] = radial_position(row1, row2, cid2 is not None)
                
            if wall_id in position:
                self.x_rel[wall_id] = (position[wall_id][0] - self.x_min) / (self.x_max - self.x_min) if (self.x_max - self.x_min) > 0 else 0.0

    def get_relative_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute and return r_rel (radial position to endodermis) and 
        x_rel (relative position to x bounding) for all walls.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - r_rel: Radial position mapping of the walls related to the endodermis.
            - x_rel: Relative mapping of the wall nodes along the x bounding box.
        """
        if getattr(self, 'r_rel', None) is None or getattr(self, 'x_rel', None) is None:
            self.compute_relative_positions()
        return self.r_rel, self.x_rel
