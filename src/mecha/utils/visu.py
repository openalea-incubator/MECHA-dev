import geopandas as gpd
from shapely.ops import polygonize
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Polygon
from typing import Tuple, Dict, List, Any
from mecha.utils.network_builder import NetworkBuilder
import networkx as nx


def prep_section(cellset_data) -> gpd.GeoDataFrame:
    """
    Constructs cell polygons from cellset data.
    If polygonization fails for a cell (open or disjoint boundary),
    a fallback method orders all cell wall points around their centroid
    and creates a Polygon manually.

    Args
    ----
    cellset_data : Dict[str, Any]
        Dictionary containing parsed cellset data from parse_cellset.

    Returns
    -------
    gpd.GeoDataFrame
        Columns: id_cell, type, geometry (Polygon).
    """
    root = cellset_data['root']

    # --- Parse groups (cell types) ---
    group_map = {}
    cellgroups_elem = root.find("groups/cellgroups")
    if cellgroups_elem is not None:
        for group_elem in cellgroups_elem.findall("group"):
            group_id = int(group_elem.get("id"))
            if group_id == 4:
                group_name = "cortex"
            elif group_id == 3:
                group_name = "endodermis"
            else:
                group_name = group_elem.get("name")
            group_map[group_id] = group_name

    # --- Parse walls into shapely LineStrings ---
    wall_linestrings: Dict[str, LineString] = {}
    walls_elem = root.find("walls")
    if walls_elem is not None:
        for wall_elem in walls_elem.findall("wall"):
            wall_id = int(wall_elem.get("id"))
            points_elem = wall_elem.find("points")
            if points_elem is None:
                continue
            points = [
                (float(p.get("x")), float(p.get("y")))
                for p in points_elem.findall("point")
                if p.get("x") and p.get("y")
            ]
            if len(points) >= 2:
                wall_linestrings[wall_id] = LineString(points)

    # --- Helper: order points around centroid (fallback) ---
    def order_polygon(points: List[Tuple[float, float]]) -> Polygon:
        """
        Given a list of (x, y) coordinates, order them around the centroid.
        """
        arr = np.array(points)
        cx, cy = arr[:, 0].mean(), arr[:, 1].mean()
        angles = np.arctan2(arr[:, 1] - cy, arr[:, 0] - cx)
        ordered = arr[np.argsort(angles)]
        return Polygon(ordered)

    # --- Parse cells and reconstruct polygons ---
    records = []
    cells_elem = root.find("cells")
    if cells_elem is not None:
        for cell_elem in cells_elem.findall("cell"):
            cell_id = int(cell_elem.get("id"))
            group_id = int(cell_elem.get("group"))
            cell_type = group_map.get(group_id, f"unknown_group_{group_id}")

            # Gather walls forming the cell boundary
            cell_lines: List[LineString] = []
            cell_points: List[Tuple[float, float]] = []
            walls_ref_elem = cell_elem.find("walls")
            if walls_ref_elem is not None:
                for wall_ref in walls_ref_elem.findall("wall"):
                    wall_id = int(wall_ref.get("id"))
                    wall = wall_linestrings.get(wall_id)
                    if wall is not None:
                        cell_lines.append(wall)
                        cell_points.extend(list(wall.coords))

            # Try polygonize first
            cell_polygon = None
            if cell_lines:
                polygons = list(polygonize(cell_lines))
                if polygons:
                    cell_polygon = polygons[0]
                else:
                    print(f"Cell {cell_id} could not form a valid polygon. Fallback: use ordered centroid method")
                    cell_polygon = order_polygon(cell_points)
                    cell_type = "fallback"

            if cell_polygon is not None and not cell_polygon.is_empty:
                records.append({
                    "id_cell": int(cell_id),
                    "type": cell_type,
                    "geometry": cell_polygon
                })
            else:
                print(f"Cell {cell_id} has invalid geometry (empty or None)")

    # --- Create GeoDataFrame ---
    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    return gdf

def plot_root_section(root_gdf: gpd.GeoDataFrame):
    """Display the root section as polygons using GeoPandas and Matplotlib."""
    if root_gdf.empty:
        print("GeoDataFrame is empty, cannot plot.")
        return

    # GeoPandas handles the figure creation and geometry plotting
    fig, ax = plt.subplots(figsize=(8, 8))

    root_gdf.plot(
        ax=ax,
        column='type',           # Color polygons by the 'type' column
        cmap='viridis',          # Use a nice color map
        edgecolor='black',       # Outline the cells
        linewidth=0.5,           # Line width for the outline
        alpha=0.5,               # Transparency
        legend=True,             # Display the legend
        legend_kwds={'title': 'Cell Type', 'loc': 'best'}
    )
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Cross Section Preview")
    plt.tight_layout()
    plt.show()
    # plt.show() # Use this for local testing

def visualize(obj: Any,
              visu_type: str = 'polygon',
              **kwargs: Dict[str, Any]) -> None:
    """Visualize cellset data using the functions above."""

    if visu_type == 'polygon':
        _visualize_polygon(obj, **kwargs)
    elif visu_type == 'network':
        _visualize_network(obj, **kwargs)
    elif visu_type == 'paraview':
        _visualize_pv(obj, **kwargs)
    elif visu_type == 'water_potential':
        _visualize_water_potential(obj, **kwargs)
    else:
        raise ValueError(f"Unknown visualization type: {visu_type}")

def _visualize_polygon(
    obj: Any,
    **kwargs: Dict[str, Any]) -> None:
    """
    Visualize polygons from a GeoDataFrame or cellset data.

    Parameters
    ----------
    obj : Union[gpd.GeoDataFrame, Dict[str, Any]]
        GeoDataFrame or cellset data to visualize.
    **kwargs : Dict[str, Any]
        Additional keyword arguments for customizing the plot.
    """
    if isinstance(obj, dict):
        root_gdf = prep_section(obj)
        plot_root_section(root_gdf)
    elif isinstance(obj, gpd.GeoDataFrame):
        plot_root_section(obj)
    else:
        raise ValueError("Unsupported object type for polygon visualization.")


def _visualize_network(
    obj: Any,
    **kwargs: Dict[str, Any]) -> None:
    """
    Visualize network data.

    Parameters
    ----------
    obj : Any
        Network object to visualize.
    **kwargs : Dict[str, Any]
        Additional keyword arguments for customizing the plot.
    """
    if isinstance(obj, NetworkBuilder):
        graph = obj.graph
    elif isinstance(obj, nx.Graph):
        graph = obj
    else:
        raise ValueError("Unsupported object type for network visualization.")

    position = kwargs.get('position', nx.get_node_attributes(graph, 'position'))
    node_types = kwargs.get('node_types', nx.get_node_attributes(graph, 'type'))

    # Default color map
    default_color_map = {'apo': 'red', 'sym': 'yellow'}
    node_color_map = kwargs.get('node_color_map', default_color_map)

    default_edge_color_map = {'wall': 'purple', 'membrane': 'green', 'plasmodesmata': 'gray'}
    edge_color_map = kwargs.get('edge_color_map', default_edge_color_map)

    # Determine node colors
    node_colors = []
    for node in graph.nodes():
        node_type = node_types.get(node, 'sym')  # Default to 'sym' if type is not found
        node_colors.append(node_color_map.get(node_type, 'blue'))  # Default to 'blue' if color not found

        # Determine edge colors
    edge_colors = []
    for u, v, edge_attrs in graph.edges(data=True):
        edge_type = edge_attrs.get('path', 'wall')  # Default to 'wall' if path is not found
        edge_colors.append(edge_color_map.get(edge_type, 'purple'))

    # Draw the network
    ax = kwargs.get('ax')
    show_plot = False
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 10)))
        show_plot = True

    nx.draw(
        graph,
        position,
        ax=ax,
        node_color=node_colors,
        with_labels=kwargs.get('with_labels', False),
        node_size=kwargs.get('node_size', 10),
        edge_color=edge_colors,
        width=kwargs.get('width', 1),
        alpha=kwargs.get('alpha', 0.7)
    )

    ax.set_aspect("equal", "box")
    ax.set_title(kwargs.get('title', 'Network Visualization'))
    
    if show_plot:
        plt.tight_layout()
        plt.show()


def plot_water_potential_map(root_gdf: gpd.GeoDataFrame, title: str = "Water Potential"):
    """Display the root section with water potential colormap."""
    if root_gdf.empty:
        print("GeoDataFrame is empty, cannot plot.")
        return
    if 'water_potential' not in root_gdf.columns:
        print("water_potential column missing in GeoDataFrame")
        return
    if np.isnan(root_gdf['water_potential']).all():
        print("All water potential values are NaN, cannot plot.")
        root_gdf['water_potential'] = 0

    fig, ax = plt.subplots(figsize=(10, 8))
    root_gdf.plot(
        ax=ax,
        column='water_potential',
        cmap='viridis', 
        edgecolor='black',
        linewidth=0.5,
        legend=True,
        legend_kwds={'label': 'Water Potential (MPa)', 'orientation': 'vertical'}
    )
    ax.set_aspect("equal", "box")
    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    plt.tight_layout()
    plt.show()

def _visualize_water_potential(obj: Any, **kwargs: Dict[str, Any]) -> None:
    """
    Visualize water potential from a Mecha object.
    
    Parameters
    ----------
    obj : Any
        Mecha object containing results and cellset_data.
    **kwargs : Dict[str, Any]
        maturity_idx : int, default 0
        scenario_idx : int, default 0
    """
    if not hasattr(obj, 'results'):
        print("Object does not have results attribute.")
        return

    results = obj.results
    if not results:
        print("Results are empty.")
        return

    maturity_idx = kwargs.get('maturity_idx', 0)
    scenario_idx = kwargs.get('scenario_idx', 0)

    # Find the matching result
    target_res = None
    if isinstance(results, list):
        for res in results:
            if res.get('maturity stage') == maturity_idx and res.get('scenario') == scenario_idx:
                target_res = res
                break
    
    if target_res is None:
        print(f"No results found for maturity {maturity_idx} and scenario {scenario_idx}")
        print("Available results:", [(r.get('maturity stage'), r.get('scenario')) for r in results])
        return

    solution = target_res['solution']
    
    rhs = target_res.get('rhs')
        
    matrix_W = target_res.get('matrix_W')
    if matrix_W is not None:
         if hasattr(matrix_W, 'toarray') or hasattr(matrix_W, 'tocoo'):
             # Sparse matrix
             if hasattr(matrix_W, 'data'):
                 mat_data = matrix_W.data
                 print(f"DEBUG: MatrixW (sparse) stats - NaNs: {np.isnan(mat_data).sum()}, Min: {np.nanmin(mat_data)}, Max: {np.nanmax(mat_data)}")
         else:
             # Dense matrix
             print(f"DEBUG: MatrixW (dense) stats - NaNs: {np.isnan(matrix_W).sum()}, Min: {np.nanmin(matrix_W)}, Max: {np.nanmax(matrix_W)}")

    # Check for cellset_data
    if not hasattr(obj, 'cellset_data'):
        print("Object does not have cellset_data attribute.")
        return
        
    gdf = prep_section(obj.cellset_data)
    
    # Check for network offset
    if not hasattr(obj, 'network') or not hasattr(obj.network, 'n_wall_junction'):
         print("Object does not have valid network structure.")
         return

    offset = obj.network.n_walls
    offset += obj.network.n_wall_junction
    
    
    def get_pot(cid):
        idx = int(offset + cid)
        try:
             # solution is typically (n_nodes, 1) or (n_nodes,)
             val = solution[idx]
             if hasattr(val, '__getitem__') and hasattr(val, '__len__') and len(val) > 0:
                  return float(val[0])
             return float(val)
        except (IndexError, TypeError):
             print(f"DEBUG: Error retrieving pot for cid {cid} at idx {idx}")
             return np.nan
        print(f"DEBUG: Pot for cid {cid} at idx {idx}: {val}")

    gdf['water_potential'] = gdf['id_cell'].apply(get_pot)
    
    plot_water_potential_map(gdf, title=f"Water Potential (Mat: {maturity_idx}, Scen: {scenario_idx})")


def _visualize_standardized_results(obj: Any, **kwargs: Dict[str, Any]) -> None:
    """
    Visualize standardized results from a Mecha object.
    
    Parameters
    ----------
    obj : Any
        Mecha object containing results and cellset_data.
    """
    if not hasattr(obj, 'standardized_results'):
        print("Object does not have standardized_results attribute.")
        return

    results = obj.standardized_results[0]
    if not results:
        print("Results are empty.")
        return

    
    solution = results['solution']
    
    # Check for cellset_data
    if not hasattr(obj, 'cellset_data'):
        print("Object does not have cellset_data attribute.")
        return
        
    gdf = prep_section(obj.cellset_data)
    
    # Check for network offset
    if not hasattr(obj, 'network') or not hasattr(obj.network, 'n_wall_junction'):
         print("Object does not have valid network structure.")
         return

    def get_pot(cid):
        
        obj.network.graph.nodes[cid]['water_potential'] = solution[cid][0]
        
    for node, edges in obj.network.graph.adjacency() : #adjacency_iter returns an iterator of (node, adjacency dict) tuples for all nodes. This is the fastest way to look at every edge. For directed graphs, only outgoing adjacencies are included.
        i = obj.network.indice[node] #Node ID number
        if i<obj.network.n_walls: #wall ID 
            psi = solution[i][0]



    offset = obj.network.n_wall_junction
    offset += obj.network.n_walls
    
    

    gdf['water_potential'] = gdf['id_cell'].apply(get_pot)
    
    plot_water_potential_map(gdf, title=f"Water Potential (Mat: {maturity_idx}, Scen: {scenario_idx})")