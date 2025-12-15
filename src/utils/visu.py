import geopandas as gpd
from shapely.ops import polygonize
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Polygon
from typing import Tuple, Dict, List, Any
from src.utils.network_builder import NetworkBuilder
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

    # Determine node colors
    node_colors = []
    for node in graph.nodes():
        node_type = node_types.get(node, 'sym')  # Default to 'sym' if type is not found
        node_colors.append(node_color_map.get(node_type, 'blue'))  # Default to 'blue' if color not found

    # Draw the network
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 10)))

    nx.draw(
        graph,
        position,
        ax=ax,
        node_color=node_colors,
        with_labels=kwargs.get('with_labels', False),
        node_size=kwargs.get('node_size', 10),
        edge_color=kwargs.get('edge_color', 'gray'),
        width=kwargs.get('width', 1),
        alpha=kwargs.get('alpha', 0.7)
    )

    ax.set_title(kwargs.get('title', 'Network Visualization'))
    plt.tight_layout()
    plt.show()

