# -*- coding: utf-8 -*-
#
#       File author(s):
#           Adrien Heymans
#
#       Copyright © by UCLouvain
#       Distributed under the LGPL License.
#       See accompanying file LICENSE.txt or copy at
#           https://www.gnu.org/licenses/lgpl-3.0.en.html
#
# -----------------------------------------------------------------------
"""
Visualization utilities for MECHA.
Includes polygon visualization, ParaView export, water potential mapping,
and network-level visualization functions.

Examples:
    >>> from openalea.mecha.utils.visu import visualize
    >>> from openalea.mecha import Mecha
    
    >>> # Basic visualization of the root organ section
    >>> visualize(mecha_instance, visu_type='polygon')
    
    >>> # Visualize water potential map
    >>> visualize(mecha_instance, visu_type='water_potential', 
    ...           maturity_idx=0, scenario_idx='standard water flow')
    
    >>> # Export to ParaView-readable files
    >>> visualize(mecha_instance, visu_type='paraview', 
    ...           prefix='my_simulation', export_cells=True, 
    ...           export_walls=False, export_plasmodesmata=True)
    
    >>> # Visualize flow pathways with custom colormap
    >>> visualize(mecha_instance, visu_type='flow_pathway', 
    ...           maturity_idx=0)
    
    >>> # Visualize xylem conductivity network
    >>> visualize(mecha_instance, visu_type='conductance', 
    ...           maturity_idx=0)
    
    >>> # Visualize water flow vectors
    >>> visualize(mecha_instance, visu_type='flow', maturity_idx=0)
    
    >>> # View radial profile of water potential
    >>> visualize(mecha_instance, visu_type='psi_profile', maturity_idx=0)
    
    >>> # Pass GeoDataFrame directly
    >>> gdf = mecha_instance.network._cells_gdf
    >>> visualize(gdf, visu_type='polygon')
    
"""

import geopandas as gpd
from shapely.ops import polygonize
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np

from shapely.geometry import LineString, Polygon
from typing import Tuple, Dict, List, Any, Optional
from openalea.mecha import Mecha, NetworkBuilder

import networkx as nx
import pandas as pd
# Network visualization functions
from openalea.mecha.utils.network_export import *
from openalea.mecha.utils.paraview_export import export_to_vtk


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

def plot_organ_section(organ_gdf: gpd.GeoDataFrame):
    """Display the organ section as polygons using GeoPandas and Matplotlib."""
    if organ_gdf.empty:
        print("GeoDataFrame is empty, cannot plot.")
        return

    # GeoPandas handles the figure creation and geometry plotting
    fig, ax = plt.subplots(figsize=(8, 8))

    organ_gdf.plot(
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
    ax.set_title("Organ Cross Section")
    plt.tight_layout()
    plt.show()
    # plt.show() # Use this for local testing

def visualize(obj: Any,
              visu_type: str = 'polygon',
              **kwargs: Dict[str, Any]) -> None:
    """Dispatch to the appropriate visualization function.

    Parameters
    ----------
    obj : Any
        The object to visualize (Mecha, NetworkBuilder, GeoDataFrame, …).
    visu_type : str
        One of:
        - ``'polygon'``        – cell cross-section polygons (matplotlib)
        - ``'network'``        – hydraulic graph (networkx/matplotlib)
        - ``'paraview'``       – export to ``.vtk`` files for ParaView
        - ``'water_potential'``– water-potential choropleth map
        - ``'conductance'``    – conductance K on graph edges (tri-panel)
        - ``'flow'``           – flow Q arrows on graph edges
        - ``'flow_pathway'``   – stacked area % by pathway vs. radius
        - ``'psi_profile'``    – Psi vs. radial distance profile
    **kwargs
        Forwarded verbatim to the selected function.  See each function's
        docstring for supported keyword arguments.
    """
    if visu_type == 'polygon':
        _visualize_polygon(obj, **kwargs)
    elif visu_type == 'network':
        visualize_network(obj, **kwargs)     
    elif visu_type == 'paraview':
        _visualize_pv(obj, **kwargs)
    elif visu_type == 'water_potential':
        _visualize_water_potential(obj, **kwargs)
    elif visu_type == 'conductance':
        plot_K_network(obj, **kwargs)
    elif visu_type == 'flow':
        plot_flow_network(obj, **kwargs)
    elif visu_type == 'flow_pathway':
        plot_flow_pathway_breakdown(obj, **kwargs)
    elif visu_type == 'psi_profile':
        plot_radial_profile(obj, **kwargs)
    elif visu_type == 'velocity':
        plot_velocity_network(obj, **kwargs)
    else:
        raise ValueError(
            f"Unknown visualization type: '{visu_type}'. "
            "Choose from: 'polygon', 'network', 'paraview', 'water_potential', "
            "'conductance', 'flow', 'flow_pathway', 'psi_profile', 'velocity'."
        )

def _visualize_pv(
    obj: Any,
    **kwargs: Dict[str, Any]) -> None:
    """Export to ParaView-readable VTK files via :func:`export_to_vtk`.

    Parameters
    ----------
    obj : Mecha
        A solved ``Mecha`` instance.
    prefix : str, optional
        File path prefix for generated ``.vtk`` files (default ``'mecha_pv'``).
    maturity_idx : int, optional
        Maturity stage index to export (default 0).
    scenario_idx : str, optional
        Scenario name to export (default ``'standard water flow'``).
    extrude_z : float, optional
        Z-extrusion depth in µm for 2-D geometry (default 5.0).
    pd_radius : float, optional
        Plasmodesmata cylinder radius in µm (default 0.05).
    export_cells, export_walls, export_membranes,
    export_plasmodesmata, export_flow_vectors : bool, optional
        Toggle individual output files (all True by default).
    """
    if not isinstance(obj, Mecha):
        raise ValueError(
            "visu_type='paraview' requires a Mecha instance. "
            f"Got {type(obj).__name__}."
        )
    prefix = kwargs.pop('prefix', 'mecha_pv')
    export_to_vtk(obj, prefix=prefix, **kwargs)


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
        cells_gdf = prep_section(obj)
        plot_organ_section(cells_gdf)
    elif isinstance(obj, gpd.GeoDataFrame):
        plot_organ_section(obj)
    elif isinstance(obj, Mecha):
        if obj.network is None:
            raise ValueError("Mecha object has no network.")
        if obj.network._cells_gdf is None:
            raise ValueError("Mecha object has no _cells_gdf.")
        cells_gdf = obj.network._cells_gdf
        plot_organ_section(cells_gdf)
    else:
        raise ValueError("Unsupported object type for polygon visualization.")




def plot_water_potential_map(organ_gdf: gpd.GeoDataFrame, title: str = "Water Potential"):
    """Display the root section with water potential colormap."""
    if organ_gdf.empty:
        print("GeoDataFrame is empty, cannot plot.")
        return
    if 'water_potential' not in organ_gdf.columns:
        print("water_potential column missing in GeoDataFrame")
        return
    if np.isnan(organ_gdf['water_potential']).all():
        print("All water potential values are NaN, cannot plot.")
        organ_gdf['water_potential'] = 0

    fig, ax = plt.subplots(figsize=(10, 8))
    organ_gdf.plot(
        ax=ax,
        column='water_potential',
        cmap='viridis', 
        edgecolor='black',
        linewidth=0.5,
        legend=True,
        legend_kwds={'label': 'Water Potential (hPa)', 'orientation': 'vertical'}
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
        results list: list of solutions, each solution is a numpy array of water potentials for each cell. It's obtained by the 
        solve_W() or water_flow() method.

    """
    # Use pre-computed _cells_gdf if available, else fall back to prep_section
    if hasattr(obj.network, '_cells_gdf') and obj.network._cells_gdf is not None:
        gdf = obj.network._cells_gdf.copy()
    else:
        gdf = prep_section(obj.cellset_data)
    
    # Check for network and indices
    if not hasattr(obj, 'network') or not hasattr(obj, 'indice'):
         print("Object does not have valid network structure or indice mapping.")
         return

    nwj = obj.network.n_wall_junction
    # Support results list or direct solution
    if hasattr(obj, 'results') and obj.results:
        standardized_results = kwargs.get('standardized_results', True)
        maturity_idx = kwargs.get('maturity_idx', 0)
        scenario_idx = kwargs.get('scenario_idx', "standard water flow")
        target_res = None
        for res in obj.results:
            if res.get('maturity stage') == maturity_idx and res.get('scenario') == scenario_idx:
                target_res = res
                break
        if target_res:
            sol = np.asarray(target_res['solution']).ravel()
        else:
            print(f"No results found for maturity {maturity_idx} and scenario {scenario_idx}")
            return
    else:
        sol = np.asarray(obj.solution).ravel()
    
    def get_pot(cid):
        """Map cell ID to water potential using the network indice mapping."""
        node_id = nwj + cid
        try:
            idx = obj.indice[node_id]
            return float(sol[idx])
        except (KeyError, IndexError):
             return np.nan

    gdf['water_potential'] = gdf['id_cell'].apply(get_pot)
    
    plot_water_potential_map(gdf, title=kwargs.get('title', "Water Potential Map"))
