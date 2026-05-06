import geopandas as gpd
from shapely.ops import polygonize
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
from shapely.geometry import LineString, Polygon
from typing import Tuple, Dict, List, Any, Optional
from mecha.utils.network_builder import NetworkBuilder
from mecha.mecha_class import Mecha
import networkx as nx
import pandas as pd

_PATH_COLORS = {
    'wall':          '#e07b4a',   # warm orange
    'membrane':      '#4ab5e0',   # sky blue
    'plasmodesmata': '#7be04a',   # lime green
}
_PATH_LABELS = {
    'wall':          'Apoplastic (cell wall)',
    'membrane':      'Transcellular (cell membrane)',
    'plasmodesmata': 'Symplastic (plasmodesmata)',
}


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
    """Visualize cellset data using the functions above."""

    if visu_type == 'polygon':
        _visualize_polygon(obj, **kwargs)
    elif visu_type == 'network':
        _visualize_network(obj, **kwargs)
    elif visu_type == 'paraview':
        _visualize_pv(obj, **kwargs)
    elif visu_type == 'water_potential':
        _visualize_water_potential(obj, **kwargs)
    elif visu_type == 'conductance':
        _plot_K_network(obj, **kwargs)
    elif visu_type == 'flow':
        _plot_flow_network(obj, **kwargs)
    elif visu_type == 'flow_pathway':
        _plot_flow_pathway_breakdown(obj, **kwargs)
    elif visu_type == 'psi_profile':
        _plot_psi_radial_profile(obj, **kwargs)
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
    elif isinstance(obj, Mecha):
        graph = obj.network.graph
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
    
    plot_water_potential_map(gdf, title=kwargs.get('title', "Water Potential"))

def _get_result_data(obj: Any, **kwargs: Any) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """Helper to extract solution and result dict from Mecha object."""
    maturity_idx = kwargs.get('maturity_idx', 0)
    scenario_idx = kwargs.get('scenario_idx', 'standard water flow')
    
    if hasattr(obj, 'results') and obj.results:
        for res in obj.results:
            if res.get('maturity stage') == maturity_idx and res.get('scenario') == scenario_idx:
                return np.asarray(res['solution']).ravel(), res
    
    if hasattr(obj, 'solution'):
        return np.asarray(obj.solution).ravel(), None
        
    return None, None

def _plot_flow_network(obj, **kwargs):
    """Draw arrows on edges proportional to |Q|, coloured by edge type.

    Arrow head direction follows sign of Q (u→v if Q>0, v→u if Q<0).
    """
    sol, res = _get_result_data(obj, **kwargs)
    if sol is not None:
        # Update flows in graph if we have a solution
        indices = getattr(obj, 'indice', nx.get_node_attributes(obj.network.graph, 'indice'))
        for u, v, eattr in obj.network.graph.edges(data=True):
            K = eattr.get('K')
            Q = eattr.get('Q', None)
            if Q is None and K is not None:
                psi_u = float(sol[indices[u]])
                psi_v = float(sol[indices[v]])
                eattr['Q'] = K * (psi_u - psi_v)

    graph  = obj.network.graph
    pos    = nx.get_node_attributes(graph, 'position')

    save_path = kwargs.get('save_path', None)

    fig, ax = plt.subplots(figsize=(12, 12), facecolor='#0d0d1a')
    ax.set_facecolor('#0d0d1a')
    ax.set_aspect('equal')
    ax.set_title(f"Water Flow Q on network edges (Mat {kwargs.get('maturity_idx', 0)})", color='white', fontsize=13)

    # Background: all nodes
    nx.draw_networkx_nodes(graph, pos, ax=ax,
                           node_size=2, node_color='#333355', alpha=0.3)

    # Gather all Q values for normalisation
    all_Q = [abs(d.get('Q', 0.0))
             for _, _, d in graph.edges(data=True)
             if d.get('Q') is not None]
    q_max = max(all_Q) if all_Q else 1.0

    from matplotlib.patches import FancyArrowPatch

    legend_handles = []
    for path_type, color in _PATH_COLORS.items():
        drawn = False
        for u, v, eattr in graph.edges(data=True):
            if eattr.get('path') != path_type:
                continue
            Q = eattr.get('Q')
            K = eattr.get('K')
            if Q is None or K is None or K == 0:
                continue

            pu = pos.get(u, (0, 0))
            pv = pos.get(v, (0, 0))

            # Arrow direction follows sign of Q
            if Q >= 0:
                src, dst = pu, pv
            else:
                src, dst = pv, pu

            magnitude = abs(Q) / q_max       # 0…1
            if magnitude < 1e-12:
                continue

            lw  = 0.2 + 4.0 * magnitude
            hw  = 0.4 + 5.0 * magnitude      # head width
            alpha = 0.35 + 0.55 * magnitude  # more opaque for stronger flow

            ax.annotate(
                "", xy=dst, xytext=src,
                arrowprops=dict(
                    arrowstyle=f"->,head_width={hw:.2f},head_length={hw*0.8:.2f}",
                    color=color,
                    lw=lw,
                    alpha=alpha,
                    connectionstyle="arc3,rad=0.0"
                )
            )
            if not drawn:
                drawn = True

        # Legend proxy
        from matplotlib.lines import Line2D
        legend_handles.append(
            Line2D([0], [0], color=color, lw=2, label=_PATH_LABELS[path_type])
        )

    ax.legend(handles=legend_handles, loc='upper right',
              facecolor='#222244', labelcolor='white', edgecolor='white',
              fontsize=8)
    plt.tight_layout()
    if save_path is not None:
        out = save_path
        plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
        print(f"Saved → {out}")
    else:
        plt.show()

    
def _plot_K_network(obj, **kwargs):
    """Draw the graph with edges coloured and scaled by conductance K.

    Three subplots: one per path type (wall / membrane / PD).
    """
    if hasattr(obj, 'network'):
        graph  = obj.network.graph
    elif hasattr(obj, 'graph'):
        graph  = obj.graph
    else:
        raise ValueError("Object must have a network or graph.")

    pos    = nx.get_node_attributes(graph, 'position')

    save_path = kwargs.get('save_path', None)
    maturity_idx = kwargs.get('maturity_idx', 0)
    
    # If obj is Mecha, we might need to re-run build_matrices to get the right K for the maturity
    if hasattr(obj, 'build_matrices') and maturity_idx != getattr(obj, '_current_visu_maturity', -1):
        print(f"Re-building matrices for maturity {maturity_idx} to update graph K attributes...")
        obj.build_matrices(h=kwargs.get('h', 0), i_maturity=maturity_idx)
        obj._current_visu_maturity = maturity_idx

    title = kwargs.get('title', f"Tri-pathways Hydraulic Conductance (K) - Mat {maturity_idx}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='#111111')
    fig.suptitle(title, color='white', fontsize=14, y=1.01)

    for ax, path_type in zip(axes, ['wall', 'membrane', 'plasmodesmata']):
        ax.set_facecolor('#1a1a2e')
        ax.set_aspect('equal')

        # Draw background nodes (small, grey)
        nx.draw_networkx_nodes(
            graph, pos, ax=ax,
            node_size=2, node_color='#444466', alpha=0.4
        )

        # Collect edges of this type
        edge_list, K_vals = [], []
        for u, v, eattr in graph.edges(data=True):
            if eattr.get('path') == path_type:
                K = eattr.get('K')
                if K is not None and K > 0:
                    edge_list.append((u, v))
                    K_vals.append(K)

        if not edge_list:
            ax.set_title(f"{_PATH_LABELS[path_type]}\n(no edges)", color='white')
            continue

        K_arr  = np.array(K_vals)
        log_K  = np.log10(K_arr + 1e-30)
        log_K -= log_K.min()
        if log_K.max() > 0:
            log_K /= log_K.max()   # normalise 0→1 for width

        widths = 0.3 + 4.5 * log_K

        # Colour map: low K = dark, high K = bright
        cmap   = cm.get_cmap('plasma')
        colors = [cmap(v) for v in log_K]

        nx.draw_networkx_edges(
            graph, pos, ax=ax,
            edgelist=edge_list,
            edge_color=colors,
            width=widths,
            alpha=0.85
        )

        # Colourbar
        sm = cm.ScalarMappable(
            cmap='plasma',
            norm=mcolors.LogNorm(vmin=max(K_arr.min(), 1e-30), vmax=K_arr.max())
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label('K  [cm³ hPa⁻¹ d⁻¹]', color='white', fontsize=8)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

        ax.set_title(
            f"{_PATH_LABELS[path_type]}\n"
            f"n={len(edge_list)}  K∈[{K_arr.min():.2e}, {K_arr.max():.2e}]",
            color='white', fontsize=9
        )

    plt.tight_layout()

    if save_path is not None:
        out = save_path
        plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#111111')
        print(f"Saved → {out}")
    else:
        plt.show()



def plot_matrix_difference(m1, m2, title="Absolute Difference between Matrices", threshold=1e-12):
    """
    Plots the absolute difference between two matrices.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert sparse matrices to dense if needed
    if hasattr(m1, 'toarray'): m1 = m1.toarray()
    if hasattr(m2, 'toarray'): m2 = m2.toarray()

    diff = np.abs(m1 - m2)

    rows, cols = np.where(diff > threshold)
    values = diff[rows, cols]

    plt.figure(figsize=(10, 8))
    if len(values) > 0:
        plt.scatter(cols, rows, c=values, cmap='hot', s=50, alpha=0.8, vmin=np.min(values), vmax=np.max(values))
        plt.colorbar(label='Absolute Difference')
    else:
        plt.text(0.5, 0.5, 'No differences found', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    
    plt.title(title)
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    if hasattr(m1, 'shape'):
        plt.xlim(-0.5, m1.shape[1]-0.5)
        plt.ylim(m1.shape[0]-0.5, -0.5) # Invert y axis
    plt.tight_layout()
    plt.show()


def plot_edge_and_node_differences(net1, net2, title="Detailed Topology Differences", distance_threshold=1e-6):
    """
    Locates and plots differences between two networks: Code colored by type.
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(14, 14))
    g1 = net1.graph if hasattr(net1, 'graph') else net1
    g2 = net2.graph if hasattr(net2, 'graph') else net2
    
    pos1 = nx.get_node_attributes(g1, 'position')
    pos2 = nx.get_node_attributes(g2, 'position')

    # Defaults for positions
    default_pos1 = {n: (0,0) for n in g1.nodes if n not in pos1}
    pos1.update(default_pos1)
    default_pos2 = {n: (0,0) for n in g2.nodes if n not in pos2}
    pos2.update(default_pos2)

    nodes1 = set(g1.nodes())
    nodes2 = set(g2.nodes())

    missing_in_2 = list(nodes1 - nodes2)
    missing_in_1 = list(nodes2 - nodes1)

    common_nodes = nodes1.intersection(nodes2)
    moved_nodes = []
    stable_nodes = []
    
    for n in common_nodes:
        p1, p2 = pos1[n], pos2[n]
        d = np.hypot(p1[0]-p2[0], p1[1]-p2[1])
        if d > distance_threshold:
            moved_nodes.append(n)
        else:
            stable_nodes.append(n)

    # Convert edges to canonical forms (min, max) so direction doesn't matter
    edges1 = set([tuple(sorted((u, v))) for u, v in g1.edges()])
    edges2 = set([tuple(sorted((u, v))) for u, v in g2.edges()])

    missing_edges_in_2 = list(edges1 - edges2)
    missing_edges_in_1 = list(edges2 - edges1)
    common_edges = list(edges1.intersection(edges2))

    # Base plot (common stuff in light gray to provide context)
    nx.draw_networkx_nodes(g1, pos1, nodelist=stable_nodes, node_color='lightgray', node_size=10, ax=ax, alpha=0.5)
    nx.draw_networkx_edges(g1, pos1, edgelist=common_edges, edge_color='lightgray', width=1, ax=ax, alpha=0.5)

    # Draw missing nodes in 2 (i.e., only in 1) -> Red
    nx.draw_networkx_nodes(g1, pos1, nodelist=missing_in_2, node_color='red', node_size=30, ax=ax, label='Node only in net1')
    
    # Draw missing nodes in 1 (i.e., only in 2) -> Blue
    nx.draw_networkx_nodes(g2, pos2, nodelist=missing_in_1, node_color='blue', node_size=30, ax=ax, label='Node only in net2')

    # Draw moved nodes 
    # Draw them in Net 1 position (Orange) and Net 2 position (Purple) with an arrow between them
    nx.draw_networkx_nodes(g1, pos1, nodelist=moved_nodes, node_color='orange', node_size=50, ax=ax, label='Moved node (Net1 pos)')
    nx.draw_networkx_nodes(g2, pos2, nodelist=moved_nodes, node_color='purple', node_size=20, ax=ax, label='Moved node (Net2 pos)')
    for n in moved_nodes:
        ax.annotate("", xy=pos2[n], xycoords='data', xytext=pos1[n], textcoords='data',
                    arrowprops=dict(arrowstyle="->", color="black", shrinkA=0, shrinkB=0, alpha=0.5))

    # Draw missing edges in 2 (i.e., only in net 1) -> Red edges
    # Be careful some nodes might not exist in pos2 if they are only in 1, so use pos1
    nx.draw_networkx_edges(g1, pos1, edgelist=missing_edges_in_2, edge_color='red', width=2, ax=ax, label='Edge only in net1')
    
    # Draw missing edges in 1 (i.e., only in net 2) -> Blue edges
    nx.draw_networkx_edges(g2, pos2, edgelist=missing_edges_in_1, edge_color='blue', width=2, ax=ax, label='Edge only in net2')

    ax.set_aspect("equal", "box")
    ax.set_title(title)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Node only in net1', markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Node only in net2', markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Moved node (Net1 pos)', markerfacecolor='orange', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Moved node (Net2 pos)', markerfacecolor='purple', markersize=10),
        Line2D([0], [0], color='red', lw=2, label='Edge only in net1'),
        Line2D([0], [0], color='blue', lw=2, label='Edge only in net2'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()

def _plot_network_property(network, prop_name, ax, title, node_size=30):
    import networkx as nx
    from matplotlib import colormaps
    graph = network.graph if hasattr(network, 'graph') else network
    
    position = nx.get_node_attributes(graph, 'position')
    default_pos = {n: (0,0) for n in graph.nodes if n not in position}
    position.update(default_pos)
    
    props = nx.get_node_attributes(graph, prop_name)
    has_props_in_graph = len(props) > 0
    
    if not has_props_in_graph and hasattr(network, 'cell_manager'):
        props = {}
        for c in network.cell_manager:
            props[c.node_id] = getattr(c, prop_name, -1)
            
    # Extract unique properties
    prop_values = list(props.values())
    unique_props = list(set(prop_values))
    cmap = colormaps.get_cmap('tab20')
    color_map = {val: cmap(i / len(unique_props)) if len(unique_props) > 0 else 'gray' for i, val in enumerate(unique_props)}
    
    node_colors = []
    for node in graph.nodes():
        val = props.get(node, None) # None for non-cell nodes
        if val is not None:
            node_colors.append(color_map[val])
        else:
            node_colors.append('lightgray') # Juncs or walls
            
    nx.draw(
        graph,
        position,
        ax=ax,
        node_color=node_colors,
        node_size=node_size,
        edge_color='black',
        alpha=0.7
    )
    ax.set_aspect("equal", "box")
    ax.set_title(title)


def plot_networks_interC(net1, net2, title1="Network 1", title2="Network 2"):
    """
    Plots two networks side by side colored by cgroup.
    """
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    _plot_network_property(net1, 'count_interC', ax=ax1, title=title1 + ' interC')
    _plot_network_property(net2, 'count_interC', ax=ax2, title=title2 + ' interC')
    plt.tight_layout()
    plt.show()

def plot_networks_cgroup(net1, net2, title1="Network 1", title2="Network 2"):
    """
    Plots two networks side by side colored by cgroup.
    """
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    _plot_network_property(net1, 'cgroup', ax=ax1, title=title1 + ' (cgroup)')
    _plot_network_property(net2, 'cgroup', ax=ax2, title=title2 + ' (cgroup)')
    plt.tight_layout()
    plt.show()


def plot_networks_rank(net1, net2, title1="Network 1", title2="Network 2"):
    """
    Plots two networks side by side colored by rank.
    """
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    _plot_network_property(net1, 'rank', ax=ax1, title=title1 + ' (rank)', node_size=10)
    _plot_network_property(net2, 'rank', ax=ax2, title=title2 + ' (rank)', node_size=10)
    plt.tight_layout()
    plt.show()


def plot_network_difference(net1, net2, title="Network Spatial Difference"):
    """
    Plots two networks overplotted in different colors to highlight spatial and topological differences.
    Red nodes/edges belong to Net1, Blue to Net2. Overlap appears as a mix.
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    
    fig, ax = plt.subplots(figsize=(12, 12))
    g1 = net1.graph if hasattr(net1, 'graph') else net1
    g2 = net2.graph if hasattr(net2, 'graph') else net2
    
    pos1 = nx.get_node_attributes(g1, 'position')
    pos2 = nx.get_node_attributes(g2, 'position')
    
    # Plot net1 elements in red (alpha=0.5)
    nx.draw(g1, pos1, ax=ax, node_color='red', edge_color='red', node_size=15, width=2, alpha=0.5, label='Net1')
    
    # Plot net2 elements in blue (alpha=0.5)
    nx.draw(g2, pos2, ax=ax, node_color='blue', edge_color='blue', node_size=10, width=1, alpha=0.5, label='Net2')
    
    ax.set_aspect("equal", "box")
    ax.set_title(title)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Net1', markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Net2', markerfacecolor='blue', markersize=10)
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()

def plot_intercellular_spaces(net1, net2, title1="Net 1", title2="Net 2"):
    """
    Plots two networks side by side and highlights the intercellular spaces.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)

    def _plot_intercellular(network, ax, title):
        graph = network.graph if hasattr(network, 'graph') else network
        pos = nx.get_node_attributes(graph, 'position')
        
        # Draw everything in light gray
        nx.draw_networkx_nodes(graph, pos, node_color='lightgray', node_size=10, ax=ax, alpha=0.5)
        nx.draw_networkx_edges(graph, pos, edge_color='lightgray', width=1, ax=ax, alpha=0.5)
        
        # Highlight intercellular nodes
        if hasattr(network, 'intercellular_cells'):
            intercellular_nodes = [network.n_wall_junction + cid for cid in network.intercellular_cells]
        else:
            intercellular_nodes = []
            
        nx.draw_networkx_nodes(graph, pos, nodelist=intercellular_nodes, node_color='blue', node_size=50, ax=ax, label='Intercellular Space')
        
        ax.set_aspect("equal", "box")
        ax.set_title(title)
        
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Intercellular Space', markerfacecolor='blue', markersize=10),
        ]
        if intercellular_nodes:
            ax.legend(handles=legend_elements, loc='upper right')

    _plot_intercellular(net1, ax1, title1)
    _plot_intercellular(net2, ax2, title2)
    
    plt.tight_layout()
    plt.show()

def _plot_psi_radial_profile(obj, **kwargs):
    """Plot average Psi vs radial distance (rank)."""
    sol, _ = _get_result_data(obj, **kwargs)
    if sol is None:
        print("No solution found for specified maturity/scenario.")
        return

    cm = obj.network.cell_manager
    data = []
    
    # 1) Collect node potentials and assign to 'out', 'mid', or 'in' for each cell rank
    for cell in cm:
        rc = np.hypot(cell.x, cell.y)
        if rc < 1e-6:
            continue
            
        psi_c = sol[cell.node_id]
        data.append({'rank': cell.rank, 'side': 'mid', 'r': rc, 'psi': psi_c})
        
        vc_x, vc_y = cell.x, cell.y
        norm_vc = rc
        
        for w in cell.walls:
            rw = np.hypot(w.x, w.y)
            vw_x, vw_y = w.x - cell.x, w.y - cell.y
            norm_vw = np.hypot(vw_x, vw_y)
            if norm_vw < 1e-6:
                continue
                
            # Angle between vector-to-cell-center and cell-to-wall vector
            cos_theta = (vc_x * vw_x + vc_y * vw_y) / (norm_vc * norm_vw)
            psi_w = sol[w.node_id]
            
            if cos_theta > 0.4:
                data.append({'rank': cell.rank, 'side': 'out', 'r': rw, 'psi': psi_w})
            elif cos_theta < -0.4:
                data.append({'rank': cell.rank, 'side': 'in', 'r': rw, 'psi': psi_w})
    
    df = pd.DataFrame(data)
    df = df.dropna(subset=['psi'])
    
    # Group by rank and side to get the 3 mean radial distances and potentials per row
    profile = df.groupby(['rank', 'side']).agg({'r': 'mean', 'psi': ['mean', 'std']}).reset_index()
    profile.columns = ['rank', 'side', 'r', 'psi_mean', 'psi_std']
    
    # Separate Symplastic (mid/cell nodes) and Apoplastic (out/in/wall nodes)
    sym_df = profile[profile['side'] == 'mid'].sort_values('r')
    apo_df = profile[profile['side'].isin(['out', 'in'])].sort_values('r')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Symplastic
    ax.errorbar(sym_df['r'], sym_df['psi_mean'], yerr=sym_df['psi_std'], fmt='-o', 
                capsize=3, color='#7be04a', ecolor='#7be04a', alpha=0.8, label='Symplastic')
                
    # Plot Apoplastic
    ax.errorbar(apo_df['r'], apo_df['psi_mean'], yerr=apo_df['psi_std'], fmt='-s', 
                capsize=3, color='#e07b4a', ecolor='#e07b4a', alpha=0.8, label='Apoplastic')
    
    ax.set_xlabel('Radial Distance (µm)')
    ax.set_ylabel('Water Potential (hPa)')
    ax.set_title(f"Radial Psi Profile - Mat {kwargs.get('maturity_idx', 0)}, Scen {kwargs.get('scenario_idx', 'standard')}")
    ax.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def _plot_flow_pathway_breakdown(obj, **kwargs):
    """Stacked area plot of flow through pathways vs radial distance."""
    sol, res = _get_result_data(obj, **kwargs)
    if sol is None:
        print("No solution found for specified maturity/scenario.")
        return

    graph = obj.network.graph
    indices = getattr(obj, 'indice', nx.get_node_attributes(graph, 'indice'))
    pos = nx.get_node_attributes(graph, 'position')
    cm = obj.network.cell_manager
    
    # 1) Get the discrete set of 3 mean radial distances per rank
    data = []
    for cell in cm:
        rc = np.hypot(cell.x, cell.y)
        if rc < 1e-6: continue
        data.append({'rank': cell.rank, 'side': 'mid', 'r': rc})
        vc_x, vc_y = cell.x, cell.y
        norm_vc = rc
        for w in cell.walls:
            rw = np.hypot(w.x, w.y)
            vw_x, vw_y = w.x - cell.x, w.y - cell.y
            norm_vw = np.hypot(vw_x, vw_y)
            if norm_vw < 1e-6: continue
            cos_theta = (vc_x * vw_x + vc_y * vw_y) / (norm_vc * norm_vw)
            if cos_theta > 0.4:
                data.append({'rank': cell.rank, 'side': 'out', 'r': rw})
            elif cos_theta < -0.4:
                data.append({'rank': cell.rank, 'side': 'in', 'r': rw})
                
    df_r = pd.DataFrame(data).groupby(['rank', 'side']).mean().reset_index()
    R_list = np.sort(df_r['r'].unique())
    if len(R_list) == 0:
        print("No radial distances calculated.")
        return
    
    flow_data = []
    for u, v, eattr in graph.edges(data=True):
        K = eattr.get('K')
        path = eattr.get('path', 'wall')
        if K is None or path not in _PATH_COLORS:
            continue
            
        psi_u = sol[indices[u]]
        psi_v = sol[indices[v]]
        Q = K * (psi_u - psi_v)
        
        pu = pos.get(u, (0,0))
        pv = pos.get(v, (0,0))
        
        v_edge_x, v_edge_y = pv[0] - pu[0], pv[1] - pu[1]
        mid_x, mid_y = (pu[0] + pv[0]) / 2, (pu[1] + pv[1]) / 2
        norm_mid = np.hypot(mid_x, mid_y)
        norm_edge = np.hypot(v_edge_x, v_edge_y)
        
        # Calculate radial component of flow Q
        if norm_mid < 1e-6 or norm_edge < 1e-6:
            Q_rad = abs(Q)
        else:
            rad_x, rad_y = mid_x / norm_mid, mid_y / norm_mid
            dr = v_edge_x * rad_x + v_edge_y * rad_y
            Q_rad = abs(Q) * abs(dr) / norm_edge
            
        # Assign to nearest discrete radial distance
        closest_R = R_list[np.argmin(np.abs(R_list - norm_mid))]
        flow_data.append({'r': closest_R, 'path': path, 'Q_rad': Q_rad})

    df = pd.DataFrame(flow_data)
    if df.empty:
        print("No flow data available.")
        return
        
    # Pivot: discrete r as index, paths as columns, sum of Q_rad as values
    pivot_df = df.pivot_table(index='r', columns='path', values='Q_rad', aggfunc='sum').fillna(0)
    
    # Ensure all paths exist in columns
    for p in _PATH_COLORS:
        if p not in pivot_df.columns:
            pivot_df[p] = 0.0
            
    # Sort columns to match _PATH_COLORS order for consistent coloring
    pivot_df = pivot_df[['wall', 'plasmodesmata', 'membrane']]
    
    # Convert to percentages
    row_sums = pivot_df.sum(axis=1)
    row_sums[row_sums == 0] = 1.0 # Avoid division by zero
    pivot_pct = pivot_df.div(row_sums, axis=0) * 100

    # Rename columns for the automatic legend and map colors
    colors = [_PATH_COLORS[c] for c in pivot_pct.columns]
    pivot_pct.rename(columns=_PATH_LABELS, inplace=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot_pct.plot.area(ax=ax, color=colors, alpha=0.7)
    
    ax.set_xlabel('Radial Distance (µm)')
    ax.set_ylabel('Pathway Contribution (%)')
    ax.set_title(f"Flow Pathway Breakdown - Mat {kwargs.get('maturity_idx', 0)}")
    ax.set_ylim(0, 100)
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.show()

def _gather_edge_data(graph):
    """Return a dict of lists: {path_type: {'K': [...], 'Q': [...], 'pos': [...]}}."""
    data = {k: {'K': [], 'Q': [], 'xy_mid': []} for k in _PATH_COLORS}
    pos = nx.get_node_attributes(graph, 'position')

    for u, v, eattr in graph.edges(data=True):
        path = eattr.get('path', 'wall')
        if path not in data:
            continue
        K = eattr.get('K')
        Q = eattr.get('Q')
        if K is None:
            continue
        pu = pos.get(u, (0, 0))
        pv = pos.get(v, (0, 0))
        mid = ((pu[0] + pv[0]) / 2, (pu[1] + pv[1]) / 2)
        data[path]['K'].append(K)
        data[path]['Q'].append(Q if Q is not None else 0.0)
        data[path]['xy_mid'].append(mid)
    return data