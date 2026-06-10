#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#       mecha.utils.network_export
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
Network export and visualization for MECHA Hydraulic Networks
============================================================

Contains functions to visualize and export MECHA networks.
Can export to graphml for use in external tools like Gephi.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from mecha.utils.network_builder import NetworkBuilder

# Constants moved from visu.py
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
_CONTINUOUS_PROPS = ['psi', 'psi_p', 'psi_os', 'psi_total', 'length', 'wall_thickness', 'Q_in', 'Q_out', 'Q', 'A', 'velocity', 'rank'] 

def export_to_graphml(obj: Any, filepath: str) -> None:
    """
    Export the hydraulic network to a .graphml file.
    
    Parameters
    ----------
    obj : Union[Mecha, NetworkBuilder, nx.Graph]
        The object containing the graph to export.
    filepath : str
        Target file path for the .graphml file.
    """
    if hasattr(obj, 'network') and obj.network is not None:
        graph = obj.network.graph
    elif hasattr(obj, 'graph'):
        graph = obj.graph
    elif isinstance(obj, nx.Graph):
        graph = obj
    else:
        raise ValueError("Unsupported object type for graphml export.")

    # GraphML doesn't support complex attributes like tuples or objects directly easily
    # We should ensure positions and other attributes are converted to simple types if needed
    G = graph.copy()
    position = nx.get_node_attributes(G, 'position')
    
    # Convert 'position' tuple (x, y) to 'x' and 'y' attributes for GraphML
    def convert_node_position(G: nx.Graph, position: Dict[str, Tuple[float, float]]) -> nx.Graph:
        for node in G.nodes():
            x, y = position.get(node, (0, 0))

            G.nodes[node]["x"] = float(x)
            G.nodes[node]["y"] = float(y)
            G.nodes[node]["z"] = 0.0
        return G
            
    def sanitize(G: nx.Graph) -> nx.Graph:
        for _, attrs in G.nodes(data=True):
            for k, v in list(attrs.items()):
                if not isinstance(v, (str, int, float, bool)):
                    attrs[k] = str(v)

        for _, _, attrs in G.edges(data=True):
            for k, v in list(attrs.items()):
                if not isinstance(v, (str, int, float, bool)):
                    attrs[k] = str(v)
        return G

    G = convert_node_position(G, position)
    G = sanitize(G)
    if not filepath.endswith(".graphml"):
        filepath += ".graphml"
    if G == None:
        raise ValueError("Graph is empty, cannot export to graphml.")
    if G.nodes() == 0:
        raise ValueError("Graph is empty, cannot export to graphml.")
    else:
        nx.write_graphml(G, filepath)
        print(f"[network_export] Exported graph to {filepath}")


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


def visualize_network(
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
    # Import Mecha locally to avoid circular dependencies if needed
    from mecha.mecha_class import Mecha

    if isinstance(obj, NetworkBuilder):
        graph = obj.graph
        network = obj
    elif isinstance(obj, nx.Graph):
        graph = obj
        network = None
    elif isinstance(obj, Mecha):
        _get_result_data(obj, **kwargs)
        graph = obj.network.graph
        network = obj.network
    else:
        raise ValueError("Unsupported object type for network visualization.")

    if network is not None and 'prop_name' in kwargs:
        prop_name = kwargs.pop('prop_name')
        ax = kwargs.get('ax')
        title = kwargs.get('title', '')
        show_plot = kwargs.get('show_plot', True)
        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 10)))
        _plot_network_property(network, prop_name, ax, title, **kwargs)
        if show_plot:
            plt.tight_layout()
            plt.show()
        return

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


def _plot_network_property(network, prop_name, ax, title, node_size=30, **kwargs):
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
            
    if prop_name in _CONTINUOUS_PROPS:
        # Continuous scale for Psi
        node_ids = []
        node_colors = []
        for node in graph.nodes():
            val = props.get(node)
            if val is not None:
                node_ids.append(node)
                node_colors.append(val)
        
        # Background nodes (no psi)
        bg_nodes = [n for n in graph.nodes() if n not in node_ids]
        nx.draw_networkx_nodes(graph, position, nodelist=bg_nodes, ax=ax, node_color='lightgray', node_size=node_size, alpha=0.3)
        
        # Foreground nodes (with psi)
        nodes = nx.draw_networkx_nodes(
            graph, position, nodelist=node_ids, ax=ax,
            node_color=node_colors, node_size=node_size,
            cmap='viridis', alpha=0.8
        )
        
        # Colorbar
        if node_ids:
            plt.colorbar(nodes, ax=ax, label=f'{prop_name}')
            
        nx.draw_networkx_edges(graph, position, ax=ax, edge_color='black', alpha=0.2)
        
    else:
        # Extract unique properties for discrete scale
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
    Plots two networks side by side colored by interC property.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    _plot_network_property(net1, 'count_interC', ax=ax1, title=title1 + ' interC')
    _plot_network_property(net2, 'count_interC', ax=ax2, title=title2 + ' interC')
    plt.tight_layout()
    plt.show()


def plot_networks_cgroup(net1, net2, title1="Network 1", title2="Network 2"):
    """
    Plots two networks side by side colored by cgroup.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    _plot_network_property(net1, 'cgroup', ax=ax1, title=title1 + ' (cgroup)')
    _plot_network_property(net2, 'cgroup', ax=ax2, title=title2 + ' (cgroup)')
    plt.tight_layout()
    plt.show()


def plot_networks_rank(net1, net2, title1="Network 1", title2="Network 2"):
    """
    Plots two networks side by side colored by rank.
    """
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


def plot_matrix_difference(m1, m2, title="Absolute Difference between Matrices", threshold=1e-12):
    """
    Plots the absolute difference between two matrices.
    """
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


def plot_flow_network(obj: Any, **kwargs: Any) -> None:
    """Draw arrows on edges proportional to flow magnitude |Q|, coloured by edge type.

    Arrow direction follows the sign of Q (u→v if Q>0, v→u if Q<0).
    """
    # get the right solution to display

    _plot_edge_vector_property(obj, prop_name='Q', unit='cm³ d⁻¹', **kwargs)


def plot_velocity_network(obj: Any, **kwargs: Any) -> None:
    """Draw arrows on edges proportional to velocity magnitude |v|, coloured by edge type.

    Arrow direction follows the sign of velocity (same as Q).
    """
    _plot_edge_vector_property(obj, prop_name='velocity', unit='cm d⁻¹', **kwargs)


def _plot_edge_vector_property(obj: Any, prop_name: str, unit: str = '', **kwargs: Any) -> None:
    """Internal helper to draw arrows on edges proportional to a scalar property.

    Performance: uses ``LineCollection`` and ``ax.quiver`` (vectorised) instead
    of one ``ax.annotate`` call per edge, giving ~100× speedup on large graphs.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D

    graph = obj.network.graph
    pos   = nx.get_node_attributes(graph, 'position')

    maturity_idx: int = kwargs.get('maturity_idx', 0)
    scenario_idx = kwargs.get('scenario_idx', 'current')

    # --- 0. Restore scenario state when a specific scenario is requested ----
    if scenario_idx != 'current' and hasattr(obj, 'restore_scenario_state'):
        print(f"Restoring scenario state for maturity={maturity_idx}, scenario={scenario_idx}")
        _ = obj.restore_scenario_state(maturity_idx, scenario_idx)

    # ------------------------------------------------------------------ #
    # 1. Compute Q for all edges in a single pass (needed for direction)  #
    # ------------------------------------------------------------------ #
    sol, _ = _get_result_data(obj, **kwargs)
    indices = getattr(obj, 'indice', {})

    if sol is not None and not indices:
        indices = nx.get_node_attributes(graph, 'indice')

    for u, v, eattr in graph.edges(data=True):
        if eattr.get('Q') is None:
            K = eattr.get('K')
            if K is not None and sol is not None and u in indices and v in indices:
                eattr['Q'] = K * (float(sol[indices[u]]) - float(sol[indices[v]]))

    # ------------------------------------------------------------------ #
    # 2. Gather all property values for global normalisation             #
    # ------------------------------------------------------------------ #
    all_vals = []
    for _, _, d in graph.edges(data=True):
        val = d.get(prop_name)
        if val is not None:
            all_vals.append(abs(float(val)))
            
    val_max_raw = max(all_vals) if all_vals else 1.0
    
    if prop_name == 'velocity' and all_vals:
        log_vals = np.log10(np.array(all_vals) + 1)
        val_min = log_vals.min()
        val_max = log_vals.max() - val_min
        if val_max == 0:
            val_max = 1.0
    else:
        val_min = 0.0
        val_max = val_max_raw
    
    # summary of val_max, val_min, mean, median, std
    summary = kwargs.get('summary', False)
    if summary:
        print(f"Summary of {prop_name}: val_max={val_max_raw:.2e}, val_min={min(all_vals) if all_vals else 0.0:.2e}, mean={np.mean(all_vals) if all_vals else 0.0:.2e}, median={np.median(all_vals) if all_vals else 0.0:.2e}, std={np.std(all_vals) if all_vals else 0.0:.2e}")
        if all_vals:
            plt.figure(figsize=(6, 4))
            if prop_name == 'velocity':
                plt.hist(np.log10(np.array(all_vals) + 1), bins=50, color='skyblue', edgecolor='black')
                plt.xlabel(f"log10({prop_name})")
            else:
                plt.hist(all_vals, bins=50, color='skyblue', edgecolor='black')
                plt.xlabel(f"{prop_name}")
            plt.ylabel("Frequency")
            plt.title(f"Histogram of edge {prop_name}")
            plt.tight_layout()

    # ------------------------------------------------------------------ #
    # 3. Collect segments per path type in one O(E) pass                 #
    # ------------------------------------------------------------------ #
    # Structure per path type: list of (src, dst, magnitude)
    path_data: Dict[str, list] = {pt: [] for pt in _PATH_COLORS}

    for u, v, eattr in graph.edges(data=True):
        path = eattr.get('path')
        if path not in path_data:
            continue
        val = eattr.get(prop_name)
        K = eattr.get('K')
        if val is None or K is None or K == 0:
            continue

        if prop_name == 'velocity':
            log_v = np.log10(abs(float(val)) + 1)
            mag = (log_v - val_min) / val_max
        else:
            mag = abs(float(val)) / val_max

        if mag < 1e-12:
            continue

        # Direction is always determined by Q
        Q_sign = eattr.get('Q', 0.0)
        
        pu = pos.get(u, (0.0, 0.0))
        pv = pos.get(v, (0.0, 0.0))
        src, dst = (pu, pv) if Q_sign >= 0 else (pv, pu)
        path_data[path].append((src, dst, mag))

    # ------------------------------------------------------------------ #
    # 4. Render                                                           #
    # ------------------------------------------------------------------ #
    save_path = kwargs.get('save_path', None)

    if kwargs.get('ax') is not None:
        ax = kwargs['ax']
    else:
        fig, ax = plt.subplots(figsize=(12, 12), facecolor='#0d0d1a')
    ax.set_facecolor('#0d0d1a')
    ax.set_aspect('equal')
    ax.set_title(
        f"{prop_name} {f'[{unit}]' if unit else ''} – edges (Mat {kwargs.get('maturity_idx', 0)}) ",
        color='white', fontsize=13,
    )

    # Background nodes (one call)
    node_xy = np.array([pos.get(n, (0.0, 0.0)) for n in graph.nodes()])
    if node_xy.size:
        ax.scatter(node_xy[:, 0], node_xy[:, 1],
                   s=2, c='#333355', alpha=0.3, linewidths=0)

    legend_handles = []
    for path_type, hex_color in _PATH_COLORS.items():
        entries = path_data[path_type]
        if not entries:
            legend_handles.append(
                Line2D([0], [0], color=hex_color, lw=2, label=_PATH_LABELS[path_type])
            )
            continue

        srcs = np.array([e[0] for e in entries])   # (N, 2)
        dsts = np.array([e[1] for e in entries])   # (N, 2)
        mags = np.array([e[2] for e in entries])   # (N,)

        # -- Lines via LineCollection (one artist for all edges) --
        rgba = to_rgba(hex_color)
        alphas = 0.25 + 0.55 * mags                # per-segment alpha
        # LineCollection needs per-segment colours as (N, 4) RGBA
        colors_arr = np.tile(rgba, (len(mags), 1))
        colors_arr[:, 3] = alphas

        segs = np.stack([srcs, dsts], axis=1)      # (N, 2, 2)
        lwidths = 0.3 + 3.5 * mags
        lc = LineCollection(segs, colors=colors_arr, linewidths=lwidths, zorder=2)
        ax.add_collection(lc)

        # -- Arrowheads via quiver (one call for all edges of this type) --
        mid = 0.85 * srcs + 0.15 * dsts            # arrow base near dst
        dx  = dsts[:, 0] - srcs[:, 0]
        dy  = dsts[:, 1] - srcs[:, 1]
        # Scale arrow length to 0 so only the head shows
        ax.quiver(
            mid[:, 0], mid[:, 1], dx, dy,
            color=hex_color,
            alpha=np.clip(alphas, 0, 1).mean(),    # one global alpha for quiver
            angles='xy', scale_units='xy',
            scale=8.0,                             # tune for arrow head size
            width=0.002,
            headwidth=5, headlength=5,
            headaxislength=4,
            zorder=3,
        )

        legend_handles.append(
            Line2D([0], [0], color=hex_color, lw=2, label=_PATH_LABELS[path_type])
        )

    ax.legend(handles=legend_handles, loc='upper right',
              facecolor='#222244', labelcolor='white', edgecolor='white',
              fontsize=8)

    # Auto-scale axes to data (LineCollection doesn't do this automatically)
    ax.autoscale_view()

    show_plot = kwargs.get('show_plot', True)
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
        print(f"Saved → {save_path}")
    elif show_plot:
        plt.tight_layout()
        plt.show()


def plot_K_network(obj, **kwargs):
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
                if K is None or K <= 0:
                    continue
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


def plot_radial_profile(obj: Any, **kwargs: Any) -> None:
    """Plot radial water potential profile including osmotic components and xylem/phloem.

    Merges the former ``plot_psi_radial_profile`` and ``plot_osmotic_radial_profile``
    into a single function.  When osmotic data (``psi_p``, ``psi_os``,
    ``psi_total``) is present in the graph node attributes the function plots
    all three components; otherwise it falls back to the raw solution vector
    (hydrostatic potential only).

    Xylem and sieve-tube (phloem) cells are excluded from the main
    symplastic/apoplastic radial bands and rendered instead as scatter-plot
    overlays with a dedicated colour scale so that their typically large
    potential values do not distort the cortex/endodermis profiles.

    Parameters
    ----------
    obj : Any
        A solved ``Mecha`` instance.
    maturity_idx : int, optional
        Maturity stage index (default ``0``).
    scenario_idx : int or str, optional
        Scenario key to display.  When provided the function restores that
        scenario's saved state via ``restore_scenario_state`` so that the
        graph reflects the correct potentials.  Defaults to ``'current'``
        (whatever is currently on the graph).
    **kwargs : Any
        Additional keyword arguments passed through.
    """
    if not hasattr(obj, 'network'):
        print("Object must have a network attribute.")
        return

    graph = obj.network.graph
    cm = obj.network.cell_manager

    maturity_idx: int = kwargs.get('maturity_idx', 0)
    scenario_idx = kwargs.get('scenario_idx', 'current')
    show_plot: bool = kwargs.get('show_plot', True)
    ax: Optional[Axes] = kwargs.get('ax', None)

    # --- 1. Restore scenario state when a specific scenario is requested ----
    if scenario_idx != 'current' and hasattr(obj, 'restore_scenario_state'):
        print(f"Restoring scenario state for maturity={maturity_idx}, scenario={scenario_idx}")
        restored = obj.restore_scenario_state(maturity_idx, scenario_idx)
        if not restored:
            print(
                f"[plot_radial_profile] WARNING: could not restore state for "
                f"mat={maturity_idx}, scen={scenario_idx}. "
                "Falling back to current graph state."
            )
            sol, _ = _get_result_data(obj, **kwargs)
            if sol is not None and hasattr(obj, 'compute_edge_flows'):
                obj.compute_edge_flows(
                    sol,
                    i_maturity=maturity_idx,
                    i_scenario=0 if scenario_idx == 'standard water flow' else scenario_idx,
                )

    # --- 2. Try to read osmotic data from graph attributes ------------------
    psi_p_attr: Dict[int, float] = nx.get_node_attributes(graph, 'psi_p')
    psi_os_attr: Dict[int, float] = nx.get_node_attributes(graph, 'psi_os')
    psi_total_attr: Dict[int, float] = nx.get_node_attributes(graph, 'psi_total')
    has_osmotic: bool = bool(psi_p_attr)

    # Fallback: run compute_edge_flows if no graph attributes exist yet
    if not has_osmotic:
        sol, _ = _get_result_data(obj, **kwargs)
        if sol is not None and hasattr(obj, 'compute_edge_flows'):
            obj.compute_edge_flows(sol, i_maturity=maturity_idx)
            psi_p_attr = nx.get_node_attributes(graph, 'psi_p')
            psi_os_attr = nx.get_node_attributes(graph, 'psi_os')
            psi_total_attr = nx.get_node_attributes(graph, 'psi_total')
            has_osmotic = bool(psi_p_attr)

    # Hydrostatic-only fallback
    sol, _ = _get_result_data(obj, **kwargs)
    hydrostatic_only: bool = (not has_osmotic) and (sol is not None)

    if not has_osmotic and sol is None:
        print(
            "No potential data found.  Make sure compute_edge_flows() has been called."
        )
        return

    # --- 3. Build per-node identifiers for special cell types ---------------
    xylem_ids: set = {x.node_id for x in cm.xylem} if hasattr(cm, 'xylem') else set()
    sieve_ids: set = {s.node_id for s in cm.sieve} if hasattr(cm, 'sieve') else set()
    inter_ids: set = (
        {i.node_id for i in cm.intercellular} if hasattr(cm, 'intercellular') else set()
    )
    special_ids: set = xylem_ids | sieve_ids | inter_ids

    # --- 4. Collect data ----------------------------------------------------
    data: List[Dict] = []
    xylem_data: List[Dict] = []
    sieve_data: List[Dict] = []

    def _psi_p(node_id: int) -> float:
        if has_osmotic:
            return float(psi_p_attr.get(node_id, 0.0))
        return float(sol[node_id]) if sol is not None else 0.0

    def _psi_os(node_id: int) -> float:
        return float(psi_os_attr.get(node_id, 0.0)) if has_osmotic else 0.0

    def _psi_tot(node_id: int) -> float:
        if has_osmotic:
            p = psi_p_attr.get(node_id, 0.0)
            o = psi_os_attr.get(node_id, 0.0)
            return float(psi_total_attr.get(node_id, p + o))
        return float(sol[node_id]) if sol is not None else 0.0

    for cell in cm:
        rc = np.hypot(cell.x, cell.y)
        if rc < 1e-6:
            continue

        row = {
            'rank': cell.rank, 'side': 'mid', 'r': rc,
            'psi_p': _psi_p(cell.node_id),
            'psi_os': _psi_os(cell.node_id),
            'psi_total': _psi_tot(cell.node_id),
        }

        if cell.node_id in xylem_ids:
            xylem_data.append(row)
        elif cell.node_id in sieve_ids:
            sieve_data.append(row)
        elif cell.node_id not in special_ids:
            data.append(row)

            # Wall nodes (apoplastic sides)
            vc_x, vc_y = cell.x, cell.y
            norm_vc = rc
            for w in cell.walls:
                rw = np.hypot(w.x, w.y)
                vw_x = w.x - cell.x
                vw_y = w.y - cell.y
                norm_vw = np.hypot(vw_x, vw_y)
                if norm_vw < 1e-6:
                    continue
                cos_theta = (vc_x * vw_x + vc_y * vw_y) / (norm_vc * norm_vw)
                side = None
                if cos_theta > 0.4:
                    side = 'out'
                elif cos_theta < -0.4:
                    side = 'in'
                if side is not None:
                    data.append({
                        'rank': cell.rank, 'side': side, 'r': rw,
                        'psi_p': _psi_p(w.node_id),
                        'psi_os': _psi_os(w.node_id),
                        'psi_total': _psi_tot(w.node_id),
                    })

    if not data:
        print("No data collected for radial profile.")
        return

    # --- 5. Aggregate -------------------------------------------------------
    df = pd.DataFrame(data).dropna(subset=['psi_p'])
    agg_cols = {'r': 'mean', 'psi_p': ['mean', 'std'],
                'psi_os': ['mean', 'std'], 'psi_total': ['mean', 'std']}
    profile = df.groupby(['rank', 'side']).agg(agg_cols).reset_index()
    profile.columns = [
        'rank', 'side', 'r',
        'psi_p_mean', 'psi_p_std',
        'psi_os_mean', 'psi_os_std',
        'psi_total_mean', 'psi_total_std',
    ]

    sym_df = profile[profile['side'] == 'mid'].sort_values('r')
    apo_df = profile[profile['side'].isin(['out', 'in'])].sort_values('r')

    # --- 6. Plot ------------------------------------------------------------
    if show_plot:
        fig, ax = plt.subplots(figsize=(12, 7))
    
    if has_osmotic:
        # Symplastic — solid lines / circle markers
        ax.errorbar(sym_df['r'], sym_df['psi_p_mean'], yerr=sym_df['psi_p_std'],
                    fmt='-o', capsize=3, color='#4a90d9', label='Symplastic Ψ_p')
        ax.errorbar(sym_df['r'], sym_df['psi_os_mean'], yerr=sym_df['psi_os_std'],
                    fmt='-o', capsize=3, color='#e04a4a', label='Symplastic Ψ_os')
        ax.errorbar(sym_df['r'], sym_df['psi_total_mean'], yerr=sym_df['psi_total_std'],
                    fmt='-o', capsize=3, color='#222222', label='Symplastic Ψ_total')
        # Apoplastic — dashed lines / square markers
        ax.errorbar(apo_df['r'], apo_df['psi_p_mean'], yerr=apo_df['psi_p_std'],
                    fmt='--s', capsize=3, color='#4a90d9', alpha=0.6,
                    label='Apoplastic Ψ_p')
        ax.errorbar(apo_df['r'], apo_df['psi_os_mean'], yerr=apo_df['psi_os_std'],
                    fmt='--s', capsize=3, color='#e04a4a', alpha=0.6,
                    label='Apoplastic Ψ_os')
        ax.errorbar(apo_df['r'], apo_df['psi_total_mean'], yerr=apo_df['psi_total_std'],
                    fmt='--s', capsize=3, color='#222222', alpha=0.6,
                    label='Apoplastic Ψ_total')
        y_label = 'Water Potential (hPa)'
    else:
        # Hydrostatic only
        ax.errorbar(sym_df['r'], sym_df['psi_p_mean'], yerr=sym_df['psi_p_std'],
                    fmt='-o', capsize=3, color='#7be04a', ecolor='#7be04a',
                    alpha=0.8, label='Symplastic Ψ')
        ax.errorbar(apo_df['r'], apo_df['psi_p_mean'], yerr=apo_df['psi_p_std'],
                    fmt='-s', capsize=3, color='#e07b4a', ecolor='#e07b4a',
                    alpha=0.8, label='Apoplastic Ψ')
        y_label = 'Hydrostatic Potential (hPa)'

    # Xylem overlay — cyan diamonds
    if xylem_data:
        xdf = pd.DataFrame(xylem_data).sort_values('r')
        col = 'psi_p' if has_osmotic else 'psi_total'
        ax.scatter(xdf['r'], xdf[col], marker='D', color='#00c8ff',
                   zorder=5, s=60, label='Xylem Ψ_p')

    # Phloem / sieve overlay — magenta triangles
    if sieve_data:
        sdf = pd.DataFrame(sieve_data).sort_values('r')
        col = 'psi_p' if has_osmotic else 'psi_total'
        ax.scatter(sdf['r'], sdf[col], marker='^', color='#d048c8',
                   zorder=5, s=60, label='Phloem Ψ_p')

    ax.set_xlabel('Radial Distance (µm)')
    ax.set_ylabel(y_label)
    ax.set_title(
        f"Radial Profile — Mat {maturity_idx}, Scen {scenario_idx}"
    )
    ax.legend(loc='best', fontsize=8)

    if show_plot:
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()


def plot_flow_pathway_breakdown(obj, **kwargs):
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
    pivot_df = pivot_df[['plasmodesmata', 'membrane', 'wall']]
    
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

