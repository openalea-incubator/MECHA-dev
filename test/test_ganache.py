# test GANACHE
# GRANAP connection with MECHA

from openalea.mecha.mecha_class import Mecha
from openalea.mecha.utils.data_loader import InData
from openalea.mecha.utils.network_builder import NetworkBuilder
from openalea.mecha.utils.prepare_paraview import prepare_geometrical_properties
from granap.network_base import AbstractNetwork
from granap.root_class import RootAnatomy
from granap.anatomy_writer import AnatomyWriter
from openalea.mecha.utils.visu import visualize

from utils import assert_close_range

import math
import numpy as np


def test_compare_granar_ganache():
    # Create a Mecha instance with a GRANAP network
    root = RootAnatomy()
    root.update_params("aerenchyma", "aerenchyma_proportion", 0.02)
    root.update_params("aerenchyma", "n_files", 1)
    root.update_params("inter_cellular_spaces", "tissue", "cortex")

    _ = root.export_to_adjencymatrix()

    root.write_to_xml("outputs/test_ganache.xml")
    root.write_xml_geometry("outputs/test_ganache_geometry.xml")

    # root.plot_cells()
    # root.plot_network()

    # Create a default input for Mecha use with the GRANAP network
    default_input = InData()
    default_input.geometry.set_maturity_stages([1, 3])

    ganache_network = NetworkBuilder(root)
    ganache_network.populate_from_network()
    mecha_ganache_1 = Mecha(default_input, network=ganache_network)
    solution, _, matrix_W, Kmb, rhs_s = mecha_ganache_1.solve_W(h=0, i_maturity=1)
    m1 = matrix_W
    

    # Create a default input for Mecha use with cellset data
    Ganache_input = InData(cellset_file="outputs/test_ganache.xml", geometry_file="outputs/test_ganache_geometry.xml")
    Ganache_input.geometry.set_maturity_stages([1,3])
    mecha_ganache_2 = Mecha(Ganache_input)

    solution, _, matrix_W, Kmb, rhs_s = mecha_ganache_2.solve_W(h=0, i_maturity=1)

    m2 = matrix_W


    # Ensure same shape
    if m1.shape != m2.shape:
        print(f"Shapes differ: {m1.shape} vs {m2.shape}")
    else:
        # Convert sparse matrices to dense for element-wise comparison
        d1 = m1.toarray() if hasattr(m1, 'toarray') else np.asarray(m1)
        d2 = m2.toarray() if hasattr(m2, 'toarray') else np.asarray(m2)
        diff = np.abs(d1 - d2)
        max_diff = np.max(diff)
        print(f"Max difference in Matrix_W: {max_diff}")

        if max_diff > 1e-10:
            i, j = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"Max difference occurs at ({i}, {j}): m1={d1[i,j]:.6e} vs m2={d2[i,j]:.6e}")
            if i == j:
                print("Diagonal element is different")
                # node type

            flat_indices = np.argsort(diff.flatten())[::-1]
            print("Top 10 differences:")
            for count, idx in enumerate(flat_indices[:10]):
                r, c = np.unravel_index(idx, diff.shape)
                if diff[r, c] < 1e-10:
                    break
                print(
                    f"({r}, {c}): m1={d1[r, c]:.6g}, m2={d2[r, c]:.6g}, diff={diff[r, c]:.6g}"
                )
        else:
            print("Matrices are practically EXACTLY identical.")

    print("mecha ganache 1")
    mecha_ganache_1.compute_conductivities()
    for i in range(len(mecha_ganache_1.root_hydraulic_properties)):
        print(mecha_ganache_1.root_hydraulic_properties[i])

    print("mecha ganache 2")
    mecha_ganache_2.compute_conductivities()
    for i in range(len(mecha_ganache_2.root_hydraulic_properties)):
        print(mecha_ganache_2.root_hydraulic_properties[i])

    print("--- Endodermis cells repr showcase ---")
    for cell in mecha_ganache_2.network.cell_manager.get_by_cgroup(3):
        print(repr(cell))
        break

    def print_network_summary(name, network, filename):
        # write the output to a file
        with open(filename, "w") as f:
            f.write(f"--- {name} Summary ---\n")
            f.write(f"  Cells: {len(network.cell_manager)}\n")
            f.write(f"  Walls: {len(network.cell_manager.walls)}\n")
            f.write(f"  Xylem cells: {len(network.cell_manager.xylem)}\n")
            f.write(f"  Sieve cells: {len(network.cell_manager.sieve)}\n")
            f.write(f"  Passage cells: {len(network.cell_manager.passage)}\n")
            f.write(
                f"  Intercellular cells: {len(network.cell_manager.intercellular)}\n"
            )
            f.write(f"  Epidermis cells: {len(network.cell_manager.epidermis)}\n")

            for cell in network.cell_manager.xylem:
                f.write(f"  Cell {cell.node_id}: {cell.rank}\n")
                f.write(f"  Cell {cell.node_id}: {cell.cgroup}\n")
                for wall in cell.walls:
                    f.write(f"  Wall {wall.node_id}: {wall.length}\n")
        cm = network.cell_manager
        if cm is None:
            print(f"--- {name}: No Cell Manager ---")
            return
        print(f"\n--- {name} Summary ---")
        print(f"  Cells: {len(cm)}")
        print(f"  Walls: {len(cm.walls)}")
        print(f"  Xylem cells: {len(cm.xylem)}")
        print(f"  Sieve cells: {len(cm.sieve)}")
        print(f"  Passage cells: {len(cm.passage)}")
        print(f"  Intercellular cells: {len(cm.intercellular)}")
        print(f"  Epidermis cells: {len(cm.epidermis)}")


    # print_network_summary("Ganache network 1", mecha_ganache_1.network, "compare_nx_ganache_1.txt")
    # print_network_summary("Ganache network 2", mecha_ganache_2.network, "compare_nx_ganache_2.txt")

    plotting = True
    if plotting:
        import matplotlib.pyplot as plt
        from mecha.utils.network_export import (
            plot_network_difference, plot_matrix_difference, 
            plot_networks_interC, plot_networks_rank, 
            plot_edge_and_node_differences,
            plot_intercellular_spaces
        )

        # plot_networks_interC(mecha_ganache_1.network, mecha_ganache_2.network)
        plot_edge_and_node_differences(mecha_ganache_1.network, mecha_ganache_2.network)
        # plot_intercellular_spaces(mecha_ganache_1.network, mecha_ganache_2.network)
        visualize(mecha_ganache_2, visu_type="water_potential", i_maturity=1)
        visualize(mecha_ganache_2, visu_type="flow")
        visualize(mecha_ganache_2, visu_type="conductance")
        # Plot absolute difference between the two matrices
        # plot_matrix_difference(m1, m2, title="Absolute Difference between Matrix_W (Net 1 vs Net 2)")


    for i in range(len(mecha_ganache_1.root_hydraulic_properties)):
        mecha_props = mecha_ganache_1.root_hydraulic_properties[i]
        for j in range(len(mecha_ganache_2.root_hydraulic_properties)):
            ganache_props = mecha_ganache_2.root_hydraulic_properties[j]
            if mecha_props['barrier'] == ganache_props['barrier']:
                assert_close_range(ganache_props['kr'], mecha_props['kr'], "different kr")
                assert_close_range(ganache_props['Kx'], mecha_props['Kx'], "different Kx")
        for j in range(len(mecha_ganache_2.root_hydraulic_properties)):
            ganache_props = mecha_ganache_2.root_hydraulic_properties[j]
            if mecha_props["barrier"] == ganache_props["barrier"]:
                assert_close_range(ganache_props["kr"], mecha_props["kr"], "different kr")
                assert_close_range(ganache_props["Kx"], mecha_props["Kx"], "different Kx")


if __name__ == "__main__":
    test_compare_granar_ganache()
