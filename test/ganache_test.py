
# test GANACHE
# GRANAP connection with MECHA

from mecha.mecha_class import Mecha
from mecha.utils.data_loader import InData
from mecha.utils.network_builder import NetworkBuilder
from mecha.utils.prepare_paraview import prepare_geometrical_properties
from granap.network_base import AbstractNetwork
from granap.root_class import RootAnatomy
from mecha.utils.visu import visualize
import math


def test_compare_granar_ganache():
    # Create a Mecha instance with a GRANAP network
    root = RootAnatomy()
    _ = root.export_to_adjencymatrix()

    # root.plot_cells()
    # root.plot_network()

    # Create a default input for Mecha use with the GRANAP network
    default_input = InData()
    default_input.geometry.set_maturity_stages([1])
    ganache_network = NetworkBuilder(root)
    ganache_network.populate_from_network()
    mecha_ganache = Mecha(default_input, network=ganache_network)

    # Create a default input for Mecha use with cellset data
    Granar_input = InData(cellset_file="inputs/current_root.xml")
    Granar_input.geometry.set_maturity_stages([1])

    # Create a Mecha instance with the default input
    mecha = Mecha(Granar_input)

    print("mecha classic")
    mecha.compute_conductivities()
    for i in range(len(mecha.root_hydraulic_properties)):
        print(mecha.root_hydraulic_properties[i])

    print("mecha ganache")
    mecha_ganache.compute_conductivities()
    for i in range(len(mecha_ganache.root_hydraulic_properties)):
        print(mecha_ganache.root_hydraulic_properties[i])

    perim_ganache = mecha_ganache.network.perimeter
    perim_mecha = mecha.network.perimeter
    assert perim_ganache != 0 and perim_mecha != 0, "Zero has no log10 order of magnitude"
    assert math.floor(math.log10(abs(perim_ganache))) == math.floor(math.log10(abs(perim_mecha)))

    plotting = True
    if plotting:
        import matplotlib.pyplot as plt

        # Test the connection visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)

        visualize(mecha.network, "network", ax=ax1, title="Root network")
        visualize(mecha_ganache.network, "network", ax=ax2, title="Ganache network")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    test_compare_granar_ganache()
