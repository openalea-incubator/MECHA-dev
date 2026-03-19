
# test GANACHE
# GRANAP connection with MECHA

from mecha.mecha_class import Mecha
from mecha.utils.data_loader import InData
from mecha.utils.network_builder import NetworkBuilder
# from mecha.utils.prepare_paraview import prepare_geometrical_properties
# from granap.network_base import AbstractNetwork
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
    default_input.geometry.set_maturity_stages([1,3])

    ganache_network = NetworkBuilder(root)
    ganache_network.populate_from_network()
    mecha_ganache = Mecha(default_input, network=ganache_network)

    # Create a default input for Mecha use with cellset data
    Granar_input = InData(cellset_file="inputs/current_root5.xml")
    Granar_input.geometry.set_maturity_stages([1,3])

    # Create a Mecha instance with the default input
    mecha = Mecha(Granar_input)

    mecha.compute_conductivities()
    for i in range(len(mecha.root_hydraulic_properties)):
        print(mecha.root_hydraulic_properties[i])

    print("mecha ganache")
    mecha_ganache.compute_conductivities()
    for i in range(len(mecha_ganache.root_hydraulic_properties)):
        print(mecha_ganache.root_hydraulic_properties[i])

    plotting = False
    if plotting:
        import matplotlib.pyplot as plt

        # Test the connection visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)

        visualize(mecha.network, "network", ax=ax1, title="Granar - cellset network")
        visualize(mecha_ganache.network, "network", ax=ax2, title="Ganache network")

        plt.tight_layout()
        plt.show()


    for i in range(len(mecha.root_hydraulic_properties)):
        mecha_props = mecha.root_hydraulic_properties[i]
        for j in range(len(mecha_ganache.root_hydraulic_properties)):
            ganache_props = mecha_ganache.root_hydraulic_properties[j]
            if mecha_props['barrier'] == ganache_props['barrier']:
                assert_close_range(ganache_props['kr'], mecha_props['kr'], "different kr")
                assert_close_range(ganache_props['Kx'], mecha_props['Kx'], "different Kx")

    print("Test granar / ganache comparision was successfull")

def assert_close_range(a, b, msg: str = ""):
    assert a != 0 and b != 0, "Zero has no log10 order of magnitude"
    assert math.floor(math.log10(abs(a))) == math.floor(math.log10(abs(b))), f'{msg}: {a} vs. {b}'


if __name__ == "__main__":
    test_compare_granar_ganache()
