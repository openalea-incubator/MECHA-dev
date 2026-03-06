
# test GANACHE
# GRANAP connection with MECHA

from mecha.mecha_class import Mecha
from mecha.utils.data_loader import InData
from mecha.utils.network_builder import NetworkBuilder
from mecha.utils.prepare_paraview import prepare_geometrical_properties
from granap.network_base import AbstractNetwork
from granap.root_class import RootAnatomy
from mecha.utils.visu import visualize

import numpy as np

def test_full_mecha_run():
    # Create a Mecha instance with a GRANAP network
    root = RootAnatomy()
    _ = root.export_to_adjencymatrix()

    # Create a default input for Mecha
    # default_input = InData()
    # root_network = NetworkBuilder(root)

    Granar_input = InData(cellset_file="inputs/current_root.xml")
    Granar_input.geometry.set_maturity_stages([1])

    # Create a Mecha instance with the default input

    mecha = Mecha(Granar_input)

    # Test the connection
    # visualize(mecha.network, "network")

    mecha.compute_conductivities()
    for i in range(len(mecha.root_hydraulic_properties)):
        print(mecha.root_hydraulic_properties[i])

    assert not np.isnan(mecha.root_hydraulic_properties[-1]["kr"])
    assert not np.isnan(mecha.root_hydraulic_properties[-1]["Kx"])


if __name__ == "__main__":
    test_full_mecha_run()