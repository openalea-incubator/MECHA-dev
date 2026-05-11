# MECHA-dev
Development version of MECHA. Private repo

# Mecha Package

A Python package for simulating hydraulic scenarios in plant root systems.

## Installation

In MECHA-dev root directory, install the environment with
```bash
mamba create -f ./conda/environment.yaml -y
mamba activate mecha 
```

For source to be recognized in the environment, run the following at the root of the MECHA-env directory
```bash
pip install -e . 
```

## Usage

```python

from mecha.utils import InData, visualize
from mecha import MECHA

# --- Loading data ---
AllIn = InData(cellset_file="extdata/current_root.xml")
# --- Visualize root anatomy ---
visualize(AllIn.cellset_data, "polygon")

# --- Setting up and solving the MECHA section ---
# 1: Casparian strip on the endodermis
# 3: fully suberized endodermis
AllIn.geometry.set_maturity_stages([1, 3])
# --- Instantiate the MECHA class
section = mecha(AllIn)
# --- Estimation of radial hydraulic conductivity ---
section.compute_conductivities()

# --- Pathway breakdown visualization ---
# 3: fully suberized endodermis
visualize(section, "flow_pathway", maturity_idx=1)

# --- Water fluxes visualization ---
# 1: Casparian strip on the endodermis
visualize(section, "flow", maturity_idx=0)

```

### Coupling with GRANAP

```python

from granap import RootAnatomy
from mecha.utils import NetworkBuilder, InData

root = RootAnatomy()
_ = root.export_to_adjencymatrix()

ganache_network = NetworkBuilder(root)
ganache_network.populate_from_network()

default_input = InData()
# set the maturity stages for the root
# 3: suberized endodermis
# 4: suberized endodermis + Casparian strip on the exodermis
# 5: Casparian strip on the endodermis & exodermis
# 6: suberized exodermis + Casparian strip on the endodermis
# 7: Casparian strip on the exodermis
# 8: suberized endodermis & exodermis
# 9: lignin cap on the exodermis + Casparian strip on the endodermis
default_input.geometry.set_maturity_stages(range(3,10)) # [3,4,5,6,7,8,9]
granap_mecha = Mecha(default_input, network=ganache_network)

granap_mecha.compute_conductivities()

for i in range(len(granap_mecha.root_hydraulic_properties)):
    print(granap_mecha.root_hydraulic_properties[i])


```
