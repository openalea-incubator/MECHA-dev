# MECHA-dev
Development version of MECHA. Private repo

# Mecha Package

A Python package for simulating hydraulic scenarios in plant root systems.

## Installation

### Public installation

```bash
conda install -c openalea3 openalea.mecha
```

### Developer installation

Git clone `Granap` and `Mecha` in the same parent directory

```bash
git clone https://github.com/openalea-incubator/GRANAP-dev.git 
git clone https://github.com/openalea-incubator/MECHA-dev.git
cd MECHA-dev
```

In MECHA-dev root directory, install the environment with
```bash
mamba create -f ./conda/environment.yaml -y
mamba activate mecha 
```

For source to be recognized in the environment, run the following at the root of the MECHA-env directory
```bash
pip install -e .
cd ../GRANAP-dev
pip install -e .
```

## Usage

```python

from openalea.mecha.utils import InData, visualize
from openalea.mecha import MECHA

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

from openalea.granap import RootAnatomy
from openalea.mecha.utils import NetworkBuilder, InData

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


## repository structure

```bash
MECHA-dev/
├── conda/                 # Conda environment file (environment.yaml)
├── docs/                   # Documentation
├── extdata/               # External datasets (e.g., .xml files)
├── src/
│   └── openalea/
│       └── mecha/
│           ├── mecha_class.py         # Main MECHA class
│           ├── __init__.py            # Package initialization
│           ├── GUI/                   # GUI
│           │   └── app.py               # MECHA GUI (In progress)
│           └── utils/
│               ├── network_builder.py   # Network construction
│               ├── scenario_builder.py  # Scenario setup
│               ├── solute_transport.py  # Solute transport algorithms
│               ├── hydraulic_solver.py  # Hydraulic solving utilities
│               ├── data_loader.py       # Data loading (InData)
│               ├── visu.py              # Visualization functions
│               └── network_export.py    # Network export tools
│
├── tutorials/             # Examples of how to use the package
├── test/                  # Unit tests and integration tests
│   ├── test_advdiff.py
│   ├── test_osmotic.py
│   ├── test_transport.py
│   └── test_visu_types.py
├── .gitignore
├── LICENSE
├── pyproject.toml
└── README.md

```
