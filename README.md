# MECHA-dev
Development version of MECHA. Private repo

# Mecha Package

A Python package for simulating hydraulic scenarios in plant root systems.

## Installation

MECHA, GRANAP and metafspm are not installable from the `openalea3` channel yet in the versions used here: clone the
three repositories on their working branches, next to each other, and install them from source (`pip install -e`)
in one environment.

```bash
mkdir fspm && cd fspm                       # any folder: the three repositories side by side
git clone -b data_structure_api https://github.com/openalea/metafspm.git metafspm
git clone -b to_metafspm https://github.com/openalea-incubator/GRANAP-dev.git GRANAP
git clone -b to_metafspm https://github.com/openalea-incubator/MECHA-dev.git MECHA-dev

mamba env create -f MECHA-dev/conda/environment.yaml     # the dependencies of the three, and MECHA itself
mamba activate mecha
pip install -e metafspm
pip install -e GRANAP
pip install -e MECHA-dev
```

Keep the folder names above: GRANAP's own environment file (`GRANAP/conda/environment.yaml`) expects `metafspm` next
to it. To update, `git pull` in each repository; the editable installs follow.

## Tutorials

Executed notebooks in [`example/tutorials/`](example/tutorials/), on MECHA's metafspm components (start with GRANAP's
[cross-section tutorial](https://github.com/openalea-incubator/GRANAP-dev/blob/to_metafspm/example/tutorials/tutorial_metafspm_01_cross_sections.ipynb) for the anatomies):

| tutorial | what it shows |
|---|---|
| [01 — GRANAP and MECHA](example/tutorials/tutorial_metafspm_01_granap_mecha.ipynb) | coupling GRANAP's anatomies and MECHA's hydraulics in a plant model; MECHA's inputs by role; its parameters in the plants' scenarios; the barriers |
| [02 — a soil gradient](example/tutorials/tutorial_metafspm_02_soil_gradient.ipynb) | the plant problem with a soil drier on one side of a root section: where the water enters and leaves, hydraulic redistribution, the barriers' role |
| [03 — a pine needle's water and sugars](example/tutorials/tutorial_metafspm_03_needle_solutes.ipynb) | an organ of three sections joined axially; its transpiration as an Organ-scale input; MechaSolutes over a day (photosynthesis, starch, the sugars' osmotic feedback) |
| [04 — a whole plant in its soil](example/tutorials/tutorial_metafspm_04_full_plant.ipynb) | the soil–plant–atmosphere seedling: the plant model, the soil model and their links, run to the fixed point |

Run them locally with `jupyter lab example/tutorials/` (the `mecha` environment above). Examples:
[`example/hydraulics/`](example/hydraulics/) (sections, a root axis, the osmotic presets),
[`example/soil_plant_atmosphere/`](example/soil_plant_atmosphere/) (the full plant), and
[`example/comparison/`](example/comparison/) (the port against the original MECHA).

## The original MECHA (`openalea.mecha.legacy`)

The original code is kept, frozen, as `openalea.mecha.legacy`: the reference the port is compared with
(`test/comparison/`, `example/comparison/`).

```python

from openalea.mecha.legacy import InData, Mecha, visualize

# --- Loading data ---
AllIn = InData(cellset_file="extdata/current_root.xml")
# --- Visualize root anatomy ---
visualize(AllIn.cellset_data, "polygon")

# --- Setting up and solving the MECHA section ---
# 1: Casparian strip on the endodermis
# 3: fully suberized endodermis
AllIn.geometry.set_maturity_stages([1, 3])
# --- Instantiate the MECHA class
section = Mecha(AllIn)
# --- Estimation of radial hydraulic conductivity ---
section.compute_conductivities()

# --- Pathway breakdown visualization ---
# 3: fully suberized endodermis
visualize(section, "flow_pathway", maturity_idx=1)

# --- Water fluxes visualization ---
# 1: Casparian strip on the endodermis
visualize(section, "flow", maturity_idx=0)

```

### Coupling with GRANAP (original API)

```python

from openalea.granap.legacy.root_class import RootAnatomy
from openalea.mecha.legacy import InData, Mecha, NetworkBuilder

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
├── conda/                     # Conda environment file (environment.yaml)
├── docs/                      # Documentation
├── extdata/                   # External datasets (e.g., .xml files)
├── src/
│   └── openalea/
│       └── mecha/
│           ├── component.py           # MechaHydraulics: the metafspm component (conductances, standard and plant problems)
│           ├── conductance.py         # the conductances of walls, membranes, plasmodesmata, the vapour path
│           ├── geometry.py            # MECHA's geometry of a section, from its DataStructure variables
│           ├── osmotic.py             # MechaOsmoticPresets: the osmotic potentials and reflection coefficients
│           ├── solutes.py             # MechaSolutes: sucrose and starch transport
│           ├── plots.py               # section figures
│           ├── legacy/                # the original MECHA, frozen
│           └── legacy_reactions/      # the reactions branch, frozen (the solutes' reference)
├── example/
│   ├── tutorials/             # the metafspm tutorials (notebooks), and the original's tutorials ported
│   ├── hydraulics/            # sections and a root axis
│   ├── soil_plant_atmosphere/ # the full plant in its soil
│   └── comparison/            # the port against the original
├── test/                      # component/ (the port), comparison/ (against the original), legacy/
├── LICENSE
├── pyproject.toml
└── README.md

```
