# MECHA-dev
Development version of MECHA. Private repo

# Mecha Package

A Python package for simulating hydraulic scenarios in plant root systems.

## Installation

In MECHA-dev root directory, install the environment with
```bash
mamba create -f ./conda/environment.yaml -y
mamba activate mecha_env 
```

For source to be recognized in the environment, run the following at the root of the MECHA-env directory
```bash
pip install -e . 
```

## Usage

```python

from mecha.utils import InData, visualize
from mecha import MECHA

AllIn = InData(cellset_file="extdata/current_root.xml")
visualize(AllIn.cellset_data, "polygon")

section = mecha(AllIn)
section.solve()
results = section.solution

visualize(results[0], "wall")


```