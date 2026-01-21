# MECHA-dev
Development version of MECHA. Private repo

# Mecha Package

A Python package for simulating hydraulic scenarios in plant root systems.

## Installation

```bash
pip install . # not working

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