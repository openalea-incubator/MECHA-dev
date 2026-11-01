"""
The tutorials' example data: SYNTHETIC, not measurements. They stand for what an experiment would give, so that the
user tutorial 05 can show how to give MECHA measured inputs, compare it with observations and fit a parameter.

- measured_roots.csv: five roots of one plant, the number of cortex cell layers of each (as counted on microscope
  images of their cross sections);
- observed_kr.csv: the radial conductivity kr of each root (m s-1 MPa-1, as a root pressure probe or a pressure
  chamber would give), made by MECHA with kaqp = 8e-4 cm hPa-1 d-1 (the default is 4.3e-4) on GRANAP's monocot root
  with that number of cortex layers, with 8 % noise;
- observed_krs.csv (user tutorial 08): a seedling's root system conductance Krs (cm3 hPa-1 d-1, as a pressure chamber
  on the whole root system would give), measured twice: as it is ("control") and with its aquaporins inhibited
  ("inhibited", their contribution kaqp divided by 10), made by MECHA with kaqp = 2e-4 and kmb = 6e-5 cm hPa-1 d-1
  (the aquaporins' and the rest of the membranes' conductivities; the defaults are 4.3e-4 and 3e-5) on the seedling
  SEEDLING, with 3 % noise.

    python make_synthetic_data.py          (rewrites the files in this folder)
    python make_synthetic_data.py krs      (only observed_krs.csv)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from openalea.mecha.simulation import root_system, simulate

HERE = Path(__file__).parent
TRUE_KAQP = 8e-4
SEEDLING = dict(main=(3, 1, 1), laterals=((1, (1,)),))     # a main root of three segments, a lateral on the second
TRUE_SEEDLING = {"kaqp": 2e-4, "kmb": 6e-5}


def main():
    noise = np.random.default_rng(0)
    roots = pd.DataFrame({"root": ["R1", "R2", "R3", "R4", "R5"], "cortex_layers": [3, 4, 5, 6, 8]})
    roots.to_csv(HERE / "measured_roots.csv", index=False)
    kr = [simulate({"kaqp": TRUE_KAQP}, anatomy={"cortex": {"n_layers": int(layers)}}).summary["kr_SI"]
          for layers in roots["cortex_layers"]]
    observed = pd.DataFrame({"root": roots["root"],
                             "kr": [float(f"{value:.3g}") for value in kr * noise.normal(1., 0.08, len(kr))]})
    observed.to_csv(HERE / "observed_kr.csv", index=False)


def krs():
    noise = np.random.default_rng(1)
    values = [root_system(parameters=TRUE_SEEDLING, experiment=experiment, **SEEDLING).conductance
              for experiment in ("control", "inhibited")]
    pd.DataFrame({"variable": "root_system_conductance", "experiment": ["control", "inhibited"],
                  "value": [float(f"{v:.3g}") for v in values * noise.normal(1., 0.03, 2)],
                  "unit": "cm3 hPa-1 d-1"}).to_csv(HERE / "observed_krs.csv", index=False)


if __name__ == "__main__":
    if sys.argv[1:] != ["krs"]:
        main()
    krs()
