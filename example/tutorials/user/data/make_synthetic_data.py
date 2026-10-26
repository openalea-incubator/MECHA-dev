"""
The tutorials' example data: SYNTHETIC, not measurements. They stand for what an experiment would give, so that the
user tutorial 05 can show how to give MECHA measured inputs, compare it with observations and fit a parameter.

- measured_roots.csv: five roots of one plant, the number of cortex cell layers of each (as counted on microscope
  images of their cross sections);
- observed_kr.csv: the radial conductivity kr of each root (m s-1 MPa-1, as a root pressure probe or a pressure
  chamber would give), made by MECHA with kaqp = 8e-4 cm hPa-1 d-1 (the default is 4.3e-4) on GRANAP's monocot root
  with that number of cortex layers, with 8 % noise.

    python make_synthetic_data.py          (rewrites the two files in this folder)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from openalea.mecha.simulation import simulate

HERE = Path(__file__).parent
TRUE_KAQP = 8e-4


def main():
    noise = np.random.default_rng(0)
    roots = pd.DataFrame({"root": ["R1", "R2", "R3", "R4", "R5"], "cortex_layers": [3, 4, 5, 6, 8]})
    roots.to_csv(HERE / "measured_roots.csv", index=False)
    kr = [simulate({"kaqp": TRUE_KAQP}, anatomy={"cortex": {"n_layers": int(layers)}}).summary["kr_SI"]
          for layers in roots["cortex_layers"]]
    observed = pd.DataFrame({"root": roots["root"],
                             "kr": [float(f"{value:.3g}") for value in kr * noise.normal(1., 0.08, len(kr))]})
    observed.to_csv(HERE / "observed_kr.csv", index=False)


if __name__ == "__main__":
    main()
