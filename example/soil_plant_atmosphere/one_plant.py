"""
One seedling in a soil column under a dry air, as metafspm's soil–plant–atmosphere example, but with GRANAP's
anatomies and MECHA's hydraulics in the plant: water from the water table, through the soil, the roots' cells and
walls, the stem and the leaves, out of the stomata. The plant and the soil exchange at each scene step until the
water potentials stop changing (a lagged fixed point).

    python one_plant.py [output folder]      (from MECHA-dev/example/soil_plant_atmosphere; default outputs/one_plant)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

from openalea.metafspm.scene.scene import Scene

from seedling_spac import (CM3_PER_DAY_TO_MM3_PER_S, SCENE_TRANSLATOR, SeedlingHydraulics, Soil, largest_change,
                           root_surface_mappings)
import plots


def scene(output_dirpath=None, tolerance=1e-4):
    table = pd.DataFrame([dict(plant="seedling", model=SeedlingHydraulics, x=0.1, y=0.1, z=0., rotation=0.,
                               scenario={"parameters": {}})])
    table.attrs.update(xrange=0.2, yrange=0.2)                       # a 0.2 m x 0.2 m column of soil
    stop = largest_change(tolerance)
    built = Scene(table, environment=[Soil], environment_scenarios=[{}], translator=SCENE_TRANSLATOR, time_step=3600,
                  mappings=root_surface_mappings, stop_when=stop, output_dirpath=output_dirpath)
    return built, stop


def summary(built, stop):
    plant = built.populations[0].data_structure
    soil = next(model for model in built.environment if isinstance(model, Soil)).grid
    transpiration = plant.get("evaporation").sum() * CM3_PER_DAY_TO_MM3_PER_S
    uptake = plant.get("root_uptake").sum() * CM3_PER_DAY_TO_MM3_PER_S
    print(f"converged in {built.iteration} steps; largest change of Ψ per step (MPa): "
          + ", ".join(f"{change:.1e}" for change in stop.history))
    print(f"{plant.n_nodes()} Compartments, {plant.n_edges()} Connections")
    print(f"transpiration {transpiration:.4f} mm3 s-1, root uptake {uptake:.4f} mm3 s-1, soil uptake "
          f"{soil.get('plant_uptake').sum():.4f} mm3 s-1, soil evaporation {soil.get('evaporation').sum():.4f} "
          f"mm3 s-1; leaf ψ down to {plant.get('water_potential').min() / 1e4:.2f} MPa")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                "outputs", "one_plant")
    start = time.perf_counter()
    built, stop = scene(output_dirpath=folder)
    print(f"built in {time.perf_counter() - start:.0f} s")
    start = time.perf_counter()
    built.simulate(30)
    print(f"simulated in {time.perf_counter() - start:.0f} s")
    summary(built, stop)
    plots.all_figures(built, folder)
