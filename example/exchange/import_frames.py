"""
Importing fspm-TF files: the solved sections export_frames.py wrote (run first when missing), read two ways.

    python import_frames.py [--show]          figure in ./outputs/water_potential_by_tissue.png

1. As any platform reads them, with openalea.fspm_tf alone: the Compartment table gives the protoplasts' water
   potential (standard name water_potential, in the unit its declaration gives) tissue by tissue from the soil to
   the xylem, and the water the section takes up from the soil, for each barrier.
2. Back into metafspm, with openalea.metafspm.io.fspm_tf.import_frame: each frame rebuilds the section's
   DataStructure (its MPG, its variables, bit for bit), with the same uptake.
"""
import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

from openalea import fspm_tf
from openalea.metafspm.io.fspm_tf import import_frame

from export_frames import BARRIERS, OUTPUTS, export

PROTOPLAST = 0                                                       # GRANAP's kind of a protoplast Compartment
OUTSIDE_IN = ("epidermis", "exodermis", "cortex", "endodermis", "pericycle", "stele", "phloem", "protoxylem",
              "metaxylem")


def profile(path: str) -> tuple:
    """(the protoplasts' mean water potential per tissue (hPa), the section's water uptake (cm3 d-1)), read with
    openalea.fspm_tf."""
    compartments = fspm_tf.read(path).data_structures[0].entity_set("Compartment")
    declared = compartments.variables["water_potential"]
    assert declared.unit == "hPa" and declared.standard_name == "water_potential", (declared.unit, declared.standard_name)
    protoplasts = compartments.columns["kind"] == PROTOPLAST
    table = pd.DataFrame({"tissue": compartments.columns["tissue"][protoplasts],
                          "water_potential": compartments.columns["water_potential"][protoplasts]})
    means = table.groupby("tissue")["water_potential"].mean()
    uptake = float(np.nansum(compartments.columns["root_uptake"]))
    return means.reindex([tissue for tissue in OUTSIDE_IN if tissue in means.index]), uptake


def main():
    parser = argparse.ArgumentParser(description="MECHA's fspm-TF sections read back")
    parser.add_argument("--show", action="store_true")
    show = parser.parse_args().show
    paths = {barrier: os.path.join(OUTPUTS, "frames", f"barrier_{barrier}.glb") for barrier in BARRIERS}
    if not all(os.path.exists(path) for path in paths.values()):
        export()
    profiles, uptakes = {}, {}
    for barrier, path in paths.items():
        profiles[f"barrier {barrier}"], uptakes[barrier] = profile(path)
        identifier, ds = next(iter(import_frame(path).items()))       # {id: DataStructure}, as metafspm wrote them
        rebuilt = float(np.nansum(ds.get("root_uptake")))
        assert rebuilt == uptakes[barrier]
        print(f"barrier {barrier}: uptake {uptakes[barrier]:.4g} cm3 d-1, the same in the DataStructure metafspm "
              f"rebuilt ('{identifier}', {ds.n_nodes()} nodes)")
    table = pd.DataFrame(profiles)
    print("\nRead with openalea.fspm_tf (the protoplasts' mean water potential, hPa, from the soil to the xylem):")
    print(table.round(1).to_string())

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(figsize=(7, 4))
    table.plot(ax=axes, marker="o")
    axes.set_xticks(range(len(table.index)), table.index, rotation=30, ha="right")
    axes.set_ylabel("water potential (hPa)")
    axes.set_title("MECHA's sections, read back from their fspm-TF frames")
    figure.tight_layout()
    path = os.path.join(OUTPUTS, "water_potential_by_tissue.png")
    figure.savefig(path, dpi=120)
    print(f"figure: {path}")
    if show:
        plt.show()


if __name__ == "__main__":
    main()
