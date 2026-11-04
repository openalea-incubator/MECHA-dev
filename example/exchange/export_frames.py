"""
Exporting solved root cross-sections as encoding3d files: one frame per apoplastic barrier (an endodermal Casparian
strip, then a fully suberized endodermis), the format the other FSPM platforms (CPlantBox, GroIMP, VirtualPlantLab)
and ParaView read.

    python export_frames.py [--barriers B ...]          frames in ./outputs/frames: barrier_<B>.glb

Each section is the one openalea.mecha.simulation.simulate solves: GRANAP's monocot root anatomy, MECHA's hydraulics,
the soil at 0 hPa and the xylem at -5000 hPa, with a barrier of maturity code B (1: an endodermal Casparian strip,
3: a fully suberized endodermis; 2, suberin with passage cells, solves as 3 on this section). A frame holds the
section's whole MPG, its compartments with their water potentials, its connections with their conductances and water
flows, and the cells' polygons as a display mesh. Each frame is then checked with encoding3d's validator.

A second, filtered export shows logging: the compartments and their connections only, with the water potential,
the uptake and the flows, in ./outputs/log/frames (outputs only: such frames are not reimported).

Open a frame in ParaView with encoding3d's reader plugin (encoding3d/paraview/encoding3d_reader.py), colour by
Compartment.water_potential; import_frames.py reads the files back.
"""
import argparse
import os

from openalea.encoding3d.validate import validate_file
from openalea.metafspm.io.encoding3d import export_scene

from openalea.mecha.simulation import simulate

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.join(HERE, "outputs")
BARRIERS = (1, 3)
# a log of the water: the compartments and their connections, their water potential, uptake and flows
LOG = dict(scales=["Compartment"], variables=["tissue", "kind", "water_potential", "root_uptake", "flow"])


def export(barriers=BARRIERS, outputs: str = OUTPUTS, options: dict = None) -> list:
    """Solve a section for each barrier and write it as a frame (filtered by *options*, e.g. LOG); the frames'
    paths."""
    folder = os.path.join(outputs, "frames")
    os.makedirs(folder, exist_ok=True)
    paths = []
    for barrier in barriers:
        section = simulate(barrier=barrier)
        path = os.path.join(folder, f"barrier_{barrier}.glb")
        export_scene(path, section.scene, **(options or {}))
        print(f"barrier {barrier}: radial conductivity kr {section.summary['kr']:.3g} cm hPa-1 d-1")
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description="MECHA's solved sections as encoding3d frames, one per barrier")
    parser.add_argument("--barriers", nargs="+", type=int, default=list(BARRIERS))
    barriers = parser.parse_args().barriers
    for path in export(barriers):
        report = validate_file(path)
        print(f"{path} ({os.path.getsize(path) / 1e6:.1f} MB): encoding3d validator {len(report.errors)} error(s), "
              f"{len(report.warnings)} warning(s)")
    for path in export(barriers, os.path.join(OUTPUTS, "log"), options=LOG):
        report = validate_file(path)
        print(f"logged {os.path.basename(path)} (scales {LOG['scales']}, {len(LOG['variables'])} variables): "
              f"{os.path.getsize(path) / 1e6:.1f} MB, {len(report.errors)} error(s)")


if __name__ == "__main__":
    main()
