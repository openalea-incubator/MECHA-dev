"""
Exporting solved root cross-sections as fspm-TF files: one frame per apoplastic barrier (an endodermal Casparian
strip, then a fully suberized endodermis), the format the other FSPM platforms (CPlantBox, GroIMP, VirtualPlantLab)
and ParaView read.

    python export_frames.py [--barriers B ...]          frames in ./outputs/frames: barrier_<B>.glb

Each section is the one openalea.mecha.simulation.simulate solves: GRANAP's monocot root anatomy, MECHA's hydraulics,
the soil at 0 hPa and the xylem at -5000 hPa, with a barrier of maturity code B (1: an endodermal Casparian strip,
3: a fully suberized endodermis; 2, suberin with passage cells, solves as 3 on this section). A frame holds the
section's whole MPG, its compartments with their water potentials, its connections with their conductances and water
flows, and the cells' polygons as a display mesh. Each frame is then checked with fspm-TF's validator.

Open a frame in ParaView with fspm-TF's reader plugin (fspm-TF/paraview/fspm_tf_reader.py), colour by
Compartment.water_potential; import_frames.py reads the files back.
"""
import argparse
import os

from openalea.fspm_tf.validate import validate_file
from openalea.metafspm.io.fspm_tf import export_scene

from openalea.mecha.simulation import simulate

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.join(HERE, "outputs")
BARRIERS = (1, 3)


def export(barriers=BARRIERS, outputs: str = OUTPUTS) -> list:
    """Solve a section for each barrier and write it as a frame; the frames' paths."""
    folder = os.path.join(outputs, "frames")
    os.makedirs(folder, exist_ok=True)
    paths = []
    for barrier in barriers:
        section = simulate(barrier=barrier)
        path = os.path.join(folder, f"barrier_{barrier}.glb")
        export_scene(path, section.scene)
        print(f"barrier {barrier}: radial conductivity kr {section.summary['kr']:.3g} cm hPa-1 d-1")
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description="MECHA's solved sections as fspm-TF frames, one per barrier")
    parser.add_argument("--barriers", nargs="+", type=int, default=list(BARRIERS))
    for path in export(parser.parse_args().barriers):
        report = validate_file(path)
        print(f"{path} ({os.path.getsize(path) / 1e6:.1f} MB): fspm-TF validator {len(report.errors)} error(s), "
              f"{len(report.warnings)} warning(s)")


if __name__ == "__main__":
    main()
