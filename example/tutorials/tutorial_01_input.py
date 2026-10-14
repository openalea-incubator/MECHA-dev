"""
Tutorial 01, ported (example/legacy/tutorial_01_input.ipynb): an anatomy into MECHA, its radial conductivity kr and
axial conductance Kx at two maturity stages, and the views of its hydraulics.

In the port, MECHA is a metafspm component on the plant's DataStructure, and its anatomies come from GRANAP's
component, either generated (the root preset) or read from a CellSet XML file (GRANAP's CellSetOrgan: the legacy's
own XML builder is not ported, Q-M7; here a Brassicaceae root from GRANAP's tests). The same section is then solved at barrier 1 (Casparian strip) and 3
(suberized endodermis).

    python tutorial_01_input.py [cellset]     (from MECHA-dev/example/tutorials; figures in ./outputs/)
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import StructuralComponent
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.scene.population import build_population

from openalea.granap.cellset_organ import CellSetOrgan
from openalea.granap.metafspm.component import GranapAnatomy, junction_rules
from openalea.granap.metafspm.library import AnatomyLibrary
from openalea.mecha import plots
from openalea.mecha.component import MechaHydraulics

HERE = os.path.dirname(os.path.abspath(__file__))
CELLSET = os.path.join(HERE, "inputs", "partial_CellSet_brasicaceae.xml")    # a CellSet file (µm), from GRANAP
HEIGHT = 200e-6                                  # m: the legacy's default section height (200 µm)
BARRIERS = (1, 3)


class Root(StructuralComponent):
    """One root segment, 200 µm long (the section MECHA solves)."""

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        s, L = g.scales, g.labels
        complex_ = plant
        for scale, label in ((s.Axis, L.Axis.Root), (s.GrowthUnit, L.GrowthUnit.Root), (s.Phytomer, L.Phytomer.Root),
                             (s.Organ, L.Organ.RootInternode)):
            complex_ = g.add_component(complex_, **PropsConfig(scale=scale, edge_type='/', label=label))
        segment = g.add_component(complex_, **PropsConfig(scale=s.SubOrgan, edge_type='/',
                                                          label=L.SubOrgan.RootSegment))
        g.property("radius")[segment] = 0.55e-3
        g.property("length")[segment] = HEIGHT
        g.property("distance_from_tip")[segment] = 0.


def cellset_organ_class(path, scale=0.001):
    """
    An organ class for GRANAP's library reading the cellset at *path* (whatever the class's parameters); *scale*:
    the file's unit in mm (GRANAR's files are in mm, CellSet's in µm: 0.001).
    """
    def organ(parameters, seed=0):
        return CellSetOrgan(path, scale=scale, seed=seed)
    return organ


def section(source):
    """The plant, its anatomy (generated, or from the cellset), the DataStructure and MECHA's component."""

    class Anatomy(GranapAnatomy):
        library = (AnatomyLibrary() if source == "granap" else
                   AnatomyLibrary(organ_class=cellset_organ_class(CELLSET)))

    Choregrapher().add_simulation_time_step(3600.)
    plants = pd.DataFrame([dict(plant="p0", model=None, x=0., y=0., z=0., rotation=0., scenario={"parameters": {}})])
    g, _ = build_population(plants, initiators=(Root, Anatomy))
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=junction_rules(g))
    Anatomy(data_structure=ds)()
    ds.register("barrier", np.ones(ds.n_nodes()), location="node")
    return ds, MechaHydraulics(data_structure=ds), Anatomy.library


def main(source="granap"):
    ds, mecha, library = section(source)
    segment = int(ds.entity_ids("SubOrgan")[0])
    frames = {}
    print(f"Hydraulic properties ({source}):")
    for barrier in BARRIERS:
        ds.set("barrier", np.full(ds.n_nodes(), float(barrier)))
        mecha()
        print(f"  barrier {barrier} | kr = {ds.get('kr')[0]:.4e} cm hPa-1 d-1 | Kx = {ds.get('Kx')[0]:.4e} "
              f"cm4 hPa-1 d-1")
        frames[barrier] = plots.port_frame(mecha, segment, libraries=(library,))

    folder = os.path.join(HERE, "outputs")
    os.makedirs(folder, exist_ok=True)
    frame = frames[BARRIERS[-1]]
    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    plots.tissue_map(axes[0, 0], frame, title=f"the section ({source})")
    points = plots.psi_map(axes[0, 1], frame, title=f"water potential, barrier {BARRIERS[-1]}")
    fig.colorbar(points, ax=axes[0, 1], label="ψ (hPa)", shrink=0.8)
    for barrier, style in zip(BARRIERS, ("-", "--")):
        plots.radial_profile(axes[0, 2], frames[barrier], label=f"barrier {barrier}", linestyle=style)
    axes[0, 2].legend(fontsize=7)
    lines = plots.conductance_by_path(axes[1], frame)
    fig.colorbar(lines, ax=axes[1, :], label="log10 K (cm³ hPa⁻¹ d⁻¹)", shrink=0.8)
    fig.savefig(os.path.join(folder, f"tutorial_01_{source}_section.png"), dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    plots.pathway_breakdown(axes[0], frame, title=f"pathways of the radial flow, barrier {BARRIERS[-1]}")
    plots.flow_vectors(axes[1], frame, "membrane")
    plots.flow_vectors(axes[2], frame, "plasmodesmata")
    fig.tight_layout()
    fig.savefig(os.path.join(folder, f"tutorial_01_{source}_flows.png"), dpi=300)
    plt.close(fig)
    print(folder)


if __name__ == "__main__":
    main("cellset" if "cellset" in sys.argv[1:] else "granap")
