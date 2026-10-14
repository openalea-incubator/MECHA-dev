"""
MECHA as a metafspm functional component on the MPG DataStructure (devplan_metafspm.md, §3 and §8).

    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=junction_rules(g))
    anatomy = GranapAnatomy(data_structure=ds)
    mecha = MechaHydraulics(data_structure=ds)

It reads each section's network from the DataStructure (GRANAP's declared outputs: role "Anatomy"), derives MECHA's
geometry once per distinct anatomy (sections copied from the same template share it), and writes its results on the
Compartments, Connections and segments.
"""
from dataclasses import dataclass

import numpy as np

from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig
from openalea.metafspm.solve.decorator import state

from openalea.mecha.geometry import section_geometry, section_of, section_parts

SCALES = ScalesConfig()


def _doc(unit, description, **values):
    return dict(unit=unit, unit_comment="", description=description, min_value=-1e12, max_value=1e12,
                value_comment="", references="MECHA (Couvreur et al. 2018)", DOI=["10.1104/pp.18.01006"], **values)


def _node(unit, description):
    return dict(**_doc(unit, description), by="Anatomy", scale=SCALES.Compartment, location="node",
                state_variable_type="descriptor")


def _edge(unit, description):
    return dict(**_doc(unit, description), by="Anatomy", scale=SCALES.Connection, location="edge",
                state_variable_type="descriptor")


@dataclass
class MechaHydraulics(FunctionalComponent):
    """
    MECHA's cross-section hydraulics on every section of the DataStructure.

    Inputs (role "Anatomy": GRANAP's anatomy component): each section's network, its nodes (protoplasts, walls,
    wall junctions) and edges (membranes, walls, plasmodesmata), in GRANAP's units (mm, mm2).
    """
    # ── Inputs: the sections' networks ──
    kind: float = input_variable(**_node("", "0 protoplast, 1 wall, 2 wall junction"))
    cgroup: float = input_variable(**_node("", "the protoplast's MECHA cell group (NaN: apoplast)"))
    tissue: object = input_variable(**_node("", "the protoplast's tissue (None: a wall or a junction)"))
    section_x: float = input_variable(**_node("mm", "position in the section"))
    section_y: float = input_variable(**_node("mm", "position in the section"))
    wall_length: float = input_variable(**_node("mm", "the wall's length in the section (0: not a wall)"))
    cell_area: float = input_variable(**_node("mm2", "the protoplast's area in the section (0: apoplast)"))
    network_path: float = input_variable(**_edge("", "edge type: membrane 0, wall_air 1, wall 2, plasmodesmata 3, "
                                                     "air_link 4, axial junction 5"))
    edge_length: float = input_variable(**_edge("mm", "the edge's length"))
    edge_dist: float = input_variable(**_edge("mm", "the distance between the edge's nodes"))
    anatomy_template: float = input_variable(**_doc("", "the segment's anatomy template (-1: none): sections of the "
                                                        "same template share MECHA's geometry"),
                                             by="Anatomy", scale=SCALES.SubOrgan, state_variable_type="descriptor")

    # ── State variables: MECHA's geometry ──
    cell_rank: float = state_variable(**_doc("", "MECHA's cell rank: the cgroup, or the cortex (40-49) and stele "
                                                 "(50-61) layer (-1: apoplast)"),
                                      initialize=-1., scale=SCALES.Compartment, location="node",
                                      state_variable_type="descriptor")
    perimeter: float = state_variable(**_doc("cm", "the section's perimeter: 2π × the epidermis walls' mean distance "
                                                   "to the endodermis' centre"),
                                      initialize=np.nan, scale=SCALES.SubOrgan, state_variable_type="descriptor")

    def __post_init__(self):
        super().__post_init__()
        self._geometries = {}                 # {anatomy key: (node count, SectionGeometry)}

    # ── Geometry, per distinct anatomy ──
    def section_geometries(self) -> dict:
        """
        {SubOrgan vid: (nodes, edges, SectionGeometry)} of every section with an anatomy: its Compartments and
        Connections (DataStructure indices) and MECHA's geometry, computed once per anatomy template.
        """
        ds = self.data_structure
        template = dict(zip(ds.entity_ids("SubOrgan").tolist(), np.asarray(ds.get("anatomy_template")).tolist()))
        out = {}
        for segment, (nodes, edges) in section_parts(ds).items():
            if nodes.size == 0:
                continue
            index = template.get(segment, -1.)
            key = ("template", int(index)) if index >= 0 else ("segment", segment)
            cached = self._geometries.get(key)
            if cached is None or cached[0] != (nodes.size, edges.size):
                cached = self._geometries[key] = ((nodes.size, edges.size),
                                                  section_geometry(section_of(ds, nodes, edges)))
            out[segment] = (nodes, edges, cached[1])
        return out

    @state(include_inactive=True)
    def _cell_rank(self, kind, anatomy_template) -> tuple[np.ndarray, str, np.ndarray]:
        """
        MECHA's cell ranks on the Compartments, and each section's perimeter. They depend on the sections' networks
        (their templates' geometry, cached), read whole from the DataStructure.
        """
        ds = self.data_structure
        rank = np.full(ds.n_nodes(), -1.)
        segments = ds.entity_ids("SubOrgan")
        perimeter = np.full(segments.size, np.nan)
        where = {int(v): i for i, v in enumerate(segments)}
        for segment, (nodes, _, geometry) in self.section_geometries().items():
            rank[nodes] = geometry.cell_rank
            perimeter[where[segment]] = geometry.perimeter
        return rank, "perimeter", perimeter
