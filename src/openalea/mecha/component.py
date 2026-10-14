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

from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, parameter, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig
from openalea.metafspm.solve.decorator import state

from openalea.mecha.conductance import (FPLXHEIGHT, Network, assemble, boundary_terms, cortex_parameters,
                                       edge_conductances, fplxheight_name, set_edge_ends)
from openalea.mecha.geometry import phi_thick_cells, section_geometry, section_of, section_parts
from openalea.mecha.postprocess import axial_conductances, radial_conductivity, soil_uptake
from openalea.mecha.solve import section_potentials

SCALES = ScalesConfig()


def _doc(unit, description, **values):
    return dict(unit=unit, unit_comment="", description=description, min_value=-1e12, max_value=1e12,
                value_comment="", references="MECHA (Couvreur et al. 2018)", DOI=["10.1104/pp.18.01006"], **values)


def _node(unit, description):
    return dict(**_doc(unit, description), by="Anatomy", scale=SCALES.Compartment, location="node",
                state_variable_type="descriptor")


def _p(unit, description, default):
    return parameter(**_doc(unit, description), by="MECHA", default=default)


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

    barrier: float = input_variable(**_doc("", "the segment's apoplastic barriers (MECHA's maturity code: 0 none, 1 "
                                                 "endodermal Casparian strip, 2 + suberin with passage cells, 3 "
                                                 "endodermis fully suberized, 4-9 exodermis variants)"),
                                    by="Development", initialize=1., scale=SCALES.SubOrgan, location="node",
                                    mapping="broadcast", state_variable_type="descriptor")
    length: float = input_variable(**_doc("m", "the segment's length (MECHA's section height)"),
                                   by="Architecture", scale=SCALES.SubOrgan, location="node",
                                   state_variable_type="intensive")

    # ── Parameters (per plant; MECHA's units: cm hPa-1 d-1 for kw, kmb, kaqp; cm3 hPa-1 d-1 for kpl, k_xyl) ──
    kw: float = _p("cm2 hPa-1 d-1", "cell wall hydraulic conductivity", 0.00024)
    kw_septa: float = _p("cm2 hPa-1 d-1", "conductivity of the septa between aerenchyma (intercellular) cells", 0.00012)
    kw_casparian: float = _p("cm2 hPa-1 d-1", "wall conductivity across a Casparian strip", 1.00E-16)
    kw_suberin_in: float = _p("cm2 hPa-1 d-1", "conductivity of suberized walls, inner face", 1.00E-16)
    kw_suberin_out: float = _p("cm2 hPa-1 d-1", "conductivity of suberized walls, outer face", 1.00E-16)
    kw_lignin: float = _p("cm2 hPa-1 d-1", "conductivity of lignified walls (lignin cap, phi thickenings)", 1.00E-16)
    thickness: float = _p("µm", "cell wall thickness (one value for every wall, as MECHA's)", 1.5)
    xylem_pieces: float = _p("", "1: the walls between two xylem cells are septa of near-zero conductance", 0.)
    kmb: float = _p("cm hPa-1 d-1", "membrane hydraulic conductivity, without aquaporins", 3.0E-5)
    kaqp: float = _p("cm hPa-1 d-1", "aquaporin contribution to the membrane conductivity", 0.000430)
    kaqp_stele_factor: float = _p("", "factor of kaqp in the stele", 1.0)
    kaqp_endo_factor: float = _p("", "factor of kaqp in the endodermis", 1.0)
    kaqp_exo_factor: float = _p("", "factor of kaqp in the exodermis", 1.0)
    kaqp_epi_factor: float = _p("", "factor of kaqp in the epidermis", 1.0)
    kaqp_cortex_factor: float = _p("", "factor of kaqp in the cortex", 1.0)
    ratio_cortex: float = _p("", "ratio of the cortex's outer to inner aquaporin activity (1: uniform)", 1.0)
    k_interc: float = _p("cm hPa-1 d-1", "membrane conductivity of intercellular spaces (with barriers)", 0.0)
    kpl: float = _p("cm3 hPa-1 d-1", "conductance of one plasmodesma", 5.3E-12)
    kpl_phloem_companion_cell_factor: float = _p("", "plasmodesmata factor at the companion-cell interfaces", 0.0)
    kpl_phloem_sieve_tube_factor: float = _p("", "plasmodesmata factor at the sieve-tube interfaces", 0.0)
    fplxheight_default: float = _p("cm-1", "plasmodesmata per unit wall length and height, other interfaces", 8.0E5)
    x_contact: float = _p("µm", "the border walls at x >= x_contact are in contact with the soil", -15E9)
    k_xyl: float = _p("cm3 hPa-1 d-1", "conductance between a xylem vessel and the xylem boundary", 1.0E-6)
    k_sieve: float = _p("cm3 hPa-1 d-1", "conductance between a protophloem cell and the phloem boundary", 1.0E-6)
    n_phi_layers: float = _p("", "number of cortex layers with phi thickenings", 0.)
    phi_type: float = _p("", "phi thickenings: 1 inner cortex, 2 outer cortex, 3 mid cortex", 1.)

    standard_solve: float = _p("", "1: solve MECHA's standard problem on each section alone (its kr; the legacy's "
                                   "definition), 0: skip it", 1.)
    standard_psi_soil: float = _p("hPa", "the soil's potential in the standard problem", 0.0)
    standard_psi_xylem: float = _p("hPa", "the xylem's potential in the standard problem (barrier > 0)", -5.0E3)
    standard_psi_sieve: float = _p("hPa", "the protophloem's potential in the standard problem (barrier 0; NaN: "
                                   "no phloem boundary, the legacy's default)", float("nan"))
    # plasmodesmata per unit wall length and height, by interface (cell groups' names)
    fplxheight_exo_exo: float = _p("cm-1", "plasmodesmata frequency, exo–exo interfaces", 800000.0)
    fplxheight_epi_epi: float = _p("cm-1", "plasmodesmata frequency, epi–epi interfaces", 800000.0)
    fplxheight_exo_epi: float = _p("cm-1", "plasmodesmata frequency, exo–epi interfaces", 1080000.0)
    fplxheight_exo_cortex: float = _p("cm-1", "plasmodesmata frequency, exo–cortex interfaces", 2280000.0)
    fplxheight_cortex_cortex: float = _p("cm-1", "plasmodesmata frequency, cortex–cortex interfaces", 860000.0)
    fplxheight_endo_cortex: float = _p("cm-1", "plasmodesmata frequency, endo–cortex interfaces", 880000.0)
    fplxheight_endo_endo: float = _p("cm-1", "plasmodesmata frequency, endo–endo interfaces", 640000.0)
    fplxheight_endo_peri: float = _p("cm-1", "plasmodesmata frequency, endo–peri interfaces", 960000.0)
    fplxheight_endo_stele: float = _p("cm-1", "plasmodesmata frequency, endo–stele interfaces", 960000.0)
    fplxheight_endo_tparenchyma: float = _p("cm-1", "plasmodesmata frequency, endo–tparenchyma interfaces", 1080000.0)
    fplxheight_endo_ttracheid: float = _p("cm-1", "plasmodesmata frequency, endo–ttracheid interfaces", 0.0)
    fplxheight_stele_peri: float = _p("cm-1", "plasmodesmata frequency, stele–peri interfaces", 1080000.0)
    fplxheight_stele_sieve: float = _p("cm-1", "plasmodesmata frequency, stele–sieve interfaces", 900000.0)
    fplxheight_stele_protophloem: float = _p("cm-1", "plasmodesmata frequency, stele–protophloem interfaces", 900000.0)
    fplxheight_stele_comp: float = _p("cm-1", "plasmodesmata frequency, stele–comp interfaces", 980000.0)
    fplxheight_stele_xylem: float = _p("cm-1", "plasmodesmata frequency, stele–xylem interfaces", 640000.0)
    fplxheight_stele_stele: float = _p("cm-1", "plasmodesmata frequency, stele–stele interfaces", 640000.0)
    fplxheight_stele_tparenchyma: float = _p("cm-1", "plasmodesmata frequency, stele–tparenchyma interfaces", 640000.0)
    fplxheight_stele_ttracheid: float = _p("cm-1", "plasmodesmata frequency, stele–ttracheid interfaces", 0.0)
    fplxheight_sieve_comp: float = _p("cm-1", "plasmodesmata frequency, sieve–comp interfaces", 1760000.0)
    fplxheight_sieve_peri: float = _p("cm-1", "plasmodesmata frequency, sieve–peri interfaces", 720000.0)
    fplxheight_sieve_xylem: float = _p("cm-1", "plasmodesmata frequency, sieve–xylem interfaces", 0.0)
    fplxheight_sieve_sieve: float = _p("cm-1", "plasmodesmata frequency, sieve–sieve interfaces", 0.0)
    fplxheight_sieve_protophloem: float = _p("cm-1", "plasmodesmata frequency, sieve–protophloem interfaces", 0.0)
    fplxheight_sieve_tparenchyma: float = _p("cm-1", "plasmodesmata frequency, sieve–tparenchyma interfaces", 640000.0)
    fplxheight_sieve_ttracheid: float = _p("cm-1", "plasmodesmata frequency, sieve–ttracheid interfaces", 0.0)
    fplxheight_comp_xylem: float = _p("cm-1", "plasmodesmata frequency, comp–xylem interfaces", 980000.0)
    fplxheight_comp_peri: float = _p("cm-1", "plasmodesmata frequency, comp–peri interfaces", 700000.0)
    fplxheight_comp_comp: float = _p("cm-1", "plasmodesmata frequency, comp–comp interfaces", 680000.0)
    fplxheight_comp_protophloem: float = _p("cm-1", "plasmodesmata frequency, comp–protophloem interfaces", 680000.0)
    fplxheight_comp_tparenchyma: float = _p("cm-1", "plasmodesmata frequency, comp–tparenchyma interfaces", 1080000.0)
    fplxheight_comp_ttracheid: float = _p("cm-1", "plasmodesmata frequency, comp–ttracheid interfaces", 0.0)
    fplxheight_xylem_peri: float = _p("cm-1", "plasmodesmata frequency, xylem–peri interfaces", 1080000.0)
    fplxheight_xylem_xylem: float = _p("cm-1", "plasmodesmata frequency, xylem–xylem interfaces", 640000.0)
    fplxheight_xylem_protophloem: float = _p("cm-1", "plasmodesmata frequency, xylem–protophloem interfaces", 0.0)
    fplxheight_xylem_tparenchyma: float = _p("cm-1", "plasmodesmata frequency, xylem–tparenchyma interfaces", 1080000.0)
    fplxheight_xylem_ttracheid: float = _p("cm-1", "plasmodesmata frequency, xylem–ttracheid interfaces", 1760000.0)
    fplxheight_tparenchyma_tparenchyma: float = _p("cm-1", "plasmodesmata frequency, tparenchyma–tparenchyma interfaces", 800000.0)
    fplxheight_tparenchyma_ttracheid: float = _p("cm-1", "plasmodesmata frequency, tparenchyma–ttracheid interfaces", 0.0)
    fplxheight_ttracheid_ttracheid: float = _p("cm-1", "plasmodesmata frequency, ttracheid–ttracheid interfaces", 0.0)
    fplxheight_protophloem_protophloem: float = _p("cm-1", "plasmodesmata frequency, protophloem–protophloem interfaces", 0.0)

    # ── State variables: MECHA's geometry ──
    cell_rank: float = state_variable(**_doc("", "MECHA's cell rank: the cgroup, or the cortex (40-49) and stele "
                                                 "(50-61) layer (-1: apoplast)"),
                                      initialize=-1., scale=SCALES.Compartment, location="node",
                                      state_variable_type="descriptor")
    perimeter: float = state_variable(**_doc("cm", "the section's perimeter: 2π × the epidermis walls' mean distance "
                                                   "to the endodermis' centre"),
                                      initialize=np.nan, scale=SCALES.SubOrgan, state_variable_type="descriptor")

    # ── State variables: the conductances ──
    conductance: float = state_variable(**_doc("cm3 hPa-1 d-1", "the edge's hydraulic conductance (walls, membranes, "
                                                                "plasmodesmata; 0 on the edges MECHA leaves out)"),
                                        initialize=0., scale=SCALES.Connection, location="edge",
                                        state_variable_type="NonInertialIntensive")
    soil_conductance: float = state_variable(**_doc("cm3 hPa-1 d-1", "conductance between a border wall or junction "
                                                                     "and the soil (0 elsewhere)"),
                                             initialize=0., scale=SCALES.Compartment, location="node",
                                             state_variable_type="NonInertialIntensive")
    axial_boundary_conductance: float = state_variable(**_doc("cm3 hPa-1 d-1", "conductance between a xylem vessel "
                                                              "(barrier > 0) or a protophloem cell (barrier 0) and "
                                                              "the axial boundary, on the matrix diagonal (0 elsewhere)"),
                                                       initialize=0., scale=SCALES.Compartment, location="node",
                                                       state_variable_type="NonInertialIntensive")

    # ── State variables: the standard problem's solution, and the sections' results ──
    psi: float = state_variable(**_doc("hPa", "water potential, in the standard problem of its section"),
                                initialize=np.nan, scale=SCALES.Compartment, location="node",
                                state_variable_type="NonInertialIntensive")
    flow: float = state_variable(**_doc("cm3 d-1", "water flow along the edge, from its first node to its second, in "
                                                   "the standard problem"),
                                 initialize=0., scale=SCALES.Connection, location="edge",
                                 state_variable_type="NonInertialIntensive")
    uptake: float = state_variable(**_doc("cm3 d-1", "the section's water uptake from the soil (standard problem)"),
                                   initialize=np.nan, scale=SCALES.SubOrgan, state_variable_type="NonInertialIntensive")
    kr: float = state_variable(**_doc("cm hPa-1 d-1", "the section's radial conductivity (standard problem): uptake / "
                                                      "(ψ soil − ψ xylem) / surface"),
                               initialize=np.nan, scale=SCALES.SubOrgan, state_variable_type="NonInertialIntensive")
    Kx: float = state_variable(**_doc("cm4 hPa-1 d-1", "the section's xylem axial conductance (Poiseuille; 0 without "
                                                       "mature xylem, barrier 0)"),
                               initialize=np.nan, scale=SCALES.SubOrgan, state_variable_type="NonInertialIntensive")
    Kx_phloem: float = state_variable(**_doc("cm4 hPa-1 d-1", "the section's phloem axial conductance (Poiseuille: "
                                                              "sieve tubes, or protophloem at barrier 0)"),
                                      initialize=np.nan, scale=SCALES.SubOrgan,
                                      state_variable_type="NonInertialIntensive")

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

    # ── Conductances ──
    PARAMETERS = ()                       # filled below: the names of the numeric parameters

    def parameter_table(self, location: str) -> dict:
        """{parameter: value per entity of *location*} (each entity its plant's value)."""
        return {name: np.asarray(self.parameter_values(name, location)) for name in self.PARAMETERS}

    def network(self, geometries: dict = None) -> Network:
        """The sections' MECHA geometry on the whole DataStructure, their φ-thickened cells included."""
        ds = self.data_structure
        geometries = self.section_geometries() if geometries is None else geometries
        items = list(geometries.values())
        layers = self.parameter_table("node")
        phi = [phi_thick_cells(geo, int(layers["n_phi_layers"][nodes[0]]), int(layers["phi_type"][nodes[0]]))
               for nodes, _, geo in items]
        net = assemble(ds.n_nodes(), ds.n_edges(), items, phi)
        view = ds.to_graph_view()
        set_edge_ends(net, view.tail, view.head, np.asarray(ds.get("network_path")))
        return net

    def cortex_gradient(self, geometries: dict, height_by_node, P_node) -> tuple:
        """Per node: (a_cortex, b_cortex) of its section, from its plant's parameters and its height."""
        a, b = np.zeros(self.data_structure.n_nodes()), np.zeros(self.data_structure.n_nodes())
        for nodes, _, geo in geometries.values():
            v = nodes[0]
            kaqp_cortex = float(P_node["kaqp"][v]) * float(P_node["kaqp_cortex_factor"][v])
            a[nodes], b[nodes] = cortex_parameters(geo, kaqp_cortex, float(P_node["ratio_cortex"][v]),
                                                   float(height_by_node[v]))
        return a, b

    @state(include_inactive=True)
    def _conductance(self, network_path, barrier, length, anatomy_template) \
            -> tuple[np.ndarray, str, np.ndarray, str, np.ndarray]:
        """
        Every edge's conductance, and the boundary conductances of the nodes (soil, axial), from MECHA's geometry,
        the segments' barriers and heights (their lengths), and the plants' parameters.
        """
        geometries = self.section_geometries()
        net = self.network(geometries)
        height_node = np.asarray(length, dtype=float) * 1.0E6                     # m -> µm
        P_node, P_edge = self.parameter_table("node"), self.parameter_table("edge")
        a_cortex, b_cortex = self.cortex_gradient(geometries, height_node, P_node)
        tail = np.maximum(net.a, 0)
        P_edge["a_cortex"], P_edge["b_cortex"] = a_cortex[tail], b_cortex[tail]
        barrier_node = np.asarray(barrier)
        K, in_matrix = edge_conductances(net, P_edge, barrier_node[tail], height_node[tail])
        soil, axial, _ = boundary_terms(net, P_node, barrier_node, height_node)
        return np.where(in_matrix, K, 0.), "soil_conductance", soil, "axial_boundary_conductance", axial


    @state(include_inactive=True)
    def _psi(self, conductance, soil_conductance, axial_boundary_conductance, barrier, length) \
            -> tuple[np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray]:
        """
        MECHA's standard problem, each section alone (Q-M5): its border walls and junctions at the soil's potential,
        its xylem vessels (barrier > 0) or protophloem cells (barrier 0) at the axial potential. Then the flows, each
        section's uptake and radial conductivity kr, and its axial conductances (Poiseuille). Optional
        (standard_solve = 0: none of them is computed).
        """
        ds = self.data_structure
        P = self.parameter_table("node")
        if not (P["standard_solve"] > 0).any():
            return None
        geometries = self.section_geometries()
        net = self.network(geometries)
        K, soil, axial = (np.asarray(v, dtype=float) for v in (conductance, soil_conductance,
                                                               axial_boundary_conductance))
        barrier, height = np.asarray(barrier), np.asarray(length, dtype=float) * 1.0E6
        psi, flow = np.full(ds.n_nodes(), np.nan), np.zeros(ds.n_edges())
        segments = ds.entity_ids("SubOrgan")
        results = {name: np.full(segments.size, np.nan) for name in ("uptake", "kr", "Kx", "Kx_phloem")}
        where = {int(v): i for i, v in enumerate(segments)}
        in_matrix = np.ones(K.size, dtype=bool)               # the edges MECHA leaves out have K = 0
        for segment, (nodes, edges, geo) in geometries.items():
            v = nodes[0]
            if not P["standard_solve"][v] > 0:
                continue
            b, h = int(barrier[v]), float(height[v])
            psi_soil = float(P["standard_psi_soil"][v])
            psi_axial = float(P["standard_psi_xylem"][v] if b > 0 else P["standard_psi_sieve"][v])
            k_axial = float(P["k_xyl"][v] if b > 0 else P["k_sieve"][v])
            if np.isnan(psi_axial):                           # no axial boundary: the soil's alone
                diag, rhs_axial, psi_axial_value = np.zeros_like(axial), np.zeros_like(axial), 0.
            else:
                diag, rhs_axial, psi_axial_value = axial, np.where(axial > 0, k_axial, 0.), psi_axial
            psi[nodes] = section_potentials(net, K, in_matrix, soil, diag, rhs_axial, nodes, edges, psi_soil,
                                            psi_axial_value)
            flow[edges] = K[edges] * (psi[net.a[edges]] - psi[net.b[edges]])
            i = where[segment]
            results["uptake"][i] = soil_uptake(geo, soil[nodes], psi[nodes], psi_soil)
            results["kr"][i] = radial_conductivity(results["uptake"][i], psi_soil, psi_axial, geo.perimeter, h)
            results["Kx"][i], results["Kx_phloem"][i] = axial_conductances(geo, h, b)
        return (psi, "flow", flow, "uptake", results["uptake"], "kr", results["kr"], "Kx", results["Kx"],
                "Kx_phloem", results["Kx_phloem"])


MechaHydraulics.PARAMETERS = tuple(MechaHydraulics._plant_parameter_fields())

