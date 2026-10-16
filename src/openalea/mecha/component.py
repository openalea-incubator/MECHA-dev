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
from scipy.sparse import diags

from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, parameter, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig
from openalea.metafspm.solve.decorator import (boundary_set, graph_jacobian, graph_output, graph_system,
                                                node_balance, rate, totalstate)

from openalea.mecha.conductance import (FPLXHEIGHT, Network, assemble, boundary_terms, cortex_parameters,
                                       edge_conductances, fplxheight_name, set_edge_ends)
from openalea.mecha.geometry import (AXIAL_JUNCTION, XYLEM_CGROUPS, phi_thick_cells, section_geometry,
                                     section_of, section_parts)
from openalea.mecha.postprocess import (VISCOSITY_PHLOEM, VISCOSITY_XYLEM, axial_conductances, poiseuille,
                                        radial_conductivity, soil_uptake)

PHLOEM_CGROUPS = (11, 23)
R, WATER_MOLAR_VOLUME, PRESSURE = 8.314, 1.8e-5, 101325.          # J mol-1 K-1, m3 mol-1, Pa
CELSIUS = 20.
AIR_PSI = R * (CELSIUS + 273.15) / WATER_MOLAR_VOLUME * np.log(0.5) * 1e-2      # hPa: 50 % RH at 20 °C
VAPOUR_FACTOR = (610.78 * np.exp(17.27 * CELSIUS / (CELSIUS + 237.3)) * WATER_MOLAR_VOLUME ** 2
                 / (R * (CELSIUS + 273.15) * PRESSURE) * 1e6 * 1e9)          # as metafspm's SPAC example
TO_MECHA = 1e-3 * 86400 * 1e-4         # mm3 s-1 MPa-1 -> cm3 d-1 hPa-1

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
    collar_condition: float = _p("", "at the first root section's xylem vessels: 1 the potential collar_psi, 2 the "
                                     "flow collar_flow, 0 none (e.g. a shoot above)", 1.)
    collar_psi: float = _p("hPa", "the collar's xylem potential (collar_condition 1)", -5.0E3)
    collar_flow: float = _p("cm3 d-1", "the flow leaving the root at the collar (collar_condition 2)", 0.)
    stomatal_conductance: float = _p("mol m-2 s-1", "stomatal conductance to water vapour (leaves' pores)", 0.2)
    vapour_factor: float = _p("mm3 s-1 MPa-1 per (mol m-2 s-1 m2)", "e_sat V_w² / (R T P): from a vapour conductance "
                              "to a liquid one (20 °C)", VAPOUR_FACTOR)
    standard_axial_condition: float = _p("", "the standard problem's xylem condition: 1 its potential "
                                         "(standard_psi_xylem, through k_xyl), 2 its flow (standard_axial_flow)", 1.)
    standard_axial_flow: float = _p("cm3 d-1", "the flow into the xylem in the standard problem (condition 2), "
                                    "split among the vessels as their areas (the legacy's distribution)", 0.)
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

    standard_soil_psi: float = state_variable(**_doc("hPa", "the soil's potential at the border walls and junctions in "
                                                            "the standard problem (its plant's standard_psi_soil)"),
                                              initialize=0., scale=SCALES.Compartment, location="node",
                                              state_variable_type="NonInertialIntensive")
    standard_axial_psi: float = state_variable(**_doc("hPa", "the axial boundary's potential at the xylem vessels "
                                                             "(barrier > 0) or protophloem cells (barrier 0) in the "
                                                             "standard problem"),
                                               initialize=0., scale=SCALES.Compartment, location="node",
                                               state_variable_type="NonInertialIntensive")
    standard_axial_inflow: float = state_variable(**_doc("cm3 d-1", "the water entering a xylem vessel from the axial "
                                                         "boundary in the standard problem (condition 2; negative: "
                                                         "the vessel's share of the flow into the xylem)"),
                                                  initialize=0., scale=SCALES.Compartment, location="node",
                                                  state_variable_type="NonInertialIntensive")
    in_standard_problem: float = state_variable(**_doc("", "1 on the Compartments of the plants solving the standard "
                                                           "problem (standard_solve)"),
                                                initialize=0., scale=SCALES.Compartment, location="node",
                                                state_variable_type="descriptor")

    # ── State variables: the axial junctions between sections ──
    axial_conduit: float = state_variable(**_doc("", "on the axial junctions: 1 between xylem vessels, 2 between "
                                                     "phloem cells (0 elsewhere)"),
                                          initialize=0., scale=SCALES.Connection, location="edge",
                                          state_variable_type="descriptor")
    xylem_axial_conductance: float = state_variable(**_doc("cm3 hPa-1 d-1", "a xylem junction's conductance: "
                                                           "Poiseuille through its two vessels, each over half its "
                                                           "segment, in series (0 without mature xylem)"),
                                                    initialize=0., scale=SCALES.Connection, location="edge",
                                                    state_variable_type="NonInertialIntensive")
    phloem_axial_conductance: float = state_variable(**_doc("cm3 hPa-1 d-1", "a phloem junction's conductance "
                                                            "(Poiseuille, phloem sap); not in the water problem (Q-G1)"),
                                                     initialize=0., scale=SCALES.Connection, location="edge",
                                                     state_variable_type="NonInertialIntensive")

    # ── Inputs and state variables of the plant problem ──
    soil_psi: float = input_variable(**_doc("hPa", "the soil's water potential around the Compartment"), by="Soil",
                                     initialize=0., scale=SCALES.Compartment, location="node",
                                     state_variable_type="intensive")
    air_psi: float = input_variable(**_doc("hPa", "the air's water potential around the leaves (default: 50 % relative "
                                                   "humidity at 20 °C)"), by="Atmosphere", initialize=AIR_PSI,
                                    scale=SCALES.Compartment, location="node", state_variable_type="intensive")
    evaporation_conductance: float = state_variable(**_doc("cm3 hPa-1 d-1", "liquid–vapour conductance of a stomatal "
                                                           "pore to the air: g_s × the leaf surface it serves × the "
                                                           "vapour factor (0 elsewhere)"),
                                                    initialize=0., scale=SCALES.Compartment, location="node",
                                                    state_variable_type="NonInertialIntensive")
    root_uptake: float = state_variable(**_doc("cm3 d-1", "water entering a border wall or junction from the soil "
                                                          "(plant problem)"),
                                        initialize=0., scale=SCALES.Compartment, location="node",
                                        state_variable_type="extensive")
    evaporation: float = state_variable(**_doc("cm3 d-1", "water leaving a stomatal pore to the air (plant problem)"),
                                        initialize=0., scale=SCALES.Compartment, location="node",
                                        state_variable_type="extensive")
    collar_kind: float = state_variable(**_doc("", "the collar condition of the node: 1 potential, 2 flow (0: none)"),
                                        initialize=0., scale=SCALES.Compartment, location="node",
                                        state_variable_type="descriptor")
    collar_value: float = state_variable(**_doc("hPa or cm3 d-1", "the collar's potential (kind 1), or the node's "
                                                "inflow (kind 2, negative: water leaving the root)"),
                                         initialize=0., scale=SCALES.Compartment, location="node",
                                         state_variable_type="NonInertialIntensive")

    # ── State variables: the solutions, and the sections' results ──
    standard_psi: float = state_variable(**_doc("hPa", "water potential, in its section's standard problem"),
                                         initialize=0., scale=SCALES.Compartment, location="node",
                                         state_variable_type="NonInertialIntensive")
    standard_flow: float = state_variable(**_doc("cm3 d-1", "flow along the edge, from its first node to its second, "
                                                            "in the standard problem"),
                                          initialize=0., scale=SCALES.Connection, location="edge",
                                          state_variable_type="NonInertialIntensive")
    psi: float = state_variable(**_doc("hPa", "water potential, in the plant problem"),
                                initialize=0., scale=SCALES.Compartment, location="node",
                                state_variable_type="NonInertialIntensive")
    flow: float = state_variable(**_doc("cm3 d-1", "flow along the edge, from its first node to its second, in the "
                                                   "plant problem"),
                                 initialize=0., scale=SCALES.Connection, location="edge",
                                 state_variable_type="NonInertialIntensive")
    standard_uptake: float = state_variable(**_doc("cm3 d-1", "the section's water uptake from the soil, standard "
                                                              "problem"),
                                            initialize=np.nan, scale=SCALES.SubOrgan,
                                            state_variable_type="NonInertialIntensive")
    uptake: float = state_variable(**_doc("cm3 d-1", "the section's water uptake from the soil, plant problem"),
                                   initialize=np.nan, scale=SCALES.SubOrgan, state_variable_type="NonInertialIntensive")
    kr: float = state_variable(**_doc("cm hPa-1 d-1", "the section's radial conductivity, MECHA's definition (standard "
                                                      "problem): uptake / (ψ soil − ψ xylem) / surface"),
                               initialize=np.nan, scale=SCALES.SubOrgan, state_variable_type="NonInertialIntensive")
    kr_effective: float = state_variable(**_doc("cm hPa-1 d-1", "the section's radial conductivity in the plant "
                                                                "problem: uptake / (ψ soil − mean ψ of its xylem) / "
                                                                "surface (NaN at barrier 0: no mature xylem)"),
                                         initialize=np.nan, scale=SCALES.SubOrgan,
                                         state_variable_type="NonInertialIntensive")
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

    def section_roles(self) -> tuple:
        """
        (first, roots): the SubOrgan vids without a parent (a root system's first section) and those labelled
        RootSegment; read from the MPG once per topology (reading it flushes the DataStructure's variables).
        """
        ds = self.data_structure
        cached = self.__dict__.get("_section_roles")
        if cached is None or cached[0] != ds.topology_version:
            g = ds.mtg
            labels, root = g.property("label"), g.labels.SubOrgan.RootSegment
            segments = [int(v) for v in ds.entity_ids("SubOrgan")]
            cached = self.__dict__["_section_roles"] = (ds.topology_version, (
                {v for v in segments if g.parent(v) is None}, {v for v in segments if labels.get(v) == root}))
        return cached[1]

    def _root_nodes(self, geometries) -> np.ndarray:
        """bool per node: in a root section."""
        roots = self.section_roles()[1]
        out = np.zeros(self.data_structure.n_nodes(), dtype=bool)
        for segment, (nodes, _, _) in geometries.items():
            out[nodes] = segment in roots
        return out

    @rate(include_inactive=True)
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
        roots = self.section_roles()[1]
        for segment, (nodes, _, geometry) in self.section_geometries().items():
            rank[nodes] = geometry.cell_rank
            if segment in roots:                                       # a root quantity (E1)
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

    @rate(include_inactive=True)
    def _conductance(self, network_path, barrier, length, anatomy_template) \
            -> tuple[np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray,
                     str, np.ndarray, str, np.ndarray]:
        """
        Every edge's conductance, and the boundary conductances of the nodes (soil, axial), from MECHA's geometry,
        the segments' barriers and heights (their lengths), and the plants' parameters; the standard problem's
        boundary potentials (the axial boundary only where its potential is given: not at barrier 0 by default).
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
        axial_psi = np.where(barrier_node > 0, P_node["standard_psi_xylem"], P_node["standard_psi_sieve"])
        axial = np.where(np.isnan(axial_psi), 0., axial)
        flow_condition = (P_node["standard_axial_condition"] == 2) & (barrier_node > 0)
        axial = np.where(flow_condition, 0., axial)                    # the flow condition replaces k_xyl's
        inflow = self._standard_axial_inflow(geometries, P_node, flow_condition)
        root = self._root_nodes(geometries)
        soil = np.where(root, soil, 0.)                               # stems and leaves: no soil (E1)
        evaporation = self._evaporation_conductance(geometries, height_node, P_node)
        return (np.where(in_matrix, K, 0.), "soil_conductance", soil, "axial_boundary_conductance", axial,
                "standard_soil_psi", P_node["standard_psi_soil"].astype(float),
                "standard_axial_psi", np.where(np.isnan(axial_psi), 0., axial_psi),
                "in_standard_problem", ((P_node["standard_solve"] > 0) & root).astype(float),
                "evaporation_conductance", evaporation, "standard_axial_inflow", inflow)

    def _standard_axial_inflow(self, geometries, P_node, flow_condition) -> np.ndarray:
        """
        The standard problem's flow condition (2): the flow into the xylem split among the vessels (the network's,
        by cgroup) as their areas. The legacy (``_distribute_xylem_flow``) computes the ratios on its cell manager's
        xylem (by tissue) and gives them by rank to the network's, so that a vessel may get another's (§2 o, fixed).
        """
        out = np.zeros(self.data_structure.n_nodes())
        for nodes, _, geo in geometries.values():
            v = nodes[0]
            if not flow_condition[v]:
                continue
            vessels = geo.xylem_cells
            total = 0.
            for c in vessels:
                total += float(geo.cell_area[c])
            for c in vessels:
                out[nodes[c]] = -float(P_node["standard_axial_flow"][v]) * (float(geo.cell_area[c]) / total)
        return out

    def _evaporation_conductance(self, geometries, height_node, P_node) -> np.ndarray:
        """
        Per stomatal pore (tissue 'pore', E3): g_s × A × the vapour factor, A the leaf surface it serves (the
        section's outer walls' length × its height, shared among its pores), in cm³ d⁻¹ hPa⁻¹.
        """
        out = np.zeros(self.data_structure.n_nodes())
        for nodes, _, geo in geometries.values():
            pores = geo.cells[geo.tissue[geo.cells] == "pore"]
            if pores.size == 0:
                continue
            outer = geo.walls[geo.border_wall[geo.walls] | geo.border_aerenchyma[geo.walls]]
            v = nodes[pores]
            area = geo.wall_length[outer].sum() * height_node[v] * 1e-12 / pores.size        # m²
            out[v] = P_node["stomatal_conductance"][v] * area * P_node["vapour_factor"][v] * TO_MECHA
        return out

    @rate(include_inactive=True)
    def _axial_conduit(self, network_path, cgroup) -> np.ndarray:
        """The axial junctions' conduit: 1 between two xylem vessels, 2 between two phloem cells."""
        view = self.data_structure.to_graph_view()
        junction = np.asarray(network_path) == AXIAL_JUNCTION
        cg = np.asarray(cgroup)
        a, b = cg[view.tail], cg[view.head]
        xylem = np.isin(a, XYLEM_CGROUPS) & np.isin(b, XYLEM_CGROUPS)
        phloem = np.isin(a, PHLOEM_CGROUPS) & np.isin(b, PHLOEM_CGROUPS)
        return np.select([junction & xylem, junction & phloem], [1., 2.], 0.)

    def _junction_conductance(self, conduit, cell_area, length, viscosity, barrier=None) -> np.ndarray:
        """Poiseuille through each junction of *conduit* (its two cells, each over half its segment, in series)."""
        ds = self.data_structure
        view = ds.to_graph_view()
        edges = np.flatnonzero(np.asarray(ds.get("axial_conduit")) == conduit)
        a, b = view.tail[edges], view.head[edges]
        area = np.asarray(cell_area, dtype=float) * 1.0E6                          # mm2 -> µm2
        half = np.asarray(length, dtype=float) * 1.0E6 / 2                         # m -> µm
        k_a, k_b = (poiseuille(area[n], half[n], viscosity) for n in (a, b))
        with np.errstate(divide="ignore"):
            K = np.where((k_a > 0) & (k_b > 0), 1 / (1 / k_a + 1 / k_b), 0.)
        if barrier is not None:
            barrier = np.asarray(barrier)
            K = np.where((barrier[a] > 0) & (barrier[b] > 0), K, 0.)
        return K

    @rate(filters={"axial_conduit": 1}, include_inactive=True)
    def _xylem_axial_conductance(self, axial_conduit, cell_area, length, barrier) -> np.ndarray:
        """The xylem junctions' conductances (water); none where either section has no mature xylem (barrier 0)."""
        return self._junction_conductance(1, cell_area, length, VISCOSITY_XYLEM, barrier)

    @rate(filters={"axial_conduit": 2}, include_inactive=True)
    def _phloem_axial_conductance(self, axial_conduit, cell_area, length) -> np.ndarray:
        """The phloem junctions' conductances (phloem sap)."""
        return self._junction_conductance(2, cell_area, length, VISCOSITY_PHLOEM)

    @rate(include_inactive=True)
    def _collar_kind(self, kind, cgroup, cell_area) -> tuple[np.ndarray, str, np.ndarray]:
        """
        The collar condition (Q-G2) on the xylem vessels of each root system's first section (a segment without a
        parent): the collar's potential (collar_condition 1), or the flow collar_flow leaving the root, split among
        the vessels as their Poiseuille conductances (∝ area², collar_condition 2).
        """
        ds = self.data_structure
        P = self.parameter_table("node")
        kinds, values = np.zeros(ds.n_nodes()), np.zeros(ds.n_nodes())
        segments = ds.entity_ids("SubOrgan")
        owner = ds.owner("SubOrgan")
        xylem = np.isin(np.asarray(cgroup), XYLEM_CGROUPS)
        area = np.asarray(cell_area, dtype=float)
        first = self.section_roles()[0]
        for i, segment in enumerate(segments):
            if int(segment) not in first:
                continue
            vessels = np.flatnonzero((owner == i) & xylem)
            if vessels.size == 0:
                continue
            condition = int(P["collar_condition"][vessels[0]])
            if condition == 1:
                kinds[vessels], values[vessels] = 1., P["collar_psi"][vessels]
            elif condition == 2:
                share = area[vessels] ** 2 / (area[vessels] ** 2).sum()
                kinds[vessels], values[vessels] = 2., -P["collar_flow"][vessels] * share
        return kinds, "collar_value", values

    # ── The standard problem: each section alone, MECHA's definition of kr (Q-M5) ──
    @graph_system(node_unknowns=["standard_psi"], solver="newton", max_iter=3, schedule_as="state",
                  filters={"in_standard_problem": ">0"})
    class _standard_problem:
        """
        Each section alone (the junctions have no conductance here): its border walls and junctions exchange with
        the soil, its xylem vessels (barrier > 0) or protophloem cells (barrier 0) with the axial boundary.
        """

        @node_balance(field="standard_psi")
        def _balance(self, standard_psi, conductance):
            B = self._graph_view.incidence
            return np.asarray(B @ (conductance * np.asarray(B.T @ standard_psi).reshape(-1))).reshape(-1)

        soil = boundary_set(filters={"soil_conductance": ">0"}, kind="robin", value="standard_soil_psi",
                            weight="soil_conductance", field="standard_psi")
        axial = boundary_set(filters={"axial_boundary_conductance": ">0"}, kind="robin", value="standard_axial_psi",
                             weight="axial_boundary_conductance", field="standard_psi")
        axial_flow = boundary_set(filters={"standard_axial_inflow": "<0"}, kind="neumann", value="standard_axial_inflow",
                                  field="standard_psi")

        @graph_jacobian
        def _jacobian(self, conductance):
            B = self._graph_view.incidence
            return (B @ diags(conductance) @ B.T).tocsr()

        @graph_output("standard_flow", location="edge")
        def _standard_flow(self, standard_psi, conductance):
            return conductance * np.asarray(self._graph_view.incidence.T @ standard_psi).reshape(-1)

    # ── The plant problem: the sections joined by their xylem junctions ──
    @graph_system(node_unknowns=["psi"], solver="newton", max_iter=3, schedule_as="state")
    class _plant_problem:
        """
        The whole plant: every section's edges and the xylem junctions between sections; the soil around the root
        surface (soil_psi), the collar's condition on the first root section's vessels, the air at the leaves'
        stomatal pores (air_psi).
        """

        @node_balance(field="psi")
        def _balance(self, psi, conductance, xylem_axial_conductance):
            B = self._graph_view.incidence
            K = conductance + xylem_axial_conductance
            return np.asarray(B @ (K * np.asarray(B.T @ psi).reshape(-1))).reshape(-1)

        soil = boundary_set(filters={"soil_conductance": ">0"}, kind="robin", value="soil_psi",
                            weight="soil_conductance", field="psi")
        collar = boundary_set(filters={"collar_kind": ">0"}, kinds="collar_kind", value="collar_value", weight=0.,
                              field="psi")
        air = boundary_set(filters={"evaporation_conductance": ">0"}, kind="robin", value="air_psi",
                           weight="evaporation_conductance", field="psi")

        @graph_jacobian
        def _jacobian(self, conductance, xylem_axial_conductance):
            B = self._graph_view.incidence
            return (B @ diags(conductance + xylem_axial_conductance) @ B.T).tocsr()

        @graph_output("flow", location="edge")
        def _flow(self, psi, conductance, xylem_axial_conductance):
            return (conductance + xylem_axial_conductance) * np.asarray(self._graph_view.incidence.T @ psi).reshape(-1)

        @graph_output("root_uptake", location="node")
        def _root_uptake(self, psi, soil_conductance, soil_psi):
            return soil_conductance * (soil_psi - psi)

        @graph_output("evaporation", location="node")
        def _evaporation(self, psi, evaporation_conductance, air_psi):
            return evaporation_conductance * (psi - air_psi)

    # ── The sections' results ──
    @totalstate(include_inactive=True)
    def _kr(self, standard_psi, psi, soil_conductance, soil_psi, barrier, length) \
            -> tuple[np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray]:
        """
        Each section's uptake and radial conductivity: MECHA's (its standard problem, the legacy's operations) and
        the effective ones of the plant problem; its axial conductances (Poiseuille).
        """
        ds = self.data_structure
        P = self.parameter_table("node")
        soil, standard, plant = (np.asarray(v, dtype=float) for v in (soil_conductance, standard_psi, psi))
        soil_psi = np.asarray(soil_psi, dtype=float)
        barrier, height = np.asarray(barrier), np.asarray(length, dtype=float) * 1.0E6
        segments = ds.entity_ids("SubOrgan")
        names = ("kr", "standard_uptake", "uptake", "kr_effective", "Kx", "Kx_phloem")
        out = {name: np.full(segments.size, np.nan) for name in names}
        where = {int(v): i for i, v in enumerate(segments)}
        cgroup = np.asarray(ds.get("cgroup"))
        roots = self.section_roles()[1]
        for segment, (nodes, edges, geo) in self.section_geometries().items():
            i, v = where[segment], nodes[0]
            b, h = int(barrier[v]), float(height[v])
            out["Kx"][i], out["Kx_phloem"][i] = axial_conductances(geo, h, b)
            if segment not in roots:                                  # kr, uptake: root quantities (E1)
                continue
            if P["standard_solve"][v] > 0:
                psi_soil = float(P["standard_psi_soil"][v])
                psi_axial = float(P["standard_psi_xylem"][v] if b > 0 else P["standard_psi_sieve"][v])
                if b > 0 and P["standard_axial_condition"][v] == 2:      # the xylem's potential: its vessels' mean
                    vessels = geo.xylem_cells
                    psi_axial = 0.0
                    for c in vessels:
                        psi_axial += standard[nodes[c]] / len(vessels)
                    psi_axial = float(psi_axial)
                out["standard_uptake"][i] = soil_uptake(geo, soil[nodes], standard[nodes], psi_soil)
                out["kr"][i] = radial_conductivity(out["standard_uptake"][i], psi_soil, psi_axial, geo.perimeter, h)
            contact = nodes[soil[nodes] > 0]
            out["uptake"][i] = float(np.sum(soil[contact] * (soil_psi[contact] - plant[contact])))
            vessels = nodes[np.isin(cgroup[nodes], XYLEM_CGROUPS)]
            if contact.size and vessels.size and b > 0:                # no mature xylem at barrier 0 (D3)
                out["kr_effective"][i] = radial_conductivity(out["uptake"][i], soil_psi[contact].mean(),
                                                             plant[vessels].mean(), geo.perimeter, h)
        return tuple(x for name in names for x in ((out[name],) if name == "kr" else (name, out[name])))


MechaHydraulics.PARAMETERS = tuple(MechaHydraulics._plant_parameter_fields())

