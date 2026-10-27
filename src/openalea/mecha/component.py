"""
MECHA as a metafspm functional component on the MPG DataStructure (doc/equations.rst, doc/legacy.rst).

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

from openalea.mecha.conductance import (Network, assemble, boundary_terms, cortex_parameters, edge_conductances,
                                       elongation_sink, kelvin_alpha, set_edge_ends, set_wall_thickness,
                                       sieve_tube_terms, vapour_conductances)
from openalea.mecha.geometry import (AXIAL_JUNCTION, LEAF_SURFACE, MEMBRANE, STOMATAL, XYLEM_CGROUPS, phi_thick_cells,
                                     section_geometry, section_of, section_parts)
from openalea.mecha.postprocess import (VISCOSITY_PHLOEM, VISCOSITY_XYLEM, axial_conductances, poiseuille,
                                        radial_conductivity, soil_uptake)

PHLOEM_CGROUPS = (11, 23)

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


def _node_thickness(net, P_node):
    """Per node, the thickness its volume takes (µm): thickness_source 1 the plant's one value, 2 the node's own."""
    own = np.asarray(P_node["thickness_source"]) == 2
    return np.where(own & np.isfinite(net.wall_thickness), net.wall_thickness, np.asarray(P_node["thickness"], float))


def _wall_air_flow(B, psi, tail_conductance, head_conductance, offset):
    """The wall_air edges' flows from tail to head: a ψ_tail − b ψ_head + c (0 on the other edges)."""
    psi = np.asarray(psi, dtype=float)
    tail = np.asarray(B.maximum(0).T @ psi).reshape(-1)
    head = np.asarray((-B).maximum(0).T @ psi).reshape(-1)
    return tail_conductance * tail - head_conductance * head + offset


def _wall_air_jacobian(B, tail_conductance, head_conductance):
    """The wall_air flows' derivative in the balances: B diag(a) B_tail^T − B diag(b) B_head^T."""
    return B @ diags(tail_conductance) @ B.maximum(0).T - B @ diags(head_conductance) @ (-B).maximum(0).T


@dataclass
class MechaSections(FunctionalComponent):
    """
    The sections MECHA's components work on: their networks (role "Anatomy": GRANAP's anatomy component; its nodes,
    protoplasts, walls and wall junctions, and edges, membranes, walls and plasmodesmata, in GRANAP's units, mm and
    mm2), their barriers and heights, and MECHA's geometry derived from them, once per distinct anatomy.
    """
    # ── Inputs: the sections' networks ──
    kind: float = input_variable(**_node("", "0 protoplast, 1 wall, 2 wall junction"))
    cgroup: float = input_variable(**_node("", "the protoplast's MECHA cell group (NaN: apoplast)"))
    tissue: object = input_variable(**_node("", "the protoplast's tissue (None: a wall or a junction)"))
    section_x: float = input_variable(**_node("mm", "position in the section"))
    section_y: float = input_variable(**_node("mm", "position in the section"))
    wall_length: float = input_variable(**_node("mm", "the wall's length in the section (0: not a wall)"))
    cell_area: float = input_variable(**_node("mm2", "the protoplast's area in the section (0: apoplast)"))
    wall_thickness: float = input_variable(**_node("µm", "the wall's thickness, its two cells' layers (NaN: not a wall; "
                                                         "thickness_source 2)"))
    pore_width: float = input_variable(**_node("mm", "a stomatal pore's aperture width (NaN: not a pore)"))
    pore_depth: float = input_variable(**_node("mm", "a stomatal pore's depth through the epidermis (NaN: not a pore)"))
    network_path: float = input_variable(**_edge("", "edge type: membrane 0, wall_air 1, wall 2, plasmodesmata 3, "
                                                     "air_link 4, axial junction 5"))
    edge_length: float = input_variable(**_edge("mm", "the edge's length"))
    edge_dist: float = input_variable(**_edge("mm", "the distance between the edge's nodes"))
    anatomy_template: float = input_variable(**_doc("", "the segment's anatomy template (-1: none): sections of the "
                                                        "same template share MECHA's geometry"),
                                             by="Anatomy", scale=SCALES.SubOrgan, state_variable_type="descriptor")

    barrier: float = input_variable(**_doc("", "the segment's apoplastic barriers (MECHA's maturity code: 0 none, 1 "
                                                 "endodermal Casparian strip, 2 + suberin with passage cells, 3 "
                                                 "endodermis fully suberized, 4-9 exodermis variants, 10 exodermis "
                                                 "suberized with a Casparian strip, 11 endodermis and exodermis "
                                                 "suberized and the sclerenchyma's walls lignified)"),
                                    by="Development", initialize=1., scale=SCALES.SubOrgan, location="node",
                                    mapping="broadcast", state_variable_type="descriptor")
    length: float = input_variable(**_doc("m", "the segment's length (MECHA's section height)"),
                                   by="Architecture", scale=SCALES.SubOrgan, location="node",
                                   state_variable_type="intensive")

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
            cache = self.kept("geometries")                          # {anatomy key: (sizes, SectionGeometry)}
            if cache is None:
                cache = {}
                self.keep("geometries", cache, saved=False)          # rebuilt when missing: not checkpointed
            cached = cache.get(key)
            if cached is None or cached[0] != (nodes.size, edges.size):
                cached = cache[key] = ((nodes.size, edges.size),
                                                  section_geometry(section_of(ds, nodes, edges)))
            out[segment] = (nodes, edges, cached[1])
        return out

    def section_roles(self) -> tuple:
        """
        (first, roots, leaves): the SubOrgan vids without a parent (a root system's first section), those labelled
        RootSegment and those labelled LeafElement; read from the MPG once per topology.
        """
        def build():
            ds = self.data_structure
            g = ds.structure
            segments = [int(v) for v in ds.entity_ids("SubOrgan")]
            labels = ds.labels(of="SubOrgan", per_node=False).tolist()
            return ({v for v in segments if g.parent(v) is None},
                    {v for v, label in zip(segments, labels) if label == "RootSegment"},
                    {v for v, label in zip(segments, labels) if label == "LeafElement"})
        return self.cached("section_roles", build)

    def _leaf_nodes(self, geometries) -> np.ndarray:
        """bool per node: in a leaf section."""
        leaves = self.section_roles()[2]
        out = np.zeros(self.data_structure.n_nodes(), dtype=bool)
        for segment, (nodes, _, _) in geometries.items():
            out[nodes] = segment in leaves
        return out

    def _root_nodes(self, geometries) -> np.ndarray:
        """bool per node: in a root section."""
        roots = self.section_roles()[1]
        out = np.zeros(self.data_structure.n_nodes(), dtype=bool)
        for segment, (nodes, _, _) in geometries.items():
            out[nodes] = segment in roots
        return out


@dataclass
class MechaHydraulics(MechaSections):
    """
    MECHA's cross-section hydraulics on every section of the DataStructure (its sections: MechaSections).
    """
    # ── Inputs: the osmotic potentials (role "Solutes": a metabolism model, or MECHA's presets, MechaOsmoticPresets) ──
    osmotic_potential: float = input_variable(**_doc("hPa", "osmotic potential of the protoplast or the wall's solution "
                                                            "(0: none)"),
                                              by="Solutes", initialize=0., scale=SCALES.Compartment, location="node",
                                              state_variable_type="NonInertialIntensive")
    reflection_coefficient: float = input_variable(**_doc("", "the membrane's reflection coefficient (0: none)"),
                                                   by="Solutes", initialize=0., scale=SCALES.Connection,
                                                   location="edge", state_variable_type="NonInertialIntensive")
    soil_contact_cell: float = input_variable(**_doc("", "1 on a protoplast whose border walls touch the soil, whatever "
                                                         "x_contact (MECHA's Contactrange: a root partly in contact)"),
                                              by="Soil", initialize=0., scale=SCALES.Compartment, location="node",
                                              state_variable_type="descriptor")
    # ── Parameters (per plant; MECHA's units: cm hPa-1 d-1 for kw, kmb, kaqp; cm3 hPa-1 d-1 for kpl, k_xyl) ──
    kw: float = _p("cm2 hPa-1 d-1", "cell wall hydraulic conductivity", 0.00024)
    kw_septa: float = _p("cm2 hPa-1 d-1", "conductivity of the septa between aerenchyma (intercellular) cells", 0.00012)
    kw_casparian: float = _p("cm2 hPa-1 d-1", "wall conductivity across a Casparian strip", 1.00E-16)
    kw_suberin_in: float = _p("cm2 hPa-1 d-1", "conductivity of suberized walls, inner face", 1.00E-16)
    kw_suberin_out: float = _p("cm2 hPa-1 d-1", "conductivity of suberized walls, outer face", 1.00E-16)
    kw_lignin: float = _p("cm2 hPa-1 d-1", "conductivity of lignified walls (lignin cap, phi thickenings)", 1.00E-16)
    thickness_source: float = _p("", "the walls' thickness: 1 one value for every wall (thickness, MECHA's), 2 each "
                                     "wall's own (wall_thickness, GRANAP's; a junction's: the mean of its walls')", 2.)
    thickness: float = _p("µm", "cell wall thickness, every wall's (thickness_source 1, MECHA's)", 1.5)
    xylem_pieces: float = _p("", "1: the walls between two xylem cells are septa of near-zero conductance", 0.)
    kmb: float = _p("cm hPa-1 d-1", "membrane hydraulic conductivity, without aquaporins", 3.0E-5)
    kaqp: float = _p("cm hPa-1 d-1", "aquaporin contribution to the membrane conductivity", 0.000430)
    kaqp_stele_factor: float = _p("adim", "factor of kaqp in the stele", 1.0)
    kaqp_endo_factor: float = _p("adim", "factor of kaqp in the endodermis", 1.0)
    kaqp_exo_factor: float = _p("adim", "factor of kaqp in the exodermis", 1.0)
    kaqp_epi_factor: float = _p("adim", "factor of kaqp in the epidermis", 1.0)
    kaqp_cortex_factor: float = _p("adim", "factor of kaqp in the cortex", 1.0)
    ratio_cortex: float = _p("adim", "the cortex's aquaporin activity at its innermost walls over its outermost "
                                 "(1: uniform)", 1.0)
    k_interc: float = _p("cm hPa-1 d-1", "membrane conductivity of intercellular spaces (with barriers)", 0.0)
    pd_spatial_profile: float = _p("", "the plasmodesmata's aperture by their distance to the soil contact (develop's "
                                       "pd_spatial_config): 0 none, 1 linear, 2 exponential, from pd_spatial_c_wet "
                                       "at the nearest to pd_spatial_c_dry", 0.)
    pd_spatial_c_wet: float = _p("adim", "the plasmodesmata's aperture coefficient nearest the soil contact", 1.0)
    pd_spatial_c_dry: float = _p("adim", "the plasmodesmata's aperture coefficient farthest from the soil contact "
                                         "(and where not connected to it)", 0.6)
    pd_spatial_length_scale: float = _p("µm", "the exponential profile's length (NaN: a third of the plasmodesmata's "
                                              "distance range)", float("nan"))
    tracheid_as_xylem: float = _p("", "1: the transfusion tracheids' (cgroup 18) membranes are the xylem's (kaqp_stele "
                                      "x 1e4 with barriers, the vessels' apoplastic solute term), as water-filled "
                                      "conduits, without the xylem's boundary (develop's); 0: a stele cell's", 0.)
    membrane_floor: float = _p("", "a membrane without aquaporin activity outside the barriers (an air-filled "
                                   "intercellular space): 1 cut (1e-16, the original MECHA's), 2 cut only where kmb is "
                                   "0 too (the reactions branch's and develop's, for the comparisons)", 1.)
    kpl: float = _p("cm3 hPa-1 d-1", "conductance of one plasmodesma", 5.3E-12)
    plasmodesmata_rules: float = _p("", "the plasmodesmata's rules: 1 the original MECHA's, 2 the refactored legacy's "
                                        "(develop's factor map; for the comparison with it, doc/legacy.rst)", 1.)
    kpl_cortex_factor: float = _p("adim", "plasmodesmata aperture factor of the cortex (outer layer-cortex, "
                                          "cortex-cortex, cortex-endodermis)", 1.0)
    kpl_endo_in_factor: float = _p("adim", "plasmodesmata factor on the stelar side of the endodermis", 1.0)
    kpl_endo_out_factor: float = _p("adim", "plasmodesmata factor on the cortical side of the endodermis", 1.0)
    kpl_PPP_factor: float = _p("adim", "plasmodesmata aperture factor of the phloem-pole pericycle", 1.0)
    kpl_PCC_factor: float = _p("adim", "plasmodesmata aperture factor of the companion cells", 1.0)
    kpl_PST_factor: float = _p("adim", "plasmodesmata aperture factor of the sieve tubes (the legacy's rules only; the "
                                   "original uses it for the axial plasmodesmata of 3D sections)", 1.0)
    kpl_cambium_stele_factor: float = _p("adim", "plasmodesmata aperture factor between the stele and the cambium (the "
                                                 "legacy's rules only)", 1.0)
    x_contact: float = _p("µm", "the border walls at x >= x_contact are in contact with the soil", -15E9)
    contact_junction_rule: float = _p("", "a border junction between protoplasts in contact (soil_contact_cell) is in "
                                          "contact: 1 when all its cells are (develop's), 2 when any is (the reactions "
                                          "branch's)", 1.)
    standard_axial_source: float = _p("", "the standard problem's conductance between a conduit and the axial "
                                          "boundary: 1 its Poiseuille conductance over the height (the original "
                                          "MECHA's), 2 k_xyl and k_sieve (the legacy's)", 1.)
    k_xyl: float = _p("cm3 hPa-1 d-1", "conductance between a xylem vessel and the xylem boundary "
                                       "(standard_axial_source 2)", 1.0E-6)
    k_sieve: float = _p("cm3 hPa-1 d-1", "conductance between a protophloem cell and the phloem boundary "
                                         "(standard_axial_source 2)", 1.0E-6)
    n_phi_layers: float = _p("", "number of cortex layers with phi thickenings", 0.)
    phi_type: float = _p("", "phi thickenings: 1 inner cortex, 2 outer cortex, 3 mid cortex", 1.)

    standard_solve: float = _p("", "1: solve MECHA's standard problem on each section alone (its kr; the legacy's "
                                   "definition), 0: skip it", 1.)
    standard_osmotic: float = _p("", "1: the standard problem with the membranes' osmotic terms (the legacy's scenarios "
                                     "≥ 1), 0: without (its scenario 0, MECHA's kr)", 0.)
    collar_condition: float = _p("", "at the first root section's xylem vessels: 1 the potential collar_psi, 2 the "
                                     "flow collar_flow, 0 none (e.g. a shoot above)", 1.)
    collar_psi: float = _p("hPa", "the collar's xylem potential (collar_condition 1)", -5.0E3)
    collar_flow: float = _p("cm3 d-1", "the flow leaving the root at the collar (collar_condition 2)", 0.)
    pd_section: float = _p("µm2", "a plasmodesma's open cross-section (the solutes' symplastic diffusion)", 1.92E-3)
    kw_meso_hypo: float = _p("cm2 hPa-1 d-1", "the conductivity of the (2, 4) walls, whose plasmodesmata it also cuts "
                                              "(the reactions' needle; NaN: none)", float("nan"))
    vapour_diffusivity: float = _p("m2 s-1", "water vapour's diffusivity in air (the leaves' vapour path)", 2.4E-5)
    leaf_temperature: float = _p("°C", "the leaves' temperature (saturation vapour pressure, Kelvin's equation)", 20.)
    psi_ref_wall_air: float = _p("hPa", "the potential the walls' evaporation (Kelvin) is linearised at", 0.0)
    pore_length: float = _p("µm", "a stomatal pore's length along the leaf (its aperture's other side)", 20.)
    stomatal_spacing: float = _p("µm", "the distance between successive stomata of a file along the leaf: a section's "
                                       "pore stands for the segment's length / stomatal_spacing of them", 100.)
    standard_axial_condition: float = _p("", "the standard problem's xylem condition: 1 its potential "
                                         "(standard_psi_xylem, see standard_axial_source), 2 its flow (standard_axial_flow)", 1.)
    standard_axial_flow: float = _p("cm3 d-1", "the flow into the xylem in the standard problem (condition 2), "
                                    "split among the vessels as their areas (the legacy's distribution)", 0.)
    standard_psi_soil: float = _p("hPa", "the soil's potential in the standard problem (on the left of the section, "
                                         "x_rel 0, with standard_psi_soil_right)", 0.0)
    standard_psi_soil_right: float = _p("hPa", "the soil's potential on the right of the section (x_rel 1) in the "
                                               "standard problem, linear in x between the two (the legacy's "
                                               "psi_soil_right; NaN: standard_psi_soil all around)", float("nan"))
    standard_psi_xylem: float = _p("hPa", "the xylem's potential in the standard problem (barrier > 0)", -5.0E3)
    standard_psi_sieve: float = _p("hPa", "the protophloem's potential in the standard problem (barrier 0; NaN: "
                                   "no phloem boundary, the legacy's default)", float("nan"))
    standard_psi_sieve_tube: float = _p("hPa", "the sieve tubes' potential in the standard problem (barrier > 0: the "
                                               "legacy's phloem pressure of its scenarios >= 1, with k_sieve or "
                                               "Poiseuille as standard_axial_source; NaN: no phloem boundary)",
                                        float("nan"))
    elongation_midpoint_rate: float = _p("cm d-1", "the cells' elongation rate at the section's middle (barrier 0, "
                                                   "the elongation zone; MECHA's BCs.xml 2.8 for maize, Pritchard et "
                                                   "al. 1988; 0: none)", 0.)
    elongation_side_rate_difference: float = _p("cm d-1", "the elongation rate's difference from the section's left "
                                                          "to its right (positive: faster on the right)", 0.)
    water_fraction_apoplast: float = _p("adim", "the walls' volumetric water content (the elongation's water)", 0.69)
    water_fraction_symplast: float = _p("adim", "the protoplasts' volumetric water content (the elongation's "
                                                "water)", 0.7)
    standard_elongation: float = _p("", "1: the standard problem with the elongation sink (the legacy's scenarios "
                                        ">= 1), 0: without (its scenario 0, MECHA's kr)", 0.)
    # plasmodesmata frequency × height (cm-1; × a wall's length: its plasmodesmata), by interface: the original's,
    # then those of the tissues the legacy added, with its values (the original rules)
    fplxheight_default: float = _p("cm-1", "plasmodesmata frequency, other interfaces", 800000.0)
    fplxheight_epi_exo: float = _p("cm-1", "plasmodesmata frequency, epidermis-exodermis", 1080000.0)
    fplxheight_outer_cortex: float = _p("cm-1", "plasmodesmata frequency, outer layer-cortex", 2280000.0)
    fplxheight_cortex_cortex: float = _p("cm-1", "plasmodesmata frequency, cortex-cortex", 860000.0)
    fplxheight_cortex_endo: float = _p("cm-1", "plasmodesmata frequency, cortex-endodermis", 880000.0)
    fplxheight_endo_endo: float = _p("cm-1", "plasmodesmata frequency, endodermis-endodermis", 640000.0)
    fplxheight_endo_peri: float = _p("cm-1", "plasmodesmata frequency, endodermis-pericycle", 960000.0)
    fplxheight_peri_stele: float = _p("cm-1", "plasmodesmata frequency, pericycle-stele", 1080000.0)
    fplxheight_stele_stele: float = _p("cm-1", "plasmodesmata frequency, stele-stele", 640000.0)
    fplxheight_stele_comp: float = _p("cm-1", "plasmodesmata frequency, stele-companion cell", 980000.0)
    fplxheight_peri_comp: float = _p("cm-1", "plasmodesmata frequency, pericycle-companion cell", 700000.0)
    fplxheight_comp_comp: float = _p("cm-1", "plasmodesmata frequency, companion cells", 680000.0)
    fplxheight_comp_sieve: float = _p("cm-1", "plasmodesmata frequency, companion cell-sieve tube", 1760000.0)
    fplxheight_peri_sieve: float = _p("cm-1", "plasmodesmata frequency, pericycle-sieve tube", 720000.0)
    fplxheight_stele_sieve: float = _p("cm-1", "plasmodesmata frequency, stele-sieve tube", 900000.0)
    fplxheight_endo_tparenchyma: float = _p("cm-1", "plasmodesmata frequency, endo-tparenchyma", 1080000.0)
    fplxheight_endo_ttracheid: float = _p("cm-1", "plasmodesmata frequency, endo-ttracheid", 0.0)
    fplxheight_stele_protophloem: float = _p("cm-1", "plasmodesmata frequency, stele-protophloem", 900000.0)
    fplxheight_stele_tparenchyma: float = _p("cm-1", "plasmodesmata frequency, stele-tparenchyma", 640000.0)
    fplxheight_stele_ttracheid: float = _p("cm-1", "plasmodesmata frequency, stele-ttracheid", 0.0)
    fplxheight_sieve_protophloem: float = _p("cm-1", "plasmodesmata frequency, sieve-protophloem", 0.0)
    fplxheight_sieve_tparenchyma: float = _p("cm-1", "plasmodesmata frequency, sieve-tparenchyma", 640000.0)
    fplxheight_sieve_ttracheid: float = _p("cm-1", "plasmodesmata frequency, sieve-ttracheid", 0.0)
    fplxheight_comp_protophloem: float = _p("cm-1", "plasmodesmata frequency, comp-protophloem", 680000.0)
    fplxheight_comp_tparenchyma: float = _p("cm-1", "plasmodesmata frequency, comp-tparenchyma", 1080000.0)
    fplxheight_comp_ttracheid: float = _p("cm-1", "plasmodesmata frequency, comp-ttracheid", 0.0)
    fplxheight_xylem_protophloem: float = _p("cm-1", "plasmodesmata frequency, xylem-protophloem", 0.0)
    fplxheight_xylem_tparenchyma: float = _p("cm-1", "plasmodesmata frequency, xylem-tparenchyma", 1080000.0)
    fplxheight_xylem_ttracheid: float = _p("cm-1", "plasmodesmata frequency, xylem-ttracheid", 1760000.0)
    fplxheight_tparenchyma_tparenchyma: float = _p("cm-1", "plasmodesmata frequency, tparenchyma-tparenchyma", 800000.0)
    fplxheight_tparenchyma_ttracheid: float = _p("cm-1", "plasmodesmata frequency, tparenchyma-ttracheid", 0.0)
    fplxheight_ttracheid_ttracheid: float = _p("cm-1", "plasmodesmata frequency, ttracheid-ttracheid", 0.0)
    fplxheight_protophloem_protophloem: float = _p("cm-1", "plasmodesmata frequency, protophloem-protophloem", 0.0)
    fplxheight_endo_cambium: float = _p("cm-1", "plasmodesmata frequency, endo-cambium", 1080000.0)
    fplxheight_stele_cambium: float = _p("cm-1", "plasmodesmata frequency, stele-cambium", 1080000.0)
    fplxheight_sieve_cambium: float = _p("cm-1", "plasmodesmata frequency, sieve-cambium", 1200000.0)
    fplxheight_comp_cambium: float = _p("cm-1", "plasmodesmata frequency, comp-cambium", 1200000.0)
    fplxheight_xylem_cambium: float = _p("cm-1", "plasmodesmata frequency, xylem-cambium", 640000.0)
    fplxheight_cambium_cambium: float = _p("cm-1", "plasmodesmata frequency, cambium-cambium", 1000000.0)

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
    elongation_sink: float = state_variable(**_doc("cm3 d-1", "the water the elongating wall or protoplast withdraws "
                                                          "(barrier 0; plant problem)"),
                                            initialize=0., scale=SCALES.Compartment, location="node",
                                            state_variable_type="NonInertialExtensive")
    standard_elongation_sink: float = state_variable(**_doc("cm3 d-1", "the elongation sink in the standard problem "
                                                                       "(0 unless standard_elongation)"),
                                                     initialize=0., scale=SCALES.Compartment, location="node",
                                                     state_variable_type="NonInertialExtensive")
    in_standard_problem: float = state_variable(**_doc("", "1 on the Compartments of the plants solving the standard "
                                                           "problem (standard_solve)"),
                                                initialize=0., scale=SCALES.Compartment, location="node",
                                                state_variable_type="descriptor")

    # ── State variables: the osmotic terms ──
    osmotic_drive: float = state_variable(**_doc("hPa", "σ × the osmotic potentials' difference along the edge (as its "
                                                        "potentials': tail − head); its flow is K (Δψ + this)"),
                                          initialize=0., scale=SCALES.Connection, location="edge",
                                          state_variable_type="NonInertialIntensive")
    standard_osmotic_drive: float = state_variable(**_doc("hPa", "the osmotic drive in the standard problem (0 unless "
                                                                 "standard_osmotic)"),
                                                   initialize=0., scale=SCALES.Connection, location="edge",
                                                   state_variable_type="NonInertialIntensive")
    water_potential: float = state_variable(**_doc("hPa", "water potential, plant problem: psi (the pressure "
                                                          "potential) + osmotic_potential"),
                                            initialize=0., scale=SCALES.Compartment, location="node",
                                            state_variable_type="NonInertialIntensive")
    standard_water_potential: float = state_variable(**_doc("hPa", "water potential, standard problem: standard_psi + "
                                                                   "osmotic_potential (with standard_osmotic)"),
                                                     initialize=0., scale=SCALES.Compartment, location="node",
                                                     state_variable_type="NonInertialIntensive")

    # ── State variables: the solutes' diffusion factors and the nodes' volumes (MechaSolutes) ──
    apoplastic_diffusion_factor: float = state_variable(**_doc("cm", "× the apoplastic diffusivity: the edge's "
                                                               "solute conductance (walls: their geometry × kw ratio; "
                                                               "mature vessels' membranes)"),
                                                        initialize=0., scale=SCALES.Connection, location="edge",
                                                        state_variable_type="NonInertialIntensive")
    symplastic_diffusion_factor: float = state_variable(**_doc("cm", "× the plasmodesmata's diffusivity (their open "
                                                               "cross-section × quantity / the crossed wall's "
                                                               "thickness)"),
                                                        initialize=0., scale=SCALES.Connection, location="edge",
                                                        state_variable_type="NonInertialIntensive")
    membrane_diffusion_factor: float = state_variable(**_doc("cm", "× the membranes' passive diffusivity (length × "
                                                             "height / distance)"),
                                                      initialize=0., scale=SCALES.Connection, location="edge",
                                                      state_variable_type="NonInertialIntensive")
    node_volume: float = state_variable(**_doc("cm3", "the node's volume: a wall's length × height × thickness, a "
                                                      "junction's half, a protoplast's area × height"),
                                        initialize=0., scale=SCALES.Compartment, location="node",
                                        state_variable_type="NonInertialExtensive")
    contact_distance: float = state_variable(**_doc("µm", "the node's distance to its section's soil contact along "
                                                          "its graph (with pd_spatial_profile > 0; NaN otherwise)"),
                                             initialize=float("nan"), scale=SCALES.Compartment, location="node",
                                             state_variable_type="NonInertialIntensive")

    # ── State variables: the leaves' vapour path, from their evaporating walls to their surface ──
    surface_inflow: float = state_variable(**_doc("cm3 d-1", "on a leaf organ's surface node: minus its "
                                                  "transpiration (the Neumann condition; 0 elsewhere)"),
                                           initialize=0., scale=SCALES.Compartment, location="node",
                                           state_variable_type="NonInertialIntensive")
    air_tail_conductance: float = state_variable(**_doc("cm3 hPa-1 d-1", "wall_air edges: the flow from tail to head "
                                                        "is air_tail_conductance ψ_tail − air_head_conductance ψ_head "
                                                        "+ air_offset (the linearised Kelvin exchange)"),
                                                 initialize=0., scale=SCALES.Connection, location="edge",
                                                 state_variable_type="NonInertialIntensive")
    air_head_conductance: float = state_variable(**_doc("cm3 hPa-1 d-1", "wall_air edges: see air_tail_conductance"),
                                                 initialize=0., scale=SCALES.Connection, location="edge",
                                                 state_variable_type="NonInertialIntensive")
    air_offset: float = state_variable(**_doc("cm3 d-1", "wall_air edges: see air_tail_conductance"),
                                       initialize=0., scale=SCALES.Connection, location="edge",
                                       state_variable_type="NonInertialIntensive")
    vapour_node: float = state_variable(**_doc("", "1 on the air nodes (the air spaces with wall_air edges, the "
                                                   "stomatal pores, the leaf surfaces): their ψ holds their vapour "
                                                   "pressure, relative to saturation"),
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
    leaf_transpiration: float = input_variable(**_doc("cm3 d-1", "a leaf organ's transpiration, leaving at its surface "
                                                                 "node (role Transpiration: constant for now)"),
                                               by="Transpiration", initialize=0., scale=SCALES.Organ, location="Organ",
                                               state_variable_type="extensive")
    stomatal_aperture: float = input_variable(**_doc("", "a stomatal pore's aperture fraction of its anatomical maximum "
                                                         "(role Stomata: 1 for now)"),
                                              by="Stomata", initialize=1., scale=SCALES.Compartment, location="node",
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
    standard_psi: float = state_variable(**_doc("hPa", "water potential, in its section's standard problem (its "
                                                       "pressure potential with standard_osmotic 1)"),
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
    standard_transmembrane_fraction: float = state_variable(**_doc("adim", "the membrane's share of the section's "
                                                                   "standard uptake: its standard flow from wall to "
                                                                   "cell / the uptake (MECHA's standard transmembrane "
                                                                   "fraction; 0 on the other edges and membranes_layer "
                                                                   "-1)"),
                                                             initialize=0., scale=SCALES.Connection, location="edge",
                                                             state_variable_type="NonInertialIntensive")
    membrane_layer: float = state_variable(**_doc("", "the membrane's layer row in its section (MECHA's radial "
                                                      "discretisation, from the stele, 0, to the epidermis; -1: not a "
                                                      "membrane, or one without transmembrane flow: an intercellular "
                                                      "space's, a mature xylem vessel's)"),
                                           initialize=-1., scale=SCALES.Connection, location="edge",
                                           state_variable_type="descriptor")
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
        set_wall_thickness(net, np.asarray(ds.get("wall_thickness"), dtype=float))
        net.soil_contact_cell = (np.asarray(ds.get("soil_contact_cell"), dtype=float) > 0) & (net.kind == 0)
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
    def _conductance(self, network_path, barrier, length, anatomy_template, leaf_transpiration, stomatal_aperture,
                     pore_width, pore_depth) \
            -> tuple[np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray,
                     str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str,
                     np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray]:
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
        factors = {}
        K, in_matrix = edge_conductances(net, P_edge, barrier_node[tail], height_node[tail], factors=factors)
        soil, axial, _ = boundary_terms(net, P_node, barrier_node, height_node)
        axial_psi = np.where(barrier_node > 0, P_node["standard_psi_xylem"], P_node["standard_psi_sieve"])
        axial = np.where(np.isnan(axial_psi), 0., axial)
        flow_condition = (P_node["standard_axial_condition"] == 2) & (barrier_node > 0)
        axial = np.where(flow_condition, 0., axial)                    # the flow condition replaces k_xyl's
        sieve_tube = sieve_tube_terms(net, P_node, barrier_node, height_node)       # the phloem, barrier > 0
        sieve_tube = np.where(np.isnan(P_node["standard_psi_sieve_tube"]), 0., sieve_tube)
        axial = np.where(sieve_tube > 0, sieve_tube, axial)
        axial_psi = np.where(sieve_tube > 0, P_node["standard_psi_sieve_tube"], axial_psi)
        relative_x = self.relative_x(geometries)
        right = P_node["standard_psi_soil_right"]
        standard_soil = np.where(np.isnan(right), P_node["standard_psi_soil"],
                                 P_node["standard_psi_soil"] * (1 - relative_x) + right * relative_x)
        inflow = self._standard_axial_inflow(geometries, P_node, flow_condition)
        root = self._root_nodes(geometries)
        soil = np.where(root, soil, 0.)                               # stems and leaves: no soil (E1)
        K = np.where(in_matrix, K, 0.)
        ds = self.data_structure
        kind = np.asarray(ds.get("kind"))
        pore_width, pore_depth = np.asarray(pore_width, dtype=float), np.asarray(pore_depth, dtype=float)
        view = ds.to_graph_view()
        K_vapour, K_wall, K_air, offset, wall_end, air_end = vapour_conductances(
            net, P_edge, height_node[tail], pore_width, pore_depth, np.asarray(ds.get("cell_area"), dtype=float),
            np.asarray(stomatal_aperture, dtype=float), height_node[view.tail])
        K = K + K_vapour
        a_is_wall = (wall_end >= 0) & (wall_end == net.a)
        air_tail = np.where(a_is_wall, K_wall, K_air)
        air_head = np.where(a_is_wall, K_air, K_wall)
        air_offset = np.where(a_is_wall, offset, -offset)
        vapour = np.zeros(ds.n_nodes(), dtype=bool)
        vapour[air_end[wall_end >= 0]] = True
        vapour |= np.isfinite(pore_width) | (kind == LEAF_SURFACE)
        surface = kind == LEAF_SURFACE
        # each organ's transpiration onto its surface node, conserved (an organ transpiring without one raises)
        surface_inflow = -ds.map_to(leaf_transpiration, "Organ", "node", "split", weights=surface)
        standard = (P_node["standard_solve"] > 0) & root
        return (K, "soil_conductance", soil, "axial_boundary_conductance", axial,
                "standard_soil_psi", standard_soil.astype(float),
                "standard_axial_psi", np.where(np.isnan(axial_psi), 0., axial_psi),
                "in_standard_problem", standard.astype(float), "standard_axial_inflow", inflow,
                "air_tail_conductance", np.where(wall_end >= 0, air_tail, 0.),
                "air_head_conductance", np.where(wall_end >= 0, air_head, 0.),
                "air_offset", np.where(wall_end >= 0, air_offset, 0.),
                "vapour_node", vapour.astype(float), "surface_inflow", surface_inflow,
                "apoplastic_diffusion_factor", np.where(in_matrix, factors["apoplastic"], 0.),
                "symplastic_diffusion_factor", np.where(in_matrix, factors["symplastic"], 0.),
                "membrane_diffusion_factor", np.where(in_matrix, factors["membrane"], 0.),
                "node_volume", self._node_volumes(net, P_node, height_node, kind),
                "contact_distance", factors["contact_distance"])

    def relative_x(self, geometries: dict) -> np.ndarray:
        """Per node, its x relative to its section's cells' x range (0 left, 1 right; the legacy's x_rel, from the
        cells' centroids: doc/legacy.rst, ME-22)."""
        out = np.zeros(self.data_structure.n_nodes())
        for nodes, _, geo in geometries.values():
            x_cells = geo.x[geo.cells]
            low, high = x_cells.min(), x_cells.max()
            out[nodes] = (geo.x - low) / (high - low) if high > low else 0.
        return out

    @rate(include_inactive=True)
    def _elongation_sink(self, barrier, anatomy_template) -> tuple[np.ndarray, str, np.ndarray]:
        """The elongation sink of the sections without barriers (barrier 0), in the plant problem, and in the
        standard problem of the plants with standard_elongation."""
        geometries = self.section_geometries()
        net = self.network(geometries)
        P_node = self.parameter_table("node")
        sink = elongation_sink(net, P_node, np.asarray(barrier), self.relative_x(geometries))
        return sink, "standard_elongation_sink", np.where(P_node["standard_elongation"] > 0, sink, 0.)

    def _node_volumes(self, net, P_node, height, kind):
        """Each node's volume (cm³; the reactions' SoluteGeometry.node_volumes): a wall's length × height ×
        thickness, a junction's half (its length), a protoplast's area × height; 1e-30 at the least."""
        thickness = _node_thickness(net, P_node)
        volume = np.where(kind == 1, net.wall_length * height * thickness * 1e-12,
                          np.where(kind == 2, height * thickness * net.wall_length / 2.0 * 1e-12,
                                   net.cell_area * height * 1e-12))
        return np.where(net.section_of_node >= 0, np.maximum(volume, 1e-30), 0.)

    def _liquid_equivalent(self, value, vapour_node, location="node"):
        """The water potential (hPa) of the vapour nodes: Kelvin's liquid equivalent of their relative vapour
        pressure x, ln(x) / α (α = V_w / (R T))."""
        alpha = kelvin_alpha(np.asarray(self.parameter_values("leaf_temperature", location), dtype=float))
        vapour = np.asarray(vapour_node) > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            kelvin = np.log(np.maximum(value, 1e-300)) / alpha
        return vapour, kelvin

    def _standard_axial_inflow(self, geometries, P_node, flow_condition) -> np.ndarray:
        """
        The standard problem's flow condition (2): the flow into the xylem split among the vessels (the network's,
        by cgroup) as their areas. The legacy (``_distribute_xylem_flow``) computes the ratios on its cell manager's
        xylem (by tissue) and gives them by rank to the network's, so that a vessel may get another's (doc/legacy.rst, o: fixed).
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

    @rate(include_inactive=True)
    def _osmotic_drive(self, osmotic_potential, reflection_coefficient) -> tuple[np.ndarray, str, np.ndarray]:
        """
        Each edge's osmotic drive σ (π_tail − π_head) (the membranes': the legacy's ``rhs_o``, as a flow K σ Δπ added
        to K Δψ), and the standard problem's (its plants with standard_osmotic).
        """
        ds = self.data_structure
        view = ds.to_graph_view()
        pi = np.asarray(osmotic_potential, dtype=float)
        drive = np.asarray(reflection_coefficient, dtype=float) * (pi[view.tail] - pi[view.head])
        standard = np.asarray(self.parameter_values("standard_osmotic", "edge")) > 0
        return drive, "standard_osmotic_drive", np.where(standard, drive, 0.)

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
    # tol (cm3 d-1): the pinned walls' penalty puts their residuals' round-off at ~1e-9; the systems are linear,
    # solved by the first Newton step
    @graph_system(node_unknowns=["standard_psi"], solver="newton", max_iter=3, tol=1e-8, schedule_as="state",
                  filters={"in_standard_problem": ">0"})
    class _standard_problem:
        """
        Each section alone (the junctions have no conductance here): its border walls and junctions exchange with
        the soil, its xylem vessels (barrier > 0) or protophloem cells (barrier 0) with the axial boundary.
        """

        @node_balance(field="standard_psi")
        def _balance(self, standard_psi, conductance, standard_osmotic_drive, air_tail_conductance,
                     air_head_conductance, air_offset, standard_elongation_sink):
            B = self._graph_view.incidence
            drop = np.asarray(B.T @ standard_psi).reshape(-1) + standard_osmotic_drive
            q = _wall_air_flow(B, standard_psi, air_tail_conductance, air_head_conductance, air_offset)
            return np.asarray(B @ (conductance * drop + q)).reshape(-1) + standard_elongation_sink

        soil = boundary_set(filters={"soil_conductance": ">0"}, kind="robin", value="standard_soil_psi",
                            weight="soil_conductance", field="standard_psi")
        axial = boundary_set(filters={"axial_boundary_conductance": ">0"}, kind="robin", value="standard_axial_psi",
                             weight="axial_boundary_conductance", field="standard_psi")
        axial_flow = boundary_set(filters={"standard_axial_inflow": "<0"}, kind="neumann", value="standard_axial_inflow",
                                  field="standard_psi")

        @graph_jacobian
        def _jacobian(self, conductance, air_tail_conductance, air_head_conductance):
            B = self._graph_view.incidence
            return (B @ diags(conductance) @ B.T
                    + _wall_air_jacobian(B, air_tail_conductance, air_head_conductance)).tocsr()

        @graph_output("standard_flow", location="edge")
        def _standard_flow(self, standard_psi, conductance, standard_osmotic_drive, air_tail_conductance,
                           air_head_conductance, air_offset):
            B = self._graph_view.incidence
            return conductance * (np.asarray(B.T @ standard_psi).reshape(-1) + standard_osmotic_drive) + \
                _wall_air_flow(B, standard_psi, air_tail_conductance, air_head_conductance, air_offset)

    # ── The plant problem: the sections joined by their xylem junctions ──
    @graph_system(node_unknowns=["psi"], solver="newton", max_iter=3, tol=1e-8, schedule_as="state")
    class _plant_problem:
        """
        The whole plant: every section's edges and the xylem junctions between sections; the soil around the root
        surface (soil_psi), the collar's condition on the first root section's vessels; the leaves' vapour path,
        from their evaporating walls through their air spaces and stomata to their surface nodes, where their organs'
        transpiration leaves (a Neumann condition).
        """

        @node_balance(field="psi")
        def _balance(self, psi, conductance, xylem_axial_conductance, osmotic_drive, air_tail_conductance,
                     air_head_conductance, air_offset, elongation_sink):
            B = self._graph_view.incidence
            K = conductance + xylem_axial_conductance
            q = _wall_air_flow(B, psi, air_tail_conductance, air_head_conductance, air_offset)
            return np.asarray(B @ (K * (np.asarray(B.T @ psi).reshape(-1) + osmotic_drive) + q)).reshape(-1) + \
                elongation_sink

        soil = boundary_set(filters={"soil_conductance": ">0"}, kind="robin", value="soil_psi",
                            weight="soil_conductance", field="psi")
        collar = boundary_set(filters={"collar_kind": ">0"}, kinds="collar_kind", value="collar_value", weight=0.,
                              field="psi")
        leaf_surfaces = boundary_set(filters={"surface_inflow": "<0"}, kind="neumann", value="surface_inflow",
                                     field="psi")

        @graph_jacobian
        def _jacobian(self, conductance, xylem_axial_conductance, air_tail_conductance, air_head_conductance):
            B = self._graph_view.incidence
            return (B @ diags(conductance + xylem_axial_conductance) @ B.T
                    + _wall_air_jacobian(B, air_tail_conductance, air_head_conductance)).tocsr()

        @graph_output("flow", location="edge")
        def _flow(self, psi, conductance, xylem_axial_conductance, osmotic_drive, air_tail_conductance,
                  air_head_conductance, air_offset):
            B = self._graph_view.incidence
            return (conductance + xylem_axial_conductance) * (np.asarray(B.T @ psi).reshape(-1) + osmotic_drive) + \
                _wall_air_flow(B, psi, air_tail_conductance, air_head_conductance, air_offset)

        @graph_output("root_uptake", location="node")
        def _root_uptake(self, psi, soil_conductance, soil_psi):
            return soil_conductance * (soil_psi - psi)

        @graph_output("evaporation", location="node")
        def _evaporation(self, psi, conductance, network_path):
            """Each stomatal pore's flow to its leaf's surface (the stomatal Connections run from the pore)."""
            B = self._graph_view.incidence
            stomatal = np.asarray(network_path) == STOMATAL
            q = np.where(stomatal, conductance * np.asarray(B.T @ psi).reshape(-1), 0.)
            return np.asarray(B.maximum(0) @ q).reshape(-1)

        @graph_output("water_potential", location="node")
        def _water_potential(self, psi, osmotic_potential, vapour_node):
            vapour, kelvin = self._liquid_equivalent(psi, vapour_node)
            return np.where(vapour, kelvin, psi + osmotic_potential)

    # ── The sections' results ──
    @totalstate(include_inactive=True)
    def _kr(self, standard_psi, psi, soil_conductance, soil_psi, barrier, length, osmotic_potential,
            standard_soil_psi) \
            -> tuple[np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray,
                     str, np.ndarray]:
        """
        Each section's uptake and radial conductivity: MECHA's (its standard problem, the legacy's operations) and
        the effective ones of the plant problem; its axial conductances (Poiseuille); the standard problem's water
        potentials (with the osmotic potentials where it has their terms, standard_osmotic).
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
                contact = nodes[soil[nodes] > 0]
                standard_soil = np.asarray(standard_soil_psi, dtype=float)
                psi_soil = float(standard_soil[contact].mean()) if contact.size else float(P["standard_psi_soil"][v])
                psi_axial = float(P["standard_psi_xylem"][v] if b > 0 else P["standard_psi_sieve"][v])
                if b > 0 and P["standard_axial_condition"][v] == 2:      # the xylem's potential: its vessels' mean
                    vessels = geo.xylem_cells
                    psi_axial = 0.0
                    for c in vessels:
                        psi_axial += standard[nodes[c]] / len(vessels)
                    psi_axial = float(psi_axial)
                out["standard_uptake"][i] = soil_uptake(geo, soil[nodes], standard[nodes], standard_soil[nodes])
                out["kr"][i] = radial_conductivity(out["standard_uptake"][i], psi_soil, psi_axial, geo.perimeter, h)
            contact = nodes[soil[nodes] > 0]
            out["uptake"][i] = float(np.sum(soil[contact] * (soil_psi[contact] - plant[contact])))
            vessels = nodes[np.isin(cgroup[nodes], XYLEM_CGROUPS)]
            if contact.size and vessels.size and b > 0:                # no mature xylem at barrier 0 (D3)
                out["kr_effective"][i] = radial_conductivity(out["uptake"][i], soil_psi[contact].mean(),
                                                             plant[vessels].mean(), geo.perimeter, h)
        with_osmotic = np.asarray(self.parameter_values("standard_osmotic", "node")) > 0
        water = standard + np.where(with_osmotic, np.asarray(osmotic_potential, dtype=float), 0.)
        vapour, kelvin = self._liquid_equivalent(standard, ds.get("vapour_node"))
        water = np.where(vapour, kelvin, water)
        return tuple(x for name in names for x in ((out[name],) if name == "kr" else (name, out[name]))) + (
            "standard_water_potential", water)

    @totalstate(include_inactive=True)
    def _standard_transmembrane_fraction(self, standard_flow, standard_uptake, barrier, network_path) \
            -> tuple[np.ndarray, str, np.ndarray]:
        """
        Each membrane's standard transmembrane fraction (its standard flow from wall to cell / its section's standard
        uptake: ``Mecha.standard_transmembrane_fractions``) and its layer row (postprocess.transmembrane_layers sums
        them by layer), on the root sections solving the standard problem.
        """
        from openalea.mecha.osmotic import membrane_rows             # osmotic imports this module
        ds = self.data_structure
        view = ds.to_graph_view()
        fraction, layer = np.zeros(ds.n_edges()), np.full(ds.n_edges(), -1.)
        flow, path, kind = np.asarray(standard_flow, dtype=float), np.asarray(network_path), np.asarray(ds.get("kind"))
        uptake, barrier = np.asarray(standard_uptake, dtype=float), np.asarray(barrier)
        where = {int(v): i for i, v in enumerate(ds.entity_ids("SubOrgan"))}
        standard = self.parameter_table("node")["standard_solve"] > 0
        roots = self.section_roles()[1]
        for segment, (nodes, edges, geo) in self.section_geometries().items():
            if segment not in roots or not standard[nodes[0]]:
                continue
            local = np.full(ds.n_nodes(), -1, dtype=np.int64)
            local[nodes] = np.arange(nodes.size)
            membranes = edges[path[edges] == MEMBRANE]
            tail, head = view.tail[membranes], view.head[membranes]
            wall_is_tail = kind[tail] != 0
            walls, cells = np.where(wall_is_tail, tail, head), np.where(wall_is_tail, head, tail)
            rows = membrane_rows(geo, int(barrier[nodes[0]]), local[walls], local[cells])
            into_cell = np.where(wall_is_tail, flow[membranes], -flow[membranes])
            fraction[membranes] = np.where(rows >= 0, into_cell / uptake[where[segment]], 0.)
            layer[membranes] = rows
        return fraction, "membrane_layer", layer


MechaHydraulics.PARAMETERS = tuple(MechaHydraulics._plant_parameter_fields())

