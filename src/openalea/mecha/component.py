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

from openalea.metafspm.coupling.component import (FunctionalComponent, input_variable, parameter, setting,
                                                   state_variable)
from openalea.metafspm.data_structure.configs import ScalesConfig
from openalea.metafspm.solve.conservation import Conservation
from openalea.metafspm.solve.decorator import (boundary_set, declared_system, explicit_term, graph_jacobian,
                                               graph_output, graph_system, input_jacobian, kind_derived,
                                               node_balance, rate, totalstate)

from openalea.mecha.conductance import (Network, assemble, boundary_terms, cortex_parameters, edge_conductances,
                                       elongation_sink, kelvin_alpha, set_edge_ends, set_wall_thickness,
                                       sieve_tube_terms, vapour_conductances)
from openalea.mecha.geometry import (AXIAL_JUNCTION, LEAF_SURFACE, MEMBRANE, SOIL, SOIL_NODE, STOMATAL, XYLEM_CGROUPS,
                                     AIR_LINK, WALL, WALL_AIR, local_index,
                                     phi_thick_cells,
                                     section_geometry, section_of, section_parts, template_section, template_values)
from openalea.mecha.postprocess import (VISCOSITY_PHLOEM, VISCOSITY_XYLEM, axial_conductances, conduit_axial_conductances,
                                       poiseuille,
                                        radial_conductivity, soil_uptake)

PHLOEM_CGROUPS = (11, 23)
# The root sections' SubOrgan labels: the root segments and the root apices (an axis' tip, ArchiSimple's, with an
# anatomy of its own: GRANAP's root apex kind), the sections with a soil contact, a kr and an uptake
ROOT_LABELS = ("RootSegment", "RootApex")
# The plant problem's water, conserved: nothing held (a steady flow), in from the soil, out by the organs'
# transpiration and to the elongating cells (cm3 d-1 x s, in cm3); its residual, each step, is the plant's imbalance
PLANT_WATER = Conservation(inflow="root_uptake", outflow=("leaf_transpiration", "elongation_sink"), unit="cm3")

SCALES = ScalesConfig()


def _doc(unit, description, **values):
    return dict(unit=unit, unit_comment="", description=description, min_value=-1e12, max_value=1e12,
                value_comment="", references="MECHA (Couvreur et al. 2018)", DOI=["10.1104/pp.18.01006"], **values)


def _node(unit, description):
    return dict(**_doc(unit, description), by="Anatomy", scale=SCALES.Compartment, location="node",
                kind="descriptor")


def _p(unit, description, default):
    return parameter(**_doc(unit, description), by="MECHA", default=default)


def _bounded(unit, description, default, low, high, comment):
    """A parameter with bounds of its own (calibrations and variants read them), *comment* saying where they come
    from."""
    declared = _doc(unit, description)
    declared.update(min_value=low, max_value=high, value_comment=comment)
    return parameter(**declared, by="MECHA", default=default)


def _edge(unit, description):
    return dict(**_doc(unit, description), by="Anatomy", scale=SCALES.Connection, location="edge",
                kind="descriptor")


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
                                             by="Anatomy", scale=SCALES.SubOrgan, kind="descriptor")

    barrier: float = input_variable(**_doc("", "the segment's apoplastic barriers (MECHA's maturity code: 0 none, 1 "
                                                 "endodermal Casparian strip, 2 + suberin with passage cells, 3 "
                                                 "endodermis fully suberized, 4-9 exodermis variants, 10 exodermis "
                                                 "suberized with a Casparian strip, 11 endodermis and exodermis "
                                                 "suberized and the sclerenchyma's walls lignified)"),
                                    by="Development", default=1., scale=SCALES.SubOrgan, location="node",
                                    aggregation="broadcast", kind="descriptor")
    length: float = input_variable(**_doc("m", "the segment's length (MECHA's section height)"),
                                   by="Architecture", scale=SCALES.SubOrgan, location="node",
                                   kind="intensive")

    # ── Geometry, per distinct anatomy ──
    def section_geometries(self) -> dict:
        """
        {SubOrgan vid: (nodes, edges, SectionGeometry)} of every section with an anatomy: its Compartments and
        Connections (DataStructure indices) and MECHA's geometry, computed once per anatomy template.
        """
        ds = self.data_structure
        templates = np.asarray(ds.get("anatomy_template"))
        reduced = self._reduced_nodes()
        stamp = (ds.structure_stamp(), templates.tobytes(),
                 reduced.tobytes())
        kept = self.caches().get("section_geometries")              # the component's caches (dropped on a load)
        if kept is not None and kept[0] == stamp:                   # the same sections: the same dict (read only)
            return kept[1]
        template = dict(zip(ds.entity_ids("SubOrgan").tolist(), templates.tolist()))
        out = {}
        for segment, (nodes, edges) in section_parts(ds, exclude=reduced).items():
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
        self.caches()["section_geometries"] = (stamp, out)
        return out

    def section_layout(self, geometries: dict = None) -> tuple:
        """
        (node_section, section_geometry, distinct) of *geometries* (section_geometries' by default), for the per-node
        quantities of the sections' geometries in one gather (ds.local_gather): each node's section (its row among
        the sections, -1 for none), each section's geometry (its index in *distinct*, the distinct SectionGeometry
        objects); kept per geometries dict.
        """
        geometries = self.section_geometries() if geometries is None else geometries
        kept = self.caches().get("section_layout")
        if kept is not None and kept[0] is geometries:
            return kept[1]
        node_section = np.full(self.data_structure.n_nodes(), -1, dtype=np.int64)
        index, distinct = {}, []
        section_geometry = np.empty(len(geometries), dtype=np.int64)
        for row, (nodes, _, geometry) in enumerate(geometries.values()):
            node_section[nodes] = row
            if id(geometry) not in index:
                index[id(geometry)] = len(distinct)
                distinct.append(geometry)
            section_geometry[row] = index[id(geometry)]
        value = (node_section, section_geometry, distinct)
        self.caches()["section_layout"] = (geometries, value)
        return value

    def per_node(self, quantity, geometries: dict = None, fill=np.nan) -> np.ndarray:
        """Per node, *quantity*(geometry)[its local row] of its section's geometry (*fill* outside the sections): one
        table per distinct geometry, gathered at once (section_layout, ds.local_gather)."""
        node_section, section_geometry, distinct = self.section_layout(geometries)
        return self.data_structure.local_gather(node_section, section_geometry,
                                                [quantity(geometry) for geometry in distinct], fill=fill)

    def _reduced_nodes(self) -> np.ndarray:
        """bool per node: in a reduced group (the plant model's Reduction, solve.reduction: no cells here)."""
        ds = self.data_structure
        out = np.zeros(ds.n_nodes(), dtype=bool)
        reduction = ds.reduction
        if reduction is None:
            return out
        groups = reduction.reduced_groups(ds)
        if groups:
            out = np.isin(np.asarray(ds.owner(reduction.location)), list(groups))
        return out

    def _reduction(self):
        """The plant model's Reduction (solve.reduction) when it reduces this component's plant problem, else None."""
        reduction = self.data_structure.reduction
        return reduction if reduction is not None and self._plant_problem_name(reduction) is not None else None

    def _plant_problem_name(self, reduction) -> str:
        """The name among *reduction*'s systems of this component's plant problem: its class's or a base class's (a
        subclass of MechaHydraulics runs the same reduced system), or None."""
        return declared_system(self, "_plant_problem", reduction.systems)

    def section_roles(self) -> tuple:
        """
        (first, roots): the SubOrgan vids without a parent (a root system's first section) and the root sections,
        labelled RootSegment or RootApex (an axis' tip with an anatomy, GRANAP's root apex kind); read once per
        topology. The leaves are not labels: an organ transpires from the leaf surface
        its anatomy gave it (GRANAP's Compartment of kind 3, LEAF_SURFACE: a leaf's, Adel's blade's or sheath's).
        """
        def build():
            ds = self.data_structure
            segments = [int(v) for v in ds.entity_ids("SubOrgan")]
            labels = ds.labels(location="SubOrgan", per_node=False).tolist()
            return ({segments[i] for i in ds.roots(location="SubOrgan").tolist()},
                    {v for v, label in zip(segments, labels) if label in ROOT_LABELS})
        return self.cached("section_roles", build)

    def _root_nodes(self, geometries) -> np.ndarray:
        """bool per node: in a root section."""
        roots = self.section_roles()[1]
        node_section = self.section_layout(geometries)[0]
        is_root = np.array([segment in roots for segment in geometries] + [False], dtype=bool)   # row -1: none
        return is_root[node_section]


@dataclass
class MechaHydraulics(MechaSections):
    """
    MECHA's cross-section hydraulics on every section of the DataStructure (its sections: MechaSections).

    Its named entity set (entity_sets): soil_contacts, the border walls and junctions in contact with the soil
    (soil_conductance > 0, soil_boundary "per_node"), e.g. the soil links' mapping's mask
    (define_mask("root_surface", entities="MechaHydraulics.soil_contacts")).
    """
    entity_sets = {"soil_contacts": {"soil_conductance": ">0"}}
    # ── Inputs: the osmotic potentials (role "Solutes": a metabolism model, or MECHA's presets, MechaOsmoticPresets) ──
    osmotic_potential: float = input_variable(**_doc("hPa", "osmotic potential of the protoplast or the wall's solution "
                                                            "(0: none)"),
                                              by="Solutes", default=0., scale=SCALES.Compartment, location="node",
                                              kind="NonInertialIntensive")
    reflection_coefficient: float = input_variable(**_doc("", "the membrane's reflection coefficient (0: none)"),
                                                   by="Solutes", default=0., scale=SCALES.Connection,
                                                   location="edge", kind="NonInertialIntensive")
    soil_contact_cell: float = input_variable(**_doc("", "1 on a protoplast whose border walls touch the soil, whatever "
                                                         "x_contact (MECHA's Contactrange: a root partly in contact)"),
                                              by="Soil", default=0., scale=SCALES.Compartment, location="node",
                                              kind="descriptor")
    measured_axial_conductivity: float = input_variable(**_doc("cm4 hPa-1 d-1", "a xylem vessel's or phloem conduit's "
                                                               "measured specific axial conductance, its conductance "
                                                               "× its length (MECHA's Kax_source 2, its K_xyl and "
                                                               "K_sieve by cell; NaN: Poiseuille's)"),
                                                        by="Anatomy", default=float("nan"), scale=SCALES.Compartment,
                                                        location="node", kind="descriptor")
    # ── Parameters (per plant; MECHA's units: cm hPa-1 d-1 for kw, kmb, kaqp; cm3 hPa-1 d-1 for kpl, k_xyl) ──
    kw: float = _bounded("cm2 hPa-1 d-1", "cell wall hydraulic conductivity", 0.00024, 2.4e-6, 2.4e-2,
                         "bounds: two orders of magnitude around MECHA's default, for calibrations")
    kw_septa: float = _p("cm2 hPa-1 d-1", "conductivity of the septa between aerenchyma (intercellular) cells", 0.00012)
    kw_casparian: float = _p("cm2 hPa-1 d-1", "wall conductivity across a Casparian strip", 1.00E-16)
    kw_suberin_in: float = _p("cm2 hPa-1 d-1", "conductivity of suberized walls, inner face", 1.00E-16)
    kw_suberin_out: float = _p("cm2 hPa-1 d-1", "conductivity of suberized walls, outer face", 1.00E-16)
    kw_lignin: float = _p("cm2 hPa-1 d-1", "conductivity of lignified walls (lignin cap, phi thickenings)", 1.00E-16)
    thickness_source: float = _p("", "the walls' thickness: 1 one value for every wall (thickness, MECHA's), 2 each "
                                     "wall's own (wall_thickness, GRANAP's; a junction's: the mean of its walls')", 2.)
    thickness: float = _p("µm", "cell wall thickness, every wall's (thickness_source 1, MECHA's)", 1.5)
    xylem_pieces: float = _p("", "1: the walls between two xylem cells are septa of near-zero conductance", 0.)
    kmb: float = _bounded("cm hPa-1 d-1", "membrane hydraulic conductivity, without aquaporins", 3.0E-5, 3e-7, 3e-3,
                          "bounds: two orders of magnitude around MECHA's default, for calibrations")
    kaqp: float = _bounded("cm hPa-1 d-1", "aquaporin contribution to the membrane conductivity", 0.000430, 0.,
                           4.3e-2, "bounds: none to two orders of magnitude above MECHA's default, for calibrations")
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
    xylem_viscosity: float = _p("hPa s", "the xylem sap's viscosity (the vessels' Poiseuille conductances): water at "
                                         "10 °C, 1.3076E-3 Pa s (the legacy's)", VISCOSITY_XYLEM)
    phloem_viscosity: float = _p("hPa s", "the phloem sap's viscosity (the sieve tubes' and protosieve cells' "
                                          "Poiseuille conductances): 1.7E-3 Pa s (Ross-Elliott et al. 2017, the "
                                          "legacy's)", VISCOSITY_PHLOEM)
    n_phi_layers: float = _p("", "number of cortex layers with phi thickenings", 0.)
    phi_type: float = _p("", "phi thickenings: 1 inner cortex, 2 outer cortex, 3 mid cortex", 1.)

    standard_solve: float = _p("", "1: solve MECHA's standard problem on each section alone (its kr; the legacy's "
                                   "definition), 0: skip it", 1.)
    standard_osmotic: float = _p("", "1: the standard problem with the membranes' osmotic terms (the legacy's scenarios "
                                     "≥ 1), 0: without (its scenario 0, MECHA's kr)", 0.)
    plant_solver: str = setting(description="the linear solves of the plant problem and of the standard problem: "
                                            "\"full\", one sparse solve of the whole network; \"condensed\", each "
                                            "section's interior eliminated, one factorisation per kind of section "
                                            "(solve.network.condensed_solve; the soil, collar and junction nodes "
                                            "kept): the same potentials to round-off",
                                default="full", choices=("full", "condensed"))
    soil_boundary: str = setting(description="the soil condition of the plant problem: \"per_node\", a Robin term on "
                                             "each border wall and junction in contact (MECHA's); \"node_dirichlet\", "
                                             "one per root section, its soil node held at the section's soil_psi, "
                                             "joined to its contact nodes through their soil conductances (GRANAP's "
                                             "soil_node; with a uniform soil around the section, the same as "
                                             "per_node); \"node_robin\", its soil node joined to soil_psi through "
                                             "the section's soil_bulk_conductance",
                                 default="per_node", choices=("per_node", "node_dirichlet", "node_robin"))
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
    standard_psi_sieve: float = _p("hPa", "the protophloem's potential in the standard problem (barrier 0: the stele's "
                                   "boundary, against which kr is defined there; MECHA's BCs.xml: barley phloem's "
                                   "pressure, Pritchard 1996; standard_sieve_flow replaces it; NaN: no phloem "
                                   "boundary, kr undefined at barrier 0, the legacy's default without BCs.xml)", 1.1E4)
    standard_psi_sieve_tube: float = _p("hPa", "the sieve tubes' potential in the standard problem (barrier > 0: the "
                                               "legacy's phloem pressure of its scenarios >= 1, with k_sieve or "
                                               "Poiseuille as standard_axial_source; NaN: no phloem boundary)",
                                        float("nan"))
    standard_sieve_flow: float = _p("cm3 d-1", "the flow into the phloem in the standard problem (the legacy's "
                                               "phloem flowrate): into the protophloem at barrier 0 (in place of "
                                               "standard_psi_sieve), the sieve tubes at barrier > 0, where their "
                                               "potential is not given (NaN), split among them as their areas; "
                                               "negative: the phloem unloading water into the section (NaN: none)",
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
                                      default=-1., scale=SCALES.Compartment, location="node",
                                      kind="descriptor")
    perimeter: float = state_variable(**_doc("cm", "the section's perimeter: 2π × the epidermis walls' mean distance "
                                                   "to the endodermis' centre"),
                                      default=np.nan, scale=SCALES.SubOrgan, kind="descriptor")

    # ── State variables: the conductances ──
    conductance: float = state_variable(**_doc("cm3 hPa-1 d-1", "the edge's hydraulic conductance (walls, membranes, "
                                                                "plasmodesmata; 0 on the edges MECHA leaves out)"),
                                        default=0., scale=SCALES.Connection, location="edge",
                                        kind="NonInertialIntensive")
    soil_conductance: float = state_variable(**_doc("cm3 hPa-1 d-1", "conductance between a border wall or junction "
                                                                     "and the soil (0 elsewhere)"),
                                             default=0., scale=SCALES.Compartment, location="node",
                                             kind="NonInertialIntensive")
    plant_soil_conductance: float = state_variable(**_doc("cm3 hPa-1 d-1", "the plant problem's soil term on a border "
                                                          "wall or junction: soil_conductance (soil_boundary "
                                                          "per_node), 0 with a soil node"),
                                                   default=0., scale=SCALES.Compartment, location="node",
                                                   kind="NonInertialIntensive")
    soil_node_kind: float = state_variable(**_doc("", "on a root section's soil node (GRANAP's soil_node): its "
                                                  "condition, 1 the potential soil_psi (soil_boundary node_dirichlet, "
                                                  "and per_node: the node then unused), 3 joined to soil_psi through "
                                                  "soil_bulk_conductance (node_robin); 0 elsewhere"),
                                           default=0., scale=SCALES.Compartment, location="node",
                                           kind="NonInertialIntensive")
    axial_boundary_conductance: float = state_variable(**_doc("cm3 hPa-1 d-1", "conductance between a xylem vessel "
                                                              "(barrier > 0) or a protophloem cell (barrier 0) and "
                                                              "the axial boundary, on the matrix diagonal (0 elsewhere)"),
                                                       default=0., scale=SCALES.Compartment, location="node",
                                                       kind="NonInertialIntensive")

    standard_soil_psi: float = state_variable(**_doc("hPa", "the soil's potential at the border walls and junctions in "
                                                            "the standard problem (its plant's standard_psi_soil)"),
                                              default=0., scale=SCALES.Compartment, location="node",
                                              kind="NonInertialIntensive")
    standard_axial_psi: float = state_variable(**_doc("hPa", "the axial boundary's potential at the xylem vessels "
                                                             "(barrier > 0) or protophloem cells (barrier 0) in the "
                                                             "standard problem"),
                                               default=0., scale=SCALES.Compartment, location="node",
                                               kind="NonInertialIntensive", weight="liquid_volume")
    standard_axial_inflow: float = state_variable(**_doc("cm3 d-1", "the water entering a xylem vessel or a phloem "
                                                         "conduit from the axial boundary in the standard problem "
                                                         "(standard_axial_condition 2, standard_sieve_flow; negative: "
                                                         "its share of the flow into the xylem or the phloem)"),
                                                  default=0., scale=SCALES.Compartment, location="node",
                                                  kind="NonInertialIntensive")
    elongation_sink: float = state_variable(**_doc("cm3 d-1", "the water the elongating wall or protoplast withdraws "
                                                          "(barrier 0; plant problem)"),
                                            default=0., scale=SCALES.Compartment, location="node",
                                            kind="NonInertialExtensive")
    standard_elongation_sink: float = state_variable(**_doc("cm3 d-1", "the elongation sink in the standard problem "
                                                                       "(0 unless standard_elongation)"),
                                                     default=0., scale=SCALES.Compartment, location="node",
                                                     kind="NonInertialExtensive")
    in_standard_problem: float = state_variable(**_doc("", "1 on the Compartments of the plants solving the standard "
                                                           "problem (standard_solve)"),
                                                default=0., scale=SCALES.Compartment, location="node",
                                                kind="descriptor")

    # ── State variables: the osmotic terms ──
    osmotic_drive: float = state_variable(**_doc("hPa", "σ × the osmotic potentials' difference along the edge (as its "
                                                        "potentials': tail − head); its flow is K (Δψ + this)"),
                                          default=0., scale=SCALES.Connection, location="edge",
                                          kind="NonInertialIntensive")
    standard_osmotic_drive: float = state_variable(**_doc("hPa", "the osmotic drive in the standard problem (0 unless "
                                                                 "standard_osmotic)"),
                                                   default=0., scale=SCALES.Connection, location="edge",
                                                   kind="NonInertialIntensive")
    water_potential: float = state_variable(**_doc("hPa", "water potential, plant problem: psi (the pressure "
                                                          "potential) + osmotic_potential"),
                                            default=0., scale=SCALES.Compartment, location="node",
                                            kind="NonInertialIntensive", weight="liquid_volume")
    standard_water_potential: float = state_variable(**_doc("hPa", "water potential, standard problem: standard_psi + "
                                                                   "osmotic_potential (with standard_osmotic)"),
                                                     default=0., scale=SCALES.Compartment, location="node",
                                                     kind="NonInertialIntensive", weight="liquid_volume")

    # ── State variables: the solutes' diffusion factors and the nodes' volumes (MechaSolutes) ──
    apoplastic_diffusion_factor: float = state_variable(**_doc("cm", "× the apoplastic diffusivity: the edge's "
                                                               "solute conductance (walls: their geometry × kw ratio; "
                                                               "mature vessels' membranes)"),
                                                        default=0., scale=SCALES.Connection, location="edge",
                                                        kind="NonInertialIntensive")
    symplastic_diffusion_factor: float = state_variable(**_doc("cm", "× the plasmodesmata's diffusivity (their open "
                                                               "cross-section × quantity / the crossed wall's "
                                                               "thickness)"),
                                                        default=0., scale=SCALES.Connection, location="edge",
                                                        kind="NonInertialIntensive")
    membrane_diffusion_factor: float = state_variable(**_doc("cm", "× the membranes' passive diffusivity (length × "
                                                             "height / distance)"),
                                                      default=0., scale=SCALES.Connection, location="edge",
                                                      kind="NonInertialIntensive")
    node_volume: float = state_variable(**_doc("cm3", "the node's volume: a wall's length × height × thickness, a "
                                                      "junction's half, a protoplast's area × height; on a reduced "
                                                      "section's kept node, plus the eliminated nodes' it condenses"),
                                        default=0., scale=SCALES.Compartment, location="node",
                                        kind="NonInertialExtensive")
    liquid_volume: float = state_variable(**_doc("cm3", "the node's liquid water volume: its node_volume, 0 on the "
                                                        "vapour nodes (the air spaces, the stomatal pores, a leaf's "
                                                        "surface) and the soil nodes; the weight of a section's mean "
                                                        "water potential (e.g. a leaf element's, for a gas "
                                                        "exchange's stomata). A reduced section's kept nodes hold "
                                                        "the eliminated nodes' they condense (a root's soil node "
                                                        "too): its total the full anatomy's"),
                                          default=0., scale=SCALES.Compartment, location="node",
                                          kind="NonInertialExtensive")
    contact_distance: float = state_variable(**_doc("µm", "the node's distance to its section's soil contact along "
                                                          "its graph (with pd_spatial_profile > 0; NaN otherwise)"),
                                             default=float("nan"), scale=SCALES.Compartment, location="node",
                                             kind="NonInertialIntensive")

    # ── State variables: the leaves' vapour path, from their evaporating walls to their surface ──
    surface_inflow: float = state_variable(**_doc("cm3 d-1", "on a leaf organ's surface node: minus its "
                                                  "transpiration (the Neumann condition; 0 elsewhere)"),
                                           default=0., scale=SCALES.Compartment, location="node",
                                           kind="NonInertialIntensive")
    air_tail_conductance: float = state_variable(**_doc("cm3 hPa-1 d-1", "wall_air edges: the flow from tail to head "
                                                        "is air_tail_conductance ψ_tail − air_head_conductance ψ_head "
                                                        "+ air_offset (the linearised Kelvin exchange)"),
                                                 default=0., scale=SCALES.Connection, location="edge",
                                                 kind="NonInertialIntensive")
    air_head_conductance: float = state_variable(**_doc("cm3 hPa-1 d-1", "wall_air edges: see air_tail_conductance"),
                                                 default=0., scale=SCALES.Connection, location="edge",
                                                 kind="NonInertialIntensive")
    air_offset: float = state_variable(**_doc("cm3 d-1", "wall_air edges: see air_tail_conductance"),
                                       default=0., scale=SCALES.Connection, location="edge",
                                       kind="NonInertialIntensive")
    vapour_node: float = state_variable(**_doc("", "1 on the air nodes (the air spaces with wall_air edges, the "
                                                   "stomatal pores, the leaf surfaces): their ψ holds their vapour "
                                                   "pressure, relative to saturation"),
                                        default=0., scale=SCALES.Compartment, location="node",
                                        kind="descriptor")

    # ── State variables: the axial junctions between sections ──
    axial_conduit: float = state_variable(**_doc("", "on the axial junctions: 1 between xylem vessels, 2 between "
                                                     "phloem cells (0 elsewhere)"),
                                          default=0., scale=SCALES.Connection, location="edge",
                                          kind="descriptor")
    xylem_axial_conductance: float = state_variable(**_doc("cm3 hPa-1 d-1", "a xylem junction's conductance: "
                                                           "Poiseuille through its two vessels, each over half its "
                                                           "segment, in series (0 without mature xylem)"),
                                                    default=0., scale=SCALES.Connection, location="edge",
                                                    kind="NonInertialIntensive")
    phloem_axial_conductance: float = state_variable(**_doc("cm3 hPa-1 d-1", "a phloem junction's conductance "
                                                            "(Poiseuille, phloem sap); not in the water problem (Q-G1)"),
                                                     default=0., scale=SCALES.Connection, location="edge",
                                                     kind="NonInertialIntensive")

    # ── Inputs and state variables of the plant problem ──
    soil_psi: float = input_variable(**_doc("hPa", "the soil's water potential around the Compartment"), by="Soil",
                                     default=0., scale=SCALES.Compartment, location="node",
                                     kind="intensive")
    conduit_osmotic_potential: float = input_variable(**_doc("hPa", "a reduced section's conduit's osmotic "
                                                             "potential (the plant model's Reduction inputs; NaN: "
                                                             "its kind's)"), by="Solutes", default=float("nan"),
                                                      scale=SCALES.Compartment, location="node",
                                                      kind="NonInertialIntensive")
    tissue_osmotic_potential: object = input_variable(**_doc("hPa", "a reduced section's hidden tissues' osmotic "
                                                             "potentials, {tissue: potential} (the plant model's "
                                                             "Reduction inputs; a tissue not given: its kind's)"),
                                                      by="Solutes", default=None, scale=SCALES.SubOrgan,
                                                      kind="descriptor", dtype=object)
    soil_bulk_conductance: float = input_variable(**_doc("cm3 hPa-1 d-1", "on a root section's soil node "
                                                         "(soil_boundary node_robin): the conductance between it and "
                                                         "the soil at soil_psi (e.g. the bulk soil's around the "
                                                         "section)"), by="Soil", default=0.,
                                                  scale=SCALES.Compartment, location="node", kind="intensive")
    leaf_transpiration: float = input_variable(**_doc("cm3 d-1", "an organ's transpiration, leaving at the leaf surface "
                                                                 "node its anatomy gave it (GRANAP's leaves, Adel's "
                                                                 "blades and sheaths; role Transpiration)"),
                                               by="Transpiration", default=0., scale=SCALES.Organ, location="Organ",
                                               kind="extensive")
    stomatal_aperture: float = input_variable(**_doc("", "a stomatal pore's aperture fraction of its anatomical maximum "
                                                         "(role Stomata: 1 for now)"),
                                              by="Stomata", default=1., scale=SCALES.Compartment, location="node",
                                              kind="NonInertialIntensive")
    root_uptake: float = state_variable(**_doc("cm3 d-1", "water entering a border wall or junction from the soil "
                                                          "(plant problem)"),
                                        default=0., scale=SCALES.Compartment, location="node",
                                        kind="extensive")
    evaporation: float = state_variable(**_doc("cm3 d-1", "water leaving a stomatal pore to the air (plant problem)"),
                                        default=0., scale=SCALES.Compartment, location="node",
                                        kind="extensive")
    collar_kind: float = state_variable(**_doc("", "the collar condition of the node: 1 potential, 2 flow (0: none)"),
                                        default=0., scale=SCALES.Compartment, location="node",
                                        kind="descriptor")
    collar_value: float = state_variable(**_doc("hPa or cm3 d-1", "the collar's potential (kind 1), or the node's "
                                                "inflow (kind 2, negative: water leaving the root)"),
                                         default=0., scale=SCALES.Compartment, location="node",
                                         kind="NonInertialIntensive")

    # ── State variables: the solutions, and the sections' results ──
    standard_psi: float = state_variable(**_doc("hPa", "water potential, in its section's standard problem (its "
                                                       "pressure potential with standard_osmotic 1)"),
                                         default=0., scale=SCALES.Compartment, location="node",
                                         kind="NonInertialIntensive", weight="liquid_volume")
    standard_flow: float = state_variable(**_doc("cm3 d-1", "flow along the edge, from its first node to its second, "
                                                            "in the standard problem"),
                                          default=0., scale=SCALES.Connection, location="edge",
                                          kind="NonInertialIntensive")
    psi: float = state_variable(**_doc("hPa", "water potential, in the plant problem"),
                                default=0., scale=SCALES.Compartment, location="node",
                                kind="NonInertialIntensive", weight="liquid_volume")
    flow: float = state_variable(**_doc("cm3 d-1", "flow along the edge, from its first node to its second, in the "
                                                   "plant problem"),
                                 default=0., scale=SCALES.Connection, location="edge",
                                 kind="NonInertialIntensive")
    standard_transmembrane_fraction: float = state_variable(**_doc("adim", "the membrane's share of the section's "
                                                                   "standard uptake: its standard flow from wall to "
                                                                   "cell / the uptake (MECHA's standard transmembrane "
                                                                   "fraction; 0 on the other edges and membranes_layer "
                                                                   "-1)"),
                                                             default=0., scale=SCALES.Connection, location="edge",
                                                             kind="NonInertialIntensive")
    membrane_layer: float = state_variable(**_doc("", "the membrane's layer row in its section (MECHA's radial "
                                                      "discretisation, from the stele, 0, to the epidermis; -1: not a "
                                                      "membrane, or one without transmembrane flow: an intercellular "
                                                      "space's, a mature xylem vessel's)"),
                                           default=-1., scale=SCALES.Connection, location="edge",
                                           kind="descriptor")
    standard_uptake: float = state_variable(**_doc("cm3 d-1", "the section's water uptake from the soil, standard "
                                                              "problem"),
                                            default=np.nan, scale=SCALES.SubOrgan,
                                            kind="NonInertialExtensive")
    uptake: float = state_variable(**_doc("cm3 d-1", "the section's water uptake from the soil, plant problem"),
                                   default=np.nan, scale=SCALES.SubOrgan, kind="NonInertialExtensive")
    kr: float = state_variable(**_doc("cm hPa-1 d-1", "the section's radial conductivity, MECHA's definition (standard "
                                                      "problem): uptake / (ψ soil − ψ xylem) / surface"),
                               default=np.nan, scale=SCALES.SubOrgan, kind="NonInertialIntensive")
    kr_effective: float = state_variable(**_doc("cm hPa-1 d-1", "the section's radial conductivity in the plant "
                                                                "problem: uptake / (ψ soil − mean ψ of its xylem) / "
                                                                "surface (NaN at barrier 0: no mature xylem)"),
                                         default=np.nan, scale=SCALES.SubOrgan,
                                         kind="NonInertialIntensive")
    Kx: float = state_variable(**_doc("cm4 hPa-1 d-1", "the section's xylem axial conductance (Poiseuille; 0 without "
                                                       "mature xylem, barrier 0)"),
                               default=np.nan, scale=SCALES.SubOrgan, kind="NonInertialIntensive")
    Kx_phloem: float = state_variable(**_doc("cm4 hPa-1 d-1", "the section's phloem axial conductance (Poiseuille: "
                                                              "sieve tubes, or protophloem at barrier 0)"),
                                      default=np.nan, scale=SCALES.SubOrgan,
                                      kind="NonInertialIntensive")

    @rate(include_inactive=True)
    def _cell_rank(self, kind, anatomy_template) -> tuple[np.ndarray, str, np.ndarray]:
        """
        MECHA's cell ranks on the Compartments, and each section's perimeter. They depend on the sections' networks
        (their templates' geometry, cached), read whole from the DataStructure.
        """
        ds = self.data_structure
        rank = self.per_node(lambda geometry: geometry.cell_rank, fill=-1.)
        segments = ds.entity_ids("SubOrgan")
        perimeter = np.full(segments.size, np.nan)
        where = {int(v): i for i, v in enumerate(segments)}
        roots = self.section_roles()[1]
        for segment, (nodes, _, geometry) in self.section_geometries().items():
            if segment in roots:                                       # a root quantity (E1)
                perimeter[where[segment]] = geometry.perimeter
        reduction = self._reduction()
        if reduction is not None:                                      # a reduced group's: its kind's
            kinds = reduction.outputs(ds, "perimeter")
            perimeter = np.where(np.isfinite(kinds), kinds, perimeter)
        return rank, "perimeter", perimeter

    # ── Conductances ──
    PARAMETERS = ()                       # filled below: the names of the numeric parameters (parameter_table)

    def network(self, geometries: dict = None) -> Network:
        """The sections' MECHA geometry on the whole DataStructure, their φ-thickened cells included."""
        ds = self.data_structure
        geometries = self.section_geometries() if geometries is None else geometries
        items = list(geometries.values())
        layers = self.parameter_table(self.PARAMETERS, "node")
        phi = [phi_thick_cells(geo, int(layers["n_phi_layers"][nodes[0]]), int(layers["phi_type"][nodes[0]]))
               for nodes, _, geo in items]
        net = assemble(ds.n_nodes(), ds.n_edges(), items, phi)
        view = ds.to_graph_view()
        set_edge_ends(net, view.tail, view.head, np.asarray(ds.get("network_path")))
        set_wall_thickness(net, np.asarray(ds.get("wall_thickness"), dtype=float))
        net.soil_contact_cell = (np.asarray(ds.get("soil_contact_cell"), dtype=float) > 0) & (net.kind == 0)
        net.measured_axial_conductivity = np.where(net.kind == 0, np.asarray(ds.get("measured_axial_conductivity"),
                                                                             dtype=float), np.nan)
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

    @rate(include_inactive=True, memo=True)
    def _conductance(self, network_path, barrier, length, anatomy_template, leaf_transpiration, stomatal_aperture,
                     pore_width, pore_depth, kind, cell_area, wall_thickness, soil_contact_cell,
                     measured_axial_conductivity) \
            -> tuple[np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray,
                     str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str,
                     np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray,
                     str, np.ndarray, str, np.ndarray, str, np.ndarray]:
        """
        Every edge's conductance, and the boundary conductances of the nodes (soil, axial), from MECHA's geometry,
        the segments' barriers and heights (their lengths), and the plants' parameters; the standard problem's
        boundary potentials (the axial boundary only where its potential is given: not at barrier 0 by default).
        Computed again only when one of these, the parameters, the settings or the structure changed (memo=True).
        """
        return self._conductance_values(network_path, barrier, length, anatomy_template, leaf_transpiration,
                                        stomatal_aperture, pore_width, pore_depth, kind, cell_area)

    def _conductance_values(self, network_path, barrier, length, anatomy_template, leaf_transpiration,
                            stomatal_aperture, pore_width, pore_depth, kind, cell_area) -> tuple:
        geometries = self.section_geometries()
        net = self.network(geometries)
        height_node = np.asarray(length, dtype=float) * 1.0E6                     # m -> µm
        self._check_heights(height_node)
        self._check_soil_boundary()
        P_node, P_edge = self.parameter_table(self.PARAMETERS, "node"), self.parameter_table(self.PARAMETERS, "edge")
        a_cortex, b_cortex = self.cortex_gradient(geometries, height_node, P_node)
        tail = np.maximum(net.a, 0)
        P_edge["a_cortex"], P_edge["b_cortex"] = a_cortex[tail], b_cortex[tail]
        barrier_node = np.asarray(barrier)
        factors = {}
        K, in_matrix = edge_conductances(net, P_edge, barrier_node[tail], height_node[tail], factors=factors)
        soil, axial, _ = boundary_terms(net, P_node, barrier_node, height_node)
        axial_psi = np.where(barrier_node > 0, P_node["standard_psi_xylem"], P_node["standard_psi_sieve"])
        tip_flow = (barrier_node == 0) & np.isfinite(P_node["standard_sieve_flow"])  # the protophloem's flow instead
        axial_psi = np.where(tip_flow, np.nan, axial_psi)
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
        inflow = self._standard_axial_inflow(geometries, P_node, flow_condition, barrier_node)
        root = self._root_nodes(geometries)
        soil = np.where(root, soil, 0.)                               # stems and leaves: no soil (E1)
        K = np.where(in_matrix, K, 0.)
        ds = self.data_structure
        kind = np.asarray(kind)
        pore_width, pore_depth = np.asarray(pore_width, dtype=float), np.asarray(pore_depth, dtype=float)
        view = ds.to_graph_view()
        self._reduced_vapour_path(net, view)
        K_vapour, K_wall, K_air, offset, wall_end, air_end = vapour_conductances(
            net, P_edge, height_node[tail], pore_width, pore_depth, np.asarray(cell_area, dtype=float),
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
        reduction = ds.reduction
        if reduction is not None and reduction.library:          # a reduction's library: no boundary data needed
            surface_inflow = np.zeros(ds.n_nodes())
        else:
            surface_inflow = -ds.map_to(leaf_transpiration, "Organ", "node", "split", weights=surface)
        standard = (P_node["standard_solve"] > 0) & root
        # the soil boundary: on each contact node (per_node), or on each root section's soil node through its
        # Connections, which carry the contact nodes' soil conductances
        path_codes = np.asarray(network_path)
        soil_links = path_codes == SOIL
        node_mode = self.soil_boundary != "per_node"
        if node_mode and root.any() and not (kind == SOIL_NODE).any():
            raise ValueError(f"MechaHydraulics: soil_boundary {self.soil_boundary!r} needs a soil node per root "
                             "section (GRANAP's scenario key soil_node)")
        K = np.where(soil_links, soil[view.head] if node_mode else 0., K)
        plant_soil = np.zeros_like(soil) if node_mode else soil
        soil_node_kind = np.where(kind == SOIL_NODE, 3. if self.soil_boundary == "node_robin" else 1., 0.)
        for name in ("apoplastic", "symplastic", "membrane"):
            factors[name] = np.where(soil_links, 0., factors[name])
        volume = self._node_volumes(net, P_node, height_node, kind)
        volume, liquid = self._condensed_volumes(volume, np.where(vapour | surface | (kind == SOIL_NODE), 0., volume),
                                                 P_node, height_node, vapour | surface)
        return (K, "soil_conductance", soil, "plant_soil_conductance", plant_soil, "soil_node_kind", soil_node_kind,
                "axial_boundary_conductance", axial,
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
                "node_volume", volume, "liquid_volume", liquid,
                "contact_distance", factors["contact_distance"])

    def _check_heights(self, height_node) -> None:
        """A section's anatomy needs a length: at zero height, its walls and membranes conduct nothing (their
        conductances ∝ the height), its nodes cut off from the network, and its conduits' Poiseuille conductances
        (∝ 1 / the height) infinite. GRANAP gives no anatomy to a section without length (a pass-through until it
        grows)."""
        flat = ~(np.asarray(height_node) > 0)
        if flat.any():
            ds = self.data_structure
            sections = np.unique(np.asarray(ds.owner("SubOrgan"))[flat])
            ids = ds.entity_ids("SubOrgan")[sections[sections >= 0]]
            raise ValueError(f"MechaHydraulics: the sections (SubOrgans) {ids.tolist()[:10]} have an anatomy and no "
                             "length (0 or none given): MECHA's section height. Give their length, or no anatomy "
                             "until they have one (GRANAP's, a pass-through)")

    def _check_soil_boundary(self) -> None:
        """The per_node soil boundary acts on each root section's contact nodes, which a reduced section eliminates
        (it keeps its conduits and its soil node): its roots would be cut off from the soil."""
        if self.soil_boundary != "per_node":
            return
        reduction = self._reduction()
        groups = reduction.reduced_groups(self.data_structure) if reduction is not None else {}
        if not groups:
            return
        ds = self.data_structure
        ids = ds.entity_ids("SubOrgan")
        roots = self.section_roles()[1]
        reduced_roots = [int(ids[i]) for i in groups if int(ids[i]) in roots]
        if reduced_roots:
            raise ValueError(f"MechaHydraulics: soil_boundary 'per_node' with reduced root sections "
                             f"({len(reduced_roots)}, e.g. {reduced_roots[:5]}): a reduced section keeps its conduits "
                             "and its soil node, not its contact nodes, which per_node puts the soil on. Use "
                             "soil_boundary 'node_dirichlet' or 'node_robin' (GRANAP's soil_node), or leave the "
                             "roots unreduced (the scenario's reduced_where without RootSegment and RootApex)")

    def _reduced_vapour_path(self, net, view) -> None:
        """The reduced sections' kept vapour path (their wall-air and air-link Connections, explicit terms) in *net*,
        from their own values (GRANAP's, mm -> µm): its conductances are local, computed as a full section's."""
        reduced = self._reduced_nodes()
        if not reduced.any():
            return
        ds = self.data_structure
        path = np.asarray(ds.get("network_path"))
        edges = np.flatnonzero(reduced[view.tail] & reduced[view.head] & np.isin(path, (WALL_AIR, AIR_LINK)))
        if edges.size == 0:
            return
        net.in_section[edges] = True
        net.edge_length[edges] = np.asarray(ds.get("edge_length"), dtype=float)[edges] * 1.0E3
        net.edge_dist[edges] = np.asarray(ds.get("edge_dist"), dtype=float)[edges] * 1.0E3
        nodes = np.flatnonzero(reduced)
        net.kind[nodes] = np.asarray(ds.get("kind"), dtype=float)[nodes].astype(np.int64)

    def relative_x(self, geometries: dict) -> np.ndarray:
        """Per node, its x relative to its section's cells' x range (0 left, 1 right; the legacy's x_rel, from the
        cells' centroids: doc/legacy.rst, ME-22)."""
        def relative(geo):
            x_cells = geo.x[geo.cells]
            low, high = x_cells.min(), x_cells.max()
            return (geo.x - low) / (high - low) if high > low else np.zeros(np.size(geo.x))
        return self.per_node(relative, geometries, fill=0.)

    @rate(include_inactive=True)
    def _elongation_sink(self, barrier, anatomy_template) -> tuple[np.ndarray, str, np.ndarray]:
        """The elongation sink of the sections without barriers (barrier 0), in the plant problem, and in the
        standard problem of the plants with standard_elongation."""
        geometries = self.section_geometries()
        net = self.network(geometries)
        P_node = self.parameter_table(self.PARAMETERS, "node")
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

    def _condensed_volumes(self, volume, liquid, P_node, height, vapour) -> tuple:
        """
        (node_volume, liquid_volume) with the reduced sections' (solve.reduction): a kept node stands for the
        eliminated nodes condensed into it. Its volume is its own (its full section's, computed as a full section's
        from the full template: template_section) plus the eliminated nodes' it condenses, each eliminated node's
        volume shared among its section's kept nodes of a liquid potential (the conduits, the evaporating walls, a
        root's soil node: its potential its contact walls'; not the vapour path's air, *vapour*) as its potential
        depends on theirs: the Kron reduction's recovery weights (x_I = W x_K, its row of W over those kept nodes,
        normalised to 1). So each section's volume and liquid volume are the full one's, and a liquid-weighted mean of
        the kept nodes' potentials the full section's when its interior has no sources of its own (close to it
        otherwise: a steady state's local gradients). The eliminated nodes are liquid (the vapour path is kept:
        vapour_path); a reduced root's soil node, of no volume of its own, holds the liquid of the tissue bound to the
        soil. The volumes per unit height are each kind's (_condensed_unit_volume, computed once per kind on its
        library), times the section's height. Unchanged without a reduction and in a reduction's library.
        """
        reduction = self._reduction()
        if reduction is None or reduction.library:
            return volume, liquid
        ds = self.data_structure
        blocks = reduction.kept_outputs(ds, "condensed_unit_volume", self._plant_problem_name(reduction))
        if not blocks:
            return volume, liquid
        volume, liquid = volume.copy(), liquid.copy()
        height = np.asarray(height, dtype=float)
        for kept, unit in blocks:                        # per template: its groups × their kept nodes at once
            volume[kept] = unit * height[kept[:, :1]]
            liquid[kept] = np.where(vapour[kept], 0., volume[kept])
        return volume, liquid

    @kind_derived("condensed_unit_volume")
    def _condensed_unit_volume(self, kind, full_template, kron, nodes) -> np.ndarray:
        """Per unit height (µm), each kept node's volume of a kind of the plant problem (_condensed_volumes): its own
        (its full template's) plus its share of the eliminated nodes' (kron.shares over the kept nodes of a liquid
        potential: two single solves while the kind's factorisation is live, no dense recovery). Run on the
        reduction's library, once per kind (metafspm's kind_derived): its nodes the full group's there, their
        vapour path and the plant's thickness parameters the library's."""
        ds = self.data_structure
        P_node = self.parameter_table(self.PARAMETERS, "node")
        kept_nodes = nodes[kron.kept_nodes]
        reference = kept_nodes[0] if kept_nodes.size else nodes[0]
        plant = (float(P_node["thickness_source"][reference]), float(P_node["thickness"][reference]))
        unit = kind.contract(self._template_unit_volumes(full_template, plant))   # a merged node's: its members' sum
        vapour = (np.asarray(ds.get("vapour_node")) > 0) | (np.asarray(ds.get("kind")) == LEAF_SURFACE)
        return unit[kron.kept_nodes] + kron.shares(~vapour[kept_nodes], unit[kron.interior_nodes])

    def _template_unit_volumes(self, full, plant) -> np.ndarray:
        """Per Compartment of the full template *full*, its volume per unit height (cm³ µm⁻¹, _node_volumes at a height
        of 1 µm; 0 on the leaf surface and the soil node), with the plant's thickness parameters *plant*."""
        scales = self.data_structure.structure.scales
        section, nodes, edges, tail, head = template_section(full, scales.Compartment, scales.Connection)
        compartments = full.rows_at_scale(scales.Compartment)
        net = assemble(compartments.size, tail.size, [(nodes, edges, section_geometry(section))])
        set_edge_ends(net, tail, head, net.path)
        set_wall_thickness(net, template_values(full, "wall_thickness", compartments))
        thickness_source, thickness = plant
        parameters = {"thickness_source": np.full(compartments.size, thickness_source),
                      "thickness": np.full(compartments.size, thickness)}
        return self._node_volumes(net, parameters, 1., net.kind)

    @staticmethod
    def _liquid_equivalent(value, vapour_node, leaf_temperature):
        """The water potential (hPa) of the vapour nodes: Kelvin's liquid equivalent of their relative vapour
        pressure x, ln(x) / α (α = V_w / (R T), T the leaves' temperature per node)."""
        alpha = kelvin_alpha(np.asarray(leaf_temperature, dtype=float))
        vapour = np.asarray(vapour_node) > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            kelvin = np.log(np.maximum(value, 1e-300)) / alpha
        return vapour, kelvin

    def _standard_axial_inflow(self, geometries, P_node, flow_condition, barrier) -> np.ndarray:
        """
        The standard problem's flow conditions: the flow into the xylem (standard_axial_condition 2) split among the
        vessels (the network's, by cgroup) as their areas; the flow into the phloem (standard_sieve_flow, where its
        potential is NaN; at barrier 0, in place of it) among the protophloem's cells (barrier 0, each once:
        SectionGeometry.protophloem_cells) or the sieve tubes (barrier > 0) as their
        areas. The legacy (``_distribute_xylem_flow``) computes the xylem's ratios on its cell manager's xylem (by
        tissue) and gives them by rank to the network's, so that a vessel may get another's (doc/legacy.rst, o:
        fixed); its phloem's, the protosieve cells' areas, also by rank to the sieve tubes (ME-42: fixed).
        """
        out = np.zeros(self.data_structure.n_nodes())
        for nodes, _, geo in geometries.values():
            v = nodes[0]
            if flow_condition[v]:
                vessels = geo.xylem_cells
                total = 0.
                for c in vessels:
                    total += float(geo.cell_area[c])
                for c in vessels:
                    out[nodes[c]] = -float(P_node["standard_axial_flow"][v]) * (float(geo.cell_area[c]) / total)
            flow = float(P_node["standard_sieve_flow"][v])
            if np.isnan(flow):
                continue
            mature = barrier[v] > 0
            if mature and not np.isnan(float(P_node["standard_psi_sieve_tube"][v])):
                raise ValueError("standard_sieve_flow and the sieve tubes' potential (standard_psi_sieve_tube, at "
                                 "barrier > 0) are both given: one condition only")
            conduits = geo.sieve_cells if mature else geo.protophloem_cells
            area = np.asarray(geo.cell_area[conduits], dtype=float)
            out[nodes[conduits]] = -flow * area / area.sum()
        return out

    @rate(include_inactive=True)
    def _osmotic_drive(self, reflection_coefficient, osmotic_potential, standard_osmotic) \
            -> tuple[np.ndarray, str, np.ndarray]:
        """
        Each edge's osmotic drive σ (π_tail − π_head) (the membranes': the legacy's ``rhs_o``, as a flow K σ Δπ added
        to K Δψ), and the standard problem's (its plants with standard_osmotic; the parameter comes on the edges, the
        location of the step's first argument).
        """
        view = self.data_structure.to_graph_view()
        pi = np.asarray(osmotic_potential, dtype=float)
        drive = np.asarray(reflection_coefficient, dtype=float) * (pi[view.tail] - pi[view.head])
        standard = np.asarray(standard_osmotic) > 0
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

    def _junction_conductance(self, conduit, axial_conduit, cell_area, length, measured_axial_conductivity,
                              viscosity, barrier=None) -> np.ndarray:
        """Poiseuille through each junction of *conduit* (its two cells, each over half its segment, in series); 0 on
        the other edges. *viscosity*: per edge."""
        view = self.data_structure.to_graph_view()
        edges = np.flatnonzero(np.asarray(axial_conduit) == conduit)
        a, b = view.tail[edges], view.head[edges]
        area = np.asarray(cell_area, dtype=float) * 1.0E6                          # mm2 -> µm2
        half = np.asarray(length, dtype=float) * 1.0E6 / 2                         # m -> µm
        measured = np.asarray(measured_axial_conductivity, dtype=float)
        viscosity = np.asarray(viscosity, dtype=float)[edges]
        k_a, k_b = (np.where(np.isfinite(measured[n]), measured[n] / (half[n] / 1.0E04),
                             poiseuille(area[n], half[n], viscosity)) for n in (a, b))
        with np.errstate(divide="ignore"):
            K = np.where((k_a > 0) & (k_b > 0), 1 / (1 / k_a + 1 / k_b), 0.)
        if barrier is not None:
            barrier = np.asarray(barrier)
            K = np.where((barrier[a] > 0) & (barrier[b] > 0), K, 0.)
        out = np.zeros(np.asarray(axial_conduit).size)
        out[edges] = K
        return out

    @rate(include_inactive=True)
    def _xylem_axial_conductance(self, axial_conduit, cell_area, length, barrier, measured_axial_conductivity,
                                 xylem_viscosity) -> np.ndarray:
        """The xylem junctions' conductances (water); none where either section has no mature xylem (barrier 0)."""
        return self._junction_conductance(1, axial_conduit, cell_area, length, measured_axial_conductivity,
                                          xylem_viscosity, barrier)

    @rate(include_inactive=True)
    def _phloem_axial_conductance(self, axial_conduit, cell_area, length, measured_axial_conductivity,
                                  phloem_viscosity) -> np.ndarray:
        """The phloem junctions' conductances (phloem sap)."""
        return self._junction_conductance(2, axial_conduit, cell_area, length, measured_axial_conductivity,
                                          phloem_viscosity)

    @rate(include_inactive=True)
    def _collar_kind(self, kind, cgroup, cell_area, collar_condition, collar_psi, collar_flow) \
            -> tuple[np.ndarray, str, np.ndarray]:
        """
        The collar condition (Q-G2) on the xylem vessels of each root system's first section (a segment without a
        parent): the collar's potential (collar_condition 1), or the flow collar_flow leaving the root, split among
        the vessels as their Poiseuille conductances (∝ area², collar_condition 2).
        """
        ds = self.data_structure
        first = np.isin(ds.entity_ids("SubOrgan"), list(self.section_roles()[0]))
        vessels = np.isin(np.asarray(cgroup), XYLEM_CGROUPS) & (ds.map_to(first, "SubOrgan", "node", "broadcast") > 0)
        condition = np.where(vessels, np.asarray(collar_condition), 0.)
        squared = np.where(vessels, np.asarray(cell_area, dtype=float) ** 2, 0.)
        with np.errstate(divide="ignore", invalid="ignore"):
            share = squared / ds.map_to(ds.map_to(squared, "node", "SubOrgan", "sum"), "SubOrgan", "node", "broadcast")
        kinds = np.where((condition == 1) | (condition == 2), condition, 0.)
        values = np.select([condition == 1, condition == 2], [np.asarray(collar_psi, dtype=float),
                                                              -np.asarray(collar_flow, dtype=float) * share], 0.)
        return kinds, "collar_value", values

    # ── The standard problem: each section alone, MECHA's definition of kr (Q-M5) ──
    # tol (cm3 d-1): the pinned walls' penalty puts their residuals' round-off at ~1e-9; the systems are linear,
    # solved by the first Newton step
    @graph_system(node_unknowns=["standard_psi"], solver="newton", max_iterations=3, tolerance=1e-8, schedule_as="state",
                  filters={"in_standard_problem": ">0"}, symmetric=True,
                  condense=lambda self: "SubOrgan" if self.plant_solver == "condensed" else None)
    class _standard_problem:
        """
        Each section alone (the junctions have no conductance here): its border walls and junctions exchange with
        the soil, its xylem vessels (barrier > 0) or protophloem cells (barrier 0) with the axial boundary.
        """

        @node_balance(field="standard_psi")
        def _balance(self, standard_psi, conductance, standard_osmotic_drive, air_tail_conductance,
                     air_head_conductance, air_offset, standard_elongation_sink):
            B = self.graph_view.incidence
            drop = np.asarray(B.T @ standard_psi).reshape(-1) + standard_osmotic_drive
            q = _wall_air_flow(B, standard_psi, air_tail_conductance, air_head_conductance, air_offset)
            return np.asarray(B @ (conductance * drop + q)).reshape(-1) + standard_elongation_sink

        soil = boundary_set(filters={"soil_conductance": ">0"}, kind="robin", value="standard_soil_psi",
                            weight="soil_conductance", field="standard_psi")
        axial = boundary_set(filters={"axial_boundary_conductance": ">0"}, kind="robin", value="standard_axial_psi",
                             weight="axial_boundary_conductance", field="standard_psi")
        axial_flow = boundary_set(filters={"standard_axial_inflow": "!=0"}, kind="neumann", value="standard_axial_inflow",
                                  field="standard_psi")

        @graph_jacobian
        def _jacobian(self, conductance, air_tail_conductance, air_head_conductance):
            return (self.graph_view.laplacian(conductance)
                    + _wall_air_jacobian(self.graph_view.incidence, air_tail_conductance, air_head_conductance)).tocsr()

        @graph_output("standard_flow", location="edge")
        def _standard_flow(self, standard_psi, conductance, standard_osmotic_drive, air_tail_conductance,
                           air_head_conductance, air_offset):
            B = self.graph_view.incidence
            return conductance * (np.asarray(B.T @ standard_psi).reshape(-1) + standard_osmotic_drive) + \
                _wall_air_flow(B, standard_psi, air_tail_conductance, air_head_conductance, air_offset)

    # ── The plant problem: the sections joined by their xylem junctions ──
    @graph_system(node_unknowns=["psi"], solver="newton", max_iterations=3, tolerance=1e-8, schedule_as="state",
                  condense=lambda self: "SubOrgan" if self.plant_solver == "condensed" else None,
                  conserved={"water": PLANT_WATER})
    class _plant_problem:
        """
        The whole plant: every section's edges and the xylem junctions between sections; the soil around the root
        surface (soil_psi), the collar's condition on the first root section's vessels; the leaves' vapour path,
        from their evaporating walls through their air spaces and stomata to their surface nodes, where their organs'
        transpiration leaves (a Neumann condition). Its water is conserved (PLANT_WATER: nothing held, what the roots
        take up leaves by transpiration and to the elongating cells; a collar condition's flow is not counted).
        """

        @node_balance(field="psi")
        def _balance(self, psi, conductance, xylem_axial_conductance, osmotic_drive, elongation_sink):
            B = self.graph_view.incidence
            K = conductance + xylem_axial_conductance
            return np.asarray(B @ (K * (np.asarray(B.T @ psi).reshape(-1) + osmotic_drive))).reshape(-1) + \
                elongation_sink

        @explicit_term(field="psi", jacobian="_vapour_jacobian")
        def _vapour(self, psi, air_tail_conductance, air_head_conductance, air_offset):
            """The leaves' wall-air flows (nonlinear: kept explicit by a reduction, their nodes kept)."""
            B = self.graph_view.incidence
            return np.asarray(B @ _wall_air_flow(B, psi, air_tail_conductance, air_head_conductance,
                                                 air_offset)).reshape(-1)

        def _vapour_jacobian(self, air_tail_conductance, air_head_conductance):
            return _wall_air_jacobian(self.graph_view.incidence, air_tail_conductance, air_head_conductance).tocsr()

        @input_jacobian("osmotic_potential")
        def _osmotic_jacobian(self, conductance, xylem_axial_conductance, reflection_coefficient):
            """The balance's derivative with respect to the osmotic potentials: B (K σ) Bᵀ (the drive σ Bᵀπ)."""
            return self.graph_view.laplacian((conductance + xylem_axial_conductance) * reflection_coefficient)

        soil = boundary_set(filters={"plant_soil_conductance": ">0"}, kind="robin", value="soil_psi",
                            weight="plant_soil_conductance", field="psi")
        soil_nodes = boundary_set(filters={"soil_node_kind": ">0"}, kinds="soil_node_kind", value="soil_psi",
                                  weight="soil_bulk_conductance", field="psi")
        collar = boundary_set(filters={"collar_kind": ">0"}, kinds="collar_kind", value="collar_value", weight=0.,
                              field="psi")
        leaf_surfaces = boundary_set(filters={"surface_inflow": "<0"}, kind="neumann", value="surface_inflow",
                                     field="psi")

        @graph_jacobian
        def _jacobian(self, conductance, xylem_axial_conductance):
            return self.graph_view.laplacian(conductance + xylem_axial_conductance)

        @graph_output("flow", location="edge")
        def _flow(self, psi, conductance, xylem_axial_conductance, osmotic_drive, air_tail_conductance,
                  air_head_conductance, air_offset):
            B = self.graph_view.incidence
            return (conductance + xylem_axial_conductance) * (np.asarray(B.T @ psi).reshape(-1) + osmotic_drive) + \
                _wall_air_flow(B, psi, air_tail_conductance, air_head_conductance, air_offset)

        @graph_output("root_uptake", location="node")
        def _root_uptake(self, psi, plant_soil_conductance, soil_psi, conductance, kind):
            """The water entering from the soil: at each contact node (per_node), or at each soil node, its balance:
            the flow through its Connections into its section (and, in a reduced section, its interior's term)."""
            B = self.graph_view.incidence
            out = np.asarray(B @ (conductance * np.asarray(B.T @ psi).reshape(-1))).reshape(-1)
            reduction = self._reduction()
            if reduction is not None:
                out = out + reduction.term_values(self, self._plant_problem_name(reduction), self.data_structure, psi)
            return plant_soil_conductance * (soil_psi - psi) + np.where(np.asarray(kind) == SOIL_NODE, out, 0.)

        @graph_output("evaporation", location="node")
        def _evaporation(self, psi, conductance, network_path):
            """Each stomatal pore's flow to its leaf's surface (the stomatal Connections run from the pore)."""
            B = self.graph_view.incidence
            stomatal = np.asarray(network_path) == STOMATAL
            q = np.where(stomatal, conductance * np.asarray(B.T @ psi).reshape(-1), 0.)
            return np.asarray(B.maximum(0) @ q).reshape(-1)

        @graph_output("water_potential", location="node")
        def _water_potential(self, psi, osmotic_potential, vapour_node, leaf_temperature):
            vapour, kelvin = self._liquid_equivalent(psi, vapour_node, leaf_temperature)
            return np.where(vapour, kelvin, psi + osmotic_potential)

    # ── The sections' results ──
    @totalstate(include_inactive=True)
    def _kr(self, standard_psi, psi, soil_conductance, soil_psi, barrier, length, osmotic_potential,
            standard_soil_psi, root_uptake, kind, cgroup, tissue, cell_area, measured_axial_conductivity, vapour_node,
            standard_solve, standard_psi_soil, standard_psi_xylem, standard_psi_sieve, standard_axial_condition,
            standard_sieve_flow, standard_osmotic, leaf_temperature, xylem_viscosity, phloem_viscosity) \
            -> tuple[np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray,
                     str, np.ndarray]:
        """
        Each section's uptake and radial conductivity: MECHA's (its standard problem, the legacy's operations) and
        the effective ones of the plant problem; its axial conductances (Poiseuille); the standard problem's water
        potentials (with the osmotic potentials where it has their terms, standard_osmotic).
        """
        ds = self.data_structure
        P = {name: np.asarray(values) for name, values in (
            ("standard_solve", standard_solve), ("standard_psi_soil", standard_psi_soil),
            ("standard_psi_xylem", standard_psi_xylem), ("standard_psi_sieve", standard_psi_sieve),
            ("standard_axial_condition", standard_axial_condition), ("standard_sieve_flow", standard_sieve_flow))}
        soil, standard, plant = (np.asarray(v, dtype=float) for v in (soil_conductance, standard_psi, psi))
        soil_psi = np.asarray(soil_psi, dtype=float)
        barrier, height = np.asarray(barrier), np.asarray(length, dtype=float) * 1.0E6
        segments = ds.entity_ids("SubOrgan")
        names = ("kr", "standard_uptake", "uptake", "kr_effective", "Kx", "Kx_phloem")
        out = {name: np.full(segments.size, np.nan) for name in names}
        where = {int(v): i for i, v in enumerate(segments)}
        cgroup = np.asarray(cgroup)
        measured = np.asarray(measured_axial_conductivity, dtype=float)
        node_kind = np.asarray(kind)
        roots = self.section_roles()[1]
        soil_side = soil_psi.copy()                    # the potential on the soil side of each contact node
        if self.soil_boundary != "per_node":           # its section's soil node's
            soil_nodes = np.flatnonzero(node_kind == SOIL_NODE)
            section = np.asarray(ds.owner("SubOrgan"))
            at_section = np.full(segments.size, -1, dtype=np.int64)
            at_section[section[soil_nodes]] = soil_nodes
            here = at_section[section] >= 0
            soil_side[here] = plant[at_section[section[here]]]
        for segment, (nodes, edges, geo) in self.section_geometries().items():
            i, v = where[segment], nodes[0]
            b, h = int(barrier[v]), float(height[v])
            out["Kx"][i], out["Kx_phloem"][i] = axial_conductances(geo, h, b, measured[nodes],
                                                                   (xylem_viscosity[v], phloem_viscosity[v]))
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
                elif b == 0 and np.isfinite(P["standard_sieve_flow"][v]):  # the protophloem's: its cells' mean
                    protophloem = geo.protophloem_cells
                    psi_axial = 0.0
                    for c in protophloem:
                        psi_axial += standard[nodes[c]] / len(protophloem)
                    psi_axial = float(psi_axial)
                out["standard_uptake"][i] = soil_uptake(geo, soil[nodes], standard[nodes], standard_soil[nodes])
                out["kr"][i] = radial_conductivity(out["standard_uptake"][i], psi_soil, psi_axial, geo.perimeter, h)
            contact = nodes[soil[nodes] > 0]
            out["uptake"][i] = float(np.sum(soil[contact] * (soil_side[contact] - plant[contact])))
            vessels = nodes[np.isin(cgroup[nodes], XYLEM_CGROUPS)]
            if contact.size and vessels.size and b > 0:                # no mature xylem at barrier 0 (D3)
                out["kr_effective"][i] = radial_conductivity(out["uptake"][i], soil_side[contact].mean(),
                                                             plant[vessels].mean(), geo.perimeter, h)
        reduction = self._reduction()                                 # the reduced sections: their kind's kr
        groups = np.asarray(sorted(reduction.reduced_groups(ds)), dtype=np.int64) if reduction is not None else ()
        if len(groups):                                               # every reduced section at once
            section = np.asarray(ds.owner("SubOrgan"))
            position = np.full(segments.size, -1, dtype=np.int64)
            position[groups] = np.arange(groups.size)
            member = np.where(section >= 0, position[np.maximum(section, 0)], -1)      # each node's reduced section
            nodes = np.flatnonzero(member >= 0)
            first = np.full(groups.size, -1, dtype=np.int64)
            first[member[nodes][::-1]] = nodes[::-1]                    # each reduced section's first node
            has = first >= 0
            groups, first = groups[has], first[has]
            position[:] = -1
            position[groups] = np.arange(groups.size)
            member = np.where(section >= 0, position[np.maximum(section, 0)], -1)
            out["kr"][groups] = reduction.outputs(ds, "kr")[groups]
            out["standard_uptake"][groups] = reduction.outputs(ds, "standard_uptake")[groups]
            kind_perimeter = reduction.outputs(ds, "perimeter")[groups]
            b, h = barrier[first].astype(int), height[first].astype(float)
            out["Kx"][groups], out["Kx_phloem"][groups] = conduit_axial_conductances(
                member, groups.size, tissue, cell_area, h, b, measured,
                (np.asarray(xylem_viscosity, dtype=float)[first], np.asarray(phloem_viscosity, dtype=float)[first]))
            at = member >= 0

            def per_section(mask, values):
                chosen = at & mask
                return (np.bincount(member[chosen], weights=values[chosen], minlength=groups.size),
                        np.bincount(member[chosen], minlength=groups.size))
            uptake, soil_count = per_section(node_kind == SOIL_NODE, np.asarray(root_uptake, dtype=float))
            with_soil = soil_count > 0
            out["uptake"][groups[with_soil]] = uptake[with_soil]
            soil_psi_sum, _ = per_section(node_kind == SOIL_NODE, plant)
            vessel_psi_sum, vessel_count = per_section(np.isin(cgroup, XYLEM_CGROUPS), plant)
            effective = with_soil & (vessel_count > 0) & (b > 0)
            if effective.any():
                out["kr_effective"][groups[effective]] = radial_conductivity(
                    uptake[effective], soil_psi_sum[effective] / soil_count[effective],
                    vessel_psi_sum[effective] / vessel_count[effective], kind_perimeter[effective], h[effective])
        with_osmotic = np.asarray(standard_osmotic) > 0
        water = standard + np.where(with_osmotic, np.asarray(osmotic_potential, dtype=float), 0.)
        vapour, kelvin = self._liquid_equivalent(standard, vapour_node, leaf_temperature)
        water = np.where(vapour, kelvin, water)
        return tuple(x for name in names for x in ((out[name],) if name == "kr" else (name, out[name]))) + (
            "standard_water_potential", water)

    @totalstate(include_inactive=True, memo=True)
    def _standard_transmembrane_fraction(self, kind, standard_flow, standard_uptake, barrier, network_path,
                                         standard_solve) \
            -> tuple[np.ndarray, str, np.ndarray]:
        """
        Each membrane's standard transmembrane fraction (its standard flow from wall to cell / its section's standard
        uptake: ``Mecha.standard_transmembrane_fractions``) and its layer row (postprocess.transmembrane_layers sums
        them by layer), on the root sections solving the standard problem. Computed again only when one of these, the
        parameters, the settings or the structure changed (memo=True).
        """
        from openalea.mecha.osmotic import membrane_rows             # osmotic imports this module
        ds = self.data_structure
        view = ds.to_graph_view()
        fraction, layer = np.zeros(ds.n_edges()), np.full(ds.n_edges(), -1.)
        flow, path, kind = np.asarray(standard_flow, dtype=float), np.asarray(network_path), np.asarray(kind)
        uptake, barrier = np.asarray(standard_uptake, dtype=float), np.asarray(barrier)
        where = {int(v): i for i, v in enumerate(ds.entity_ids("SubOrgan"))}
        standard = np.asarray(standard_solve) > 0                     # per node (the step's first argument's)
        roots = self.section_roles()[1]
        for segment, (nodes, edges, geo) in self.section_geometries().items():
            if segment not in roots or not standard[nodes[0]]:
                continue
            membranes = edges[path[edges] == MEMBRANE]
            tail, head = view.tail[membranes], view.head[membranes]
            wall_is_tail = kind[tail] != 0
            walls, cells = np.where(wall_is_tail, tail, head), np.where(wall_is_tail, head, tail)
            rows = membrane_rows(geo, int(barrier[nodes[0]]), local_index(nodes, walls), local_index(nodes, cells))
            into_cell = np.where(wall_is_tail, flow[membranes], -flow[membranes])
            fraction[membranes] = np.where(rows >= 0, into_cell / uptake[where[segment]], 0.)
            layer[membranes] = rows
        return fraction, "membrane_layer", layer


MechaHydraulics.PARAMETERS = MechaHydraulics.declared_parameters()


def vapour_path(template) -> np.ndarray:
    """The vapour path's nodes of a SectionTemplate: the ends of its wall-air and air-link Connections (a leaf's
    evaporating walls, air spaces and pores; none in a section without them): a reduction's keep rule (the nodes of
    the explicit vapour terms)."""
    out = np.zeros(template.size, dtype=bool)
    edges = np.flatnonzero(vapour_edges(template))
    for link in ("n_id_a", "n_id_b"):
        where, ends = template.properties[link]
        position = np.searchsorted(where, edges)
        out[np.asarray(ends, dtype=np.int64)[position]] = True
    return out


def vapour_edges(template) -> np.ndarray:
    """The vapour path's Connections of a SectionTemplate (wall-air and air-link): a reduction's explicit edges."""
    out = np.zeros(template.size, dtype=bool)
    rows, values = template.properties.get("network_path", (np.zeros(0, dtype=np.int64), np.zeros(0)))
    out[rows] = np.isin(np.asarray(values, dtype=float), (WALL_AIR, AIR_LINK))
    return out


def air_space_of_wall(template) -> np.ndarray:
    """Per vertex of a SectionTemplate, the air space its evaporating wall faces (the row of the other end of its
    wall-air Connection; a leaf's walls each face one), -1 for the others: the walls of one air space are one node
    with plant_reduction's vapour_path="air_spaces" (a merge's labels, metafspm's Reduction(merge=))."""
    out = np.full(template.size, -1, dtype=np.int64)
    rows, values = template.properties.get("network_path", (np.zeros(0, dtype=np.int64), np.zeros(0)))
    edges = np.asarray(rows, dtype=np.int64)[np.asarray(values, dtype=float) == WALL_AIR]
    if not edges.size:
        return out
    kind = np.full(template.size, np.nan)
    kind_rows, kind_values = template.properties["kind"]
    kind[np.asarray(kind_rows, dtype=np.int64)] = np.asarray(kind_values, dtype=float)
    ends = []
    for link in ("n_id_a", "n_id_b"):
        where, linked = template.properties[link]
        ends.append(np.asarray(linked, dtype=np.int64)[np.searchsorted(where, edges)])
    a_is_wall = kind[ends[0]] == WALL
    walls, air = np.where(a_is_wall, ends[0], ends[1]), np.where(a_is_wall, ends[1], ends[0])
    out[walls] = air
    return out


def shoot_conduits(template) -> np.ndarray:
    """The conduits (GRANAP's entity set conduits) of a SectionTemplate of the shoot: none in a root's (a reduced
    root section has a soil node: MECHA's node soil boundaries, which a reduced root needs), so that
    plant_reduction's leaf_ports="bundles" leaves the roots' conduits each its own node."""
    from openalea.metafspm.coupling.declaration import template_selection
    if template_selection("GranapAnatomy.soil_nodes", template).any():
        return np.zeros(template.size, dtype=bool)
    return template_selection("GranapAnatomy.conduits", template)


_VAPOUR_PATH = vapour_path                    # plant_reduction's keep rule (its argument vapour_path is an option)

# plant_reduction's options as metafspm's Reduction(merge=) dicts: a shoot section's conduits lumped per vascular
# bundle and family (GRANAP's pole_id and family: its xylem vessels one node, its sieve tubes another), the merged
# conduit's area its members' root sum of squares (a Poiseuille conductance goes as the area squared: the junctions'
# conductances summed); a leaf's evaporating walls lumped per air space
LEAF_PORTS = {"conduits": None,
              "bundles": {"where": shoot_conduits, "by": ("pole_id", "family"),
                          "properties": {"cell_area": "root_sum_square"}}}
VAPOUR_PATHS = {"cells": None, "air_spaces": {"where": _VAPOUR_PATH, "by": air_space_of_wall}}


def plant_reduction(where=None, length_classes=0.1, osmotic: bool = False, leaf_ports: str = "conduits",
                    vapour_path: str = "cells"):
    """
    The Reduction (solve.reduction) of a plant model with GRANAP's anatomies and MechaHydraulics' plant problem: the
    sections *where* selects (a filter of the SubOrgans, metafspm's Reduction(where=): {"label": ...} their labels,
    "Organ.label" their organ's, e.g. {"label": "StemElement", "Organ.label": "StemInternode"} the internodes' and not
    the sheaths'; the root segments by default; GRANAP warns of a label it gives no anatomy) reduced onto their
    conduits (GRANAP's entity set conduits), their soil node (soil_nodes) and, in a leaf, its vapour path
    (vapour_path, its Connections kept explicit: vapour_edges); a kind per (anatomy, barrier, length class,
    parameters): the classes of relative width *length_classes* (0.1 by default: about 60 from 1 mm to 30 cm, each
    evaluated once), a section between two classes their kinds' blend (metafspm's Relative, interpolated), or
    "exact" (a kind per length: the exactness's reference); with *osmotic*, the conduits' and the hidden tissues'
    osmotic potentials as per-section inputs (conduit_osmotic_potential, tissue_osmotic_potential); the sections' kr,
    standard uptake and perimeter from their kind.

    Two approximations shrink the shoot's dense blocks (each kept node of a section a row of its dense block) by
    lumping kept nodes (metafspm's Reduction(merge=): a merged group one node, its members held at one potential):
      leaf_ports   "conduits" (default): each conduit its own node; "bundles": a shoot section's conduits per
                   vascular bundle (LEAF_PORTS: its xylem vessels one node, its sieve tubes another; the junctions
                   between sections then join bundles, the merged conduit's area the root sum of squares of its
                   members', so that its Poiseuille conductance is theirs summed); the roots' unchanged
      vapour_path  "cells" (default): each evaporating wall its own node; "air_spaces": a leaf's evaporating walls per
                   air space they face (VAPOUR_PATHS: one node, their wall-air exchanges summed on it)
    """
    import openalea.granap.metafspm.component                   # noqa: F401  (GranapAnatomy's entity sets)
    from openalea.metafspm.solve.reduction import Reduction, Relative
    if leaf_ports not in LEAF_PORTS:
        raise ValueError(f"plant_reduction: leaf_ports is one of {sorted(LEAF_PORTS)}, got {leaf_ports!r}")
    if vapour_path not in VAPOUR_PATHS:
        raise ValueError(f"plant_reduction: vapour_path is one of {sorted(VAPOUR_PATHS)}, got {vapour_path!r}")
    kind = {"barrier": None, "length": None if length_classes == "exact" else Relative(float(length_classes))}
    inputs = {"osmotic_potential": dict(by="tissue", group="tissue_osmotic_potential",
                                        kept="conduit_osmotic_potential")} if osmotic else {}
    merge = tuple(spec for spec in (LEAF_PORTS[leaf_ports], VAPOUR_PATHS[vapour_path]) if spec is not None)
    return Reduction(location="SubOrgan", where={"label": "RootSegment"} if where is None else where,
                     ports="GranapAnatomy.conduits", keep=("GranapAnatomy.soil_nodes", _VAPOUR_PATH),
                     explicit_edges=vapour_edges, kind=kind, inputs=inputs, systems=("MechaHydraulics._plant_problem",),
                     outputs=("kr", "standard_uptake", "perimeter"), merge=merge or None)


def scenario_reduction(scenario):
    """
    A plant model's reduction chosen by its scenario (``reduction = staticmethod(scenario_reduction)``, one model
    class reduced or not): with "reduced" true, plant_reduction of the sections "reduced_where" selects (a filter of
    the SubOrgans, e.g. {"label": ("RootSegment", "RootApex", "StemElement", "LeafElement", "Seed")}: every section;
    the root segments by default), their kinds by "length_classes" (0.1 by default, plant_reduction's), the osmotic
    inputs with "osmotic_reduction" (false by default), the shoot's conduits by "leaf_ports" and a leaf's vapour path
    by "vapour_path" (plant_reduction's: "conduits" and "cells" by default); None without it (the whole network).
    """
    if not scenario.get("reduced", False):
        return None
    return plant_reduction(where=scenario.get("reduced_where"), length_classes=scenario.get("length_classes", 0.1),
                           osmotic=bool(scenario.get("osmotic_reduction", False)),
                           leaf_ports=scenario.get("leaf_ports", "conduits"),
                           vapour_path=scenario.get("vapour_path", "cells"))


# the scenario entries scenario_reduction owns, read or not (metafspm accepts them with "reduced" off)
scenario_reduction.scenario_keys = ("reduced", "reduced_where", "length_classes", "osmotic_reduction", "leaf_ports",
                                    "vapour_path")
