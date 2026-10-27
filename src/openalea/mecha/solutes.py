"""
MECHA's solutes (doc/equations.rst): sucrose, mobile, and starch, immobile, on the liquid network of every
section (protoplasts, walls, junctions; not the air nodes), from the reactions branch's transport
(``SoluteTransport``, ``MultiSoluteTransport``, its needle simulations), in implicit Euler.

    presets = MechaOsmoticPresets(data_structure=ds)   # the osmotic baseline, + the solutes' −R T c on the cells
    mecha = MechaHydraulics(data_structure=ds)         # the water flows, the diffusion factors, the volumes
    solutes = MechaSolutes(data_structure=ds)          # one transport step per time step, on the water's flows

Per node and time step (dt, days): C V (c − c_prev) / dt = T c + K V c − k V c + s, with C the storage fraction
(storage_fraction_wall, _cell; the reactions' C_wall, C_cell), k the first-order loss (k_decay; their k_deg), and
  T c        the Scharfetter–Gummel fluxes along the liquid edges: per edge, its diffusion conductance D_e (the
             apoplastic, symplastic and membrane factors of MechaHydraulics × their diffusivities) and its advective
             flow F (the water's flow, × (1 − σ) on the membranes: Kedem–Katchalsky), P = F / D_e, the flow from tail
             to head D_e (B(−P) c_tail − B(P) c_head), B(x) = x / (eˣ − 1); upwind where D_e = 0
  K          the sucrose ⇄ starch reaction (d⁻¹), in the cells only (Q-U4)
  s          the sources (sucrose_source, mol d⁻¹: photosynthesis), and the sinks (Dirichlet at c_sink): the
             phloem-loading cells (Strasburger cells, else phloem) and the xylem
  and the liquid boundary (develop's conserve_boundary; solute_boundary_advection 1): the water a node exchanges
             with the outside of its section's liquid network (the soil, the axial boundaries and junctions), its
             flows' residual on the liquid and vapour edges, carries out the node's concentration where it leaves and
             brings c_boundary where it enters. Water leaving as vapour (the air spaces' walls, the pores) is an edge
             of that balance, not the boundary: it leaves its solutes behind.
The reactions' problems not ported (doc/legacy.rst): r1 (the osmotic flow counted twice), r2 (σ on the wrong
membrane end), r3 (two σ: here the membranes' one), r4 (starch in the walls), r12 (solutes along the vapour edges).
"""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import bmat, diags

from openalea.metafspm.coupling.component import input_variable, state_variable
from openalea.metafspm.solve.decorator import (boundary_set, graph_jacobian, graph_output, graph_system, node_balance,
                                                rate, totalstate)

from openalea.mecha.component import SCALES, MechaSections, _doc, _p, _wall_air_flow, _wall_air_jacobian
from openalea.mecha.geometry import AIR_LINK, MEMBRANE, PLASMODESMATA, STOMATAL, WALL_AIR, WALL_PATH, XYLEM_CGROUPS

GAS_CONSTANT = 8.314E4          # hPa cm³ mol⁻¹ K⁻¹ (van't Hoff with c in mol cm⁻³: π in hPa)
SECONDS_PER_DAY = 86400.
# the coupled steady state's balances, each over its node's total conductance (constant during the solve): the water's
# in hPa, the sucrose's in µmol cm⁻³ (× 1e6), so that one tolerance bounds the potentials and the concentrations
SUCROSE_RESIDUAL_SCALE = 1e6


def _row_scales(B, conductance, air_tail_conductance, air_head_conductance, soil_conductance,
                axial_boundary_conductance, diffusion_conductance, decay_rate):
    """(water, sucrose): each node's total water conductance (its edges', its boundaries'), and its total diffusion
    conductance and loss rate (cm³ d⁻¹; 1 where none), that the coupled balances are divided by."""
    magnitude = abs(B)
    water = (np.asarray(magnitude @ (conductance + air_tail_conductance + air_head_conductance)).reshape(-1)
             + soil_conductance + axial_boundary_conductance)
    sucrose = np.asarray(magnitude @ diffusion_conductance).reshape(-1) + decay_rate
    return np.where(water > 0, water, 1.), np.where(sucrose > 0, sucrose, 1.) / SUCROSE_RESIDUAL_SCALE


def bernoulli(x):
    """B(x) = x / (eˣ − 1), elementwise, as the reactions' ``_bernoulli``: Taylor below |x| 1e-10, its limits
    beyond |x| 700."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small, low, high = np.abs(x) < 1e-10, x < -700.0, x > 700.0
    middle = ~(small | low | high)
    out[small] = 1.0 - x[small] / 2.0 + x[small] * x[small] / 12.0
    out[low] = -x[low]
    out[high] = 0.0
    out[middle] = x[middle] / np.expm1(x[middle])
    return out


def bernoulli_derivative(x):
    """B'(x) = B(x) (1 − B(x)) / x − B(x), elementwise (from B' = B / x − B eˣ / (eˣ − 1)): −1/2 + x/6 below |x|
    1e-3, its limits beyond |x| 700 (0, −1)."""
    x = np.asarray(x, dtype=float)
    small, low, high = np.abs(x) < 1e-3, x < -700.0, x > 700.0
    middle = np.where(small | low | high, 1.0, x)
    Bx = bernoulli(middle)
    return np.select([small, low, high], [-0.5 + x / 6.0, -1.0, 0.0], Bx * (1.0 - Bx) / middle - Bx)


def weight_derivatives(De, F, upwind=False):
    """The derivatives (∂a/∂F, ∂b/∂F) of edge_weights' (a, b) in the edge's advective flow F: −B'(−P), B'(P) with
    Scharfetter–Gummel; the indicators of F > 0 and −(F < 0) upwind."""
    De, F = np.asarray(De, dtype=float), np.asarray(F, dtype=float)
    diffusive = (De > 0.0) & ~np.asarray(upwind, dtype=bool)
    P = np.divide(F, De, out=np.zeros_like(F), where=diffusive)
    return (np.where(diffusive, -bernoulli_derivative(-P), (F > 0).astype(float)),
            np.where(diffusive, bernoulli_derivative(P), -(F < 0).astype(float)))


def coupled_flow(B, psi, sucrose, conductance, reflection_coefficient, osmotic_baseline, osmotic_slope,
                 air_tail_conductance, air_head_conductance, air_offset):
    """The water's flow along each edge at the potentials *psi* and the concentrations *sucrose*: K (Δψ + σ Δπ),
    π = the baseline − R T c on the cells (osmotic_slope), plus the wall–air edges' linearised Kelvin flows."""
    pi = osmotic_baseline - osmotic_slope * sucrose
    drop = np.asarray(B.T @ psi).reshape(-1) + reflection_coefficient * np.asarray(B.T @ pi).reshape(-1)
    return conductance * drop + _wall_air_flow(B, psi, air_tail_conductance, air_head_conductance, air_offset)


def liquid_boundary_exchange(incidence, flow, counted, liquid):
    """
    (outflow, inflow) per node (cm³ d⁻¹, ≥ 0): the water each liquid node gives the boundary or takes from it, its
    net outflow along the *counted* edges (the liquid network's and the vapour edges) with its sign changed (develop's
    SoluteTransport._boundary_flux). Water leaving along a vapour edge is counted, so it is never the boundary's: the
    phase change leaves the solutes behind.
    """
    along_edges = np.asarray(incidence @ np.where(counted, np.asarray(flow, dtype=float), 0.)).ravel()
    return (np.where(liquid, np.maximum(-along_edges, 0.), 0.), np.where(liquid, np.maximum(along_edges, 0.), 0.))


def edge_weights(De, F, upwind=False):
    """
    The Scharfetter–Gummel weights (a, b) of each edge: its flow from tail to head is a c_tail − b c_head, with
    a = D_e B(−P), b = D_e B(P), P = F / D_e (B(−P) as the reactions' B(P) + P); upwind (a = max(F, 0), b = −min(F, 0))
    where D_e = 0. With *upwind* (a bool per edge), the upwind scheme: diffusion and donor-cell advection,
    a = D_e + max(F, 0), b = D_e − min(F, 0) (the reactions' ``scheme='upwind'``).
    """
    De, F = np.asarray(De, dtype=float), np.asarray(F, dtype=float)
    diffusive = (De > 0.0) & ~np.asarray(upwind, dtype=bool)
    P = np.divide(F, De, out=np.zeros_like(F), where=diffusive)
    Bp = bernoulli(P)
    Bm = Bp + P
    a = np.where(diffusive, De * Bm, De + np.maximum(F, 0.))
    b = np.where(diffusive, De * Bp, De - np.minimum(F, 0.))
    return a, b


@dataclass
class MechaSolutes(MechaSections):
    """
    Sucrose and starch in the liquid network of the sections. Inputs: MechaHydraulics' water flows,
    diffusion factors and volumes, the membranes' reflection coefficients (the "Osmotic" role: MechaOsmoticPresets),
    the photosynthetic sources (role "Photosynthesis"). Outputs: the concentrations, and the sucrose's osmotic
    potential −R T c on the cells (the presets add it to their baseline).
    """
    # ── Inputs ──
    flow: float = input_variable(**_doc("cm3 d-1", "the water's flow along the edge (MechaHydraulics' plant problem)"),
                                 by="Hydraulics", initialize=0., scale=SCALES.Connection, location="edge",
                                 state_variable_type="NonInertialExtensive")
    standard_flow: float = input_variable(**_doc("cm3 d-1", "the water's flow along the edge in MechaHydraulics' "
                                                           "standard problem (solute_flows 2)"),
                                          by="Hydraulics", initialize=0., scale=SCALES.Connection, location="edge",
                                          state_variable_type="NonInertialExtensive")
    apoplastic_diffusion_factor: float = input_variable(**_doc("cm", "MechaHydraulics'"), by="Hydraulics",
                                                        initialize=0., scale=SCALES.Connection, location="edge",
                                                        state_variable_type="NonInertialIntensive")
    symplastic_diffusion_factor: float = input_variable(**_doc("cm", "MechaHydraulics'"), by="Hydraulics",
                                                        initialize=0., scale=SCALES.Connection, location="edge",
                                                        state_variable_type="NonInertialIntensive")
    membrane_diffusion_factor: float = input_variable(**_doc("cm", "MechaHydraulics'"), by="Hydraulics",
                                                      initialize=0., scale=SCALES.Connection, location="edge",
                                                      state_variable_type="NonInertialIntensive")
    node_volume: float = input_variable(**_doc("cm3", "MechaHydraulics'"), by="Hydraulics", initialize=0.,
                                        scale=SCALES.Compartment, location="node",
                                        state_variable_type="NonInertialExtensive")
    vapour_node: float = input_variable(**_doc("", "1 on the air nodes (MechaHydraulics')"), by="Hydraulics",
                                        initialize=0., scale=SCALES.Compartment, location="node",
                                        state_variable_type="descriptor")
    reflection_coefficient: float = input_variable(**_doc("", "the membrane's reflection coefficient"), by="Osmotic",
                                                   initialize=0., scale=SCALES.Connection, location="edge",
                                                   state_variable_type="NonInertialIntensive")
    osmotic_potential: float = input_variable(**_doc("hPa", "the nodes' osmotic potential (MechaOsmoticPresets': its "
                                                            "baseline + the sucrose's −R T c on the cells)"),
                                              by="Osmotic", initialize=0., scale=SCALES.Compartment, location="node",
                                              state_variable_type="NonInertialIntensive")
    conductance: float = input_variable(**_doc("cm3 hPa-1 d-1", "MechaHydraulics' (solute_solve 2)"), by="Hydraulics",
                                        initialize=0., scale=SCALES.Connection, location="edge",
                                        state_variable_type="NonInertialIntensive")
    soil_conductance: float = input_variable(**_doc("cm3 hPa-1 d-1", "MechaHydraulics'"), by="Hydraulics",
                                             initialize=0., scale=SCALES.Compartment, location="node",
                                             state_variable_type="NonInertialIntensive")
    standard_soil_psi: float = input_variable(**_doc("hPa", "MechaHydraulics'"), by="Hydraulics", initialize=0.,
                                              scale=SCALES.Compartment, location="node",
                                              state_variable_type="NonInertialIntensive")
    axial_boundary_conductance: float = input_variable(**_doc("cm3 hPa-1 d-1", "MechaHydraulics'"), by="Hydraulics",
                                                       initialize=0., scale=SCALES.Compartment, location="node",
                                                       state_variable_type="NonInertialIntensive")
    standard_axial_psi: float = input_variable(**_doc("hPa", "MechaHydraulics'"), by="Hydraulics", initialize=0.,
                                               scale=SCALES.Compartment, location="node",
                                               state_variable_type="NonInertialIntensive")
    standard_axial_inflow: float = input_variable(**_doc("cm3 d-1", "MechaHydraulics'"), by="Hydraulics",
                                                  initialize=0., scale=SCALES.Compartment, location="node",
                                                  state_variable_type="NonInertialIntensive")
    standard_elongation_sink: float = input_variable(**_doc("cm3 d-1", "MechaHydraulics'"), by="Hydraulics",
                                                     initialize=0., scale=SCALES.Compartment, location="node",
                                                     state_variable_type="NonInertialExtensive")
    standard_psi: float = input_variable(**_doc("hPa", "MechaHydraulics' standard problem's (the coupled steady "
                                                     "state's first guess)"),
                                         by="Hydraulics", initialize=0., scale=SCALES.Compartment, location="node",
                                         state_variable_type="NonInertialIntensive")
    in_standard_problem: float = input_variable(**_doc("", "MechaHydraulics'"), by="Hydraulics", initialize=0.,
                                                scale=SCALES.Compartment, location="node",
                                                state_variable_type="descriptor")
    air_tail_conductance: float = input_variable(**_doc("cm3 hPa-1 d-1", "MechaHydraulics'"), by="Hydraulics",
                                                 initialize=0., scale=SCALES.Connection, location="edge",
                                                 state_variable_type="NonInertialIntensive")
    air_head_conductance: float = input_variable(**_doc("cm3 hPa-1 d-1", "MechaHydraulics'"), by="Hydraulics",
                                                 initialize=0., scale=SCALES.Connection, location="edge",
                                                 state_variable_type="NonInertialIntensive")
    air_offset: float = input_variable(**_doc("cm3 d-1", "MechaHydraulics'"), by="Hydraulics", initialize=0.,
                                       scale=SCALES.Connection, location="edge",
                                       state_variable_type="NonInertialIntensive")
    sucrose_source: float = input_variable(**_doc("mol d-1", "sucrose produced in the node (photosynthesis)"),
                                           by="Photosynthesis", initialize=0., scale=SCALES.Compartment,
                                           location="node", state_variable_type="NonInertialExtensive")

    # ── Parameters (per plant) ──
    d_apoplast: float = _p("cm2 d-1", "sucrose's diffusivity in the walls", 0.1)
    d_plasmodesmata: float = _p("cm2 d-1", "sucrose's diffusivity through the plasmodesmata", 0.5)
    d_membrane: float = _p("cm2 d-1", "sucrose's passive diffusivity across the membranes", 1.0E-6)
    k_starch_synthesis: float = _p("d-1", "the cells' sucrose → starch rate", 0.)
    k_starch_degradation: float = _p("d-1", "the cells' starch → sucrose rate", 0.)
    c_sink: float = _p("mol cm-3", "the sucrose concentration of the sinks (phloem loading, xylem)", 0.)
    storage_fraction_wall: float = _p("adim", "the fraction of a wall's (or junction's) volume storing sucrose (the "
                                              "reactions' C_wall)", 1.)
    storage_fraction_cell: float = _p("adim", "the fraction of a protoplast's volume storing sucrose (the reactions' "
                                              "C_cell)", 1.)
    k_decay: float = _p("d-1", "sucrose's first-order loss on every liquid node (degradation, consumption: the "
                               "reactions' and develop's k_deg)", 0.)
    solute_boundary_advection: float = _p("", "the liquid water leaving a section's liquid network (soil, axial "
                                              "boundaries) carries its solutes out, entering water brings c_boundary: "
                                              "1 (develop's conserve_boundary), 0 none (the reactions branch's)", 1.)
    c_boundary: float = _p("mol cm-3", "the sucrose concentration of the liquid water entering from the boundary", 0.)
    solute_temperature: float = _p("K", "the temperature of van't Hoff's law", 298.15)
    solute_pathway: float = _p("", "the network the solutes move in: 1 the whole liquid network, 2 the apoplast only "
                                   "(walls and junctions, along the walls), 3 the symplast only (protoplasts, through "
                                   "the plasmodesmata): the reactions' modes 'full', 'apo', 'sym'", 1.)
    solute_scheme: float = _p("", "the transport's discretisation: 1 Scharfetter–Gummel, 2 upwind (diffusion and "
                                  "donor-cell advection, the reactions' default)", 1.)
    solute_theta: float = _p("adim", "the time step's implicitness θ: 1 implicit Euler, 0.5 Crank–Nicolson, 0 explicit "
                                     "Euler (the transport, reactions, loss and boundary outflow weighed θ at the "
                                     "step's end, 1 − θ at its start)", 1.)
    solute_solve: float = _p("", "1 one transport step per time step, on the water flows given (flow or "
                                 "standard_flow); 2 the steady state of the sucrose coupled with the standard problem's "
                                 "water (its osmotic potentials −R T c), each section alone, solved by Newton (the "
                                 "reactions' coupled_water_solute_solve; no starch)", 1.)
    solute_flows: float = _p("", "the water flows carrying the solutes: 1 the plant problem's (flow), 2 the standard "
                                 "problem's (standard_flow: the legacy's standard_solute_flux, each section alone)", 1.)

    # ── State variables ──
    sucrose: float = state_variable(**_doc("mol cm-3", "sucrose concentration (liquid nodes)"), initialize=0.,
                                    scale=SCALES.Compartment, location="node", state_variable_type="intensive")
    starch: float = state_variable(**_doc("mol cm-3", "starch concentration (sucrose equivalents; cells)"),
                                   initialize=0., scale=SCALES.Compartment, location="node",
                                   state_variable_type="intensive")
    sucrose_sink: float = state_variable(**_doc("", "1 on the sinks: the phloem-loading cells (Strasburger, else "
                                                    "phloem) and the xylem"),
                                         initialize=0., scale=SCALES.Compartment, location="node",
                                         state_variable_type="descriptor")
    sucrose_sink_value: float = state_variable(**_doc("mol cm-3", "the sinks' concentration (c_sink)"),
                                               initialize=0., scale=SCALES.Compartment, location="node",
                                               state_variable_type="NonInertialIntensive")
    liquid_node: float = state_variable(**_doc("", "1 on the nodes the solutes are solved on (the liquid nodes of "
                                                   "their pathway)"), initialize=0.,
                                        scale=SCALES.Compartment, location="node", state_variable_type="descriptor")
    coupled_node: float = state_variable(**_doc("", "1 on the nodes of the sections solving the coupled steady state "
                                                    "(solute_solve 2)"), initialize=0., scale=SCALES.Compartment,
                                         location="node", state_variable_type="descriptor")
    solute_node: float = state_variable(**_doc("", "1 on the liquid nodes of the solutes' pathway (the coupled steady "
                                                   "state's sucrose nodes)"), initialize=0.,
                                        scale=SCALES.Compartment, location="node", state_variable_type="descriptor")
    coupled_psi: float = state_variable(**_doc("hPa", "the water potential of the coupled steady state (the standard "
                                                      "problem's, with the sucrose's osmotic potentials)"),
                                        initialize=0., scale=SCALES.Compartment, location="node",
                                        state_variable_type="NonInertialIntensive")
    coupled_flow: float = state_variable(**_doc("cm3 d-1", "the water's flow along the edge in the coupled steady "
                                                           "state"),
                                         initialize=0., scale=SCALES.Connection, location="edge",
                                         state_variable_type="NonInertialIntensive")
    osmotic_baseline: float = state_variable(**_doc("hPa", "the osmotic potential without the sucrose's (its −R T c "
                                                           "removed from the cells')"),
                                             initialize=0., scale=SCALES.Compartment, location="node",
                                             state_variable_type="NonInertialIntensive")
    osmotic_slope: float = state_variable(**_doc("hPa cm3 mol-1", "R T on the liquid cells (π = baseline − R T c), 0 "
                                                                  "elsewhere"),
                                          initialize=0., scale=SCALES.Compartment, location="node",
                                          state_variable_type="descriptor")
    diffusion_conductance: float = state_variable(**_doc("cm3 d-1", "the edge's diffusion conductance D_e on the "
                                                                    "solutes' edges (0 elsewhere)"),
                                                  initialize=0., scale=SCALES.Connection, location="edge",
                                                  state_variable_type="NonInertialIntensive")
    carried_fraction: float = state_variable(**_doc("adim", "the fraction of the water's flow advecting the solute: "
                                                            "1 − σ on the membranes, 1 on the other solutes' edges, 0 "
                                                            "elsewhere"),
                                             initialize=0., scale=SCALES.Connection, location="edge",
                                             state_variable_type="descriptor")
    upwind_edge: float = state_variable(**_doc("", "1 on the edges of the upwind scheme (solute_scheme 2)"),
                                        initialize=0., scale=SCALES.Connection, location="edge",
                                        state_variable_type="descriptor")
    water_edge: float = state_variable(**_doc("", "1 on the edges counted in the liquid boundary's balance (the "
                                                  "water's liquid and vapour edges)"),
                                       initialize=0., scale=SCALES.Connection, location="edge",
                                       state_variable_type="descriptor")
    boundary_concentration: float = state_variable(**_doc("mol cm-3", "c_boundary on the nodes whose boundary water "
                                                                      "carries solutes (NaN: none, "
                                                                      "solute_boundary_advection 0)"),
                                                   initialize=np.nan, scale=SCALES.Compartment, location="node",
                                                   state_variable_type="NonInertialIntensive")
    implicit_weight: float = state_variable(**_doc("adim", "the time step's θ (solute_theta)"), initialize=1.,
                                            scale=SCALES.Compartment, location="node",
                                            state_variable_type="descriptor")
    tail_weight: float = state_variable(**_doc("cm3 d-1", "Scharfetter–Gummel: the solute's flow from tail to head "
                                                         "is tail_weight c_tail − head_weight c_head"),
                                        initialize=0., scale=SCALES.Connection, location="edge",
                                        state_variable_type="NonInertialIntensive")
    head_weight: float = state_variable(**_doc("cm3 d-1", "see tail_weight"), initialize=0., scale=SCALES.Connection,
                                        location="edge", state_variable_type="NonInertialIntensive")
    storage_volume: float = state_variable(**_doc("cm3", "the node's volume storing sucrose: its volume × its storage "
                                                         "fraction"),
                                           initialize=0., scale=SCALES.Compartment, location="node",
                                           state_variable_type="NonInertialExtensive")
    decay_rate: float = state_variable(**_doc("cm3 d-1", "k_decay × V on the liquid nodes"), initialize=0.,
                                       scale=SCALES.Compartment, location="node",
                                       state_variable_type="NonInertialExtensive")
    boundary_outflow: float = state_variable(**_doc("cm3 d-1", "the liquid water the node gives the boundary (its "
                                                              "solute leaves with it; solute_boundary_advection 1)"),
                                             initialize=0., scale=SCALES.Compartment, location="node",
                                             state_variable_type="NonInertialExtensive")
    boundary_source: float = state_variable(**_doc("mol d-1", "the solute the water entering from the boundary brings "
                                                              "(× c_boundary)"),
                                            initialize=0., scale=SCALES.Compartment, location="node",
                                            state_variable_type="NonInertialExtensive")
    synthesis_rate: float = state_variable(**_doc("cm3 d-1", "k_starch_synthesis × V on the cells (0 elsewhere)"),
                                           initialize=0., scale=SCALES.Compartment, location="node",
                                           state_variable_type="NonInertialExtensive")
    degradation_rate: float = state_variable(**_doc("cm3 d-1", "k_starch_degradation × V on the cells"),
                                             initialize=0., scale=SCALES.Compartment, location="node",
                                             state_variable_type="NonInertialExtensive")
    sucrose_osmotic_potential: float = state_variable(**_doc("hPa", "−R T c on the cells (van't Hoff; 0 elsewhere)"),
                                                      initialize=0., scale=SCALES.Compartment, location="node",
                                                      state_variable_type="NonInertialIntensive")

    @rate(include_inactive=True)
    def _tail_weight(self, flow, standard_flow, reflection_coefficient, apoplastic_diffusion_factor,
                           symplastic_diffusion_factor, membrane_diffusion_factor, node_volume, vapour_node, kind,
                           cgroup, tissue, network_path, osmotic_potential, sucrose_osmotic_potential,
                           in_standard_problem, standard_psi, coupled_psi) \
            -> tuple[np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray,
                     str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str,
                     np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray,
                     str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray]:
        """Each liquid edge's Scharfetter–Gummel weights; the sinks; the cells' reaction rates; the storage and the
        first-order loss; the liquid boundary's exchanges."""
        ds = self.data_structure
        view = ds.to_graph_view()
        P_edge = {name: np.asarray(self.parameter_values(name, "edge"), dtype=float)
                  for name in ("d_apoplast", "d_plasmodesmata", "d_membrane")}
        P_node = {name: np.asarray(self.parameter_values(name, "node"), dtype=float)
                  for name in ("k_starch_synthesis", "k_starch_degradation", "c_sink", "solute_boundary_advection",
                               "c_boundary", "storage_fraction_wall", "storage_fraction_cell", "k_decay",
                               "solute_pathway", "solute_theta", "solute_solve", "solute_temperature")}
        edge_pathway = np.asarray(self.parameter_values("solute_pathway", "edge"))
        upwind = np.asarray(self.parameter_values("solute_scheme", "edge")) == 2
        kind = np.asarray(kind)
        water = np.asarray(vapour_node) == 0                               # the water's liquid nodes
        pathway = P_node["solute_pathway"]
        liquid = water & ((pathway == 1) | ((pathway == 2) & (kind != 0)) | ((pathway == 3) & (kind == 0)))
        path = np.asarray(network_path)
        standard = np.asarray(self.parameter_values("solute_flows", "edge")) == 2
        flow = np.where(standard, np.asarray(standard_flow, dtype=float), np.asarray(flow, dtype=float))
        on_water = water[view.tail] & water[view.head] & np.isin(path, (WALL_PATH, MEMBRANE, PLASMODESMATA))
        along = np.select([edge_pathway == 2, edge_pathway == 3], [path == WALL_PATH, path == PLASMODESMATA],
                          np.isin(path, (WALL_PATH, MEMBRANE, PLASMODESMATA)))
        on_liquid = liquid[view.tail] & liquid[view.head] & along
        De = (np.asarray(apoplastic_diffusion_factor) * P_edge["d_apoplast"]
              + np.asarray(symplastic_diffusion_factor) * P_edge["d_plasmodesmata"]
              + np.asarray(membrane_diffusion_factor) * P_edge["d_membrane"])
        carried = np.where(path == MEMBRANE, 1.0 - np.asarray(reflection_coefficient, dtype=float), 1.0)
        a, b = edge_weights(np.where(on_liquid, De, 0.), np.where(on_liquid, np.asarray(flow) * carried, 0.), upwind)
        sink = self._sinks(kind, np.asarray(cgroup), np.asarray(tissue, dtype=object)) & liquid
        coupled = (P_node["solute_solve"] == 2) & (np.asarray(in_standard_problem) > 0)
        transient = liquid & (P_node["solute_solve"] == 1)
        cell = (kind == 0) & liquid
        volume = np.asarray(node_volume, dtype=float)
        # the liquid boundary: the water's network, whatever the solutes' pathway (the water crossing from it to the
        # other pathway leaves its solutes behind, as the vapour leaving does)
        counted = on_water | np.isin(path, (WALL_AIR, AIR_LINK, STOMATAL))
        advected = liquid & (P_node["solute_boundary_advection"] > 0)
        boundary_outflow, boundary_inflow = liquid_boundary_exchange(view.incidence, flow, counted, advected)
        return (a, "head_weight", b, "sucrose_sink", sink.astype(float),
                "sucrose_sink_value", P_node["c_sink"],
                "liquid_node", transient.astype(float), "implicit_weight", P_node["solute_theta"],
                # Newton's start: the standard problem's water, balanced (on the flows of an unbalanced guess the
                # steady sucrose has no solution: water converging on a node without an exit)
                "coupled_psi", np.where(coupled, np.asarray(standard_psi, dtype=float), np.asarray(coupled_psi)),
                "coupled_node", coupled.astype(float), "solute_node", liquid.astype(float),
                "osmotic_baseline", np.asarray(osmotic_potential, dtype=float)
                - np.asarray(sucrose_osmotic_potential, dtype=float),
                "osmotic_slope", np.where(cell, GAS_CONSTANT * P_node["solute_temperature"], 0.),
                "diffusion_conductance", np.where(on_liquid, De, 0.),
                "carried_fraction", np.where(on_liquid, carried, 0.), "upwind_edge", upwind.astype(float),
                "water_edge", counted.astype(float),
                "boundary_concentration", np.where(advected, P_node["c_boundary"], np.nan),
                "synthesis_rate", np.where(cell, P_node["k_starch_synthesis"] * volume, 0.),
                "degradation_rate", np.where(cell, P_node["k_starch_degradation"] * volume, 0.),
                "storage_volume", volume * np.where(np.asarray(kind) == 0, P_node["storage_fraction_cell"],
                                                    P_node["storage_fraction_wall"]),
                "decay_rate", np.where(liquid, P_node["k_decay"] * volume, 0.),
                "boundary_outflow", boundary_outflow, "boundary_source", boundary_inflow * P_node["c_boundary"])

    def _sinks(self, kind, cgroup, tissue):
        """The reactions' needle sinks: per section, its Strasburger cells (cgroup 12) if it has any, else its phloem
        (cgroup 11); and its xylem (cgroup 13)."""
        out = np.zeros(kind.size, dtype=bool)
        for nodes, _, _ in self.section_geometries().values():
            cells = nodes[kind[nodes] == 0]
            strasburger = cells[(cgroup[cells] == 12) & (tissue[cells] == "Strasburger cell")]
            loading = strasburger if strasburger.size else cells[cgroup[cells] == 11]
            out[loading] = True
            out[cells[np.isin(cgroup[cells], XYLEM_CGROUPS[:1])]] = True
        return out

    @totalstate(include_inactive=True)
    def _sucrose_osmotic_potential(self, sucrose, kind, vapour_node) -> np.ndarray:
        """−R T c on the liquid cells (van't Hoff, the reactions' set_osmotic_from_concentration), after the transport
        step: the next water solve's."""
        T = np.asarray(self.parameter_values("solute_temperature", "node"), dtype=float)
        cells = (np.asarray(kind) == 0) & (np.asarray(vapour_node) == 0)
        return np.where(cells, -GAS_CONSTANT * T * np.asarray(sucrose, dtype=float), 0.)

    @graph_system(node_unknowns=["sucrose", "starch"], solver="newton", transient=True, max_iter=3, tol=1e-14,
                  schedule_as="state", filters={"liquid_node": ">0"})
    class _transport:
        """The θ method on the liquid nodes (implicit Euler with θ = 1): C V (c − c_prev) / dt + θ L(c, s) + (1 − θ)
        L(c_prev, s_prev) − S − b = 0 for the sucrose, L(c, s) = B q(c) + K V c − K' V s + k V c + O c (C V its storage
        volume, O the boundary's outflow, b the solute its inflow brings), V (s − s_prev) / dt + θ (K' V s − K V c) +
        (1 − θ) (K' V s_prev − K V c_prev) = 0 for the starch."""

        @node_balance(field="sucrose")
        def _sucrose(self, sucrose, starch, tail_weight, head_weight, storage_volume, synthesis_rate, degradation_rate,
                     decay_rate, sucrose_source, boundary_outflow, boundary_source, implicit_weight):
            B = self._graph_view.incidence

            def operator(c, s):
                q = tail_weight * np.asarray(B.maximum(0).T @ c).reshape(-1) - \
                    head_weight * np.asarray((-B).maximum(0).T @ c).reshape(-1)
                return (np.asarray(B @ q).reshape(-1) + (synthesis_rate + decay_rate + boundary_outflow) * c
                        - degradation_rate * s)
            previous_sucrose, previous_starch = self.previous("sucrose"), self.previous("starch")
            return (storage_volume * (sucrose - previous_sucrose) / (self.dt / SECONDS_PER_DAY)
                    + implicit_weight * operator(sucrose, starch)
                    + (1. - implicit_weight) * operator(previous_sucrose, previous_starch)
                    - sucrose_source - boundary_source)

        @node_balance(field="starch")
        def _starch(self, sucrose, starch, node_volume, synthesis_rate, degradation_rate, implicit_weight):
            previous_sucrose, previous_starch = self.previous("sucrose"), self.previous("starch")
            return (node_volume * (starch - previous_starch) / (self.dt / SECONDS_PER_DAY)
                    + implicit_weight * (degradation_rate * starch - synthesis_rate * sucrose)
                    + (1. - implicit_weight) * (degradation_rate * previous_starch - synthesis_rate * previous_sucrose))

        sinks = boundary_set(filters={"sucrose_sink": ">0"}, kind="dirichlet", value="sucrose_sink_value",
                             field="sucrose")

        @graph_jacobian
        def _jacobian(self, tail_weight, head_weight, node_volume, storage_volume, synthesis_rate, degradation_rate,
                      decay_rate, boundary_outflow, implicit_weight):
            B = self._graph_view.incidence
            days = self.dt / SECONDS_PER_DAY
            theta = diags(implicit_weight)
            transport = B @ diags(tail_weight) @ B.maximum(0).T - B @ diags(head_weight) @ (-B).maximum(0).T
            return bmat([[diags(storage_volume / days)
                          + theta @ (transport + diags(synthesis_rate + decay_rate + boundary_outflow)),
                          theta @ diags(-degradation_rate)],
                         [theta @ diags(-synthesis_rate),
                          diags(node_volume / days) + theta @ diags(degradation_rate)]], format="csr")

    # ── The steady state of the sucrose coupled with the standard problem's water (solute_solve 2) ──
    # tol: the water's scaled residuals (hPa) and the sucrose's (µmol cm⁻³): _row_scales
    @graph_system(node_unknowns=["coupled_psi", "sucrose"], solver="newton", max_iter=30, tol=1e-10,
                  schedule_as="state", filters={"coupled_node": ">0"})
    class _steady_coupled:
        """
        Each section's standard problem (MechaHydraulics': the soil, the axial boundary, the elongation sink) with the
        osmotic drives of π = baseline − R T c on its cells, and the steady sucrose on the water's flows: B q + (k V
        + O) c − s − b = 0 (q the Scharfetter–Gummel or upwind fluxes, O, b the liquid boundary's), c = 0 on the
        other nodes; each balance over its node's total conductance (_row_scales), so that the tolerance bounds
        potentials and concentrations. One Newton system, its Jacobian analytic: the flows' derivatives in ψ and, through π, in c; the
        weights', the boundary exchange's in the flows.
        """

        @node_balance(field="coupled_psi")
        def _water(self, coupled_psi, sucrose, conductance, reflection_coefficient, osmotic_baseline, osmotic_slope,
                   air_tail_conductance, air_head_conductance, air_offset, standard_elongation_sink, soil_conductance,
                   standard_soil_psi, axial_boundary_conductance, standard_axial_psi, standard_axial_inflow,
                   diffusion_conductance, decay_rate):
            """The water's balance with its boundaries (the soil's and the axial boundary's Robin terms, the axial
            flow's Neumann term), over the node's total conductance (hPa)."""
            B = self._graph_view.incidence
            F = coupled_flow(B, coupled_psi, sucrose, conductance, reflection_coefficient, osmotic_baseline,
                             osmotic_slope, air_tail_conductance, air_head_conductance, air_offset)
            inflow = (soil_conductance * (standard_soil_psi - coupled_psi)
                      + axial_boundary_conductance * (standard_axial_psi - coupled_psi)
                      + standard_axial_inflow)
            water, _ = _row_scales(B, conductance, air_tail_conductance, air_head_conductance, soil_conductance,
                                   axial_boundary_conductance, diffusion_conductance, decay_rate)
            return (np.asarray(B @ F).reshape(-1) + standard_elongation_sink - inflow) / water

        @node_balance(field="sucrose")
        def _sucrose(self, coupled_psi, sucrose, conductance, reflection_coefficient, osmotic_baseline, osmotic_slope,
                     air_tail_conductance, air_head_conductance, air_offset, diffusion_conductance, carried_fraction,
                     upwind_edge, water_edge, boundary_concentration, decay_rate, sucrose_source, solute_node,
                     soil_conductance, axial_boundary_conductance):
            """The sucrose's balance over the node's total diffusion conductance (µmol cm⁻³)."""
            B = self._graph_view.incidence
            F = coupled_flow(B, coupled_psi, sucrose, conductance, reflection_coefficient, osmotic_baseline,
                             osmotic_slope, air_tail_conductance, air_head_conductance, air_offset)
            a, b = edge_weights(diffusion_conductance, carried_fraction * F, upwind_edge > 0)
            q = a * np.asarray(B.maximum(0).T @ sucrose).reshape(-1) - \
                b * np.asarray((-B).maximum(0).T @ sucrose).reshape(-1)
            advected = np.isfinite(boundary_concentration)
            outflow, inflow = liquid_boundary_exchange(B, F, water_edge > 0, advected)
            balance = (np.asarray(B @ q).reshape(-1) + (decay_rate + outflow) * sucrose - sucrose_source
                       - inflow * np.nan_to_num(boundary_concentration))
            _, scale = _row_scales(B, conductance, air_tail_conductance, air_head_conductance, soil_conductance,
                                   axial_boundary_conductance, diffusion_conductance, decay_rate)
            return np.where(solute_node > 0, balance / scale, SUCROSE_RESIDUAL_SCALE * sucrose)

        sinks = boundary_set(filters={"sucrose_sink": ">0"}, kind="dirichlet", value="sucrose_sink_value",
                             field="sucrose")

        @graph_jacobian
        def _jacobian(self, coupled_psi, sucrose, conductance, reflection_coefficient, osmotic_baseline,
                      osmotic_slope, air_tail_conductance, air_head_conductance, air_offset, diffusion_conductance,
                      carried_fraction, upwind_edge, water_edge, boundary_concentration, decay_rate, solute_node,
                      soil_conductance, axial_boundary_conductance):
            B = self._graph_view.incidence
            water, scale = _row_scales(B, conductance, air_tail_conductance, air_head_conductance, soil_conductance,
                                       axial_boundary_conductance, diffusion_conductance, decay_rate)
            scale = np.where(solute_node > 0, scale, 1. / SUCROSE_RESIDUAL_SCALE)
            tail, head = B.maximum(0), (-B).maximum(0)
            F = coupled_flow(B, coupled_psi, sucrose, conductance, reflection_coefficient, osmotic_baseline,
                             osmotic_slope, air_tail_conductance, air_head_conductance, air_offset)
            # the flows' derivatives (edges × nodes)
            flow_psi = (diags(conductance) @ B.T + diags(air_tail_conductance) @ tail.T
                        - diags(air_head_conductance) @ head.T)
            flow_sucrose = -diags(conductance * reflection_coefficient) @ B.T @ diags(osmotic_slope)
            # the solute's fluxes: direct, and through their weights' flows
            advective = carried_fraction * F
            a, b = edge_weights(diffusion_conductance, advective, upwind_edge > 0)
            da, db = weight_derivatives(diffusion_conductance, advective, upwind_edge > 0)
            c_tail = np.asarray(tail.T @ sucrose).reshape(-1)
            c_head = np.asarray(head.T @ sucrose).reshape(-1)
            through_flow = diags(carried_fraction * (da * c_tail - db * c_head))
            # the liquid boundary's outflow O c and inflow b, piecewise linear in the flows' residual r
            advected = np.isfinite(boundary_concentration)
            counted = np.where(water_edge > 0, F, 0.)
            residual = np.asarray(B @ counted).reshape(-1)
            outflow, _ = liquid_boundary_exchange(B, F, water_edge > 0, advected)
            boundary = diags(np.where(advected, -(residual < 0).astype(float) * sucrose
                                      - (residual > 0).astype(float) * np.nan_to_num(boundary_concentration),
                                      0.)) @ B @ \
                diags((water_edge > 0).astype(float))
            solute_rows = diags((solute_node > 0).astype(float))
            other_rows = diags((solute_node <= 0).astype(float))
            sucrose_psi = solute_rows @ (B @ through_flow @ flow_psi + boundary @ flow_psi)
            sucrose_sucrose = solute_rows @ (B @ (diags(a) @ tail.T - diags(b) @ head.T)
                                             + diags(decay_rate + outflow)
                                             + B @ through_flow @ flow_sucrose + boundary @ flow_sucrose) + other_rows
            water_psi = B @ flow_psi + diags(soil_conductance + axial_boundary_conductance)
            return bmat([[diags(1. / water) @ water_psi, diags(1. / water) @ B @ flow_sucrose],
                         [diags(1. / scale) @ sucrose_psi, diags(1. / scale) @ sucrose_sucrose]], format="csr")

        @graph_output("coupled_flow", location="edge")
        def _coupled_flow(self, coupled_psi, sucrose, conductance, reflection_coefficient, osmotic_baseline,
                          osmotic_slope, air_tail_conductance, air_head_conductance, air_offset):
            return coupled_flow(self._graph_view.incidence, coupled_psi, sucrose, conductance, reflection_coefficient,
                                osmotic_baseline, osmotic_slope, air_tail_conductance, air_head_conductance,
                                air_offset)
