"""
MECHA's solutes (doc/equations.rst): sucrose, mobile, and starch, immobile, on the liquid network of every
section (protoplasts, walls, junctions; not the air nodes), from the reactions branch's transport
(``SoluteTransport``, ``MultiSoluteTransport``, its needle simulations), in implicit Euler.

    presets = MechaOsmoticPresets(data_structure=ds)   # the osmotic baseline, + the solutes' −R T c on the cells
    mecha = MechaHydraulics(data_structure=ds)         # the water flows, the diffusion factors, the volumes
    solutes = MechaSolutes(data_structure=ds)          # one transport step per time step, on the water's flows

Per node and time step (dt, days): V (c − c_prev) / dt = T c + K V c + s, with
  T c        the Scharfetter–Gummel fluxes along the liquid edges: per edge, its diffusion conductance D_e (the
             apoplastic, symplastic and membrane factors of MechaHydraulics × their diffusivities) and its advective
             flow F (the water's flow, × (1 − σ) on the membranes: Kedem–Katchalsky), P = F / D_e, the flow from tail
             to head D_e (B(−P) c_tail − B(P) c_head), B(x) = x / (eˣ − 1); upwind where D_e = 0
  K          the sucrose ⇄ starch reaction (d⁻¹), in the cells only (Q-U4)
  s          the sources (sucrose_source, mol d⁻¹: photosynthesis), and the sinks (Dirichlet at c_sink): the
             phloem-loading cells (Strasburger cells, else phloem) and the xylem.
The reactions' problems not ported (doc/legacy.rst): r1 (the osmotic flow counted twice), r2 (σ on the wrong
membrane end), r3 (two σ: here the membranes' one), r4 (starch in the walls), r12 (solutes along the vapour edges).
"""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import bmat, diags

from openalea.metafspm.coupling.component import input_variable, state_variable
from openalea.metafspm.solve.decorator import boundary_set, graph_jacobian, graph_system, node_balance, rate, totalstate

from openalea.mecha.component import SCALES, MechaSections, _doc, _p
from openalea.mecha.geometry import MEMBRANE, PLASMODESMATA, WALL_PATH, XYLEM_CGROUPS

GAS_CONSTANT = 8.314E4          # hPa cm³ mol⁻¹ K⁻¹ (van't Hoff with c in mol cm⁻³: π in hPa)
SECONDS_PER_DAY = 86400.


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


def edge_weights(De, F):
    """
    The Scharfetter–Gummel weights (a, b) of each edge: its flow from tail to head is a c_tail − b c_head, with
    a = D_e B(−P), b = D_e B(P), P = F / D_e (B(−P) as the reactions' B(P) + P); upwind (a = max(F, 0), b = −min(F, 0))
    where D_e = 0.
    """
    De, F = np.asarray(De, dtype=float), np.asarray(F, dtype=float)
    diffusive = De > 0.0
    P = np.divide(F, De, out=np.zeros_like(F), where=diffusive)
    Bp = bernoulli(P)
    Bm = Bp + P
    a = np.where(diffusive, De * Bm, np.maximum(F, 0.))
    b = np.where(diffusive, De * Bp, -np.minimum(F, 0.))
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
    solute_temperature: float = _p("K", "the temperature of van't Hoff's law", 298.15)

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
    liquid_node: float = state_variable(**_doc("", "1 on the nodes the solutes are solved on"), initialize=0.,
                                        scale=SCALES.Compartment, location="node", state_variable_type="descriptor")
    tail_weight: float = state_variable(**_doc("cm3 d-1", "Scharfetter–Gummel: the solute's flow from tail to head "
                                                         "is tail_weight c_tail − head_weight c_head"),
                                        initialize=0., scale=SCALES.Connection, location="edge",
                                        state_variable_type="NonInertialIntensive")
    head_weight: float = state_variable(**_doc("cm3 d-1", "see tail_weight"), initialize=0., scale=SCALES.Connection,
                                        location="edge", state_variable_type="NonInertialIntensive")
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
    def _tail_weight(self, flow, reflection_coefficient, apoplastic_diffusion_factor,
                           symplastic_diffusion_factor, membrane_diffusion_factor, node_volume, vapour_node, kind,
                           cgroup, tissue, network_path) \
            -> tuple[np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray, str, np.ndarray,
                     str, np.ndarray]:
        """Each liquid edge's Scharfetter–Gummel weights; the sinks; the cells' reaction rates."""
        ds = self.data_structure
        view = ds.to_graph_view()
        P_edge = {name: np.asarray(self.parameter_values(name, "edge"), dtype=float)
                  for name in ("d_apoplast", "d_plasmodesmata", "d_membrane")}
        P_node = {name: np.asarray(self.parameter_values(name, "node"), dtype=float)
                  for name in ("k_starch_synthesis", "k_starch_degradation", "c_sink")}
        liquid = np.asarray(vapour_node) == 0
        path = np.asarray(network_path)
        on_liquid = liquid[view.tail] & liquid[view.head] & np.isin(path, (WALL_PATH, MEMBRANE, PLASMODESMATA))
        De = (np.asarray(apoplastic_diffusion_factor) * P_edge["d_apoplast"]
              + np.asarray(symplastic_diffusion_factor) * P_edge["d_plasmodesmata"]
              + np.asarray(membrane_diffusion_factor) * P_edge["d_membrane"])
        carried = np.where(path == MEMBRANE, 1.0 - np.asarray(reflection_coefficient, dtype=float), 1.0)
        a, b = edge_weights(np.where(on_liquid, De, 0.), np.where(on_liquid, np.asarray(flow) * carried, 0.))
        sink = self._sinks(np.asarray(kind), np.asarray(cgroup), np.asarray(tissue, dtype=object)) & liquid
        cell = (np.asarray(kind) == 0) & liquid
        volume = np.asarray(node_volume, dtype=float)
        return (a, "head_weight", b, "sucrose_sink", sink.astype(float),
                "sucrose_sink_value", P_node["c_sink"],
                "liquid_node", liquid.astype(float),
                "synthesis_rate", np.where(cell, P_node["k_starch_synthesis"] * volume, 0.),
                "degradation_rate", np.where(cell, P_node["k_starch_degradation"] * volume, 0.))

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
        """Implicit Euler on the liquid nodes: V (c − c_prev) / dt + B q − K V c − s = 0, per species."""

        @node_balance(field="sucrose")
        def _sucrose(self, sucrose, starch, tail_weight, head_weight, node_volume, synthesis_rate, degradation_rate,
                     sucrose_source):
            B = self._graph_view.incidence
            tail = np.asarray(B.maximum(0).T @ sucrose).reshape(-1)
            head = np.asarray((-B).maximum(0).T @ sucrose).reshape(-1)
            q = tail_weight * tail - head_weight * head
            return (node_volume * (sucrose - self.previous("sucrose")) / (self.dt / SECONDS_PER_DAY)
                    + np.asarray(B @ q).reshape(-1) + synthesis_rate * sucrose - degradation_rate * starch
                    - sucrose_source)

        @node_balance(field="starch")
        def _starch(self, sucrose, starch, node_volume, synthesis_rate, degradation_rate):
            return (node_volume * (starch - self.previous("starch")) / (self.dt / SECONDS_PER_DAY)
                    - synthesis_rate * sucrose + degradation_rate * starch)

        sinks = boundary_set(filters={"sucrose_sink": ">0"}, kind="dirichlet", value="sucrose_sink_value",
                             field="sucrose")

        @graph_jacobian
        def _jacobian(self, tail_weight, head_weight, node_volume, synthesis_rate, degradation_rate):
            B = self._graph_view.incidence
            storage = node_volume / (self.dt / SECONDS_PER_DAY)
            transport = B @ diags(tail_weight) @ B.maximum(0).T - B @ diags(head_weight) @ (-B).maximum(0).T
            return bmat([[transport + diags(storage + synthesis_rate), diags(-degradation_rate)],
                         [diags(-synthesis_rate), diags(storage + degradation_rate)]], format="csr")
