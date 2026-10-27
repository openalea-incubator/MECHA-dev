"""
The legacy MECHA on a GRANAP section: the reference the port is compared with.

    run = legacy_section(organ, barriers=(1, 3), height=200.)
    run.network            the legacy NetworkBuilder (its derived geometry: cell_ranks, distance_center_grav, ...)
    run.mecha              the legacy Mecha
    run.stages[i]          per maturity stage: barrier, solution, matrix_W, Kmb, rhs (soil), edge K and Q (keys: the
                           graph's (u, v), Q from u to v), kr, Kx

The organ is GRANAP's (the main package, generating into its MPG); the legacy reads it through GRANAP's legacy
exports (its networkx graph, cell GeoDataFrame and cell list), as before the port.
"""
from dataclasses import dataclass, field

import numpy as np

from openalea.mecha.legacy import InData, Mecha, NetworkBuilder


# The port's options that reproduce the legacy where it departs from the original MECHA:
# its plasmodesmata rules and its aperture factors' defaults, its prescribed xylem boundary, its one wall thickness,
# its membrane floor (ME-33), its axial penalty conductances (ME-32: develop's 1.0)
LEGACY_OPTIONS = dict(plasmodesmata_rules=2., kpl_PCC_factor=0., kpl_PST_factor=0., standard_axial_source=2.,
                      thickness_source=1., membrane_floor=2., k_xyl=1.0, k_sieve=1.0)


def legacy_input(kpl=None):
    """The legacy's default input (InData()), with its phloem-pole plasmodesmata factor (PPP) 1 instead of 0: with
    develop's factors applied (doc/legacy.rst, d2), PPP 0 closes the companion cells' plasmodesmata with the stele,
    xylem and pericycle, isolating them; the port does not reproduce that. Set under the key the solver reads
    (``phloem_pericycle_pole_factor``; the loader stores the input's under ``pericycle_phloem_pole_factor``).
    *kpl*: other entries of its plasmodesmata configuration ({key the solver reads: value})."""
    data = InData()
    for entry in data.hydraulic.kpl:
        entry["phloem_pericycle_pole_factor"] = 1.0
        entry.update(kpl or {})
    return data


def legacy_options(mecha):
    """*mecha* (a MechaHydraulics) set to the legacy's choices, for the comparison; returns it."""
    for name, value in LEGACY_OPTIONS.items():
        setattr(mecha, name, value)
    return mecha


@dataclass
class LegacyRun:
    network: object
    mecha: object
    stages: list = field(default_factory=list)


def legacy_section(organ, barriers=(1,), height=200.0, h=0, kpl=None, hydraulic=None, edit_network=None) -> LegacyRun:
    """The legacy MECHA on a GRANAP organ, one solve per maturity stage (barrier code, height in µm); *kpl*: its
    plasmodesmata configuration's entries (legacy_input); *hydraulic*: its HydraulicData's attributes to set
    ({name: value}); *edit_network*: a function of its NetworkBuilder, called once the network is built."""
    organ.export_to_adjencymatrix()
    data = legacy_input(kpl)
    for name, value in (hydraulic or {}).items():
        setattr(data.hydraulic, name, value)
    data.geometry.set_maturity_stages(list(barriers), height=[height] * len(barriers))
    network = NetworkBuilder(organ)
    network.populate_from_network()
    if edit_network is not None:
        edit_network(network)
    mecha = Mecha(data, network=network)
    run = LegacyRun(network=network, mecha=mecha)
    for i, barrier in enumerate(barriers):
        solution, _, matrix_W, Kmb, rhs = mecha.solve_W(h=h, i_maturity=i)
        graph = network.graph
        edge_K = {(int(u), int(v)): float(attrs["K"]) for u, v, attrs in graph.edges(data=True) if "K" in attrs}
        edge_Q = {(int(u), int(v)): float(attrs["Q"]) for u, v, attrs in graph.edges(data=True) if "Q" in attrs}
        stage = dict(barrier=barrier, solution=np.asarray(solution).reshape(-1), matrix_W=matrix_W.copy(),
                     Kmb=np.asarray(Kmb).copy(), rhs=np.asarray(rhs).copy(), edge_K=edge_K, edge_Q=edge_Q,
                     kr=float(mecha.kr_tot[i, 0]))
        if barrier > 0:                                      # the legacy's Kx fails for barrier 0 (§2 e)
            _, stage["Kx"] = mecha.calculate_axial_conductance(i)
        run.stages.append(stage)
    return run


def legacy_neumann(organ, barrier, flow, height=200.0):
    """
    The legacy's standard problem with a flow condition at the xylem (its tutorial 02): the xylem pressure removed,
    *flow* (cm³ d⁻¹) distributed among the vessels, solved; the xylem's potential recovered as their mean, kr computed
    from it as the tutorial does. Returns dict(solution, total_flow, psi_xyl, kr).
    """
    organ.export_to_adjencymatrix()
    data = legacy_input()
    data.geometry.set_maturity_stages([barrier], height=[height])
    network = NetworkBuilder(organ)
    network.populate_from_network()
    mecha = Mecha(data, network=network)
    mecha.psi_xyl[1, :, 0] = np.nan
    mecha.distributed_flow_xyl[1, 0, 0] = flow
    mecha._distribute_xylem_flow()
    solution, _, _, _, _ = mecha.solve_W(h=0, i_maturity=0)
    psi_xyl = float(mecha.psi_xyl[1][0][0])
    total = float(np.asarray(mecha.total_flow[0][0]).reshape(-1)[0])
    psi_soil = data.boundary.scenarios[0]["psi_soil_left"]
    kr = total / (psi_soil - psi_xyl) / network.perimeter / height / 1.0E-04
    return dict(solution=np.asarray(solution).reshape(-1), total_flow=total, psi_xyl=psi_xyl, kr=kr)



def legacy_frame(run, stage, organ):
    """
    The legacy's section as the port's plots take it (openalea.mecha.plots.SectionFrame): its graph's positions and
    node kinds, its cell ranks and types, its edges with their K and Q at stage *stage*, its solution, and the GRANAP
    organ's cell polygons (µm).
    """
    from shapely import affinity
    from openalea.mecha.plots import SectionFrame
    net = run.network
    graph = net.graph
    n = net.n_wall_junction + net.n_cells
    xy = np.array([graph.nodes[i]["position"] for i in range(n)], dtype=float)
    kind = np.r_[np.ones(net.n_walls), 2 * np.ones(net.n_junctions), np.zeros(net.n_cells)].astype(int)
    rank = np.r_[-np.ones(net.n_wall_junction), net.cell_ranks].astype(int)
    tissue = np.array([None] * net.n_wall_junction + list(net.cell_types), dtype=object)
    data = run.stages[stage]
    keys = list(data["edge_K"])
    a, b = np.array([k[0] for k in keys]), np.array([k[1] for k in keys])
    path = np.array([graph.edges[u, v]["path"] for u, v in keys])
    polygons = [None if c.polygon is None else affinity.scale(c.polygon, 1000., 1000., origin=(0, 0))
                for c in organ.all_cells.cells[:net.n_cells]]
    return SectionFrame(xy=xy, kind=kind, rank=rank, tissue=tissue, a=a, b=b, path=path,
                        K=np.array([data["edge_K"][k] for k in keys]), psi=data["solution"],
                        Q=np.array([data["edge_Q"][k] for k in keys]), polygons=polygons,
                        centre=(net.x_grav, net.y_grav))


def legacy_osmotic(organ, barrier, height=200.0, scenario_zero=None, edit_network=None, hydraulic=None, **overrides):
    """
    The legacy's osmotic scenario (``Mecha.water_flux``, its scenario 1: the default scenario with *overrides*, e.g.
    os_hetero, s_hetero) on one maturity stage. *scenario_zero*: entries of its scenario 0 too (the scenario loop
    reads some of scenario i_maturity's: doc/legacy.rst, h); *edit_network*: a function of its NetworkBuilder, called
    once the network is built; *hydraulic*: its HydraulicData's attributes to set ({name: value}). Returns dict(solution, psi_os {node: π}, sigma {(wall, cell): σ}, edge_Q {(u, v): Q},
    total_flow, mecha (its Mecha, scenario 1 solved)).
    """
    organ.export_to_adjencymatrix()
    data = legacy_input()
    data.geometry.set_maturity_stages([barrier], height=[height])
    data.boundary.scenarios[0].update(scenario_zero or {})
    for name, value in (hydraulic or {}).items():
        setattr(data.hydraulic, name, value)
    scenario = dict(data.boundary.scenarios[0])
    scenario.update(overrides)
    data.boundary.add_scenario(scenario)
    network = NetworkBuilder(organ)
    network.populate_from_network()
    if edit_network is not None:
        edit_network(network)
    mecha = Mecha(data, network=network)
    mecha.water_flux(verbose=False)
    result = [r for r in mecha.results if r.get("scenario") == 1][-1]
    cm = network.cell_manager
    psi_os, sigma = {}, {}
    for u, v, attrs in network.graph.edges(data=True):
        if attrs.get("path") != "membrane":
            continue
        wall, cell = (int(u), int(v)) if int(u) < network.n_walls else (int(v), int(u))
        w, c, mb = cm.get_wall_by_node_id(wall), cm.get_by_node_id(cell), cm.get_membrane_by_edge(wall, cell)
        psi_os[wall], psi_os[cell] = w.psi_os, c.psi_os
        sigma[(wall, cell)] = mb.sigma
    edge_Q = {(int(u), int(v)): float(attrs["Q"]) for u, v, attrs in network.graph.edges(data=True) if "Q" in attrs}
    return dict(solution=np.asarray(result["solution"]).reshape(-1), psi_os=psi_os, sigma=sigma, edge_Q=edge_Q,
                total_flow=float(np.asarray(mecha.total_flow[0][1]).reshape(-1)[0]), mecha=mecha)


def legacy_reactions_needle(organ, psi_air=-1.0E4, psi_xyl=-200.0, sigma=0.7):
    """
    The reactions branch's needle water problem (``simulations/needle.py``, ``NeedleSimulation._build_network`` and
    ``_resolve_air_bc``): ``InData.needle_defaults()``, its scenario 0 (all zero: the xylem and the soil at 0) and a
    scenario 1 (the default root scenario with s_factor = σ, the xylem at *psi_xyl*), the evaporating walls pinned at
    *psi_air*; ``water_flux``. Returns per scenario dict(solution, edge_K, edge_Q, psi_os {node: π}, sigma
    {(wall, cell): σ}), and the pinned walls, the penalty k_air and the wall_air coefficients.
    """
    import copy
    from openalea.mecha.legacy_reactions import InData as RInData, Mecha as RMecha, NetworkBuilder as RNetwork
    from openalea.mecha.legacy_reactions.utils.data_loader import BoundaryData
    organ.export_to_adjencymatrix()
    network = RNetwork(organ)
    network.populate_from_network()
    data = RInData.needle_defaults()
    s1 = copy.deepcopy(BoundaryData().scenarios[0])
    s1["s_factor"] = sigma
    data.boundary.add_scenario(s1)
    mecha = RMecha(data, network=network)
    mecha.psi_xyl[1, 0, 1] = psi_xyl
    mecha.psi_sieve[1, 0, 1] = np.nan
    walls = mecha.set_air_wall_bc(psi_air)
    graph, cm = network.graph, network.cell_manager
    out = {}
    standard = None

    def record():
        edge_K = {(int(u), int(v)): float(a["K"]) for u, v, a in graph.edges(data=True) if "K" in a}
        edge_Q = {(int(u), int(v)): float(a["Q"]) for u, v, a in graph.edges(data=True) if "Q" in a}
        psi_os, sig = {}, {}
        for u, v, a in graph.edges(data=True):
            if a.get("path") != "membrane":
                continue
            wall, cell = (int(u), int(v)) if int(u) < network.n_walls else (int(v), int(u))
            w, c, mb = cm.get_wall_by_node_id(wall), cm.get_by_node_id(cell), cm.get_membrane_by_edge(wall, cell)
            psi_os[wall], psi_os[cell], sig[(wall, cell)] = w.psi_os, c.psi_os, mb.sigma
        return dict(edge_K=edge_K, edge_Q=edge_Q, psi_os=psi_os, sigma=sig)

    solution, _, _, _, _ = mecha.solve_W(h=0, i_maturity=0)
    standard = dict(solution=np.asarray(solution).reshape(-1), **record())
    wall_air = {(int(u), int(v)): (float(a["K_wall"]), float(a["K_air"]), float(a["offset"]))
                for u, v, a in graph.edges(data=True) if a.get("path") == "wall_air"}
    k_air = mecha._air_wall_bc["k_air"]
    mecha.water_flux(verbose=False)
    result = [r for r in mecha.results if r.get("scenario") == 1][-1]
    out[0] = standard
    out[1] = dict(solution=np.asarray(result["solution"]).reshape(-1), **record())
    return dict(scenarios=out, walls=list(walls), k_air=k_air, wall_air=wall_air, network=network, mecha=mecha)


def legacy_reactions_step(organ, d_apo=0.1, d_pd=0.5, d_mem=1e-6, dt=0.1, c_pulse=50e-6, starch=None, cells_only=False,
                          source=None, solute_sigma=None, osmotic_term=True, air_flows=False, reference=None,
                          c_start=None, starch_start=None, psi_air=-1.0E4, psi_xyl=-200.0, sigma=0.7, storage=None,
                          k_deg=0.0):
    """
    One transport step of the reactions branch's needle (``simulations/needle.py``, NeedleDynamicSimulation), on the
    water state of legacy_reactions_needle: its SoluteTransport ('full', SG, θ = 1), the solute's σ 0 and the air
    edges' flows set to 0 (so that its problems r1, r2, r12 vanish, plan §28), its sinks and anchors, a pulse
    c_pulse in the mesophyll; with *starch* = (k_syn, k_deg), its MultiSoluteTransport with an immobile starch (with
    *cells_only*, the reaction in the liquid cells only: the port's Q-U4); with *solute_sigma* {cgroup: σ}, its
    solute's σ (r1, r2), without its term K σ Δπ added to the flows if not *osmotic_term* (r1 removed); with
    *air_flows*, the air edges' flows kept (r12); with
    *source* {cell id: increment}, its photosynthetic increment before the step; from *c_start* (and *starch_start*)
    instead of the pulse, on the water state *reference* (a previous step's, its air flows already as *air_flows*)
    to chain steps; with *storage* (C_wall, C_cell), its capacitance's storage fractions, and *k_deg* (d⁻¹) its
    first-order degradation. Returns dict(D, volumes, flows
    {(u, v): Q}, bc, c0, c1[, starch1]).
    """
    from scipy.sparse import csgraph
    from openalea.mecha.legacy_reactions.utils.solute_transport import SoluteGeometry, SoluteTransport
    if reference is None:
        reference = legacy_reactions_needle(organ, psi_air=psi_air, psi_xyl=psi_xyl, sigma=sigma)
    mecha, network = reference["mecha"], reference["network"]
    graph, nwj = network.graph, network.n_wall_junction
    geometry = SoluteGeometry(mecha)
    cap = {"dt": dt}
    if storage is not None:
        cap.update(C_wall=storage[0], C_cell=storage[1])
    st = SoluteTransport(mecha, {"apo_wall": d_apo, "plasmodesmata": d_pd, "membrane": d_mem,
                                 "sigma": dict(solute_sigma or {})}, cap,
                         mode="full", geometry=geometry, reaction_params={"k_deg": k_deg} if k_deg else None)
    D = st.build_diffusion_matrix(0, 0)
    air = {(int(u), int(v)) for u, v, a in graph.edges(data=True) if a.get("path") in ("wall_air", "air_link")}
    for edge in [] if air_flows else mecha.edge_flux_list[0][1]:
        if (edge["source"], edge["target"]) in air or (edge["target"], edge["source"]) in air:
            edge["flux"] = 0.0
    flows = {(int(e["source"]), int(e["target"])): float(e["flux"]) for e in mecha.edge_flux_list[0][1]}
    kinds = {}
    for node, data in graph.nodes(data=True):
        i = mecha.indice[node]
        if i < nwj:
            continue
        cg, ct = int(data.get("cgroup", -1)), str(data.get("cell_type", ""))
        if network.is_wall_air_cell(node):
            kinds.setdefault("air", []).append(i)
        elif cg == 4 and ct == "mesophyll":
            kinds.setdefault("meso", []).append(i)
        elif cg == 12 and ct == "Strasburger cell":
            kinds.setdefault("strasburger", []).append(i)
        elif cg == 11:
            kinds.setdefault("phloem", []).append(i)
        elif cg == 13:
            kinds.setdefault("xylem", []).append(i)
    loading = kinds.get("strasburger") or kinds.get("phloem", [])
    isolated = list(np.flatnonzero(np.asarray(abs(D).sum(axis=1)).ravel() == 0))
    n_comp, labels = csgraph.connected_components(D, directed=False, connection="weak")
    anchored = set(loading) | set(kinds.get("xylem", [])) | set(kinds.get("meso", []))
    extra = []
    for comp in range(n_comp):
        members = np.flatnonzero(labels == comp)
        if any(m in anchored for m in members):
            continue
        rest = [m for m in members if m not in set(isolated)]
        if rest:
            extra.append(rest[0])
    bc = {int(n): 0.0 for n in isolated + extra + list(loading) + kinds.get("xylem", [])}
    c0 = np.zeros(graph.number_of_nodes())
    c0[kinds.get("meso", [])] = c_pulse
    if c_start is not None:
        c0 = np.array(c_start, dtype=float)
    for n, value in bc.items():
        c0[n] = value
    c_prev = c0.copy()
    if source:
        for i, increment in source.items():
            c_prev[i] += increment
    out = dict(D=D, volumes=geometry.node_volumes(0), flows=flows, bc=bc, c0=c0, kinds=kinds, network=network,
               reference=reference, extra=extra, isolated=isolated)
    cm = network.cell_manager                     # r1's K σ Δπ: from the cells' and walls' π, 0 while solving
    held = [] if osmotic_term else [o for o in list(cm) + [cm.get_wall_by_node_id(i) for i in range(nwj)]
                                    if o is not None and getattr(o, "psi_os", None) is not None]
    saved = [o.psi_os for o in held]
    for o in held:
        o.psi_os = 0.0
    try:
        s0 = np.zeros(c0.size) if starch_start is None else np.asarray(starch_start, dtype=float)
        _legacy_solve(out, st, mecha, geometry, cap, bc, c_prev, s0, starch, cells_only, kinds, nwj)
    finally:
        for o, value in zip(held, saved):
            o.psi_os = value
    return out


def _legacy_solve(out, st, mecha, geometry, cap, bc, c_prev, s_prev, starch, cells_only, kinds, nwj):
    """legacy_reactions_step's step: its SoluteTransport, or its MultiSoluteTransport with starch."""
    from openalea.mecha.legacy_reactions.utils.solute_transport import MultiSoluteTransport, SoluteTransport
    if starch is None:
        out["c1"] = st.solve(h=0, i_maturity=0, i_scenario=1, rhs=np.zeros(c_prev.size), boundary_conditions=bc,
                             c_prev=c_prev, theta=1.0, scheme="sg")
    else:
        immobile = SoluteTransport(mecha, {"apo_wall": 0.0, "plasmodesmata": 0.0, "membrane": 0.0}, cap,
                                   mode="full", mobile=False, geometry=geometry)
        k_syn, k_deg = starch
        multi = MultiSoluteTransport(geometry, [st, immobile], np.array([[-k_syn, k_deg], [k_syn, -k_deg]]))
        if cells_only:                                # the reaction's volumes: the liquid cells' only (Q-U4)
            volumes = multi._mode_volumes(0).copy()
            volumes[:nwj] = 0.
            volumes[kinds.get("air", [])] = 0.
            multi._mode_volumes = lambda i_maturity: volumes
        out["c1"], out["starch1"] = multi.solve_coupled(h=0, i_maturity=0, i_scenario=1,
                                                       c_prev=[c_prev, s_prev],
                                                       boundary_conditions=[bc, None], theta=1.0, scheme="sg")


def legacy_fractions(organ, barrier, height=200.0):
    """
    The legacy's standard transmembrane fractions (``Mecha.water_flux``, its scenario 0): dict(membranes [(wall,
    cell)] in its membrane order, stf_mb, stf_layer_plus, stf_layer_minus, uptake_layer_plus, uptake_layer_minus,
    stf_cell_plus, stf_cell_minus, total_flow, solution), each of its scenario 0 and maturity stage 0.
    """
    organ.export_to_adjencymatrix()
    data = legacy_input()
    data.geometry.set_maturity_stages([barrier], height=[height])
    network = NetworkBuilder(organ)
    network.populate_from_network()
    mecha = Mecha(data, network=network)
    mecha.water_flux(verbose=False)
    membranes = []
    for node, edges in network.graph.adjacency():
        i = mecha.indice[node]
        if i < network.n_walls:
            membranes += [(i, mecha.indice[j]) for j, attrs in edges.items() if attrs["path"] == "membrane"]
    out = {name: np.asarray(getattr(mecha, name))[..., 0, 0].copy() for name in
           ("stf_mb", "stf_layer_plus", "stf_layer_minus", "uptake_layer_plus", "uptake_layer_minus",
            "stf_cell_plus", "stf_cell_minus")}
    return dict(membranes=membranes, total_flow=float(np.asarray(mecha.total_flow[0][0]).reshape(-1)[0]),
                solution=np.asarray(mecha.results[0]["solution"]).reshape(-1), mecha=mecha, **out)


def legacy_standard_solute_flux(run, c_prev, boundary_conditions, d_apo=0.1, d_pd=0.5, d_mem=1e-6, dt=0.1,
                                mode="full", theta=1.0, scheme="sg", operators="T", conserve_boundary=True,
                                rhs=None):
    """
    One solute step of the legacy's ``Mecha.standard_solute_flux`` on *run* (a legacy_section's, its first maturity
    stage's standard problem: its scenario 0's edge flows), from *c_prev* (by legacy index) with the Dirichlet
    *boundary_conditions* {legacy index: c}; *dt* None: its steady state. Returns the concentrations (its mode's
    nodes).
    """
    params = {"apo_wall": d_apo, "plasmodesmata": d_pd, "membrane": d_mem, "conserve_boundary": conserve_boundary}
    size = {"full": c_prev.size, "apo": run.network.n_wall_junction, "sym": run.network.n_cells}[mode]
    return run.mecha.standard_solute_flux(h=0, i_maturity=0, i_scenario=0, diffusion_params=params,
                                          capacitance_params=None if dt is None else {"dt": dt}, mode=mode,
                                          rhs=np.zeros(size) if rhs is None else rhs,
                                          boundary_conditions=boundary_conditions,
                                          c_prev=None if dt is None else c_prev, theta=theta, operators=operators,
                                          scheme=scheme)
