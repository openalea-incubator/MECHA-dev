"""
The legacy MECHA on a GRANAP section: the reference the port is compared with (devplan_metafspm.md, §4 step 1).

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


# The port's options that reproduce the legacy where it departs from the original MECHA (devplan_metafspm.md §18):
# its plasmodesmata rules and its aperture factors' defaults, its prescribed xylem boundary, its one wall thickness
LEGACY_OPTIONS = dict(plasmodesmata_rules=2., kpl_PCC_factor=0., kpl_PST_factor=0., standard_axial_source=2.,
                      thickness_source=1.)


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


def legacy_section(organ, barriers=(1,), height=200.0, h=0) -> LegacyRun:
    """The legacy MECHA on a GRANAP organ, one solve per maturity stage (barrier code, height in µm)."""
    organ.export_to_adjencymatrix()
    data = InData()
    data.geometry.set_maturity_stages(list(barriers), height=[height] * len(barriers))
    network = NetworkBuilder(organ)
    network.populate_from_network()
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
    data = InData()
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


def legacy_osmotic(organ, barrier, height=200.0, **overrides):
    """
    The legacy's osmotic scenario (``Mecha.water_flux``, its scenario 1: the default scenario with *overrides*, e.g.
    os_hetero, s_hetero) on one maturity stage. Returns dict(solution, psi_os {node: π}, sigma {(wall, cell): σ},
    edge_Q {(u, v): Q}, total_flow).
    """
    organ.export_to_adjencymatrix()
    data = InData()
    data.geometry.set_maturity_stages([barrier], height=[height])
    scenario = dict(data.boundary.scenarios[0])
    scenario.update(overrides)
    data.boundary.add_scenario(scenario)
    network = NetworkBuilder(organ)
    network.populate_from_network()
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
                total_flow=float(np.asarray(mecha.total_flow[0][1]).reshape(-1)[0]))
