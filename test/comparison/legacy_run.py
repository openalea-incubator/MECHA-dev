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
