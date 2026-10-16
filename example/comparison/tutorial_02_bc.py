"""
Tutorial 02 against the legacy MECHA: the standard problem with the xylem's potential, then with the same uptake as a
flow into the xylem (the legacy's own procedure: pressure removed, flow distributed, solved, the xylem's potential
recovered), on the same GRANAP section (barrier 3).

The flow case's kr differs from the legacy's (1.6e-4 on the monocot root) since fix A2 (devplan_metafspm.md §18): the
port gives each vessel the share of its own area, the legacy gives them mis-ordered (its problem o).

    python tutorial_02_bc.py          (from MECHA-dev/example/comparison)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "..", "tutorials"), os.path.join(HERE, "..", "..", "test", "comparison")]

from legacy_run import legacy_neumann, legacy_section                            # noqa: E402
from tutorial_01_input import section                                             # noqa: E402

BARRIER = 3


def main():
    ds, mecha, library = section("granap")
    organ = library.by_index(int(ds.get("anatomy_template")[0])).organ
    height = float(np.asarray(ds.get("length"))[0]) * 1.0E6
    ds.set("barrier", np.full(ds.n_nodes(), float(BARRIER)))
    mecha()
    port_potential = dict(uptake=float(ds.get("standard_uptake")[0]), kr=float(ds.get("kr")[0]))
    legacy_run = legacy_section(organ, barriers=(BARRIER,), height=height)
    legacy_potential = dict(uptake=float(np.ravel(legacy_run.mecha.total_flow[0][0])[0]),
                            kr=legacy_run.stages[0]["kr"])
    mecha.standard_axial_condition, mecha.standard_axial_flow = 2., port_potential["uptake"]
    mecha()
    port_flow = dict(uptake=float(ds.get("standard_uptake")[0]), kr=float(ds.get("kr")[0]))
    legacy_flow = legacy_neumann(organ, BARRIER, port_potential["uptake"], height=height)
    legacy_flow = dict(uptake=legacy_flow["total_flow"], kr=legacy_flow["kr"])
    print(f"{'case':12s} {'quantity':10s} {'legacy':>14s} {'port':>14s} {'rel. diff':>10s}")
    for case, a, b in (("potential", legacy_potential, port_potential), ("flow", legacy_flow, port_flow)):
        for name in ("uptake", "kr"):
            print(f"{case:12s} {name:10s} {a[name]:14.8e} {b[name]:14.8e} {abs(a[name] - b[name]) / abs(a[name]):10.1e}")


if __name__ == "__main__":
    main()
