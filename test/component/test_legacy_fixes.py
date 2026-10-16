"""The legacy's problems fixed in the port (devplan_metafspm.md, §18): each fix's rule, where the legacy differs."""
from types import SimpleNamespace

import numpy as np

from openalea.mecha.conductance import boundary_terms


def test_a1_protosieve_cell_counted_once():
    """§2 f: a protosieve cell listed twice by the legacy gets k_sieve once in the phloem boundary term (barrier 0)."""
    n = 3
    net = SimpleNamespace(cgroup=np.full(n, 23), x=np.zeros(n), border_wall=np.zeros(n, bool),
                          border_junction=np.zeros(n, bool), wall_length=np.zeros(n),
                          xylem=np.zeros(n, np.int64), protosieve=np.array([0, 1, 2]))
    P = {name: np.full(n, value) for name, value in
         (("kw", 2.4e-4), ("thickness", 1.5), ("x_contact", 0.), ("k_xyl", 1e-6), ("k_sieve", 3e-6))}
    _, axial, rhs = boundary_terms(net, P, barrier=np.zeros(n), height=np.full(n, 200.))
    assert axial.tolist() == [0., 3e-6, 3e-6] and rhs.tolist() == axial.tolist()
