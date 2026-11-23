"""
What PR #67 (fix/GranapAPI, 7c4ed4f) adds to MECHA, in the port (doc/legacy.rst, d12 on; devplan §31), where the
comparisons with the legacy do not reach: the walls lining air spaces and their check, a needle's protected air
spaces left out.
"""
import numpy as np
import pytest

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.conductance import air_space_lining_walls, root_air_spaces

from plants import anatomy_data_structure, needle, needle_anatomy_component, root


def section(barrier=1., architecture=None, anatomy=None, **parameters):
    """A one-segment root's (or *architecture*'s, its *anatomy*'s) conductances and network, the component's
    *parameters* set."""
    arguments = (architecture or root(1),) + ((anatomy,) if anatomy is not None else ())
    g, data_structure, _ = anatomy_data_structure(*arguments)
    data_structure.register("barrier", np.full(data_structure.n_nodes(), float(barrier)), location="node")
    mecha = MechaHydraulics(data_structure=data_structure)
    for name, value in parameters.items():
        setattr(mecha, name, value)
    mecha()
    return np.asarray(data_structure.get("conductance")).copy(), mecha.network(), mecha


def test_walls_lining_an_air_space():
    """At barrier 1, a wall with one air space among its cells is a hydrated half (kw) and a desiccated half
    (kw_septa) in parallel: its conductance follows kw + kw_septa; every other wall keeps its own; at barrier 0
    none is halved."""
    K, net, mecha = section(kw_septa=1.0E-4)
    K_drier, _, _ = section(kw_septa=0.5E-4)
    lining = air_space_lining_walls(net, mecha.parameter_table(mecha.PARAMETERS, "edge"), np.ones(net.path.size))
    assert lining.sum() > 0
    kw = mecha.kw
    np.testing.assert_allclose(K_drier[lining] / K[lining], (kw + 0.5E-4) / (kw + 1.0E-4), rtol=1e-14)
    walls = net.in_section & net.junction_edge
    septa = walls & ~lining & (K != K_drier)
    cells = net.wall_cells[np.where(net.compartment_type[net.a] == 1, net.a, net.b)[septa]]
    assert net.intercellular[np.maximum(cells, 0)].all()          # only the septa between two air spaces differ
    K0, _, _ = section(barrier=0., kw_septa=1.0E-4)
    K0_drier, _, _ = section(barrier=0., kw_septa=0.5E-4)
    np.testing.assert_array_equal(K0[walls], K0_drier[walls])


def test_kw_septa_below_kw():
    """A desiccated wall conducts less than a hydrated one: kw_septa >= kw raises before the conductances, naming the
    sections whose walls line air spaces."""
    with pytest.raises(ValueError, match="kw_septa"):
        section(kw_septa=3.0E-4)
    section(barrier=0., kw_septa=3.0E-4)                           # no wall lining an air space at barrier 0


def test_a_needles_protected_air_spaces_are_not_halved():
    """A needle's mesophyll air spaces (GRANAP's protect_topology, joined to their walls by wall_air edges) keep the
    vapour path: none of its walls is a half wall."""
    _, net, mecha = section(architecture=needle(), anatomy=needle_anatomy_component(), kw=2.5634e-3)
    assert net.intercellular.any()
    assert not root_air_spaces(net)[net.intercellular].any()
    P = mecha.parameter_table(mecha.PARAMETERS, "edge")
    assert not air_space_lining_walls(net, P, np.ones(net.path.size)).any()
