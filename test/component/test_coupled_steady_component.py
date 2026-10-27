"""
The coupled steady state of the water and the sucrose (MechaSolutes' solute_solve 2): its analytic Jacobian against
finite differences of its balances, with the liquid boundary (solute_boundary_advection 1), in both schemes; and the
solve's conservation: at steady state the sucrose produced leaves through the sinks and the liquid boundary.
"""
import inspect

import numpy as np
import pytest

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.osmotic import MechaOsmoticPresets
from openalea.mecha.solutes import MechaSolutes, edge_weights, liquid_boundary_exchange

from plants import anatomy_data_structure, root

PRODUCTION = 2e-13                                   # mol d-1 per cortex cell


@pytest.fixture(scope="module")
def section():
    g, ds, anatomy = anatomy_data_structure(root(1))
    ds.register("barrier", np.ones(ds.n_nodes()), location="node")
    presets = MechaOsmoticPresets(data_structure=ds)
    presets.s_factor = 0.5
    mecha = MechaHydraulics(data_structure=ds)
    mecha.standard_osmotic = 1.
    solutes = MechaSolutes(data_structure=ds)
    ds.set("sucrose_source", np.where(np.asarray(ds.get("cgroup")) == 4, PRODUCTION, 0.))
    presets()
    mecha()
    return ds, solutes


def _call(function, ds, values):
    """*function* (a balance or the Jacobian of the coupled system) on the DataStructure's arrays, with *values*."""
    class View:
        _graph_view = ds.to_graph_view()
    names = list(inspect.signature(function).parameters)[1:]
    return function(View(), *[values[name] if name in values else np.asarray(ds.get(name), dtype=float)
                              for name in names])


@pytest.mark.parametrize("scheme", (1., 2.))
def test_jacobian(section, scheme):
    ds, solutes = section
    solutes.solute_scheme = scheme
    solutes.solute_solve = 1.
    solutes()                                         # the masks and weights, without the coupled solve
    inner = MechaSolutes._steady_coupled._inner_cls
    n = ds.n_nodes()
    rng = np.random.default_rng(1)
    values = dict(coupled_psi=np.asarray(ds.get("standard_psi")) + rng.normal(0, 10, n),
                  sucrose=np.where(np.asarray(ds.get("solute_node")) > 0, rng.uniform(0, 1e-5, n), 0.))

    def residual(psi, sucrose):
        state = dict(values, coupled_psi=psi, sucrose=sucrose)
        return np.r_[_call(inner._water, ds, state), _call(inner._sucrose, ds, state)]
    J = _call(inner._jacobian, ds, values)
    for direction in (np.r_[rng.normal(0, 1, n), np.zeros(n)], np.r_[np.zeros(n), rng.normal(0, 1e-7, n)]):
        eps = 1e-6                                    # small: the upwind weights and the boundary are piecewise
        step = dict(psi=eps * direction[:n], sucrose=eps * direction[n:])
        fd = (residual(values["coupled_psi"] + step["psi"], values["sucrose"] + step["sucrose"])
              - residual(values["coupled_psi"] - step["psi"], values["sucrose"] - step["sucrose"])) / (2 * eps)
        analytic = J @ direction
        for part in (slice(0, n), slice(n, 2 * n)):
            assert abs(fd[part] - analytic[part]).max() <= 1e-5 * abs(fd[part]).max()
    solutes.solute_scheme = 1.


def test_steady_state_conserves_the_sucrose(section):
    """Solved: the production equals what leaves through the sinks (their nodes' net outflow along the solute's edges)
    and with the water leaving to the boundary, to 1e-9."""
    ds, solutes = section
    solutes.solute_solve = 2.
    solutes()
    view = ds.to_graph_view()
    B = view.incidence
    c = np.asarray(ds.get("sucrose"))
    F = np.asarray(ds.get("coupled_flow"))
    a, b = edge_weights(np.asarray(ds.get("diffusion_conductance")), np.asarray(ds.get("carried_fraction")) * F,
                        np.asarray(ds.get("upwind_edge")) > 0)
    q = a * c[view.tail] - b * c[view.head]
    sink = np.asarray(ds.get("sucrose_sink")) > 0
    into_sinks = -np.asarray(B @ q).reshape(-1)[sink].sum()
    advected = np.isfinite(np.asarray(ds.get("boundary_concentration")))
    outflow, _ = liquid_boundary_exchange(B, F, np.asarray(ds.get("water_edge")) > 0, advected)
    leaving = into_sinks + (outflow * c)[~sink].sum()
    produced = np.asarray(ds.get("sucrose_source")).sum()
    assert c.max() > 0 and leaving == pytest.approx(produced, rel=1e-9)
    assert abs(np.asarray(ds.get("coupled_psi")) - np.asarray(ds.get("standard_psi"))).max() > 1e-3
    solutes.solute_solve = 1.
