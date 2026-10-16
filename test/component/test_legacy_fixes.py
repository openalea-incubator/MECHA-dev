"""The legacy's problems fixed in the port (devplan_metafspm.md, §18): each fix's rule, where the legacy differs."""
from types import SimpleNamespace

import numpy as np
import pytest

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.conductance import boundary_terms

from plants import anatomy_data_structure, root


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


def test_a2_each_vessel_gets_its_own_share():
    """§2 o: with a flow at the xylem, each vessel (the network's, by cgroup) receives the flow × its own area fraction,
    and kr's xylem potential is those vessels' mean."""
    g, ds, anatomy = anatomy_data_structure(root(1))
    ds.register("barrier", np.full(ds.n_nodes(), 3.), location="node")
    mecha = MechaHydraulics(data_structure=ds)
    mecha.standard_axial_condition, mecha.standard_axial_flow = 2., 1e-3
    mecha()
    nodes, _, geo = mecha.section_geometries()[int(ds.entity_ids("SubOrgan")[0])]
    vessels = geo.xylem_cells
    inflow = np.asarray(ds.get("standard_axial_inflow"))[nodes]
    area = geo.cell_area[vessels]
    np.testing.assert_allclose(-inflow[vessels], 1e-3 * area / area.sum(), rtol=1e-14)
    assert inflow.sum() == pytest.approx(-1e-3, rel=1e-12)
    psi_xylem = np.asarray(ds.get("standard_psi"))[nodes][vessels].mean()
    kr = 1e-3 / (mecha.standard_psi_soil - psi_xylem) / ds.get("perimeter")[0] / (ds.get("length")[0] * 1e6) / 1e-4
    assert ds.get("kr")[0] == pytest.approx(kr, rel=1e-12)
