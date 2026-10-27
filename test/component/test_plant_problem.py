"""
The axial junctions' conductances, and the plant problem, the sections joined by
their xylem junctions, the soil around them and the collar condition on the first section's vessels.
"""
import numpy as np
import pytest

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.postprocess import VISCOSITY_PHLOEM, VISCOSITY_XYLEM, poiseuille

from plants import anatomy_data_structure, root

SOIL = -1000.                                          # hPa


def _root(n, barriers=None, **parameters):
    g, ds, anatomy = anatomy_data_structure(root(n, length=5e-3))
    owner = ds.owner("SubOrgan")
    barriers = np.ones(n) if barriers is None else np.asarray(barriers, dtype=float)
    ds.register("barrier", barriers[owner], location="node")
    ds.register("soil_psi", np.full(ds.n_nodes(), SOIL), location="node")
    mecha = MechaHydraulics(data_structure=ds)
    for name, value in parameters.items():
        setattr(mecha, name, value)
    mecha()
    return g, ds, mecha


@pytest.fixture(scope="module")
def four():
    return _root(4)


def _ends(ds):
    view = ds.to_graph_view()
    return view.tail, view.head


def test_junction_conductances_poiseuille_in_series(four):
    g, ds, mecha = four
    conduit, Kx, Kp = ds.get("axial_conduit"), ds.get("xylem_axial_conductance"), ds.get("phloem_axial_conductance")
    a, b = _ends(ds)
    area, half = ds.get("cell_area") * 1e6, ds.get("length") * 1e6 / 2
    xylem, phloem = np.flatnonzero(conduit == 1), np.flatnonzero(conduit == 2)
    assert xylem.size > 0 and phloem.size > 0
    for edges, K, viscosity in ((xylem, Kx, VISCOSITY_XYLEM), (phloem, Kp, VISCOSITY_PHLOEM)):
        k_a, k_b = poiseuille(area[a[edges]], half[a[edges]], viscosity), poiseuille(area[b[edges]], half[b[edges]],
                                                                                     viscosity)
        np.testing.assert_allclose(K[edges], 1 / (1 / k_a + 1 / k_b), rtol=1e-14)
    assert (Kx[conduit != 1] == 0).all() and (Kp[conduit != 2] == 0).all()


def test_junctions_carry_the_sections_kx(four):
    """Between identical sections, each vessel joined to its copy: Σ junction K = Σ vessel K(h) = Kx·1e4/h."""
    g, ds, mecha = four
    owner = ds.owner("SubOrgan")
    a, b = _ends(ds)
    conduit, K = ds.get("axial_conduit"), ds.get("xylem_axial_conductance")
    h = float(ds.get("length")[0]) * 1e6
    between = (conduit == 1) & (owner[a] == 0) & (owner[b] == 1) | (conduit == 1) & (owner[a] == 1) & (owner[b] == 0)
    assert K[between].sum() == pytest.approx(ds.get("Kx")[0] * 1e4 / h, rel=1e-12)


def test_no_water_through_immature_xylem():
    g, ds, mecha = _root(3, barriers=(1, 1, 0))
    owner = ds.owner("SubOrgan")
    a, b = _ends(ds)
    K, conduit = ds.get("xylem_axial_conductance"), ds.get("axial_conduit")
    to_tip = (conduit == 1) & ((owner[a] == 2) | (owner[b] == 2))
    assert to_tip.any() and (K[to_tip] == 0).all()


def test_plant_problem_mass_balance_and_bounds(four):
    g, ds, mecha = four
    psi, flow, owner = ds.get("psi"), ds.get("flow"), ds.owner("SubOrgan")
    collar = ds.get("collar_kind") == 1
    assert collar.any() and (owner[collar] == 0).all()
    np.testing.assert_allclose(psi[collar], mecha.collar_psi)
    assert psi.min() >= mecha.collar_psi - 1e-6 and psi.max() <= SOIL + 1e-6
    outflow = np.asarray(ds.incidence_matrix() @ flow).reshape(-1)             # out of each node, along its edges
    extracted = -outflow[collar].sum()                                         # what the collar takes
    uptake = ds.get("uptake")
    assert (uptake > 0).all() and extracted == pytest.approx(uptake.sum(), rel=1e-9)


def test_uptake_and_xylem_potential_along_the_root(four):
    """Farther from the collar: a higher xylem potential, a smaller uptake (the axial resistance)."""
    g, ds, mecha = four
    psi, owner, cgroup = ds.get("psi"), ds.owner("SubOrgan"), ds.get("cgroup")
    xylem = np.isin(cgroup, (13, 19, 20))
    psi_x = [psi[(owner == i) & xylem].mean() for i in range(4)]
    assert np.all(np.diff(psi_x) > 0) and np.all(np.diff(ds.get("uptake")) < 0)
    kr_eff = ds.get("kr_effective")
    assert np.isfinite(kr_eff).all() and (kr_eff > 0).all()


def test_neumann_collar():
    g, ds, mecha = _root(2, collar_condition=2., collar_flow=1e-3)
    assert (ds.get("collar_kind")[ds.get("collar_kind") > 0] == 2).all()
    assert ds.get("uptake").sum() == pytest.approx(1e-3, rel=1e-9)


def test_elongation_sink_drawn_from_the_soil():
    """A root tip without barriers (barrier 0) and an older section: the elongation sink of the tip (MECHA's 2.8
    cm d-1) is the plant problem's, the standard problem's only with standard_elongation; with no collar flow the
    root takes from the soil what elongation withdraws."""
    g, ds, mecha = _root(2, barriers=(1, 0), elongation_midpoint_rate=2.8, collar_condition=2., collar_flow=0.)
    sink = np.asarray(ds.get("elongation_sink"))
    owner = ds.owner("SubOrgan")
    assert sink[owner == 1].sum() > 0 and sink[owner == 0].sum() == 0
    assert np.asarray(ds.get("standard_elongation_sink")).sum() == 0
    assert np.nansum(ds.get("root_uptake")) == pytest.approx(sink.sum(), rel=1e-9)


def test_measured_axial_conductivities():
    """measured_axial_conductivity (MECHA's Kax_source 2) replaces a conduit's Poiseuille conductance in the axial
    junctions and the standard problem's axial boundary: Poiseuille's own values change nothing; doubled on every
    vessel, the junctions' conductances double (cells in series), and so does Kx."""
    g, ds, mecha = _root(3)
    conduit = np.asarray(ds.get("axial_conduit"))
    reference = {name: np.asarray(ds.get(name)).copy() for name in ("psi", "standard_psi", "xylem_axial_conductance",
                                                                       "Kx")}
    area, length = np.asarray(ds.get("cell_area")) * 1e6, np.asarray(ds.get("length")) * 1e6
    vessels = np.isin(np.asarray(ds.get("cgroup")), (13, 19, 20))
    own = np.where(vessels, poiseuille(area, length, VISCOSITY_XYLEM) * length / 1e4, np.nan)
    ds.set("measured_axial_conductivity", own)
    mecha()
    for name in ("psi", "standard_psi", "xylem_axial_conductance", "Kx"):
        np.testing.assert_allclose(ds.get(name), reference[name], rtol=1e-12, atol=1e-12 * abs(reference[name]).max())
    ds.set("measured_axial_conductivity", 2 * own)
    mecha()
    xylem = conduit == 1
    np.testing.assert_allclose(np.asarray(ds.get("xylem_axial_conductance"))[xylem],
                               2 * reference["xylem_axial_conductance"][xylem], rtol=1e-12)
    np.testing.assert_allclose(ds.get("Kx"), 2 * reference["Kx"], rtol=1e-12)
