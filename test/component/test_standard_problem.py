"""MECHA's standard problem on each section: ψ, flows, uptake, kr, Kx, by barrier stage; optional."""
import numpy as np
import pytest

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.geometry import section_parts

from plants import anatomy_data_structure, root


@pytest.fixture(scope="module")
def solved():
    g, ds, anatomy = anatomy_data_structure(root(3))
    ds.register("barrier", np.array([0., 1., 3.])[ds.owner("SubOrgan")], location="node")
    mecha = MechaHydraulics(data_structure=ds)
    mecha()
    return g, ds, mecha


def test_potentials_between_soil_and_xylem(solved):
    g, ds, mecha = solved
    psi, owner = ds.get("standard_psi"), ds.owner("SubOrgan")
    assert np.isfinite(psi).all()
    for i in (1, 2):                                   # barriers: between the xylem's -5000 hPa and the soil's 0
        assert psi[owner == i].min() >= -5000. - 1e-6 and psi[owner == i].max() <= 1e-6
    np.testing.assert_allclose(psi[owner == 0], 0., atol=1e-9)    # barrier 0, no phloem boundary: the soil's


def test_mass_balance(solved):
    """Each section's uptake from the soil is the flow into its xylem (the axial boundary)."""
    g, ds, mecha = solved
    psi, axial, owner = ds.get("standard_psi"), ds.get("axial_boundary_conductance"), ds.owner("SubOrgan")
    uptake = ds.get("standard_uptake")
    for i in (1, 2):
        into_xylem = (axial * (psi - mecha.standard_psi_xylem))[owner == i].sum()
        assert uptake[i] > 0 and into_xylem == pytest.approx(uptake[i], rel=1e-9)


def test_kr_kx(solved):
    g, ds, mecha = solved
    kr, Kx, Kx_phloem = ds.get("kr"), ds.get("Kx"), ds.get("Kx_phloem")
    assert np.isnan(kr[0]) and kr[1] > kr[2] > 0                     # suberin (3) lowers kr below the strip's (1)
    assert Kx[0] == 0. and Kx[1] == Kx[2] > 0                        # same anatomy, same height
    assert (Kx_phloem >= 0).all()


def test_optional(solved):
    """standard_solve = 0: the standard problem is not solved, kr not computed (NaN); Kx still is."""
    g, ds, mecha = solved
    before = ds.get("standard_psi").copy()
    mecha.standard_solve = 0.
    try:
        ds.set("barrier", np.full(ds.n_nodes(), 1.))
        mecha()
        assert np.isnan(ds.get("kr")).all() and (ds.get("Kx") > 0).all()
        np.testing.assert_array_equal(ds.get("standard_psi"), before)
    finally:
        mecha.standard_solve = 1.
        ds.set("barrier", np.array([0., 1., 3.])[ds.owner("SubOrgan")])
        mecha()
