"""MECHA's standard problem on each section: ψ, flows, uptake, kr, Kx, by barrier stage (at barrier 0 against the
protophloem, MECHA's stele boundary in the tip zone); optional."""
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
    tip = psi[owner == 0]                              # barrier 0: between the soil's and the protophloem's 1.1e4 hPa
    assert tip.min() >= -1e-6 and tip.max() <= mecha.standard_psi_sieve + 1e-6 and tip.max() > 1e3


def test_mass_balance(solved):
    """Each section's uptake from the soil is the flow into its xylem (the axial boundary), to the round-off of the
    boundary's terms (the vessels' Poiseuille conductances, up to ~10 cm3 hPa-1 d-1, hold them near the xylem's
    potential: their flows are small differences of large terms)."""
    g, ds, mecha = solved
    psi, axial, owner = ds.get("standard_psi"), ds.get("axial_boundary_conductance"), ds.owner("SubOrgan")
    uptake = ds.get("standard_uptake")
    for i in (1, 2):
        into_xylem = (axial * (psi - mecha.standard_psi_xylem))[owner == i].sum()
        scale = (axial * abs(mecha.standard_psi_xylem))[owner == i].sum()
        assert uptake[i] > 0 and abs(into_xylem - uptake[i]) <= 64 * np.finfo(float).eps * scale
    into_phloem = (axial * (psi - mecha.standard_psi_sieve))[owner == 0].sum()   # barrier 0: out of the protophloem
    scale = (axial * abs(mecha.standard_psi_sieve))[owner == 0].sum()
    assert uptake[0] < 0 and abs(into_phloem - uptake[0]) <= 64 * np.finfo(float).eps * scale


def test_kr_kx(solved):
    g, ds, mecha = solved
    kr, Kx, Kx_phloem = ds.get("kr"), ds.get("Kx"), ds.get("Kx_phloem")
    assert kr[0] > kr[1] > kr[2] > 0          # no barrier (0) above the strip's (1), suberin (3) lowers kr below it
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


def test_flow_condition_at_the_xylem(solved):
    """standard_axial_condition 2: the given flow leaves through the xylem; the section's uptake is that flow."""
    g, ds, mecha = solved
    mecha.standard_axial_condition, mecha.standard_axial_flow = 2., 1e-3
    try:
        mecha()
        inflow, owner = ds.get("standard_axial_inflow"), ds.owner("SubOrgan")
        for i in (1, 2):                                  # barriers 1 and 3 (barrier 0: no mature xylem)
            assert inflow[owner == i].sum() == pytest.approx(-1e-3, rel=1e-12)
            assert ds.get("standard_uptake")[i] == pytest.approx(1e-3, rel=1e-9)
        assert np.isfinite(ds.get("kr")[1:]).all()
    finally:
        mecha.standard_axial_condition, mecha.standard_axial_flow = 1., 0.
        mecha()


def test_kr_at_barrier_0_against_the_protophloem(solved):
    """At barrier 0 (the tip zone, the xylem immature), MECHA's kr is the section's against the protophloem, its stele
    boundary: its protosieve cells, or, GRANAP's anatomies typing none, its sieve tubes, at standard_psi_sieve (1.1e4
    hPa, MECHA's BCs.xml). NaN, it has no axial boundary: the soil's potential everywhere, kr undefined. A phloem flow
    (standard_sieve_flow) replaces the potential, split among the protophloem's cells, kr against their mean."""
    g, ds, mecha = solved
    nodes, _, geo = mecha.section_geometries()[int(ds.entity_ids("SubOrgan")[0])]
    assert geo.protosieve_cells.size == 0 and geo.sieve_cells.size > 0
    np.testing.assert_array_equal(geo.protophloem_cells, geo.sieve_cells)
    axial = ds.get("axial_boundary_conductance")[nodes]
    assert (axial[geo.sieve_cells] > 0).all() and np.count_nonzero(axial) == geo.sieve_cells.size
    assert mecha.standard_psi_sieve == 1.1E4
    uptake, height = ds.get("standard_uptake")[0], ds.get("length")[0] * 1e6                    # µm
    assert ds.get("kr")[0] == pytest.approx(uptake / ((0. - 1.1E4) * ds.get("perimeter")[0] * height * 1e-4),
                                            rel=1e-12)
    try:
        mecha.standard_psi_sieve = float("nan")
        mecha()
        assert np.isnan(ds.get("kr")[0]) and np.isfinite(ds.get("kr")[1:]).all()
        np.testing.assert_allclose(ds.get("standard_psi")[nodes], 0., atol=1e-9)
        mecha.standard_psi_sieve, mecha.standard_sieve_flow = 1.1E4, -1e-3         # the phloem unloading 1e-3 cm3 d-1
        mecha()
        inflow = ds.get("standard_axial_inflow")[nodes]
        assert inflow[geo.sieve_cells].sum() == pytest.approx(1e-3, rel=1e-12) and ds.get("axial_boundary_conductance")[
            nodes].sum() == 0.
        assert ds.get("standard_uptake")[0] == pytest.approx(-1e-3, rel=1e-9)
        protophloem = ds.get("standard_psi")[nodes[geo.sieve_cells]].mean()
        assert ds.get("kr")[0] == pytest.approx(-1e-3 / ((0. - protophloem) * ds.get("perimeter")[0] * height * 1e-4),
                                                rel=1e-9)
    finally:
        mecha.standard_psi_sieve, mecha.standard_sieve_flow = 1.1E4, float("nan")
        mecha()


def test_the_standard_problem_is_declared_symmetric_and_checked():
    """@graph_system(symmetric=True): its matrix checked once per pattern (it holds: a conductance network with Robin
    terms), and solved by CHOLMOD with Compute(linear_solver="cholmod") (scikit-sparse, optional), and by PARDISO and
    MUMPS when installed, to SuperLU's potentials. The plant problem is not declared: measured on use case 1, its Jacobian is unsymmetric beyond
    round-off once the leaves transpire (pairs of kept shoot nodes, 0.049 one way and 3.6e-8 the other: the explicit
    terms of the leaves' vapour path), its reduced kinds symmetric to 3e-16."""
    from openalea.metafspm.compute import Compute
    assert MechaHydraulics._graph_system_specs["_standard_problem"]["symmetric"]
    assert not MechaHydraulics._graph_system_specs["_plant_problem"]["symmetric"]
    results = {}
    solvers = ["superlu"]
    try:
        import sksparse.cholmod  # noqa: F401
        solvers.append("cholmod")
    except ImportError:
        pass
    from openalea.metafspm.solve.linear import LINEAR_SOLVERS
    solvers += [name for name in ("pardiso", "mumps") if name in LINEAR_SOLVERS]   # optional: when installed
    for name in solvers:
        g, ds, anatomy = anatomy_data_structure(root(3))
        ds.register("barrier", np.array([0., 1., 3.])[ds.owner("SubOrgan")], location="node")
        with Compute(linear_solver=name).limits() as compute:
            MechaHydraulics(data_structure=ds)()
            checked = [pattern.symmetric_checked for key, patterns in compute.linear()._patterns.items()
                       if key[1] == "_standard_problem" for pattern in patterns]
        assert checked and all(checked)
        results[name] = np.asarray(ds.get("standard_psi"), dtype=float)
    for name in results:
        np.testing.assert_allclose(results[name], results["superlu"], rtol=1e-8, atol=1e-6)
