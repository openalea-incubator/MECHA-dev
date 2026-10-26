"""The users' entry point (openalea.mecha.simulation): its summary agrees with the component's variables, its tables
cover the section, and its arguments act as the model says."""
import numpy as np
import pytest

from openalea.mecha.simulation import KR_TO_M_PER_S_PER_MPA, MAIN_PARAMETERS, parameters, simulate, simulate_root


@pytest.fixture(scope="module")
def default():
    return simulate()


def test_the_summary_is_the_components(default):
    ds = default.data_structure
    summary = default.summary
    assert summary["kr"] == ds.get("kr")[0]
    assert summary["kr_SI"] == pytest.approx(summary["kr"] * 1.1574074e-3, rel=1e-6)
    assert KR_TO_M_PER_S_PER_MPA == pytest.approx(1 / 864., rel=1e-12)
    assert summary["uptake"] == ds.get("standard_uptake")[0]
    # kr is the uptake per unit of surface (perimeter × section height) and of potential difference
    height = ds.get("length")[0] * 100.
    assert summary["kr"] == pytest.approx(summary["uptake"] / (summary["perimeter"] * height * 5000.), rel=1e-6)
    shares = summary[["cortex_wall_share", "cortex_membrane_share", "cortex_plasmodesmata_share"]]
    assert shares.sum() == pytest.approx(1.) and (shares >= 0).all()


def test_tissues_and_pathways(default):
    tissues = default.tissues()
    assert tissues["cells"].sum() == default.summary["cells"]
    assert tissues["potential"].between(-5000. - 1e-6, 0. + 1e-6).all()
    assert tissues.index[0] == "epidermis" and "endodermis" in tissues.index
    pathways = default.pathways()
    np.testing.assert_allclose(pathways[["wall", "membrane", "plasmodesmata"]].sum(axis=1), 1., rtol=1e-12)
    assert pathways["radius"].is_monotonic_increasing


def test_parameters_table():
    table = parameters({"kw": 1e-4})
    assert tuple(table["name"][:len(MAIN_PARAMETERS)]) == MAIN_PARAMETERS
    assert table.loc[table["name"] == "kw", "value"].item() == 1e-4
    assert len(parameters(main_only=True)) == len(MAIN_PARAMETERS)


def test_the_arguments_act(default):
    assert simulate({"kw": 2.4e-5}).summary["kr"] < default.summary["kr"]
    assert simulate(barrier=3).summary["kr"] < default.summary["kr"]
    dicot = simulate(anatomy={"preset": "for_dicot_root"})
    assert dicot.summary["perimeter"] > default.summary["perimeter"]
    # the standard problem is linear: kr does not depend on the potentials, the uptake does
    drier = simulate(soil_psi=-1000.)
    assert drier.summary["kr"] == pytest.approx(default.summary["kr"], rel=1e-9)
    assert drier.summary["uptake"] == pytest.approx(default.summary["uptake"] * 4000. / 5000., rel=1e-9)


def test_draw(default):
    import matplotlib
    matplotlib.use("Agg")
    for what in ("potential", "tissues", "pathways"):
        assert default.draw(what).has_data()
    with pytest.raises(ValueError):
        default.draw("other")


def test_a_root_axis():
    table = simulate_root(barriers=(3, 1, 0), soil_psi=-1000., collar_psi=-5000.)
    assert list(table["barrier"]) == [3, 1, 0] and list(table["distance"]) == [2.5, 7.5, 12.5]
    assert table["xylem_potential"].iloc[0] == pytest.approx(-5000.)
    assert table["uptake"].iloc[1] > table["uptake"].iloc[0] > 0           # the Casparian strip alone lets more in
    assert abs(table["uptake"].iloc[2]) < 1e-12                             # the tip's xylem does not conduct yet
    ds = table.attrs["scene"].populations[0].data_structure
    np.testing.assert_allclose(table["uptake"], ds.get("uptake"))
