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


def test_osmotic_potentials():
    """With osmotic (MECHA's presets): the cells' osmotic potentials are the presets' (os_hetero 4, per tissue), and
    a soil solution more concentrated than the cells' takes the uptake down."""
    osmotic = dict(os_hetero=4, os_cortex=-8.3E3, osmotic_epi=-6.8E3, osmotic_endo=-4.8E3, osmotic_peri=-5.5E3,
                   osmotic_stele=-5.5E3, osmotic_left_soil=-1.5E3, osmotic_right_soil=-1.5E3)
    fresh = simulate(osmotic=osmotic, xylem_psi=-2.5E3)
    ds = fresh.data_structure
    tissue, potential = np.asarray(ds.get("tissue"), dtype=object), np.asarray(ds.get("osmotic_potential"))
    assert (potential[tissue == "epidermis"] == -6.8E3).all() and (potential[tissue == "stele"] == -5.5E3).all()
    salty = simulate(osmotic=dict(osmotic, osmotic_left_soil=-7.5E3, osmotic_right_soil=-7.5E3), xylem_psi=-2.5E3)
    assert salty.summary["uptake"] < fresh.summary["uptake"]


def test_an_incomplete_outer_cortex_layer():
    """A cortex rank without a layer row (an incomplete outermost layer, with no layer after it: the tomato root's rank
    49) takes its inner neighbour's row (doc/legacy.rst, ME-38): every cortex cell gets its preset."""
    from openalea.mecha.osmotic import legacy_rows
    from openalea.mecha.simulation import TOMATO_ROOT
    section = simulate(anatomy=TOMATO_ROOT, osmotic=dict(os_hetero=4, os_cortex=-8.3E3))
    geo = list(section.hydraulics.section_geometries().values())[0][2]
    rank_to_row, _, _ = legacy_rows(geo.n_layer, geo.layer_dist)
    assert any(geo.n_layer[r] > 0 and np.isnan(rank_to_row[r]) for r in range(40, 50))
    ds = section.data_structure
    cortex = np.asarray(ds.get("tissue"), dtype=object) == "cortex"
    assert cortex.sum() > 0 and np.isfinite(np.asarray(ds.get("osmotic_potential"))[cortex]).all()
