"""The users' entry point (openalea.mecha.simulation): its summary agrees with the component's variables, its tables
cover the section, and its arguments act as the model says."""
import numpy as np
import pytest

from openalea.mecha.simulation import KR_TO_M_PER_S_PER_MPA, MAIN_PARAMETERS, parameters, simulate, simulate_root


@pytest.fixture(scope="module")
def default():
    return simulate()


def test_the_summary_is_the_components(default):
    data_structure = default.data_structure
    summary = default.summary
    assert summary["kr"] == data_structure.get("kr")[0]
    assert summary["kr_SI"] == pytest.approx(summary["kr"] * 1.1574074e-3, rel=1e-6)
    assert KR_TO_M_PER_S_PER_MPA == pytest.approx(1 / 864., rel=1e-12)
    assert summary["uptake"] == data_structure.get("standard_uptake")[0]
    # kr is the uptake per unit of surface (perimeter × section height) and of potential difference
    height = data_structure.get("length")[0] * 100.
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
    data_structure = table.attrs["scene"].populations[0].data_structure
    np.testing.assert_allclose(table["uptake"], data_structure.get("uptake"))


def test_osmotic_potentials():
    """With osmotic (MECHA's presets): the cells' osmotic potentials are the presets' (os_hetero 4, per tissue), and
    a soil solution more concentrated than the cells' takes the uptake down."""
    osmotic = dict(os_hetero=4, os_cortex=-8.3E3, osmotic_epi=-6.8E3, osmotic_endo=-4.8E3, osmotic_peri=-5.5E3,
                   osmotic_stele=-5.5E3, osmotic_left_soil=-1.5E3, osmotic_right_soil=-1.5E3)
    fresh = simulate(osmotic=osmotic, xylem_psi=-2.5E3)
    data_structure = fresh.data_structure
    tissue, potential = np.asarray(data_structure.get("tissue"), dtype=object), np.asarray(data_structure.get("osmotic_potential"))
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
    data_structure = section.data_structure
    cortex = np.asarray(data_structure.get("tissue"), dtype=object) == "cortex"
    assert cortex.sum() > 0 and np.isfinite(np.asarray(data_structure.get("osmotic_potential"))[cortex]).all()


def test_the_run_configuration_is_the_scenes():
    """compute= (metafspm's Compute) is the Scene's run configuration."""
    from openalea.metafspm.compute import Compute
    compute = Compute(threads=1, memory="keep")
    assert simulate(compute=compute).scene.compute is compute


def test_the_plant_summaries_weight_kr_by_the_root_surface(tmp_path):
    """The recorder's per-plant kr is the sections' mean weighted by their root surface (perimeter × length), so that
    kr × the total surface is the sum of the sections' kr × surface; the boundary potentials are weighted as psi."""
    import warnings

    import pandas as pd

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        table = simulate_root(barriers=(3, 1, 1), outputs=str(tmp_path))
    data_structure = table.attrs["scene"].populations[0].data_structure
    kr, surface = np.asarray(data_structure.get("kr")), np.asarray(data_structure.get("radial_surface"))
    length = data_structure.map_to(np.asarray(data_structure.get("length")), "node", "SubOrgan", "mean")
    assert surface == pytest.approx(np.asarray(data_structure.get("perimeter")) * length * 100.)    # cm × m -> cm2
    rows = pd.read_csv(next((tmp_path / "summaries" / "Plant").glob("*/summaries.csv")))
    measured = np.isfinite(kr)
    assert rows["kr"].iloc[-1] == pytest.approx(np.sum((kr * surface)[measured]) / np.sum(surface[measured]),
                                                rel=1e-12)
    sections = np.asarray(data_structure.get("section_length"))
    assert sections == pytest.approx(length * 100.)                                     # m -> cm
    Kx = np.asarray(data_structure.get("Kx"))
    assert rows["Kx"].iloc[-1] == pytest.approx(np.sum(Kx * sections) / np.sum(sections), rel=1e-12)
    conductance = np.asarray(data_structure.get("soil_conductance"))                                 # in parallel: summed
    assert rows["soil_conductance"].iloc[-1] == pytest.approx(conductance.sum(), rel=1e-12)
    assert not [w for w in caught if "without a declared weight" in str(w.message)]


def _every_value(scene) -> dict:
    """Every numeric variable of every population, by population and name."""
    found = {}
    for population in scene.populations:
        data_structure = population.data_structure
        for name in data_structure.available_vars():
            values = np.asarray(data_structure.get(name))
            if values.dtype != object:
                found[f"{population.name} {name}"] = values.copy()
    return found


def _two_populations_at(processes, build, steps) -> dict:
    """*build*(Compute)'s scene run *steps* steps: every value of its populations."""
    from openalea.metafspm.compute import Compute
    scene = build(Compute(threads=1, processes=processes))
    try:
        for _ in range(steps):
            scene.run()
        assert type(scene.runner).__name__ == ("ProcessRunner" if processes > 1 else "SequentialRunner")
        return _every_value(scene)
    finally:
        scene.finish()


def _assert_same_bits(one: dict, two: dict) -> None:
    assert one.keys() == two.keys() and one
    for key, values in one.items():
        np.testing.assert_array_equal(two[key], values, err_msg=key)


def test_two_populations_in_two_processes_give_the_bits_of_one():
    """Two root axes (their collars at -5000 and -3000 hPa) in two populations (planting(population=)):
    Compute(processes=2) runs each in its worker process, with the values of one process."""
    from openalea.metafspm.scene.population import Planting, planting
    from openalea.metafspm.scene.scene import Scene

    from openalea.mecha.simulation import _RootSection, development_parameters

    distances = [2 * 5e-3, 5e-3, 0.]
    scenario = dict(collar_condition=1., collar_psi=-5000., uniform_soil_psi=-1000.,
                    **development_parameters(distances, (3, 1, 0)),
                    root_segments=[[0.55e-3, 5e-3, distance] for distance in distances])

    def build(compute):
        table = planting([Planting(), Planting(x=0.1)], _RootSection, scenario, [{}, {"collar_psi": -3000.}],
                         population=["first", "second"])
        return Scene(table, compute=compute)
    _assert_same_bits(_two_populations_at(1, build, 2), _two_populations_at(2, build, 2))
