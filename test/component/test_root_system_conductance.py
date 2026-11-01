"""
A root system's conductance (Krs) and standard uptake fractions (SUF), upscaled from the anatomical network
(RootSystemConductance), and their calibration (simulation.fit_root_system): MARSHAL's definitions checked on the
plant problem itself.
"""
import numpy as np
import pandas as pd
import pytest

from openalea.mecha.root_system import NETWORK, RootSystemConductance
from openalea.mecha.simulation import fit_root_system, root_system

SOIL, COLLAR = -1000., -5000.


@pytest.fixture(scope="module")
def seedling():
    return root_system(main=(3, 1, 1), laterals=((1, (1,)),), soil_psi=SOIL, collar_psi=COLLAR)


def test_krs_is_the_plant_problems_uptake_per_hpa_under_a_uniform_soil(seedling):
    """Without osmotic potentials, the plant problem is linear: its uptake under a uniform soil is Krs Δψ, and each
    section's share of it is its SUF."""
    uptake = seedling.sections["uptake"].to_numpy()
    assert seedling.conductance == pytest.approx(uptake.sum() / (SOIL - COLLAR), rel=1e-10)
    np.testing.assert_allclose(seedling.sections["suf"], uptake / uptake.sum(), rtol=1e-10)
    assert seedling.sections["suf"].sum() == pytest.approx(1.)
    suberized, young = seedling.sections.loc[0, "suf"], seedling.sections.loc[1, "suf"]
    assert suberized < young / 10                                        # the barriers decide where water enters


def test_one_section_alone_has_its_kr_times_its_surface():
    """A root of one section: Krs is its kr (MECHA's standard problem) times its surface, within 0.5 % (the standard
    problem has its own boundary terms: 0.12 % here)."""
    alone = root_system(main=(1,), laterals=())
    ds = alone.data_structure
    surface = float(np.asarray(ds.get("perimeter"))[0]) * 0.5                  # cm: 5 mm long
    assert alone.conductance == pytest.approx(float(alone.sections.loc[0, "kr"]) * surface, rel=5e-3)


def test_the_upscaling_is_declared():
    table = RootSystemConductance.declarations().set_index("name")
    declared = table.loc["root_system_conductance", "upscaling"]
    assert declared.law == "network" and declared.of == NETWORK and "uniform" in declared.protocol
    assert table.loc["standard_uptake_fraction", "upscaling"].of == NETWORK
    assert table.loc["conductance", "role"] == "input"


def test_the_inhibited_experiment_lowers_krs(seedling):
    inhibited = root_system(main=(3, 1, 1), laterals=((1, (1,)),), experiment="inhibited")
    assert inhibited.conductance < seedling.conductance
    assert inhibited.parameters["kaqp"] == pytest.approx(0.1 * 0.00043)
    with pytest.raises(ValueError, match="experiment"):
        root_system(experiment="bogus")


def test_a_parameter_is_recovered_from_a_measured_krs():
    truth = 0.00043 * 0.5
    measured = root_system(main=(1, 1), laterals=(), parameters={"kaqp": truth}).conductance
    observations = pd.DataFrame({"variable": ["root_system_conductance"], "value": [measured]})
    fit = fit_root_system(observations, parameters=("kaqp",), main=(1, 1), laterals=())
    assert fit.values["kaqp"] == pytest.approx(truth, rel=1e-5)
    assert abs(fit.compared.loc[0, "residual"]) < 1e-6
