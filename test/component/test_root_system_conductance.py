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


def test_instanced_anatomies_give_the_same_root_system(seedling):
    """Each segment's anatomy an instance of its class's template (GRANAP's instanced_anatomy): the same network,
    potentials, Krs and SUF, bit for bit, without the anatomies' MTG vertices."""
    instanced = root_system(main=(3, 1, 1), laterals=((1, (1,)),), soil_psi=SOIL, collar_psi=COLLAR, instanced=True)
    a, b = seedling.data_structure, instanced.data_structure
    assert instanced.conductance == seedling.conductance
    assert a.entity_ids("edge").tolist() == b.entity_ids("edge").tolist()
    for name in ("psi", "standard_uptake_fraction"):
        np.testing.assert_array_equal(a.get(name), b.get(name))
    assert len(b.structure) < len(a.structure) / 100


def test_one_soil_node_per_section_held_at_the_soil_is_the_per_node_soil(seedling):
    """soil_boundary node_dirichlet (GRANAP's soil_node: one soil node per root section, joined to its contact walls
    and junctions through their soil conductances): on a uniform soil, the same potentials, uptakes and Krs as the
    per-node soil terms."""
    held = root_system(main=(3, 1, 1), laterals=((1, (1,)),), soil_psi=SOIL, collar_psi=COLLAR,
                       soil_boundary="node_dirichlet")
    a, b = seedling.data_structure, held.data_structure
    anatomy = np.asarray(b.get("kind")) != 4
    np.testing.assert_allclose(b.get("psi")[anatomy], a.get("psi"), rtol=1e-11)
    assert held.conductance == pytest.approx(seedling.conductance, rel=1e-10)   # its soil nodes held: equivalent
    uptake = [ds.map_to(np.asarray(ds.get("root_uptake")), "node", "SubOrgan", "sum") for ds in (a, b)]
    np.testing.assert_allclose(uptake[1], uptake[0], rtol=1e-10)
    np.testing.assert_allclose(held.sections["uptake"], seedling.sections["uptake"], rtol=1e-10)


def test_a_soil_node_through_its_bulk_conductance_tends_to_the_held_soil():
    """node_robin: the section's soil node joined to the soil through one bulk conductance; the uptake grows to the
    held soil's as that conductance grows, and the condensed solve gives the same potentials."""
    kwargs = dict(main=(3, 1, 1), laterals=((1, (1,)),), soil_psi=SOIL, collar_psi=COLLAR, instanced=True)
    held = root_system(soil_boundary="node_dirichlet", **kwargs)
    total = float(np.sum(held.data_structure.get("root_uptake")))
    shares = []
    for bulk in (1e-6, 1e-3, 1.):
        robin = root_system(soil_boundary="node_robin", soil_bulk_conductance=bulk, **kwargs)
        shares.append(float(np.sum(robin.data_structure.get("root_uptake"))) / total)
    assert shares[0] < shares[1] < shares[2] < 1. and shares[2] > 0.999
    condensed = root_system(soil_boundary="node_robin", soil_bulk_conductance=1e-3, plant_solver="condensed", **kwargs)
    full = root_system(soil_boundary="node_robin", soil_bulk_conductance=1e-3, **kwargs)
    np.testing.assert_allclose(condensed.data_structure.get("psi"), full.data_structure.get("psi"), rtol=1e-10)


def test_the_condensed_solves_give_the_same_sections():
    """plant_solver condensed (the plant problem, the standard problem and Krs, each section's interior eliminated):
    MECHA's kr, the standard potentials, Krs and SUF as the full solves', to round-off."""
    kwargs = dict(main=(3, 3, 1, 1), laterals=((1, (1, 1)),), soil_psi=SOIL, collar_psi=COLLAR, instanced=True)
    full, condensed = root_system(**kwargs), root_system(plant_solver="condensed", **kwargs)
    for name in ("standard_psi", "psi"):
        np.testing.assert_allclose(condensed.data_structure.get(name), full.data_structure.get(name), rtol=1e-10)
    for column in ("kr", "suf", "uptake"):
        np.testing.assert_allclose(condensed.sections[column], full.sections[column], rtol=1e-9)
    assert condensed.conductance == pytest.approx(full.conductance, rel=1e-10)


def test_a_condensed_root_system_is_the_full_one_on_its_conduits_and_soil_nodes():
    """condensed (GRANAP's condensed_anatomy): each section its conduits and soil node, joined by the exact reduction
    of its full network, one per kind (template, barrier, length, parameters): the same potentials on those nodes,
    uptakes, kr, SUF and Krs as the full anatomies'; a section's cells reconstructed on request as the full ones."""
    kwargs = dict(main=(3, 3, 1, 1, 0), laterals=((1, (1, 0)),), soil_psi=SOIL, collar_psi=COLLAR,
                  soil_boundary="node_dirichlet")
    full, small = root_system(instanced=True, **kwargs), root_system(condensed=True, **kwargs)
    a, b = full.data_structure, small.data_structure
    assert b.n_nodes() < a.n_nodes() / 100

    def places(ds):
        node = ds.structure.values_of("template_node", ds.entity_ids("node"))
        return [(int(o), int(node[int(v)])) for o, v in zip(ds.owner("SubOrgan"), ds.entity_ids("node"))]
    row = {place: i for i, place in enumerate(places(a))}
    rows = [row[place] for place in places(b)]
    np.testing.assert_allclose(b.get("psi"), a.get("psi")[rows], rtol=1e-11)
    total = float(full.sections["uptake"].sum())
    np.testing.assert_allclose(small.sections["uptake"], full.sections["uptake"], rtol=1e-9, atol=1e-12 * total)
    np.testing.assert_allclose(small.sections["suf"], full.sections["suf"], rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(small.sections["kr"], full.sections["kr"], rtol=1e-12)
    for name in ("Kx", "Kx_phloem"):                     # from the conduits alone: every one of them is kept
        np.testing.assert_allclose(b.get(name), a.get(name), rtol=1e-12)
    assert small.conductance == pytest.approx(full.conductance, rel=1e-10)
    hydraulics = small.scene.populations[0].instance.hydraulics
    cells, psi = hydraulics.section_cells(2)
    owner = np.asarray(a.owner("SubOrgan"))
    by_node = {node: i for i, (o, node) in enumerate(places(a)) if o == 2}
    np.testing.assert_allclose(psi, a.get("psi")[[by_node[int(c)] for c in cells]], rtol=1e-10)


def test_condensed_osmotic_terms_are_the_full_anatomys():
    """condensed_osmotic presets: each kind's cells' osmotic potentials and reflection coefficients from MECHA's
    presets, changed per segment by its conduits' potential (conduit_osmotic_potential) and its hidden tissues'
    ({tissue: π}, tissue_osmotic_potential): the same potentials and uptakes as the full anatomy given the same."""
    kwargs = dict(main=(3, 1, 1, 0), laterals=((1, (1, 0)),), soil_psi=SOIL, collar_psi=COLLAR,
                  soil_boundary="node_dirichlet", osmotic={"os_hetero": 1.})
    full, small = root_system(instanced=True, **kwargs), root_system(condensed=True, **kwargs)
    a, b = full.data_structure, small.data_structure

    def places(ds):
        node = ds.structure.values_of("template_node", ds.entity_ids("node"))
        return [(int(o), int(node[int(v)])) for o, v in zip(ds.owner("SubOrgan"), ds.entity_ids("node"))]
    row = {place: i for i, place in enumerate(places(a))}
    rows = np.array([row[place] for place in places(b)])

    def same():
        np.testing.assert_allclose(b.get("psi"), a.get("psi")[rows], rtol=1e-11)
        total = float(full.sections["uptake"].abs().sum())
        np.testing.assert_allclose(small.sections["uptake"], full.sections["uptake"], rtol=1e-9, atol=1e-12 * total)
    same()
    before = b.get("psi").copy()
    section, owner_a, owner_b = 1, np.asarray(a.owner("SubOrgan")), np.asarray(b.owner("SubOrgan"))
    pi = np.array(a.get("osmotic_potential"), dtype=float)                 # the full anatomy given them directly
    conduits_b = np.flatnonzero((owner_b == section) & (np.asarray(b.get("kind")) != 4))
    pi[rows[conduits_b]] = -9000.
    pi[(owner_a == section) & (np.asarray(a.get("tissue"), dtype=object) == "cortex")] = -7000.
    a.set("osmotic_potential", pi)
    full.scene.populations[0].instance.hydraulics()
    conduit = np.full(b.n_nodes(), np.nan)                                 # the condensed one through its inputs
    conduit[conduits_b] = -9000.
    b.set("conduit_osmotic_potential", conduit)
    tissues = np.empty(len(b.entity_ids("SubOrgan")), dtype=object)
    tissues[section] = {"cortex": -7000.}
    b.set("tissue_osmotic_potential", tissues)
    small.scene.populations[0].instance.hydraulics()
    same()
    assert np.abs(b.get("psi") - before).max() > 100.                     # the change matters
