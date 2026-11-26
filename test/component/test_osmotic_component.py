"""
The osmotic potentials: MECHA's presets as the "Solutes" inputs, the membranes' osmotic
terms in the plant problem and in the standard one, the water potential, and their balance.
"""
import numpy as np
import pytest

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.geometry import MEMBRANE
from openalea.mecha.osmotic import MechaOsmoticPresets

from plants import anatomy_data_structure, root

SOIL = -1000.                                          # hPa
UNIFORM = dict(os_cortex=-500., osmotic_sieve=-500., osmotic_left_soil=-500., osmotic_right_soil=-500.,
               osmotic_xyl=-500., osmotic_endo=-500., osmotic_symmetry_xyl=1.)


def _root(n=3, presets=None, **parameters):
    g, data_structure, anatomy = anatomy_data_structure(root(n, length=5e-3))
    owner = data_structure.owner("SubOrgan")
    data_structure.register("barrier", np.array([1., 3., 1.][:n])[owner], location="node")
    data_structure.register("soil_psi", np.full(data_structure.n_nodes(), SOIL), location="node")
    osmotic = None
    if presets is not None:
        osmotic = MechaOsmoticPresets(data_structure=data_structure)
        for name, value in presets.items():
            setattr(osmotic, name, value)
        osmotic()
    mecha = MechaHydraulics(data_structure=data_structure)
    for name, value in parameters.items():
        setattr(mecha, name, value)
    mecha()
    return g, data_structure, mecha, osmotic


@pytest.fixture(scope="module")
def plain():
    return _root()


@pytest.fixture(scope="module")
def presets():
    return _root(presets={}, standard_osmotic=1.)


def test_no_solutes_no_change(plain):
    """Without a "Solutes" model, π = σ = 0: no drive, and the water potential is the pressure potential."""
    g, data_structure, mecha, _ = plain
    assert (data_structure.get("osmotic_drive") == 0).all() and (data_structure.get("osmotic_potential") == 0).all()
    np.testing.assert_array_equal(data_structure.get("water_potential"), data_structure.get("psi"))


def test_presets_on_the_root_membranes(presets):
    """σ on the membranes only; π on the cells and the walls, none on the junctions."""
    g, data_structure, mecha, _ = presets
    sigma, path, compartment_type = data_structure.get("reflection_coefficient"), data_structure.get("network_path"), data_structure.get("compartment_type")
    assert (sigma[path != MEMBRANE] == 0).all() and (sigma[path == MEMBRANE] > 0).any()
    pi = data_structure.get("osmotic_potential")
    assert (pi[compartment_type == 2] == 0).all() and (pi[compartment_type == 0] < 0).all()


def test_plant_balance_with_osmotic_terms(presets):
    """Every node's balance with the drives: the flows (K (Δψ + σ Δπ)) leaving each node equal what its boundaries
    bring in; the water potential is ψ + π."""
    g, data_structure, mecha, _ = presets
    B = data_structure.to_graph_view().incidence
    out = np.asarray(B @ data_structure.get("flow")).reshape(-1)
    psi = data_structure.get("psi")
    inflow = data_structure.get("root_uptake") - data_structure.get("evaporation")          # from the soil, to the air
    collar = data_structure.get("collar_node_condition") > 0
    scale = np.abs(data_structure.get("flow")).max()
    np.testing.assert_allclose((out - inflow)[~collar], 0., atol=1e-9 * scale)
    np.testing.assert_array_equal(data_structure.get("water_potential"), psi + data_structure.get("osmotic_potential"))


def test_uniform_potentials_cancel():
    """π equal on both sides of every membrane: no drive, the same solution as without solutes."""
    g, data_structure, mecha, osmotic = _root(presets=UNIFORM, standard_osmotic=1.)
    drive = data_structure.get("osmotic_drive")
    assert np.abs(drive).max() < 1e-12                       # the soil's left-right weights: rounding only
    with_osmotic = data_structure.get("psi").copy(), data_structure.get("standard_psi").copy()
    g2, ds2, mecha2, _ = _root()
    np.testing.assert_allclose(with_osmotic[0], ds2.get("psi"), rtol=0, atol=1e-9)
    np.testing.assert_allclose(with_osmotic[1], ds2.get("standard_psi"), rtol=0, atol=1e-9)


def test_osmotic_uptake_against_the_cells(presets, plain):
    """Cells more negative than the soil solution draw water in: the root takes up more with their osmotic terms
    (σ = 1, cortex −4800 hPa against a soil solution of −500 hPa), in the standard problem and in the plant."""
    _, data_structure, _, _ = presets
    _, ds0, _, _ = plain
    assert data_structure.get("root_uptake").sum() > ds0.get("root_uptake").sum()
    assert (data_structure.get("standard_uptake") > ds0.get("standard_uptake")).all()
