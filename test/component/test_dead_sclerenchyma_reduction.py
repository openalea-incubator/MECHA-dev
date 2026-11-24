"""
The dead sclerenchyma (PR #67's dead fibres at barrier 11, doc/legacy.rst d17) in a plant reduction: each section
draws its own dead cells, so a section whose draw acts (barrier 11, sclerenchyma in its anatomy) is a kind of its own
(dead_sclerenchyma_draw, one of plant_reduction's archetype variables), its kind evaluated with its own draw: a reduced
root system takes up, section by section, the water of the unreduced one. GRANAP's monocot root with a sclerenchyma
layer under its exodermis (its preset has none).
"""
import numpy as np
import pytest

from openalea.granap.input_data import ExodermisParams, OrganInputData, SclerenchymaParams
from openalea.granap.metafspm.component import GranapAnatomy
from openalea.granap.metafspm.library import AnatomyLibrary, default_root_parameters
from openalea.mecha.simulation import root_system


MAIN = (11, 11, 11)                                  # three segments of one length, at barrier 11: one kind without
                                                     # their draws


def _root_with_fibres() -> OrganInputData:
    """GRANAP's monocot root preset with one layer of sclerenchyma under its exodermis."""
    params = list(OrganInputData.for_root().params)
    after = next(k for k, p in enumerate(params) if isinstance(p, ExodermisParams))
    params.insert(after + 1, SclerenchymaParams(n_layers=1))
    return OrganInputData(params=params)


@pytest.fixture(scope="module")
def root_systems():
    fibre_root = _root_with_fibres()

    def parameters(key, base=None, overrides=None):
        return default_root_parameters(key, base=fibre_root if base is None else base, overrides=overrides)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(GranapAnatomy, "library", AnatomyLibrary(parameters=parameters))
        full = root_system(main=MAIN, laterals=(), instanced=True, soil_boundary="node_dirichlet")
        patch.setattr(GranapAnatomy, "library", AnatomyLibrary(parameters=parameters))
        reduced = root_system(main=MAIN, laterals=(), condensed=True, length_classes="exact")
    return full, reduced


def test_each_section_draws_its_own_dead_fibres(root_systems):
    """The sections, of one template, have sclerenchyma cells and different dead ones; each is its own draw."""
    full, _ = root_systems
    data_structure = full.data_structure
    tissue = np.asarray(data_structure.get("tissue"), dtype=object)
    owner = np.asarray(data_structure.owner("SubOrgan"))
    fibres = tissue == "sclerenchyma"
    segments = data_structure.entity_ids("SubOrgan")
    assert fibres.sum() > 0 and set(np.unique(owner[fibres]).tolist()) == set(range(segments.size))
    np.testing.assert_array_equal(data_structure.get("dead_sclerenchyma_draw"), segments)
    living = full.scene.populations[0].instance.hydraulics.network().living_cell
    patterns = {tuple(living[fibres & (owner == i)].tolist()) for i in range(segments.size)}
    assert len(patterns) == segments.size


def test_a_reduced_root_system_with_dead_fibres_takes_up_the_unreduced_ones_water(root_systems):
    """Reduced (each section onto its conduits and soil node, its kind per exact length), the sections' uptakes,
    standard uptake fractions and kr are the unreduced ones', and so is the root system's conductance: each section's
    kind holds its own dead fibres. kr, each kind's own standard problem, bit for bit; the plant problem to the
    reduction's round-off (2026-11-24: uptake 6.6e-10, SUF 1.8e-9, Krs 3.6e-9 relative; the sections sharing one kind,
    their draws ignored: 5.0e-3, 4.0e-3, 1.1e-3, kr 5.0e-3)."""
    full, reduced = root_systems
    assert reduced.data_structure.n_nodes() < full.data_structure.n_nodes() / 10
    assert len(reduced.data_structure.reduction.reduced_groups(reduced.data_structure)) == len(MAIN)
    np.testing.assert_array_equal(reduced.sections["kr"], full.sections["kr"])
    for name in ("uptake", "suf"):
        np.testing.assert_allclose(reduced.sections[name], full.sections[name], rtol=1e-8)
    assert reduced.conductance == pytest.approx(full.conductance, rel=1e-8)
