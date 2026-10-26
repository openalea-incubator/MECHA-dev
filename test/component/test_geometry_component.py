"""MECHA's component on GRANAP's anatomies: its geometry written on the DataStructure, once per anatomy."""
import numpy as np
import pytest

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.geometry import section_geometry, section_of, section_parts

from plants import anatomy_data_structure, root


@pytest.fixture(scope="module")
def three_segments():
    g, ds, anatomy = anatomy_data_structure(root(3))
    mecha = MechaHydraulics(data_structure=ds)
    mecha()
    return g, ds, mecha


def test_ranks_and_perimeters_written(three_segments):
    g, ds, mecha = three_segments
    rank, kind = ds.get("cell_rank"), ds.get("kind")
    assert (rank[kind == 0] >= 1).all() and (rank[kind != 0] == -1).all()
    perimeter = ds.get("perimeter")
    assert perimeter.shape == (3,) and np.isfinite(perimeter).all() and (perimeter > 0).all()


def test_one_geometry_per_template(three_segments):
    g, ds, mecha = three_segments
    templates = set(np.asarray(ds.get("anatomy_template")).tolist())
    assert len(mecha.kept("geometries")) == len(templates)


def test_shared_geometry_is_each_sections_own(three_segments):
    g, ds, mecha = three_segments
    for segment, (nodes, edges, geometry) in mecha.section_geometries().items():
        own = section_geometry(section_of(ds, nodes, edges))
        np.testing.assert_array_equal(own.cell_rank, geometry.cell_rank)
        assert own.perimeter == geometry.perimeter
