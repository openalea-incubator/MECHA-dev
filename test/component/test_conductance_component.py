"""MECHA's conductances on GRANAP's anatomies: per edge and boundary node, by segment barrier and plant parameters."""
import numpy as np
import pytest

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.geometry import MEMBRANE, PLASMODESMATA, WALL, section_parts
from openalea.mecha.postprocess import VISCOSITY_XYLEM, poiseuille

from plants import anatomy_data_structure, root


@pytest.fixture(scope="module")
def coupled():
    g, data_structure, anatomy = anatomy_data_structure(root(3))
    owner = data_structure.owner("SubOrgan")
    data_structure.register("barrier", np.array([0., 1., 3.])[owner], location="node")      # one barrier stage per segment
    mecha = MechaHydraulics(data_structure=data_structure)
    mecha()
    return g, data_structure, mecha


def test_every_section_edge_has_a_conductance(coupled):
    g, data_structure, mecha = coupled
    K, path = data_structure.get("conductance"), data_structure.get("network_path")
    inner = np.zeros(K.size, dtype=bool)
    for nodes, edges in section_parts(data_structure).values():
        inner[edges] = True
    assert np.isfinite(K).all() and (K >= 0).all()
    assert (K[inner & (path == MEMBRANE)] > 0).all()
    assert (K[~inner] == 0).all()                                   # the axial junctions: not MECHA's (step 5)


def test_barriers_close_the_endodermis(coupled):
    """The endodermal walls' conductance: kw without a barrier, the Casparian strip's (1e-16 × ...) with one."""
    g, data_structure, mecha = coupled
    net = mecha.network()
    K = data_structure.get("conductance")
    wall_end = np.where(net.compartment_type[np.maximum(net.a, 0)] == WALL, net.a, net.b)
    cells = net.wall_cells[np.maximum(wall_end, 0)]
    endo_endo = net.junction_edge & (cells >= 0).all(axis=1) & (net.cgroup[np.maximum(cells, 0)] == 3).all(axis=1)
    by_section = [K[endo_endo & (net.section_of_edge == i)] for i in range(3)]
    assert by_section[0].min() > 1e-12 and by_section[1].max() < 1e-15 and by_section[2].max() < 1e-15


def test_boundary_terms(coupled):
    g, data_structure, mecha = coupled
    mecha.standard_axial_source = 2.                                  # the prescribed k_xyl (the legacy's)
    try:
        mecha()
        axial = data_structure.get("axial_boundary_conductance")
        xylem = np.isin(data_structure.get("cgroup"), (13, 19, 20)) & (data_structure.owner("SubOrgan") > 0)
        assert (axial[xylem] == mecha.k_xyl).all()
    finally:
        mecha.standard_axial_source = 1.
        mecha()
    soil, axial = data_structure.get("soil_conductance"), data_structure.get("axial_boundary_conductance")
    compartment_type, cgroup = data_structure.get("compartment_type"), data_structure.get("cgroup")
    assert (soil[compartment_type == 0] == 0).all() and (soil > 0).sum() > 0
    owner = data_structure.owner("SubOrgan")
    xylem = np.isin(cgroup, (13, 19, 20))
    # the vessels' Poiseuille conductance over the height (standard_axial_source 1, the original MECHA's)
    area = np.asarray(data_structure.get("cell_area")) * 1e6                                   # mm2 -> µm2
    height = np.asarray(data_structure.get("length")) * 1e6
    expected = poiseuille(area, height, VISCOSITY_XYLEM)
    vessels = (owner > 0) & xylem
    np.testing.assert_allclose(axial[vessels], expected[vessels], rtol=1e-12)
    assert (axial[(owner > 0) & ~xylem] == 0).all()


def test_plant_parameters(coupled):
    """kw doubled: the walls' conductances (outside the barriers) double, the plasmodesmata's do not change."""
    g, data_structure, mecha = coupled
    K = data_structure.get("conductance").copy()
    path = data_structure.get("network_path")
    kw = mecha.kw
    mecha.kw = 2 * kw
    try:
        mecha()
        K2 = data_structure.get("conductance")
        net = mecha.network()
        walls = net.junction_edge & (net.section_of_edge == 0)
        np.testing.assert_allclose(K2[walls], 2 * K[walls], rtol=1e-15)
        np.testing.assert_array_equal(K2[path == PLASMODESMATA], K[path == PLASMODESMATA])
    finally:
        mecha.kw = kw
        mecha()
