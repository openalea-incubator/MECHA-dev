"""
What develop (945b2be) adds to MECHA, in the port (doc/legacy.rst, the develop branch; devplan §30): the cambium
(cgroup 22) and develop's companion cells (24, 26), barriers 10 and 11 (the sclerenchyma's lignified walls), the
transfusion tracheids as xylem. Each on a root section whose cells are retagged, against the same section untagged.
"""
import numpy as np
import pytest

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.geometry import MEMBRANE, PLASMODESMATA, WALL_PATH

from plants import anatomy_component, anatomy_data_structure, root


def section(barrier=1., retag=None, preset=None, tissues=None, **parameters):
    """A one-segment root's conductances (and its network; GRANAP's *preset*, the monocot root by default), its cells'
    cgroups changed by *retag* ({old: new}) and their tissues by *tissues* ({old: new}) before the first call, the
    component's *parameters* set."""
    g, ds, anatomy = anatomy_data_structure(root(1), anatomy_component(preset))
    cgroup = np.asarray(ds.get("cgroup"), dtype=float).copy()
    for old, new in (retag or {}).items():
        cgroup[cgroup == old] = new
    ds.set("cgroup", cgroup)
    tissue = np.asarray(ds.get("tissue"), dtype=object).copy()
    for old, new in (tissues or {}).items():
        tissue[tissue == old] = new
    ds.set("tissue", tissue)
    ds.register("barrier", np.full(ds.n_nodes(), float(barrier)), location="node")
    mecha = MechaHydraulics(data_structure=ds)
    for name, value in parameters.items():
        setattr(mecha, name, value)
    mecha()
    return np.asarray(ds.get("conductance")).copy(), mecha.network()


def cells_of(net, edges, groups):
    """The edges joining a cell of *groups[0]* to a cell (or wall) next to one of *groups[1]*."""
    a, b = net.cgroup[net.a[edges]], net.cgroup[net.b[edges]]
    return edges[(np.isin(a, groups[0]) & np.isin(b, groups[1])) | (np.isin(b, groups[0]) & np.isin(a, groups[1]))]


@pytest.mark.parametrize("rules", (1., 2.))
def test_the_cambium(rules):
    """The pericycle retagged cambium (22): its plasmodesmata with the endodermis take develop's frequency (1.08e6
    against the endodermis-pericycle's 9.6e5), with the stele the cambium-stele factor (rules 2: 2f / (f + 1)); its
    walls with the endodermis kw_endo_peri, as the pericycle's: the walls' conductances are unchanged."""
    K16, net16 = section(barrier=3., plasmodesmata_rules=rules)
    K22, net22 = section(barrier=3., retag={16: 22}, plasmodesmata_rules=rules, kpl_cambium_stele_factor=0.5)
    pd = np.flatnonzero(net22.in_section & (net22.path == PLASMODESMATA))
    endo = cells_of(net22, pd, ((22,), (3,)))
    stele = cells_of(net22, pd, ((22,), (5,)))
    assert endo.size and stele.size
    np.testing.assert_allclose(K22[endo] / K16[endo], 1.08E6 / 9.6E5, rtol=1e-14)
    np.testing.assert_allclose(K22[stele] / K16[stele], 2 * 0.5 / 1.5 if rules == 2. else 1.0, rtol=1e-14)
    walls = np.flatnonzero(net22.in_section & net22.junction_edge)
    np.testing.assert_array_equal(K22[walls], K16[walls])


def test_develop_companion_cells_in_the_original_rules():
    """With the original rules, cells tagged 26 or 24 (develop's companion cells) are companion cells (the dicot root:
    81 companion cells)."""
    K12, net = section(preset="for_dicot_root", kpl_PCC_factor=0.5)
    assert (net.cgroup == 12).sum() > 0
    for tag in (24, 26):
        K, _ = section(preset="for_dicot_root", retag={12: tag}, kpl_PCC_factor=0.5)
        np.testing.assert_array_equal(K, K12)


def test_the_sclerenchyma_lignified_at_barrier_11():
    """The cortex retagged sclerenchyma: at barrier 11 (barrier 8 and the sclerenchyma's walls lignified), the walls
    between two of its cells take kw_lignin instead of kw (barrier 8's cortex-cortex); without sclerenchyma,
    barrier 11 is barrier 8."""
    K8, net = section(barrier=8., tissues={"cortex": "sclerenchyma"})
    K11, _ = section(barrier=11., tissues={"cortex": "sclerenchyma"})
    walls = np.flatnonzero(net.in_section & net.junction_edge)
    wall = np.where(np.isin(net.a[walls], np.flatnonzero(net.kind == 1)), net.a[walls], net.b[walls])
    cells = net.wall_cells[wall]
    both = (cells >= 0).all(axis=1) & net.sclerenchyma[np.maximum(cells, 0)].all(axis=1)
    assert both.sum() > 0
    mecha = MechaHydraulics
    np.testing.assert_allclose(K11[walls[both]] / K8[walls[both]], mecha.kw_lignin / mecha.kw, rtol=1e-13)
    untouched = ~(cells >= 0).any(axis=1) | ~net.sclerenchyma[np.maximum(cells, 0)].any(axis=1)
    np.testing.assert_array_equal(K11[walls[untouched]], K8[walls[untouched]])
    plain8, _ = section(barrier=8.)
    plain11, _ = section(barrier=11.)
    np.testing.assert_array_equal(plain11, plain8)
