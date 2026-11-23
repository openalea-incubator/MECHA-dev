"""
What PR #67 (fix/GranapAPI, 7c4ed4f) adds to MECHA, in the port (doc/legacy.rst, d12 on; devplan §31), where the
comparisons with the legacy do not reach: the walls lining air spaces and their check, a needle's protected air
spaces left out; the suberin of passage cells on their membranes (GRANAP's sections have none); the dead
sclerenchyma's draw and the smaller of two cells' plasmodesmata factors (E4); a cell without aquaporins keeping kmb (U5).
"""
import numpy as np
import pytest

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.conductance import air_space_lining_walls, root_air_spaces
from openalea.mecha.geometry import MEMBRANE, PLASMODESMATA, WALL

from plants import anatomy_data_structure, needle, needle_anatomy_component, root


def section(barrier=1., architecture=None, anatomy=None, passage=False, sclerenchyma=False, living_cell=None,
            inputs=None, **parameters):
    """A one-segment root's (or *architecture*'s, its *anatomy*'s) conductances and network, the component's
    *parameters* set; with *passage*, every other endodermal cell a passage cell (its tissue); with *sclerenchyma*, the
    cortex sclerenchyma; *living_cell*: the input's value on every node; *inputs*: {node input: a function of the
    tissues giving its values}."""
    arguments = (architecture or root(1),) + ((anatomy,) if anatomy is not None else ())
    g, data_structure, _ = anatomy_data_structure(*arguments)
    tissue = np.asarray(data_structure.get("tissue"), dtype=object).copy()
    if passage:
        tissue[np.flatnonzero(tissue == "endodermis")[::2]] = "passage"
    if sclerenchyma:
        tissue[tissue == "cortex"] = "sclerenchyma"
    data_structure.set("tissue", tissue)
    for name, values in (inputs or {}).items():
        data_structure.register(name, np.asarray(values(tissue), dtype=float), location="node")
    if living_cell is not None:
        data_structure.register("living_cell", np.full(data_structure.n_nodes(), float(living_cell)), location="node")
    data_structure.register("barrier", np.full(data_structure.n_nodes(), float(barrier)), location="node")
    mecha = MechaHydraulics(data_structure=data_structure)
    for name, value in parameters.items():
        setattr(mecha, name, value)
    mecha()
    return np.asarray(data_structure.get("conductance")).copy(), mecha.network(), mecha


def test_walls_lining_an_air_space():
    """At barrier 1, a wall with one air space among its cells is a hydrated half (kw) and a desiccated half
    (kw_septa) in parallel: its conductance follows kw + kw_septa; every other wall keeps its own; at barrier 0
    none is halved."""
    K, net, mecha = section(kw_septa=1.0E-4)
    K_drier, _, _ = section(kw_septa=0.5E-4)
    lining = air_space_lining_walls(net, mecha.parameter_table(mecha.PARAMETERS, "edge"), np.ones(net.path.size))
    assert lining.sum() > 0
    kw = mecha.kw
    np.testing.assert_allclose(K_drier[lining] / K[lining], (kw + 0.5E-4) / (kw + 1.0E-4), rtol=1e-14)
    walls = net.in_section & net.junction_edge
    septa = walls & ~lining & (K != K_drier)
    cells = net.wall_cells[np.where(net.compartment_type[net.a] == 1, net.a, net.b)[septa]]
    assert net.intercellular[np.maximum(cells, 0)].all()          # only the septa between two air spaces differ
    K0, _, _ = section(barrier=0., kw_septa=1.0E-4)
    K0_drier, _, _ = section(barrier=0., kw_septa=0.5E-4)
    np.testing.assert_array_equal(K0[walls], K0_drier[walls])


def test_kw_septa_below_kw():
    """A desiccated wall conducts less than a hydrated one: kw_septa >= kw raises before the conductances, naming the
    sections whose walls line air spaces."""
    with pytest.raises(ValueError, match="kw_septa"):
        section(kw_septa=3.0E-4)
    section(barrier=0., kw_septa=3.0E-4)                           # no wall lining an air space at barrier 0


def test_a_needles_protected_air_spaces_are_not_halved():
    """A needle's mesophyll air spaces (GRANAP's protect_topology, joined to their walls by wall_air edges) keep the
    vapour path: none of its walls is a half wall."""
    _, net, mecha = section(architecture=needle(), anatomy=needle_anatomy_component(), kw=2.5634e-3)
    assert net.intercellular.any()
    assert not root_air_spaces(net)[net.intercellular].any()
    P = mecha.parameter_table(mecha.PARAMETERS, "edge")
    assert not air_space_lining_walls(net, P, np.ones(net.path.size)).any()


def test_suberin_on_the_endodermis_membranes_and_passage_cells():
    """Barrier 2 (suberin with passage cells) against barrier 1, every other endodermal cell a passage cell: the walls
    are the same (suberin is on the membranes); an endodermal cell's membrane toward the cortex is cut by
    kw_suberin_out, except on a wall with a passage cell (kw_passage: kw at barrier 2); the cortex cells' membranes
    keep kw."""
    K1, net, _ = section(barrier=1., passage=True)
    K2, _, _ = section(barrier=2., passage=True)
    assert net.passage.any()
    walls = net.in_section & net.junction_edge
    np.testing.assert_array_equal(K2[walls], K1[walls])
    membranes = np.flatnonzero(net.in_section & (net.path == MEMBRANE))
    wall_is_a = net.compartment_type[net.a[membranes]] == WALL
    wall = np.where(wall_is_a, net.a[membranes], net.b[membranes])
    owner = np.where(wall_is_a, net.b[membranes], net.a[membranes])
    cells = net.wall_cells[wall]
    has = cells >= 0
    toward_cortex = (has & (net.cgroup[np.maximum(cells, 0)] == 4)).any(axis=1) & \
        (has & (net.cgroup[np.maximum(cells, 0)] == 3)).any(axis=1)
    with_passage = (has & net.passage[np.maximum(cells, 0)]).any(axis=1)
    endodermal = net.cgroup[owner] == 3
    open_ = membranes[toward_cortex & endodermal & with_passage]
    cut = membranes[toward_cortex & endodermal & ~with_passage]
    cortical = membranes[toward_cortex & (net.cgroup[owner] == 4)]
    assert open_.size and cut.size and cortical.size
    np.testing.assert_array_equal(K2[open_], K1[open_])
    assert (K2[cut] < 1e-6 * K1[cut]).all()
    np.testing.assert_array_equal(K2[cortical], K1[cortical])


def _plasmodesmata_between(net, kinds):
    """The plasmodesmata between two cells of *kinds* (a pair of bool arrays per node), either way round."""
    pd = np.flatnonzero(net.in_section & (net.path == PLASMODESMATA))
    first, second = kinds
    return pd[(first[net.a[pd]] & second[net.b[pd]]) | (second[net.a[pd]] & first[net.b[pd]])]


def test_dead_sclerenchyma():
    """The cortex retagged sclerenchyma, at barrier 11: round(0.8 n) of its n cells dead, the same at every call, others
    with another seed; a plasmodesma takes the smaller of its two cells' factors (a dead cell's closed, two living cells'
    0.05, not 0.05²; E4), the product with sclerenchyma_plasmodesmata_rule 2; at barrier 8, or with every cell given
    living, none dead."""
    K8, net8, _ = section(barrier=8., sclerenchyma=True)
    K, net, mecha = section(barrier=11., sclerenchyma=True)
    scl = net.sclerenchyma
    n = int(scl.sum())
    assert n > 0 and int((scl & ~net.living_cell).sum()) == int(round(0.8 * n))
    np.testing.assert_array_equal(mecha.network().living_cell, net.living_cell)
    _, other, _ = section(barrier=11., sclerenchyma=True, sclerenchyma_seed=1.)
    assert (other.living_cell != net.living_cell).any()
    living, dead = scl & net.living_cell, scl & ~net.living_cell
    open_ = _plasmodesmata_between(net, (living, living))
    closed = _plasmodesmata_between(net, (dead, scl | ~scl))
    mixed = _plasmodesmata_between(net, (living, ~scl))
    mixed = mixed[K8[mixed] > 0]
    assert open_.size and closed.size and mixed.size
    np.testing.assert_allclose(K[open_] / K8[open_], 0.05, rtol=1e-14)
    np.testing.assert_allclose(K[mixed] / K8[mixed], 0.05, rtol=1e-14)
    assert (K[closed] == 0.).all()
    K_product, _, _ = section(barrier=11., sclerenchyma=True, sclerenchyma_plasmodesmata_rule=2.)
    np.testing.assert_allclose(K_product[open_] / K8[open_], 0.05 ** 2, rtol=1e-14)
    plasmodesmata = net8.in_section & (net8.path == PLASMODESMATA)
    K8_plain, _, _ = section(barrier=8.)
    np.testing.assert_array_equal(K8[plasmodesmata], K8_plain[plasmodesmata])     # barrier 8: no dead fibres
    K_living, net_living, _ = section(barrier=11., sclerenchyma=True, living_cell=1.)
    assert net_living.living_cell.all()
    conducting = closed[K8[closed] > 0]
    np.testing.assert_allclose(K_living[conducting] / K8[conducting], 0.05, rtol=1e-14)


def test_cell_and_wall_factors():
    """The cortex's aquaporins off (aquaporin_factor 0): its membranes keep kmb (the floor reads the tissue's kaqp,
    not the cell's: U5), less than with them; every third wall at 0.3 of its conductance (wall_conductivity_factor)."""
    K, net, _ = section()
    off, _, _ = section(inputs={"aquaporin_factor": lambda tissue: np.where(tissue == "cortex", 0., 1.)})
    membranes = np.flatnonzero(net.in_section & (net.path == MEMBRANE))
    owner = np.where(net.compartment_type[net.a[membranes]] == WALL, net.b[membranes], net.a[membranes])
    cortex = membranes[(net.cgroup[owner] == 4) & ~net.intercellular[owner]]
    assert cortex.size
    assert (off[cortex] > 1e-12).all() and (off[cortex] < K[cortex]).all()
    others = np.setdiff1d(np.flatnonzero(net.in_section), cortex)
    np.testing.assert_array_equal(off[others], K[others])
    walls = np.flatnonzero(net.compartment_type == WALL)[::3]
    factor = np.ones(net.compartment_type.size)
    factor[walls] = 0.3
    scaled, _, _ = section(inputs={"wall_conductivity_factor": lambda tissue: factor})
    wall_edges = np.flatnonzero(net.in_section & net.junction_edge)
    wall_end = np.where(net.compartment_type[net.a[wall_edges]] == WALL, net.a[wall_edges], net.b[wall_edges])
    np.testing.assert_allclose(scaled[wall_edges] / K[wall_edges], factor[wall_end], rtol=1e-15)
