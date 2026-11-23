"""
test_cell_wall_factors.py
=========================
Optional per-cell aquaporin factor, per-wall kw factor, structure-aware membrane
diffusion (``membrane_pd``) and ``SoluteTransport.boundary_flux``.

All features are OFF by default: the default path must be bit-identical to the code
before they existed (reference metrics in ``data/feature_defaults_reference.json`` were
computed on git HEAD before the change).
"""

import json
import os

import numpy as np
import pytest

from openalea.mecha import Mecha, InData, NetworkBuilder, SoluteTransport
from openalea.mecha.utils.hydraulic_solver import HydraulicMatrixBuilder
from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData

REF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                        "feature_defaults_reference.json")
BARRIER = 3
KW_LAM = 3.0e-9
DIFF = {'apo_wall': 1e-5, 'plasmodesmata': 1e-4, 'membrane': 1e-5}


@pytest.fixture(scope="module")
def setup():
    data = OrganInputData.for_root()
    root = RootAnatomy(data, seed=0)
    root.generate_cells()
    root.export_to_adjencymatrix()
    indata = InData()
    indata.geometry.set_maturity_stages([BARRIER])
    network = NetworkBuilder(root)
    network.populate_from_network()
    mecha = Mecha(indata, network=network)
    builder = HydraulicMatrixBuilder(
        network=mecha.network, geometry=mecha.geometry, boundary=mecha.boundary,
        hydraulic=mecha.hydraulic, hormones=mecha.hormones, general=mecha.general,
        geo_props=mecha.geo_props, position=mecha.position, indice=mecha.indice)
    builder._wall_thk = builder._build_wall_thickness()
    return mecha, builder


@pytest.fixture(autouse=True)
def _clean(setup):
    """Every test starts and ends with all optional features off."""
    mecha = setup[0]
    _reset(mecha)
    yield
    _reset(mecha)


def _reset(mecha):
    h = mecha.hydraulic
    h.aqp_factor = None
    h.aqp_factor_by_type = None
    h.kw_wall_factor = None
    h.kw_wall_factor_fn = None
    h.suberin_model = 'none'
    h.tert_wall_thickness = 0.0
    h.sec_wall_thickness = None


def _solveW(mecha):
    mecha.solve_W(h=0, i_maturity=0)
    return mecha.results[-1]['matrix_W'].tocsr().copy()


def _D(mecha, **extra):
    st = SoluteTransport(mecha, dict(DIFF, **extra), mode='full')
    return st, st.build_diffusion_matrix(0, 0).tocsr()


def _same(A, B):
    return (A != B).nnz == 0


def _membrane_edges(mecha, cell):
    """[(wall_node, edge_attr)] of the membrane edges of *cell*."""
    g = mecha.network.graph
    return [(n, g.edges[n, cell.node_id]) for n in g.neighbors(cell.node_id)
            if g.edges[n, cell.node_id].get('path') == 'membrane']


def _cells(mecha, ctype):
    return [c for c in mecha.network.cell_manager if c.cell_type == ctype]


def _mem_K(kw_series, p_half, e, kmb, kaqp, height):
    r = (p_half * 1.0e-4) / kw_series + 1.0 / (kmb + kaqp)
    return 1.0 / r * 1.0e-8 * (height + e['dist']) * e['length']


# ---------------------------------------------------------------------------
# (a) defaults unchanged
# ---------------------------------------------------------------------------

def test_defaults_match_pre_feature_reference(setup):
    mecha, _ = setup
    ref = json.load(open(REF_PATH))
    W = _solveW(mecha)
    _, D = _D(mecha)
    got = {'W_sum': W.sum(), 'W_abs': abs(W).sum(), 'W_sq': W.multiply(W).sum(),
           'W_nnz': W.nnz, 'sol_sum': np.sum(mecha.results[-1]['solution']),
           'sol_sq': np.sum(mecha.results[-1]['solution'] ** 2),
           'D_abs': abs(D).sum(), 'D_sq': D.multiply(D).sum(), 'D_nnz': D.nnz}
    for k, v in got.items():
        if k.endswith('nnz'):
            assert int(v) == ref[k], k
        else:
            assert float(v) == pytest.approx(ref[k], rel=1e-12), k


def test_default_inputs_are_none(setup):
    h = setup[0].hydraulic
    assert h.aqp_factor is None and h.aqp_factor_by_type is None
    assert h.kw_wall_factor is None and h.kw_wall_factor_fn is None
    assert h.get_aqp_factor(0, 'cortex') is None and h.get_wall_kw_factor(0, []) is None


# ---------------------------------------------------------------------------
# (b) factor 1 everywhere == default (exact)
# ---------------------------------------------------------------------------

def test_factor_one_is_identical(setup):
    mecha, _ = setup
    W0 = _solveW(mecha)
    _, D0 = _D(mecha)
    cm = mecha.network.cell_manager
    mecha.hydraulic.aqp_factor = {c.cell_id: 1.0 for c in cm}
    mecha.hydraulic.aqp_factor_by_type = {c.cell_type: 1.0 for c in cm}
    mecha.hydraulic.kw_wall_factor = {w: 1.0 for w in range(mecha.network.n_walls)}
    mecha.hydraulic.kw_wall_factor_fn = lambda wid, cells: 1.0
    W1 = _solveW(mecha)
    _, D1 = _D(mecha)
    assert _same(W0, W1)
    assert _same(D0, D1)
    # only the callable
    mecha.hydraulic.aqp_factor = mecha.hydraulic.aqp_factor_by_type = None
    mecha.hydraulic.kw_wall_factor = None
    assert _same(W0, _solveW(mecha))


# ---------------------------------------------------------------------------
# (c) aquaporin factor
# ---------------------------------------------------------------------------

def test_aqp_factor_zero_leaves_kmb_only(setup):
    mecha, builder = setup
    kw = float(mecha.hydraulic.get_kw_value(0))
    kw_cfg = mecha.hydraulic_conductivities[0, 0, BARRIER]['kw']
    height = float(mecha.geometry.maturity_stages[0]['height'])
    kmb = mecha.hydraulic.kmb
    assert kmb > 0
    cort = _cells(mecha, 'cortex')
    target, other = cort[0], cort[1]

    _solveW(mecha)
    base = {n: e['K'] for n, e in _membrane_edges(mecha, target)}
    base_other = {n: e['K'] for n, e in _membrane_edges(mecha, other)}
    aqp0 = {n: e for n, e in _membrane_edges(mecha, target)}
    assert all(e['K'] > 0 for e in aqp0.values())

    mecha.hydraulic.aqp_factor = {target.cell_id: 0.0}
    _solveW(mecha)
    cm = mecha.network.cell_manager
    for n, e in _membrane_edges(mecha, target):
        mb = cm.get_membrane_by_edge(n, target.node_id)
        assert mb.kaqp == 0.0
        cells_i = builder._get_cells_for_wall_node(n)
        kws = builder._membrane_series_kw(target.cgroup, cells_i, kw_cfg, kw)
        t = builder._local_thickness(mecha.geometry.thickness, n)
        assert e['K'] == pytest.approx(_mem_K(kws, t / 2, e, kmb, 0.0, height), rel=1e-12)
        assert e['K'] < base[n]
    # a neighbouring cell is untouched
    assert {n: e['K'] for n, e in _membrane_edges(mecha, other)} == base_other

    # partial factor scales only the aquaporin part
    mecha.hydraulic.aqp_factor = {target.cell_id: 0.5}
    _solveW(mecha)
    mb_ref = None
    for n, e in _membrane_edges(mecha, target):
        mb = cm.get_membrane_by_edge(n, target.node_id)
        kaqp_full = mb.kaqp / 0.5
        cells_i = builder._get_cells_for_wall_node(n)
        kws = builder._membrane_series_kw(target.cgroup, cells_i, kw_cfg, kw)
        t = builder._local_thickness(mecha.geometry.thickness, n)
        assert e['K'] == pytest.approx(_mem_K(kws, t / 2, e, kmb, 0.5 * kaqp_full, height), rel=1e-12)
        assert e['aqp_factor'] == 0.5


def test_aqp_factor_by_type_and_id_override(setup):
    mecha, _ = setup
    cm = mecha.network.cell_manager
    cort = _cells(mecha, 'cortex')
    _solveW(mecha)
    base = {c.cell_id: {n: cm.get_membrane_by_edge(n, c.node_id).kaqp
                        for n, _ in _membrane_edges(mecha, c)} for c in cort[:3]}
    mecha.hydraulic.aqp_factor_by_type = {'cortex': 0.0}
    mecha.hydraulic.aqp_factor = {cort[0].cell_id: 2.0}      # id entry wins over type
    _solveW(mecha)
    for c in cort[:3]:
        f = 2.0 if c.cell_id == cort[0].cell_id else 0.0
        for n, _ in _membrane_edges(mecha, c):
            assert cm.get_membrane_by_edge(n, c.node_id).kaqp == pytest.approx(
                base[c.cell_id][n] * f, rel=1e-14)


# ---------------------------------------------------------------------------
# (d) wall factor
# ---------------------------------------------------------------------------

def _two_cell_wall(mecha, builder, groups=(4, 4)):
    net = mecha.network
    for u, v, e in net.graph.edges(data=True):
        if e.get('path') != 'wall' or min(u, v) >= net.n_walls:
            continue
        cells = builder._get_cells_for_wall_node(min(u, v))
        if len(cells) == 2 and sorted(c.cgroup for c in cells) == sorted(groups):
            return min(u, v)
    raise AssertionError('no wall')


@pytest.mark.parametrize("use_fn", [False, True])
def test_wall_factor_zero_removes_wall(setup, use_fn):
    mecha, builder = setup
    g = mecha.network.graph
    w = _two_cell_wall(mecha, builder)
    nbrs = [n for n in g.neighbors(w) if g.edges[w, n].get('path') == 'wall']
    assert nbrs
    W0 = _solveW(mecha)
    _, D0 = _D(mecha)
    assert all(W0[w, n] > 0 and D0[w, n] > 0 for n in nbrs)
    other = next(n for u, v, e in g.edges(data=True) if e.get('path') == 'wall'
                 for n in [min(u, v)] if n != w and n < mecha.network.n_walls)
    K_other = g.edges[other, [x for x in g.neighbors(other)
                              if g.edges[other, x].get('path') == 'wall'][0]]['K']

    if use_fn:
        seen = []
        def fn(wid, cells):
            seen.append((wid, len(cells)))
            return 0.0 if wid == w else None
        mecha.hydraulic.kw_wall_factor_fn = fn
    else:
        mecha.hydraulic.kw_wall_factor = {w: 0.0}
    W1 = _solveW(mecha)
    _, D1 = _D(mecha)
    for n in nbrs:
        assert g.edges[w, n]['K'] == 0.0
        assert W1[w, n] == 0.0 and W1[n, w] == 0.0
        assert D1[w, n] == 0.0 and D1[n, w] == 0.0
    # nothing else changed in the wall-wall couplings elsewhere
    assert g.edges[other, [x for x in g.neighbors(other)
                           if g.edges[other, x].get('path') == 'wall'][0]]['K'] == K_other
    if use_fn:
        assert any(wid == w for wid, _ in seen)
    # partial factor scales K and the diffusive coupling identically
    mecha.hydraulic.kw_wall_factor_fn = None
    mecha.hydraulic.kw_wall_factor = {w: 0.25}
    W2 = _solveW(mecha)
    _, D2 = _D(mecha)
    for n in nbrs:
        assert W2[w, n] == pytest.approx(0.25 * W0[w, n], rel=1e-12)
        assert D2[w, n] == pytest.approx(0.25 * D0[w, n], rel=1e-12)


# ---------------------------------------------------------------------------
# (e) boundary flux conservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ['full', 'apo'])
def test_boundary_flux_conservation(setup, mode):
    mecha, _ = setup
    net = mecha.network
    st = SoluteTransport(mecha, DIFF, mode=mode)
    walls = np.array([n for n in range(net.n_walls)])
    xs = np.array([mecha.position[n][0] for n in walls], dtype=float)
    order = np.argsort(xs)
    inlet = [int(walls[k]) for k in order[:15]]
    outlet = [int(walls[k]) for k in order[-15:]]
    bc = {n: 1.0 for n in inlet}
    bc.update({n: 0.0 for n in outlet})
    n = st._matrix_size
    c = st.solve(0, 0, 0, np.zeros(n), boundary_conditions=bc, operators='D')
    assert np.all(np.isfinite(c))
    f_in = st.boundary_flux(c, inlet)            # defaults to the last solve's indices
    f_out = st.boundary_flux(c, outlet, h=0, i_maturity=0, i_scenario=0, operators='D')
    assert f_in < 0 < f_out                       # inflow negative, outflow positive
    assert f_out == pytest.approx(-f_in, rel=1e-8)
    # per-node values sum to the scalar; non-BC interior nodes carry no net flux
    assert st.boundary_flux(c, outlet, per_node=True).sum() == pytest.approx(f_out, rel=1e-12)
    interior = [k for k in range(net.n_walls) if k not in bc][:50]
    assert abs(st.boundary_flux(c, interior)) < 1e-8 * abs(f_out)


# ---------------------------------------------------------------------------
# (f) membrane_pd mode
# ---------------------------------------------------------------------------

def test_membrane_pd_air_space_and_formula(setup):
    mecha, _ = setup
    height = float(mecha.geometry.maturity_stages[0]['height'])
    P_l, P_a = 1e-3, 4e-3
    _solveW(mecha)
    air = mecha.network.cell_manager.intercellular
    assert air

    # legacy uniform D_mem path: air-space membranes DO diffuse
    st_old, D_old = _D(mecha)
    n_old = 0
    for c in air[:50]:
        for n, e in _membrane_edges(mecha, c):
            n_old += D_old[n, c.node_id] > 0
    assert n_old > 0

    st, D = _D(mecha, membrane_pd=dict(lipid=P_l, aqp=P_a))
    for c in air:
        for n, e in _membrane_edges(mecha, c):
            assert e['membrane_diffusivity'] == 0.0
            assert D[n, c.node_id] == 0.0
    # living cell: P = P_l + P_a (factor default 1), conductance = P*(h+dist)*L*1e-8
    cort = _cells(mecha, 'cortex')[0]
    for n, e in _membrane_edges(mecha, cort):
        assert D[n, cort.node_id] == pytest.approx(
            (P_l + P_a) * (height + e['dist']) * e['length'] * 1e-8, rel=1e-12)

    # aquaporin factor 0 -> lipid only; 0.5 -> halfway
    for f, p in ((0.0, P_l), (0.5, P_l + 0.5 * P_a)):
        mecha.hydraulic.aqp_factor = {cort.cell_id: f}
        _, Df = _D(mecha, membrane_pd=dict(lipid=P_l, aqp=P_a))
        for n, e in _membrane_edges(mecha, cort):
            assert Df[n, cort.node_id] == pytest.approx(
                p * (height + e['dist']) * e['length'] * 1e-8, rel=1e-12)
    mecha.hydraulic.aqp_factor = None

    # scalar = lipid only
    _, Ds = _D(mecha, membrane_pd=P_l)
    for n, e in _membrane_edges(mecha, cort):
        assert Ds[n, cort.node_id] == pytest.approx(
            P_l * (height + e['dist']) * e['length'] * 1e-8, rel=1e-12)


def test_membrane_pd_lamella_in_series(setup):
    mecha, builder = setup
    kw = float(mecha.hydraulic.get_kw_value(0))
    kw_cfg = mecha.hydraulic_conductivities[0, 0, BARRIER]['kw']
    height = float(mecha.geometry.maturity_stages[0]['height'])
    keys = ('kw_lam_endo_out', 'kw_lam_endo_in', 'kw_lam_endo_rad')
    saved = {k: kw_cfg[k] for k in keys}
    P_l, s = 1e-3, 0.5
    pd = dict(lipid=P_l, aqp=0.0)
    try:
        _solveW(mecha)
        _, D_off = _D(mecha, membrane_pd=pd)
        for k in keys:
            kw_cfg[k] = KW_LAM
        mecha.hydraulic.suberin_model = 'lamella'
        mecha.hydraulic.sec_wall_thickness = s
        _solveW(mecha)
        st, D_on = _D(mecha, membrane_pd=pd)
        found = 0
        for endo in _cells(mecha, 'endodermis'):
            for n, e in _membrane_edges(mecha, endo):
                if 'kw_lam' not in e:
                    continue
                cells = builder._get_cells_for_wall_node(n)
                if sorted(c.cgroup for c in cells) != [3, 4]:
                    continue
                P_lam = DIFF['apo_wall'] * (KW_LAM / kw) / (s * 1e-4)
                P_eff = 1.0 / (1.0 / P_l + 1.0 / P_lam)
                area = (height + e['dist']) * e['length'] * 1e-8
                assert D_on[n, endo.node_id] == pytest.approx(P_eff * area, rel=1e-12)
                assert D_on[n, endo.node_id] < D_off[n, endo.node_id]
                found += 1
        assert found > 0
        # non-suberized (cortex) membranes are unaffected by the lamella
        cort = _cells(mecha, 'cortex')[0]
        for n, e in _membrane_edges(mecha, cort):
            assert D_on[n, cort.node_id] == pytest.approx(D_off[n, cort.node_id], rel=1e-12)
    finally:
        for k, v in saved.items():
            kw_cfg[k] = v
