"""
test_wall_sigma.py
==================
Optional Casparian-band reflection coefficients ``HydraulicData.sigma_casp_endo`` /
``sigma_casp_exo`` (default 0; default path bit-identical). Only the endo-endo / exo-exo
band walls (that exist at the barrier level) carry sigma, never any other wall.

Hydraulics: a band wall-junction edge with sigma_w carries the osmotic driving term
K_w*sigma_w*(pi_wall - pi_junction) in rhs_o (junction pi = mean psi_os of its adjacent
wall nodes).  Solutes: advective carry factor 1 - sigma_w on wall edges.
"""

import json
import os

import numpy as np
import pytest

from openalea.mecha import Mecha, InData, NetworkBuilder, SoluteTransport
from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData

REF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                        "feature_defaults_reference.json")
DIFF = {'apo_wall': 1e-5, 'plasmodesmata': 1e-4, 'membrane': 1e-5,
        'conserve_boundary': False}


def _build(stage=3, scen=None, **hyd):
    data = OrganInputData.for_root()
    root = RootAnatomy(data, seed=0)
    root.generate_cells()
    root.export_to_adjencymatrix()
    indata = InData()
    indata.geometry.set_maturity_stages([stage])
    if scen is not None:
        indata.boundary.set_os_hetero_scenarios(scen)
    for k, v in hyd.items():
        setattr(indata.hydraulic, k, v)
    network = NetworkBuilder(root)
    network.populate_from_network()
    return Mecha(indata, network=network)


def _band_edges(m, cgroup):
    """Independent set of wall-junction edges of the two-cell walls between two cells of
    tissue *cgroup* (set of frozenset node pairs)."""
    g = m.network.graph
    out = set()
    for w in m.network.cell_manager.walls:
        if len(w.cells) == 2 and all(c.cgroup == cgroup for c in w.cells):
            for nb in g.neighbors(w.node_id):
                if g.edges[w.node_id, nb].get('path') == 'wall':
                    out.add(frozenset((w.node_id, nb)))
    return out


def _wall_edges(m, key='sigma_wall'):
    return [(u, v, ea) for u, v, ea in m.network.graph.edges(data=True) if key in ea]


def _edge_set(m, key='sigma_wall', val=None):
    return {frozenset((u, v)) for u, v, ea in m.network.graph.edges(data=True)
            if key in ea and (val is None or ea[key] == val)}


@pytest.fixture(scope="module")
def default_mecha():
    m = _build(3, scen=[0, 1])
    m.water_flux(verbose=False)
    return m


@pytest.fixture(scope="module")
def endo_mecha():
    m = _build(3, scen=[0, 1], sigma_casp_endo=0.5)
    m.water_flux(verbose=False)
    return m


@pytest.fixture(scope="module")
def half_mecha():
    """sigma_casp_endo = 0.5 (graph attribute on the endo-endo band edges)."""
    m = _build(3, scen=[0, 1], sigma_casp_endo=0.5)
    m.water_flux(verbose=False)
    return m


def _solveW(m):
    m.solve_W(h=0, i_maturity=0)
    return m.results[-1]['matrix_W'].tocsr().copy()


# ---------------------------------------------------------------------------
# 1. defaults bit-identical
# ---------------------------------------------------------------------------

def test_defaults_match_reference_and_no_sigma_attr():
    ref = json.load(open(REF_PATH))
    m = _build(3)
    assert m.hydraulic.sigma_casp_endo == 0.0 and m.hydraulic.sigma_casp_exo == 0.0
    W = _solveW(m)
    D = SoluteTransport(m, {k: v for k, v in DIFF.items() if k != 'conserve_boundary'},
                        mode='full').build_diffusion_matrix(0, 0).tocsr()
    got = {'W_sum': W.sum(), 'W_abs': abs(W).sum(), 'W_sq': W.multiply(W).sum(),
           'W_nnz': W.nnz, 'sol_sum': np.sum(m.results[-1]['solution']),
           'sol_sq': np.sum(m.results[-1]['solution'] ** 2),
           'D_abs': abs(D).sum(), 'D_sq': D.multiply(D).sum(), 'D_nnz': D.nnz}
    for k, v in got.items():
        if k.endswith('nnz'):
            assert int(v) == ref[k], k
        else:
            assert float(v) == pytest.approx(ref[k], rel=1e-12), k
    assert not _wall_edges(m)
    assert all(w.sigma == 0.0 for w in m.network.cell_manager.walls)


# ---------------------------------------------------------------------------
# 2. empty dict / fn -> 0 / fn -> None identical to default
# ---------------------------------------------------------------------------

def test_zero_sigma_identical_to_default():
    m0 = _build(3)
    W0 = _solveW(m0)
    s0 = m0.results[-1]['solution'].copy()
    m1 = _build(3, sigma_casp_endo=0.0, sigma_casp_exo=0.0)
    W1 = _solveW(m1)
    assert abs(W0 - W1).max() < 1e-14
    assert np.abs(s0 - m1.results[-1]['solution']).max() < 1e-14
    assert not _wall_edges(m1)


# ---------------------------------------------------------------------------
# 3. validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["sigma_casp_endo", "sigma_casp_exo"])
@pytest.mark.parametrize("bad", [-0.1, 1.0000001, 2.0])
def test_sigma_out_of_range_raises(key, bad):
    m = _build(3, **{key: bad})
    with pytest.raises(ValueError):
        m.hydraulic.get_casp_sigma(key[len('sigma_casp_'):])
    with pytest.raises(ValueError):
        m.solve_W(h=0, i_maturity=0)


# ---------------------------------------------------------------------------
# 4. band existence / edge selection
# ---------------------------------------------------------------------------

def test_no_band_at_barrier_0():
    # barrier 0: no Casparian band -> nothing carries sigma_wall or casp
    m = _build(0, scen=[0, 1], sigma_casp_endo=0.7, sigma_casp_exo=0.7)
    m.water_flux(verbose=False)
    assert not _wall_edges(m) and not _wall_edges(m, 'casp')
    m0 = _build(0, scen=[0, 1])
    m0.water_flux(verbose=False)
    for s in (1, 2):
        sa = [r for r in m0.results if r.get('scenario') == s][0]['solution']
        sb = [r for r in m.results if r.get('scenario') == s][0]['solution']
        assert np.abs(sa - sb).max() <= 1e-12 * np.abs(sa).max()


def test_endo_band_edges_exact():
    m = _build(3, sigma_casp_endo=0.5, sigma_casp_exo=0.9)   # barrier 3: endo band only
    m.solve_W(h=0, i_maturity=0)
    expected = _band_edges(m, 3)
    assert len(expected) > 0
    assert _edge_set(m, 'sigma_wall') == expected
    assert _edge_set(m, 'casp', 'endo') == expected
    assert _edge_set(m, 'casp') == expected                    # no exo band at barrier 3
    assert all(ea['sigma_wall'] == 0.5 for _, _, ea in _wall_edges(m))
    bw = {n for e in expected for n in e}
    for w in m.network.cell_manager.walls:
        assert w.sigma == (0.5 if w.node_id in bw else 0.0)
    # sigma 0 still tags the band edges (casp) but sets no sigma_wall
    m0 = _build(3)
    m0.solve_W(h=0, i_maturity=0)
    assert _edge_set(m0, 'casp') == expected and not _wall_edges(m0)


def test_exo_band_edges_exact():
    m = _build(4, sigma_casp_endo=0.5, sigma_casp_exo=0.7)   # barrier 4: endo + exo bands
    m.solve_W(h=0, i_maturity=0)
    ex, en = _band_edges(m, 1), _band_edges(m, 3)
    assert len(ex) > 0 and len(en) > 0
    assert _edge_set(m, 'casp', 'exo') == ex and _edge_set(m, 'casp', 'endo') == en
    assert _edge_set(m, 'sigma_wall', 0.7) == ex and _edge_set(m, 'sigma_wall', 0.5) == en
    assert _edge_set(m, 'sigma_wall') == ex | en
    # exo band only (barrier 7 has kw_endo_endo == kw)
    m7 = _build(7, sigma_casp_endo=0.5, sigma_casp_exo=0.7)
    m7.solve_W(h=0, i_maturity=0)
    assert _edge_set(m7, 'sigma_wall') == _band_edges(m7, 1) and not _edge_set(m7, 'casp', 'endo')


# ---------------------------------------------------------------------------
# 5. apoplastic osmotic gradient (barrier 3, endodermal Casparian walls)
# ---------------------------------------------------------------------------

def _hand_term(m):
    """Independent sum of K*sigma*(pi_wall - pi_junction) per node (junction pi = mean
    psi_os of its adjacent wall nodes that carry one)."""
    g = m.network.graph
    cm = m.network.cell_manager
    pi = {}
    for n in g.nodes():
        w = cm.get_wall_by_node_id(n)
        if w is not None and w.psi_os is not None:
            pi[n] = float(w.psi_os)
    out = np.zeros(len(g.nodes()))
    bridging = []
    for u, v, ea in g.edges(data=True):
        s = ea.get('sigma_wall', 0.0)
        if s == 0.0 or ea.get('path') != 'wall':
            continue
        wall, junc = (u, v) if u in pi else (v, u)
        assert wall in pi and junc not in pi
        nb = [pi[x] for x in g.neighbors(junc) if x in pi
              and g.edges[x, junc].get('path') == 'wall']
        pj = sum(nb) / len(nb)
        t = ea['K'] * s * (pi[wall] - pj)
        out[m.indice[wall]] += t
        out[m.indice[junc]] -= t
        if abs(pi[wall] - pj) > 1e-6:      # ignore float-rounding of a uniform mean
            bridging.append((wall, junc, ea['K'], s, pi[wall] - pj))
    return out, bridging


def test_rhs_o_gradient_term(default_mecha, endo_mecha):
    m0, m1 = default_mecha, endo_mecha
    assert len(_wall_edges(m1)) > 0
    assert all(ea['sigma_wall'] == 0.5 for _, _, ea in _wall_edges(m1))
    Kmb0 = m0.solve_W(h=0, i_maturity=0)[3]
    Kmb1 = m1.solve_W(h=0, i_maturity=0)[3]
    np.testing.assert_array_equal(Kmb0, Kmb1)
    ro0 = m0.initialize_scenarios(1, 0, Kmb0, verbose=False)[3]
    ro1 = m1.initialize_scenarios(1, 0, Kmb1, verbose=False)[3]
    expected, bridging = _hand_term(m1)
    assert len(bridging) > 0            # apoplast has a soil-side / xylem-side split
    diff = (ro1 - ro0).ravel()
    assert np.abs(diff).max() > 0
    np.testing.assert_allclose(diff, expected, rtol=1e-10, atol=1e-25)
    assert abs(diff.sum()) <= 1e-12 * np.abs(diff).max() * len(diff)   # pairwise antisymmetric
    # bridging edges join soil-side (-500) and xylem-side (-1800) walls via junctions
    assert {round(b[4]) for b in bridging} <= {433, -867}


def test_sigma_shifts_water_in_osmotic_direction(default_mecha, endo_mecha):
    """Sign convention check.  With sigma_w>0 an edge carries
    Q = K((p_u-p_v) + sigma (pi_u-pi_v)) (water is drawn toward the lower osmotic
    potential).  If rhs_o used the opposite sign, water would NOT balance at the
    junctions; here every bridging junction closes its balance (Q_in == Q_out)."""
    m0, m1 = default_mecha, endo_mecha
    _, bridging = _hand_term(m1)
    g = m1.network.graph
    for wall, junc, K, s, dpi in bridging:
        qi, qo = g.nodes[junc]['Q_in'], g.nodes[junc]['Q_out']
        assert abs(qi - qo) <= 1e-6 * (qi + qo)
    assert bridging


# ---------------------------------------------------------------------------
# 6. SoluteTransport advective carry factor
# ---------------------------------------------------------------------------

def _A_apo(m, **extra):
    st = SoluteTransport(m, dict(DIFF, **extra), mode='apo')
    return st.build_advection_matrix(0, 0).tocsr()


def _flux_edges(m):
    """{(a, b): flux} of apoplastic edges with non-zero standard flux, split into band
    (casp) and non-band."""
    g = m.network.graph
    nwj = m.network.n_wall_junction
    band, other = {}, {}
    for e in m.edge_flux_list[0][0]:
        a, b, F = e['source'], e['target'], e['flux']
        if a < nwj and b < nwj and F != 0.0 and g.has_edge(a, b):
            (band if 'casp' in g.edges[a, b] else other)[(a, b)] = F
    return band, other


def test_solute_advection_scaled_by_band_sigma(default_mecha, half_mecha):
    A0 = _A_apo(default_mecha)
    Ah = _A_apo(half_mecha)                  # graph sigma_casp_endo = 0.5
    band, other = _flux_edges(half_mecha)
    assert band and other
    for (a, b), F in band.items():
        up, dn = (a, b) if F > 0 else (b, a)
        assert A0[dn, up] != 0.0
        assert Ah[dn, up] == pytest.approx(0.5 * A0[dn, up], rel=1e-12)
    for (a, b), F in other.items():          # non-band edges untouched
        up, dn = (a, b) if F > 0 else (b, a)
        assert Ah[dn, up] == A0[dn, up]
    assert abs(_A_apo(default_mecha, sigma_casp=0.0) - A0).max() == 0.0


def test_solute_override_only_touches_band_edges(default_mecha, half_mecha):
    A0 = _A_apo(default_mecha)
    band, other = _flux_edges(default_mecha)
    assert band and other
    # the default graph (sigma 0) still tags the band edges, so the override reaches them
    for ov in (1.0, {'endo': 1.0}):
        A1 = _A_apo(default_mecha, sigma_casp=ov)
        for (a, b), F in band.items():
            up, dn = (a, b) if F > 0 else (b, a)
            assert A1[dn, up] == 0.0
        for (a, b), F in other.items():
            up, dn = (a, b) if F > 0 else (b, a)
            assert A1[dn, up] == A0[dn, up]
    # missing band key -> falls back to the graph value
    Ah = _A_apo(half_mecha)
    assert abs(_A_apo(half_mecha, sigma_casp={'exo': 1.0}) - Ah).max() == 0.0
    # override beats the graph attribute (both directions)
    assert abs(_A_apo(half_mecha, sigma_casp=0.0) - A0).max() <= 1e-12 * abs(A0).max()
    Ao = _A_apo(half_mecha, sigma_casp={'endo': 0.25})
    for (a, b), F in band.items():
        up, dn = (a, b) if F > 0 else (b, a)
        assert Ao[dn, up] == pytest.approx(0.75 * A0[dn, up], rel=1e-12)
    # sigma 1 override: the transport operator equals pure diffusion on band edges
    st = SoluteTransport(default_mecha, dict(DIFF, sigma_casp=1.0), mode='apo')
    T = st.build_transport_operator(0, 0, 0).tocsr()
    D = st.build_diffusion_matrix(0, 0).tocsr()
    T0 = SoluteTransport(default_mecha, DIFF, mode='apo').build_transport_operator(0, 0, 0).tocsr()
    for (a, b), F in band.items():
        up, dn = (a, b) if F > 0 else (b, a)
        assert T[dn, up] == pytest.approx(D[dn, up], rel=1e-12)
        assert T0[dn, up] != T[dn, up]
