"""
test_suberin_semantics.py
=========================
Suberin semantics of the hydraulic matrix builder.

Default path (``suberin_model='none'``, original MECHA semantics): suberin acts on the
membrane of the endodermal / exodermal cell that owns it; the wall keeps ``kw`` (only
the Casparian-type bands ``kw_endo_endo`` / ``kw_exo_exo`` / ``kw_cortex_cortex`` modify
walls), and the neighbouring (cortex, pericycle) membranes keep ``kw``.

Lamella path (``suberin_model='lamella'``): additive layers. Primary p = local wall
thickness t; secondary s (per suberized side) and tertiary tau (endodermis) are added:
The tertiary layer is U-shaped: only on the inner (pericycle/stele) and radial
(endodermis) faces of a non-passage endodermal cell, never toward the cortex.
wall  K = temp(t_c) * sum(t_i kw_i) / t_c  with t_c = p + s_A + s_B (the tertiary layer is
excluded from the conductance; t_wall = t_c + tau_A + tau_B is geometric only),
membrane of a suberized cell  1/K = (p/2)/kw + s/kw_lam + tau/kw + 1/(kmb + kaqp)
(tau only on the inner / radial faces); plasmodesmata unchanged (length factor 1,
Clarkson et al. 1971).
"""

import numpy as np
import pytest

from openalea.mecha import Mecha, InData, NetworkBuilder
from openalea.mecha.utils.hydraulic_solver import HydraulicMatrixBuilder
from openalea.granap.root_class import RootAnatomy
from openalea.granap.input_data import OrganInputData

BARRIER = 3          # endodermis full: kw_endo_cortex = Sub_out, kw_endo_peri = Sub_in
KW_LAM = 3.0e-9      # distinct lamella kw for the membrane-formula test


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
    assert not mecha.network.cell_manager.passage, "test assumes no passage cells"
    return mecha, builder


def _params(mecha):
    kw_cfg = mecha.hydraulic_conductivities[0, 0, BARRIER]['kw']
    return (float(mecha.hydraulic.get_kw_value(0)), kw_cfg,
            float(mecha.geometry.maturity_stages[0]['height']))


def _temp(e, height, t):
    return 1.0e-4 * ((e['lateral_distance'] + height) * t - t ** 2) / e['length']


def _endo_cortex_walls(mecha, builder):
    """[(wall_node, wall_edge_attr, endo_cell, cortex_cell)] for mid-wall endo|cortex walls."""
    net = mecha.network
    out = []
    for u, v, e in net.graph.edges(data=True):
        if e.get('path') != 'wall' or min(u, v) >= net.n_walls:
            continue
        cells = builder._get_cells_for_wall_node(min(u, v))
        if sorted(c.cgroup for c in cells) == [3, 4]:
            endo = next(c for c in cells if c.cgroup == 3)
            cort = next(c for c in cells if c.cgroup == 4)
            out.append((min(u, v), e, endo, cort))
    assert out, "no endodermis-cortex wall found"
    return out


def _membrane(mecha, wall_node, cell):
    """(edge attr, HydraulicMembrane) of the membrane between *wall_node* and *cell*."""
    e = mecha.network.graph.edges[wall_node, cell.node_id]
    return e, mecha.network.cell_manager.get_membrane_by_edge(wall_node, cell.node_id)


def _mem_K(kw_series, p_half, e, kmb, kaqp, height, extra=0.0):
    """Reference membrane conductance; *extra* = additional series resistance [1e-4 um/kw]."""
    r = (p_half * 1.0e-4) / kw_series + extra + 1.0 / (kmb + kaqp)
    return 1.0 / r * 1.0e-8 * (height + e['dist']) * e['length']


# ---------------------------------------------------------------------------
# Default path
# ---------------------------------------------------------------------------

def test_default_wall_ignores_suberin_membranes_follow_owner(setup):
    mecha, builder = setup
    mecha.hydraulic.suberin_model = 'none'
    mecha.solve_W(h=0, i_maturity=0)
    kw, kw_cfg, height = _params(mecha)
    sub_out = kw_cfg['kw_endo_cortex']
    assert sub_out < kw * 1e-3, "barrier 3 should carry a strongly suberized endo-cortex key"
    kmb = mecha.hydraulic.kmb

    for node, e, endo, cort in _endo_cortex_walls(mecha, builder)[:30]:
        t = builder._local_thickness(mecha.geometry.thickness, node, node)
        # wall: kw * temp, regardless of Sub_out
        assert e['K'] == pytest.approx(kw * _temp(e, height, t), rel=1e-12)
        # cortex-cell membrane: kw in series; endo-cell membrane: Sub_out in series
        e_c, mb_c = _membrane(mecha, node, cort)
        e_e, mb_e = _membrane(mecha, node, endo)
        assert e_c['K'] == pytest.approx(_mem_K(kw, t / 2, e_c, kmb, mb_c.kaqp, height), rel=1e-12)
        assert e_e['K'] == pytest.approx(_mem_K(sub_out, t / 2, e_e, kmb, mb_e.kaqp, height), rel=1e-12)
        assert e_e['K'] < e_c['K']


# ---------------------------------------------------------------------------
# Lamella path
# ---------------------------------------------------------------------------

def _lamella(mecha, s, tau=0.0, kw_lam=None):
    kw, kw_cfg, _ = _params(mecha)
    mecha.hydraulic.suberin_model = 'lamella'
    mecha.hydraulic.prim_wall_thickness = None
    mecha.hydraulic.sec_wall_thickness = s
    mecha.hydraulic.tert_wall_thickness = tau
    for key in ('kw_lam_endo_out', 'kw_lam_endo_in', 'kw_lam_endo_rad'):
        assert key in kw_cfg
        kw_cfg[key] = kw if kw_lam is None else kw_lam
    mecha.solve_W(h=0, i_maturity=0)


def _walls_between(mecha, builder, groups):
    """[(wall_node, edge_attr, cells)] for mid-wall walls whose cgroups == *groups*."""
    net = mecha.network
    out = []
    for u, v, e in net.graph.edges(data=True):
        if e.get('path') != 'wall' or min(u, v) >= net.n_walls:
            continue
        cells = builder._get_cells_for_wall_node(min(u, v))
        if len(cells) == 2 and sorted(c.cgroup for c in cells) == sorted(groups):
            out.append((min(u, v), e, cells))
    return out


def _reset(mecha):
    mecha.hydraulic.suberin_model = 'none'
    mecha.hydraulic.tert_wall_thickness = 0.0
    mecha.hydraulic.sec_wall_thickness = None


def test_lamella_wall_additive_thickness(setup):
    """Endo-cortex wall (no tau on the outer face): plain wall of thickness t + s."""
    mecha, builder = setup
    kw, _, height = _params(mecha)
    for s, tau in ((0.5, 0.0), (1.0, 0.0), (0.5, 1.0)):
        _lamella(mecha, s, tau)
        for node, e, endo, cort in _endo_cortex_walls(mecha, builder)[:30]:
            t = builder._local_thickness(mecha.geometry.thickness, node, node)
            assert e['K'] == pytest.approx(kw * _temp(e, height, t + s), rel=1e-12)
            assert e['t_wall'] == pytest.approx(t + s, rel=1e-12)   # no tau outside
    _reset(mecha)


@pytest.mark.parametrize("groups, n_tau", [((3, 16), 1), ((3, 3), 2)])
def test_lamella_wall_tertiary_not_conducting(setup, groups, n_tau):
    """Endo-pericycle / endo-endo walls: K independent of tau, t_wall includes the tau."""
    mecha, builder = setup
    kw, kw_cfg, height = _params(mecha)
    s, tau = 0.5, 1.0
    walls = _walls_between(mecha, builder, groups)
    assert walls, f"no {groups} wall"
    _lamella(mecha, s, tau)
    for node, e, cells in walls[:30]:
        t = builder._local_thickness(mecha.geometry.thickness, node, node)
        n_s = 1 if groups == (3, 16) else 2
        t_c = t + n_s * s
        kmid = kw_cfg['kw_endo_endo'] if groups == (3, 3) else kw
        kw_eff = (n_s * s * kw + t * kmid) / t_c            # kw_lam == kw
        assert e['K'] == pytest.approx(kw_eff * _temp(e, height, t_c), rel=1e-12)
        assert e['t_wall'] == pytest.approx(t_c + n_tau * tau, rel=1e-12)
    _reset(mecha)


def test_lamella_membrane_formula(setup):
    """Endodermis membrane: no tau toward cortex; tau toward pericycle and endodermis."""
    mecha, builder = setup
    kw, _, height = _params(mecha)
    kmb = mecha.hydraulic.kmb
    s, tau = 0.5, 1.0
    _lamella(mecha, s, tau, kw_lam=KW_LAM)
    for node, e, endo, cort in _endo_cortex_walls(mecha, builder)[:30]:
        t = builder._local_thickness(mecha.geometry.thickness, node, node)
        e_e, mb_e = _membrane(mecha, node, endo)
        e_c, mb_c = _membrane(mecha, node, cort)
        extra = (s * 1.0e-4) / KW_LAM                      # no tertiary on the outer face
        assert e_e['K'] == pytest.approx(
            _mem_K(kw, t / 2, e_e, kmb, mb_e.kaqp, height, extra), rel=1e-12)
        assert e_c['K'] == pytest.approx(_mem_K(kw, t / 2, e_c, kmb, mb_c.kaqp, height), rel=1e-12)
    extra_t = (s * 1.0e-4) / KW_LAM + (tau * 1.0e-4) / kw  # with tertiary
    for groups in ((3, 16), (3, 3)):
        walls = _walls_between(mecha, builder, groups)
        assert walls
        for node, e, cells in walls[:30]:
            t = builder._local_thickness(mecha.geometry.thickness, node, node)
            for c in cells:
                e_m, mb = _membrane(mecha, node, c)
                if c.cgroup == 3:
                    assert e_m['K'] == pytest.approx(
                        _mem_K(kw, t / 2, e_m, kmb, mb.kaqp, height, extra_t), rel=1e-12)
    _reset(mecha)


def test_lamella_pd_length_factor(setup):
    """PD length factor is 1: PD K is identical with the lamella on vs off.

    Clarkson, Robards & Sanderson 1971 (Planta 96:292): endodermal PDs sit in pits
    where the tertiary wall is only 0.05 um thick; kpl is per-PD, not wall-scaled.
    """
    mecha, builder = setup

    def pd_K():
        return {(u, v): e['K'] for u, v, e in mecha.network.graph.edges(data=True)
                if e.get('path') == 'plasmodesmata'}

    _reset(mecha)
    mecha.solve_W(h=0, i_maturity=0)
    k_off = pd_K()
    assert k_off
    _lamella(mecha, 0.5, 1.0)
    k_on = pd_K()
    _reset(mecha)
    assert k_on.keys() == k_off.keys()
    for key, k in k_off.items():
        assert k_on[key] == pytest.approx(k, rel=1e-12)
