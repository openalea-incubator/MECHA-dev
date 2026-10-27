"""
The solutes: MechaSolutes against the reactions branch's transport (its needle,
``simulations/needle.py``) on GRANAP's ``NeedleAnatomy(seed=42)``, both fed the reactions' water flows (its needle's
water state, scenario 1). The solute's σ is 0 and the air edges' flows 0 in the reactions' run, so that its problems
r1, r2 and r12 vanish (Q-U1, Q-U2): the diffusion (every edge's), the volumes, the sinks, one implicit Euler step from
a mesophyll pulse; with a photosynthetic source (the reactions' increment before the step, the port's equivalent
source C V Δc / dt); with starch, against the reactions' operator in the coupled system with the reaction in the
cells only (Q-U4: the reactions' runs it in the walls too, r4).
"""
import numpy as np
import pytest

from openalea.mecha.geometry import MEMBRANE

from legacy_run import legacy_reactions_step
from solute_case import D_APO, D_MEM, D_PD, DT, mesophyll_source, needle_case as _needle_case, port_step as _run

STARCH = (2.0, 1.0)                                           # d-1: synthesis, degradation (the diurnal runs')
PHOTOSYNTHESIS = 5e-6                                         # mol cm-3: the diurnal runs' c_photo, at g = 1


@pytest.fixture(scope="module")
def needle_case():
    return _needle_case()


def test_diffusion_and_volumes(needle_case):
    """Every edge's diffusion conductance against the reactions' D (the same pairs; to 1e-15: its membranes'
    D_mem × L × h order), the volumes exactly."""
    c, ds = needle_case, needle_case["ds"]
    De = sum(np.asarray(ds.get(f"{kind}_diffusion_factor")) * d for kind, d in
             (("apoplastic", D_APO), ("symplastic", D_PD), ("membrane", D_MEM)))[c["edges"]]
    D = c["reference"]["D"].tocsr()
    legacy = np.asarray(D[c["ends"][:, 0], c["ends"][:, 1]]).ravel()
    np.testing.assert_allclose(De, legacy, rtol=1e-15, atol=0)
    off = D.copy()
    off.setdiag(0)
    off.eliminate_zeros()
    assert (De != 0).sum() == off.nnz // 2
    volume = np.asarray(ds.get("node_volume"))[c["nodes"]]
    np.testing.assert_array_equal(volume[c["liquid"]], c["reference"]["volumes"][c["index"]][c["liquid"]])


def test_sinks(needle_case):
    c = needle_case
    c["solutes"]()
    sink = np.asarray(c["ds"].get("sucrose_sink"))[c["nodes"]] > 0
    legacy = np.isin(c["index"], list(c["reference"]["bc"]))
    assert sink.sum() > 0 and np.array_equal(sink, legacy & c["liquid"])


def test_one_step(needle_case):
    """From the mesophyll pulse: every liquid node's sucrose to 1e-11 of the range, the mass to 1e-12."""
    c = needle_case
    sucrose, _ = _run(c, c["reference"])
    reference = c["reference"]["c1"][c["index"]]
    liquid = c["liquid"]
    np.testing.assert_allclose(sucrose[liquid], reference[liquid], rtol=0, atol=1e-11 * abs(reference).max())
    volume = c["reference"]["volumes"][c["index"]]
    assert (volume * sucrose)[liquid].sum() == pytest.approx((volume * reference)[liquid].sum(), rel=1e-12)


def test_one_step_with_photosynthesis(needle_case):
    """The reactions' increment c_photo in the mesophyll before the step is the source V c_photo / dt."""
    c = needle_case
    meso = c["reference"]["kinds"]["meso"]
    reference = legacy_reactions_step(c["organ"], d_apo=D_APO, d_pd=D_PD, d_mem=D_MEM, dt=DT,
                                      source={i: PHOTOSYNTHESIS for i in meso})
    source = mesophyll_source(c, reference, PHOTOSYNTHESIS)
    sucrose, _ = _run(c, reference, source=source)
    expected = reference["c1"][c["index"]]
    liquid = c["liquid"]
    np.testing.assert_allclose(sucrose[liquid], expected[liquid], rtol=0, atol=1e-11 * abs(expected).max())


def test_one_step_with_starch(needle_case):
    """
    Sucrose and starch coupled by the reaction, against the reactions' MultiSoluteTransport with the reaction in the
    liquid cells only (Q-U4): to 1e-11 of the ranges; the walls hold no starch. Its own reaction in the walls too (r4)
    is a reviewed difference, measured.
    """
    c = needle_case
    reference = legacy_reactions_step(c["organ"], d_apo=D_APO, d_pd=D_PD, d_mem=D_MEM, dt=DT, starch=STARCH,
                                      cells_only=True)
    sucrose, starch = _run(c, reference, starch=STARCH)
    liquid = c["liquid"]
    for port, legacy in ((sucrose, reference["c1"]), (starch, reference["starch1"])):
        legacy = legacy[c["index"]]
        np.testing.assert_allclose(port[liquid], legacy[liquid], rtol=0, atol=1e-11 * abs(legacy).max())
    walls = liquid & (c["index"] < reference["network"].n_wall_junction)
    assert (starch[walls] == 0).all() and (starch[liquid & ~walls] > 0).any()
    everywhere = legacy_reactions_step(c["organ"], d_apo=D_APO, d_pd=D_PD, d_mem=D_MEM, dt=DT, starch=STARCH)
    volume = reference["volumes"]
    r4 = abs(volume * (everywhere["starch1"] - reference["starch1"])).sum() / (volume * reference["starch1"]).sum()
    print(f"r4: the walls' reaction changes the starch mass by {r4:.3e} (relative)")
    assert r4 > 0.                                   # 0.61 at this step (plan §28a)


def _difference(case, port, reference):
    """The liquid nodes' sucrose difference, relative to the reference's range."""
    reference = reference["c1"][case["index"]]
    liquid = case["liquid"]
    return abs(port - reference)[liquid].max() / abs(reference)[liquid].max()


def _sigma_case(case, groups, osmotic_term):
    """σ 0.7 for the reactions' solute on the cgroups *groups* (all: None) and for the port on the membranes whose
    cell is of these cgroups: the reactions' step and the port's sucrose."""
    ds = case["ds"]
    view = ds.to_graph_view()
    cgroup, path = np.asarray(ds.get("cgroup")), np.asarray(ds.get("network_path"))
    cell_end = np.where(np.asarray(ds.get("kind"))[view.tail] == 0, view.tail, view.head)
    reference = legacy_reactions_step(case["organ"], d_apo=D_APO, d_pd=D_PD, d_mem=D_MEM, dt=DT,
                                      solute_sigma={g: 0.7 for g in (range(40) if groups is None else groups)},
                                      osmotic_term=osmotic_term)
    on = (path == MEMBRANE) & (True if groups is None else np.isin(cgroup[cell_end], groups))
    sucrose, _ = _run(case, reference, sigma=np.where(on, 0.7, 0.))
    return sucrose, reference


def test_one_step_with_reflection(needle_case):
    """The solute's σ 0.7 on every membrane, the carried flow (1 − σ) Q: exact without the reactions' K σ Δπ added
    to the flows (r1)."""
    c = needle_case
    sucrose, reference = _sigma_case(c, None, osmotic_term=False)
    assert _difference(c, sucrose, reference) < 1e-11


def test_reviewed_differences(needle_case):
    """
    The reactions' problems not ported, measured on this step (plan §28a), against the port with the same σ:
        r1    the solute's σ 0.7 on every cgroup: the osmotic flow K σ Δπ counted twice
        r2    σ 0.7 on the most frequent cgroup but the mesophyll's and the sinks', r1 removed: σ read on a wall's
              end (its cgroup taken as the mesophyll's, 4) of the membranes stored cell → wall; latent here: the
              reactions' graph yields every membrane wall → cell (its walls inserted first), so 0 to round-off
        r12   the air edges' flows kept: the air nodes (Dirichlet 0) as sugar sinks; negligible on this water state
    """
    c = needle_case
    cells = c["nodes"][(np.asarray(c["ds"].get("kind"))[c["nodes"]] == 0) & c["liquid"]]
    groups, counts = np.unique(np.asarray(c["ds"].get("cgroup"))[cells], return_counts=True)
    free = ~np.isin(groups, (4, 11, 12, 13))                     # nor the sinks'
    other = int(groups[free][np.argmax(counts[free])])
    differences = {"r1": _difference(c, *_sigma_case(c, None, osmotic_term=True)),
                   "r2": _difference(c, *_sigma_case(c, (other,), osmotic_term=False))}
    reference = legacy_reactions_step(c["organ"], d_apo=D_APO, d_pd=D_PD, d_mem=D_MEM, dt=DT, air_flows=True)
    sucrose, _ = _run(c, c["reference"])
    differences["r12"] = _difference(c, sucrose, reference)
    print(f"reviewed differences (max |Δc| / max c; r2 on cgroup {other}):",
          {k: f"{v:.3e}" for k, v in differences.items()})
    assert differences["r1"] > 1. and differences["r2"] < 1e-10 and differences["r12"] < 1e-10


def test_one_step_with_storage_and_decay(needle_case):
    """The capacitance's storage fractions (C_wall 0.3, C_cell 0.8) and a first-order degradation (k_deg 0.5 d⁻¹) of
    the reactions' SoluteTransport: the port's storage_fraction_wall, _cell and k_decay, to 1e-11 of the range."""
    c = needle_case
    reference = legacy_reactions_step(c["organ"], d_apo=D_APO, d_pd=D_PD, d_mem=D_MEM, dt=DT, storage=(0.3, 0.8),
                                      k_deg=0.5)
    sucrose, _ = _run(c, reference, storage=(0.3, 0.8), k_decay=0.5)
    expected = reference["c1"][c["index"]]
    liquid = c["liquid"]
    np.testing.assert_allclose(sucrose[liquid], expected[liquid], rtol=0, atol=1e-11 * abs(expected).max())
    plain, _ = _run(c, c["reference"])
    assert np.abs(plain - sucrose)[liquid].max() > 1e-3 * abs(expected).max()      # the options matter


def test_crank_nicolson(needle_case):
    """θ = 0.5 (Crank–Nicolson, solute_theta): sucrose and starch coupled by the reaction against the reactions'
    MultiSoluteTransport (its reaction in the cells only, Q-U4; it takes no k_deg), and sucrose alone with storage
    fractions and a first-order loss against its SoluteTransport: to 1e-11 of the ranges."""
    c = needle_case
    liquid = c["liquid"]
    c["solutes"].solute_theta = 0.5
    try:
        reference = legacy_reactions_step(c["organ"], d_apo=D_APO, d_pd=D_PD, d_mem=D_MEM, dt=DT, starch=STARCH,
                                          cells_only=True, theta=0.5)
        sucrose, starch = _run(c, reference, starch=STARCH)
        for port, legacy in ((sucrose, reference["c1"]), (starch, reference["starch1"])):
            legacy = legacy[c["index"]]
            np.testing.assert_allclose(port[liquid], legacy[liquid], rtol=0, atol=1e-11 * abs(legacy).max())
        reference = legacy_reactions_step(c["organ"], d_apo=D_APO, d_pd=D_PD, d_mem=D_MEM, dt=DT, storage=(0.3, 0.8),
                                          k_deg=0.5, theta=0.5)
        sucrose, _ = _run(c, reference, storage=(0.3, 0.8), k_decay=0.5)
        expected = reference["c1"][c["index"]]
        np.testing.assert_allclose(sucrose[liquid], expected[liquid], rtol=0, atol=1e-11 * abs(expected).max())
    finally:
        c["solutes"].solute_theta = 1.
