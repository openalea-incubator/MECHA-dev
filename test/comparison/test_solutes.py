"""
The solutes (devplan_metafspm.md, §28): MechaSolutes against the reactions branch's transport (its needle,
``simulations/needle.py``) on GRANAP's ``NeedleAnatomy(seed=42)``, both fed the reactions' water flows (its needle's
water state, scenario 1). The solute's σ is 0 and the air edges' flows 0 in the reactions' run, so that its problems
r1, r2 and r12 vanish (Q-U1, Q-U2): the diffusion (every edge's), the volumes, the sinks, one implicit Euler step from
a mesophyll pulse; with a photosynthetic source (the reactions' increment before the step, the port's equivalent
source C V Δc / dt); with starch, against the reactions' operator in the coupled system with the reaction in the
cells only (Q-U4: the reactions' runs it in the walls too, r4).
"""
import numpy as np
import pytest

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.geometry import MEMBRANE, section_parts
from openalea.mecha.solutes import MechaSolutes

from legacy_run import legacy_options, legacy_reactions_step
from plants import anatomy_data_structure, needle, needle_anatomy_component, template_organ

D_APO, D_PD, D_MEM, DT = 0.1, 0.5, 1e-6, 1. / 24.             # cm2 d-1, d (the simulation's 3600 s)
NEEDLE = dict(kw=2.5634e-3, kw_meso_hypo=1e-16, kmb=3.6314e-4, x_contact=1e10, k_xyl=1.0, collar_condition=0.)
STARCH = (2.0, 1.0)                                           # d-1: synthesis, degradation (the diurnal runs')
PHOTOSYNTHESIS = 5e-6                                         # mol cm-3: the diurnal runs' c_photo, at g = 1


@pytest.fixture(scope="module")
def needle_case():
    g, ds, anatomy = anatomy_data_structure(needle(), needle_anatomy_component())
    segment = int(ds.entity_ids("SubOrgan")[0])
    organ = template_organ(anatomy, g, segment)
    ds.register("barrier", np.ones(ds.n_nodes()), location="node")
    mecha = legacy_options(MechaHydraulics(data_structure=ds))
    for name, value in NEEDLE.items():
        setattr(mecha, name, value)
    mecha()
    nodes, edges = section_parts(ds)[segment]
    geo = mecha.section_geometries()[segment][2]
    reference = legacy_reactions_step(organ, d_apo=D_APO, d_pd=D_PD, d_mem=D_MEM, dt=DT)
    net = reference["network"]
    mecha_index = g.property("mecha_index")
    index = np.array([int(mecha_index[int(v)]) for v in ds.entity_ids("node")[nodes]])
    index[geo.junction_order] = net.n_walls + np.arange(net.n_junctions)
    local = np.full(ds.n_nodes(), -1, dtype=np.int64)
    local[nodes] = np.arange(nodes.size)
    view = ds.to_graph_view()
    ends = np.c_[index[local[view.tail[edges]]], index[local[view.head[edges]]]]
    solutes = MechaSolutes(data_structure=ds, time_step=DT * 86400.)
    for name, value in dict(d_apoplast=D_APO, d_plasmodesmata=D_PD, d_membrane=D_MEM).items():
        setattr(solutes, name, value)
    return dict(g=g, ds=ds, organ=organ, mecha=mecha, solutes=solutes, nodes=nodes, edges=edges, index=index,
                ends=ends, reference=reference, liquid=np.asarray(ds.get("vapour_node"))[nodes] == 0)


def _run(case, reference, starch=(0., 0.), source=None, sigma=None):
    """The port's step on the reactions' flows and initial concentrations (σ 0, or *sigma* per edge)."""
    ds, nodes, edges = case["ds"], case["nodes"], case["edges"]
    flows, Q = reference["flows"], np.zeros(ds.n_edges())
    for e, (a, b) in zip(edges, case["ends"]):
        Q[e] = flows.get((a, b), -flows.get((b, a), 0.))
    ds.set("flow", Q)
    ds.set("reflection_coefficient", np.zeros(ds.n_edges()) if sigma is None else sigma)
    c0 = np.zeros(ds.n_nodes())
    c0[nodes] = reference["c0"][case["index"]]
    ds.set("sucrose", c0)
    ds.set("starch", np.zeros(ds.n_nodes()))
    ds.set("sucrose_source", np.zeros(ds.n_nodes()) if source is None else source)
    case["solutes"].k_starch_synthesis, case["solutes"].k_starch_degradation = starch
    case["solutes"]()
    return np.asarray(ds.get("sucrose"))[nodes].copy(), np.asarray(ds.get("starch"))[nodes].copy()


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
    by_legacy = dict(zip(c["index"].tolist(), c["nodes"].tolist()))
    source = np.zeros(c["ds"].n_nodes())
    volume = np.asarray(c["ds"].get("node_volume"))
    for i in meso:
        source[by_legacy[i]] = volume[by_legacy[i]] * PHOTOSYNTHESIS / DT
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
