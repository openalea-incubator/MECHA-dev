"""
The hydraulic options away from their defaults, against the legacy: φ-thickenings (n_phi_layers, phi_type: the inner,
outer or middle cortex), the cortex's aquaporin gradient (ratio_cortex), the tissues' aquaporin factors, the xylem
pieces' septa (xylem_pieces), other wall and membrane conductivities (kw, kmb, kaqp). Every edge conductance bit for
bit, at barriers 1 and 3. The legacy's cortex gradient on GRANAP's path is a one-element array, which the numpy of
this environment refuses as a float (doc/legacy.rst, ME-44): the comparison gives it its value.
"""
import numpy as np
import pytest

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.geometry import section_parts
from openalea.mecha.legacy import Mecha

from legacy_run import legacy_options, legacy_section
from plants import template_organ

KAQP = dict(cortex_factor=0.5, endo_factor=2.0, exo_factor=0.7, epi_factor=1.3, stele_factor=0.8)
OPTIONS = {
    "phi_inner": dict(port=dict(n_phi_layers=1., phi_type=1.), network=dict(n_phi_layers=1, phi_type=1)),
    "phi_outer": dict(port=dict(n_phi_layers=2., phi_type=2.), network=dict(n_phi_layers=2, phi_type=2)),
    "phi_middle": dict(port=dict(n_phi_layers=1., phi_type=3.), network=dict(n_phi_layers=1, phi_type=3)),
    "ratio_cortex": dict(port=dict(ratio_cortex=0.4), hydraulic=dict(ratio_cortex=0.4)),
    "aquaporin_factors": dict(port=dict(kaqp=6.0E-4, **{f"kaqp_{k}": v for k, v in KAQP.items()}),
                              hydraulic=dict(kaqp=[dict(value=6.0E-4, **KAQP)])),
    "xylem_pieces": dict(port=dict(xylem_pieces=1.), geometry=dict(xylem_pieces=True)),
    "conductivities": dict(port=dict(kw=3.0E-4, kmb=5.0E-5), hydraulic=dict(kw=[3.0E-4], kmb=5.0E-5)),
}
BARRIERS = (1, 3)


@pytest.fixture(scope="module", params=list(OPTIONS))
def compared(case, request):
    spec = OPTIONS[request.param]
    ds, g = case["ds"], case["g"]
    cortex_parameters = Mecha._calculate_cortex_parameters

    def as_floats(self, *arguments, **keywords):                   # ME-44: its one-element arrays, as floats
        return tuple(float(np.asarray(value).reshape(-1)[0])
                     for value in cortex_parameters(self, *arguments, **keywords))
    Mecha._calculate_cortex_parameters = as_floats
    try:
        run = legacy_section(template_organ(case["anatomy"], g, case["segment"]), barriers=BARRIERS,
                             height=case["height"], hydraulic=spec.get("hydraulic"), geometry=spec.get("geometry"),
                             network_attributes=spec.get("network"))
    finally:
        Mecha._calculate_cortex_parameters = cortex_parameters
    if not ds.has("barrier"):
        ds.register("barrier", np.ones(ds.n_nodes()), location="node")
    mecha = legacy_options(MechaHydraulics(data_structure=ds))
    defaults = {name: getattr(mecha, name) for name in spec["port"]}
    nodes, edges = section_parts(ds)[case["segment"]]
    mecha_index = g.property("mecha_index")
    legacy = np.array([int(mecha_index[int(v)]) for v in ds.entity_ids("node")[nodes]])
    geo = mecha.section_geometries()[case["segment"]][2]
    by_position = {tuple(run.network.graph.nodes[j]["position"]): j
                   for j in range(run.network.n_walls, run.network.n_wall_junction)}
    for v in geo.junctions:
        legacy[v] = by_position[(geo.x[v], geo.y[v])]
    stages = {}
    try:
        for name, value in spec["port"].items():
            setattr(mecha, name, value)
        for i, barrier in enumerate(BARRIERS):
            ds.set("barrier", np.full(ds.n_nodes(), float(barrier)))
            mecha()
            stages[barrier] = dict(K=np.asarray(ds.get("conductance")).copy(), net=mecha.network(),
                                   legacy=run.stages[i])
    finally:
        for name, value in defaults.items():
            setattr(mecha, name, value)
        ds.set("barrier", np.ones(ds.n_nodes()))
    return dict(nodes=nodes, edges=edges, legacy=legacy, stages=stages, option=request.param)


@pytest.mark.parametrize("barrier", BARRIERS)
def test_edge_conductances(compared, barrier):
    stage, legacy, edges = compared["stages"][barrier], compared["legacy"], compared["edges"]
    net = stage["net"]
    local = np.full(net.cgroup.size, -1, dtype=np.int64)
    local[compared["nodes"]] = np.arange(compared["nodes"].size)
    expected = {tuple(sorted(k)): v for k, v in stage["legacy"]["edge_K"].items()}
    key = lambda e: tuple(sorted((int(legacy[local[net.a[e]]]), int(legacy[local[net.b[e]]]))))   # noqa: E731
    in_matrix = [e for e in edges if key(e) in expected]
    assert len(in_matrix) == len(expected)
    if compared["option"].startswith("phi"):
        assert net.phi_thick.any()
    np.testing.assert_array_equal([stage["K"][e] for e in in_matrix], [expected[key(e)] for e in in_matrix])
