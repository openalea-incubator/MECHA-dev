"""
The standard transmembrane fractions (MechaHydraulics' standard_transmembrane_fraction and membrane_layer,
postprocess.transmembrane_layers and transmembrane_cells) against the legacy's
(``Mecha.standard_transmembrane_fractions``, its scenario 0). The legacy looks a cell's row up by value (``rank in
rank_to_row``: ME-40), so it counts only the membranes whose rank is one of its rows' numbers, and never excludes the
intercellular spaces (it compares their cells with ids: ME-41); the comparison takes the membranes it counts, and its
rows with the port's rows of every membrane but the intercellular spaces' kept.
"""
import numpy as np
import pytest

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.geometry import MEMBRANE, section_parts
from openalea.mecha.osmotic import membrane_rows
from openalea.mecha.postprocess import transmembrane_cells, transmembrane_layers

from legacy_run import legacy_fractions, legacy_options
from plants import template_organ


@pytest.fixture(scope="module", params=(1, 3), ids=lambda b: f"barrier{b}")
def both(case, request):
    barrier = request.param
    data_structure, g = case["data_structure"], case["mtg"]
    if not data_structure.has("barrier"):
        data_structure.register("barrier", np.ones(data_structure.n_nodes()), location="node")
    data_structure.set("barrier", np.full(data_structure.n_nodes(), float(barrier)))
    mecha = legacy_options(MechaHydraulics(data_structure=data_structure))
    mecha()
    legacy = legacy_fractions(template_organ(case["anatomy"], g, case["segment"]), barrier, height=case["height"])
    nodes, edges = section_parts(data_structure)[case["segment"]]
    geo = mecha.section_geometries()[case["segment"]][2]
    net = case["legacy"].network
    mecha_index = g.property("mecha_index")
    index = np.array([int(mecha_index[int(v)]) for v in data_structure.entity_ids("node")[nodes]])
    index[geo.junction_order] = net.n_walls + np.arange(net.n_junctions)
    view = data_structure.to_graph_view()
    local = np.full(data_structure.n_nodes(), -1, dtype=np.int64)
    local[nodes] = np.arange(nodes.size)
    membranes = edges[np.asarray(data_structure.get("network_path"))[edges] == MEMBRANE]
    by_pair = {}
    for e in membranes:
        a, b = int(index[local[view.tail[e]]]), int(index[local[view.head[e]]])
        by_pair[(min(a, b), max(a, b))] = e
    edge_of = np.array([by_pair[(w, c)] for w, c in legacy["membranes"]])
    mecha_net = legacy["mecha"].network
    rank_to_row = mecha_net.rank_to_row
    counted = np.array([(not np.isnan(mecha_net.cell_ranks[c - mecha_net.n_wall_junction])
                         and int(mecha_net.cell_ranks[c - mecha_net.n_wall_junction]) in rank_to_row)
                        for _, c in legacy["membranes"]])
    tail, head = view.tail[edge_of], view.head[edge_of]
    compartment_type = np.asarray(data_structure.get("compartment_type"))
    wall_is_tail = compartment_type[tail] != 0
    walls, cells = np.where(wall_is_tail, tail, head), np.where(wall_is_tail, head, tail)
    rows_with_spaces = membrane_rows(geo, barrier, local[walls], local[cells], intercellular=False)
    yield dict(data_structure=data_structure, legacy=legacy, edge_of=edge_of, counted=counted, rows_with_spaces=rows_with_spaces,
               segment=case["segment"], nodes=nodes, index=index)


def test_fractions_per_membrane(both):
    """Each membrane the legacy counts: its fraction, to 1e-10 (the standard flows' 1e-12 of the potentials over the
    uptake); the port's 0 on the intercellular spaces' (ME-41)."""
    fraction = np.asarray(both["data_structure"].get("standard_transmembrane_fraction"))[both["edge_of"]]
    layer = np.asarray(both["data_structure"].get("membrane_layer"))[both["edge_of"]]
    legacy = both["legacy"]["stf_mb"]
    kept = both["counted"] & (layer >= 0)
    assert kept.sum() > 0
    np.testing.assert_allclose(fraction[kept], legacy[kept], rtol=0, atol=1e-10 * abs(legacy).max())
    assert (fraction[layer < 0] == 0).all()


def test_rows(both):
    """The legacy's layer sums from its own fractions, on the port's rows (the intercellular spaces' kept)."""
    legacy, rows, counted = both["legacy"], both["rows_with_spaces"], both["counted"]
    stf = legacy["stf_mb"]
    n = legacy["stf_layer_plus"].size
    on = counted & (rows >= 0)
    plus = np.bincount(rows[on], weights=np.maximum(stf[on], 0.), minlength=n)
    minus = np.bincount(rows[on], weights=np.minimum(stf[on], 0.), minlength=n)
    np.testing.assert_allclose(plus, legacy["stf_layer_plus"], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(minus, legacy["stf_layer_minus"], rtol=1e-12, atol=1e-15)


def test_layers_and_cells(both):
    """The layers' and the cells' sums of the port's fractions: equal, and 0 in all (every cell balances the water its
    membranes take and give; the mature vessels' membranes are excluded); the flows' sums, the fractions' × the
    uptake."""
    data_structure = both["data_structure"]
    layers = transmembrane_layers(data_structure, both["segment"])
    cells = transmembrane_cells(data_structure, both["segment"])
    assert layers["fraction_plus"].sum() > 0.1
    assert cells["fraction_plus"].sum() == pytest.approx(layers["fraction_plus"].sum(), rel=1e-12)
    assert cells["fraction_minus"].sum() == pytest.approx(layers["fraction_minus"].sum(), rel=1e-12)
    assert layers["fraction_plus"].sum() + layers["fraction_minus"].sum() == pytest.approx(0., abs=1e-9)
    uptake = np.asarray(data_structure.get("standard_uptake"))[0]
    assert layers["uptake_plus"].sum() == pytest.approx(layers["fraction_plus"].sum() * uptake, rel=1e-12)
