"""
A section partly in contact with the soil by its cells (MECHA's Contactrange, the hormones file's ``contact``): the
border walls of the listed protoplasts, and the border junctions whose cells all are listed (develop's
``_contact_nodes``) or any is (the reactions branch's ``_apply_soil_boundary``). The port's soil_contact_cell and
contact_junction_rule give the legacy's soil terms bit for bit and its solution to 1e-12, on the left half of the
epidermis with x_contact beyond the section.
"""
import numpy as np
import pytest

from openalea.mecha import legacy, legacy_reactions
from openalea.mecha.component import MechaHydraulics
from openalea.mecha.geometry import section_parts

from legacy_run import legacy_options, legacy_section
from plants import template_organ

BARRIER = 1
FAR = 1.0E9                                       # x_contact beyond every wall: the cells' contact only


@pytest.mark.parametrize("rule, package", [(1, legacy), (2, legacy_reactions)], ids=["develop", "reactions"])
def test_contact_by_cells(case, rule, package):
    ds, g = case["ds"], case["g"]
    if not ds.has("barrier"):
        ds.register("barrier", np.ones(ds.n_nodes()), location="node")
    ds.set("barrier", np.full(ds.n_nodes(), float(BARRIER)))
    nodes, _ = section_parts(ds)[case["segment"]]
    mecha = legacy_options(MechaHydraulics(data_structure=ds))
    geo = mecha.section_geometries()[case["segment"]][2]
    epidermis = geo.cells[geo.cgroup[geo.cells] == 2]
    left = epidermis[geo.x[epidermis] < np.median(geo.x[epidermis])]
    mecha_index = g.property("mecha_index")
    node_ids = ds.entity_ids("node")
    run_network = case["legacy"].network
    contact_ids = [int(mecha_index[int(node_ids[nodes[c]])]) - run_network.n_wall_junction for c in left]
    flags = np.zeros(ds.n_nodes())
    flags[nodes[left]] = 1.
    ds.set("soil_contact_cell", flags)
    try:
        mecha.x_contact, mecha.contact_junction_rule = FAR, float(rule)
        mecha()
        soil, psi = np.asarray(ds.get("soil_conductance")).copy(), np.asarray(ds.get("standard_psi")).copy()
    finally:
        ds.set("soil_contact_cell", np.zeros(ds.n_nodes()))
        mecha.x_contact, mecha.contact_junction_rule = -15E9, 1.
    run = legacy_section(template_organ(case["anatomy"], g, case["segment"]), barriers=(BARRIER,),
                         height=case["height"], hydraulic=dict(xcontactrange=[FAR]),
                         hormones=dict(contact=contact_ids), package=package)
    index = np.array([int(mecha_index[int(v)]) for v in node_ids[nodes]])
    graph = run.network.graph
    by_position = {tuple(graph.nodes[j]["position"]): j for j in range(run.network.n_walls, run.network.n_wall_junction)}
    for v in geo.junctions:
        index[v] = by_position[(geo.x[v], geo.y[v])]
    stage = run.stages[0]
    in_contact = soil[nodes] > 0
    assert 0 < in_contact.sum() < (geo.border_wall | geo.border_junction).sum()      # partly in contact
    reference_soil = np.zeros(nodes.size)
    border = geo.border_wall | geo.border_junction
    reference_soil[border] = -np.asarray(stage["rhs"]).reshape(-1)[index[border]]                # the legacy's rhs_s: −K of the soil
    np.testing.assert_array_equal(soil[nodes], reference_soil)
    reference = stage["solution"][index]
    np.testing.assert_allclose(psi[nodes], reference, rtol=0, atol=1e-12 * abs(reference).max())
