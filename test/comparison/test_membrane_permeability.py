"""
The membranes' permeability to the solute from their structure (PR #67's ``membrane_pd``; the port's
membrane_lipid_permeability and membrane_aquaporin_permeability, doc/legacy.rst, d20): P = P_lipid + P_aqp × the
cell's aquaporin factor, in series with the suberin lamella of a suberized face, × the membrane's surface, in place
of the diffusivity d_membrane. One Scharfetter–Gummel step on a root section's standard flows at barrier 3, the
aquaporin factors by tissue, without and with the lamella model: every node's sucrose the legacy's to 1e-11 of the
range; with the legacy's surface (membrane_surface_rule 2: length × (height + distance), none on an intercellular
cell, U6), and the port's wall half (lamella_wall_half 1: kw on the suberized faces, the radial ones too, as the
legacy's; the section has no passage cell).
"""
import numpy as np
import pytest

from openalea.mecha.component import MechaHydraulics
from openalea.mecha.geometry import section_parts
from openalea.mecha.solutes import MechaSolutes

from legacy_run import legacy_options, legacy_section, legacy_standard_solute_flux
from plants import template_organ

D_APO, D_PD, D_MEM, DT = 0.1, 0.5, 1e-6, 1. / 24.                     # cm2 d-1, d
P_LIPID, P_AQUAPORIN = 2.0E-3, 5.0E-3                                 # cm d-1
PULSE = 50e-6                                                         # mol cm-3, in the cortex cells
BARRIER = 3
AQUAPORINS = {"cortex": 0.5, "endodermis": 2.0, "stele": 0.7, "exodermis": 1.3}
LAMELLA = dict(kw_suberin_in=3.0E-9, kw_suberin_out=5.0E-9, kw_suberin_radial=7.0E-9, secondary_wall_thickness=0.5,
               tertiary_wall_thickness=1.0)
LEGACY_LAMELLA = dict(kw_barrier_elems=[{"value": 1.0E-16, "Casp": 1.0E-16, "Sub": 7.0E-9, "Sub_in": 3.0E-9,
                                         "Sub_out": 5.0E-9, "Lig": 1.0E-16}],
                      sec_wall_thickness=0.5, tert_wall_thickness=1.0)
SOLUTES = dict(d_apoplast=D_APO, d_plasmodesmata=D_PD, d_membrane=D_MEM, solute_flows=2., solute_boundary_advection=0.)
RESET = dict(solute_flows=1., solute_boundary_advection=1., membrane_lipid_permeability=float("nan"),
             membrane_aquaporin_permeability=float("nan"))


@pytest.mark.parametrize("lamella", [False, True])
def test_one_step_with_the_membranes_permeabilities(case, lamella):
    data_structure, g = case["data_structure"], case["mtg"]
    if not data_structure.has("barrier"):
        data_structure.register("barrier", np.ones(data_structure.n_nodes()), location="node")
    mecha = legacy_options(MechaHydraulics(data_structure=data_structure))
    solutes = legacy_options(MechaSolutes(data_structure=data_structure, time_step=DT * 86400.))
    nodes, _ = section_parts(data_structure)[case["segment"]]
    geo = mecha.section_geometries()[case["segment"]][2]
    run = legacy_section(template_organ(case["anatomy"], g, case["segment"]), barriers=(BARRIER,),
                         height=case["height"],
                         hydraulic=dict(LEGACY_LAMELLA, aqp_factor_by_type=AQUAPORINS,
                                        suberin_model="lamella" if lamella else "none"))
    mecha_index = g.property("mecha_index")
    index = np.array([int(mecha_index[int(v)]) for v in data_structure.entity_ids("node")[nodes]])
    index[geo.junction_order] = run.network.n_walls + np.arange(run.network.n_junctions)
    c0 = np.where(np.asarray(data_structure.get("cgroup"))[nodes] == 4, PULSE, 0.)
    c0 = np.where((geo.compartment_type != 0) & (geo.x < geo.x[geo.cells].mean()), PULSE, c0)   # and the left walls
    start = np.zeros(data_structure.n_nodes())
    start[nodes] = c0
    tissue = np.asarray(data_structure.get("tissue"), dtype=object)
    hydraulics = dict(LAMELLA, suberin_model=1. if lamella else 0., lamella_wall_half=1.)     # the port's: R4
    defaults = {name: np.array(getattr(mecha, name), copy=True) for name in hydraulics}

    def step(**permeabilities):
        for name, value in dict(SOLUTES, **permeabilities).items():
            setattr(solutes, name, value)
        data_structure.set("sucrose", start.copy())
        solutes()
        return np.asarray(data_structure.get("sucrose"))[nodes].copy()

    try:                                     # the parameters and inputs are the shared case's: restored for the others
        for name, value in hydraulics.items():
            setattr(mecha, name, value)
        data_structure.set("aquaporin_factor", np.array([AQUAPORINS.get(t, 1.) for t in tissue]))
        data_structure.set("barrier", np.full(data_structure.n_nodes(), float(BARRIER)))
        mecha()
        resistance = np.asarray(data_structure.get("lamella_diffusion_resistance")).copy()
        by_diffusivity = step()
        sucrose = step(membrane_lipid_permeability=P_LIPID, membrane_aquaporin_permeability=P_AQUAPORIN)
        sink = np.asarray(data_structure.get("sucrose_sink"))[nodes] > 0
    finally:
        for name, value in dict(defaults, **RESET).items():
            setattr(mecha if name in defaults else solutes, name, value)
        data_structure.set("aquaporin_factor", np.ones(data_structure.n_nodes()))
        data_structure.set("barrier", np.ones(data_structure.n_nodes()))
        for name in ("sucrose", "sucrose_osmotic_potential"):       # the cells' −R T c too: the osmotic presets add it
            data_structure.set(name, np.zeros(data_structure.n_nodes()))
        mecha()
    c_prev = np.zeros(index.size)
    c_prev[index] = np.where(sink, 0., c0)
    reference = np.asarray(legacy_standard_solute_flux(
        run, c_prev, {int(i): 0.0 for i in index[sink]}, d_apo=D_APO, d_pd=D_PD, d_mem=D_MEM, dt=DT,
        conserve_boundary=False, membrane_pd=dict(lipid=P_LIPID, aqp=P_AQUAPORIN)))[index]
    assert sink.sum() > 0 and (resistance > 0).any() == lamella
    assert abs(sucrose - by_diffusivity).max() > 1e-3 * PULSE            # the permeabilities' path is another one
    np.testing.assert_allclose(sucrose, reference, rtol=0, atol=1e-11 * abs(reference).max())
