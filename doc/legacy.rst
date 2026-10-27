=================================
The legacy: algorithms and quirks
=================================

Two versions of MECHA are kept, frozen, beside the port, as its references:

- ``openalea.mecha.legacy``: the refactored MECHA, the ``develop`` branch at 945b2be (2026-10-26; the port started
  from the ``to_metafspm`` branch at 2b995b2), with its tests in ``test/legacy/`` and its tutorials in
  ``example/legacy/`` (inputs, boundary conditions, diffusion, capacitance and reaction, a tomato root). One repair:
  the header of ``HydraulicMatrixBuilder._contact_nodes``, which ``develop``'s merge had turned into a second
  ``_apply_soil_boundary`` (every solve raised AttributeError);
- ``openalea.mecha.legacy_reactions``: the ``reactions`` branch at 4c9c8c2 (solute transport, needle transpiration,
  calibration), with its seven simulation scripts in ``example/legacy_reactions/simulations/``.

Only their imports are rewritten. The port is compared with them in ``test/comparison/`` and ``example/comparison/``.
Where the refactored legacy departs from the original MECHA (the MECHA v4 script, ``6_MECHAv4_SRS2025.py``, with its
GRANAR project inputs), the original is the reference for what is fixed. This page maps the legacy to the port,
describes the algorithms as the port implements them, then registers every legacy behaviour found during the port
with what the port does. The equations are in :doc:`equations`, the timings in :doc:`performance`.


The legacy's chain and the port's modules
=========================================

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - step
     - legacy
     - port
   * - inputs: hydraulics, geometry, boundary conditions, scenarios
     - ``InData`` (``data_loader``, XML files or defaults)
     - per-plant parameters of the components (``MechaHydraulics``, ``MechaOsmoticPresets``, ``MechaSolutes``), set by
       the plants' scenarios
   * - the section's network
     - ``NetworkBuilder`` (``network_builder``): from a GRANAR cellset XML (``build_network``) or a GRANAP organ
       (``populate_from_network``, through its networkx graph)
     - GRANAP's anatomy component writes the network on the DataStructure; ``geometry.section_parts``,
       ``geometry.section_of``. Cellset XML files are read by GRANAP's ``CellSetOrgan``
   * - the derived geometry (junctions, ranks, centre, perimeter, plasmodesmata ratios, cell sets)
     - ``NetworkBuilder.populate_from_network``, ``rank_cells``, ``compute_cell_surface``
     - ``geometry.section_geometry`` (``_ranks``, ``_plasmodesmata_ratios``), cached per anatomy template
       (``MechaSections.section_geometries``)
   * - one Python object per wall, cell, membrane, plasmodesma
     - ``HydraulicCellManager`` (``hydraulic_cell``)
     - arrays: ``conductance.Network`` (``assemble``)
   * - the conductances and boundary terms
     - ``HydraulicMatrixBuilder`` (``hydraulic_solver``: ``_fill_wall``, ``_fill_membrane``, ``_fill_plasmodesmata``,
       ``_apply_soil_boundary``, ``_apply_xylo_phloem_boundary``); ``Mecha._calculate_cortex_parameters``
     - ``conductance.edge_conductances``, ``boundary_terms``, ``cortex_parameters``; the step
       ``MechaHydraulics._conductance``
   * - the solve
     - the Doussan matrix assembled from the networkx adjacency, ``scipy.sparse.linalg.spsolve``, one per maturity
       stage (``solve_W``) and scenario (``water_flux``)
     - two metafspm graph systems: ``_standard_problem`` (each section alone) and ``_plant_problem`` (the sections
       joined); ``conductance.section_matrix`` builds the legacy's matrix for the comparison only
   * - kr, Kx, the edge flows
     - ``Mecha.standard_water_flow``, ``calculate_axial_conductance``, ``compute_edge_flows``
     - ``postprocess.soil_uptake``, ``radial_conductivity``, ``axial_conductances``; the step ``_kr``; the graph
       outputs ``standard_flow`` and ``flow``
   * - the osmotic scenarios
     - ``Mecha.initialize_scenarios`` (``os_hetero``, ``s_hetero``), ``_count_surrounding_cells``,
       ``NetworkBuilder.compute_relative_positions``, ``create_layer_discretization`` (the rows)
     - ``osmotic.MechaOsmoticPresets``: ``legacy_rows``, ``relative_positions``, ``wall_counts``, ``tissue_values``,
       ``section_presets``
   * - φ-thickenings
     - ``HydraulicCellManager.tag_phi_thick_cells``
     - ``geometry.phi_thick_cells``
   * - the flow condition at the xylem
     - ``Mecha._distribute_xylem_flow``
     - ``MechaHydraulics._standard_axial_inflow``
   * - solute transport (``reactions`` branch)
     - ``SoluteTransport``, ``MultiSoluteTransport`` (``solute_transport``); ``HydraulicMatrixBuilder`` in its
       ``c_flag`` mode
     - ``solutes.MechaSolutes``; the diffusion factors in ``edge_conductances(factors=...)``
   * - needle transpiration (``reactions`` branch)
     - ``wall_air`` and ``air_link`` edges, ``set_air_wall_bc`` (pinned evaporating walls)
     - ``conductance.vapour_conductances``: Kelvin walls, air links, stomata to a leaf-surface node
   * - plots and exports
     - ``visu``, ``network_export``, ``paraview_export``, ``prepare_paraview``
     - ``openalea.mecha.plots`` (tissues, ψ maps, radial profiles, conductances by pathway, pathway shares, flow
       vectors); no file export
   * - the steady coupled water and solutes (``reactions`` branch)
     - ``coupled_solver.coupled_water_solute_solve`` (Picard, JFNK, continuation)
     - MechaSolutes' ``solute_solve`` 2: one Newton system (``_steady_coupled``), its Jacobian analytic
   * - not ported
     - ``scenario_builder`` (XML scenarios: the port's scenarios are its plants' parameters), ``calibration``
       (fitting the transport coefficients on needle anatomies: a tool on top of the model, its fits rerunnable with
       the port's components), the legacy's ``SoluteTransport`` itself
     - —

The legacy's tutorials are in ``example/legacy/`` (with its inputs); ``example/comparison/tutorial_01_input.py``
and ``tutorial_02_bc.py`` run their sections through the legacy and the port, ``compare_solutes.py`` and
``compare_solutes_day.py`` the solute transport. The port's tutorials are in ``example/tutorials/``: the user series,
and the developer series, whose tutorials 07–09 go through every option of the coverage table below.


Geometry
========

**Units.** GRANAP's network is in mm; MECHA's in µm (lengths × 1000, areas × 10⁶). The legacy numbers its nodes walls,
then junctions, then cells (``SectionGeometry.legacy_order``); the port keeps the DataStructure's order and reproduces
the legacy's orders wherever a sum depends on them.

**Junctions.** The legacy deletes GRANAP's junctions and rebuilds them from the walls' positions (rounded to 6
decimals). On GRANAP's sections this is a bijection with the same walls on each junction, so the port keeps GRANAP's
junctions, rounds their positions to 6 decimals, drops the junction edges of a wall with fewer than two junctions (the
rebuild's rule), and reproduces the rebuilt junction ids (``junction_order``: by first appearance, walls in order, each
wall's junctions in neighbour order).

**Border walls and junctions.** A wall with one membrane, to an epidermis cell, is a border wall (the soil interface);
to another cell, an aerenchyma border. A junction between two border walls is a border junction, of length the sum of
its walls' quarter lengths.

**Lateral distance** of a wall: the sum of its membranes' distances. **Gravity centre**: the mean of the endodermis
cells' positions, summed in cell order (of every cell where there is no endodermis: the port's extension to stems and
leaves). **Distances to the centre**: per wall, with scalar arithmetic (numpy's scalar ``x ** 2`` calls libm's
``pow``, which can differ by one unit in the last place from the arrays' ``x*x``). **Perimeter**:
2π × the epidermis walls' mean distance, in wall order.

**Ranks** (``_ranks``, the legacy's ``_rank_cells_from_graph``). Each cell starts at its cgroup (19, 20 as xylem 13; 21
as pericycle 16; 26 as companion 12). The cortex cells next to the endodermis become 40, those next to the outer layer
(the exodermis, or the epidermis without one) 49, the stele cells next to the pericycle (the endodermis without one)
50. Then 12 sweeps: in the first four, the cortex cells next to layer 40 + i become 41 + i and those next to 49 − i
become 48 − i; in the first ten, the stele cells next to 50 + i become 51 + i; the last two put the remaining stele
cells in 61. The legacy updates in place, but within one sweep no cell's test reads a rank given in the same sweep, so
each sweep is computed at once. Each layer's mean distance to the centre is summed in the legacy's pass order.

**Plasmodesmata ratios** (``_plasmodesmata_ratios``, the legacy's ``compute_cell_surface``): at the outer
layer–cortex, cortex–cortex and cortex–endodermis interfaces, the plasmodesmata length from non-intercellular cells
over the part whose other cell is non-intercellular too, summed over the cells in cell order and their neighbours in the
order networkx holds them after ``Graph.copy()`` (earlier nodes first by node order, then later ones in insertion
order).

**Cell sets.** Two sets coexist in the legacy: the network's (by cgroup, an elif chain: xylem, sieve, protosieve,
passage, intercellular) and the cell manager's (by tissue name). The port keeps both (``xylem_cells`` …,
``typed``), each where the legacy uses it.


Conductances
============

**Walls** (``_fill_wall``). The interface of a wall's two cells gives its ``kw`` key; the barrier code gives each key's
value:

.. list-table:: The wall conductivity of each interface, by barrier code
   :header-rows: 1
   :widths: 8 13 12 12 12 11 13 13 11

   * - code
     - endodermis
     - exodermis
     - exodermis–epidermis
     - exodermis–cortex
     - cortex
     - endodermis–stele
     - endodermis–cortex
     - passage cell
   * - 0
     - kw
     - kw
     - kw
     - kw
     - kw
     - kw
     - kw
     - kw
   * - 1
     - casparian
     - kw
     - kw
     - kw
     - kw
     - kw
     - kw
     - kw
   * - 2
     - casparian
     - kw
     - kw
     - kw
     - kw
     - suberin in
     - suberin out
     - kw
   * - 3
     - casparian
     - kw
     - kw
     - kw
     - kw
     - suberin in
     - suberin out
     - suberin in
   * - 4
     - casparian
     - casparian
     - kw
     - kw
     - kw
     - suberin in
     - suberin out
     - suberin in
   * - 5
     - casparian
     - casparian
     - kw
     - kw
     - kw
     - kw
     - kw
     - kw
   * - 6
     - casparian
     - casparian
     - suberin out
     - suberin in
     - kw
     - kw
     - kw
     - kw
   * - 7
     - kw
     - casparian
     - kw
     - kw
     - kw
     - kw
     - kw
     - kw
   * - 8
     - casparian
     - casparian
     - suberin out
     - suberin in
     - kw
     - suberin in
     - suberin out
     - kw
   * - 9
     - casparian
     - casparian
     - lignin
     - kw
     - kw
     - kw
     - kw
     - kw

(``kw_casparian``, ``kw_suberin_in``, ``kw_suberin_out``, ``kw_lignin``; a code outside the table is read as 0.)
Then, in order: ``kw_lignin`` between two φ-thickened cells of the same rank; ``kw_passage`` next to a passage cell;
``kw_septa`` for a ghost wall (two intercellular cells at barrier > 0, or two xylem cells with ``xylem_pieces``), and
10⁻¹⁶ for a ghost wall between xylem pieces. The geometric factor's ``t ** 2`` is computed as the legacy's Python
scalars do (``_pow2``: libm's ``pow`` per distinct value).

**Membranes** (``_fill_membrane``). The aquaporin activity by the cell (an ``np.select`` in the legacy's order:
exodermis, epidermis, endodermis, mature xylem × 10⁴, immature xylem, stele, intercellular at barrier > 0, cortex
gradient), the wall's ``kw`` (``kw_passage`` at a passage cell's endodermis wall), the half wall in series, and the
10⁻¹⁶ floor where the aquaporin activity is 0 outside a barrier interface.

**Plasmodesmata.** Two rule sets (``plasmodesmata_rules``): the original MECHA's (1, default) and the legacy's (2), see
the registry (a, b and their rows) and :doc:`equations`.

**Boundary terms** (``_apply_soil_boundary``, ``_apply_xylo_phloem_boundary``). The soil's on the border walls and
junctions in contact (``x_contact``), the axial boundary's on the xylem vessels (barrier > 0) or the protophloem cells
(barrier 0).


The solve and its results
=========================

The legacy assembles the Doussan matrix (−ΣK on the diagonal, K off it, minus the boundary terms on the diagonal) and
solves it with ``spsolve``, one maturity stage (barrier code, height) at a time. The port solves the same balance as a
metafspm graph system, each section a SubOrgan with its own barrier and its length as the height, so the legacy's
stages are sections. Its ``_standard_problem`` reproduces the legacy's per-section problem (scenario 0; with
``standard_osmotic`` 1, its osmotic scenarios), and ``_plant_problem`` joins the sections through Poiseuille
conductances on the axial junctions, with the soil from a soil model, a collar condition and the leaves' transpiration.

**kr** sums the border walls' then the border junctions' soil flows, in the legacy's id order; the legacy's ``sum`` is
numpy's pairwise sum (``mecha_class`` star-imports pylab), and the port uses numpy's on arrays laid out as the legacy's.
**Kx** sums the vessels' Poiseuille conductances in cell order, the same way.

**Exactness.** Before the solve everything is bit-exact (geometry, every edge's K, every boundary term, Kx, and kr and
the flows on the legacy's own solution). The assembled matrix's diagonal sums in the legacy's networkx order and its
node order differs, so the matrix is compared to 4 units in the last place, and ψ, the flows and kr from the port's
solve to a relative 10⁻¹² (10⁻¹¹ of the range for the flow condition, where only the soil anchors the potentials).
The comparison sets the legacy's choices through ``legacy_options`` (``test/comparison/legacy_run.py``):
``plasmodesmata_rules`` 2, ``kpl_PCC_factor`` 0, ``kpl_PST_factor`` 0, ``standard_axial_source`` 2,
``thickness_source`` 1; the osmotic comparison adds ``presets_rules`` 2, the needle's ``presets_sections`` 2.


The osmotic scenarios
=====================

The legacy's scenario 0 is the standard problem; each further scenario solves the same section again with the osmotic
terms on the membranes, the soil's left and right potentials, and, at barrier 0, the elongation sink. The port's
presets (``MechaOsmoticPresets``) reproduce its π and σ rules: the layer rows (``legacy_rows``, transcribed: stele
rows from rank 61 down, the pericycle, four endodermis rows, a cortex layer with fewer than 0.75 of the previous row's cells
merged with a neighbour, two exodermis rows, the epidermis), each wall's radial position relative to the
endodermis and its relative x (``relative_positions``), the wall's membranes counted by kind (``wall_counts``), the
tissues' values (``tissue_values``), and per membrane, in wall order, the cell's and the wall's π and the membrane's σ
(``section_presets``; a cell's π is the one its last membrane sets).


The solutes (``reactions`` branch)
==================================

The reactions branch solves one or several species on every node, ``C dc/dt = (D + A − R) c + s``: diffusion along the
walls (scaled by their kw ratio), through the plasmodesmata and across the membranes; advection by the water's flows
with the Scharfetter–Gummel scheme; a reaction matrix between species; a θ-method in time with Dirichlet rows; a
lagged water solve per step with each cell's π = baseline − R T c. The port keeps its diffusion conductances, volumes,
Scharfetter–Gummel kernel and sinks (compared exactly: diffusion to 10⁻¹⁵ relative, volumes exact, one step to
1.9 × 10⁻¹² of the range), with implicit Euler (θ = 1, the only value its scenarios use) as a metafspm graph system on
the liquid nodes, after the fixes below.


Coverage
========

Every input, option, boundary condition, solver path and output of the two legacies (2026-10-28, ``develop`` at
945b2be and ``reactions`` at 4c9c8c2, module by module: their loaders' XML keys, ``Mecha``'s, the matrix builder's,
the solute transport's and the coupled solver's branches), with the port's equivalent and the test comparing them.
Each legacy problem met on the way is in the registry below, with what the port does.

.. list-table::
   :header-rows: 1
   :widths: 34 38 28

   * - legacy
     - port
     - compared in
   * - ``Geometry``: the maturity stages (barrier codes 0–11, height), the wall thickness, ``Xylem_pieces``,
       ``kInterC``, ``PD_section``; ``passage_cell_range``, ``aerenchyma_range`` (cell ids)
     - ``barrier`` and ``length`` per segment; ``thickness_source``, ``thickness``, ``xylem_pieces``, ``k_interc``,
       ``pd_section``; passage cells and air spaces by their tissue (GRANAP's labels on the DataStructure)
     - ``test_develop_barriers.py`` (every code, bit for bit), ``test_hydraulic_options.py``, ``test_conductance.py``
   * - ``Hydraulics``: ``kw``, ``kw_septa``, the barriers' ``kw_barrier`` (Casparian, suberin in and out, lignin, φ),
       ``km``, ``kAQP`` and its tissue factors, ``ratio_cortex``, ``phi_thick``, ``Kpl`` and its factors, the
       frequencies (``Fplxheight``), ``Xcontact``, ``K_xyl``, ``K_sieve``, ``Kax_source``
     - ``kw``, ``kw_septa``, ``kw_casparian``, ``kw_suberin_in``, ``kw_suberin_out``, ``kw_lignin``, ``kmb``,
       ``kaqp``, ``kaqp_*_factor``, ``ratio_cortex``, ``n_phi_layers``, ``phi_type``, ``kpl``, ``kpl_*_factor``,
       ``plasmodesmata_rules``, ``fplxheight_*``, ``x_contact``, ``k_xyl``, ``k_sieve``, ``standard_axial_source``,
       ``measured_axial_conductivity``
     - ``test_conductance.py``, ``test_hydraulic_options.py``, ``test_plasmodesmata_factors.py``, ``test_postprocess.py``
   * - ``develop``'s hydraulics: the cambium, barriers 10 and 11 (``kw_scler``), ``tracheid_as_xylem``,
       ``pd_spatial_config``, ``membrane floor``
     - the same (d3–d6), ``membrane_floor``, ``pd_spatial_*``, ``contact_distance``
     - ``test_develop_barriers.py``
   * - the soil contact by cells (``Contactrange``)
     - ``soil_contact_cell``, ``contact_junction_rule`` (d10)
     - ``test_soil_contact_cells.py`` (both legacies)
   * - ``BCs``: the soil's left and right potentials, the xylem's pressure or flow, the phloem's pressure or flow,
       the elongation (midpoint rate, side difference, water fractions); the osmotic scenarios (``Os_hetero``,
       ``s_hetero``, ``s_factor``, ``Os_cortex``, ``osmotic_sieve``, the soil's and the xylem's solutions: symmetry,
       shape, diffusivity)
     - ``standard_psi_soil``, ``standard_psi_soil_right``, ``standard_psi_xylem``, ``standard_axial_condition``,
       ``standard_axial_flow``, ``standard_psi_sieve``, ``standard_psi_sieve_tube``, ``standard_sieve_flow``,
       ``standard_elongation``, ``elongation_*``, ``water_fraction_*``, ``standard_osmotic``; MechaOsmoticPresets'
       parameters of the same names; the plant problem's ``collar_*`` and ``soil_psi``
     - ``test_scenario_boundaries.py``, ``test_boundary_conditions.py``, ``test_osmotic.py``
   * - the solve (``solve_W``, ``water_flux``), kr, Kx, the edge flows, the transmembrane fractions by membrane,
       cell and layer, the axial flows by layer
     - the standard problem (each section) and the plant problem (the sections joined); ``kr``, ``Kx``,
       ``Kx_phloem``, ``standard_flow``, ``standard_transmembrane_fraction``, ``membrane_layer``,
       ``postprocess.transmembrane_layers``, ``transmembrane_cells``, ``axial_layers``
     - ``test_conductance.py``, ``test_postprocess.py``, ``test_transmembrane_fractions.py``
   * - needle transpiration (``reactions``): ``wall_air``, ``air_link``, the evaporating walls
     - the vapour path (``vapour_conductances``: Kelvin walls, air links, stomata), ``leaf_transpiration``,
       ``stomatal_aperture``
     - ``test_solutes.py`` (the needle), ``test_shoot.py``; r11, ME-35
   * - solutes (``SoluteTransport``, ``MultiSoluteTransport``): the diffusivities, σ, the modes ``full``, ``apo``,
       ``sym``, the schemes (upwind, Scharfetter–Gummel), θ, the storage fractions, ``k_deg``, the sucrose–starch
       reactions, the sources, ``conserve_boundary``, ``c_boundary``, the Dirichlet nodes, the steady state; the
       standard problem's flows (``standard_solute_flux``)
     - MechaSolutes: ``d_*``, the membranes' σ (the water's), ``solute_pathway``, ``solute_scheme``,
       ``solute_theta``, ``storage_fraction_*``, ``k_decay``, ``k_starch_*``, ``sucrose_source``,
       ``solute_boundary_advection``, ``c_boundary``, ``fixed_sucrose``, ``solute_sinks``, ``c_sink``,
       ``solute_solve`` 1 and 3, ``solute_flows``
     - ``test_solutes.py``, ``test_standard_solutes.py``, ``test_solute_boundary_exchange.py``
   * - the steady coupled water and solute (``coupled_water_solute_solve``: Picard, JFNK, continuation, relaxation;
       its time stepping)
     - ``solute_solve`` 2 (Newton); the time stepping: ``solute_solve`` 1, the sucrose's osmotic potential fed to
       the next water solve
     - ``test_coupled_steady.py``
   * - the views (``visu``: the anatomy, ψ, flows, pathways, the wet–dry transect, the contact distance)
     - ``openalea.mecha.plots``
     - ``test_plots_contact.py``

Not ported, each for its reason:

- **inputs and outputs**: the XML files themselves (the port's inputs are its parameters and the DataStructure's
  variables), ``Output`` paths, the paraview, graphml and pickle exports (``General``'s ``Paraview*``, ``par_track``,
  ``sym_contagion``, ``apo_contagion`` display flags; ME-29), the solver choice ``sparse_matrix`` (the graph systems are
  sparse), the debug prints (m);
- **dead in both legacies**: the hormones file's movement, active transport and contagion (``Hormone_movement``:
  ``Degradation_constant_H1``, ``Diffusivity_*_H1``, ``H1_D2O``; ``Hormone_active_transport``; ``Sym_Contagion``,
  ``Apo_Contagion``): the matrix builder's ``c_flag`` branch builds their terms under keys the loader never sets
  (``diff1_pw1``, ``diff1_pd1``), ``_initialize_apo_j_zombies0`` is empty, and nothing solves the matrix (develop's
  ``SoluteTransport`` empties the carriers and the sources before using the builder); the hydropatterning and
  hydrotropism arrays (never computed); the distal pressures and flows and the pressure differences (``pressure_*_dist``, ``delta_p_*``: only a
  consistency check, no 3D solve); the ``Geometry``'s ``cell_per_layer``, ``diffusion_length``, ``pile_up``,
  ``xwalls``, ``print_layer``, ``interc_perim_search`` (never read); the ``reactions`` operators ``'D'`` and ``'A'``
  alone (a diagnostic, given by the physical parameters: ME-37);
- **replaced**: the cellset XML builder and its perimeter search for intercellular spaces (``interc_perims``, ME-27:
  GRANAP's ``CellSetOrgan``), the combinatorial scenario sets (ME-28), the pinned evaporating walls (ME-35);
- **tools on top of the model**: the scenario builder (F10) and the calibration package (F9).

Every registered problem is fixed in the port's defaults; the options that reproduce one (``plasmodesmata_rules`` 2,
``thickness_source`` 1, ``standard_axial_source`` 2, ``membrane_floor`` 2, ``presets_rules`` 2,
``solute_boundary_advection`` 0, ``contact_junction_rule`` 2) are for the comparisons. Kept on purpose: the
original MECHA's own choices (ME-17, ME-21), the radial profiles' flat convective form (ME-24, the original's
velocity never computed), the needle's root presets (r7) and ``kw_meso_hypo``'s interface (r9).


The registry
============

Each behaviour of the legacy found during the port, with what the port does: **reproduced** (the port computes the
same, by default or with the named option), **corrected** by the named option (its default is the correction; the
legacy's value stays available for the comparison), **fixed** (in the port whatever the options), **kept**
(deliberately kept), **dropped** or **not ported** (input files, dead code, features no scenario uses). The IDs a–q are
the legacy's problems as first recorded, r1–r12 the reactions branch's; ME-NN are the others.

Parameters silently not applied
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 8 52 40

   * - ID
     - legacy
     - port
   * - a
     - the plasmodesmata factors' keys (``cortex_factor``, ``endo_in_factor`` …) do not match the configuration's
       (``cortex``, ``endo_in`` …): only the companion-cell and sieve-tube factors apply, the others fall back to 1; a
       duplicate ``(11, 11)`` key
     - corrected: ``plasmodesmata_rules`` 1 (the original's rules and factors, ``kpl_cortex_factor``,
       ``kpl_endo_in_factor``, ``kpl_endo_out_factor``, ``kpl_PPP_factor``, ``kpl_PCC_factor``); fixed upstream by
       ``develop`` (d2), whose map ``plasmodesmata_rules`` 2 reproduces (``test_plasmodesmata_factors.py``)
   * - b
     - the ``Fplxheight`` values of the XML file are never applied (the interface map is filled after loading)
     - dropped (no XML input); every frequency is a parameter (``fplxheight_*``)
   * - c
     - the flow boundary conditions are never applied (``flowrate_prox`` read, ``flow_xyl_prox`` stored)
     - dropped (no XML input); the port's flow condition is ``standard_axial_condition`` 2 with
       ``standard_axial_flow``
   * - d
     - GRANAP's per-wall thickness ignored: one thickness (1.5 µm) everywhere
     - corrected: ``thickness_source`` 2 (each wall's own, a junction the mean of its walls'); 1 reproduces. kr +3.6 %
       (monocot root) and +8.7 % (dicot root) at barrier 1, +1.0 % and +3.4 % at barrier 3. The solutes'
       plasmodesmata diffusion also divides by the crossed wall's own thickness (the reactions branch: the one
       value; ``thickness_source`` 1 reproduces it)

Crashes and wrong values
------------------------

.. list-table::
   :header-rows: 1
   :widths: 8 52 40

   * - ID
     - legacy
     - port
   * - e
     - ``calculate_axial_conductance`` fails (an unbound variable) at barrier 0, so ``compute_conductivities``
       crashes on any stage without a barrier
     - fixed: Kx = 0 at barrier 0 (no mature xylem), and ``Kx_phloem`` the protophloem's Poiseuille sum
   * - f
     - the protophloem cells listed twice on the GRANAP path: the phloem boundary term added twice for them
     - fixed: each protophloem cell once (``boundary_terms``); the geometry keeps the legacy's list for its comparison.
       Only matters at barrier 0 with a phloem potential given
   * - g
     - ``k_xyl`` and ``k_sieve`` are lists, used as scalars in places
     - dropped (scalar parameters)
   * - h
     - the scenario loop reads ``scenarios[i_maturity]`` for the elongation's side rate difference; the interface
       flows take the left soil potential even with a left–right profile
     - fixed: the plant's own parameters (no scenario loop); the uptake with each border node's soil potential
       (``test_scenario_boundaries.py``)
   * - i
     - GRANAP's ``wall_air`` and ``air_link`` edges (leaves, needles) ignored by the matrix
     - corrected: the leaves' vapour path (Kelvin exchange on ``wall_air``, vapour diffusion on ``air_link``, stomata to
       a leaf-surface node) in the plant problem
   * - j
     - the shipped ``extdata/Hydraulics.xml`` does not load (barrier attributes missing)
     - dropped (no XML input)
   * - o
     - the flow condition's shares are the cell manager's xylem area ratios (by tissue), given by rank to the
       network's vessels (by cgroup): in the monocot root a vessel gets another's share
     - fixed: each vessel (by cgroup) its own area fraction, and the xylem's potential recovered on the same vessels.
       On tutorial 02's flow case (monocot root, barrier 3): kr 4.028609 × 10⁻⁵ against 4.029255 × 10⁻⁵, ψ up to 177
       hPa (3.5 %) apart in the stele; the dicot root's two orders agree

Code health
-----------

.. list-table::
   :header-rows: 1
   :widths: 8 52 40

   * - ID
     - legacy
     - port
   * - k
     - ``tag_phi_thick_cells`` called with its arguments swapped (harmless through a guard)
     - fixed: ``phi_thick_cells`` called in order
   * - l
     - two sources for the same cell sets: by cgroup in the solver, by tissue in the class
     - reproduced where each is used (for exactness); the flow condition fixed (o)
   * - m
     - debug prints, dead code, results lists that grow at every call
     - dropped
   * - n
     - ``list_ghostwalls`` and ``list_ghostjunctions`` created empty, never filled: the ghost-wall bookkeeping is dead;
       the ghost branch only depends on the intercellular count and the barrier
     - dropped (the bookkeeping); the ghost rule reproduced (``kw_septa``, 10⁻¹⁶ between xylem pieces)
   * - ME-01
     - ``solve_W`` returns its matrix after removing the axial terms in place (``remove_xyl_phloem_BC``), added on the
       network's xylem (by cgroup) and removed on the cell manager's (by tissue)
     - the comparison emulates it on the legacy's matrix; the port never removes anything
   * - ME-02
     - the junctions rebuilt from the walls' positions (rounded to 6 decimals): only a renumbering of GRANAP's
     - kept GRANAP's junctions; the rounding, the dropped edges of a wall with fewer than two junctions and the rebuilt
       ids reproduced
   * - ME-03
     - one Python object per wall, cell, membrane and plasmodesma (``HydraulicCellManager``)
     - dropped (arrays)
   * - ME-04
     - the legacy's tests: no stored values; several without assertion, some blocking on ``plt.show()``, three
       ``assert True``; the README's usage does not run (wrong imports)
     - the frozen tests kept in ``test/legacy/``, not run (they call GRANAP interfaces since removed); the port's
       tests assert, headless; the README's imports fixed (``openalea.mecha.legacy``)

Numerics
--------

.. list-table::
   :header-rows: 1
   :widths: 8 52 40

   * - ID
     - legacy
     - port
   * - ME-05
     - summation orders: cell order (gravity centre, layers), wall order (epidermis distances), networkx's neighbour
       order after ``Graph.copy()`` (plasmodesmata ratios), numpy's pairwise ``sum`` through ``from pylab import *``
       (kr, Kx)
     - reproduced
   * - ME-06
     - scalar ``x ** 2`` (libm's ``pow``), one unit in the last place from the arrays' ``x*x``
     - reproduced (per scalar, ``_pow2``)
   * - ME-07
     - Poiseuille's law written with 3.141592 for π
     - reproduced, in every Poiseuille term (Kx, the axial boundary, the axial junctions)
   * - ME-08
     - the Doussan matrix solved by ``spsolve`` per stage and scenario
     - replaced by the graph systems (sparse analytic Jacobians, one Newton step); the matrix to 4 units in the last
       place, the solution to 10⁻¹²

Plasmodesmata, against the original MECHA
-----------------------------------------

The legacy's tables (``interface_kpl_factor_map``, ``fplxheight_map``) against the original's radial rules. All are
corrected by ``plasmodesmata_rules`` 1 (default) and reproduced by 2 (with ``kpl_PCC_factor`` = ``kpl_PST_factor`` =
0, the legacy's defaults without an XML file). The legacy column describes the refactored legacy before ``develop``;
``develop`` applies its factors (d2): the cortex's at cortex–cortex, the endodermis' at endodermis–stele,
endodermis–pericycle (inner) and endodermis–cortex (outer), PPP at stele–companion, companion–xylem and
companion–pericycle, PCC at companion–companion and companion–sieve, PST at sieve–sieve, sieve–pericycle,
sieve–xylem and stele–sieve, ``kpl_cambium_stele_factor`` at stele–cambium; still not where the original puts PPP
(the phloem-pole pericycle) and PCC (also stele and xylem–companion). Reviewed difference (standard problem): monocot root, kr +0.01 %
(stele–sieve opened); dicot root, kr −0.06 % at barrier 1 and −0.03 % at barrier 3.

.. list-table::
   :header-rows: 1
   :widths: 8 52 40

   * - ID
     - legacy
     - port (``plasmodesmata_rules`` 1)
   * - ME-09
     - ``cortex_factor`` never applied (the original: outer layer–cortex and cortex–endodermis as a harmonic mean with 1,
       cortex–cortex as is)
     - the original's, ``kpl_cortex_factor``
   * - ME-10
     - ``PPP_factor`` never applied (the original: the phloem-pole pericycle at its interfaces with the endodermis,
       stele, companion cells and sieve tubes)
     - the original's, ``kpl_PPP_factor``
   * - ME-11
     - ``PCC_factor`` only at companion–companion and companion–sieve (the original: also stele and xylem–companion)
     - the original's, ``kpl_PCC_factor``
   * - ME-12
     - ``PST_factor`` at sieve–pericycle, sieve–xylem, stele–sieve (the original: no radial interface)
     - not applied by the original's rules; ``kpl_PST_factor`` with the legacy's rules only
   * - ME-13
     - ``endo_in_factor`` and ``endo_out_factor`` never applied (the original: by the side of the endodermis)
     - the original's, ``kpl_endo_in_factor``, ``kpl_endo_out_factor``
   * - ME-14
     - the factors default to 0.0 without an XML file (the original's projects: 1.0): no plasmodesmata at
       companion–companion, companion–sieve and the sieve interfaces
     - every factor 1.0
   * - ME-15
     - frequencies: outer layer–cortex only from the exodermis (the original: the actual outer layer, the epidermis
       without an exodermis); stele–sieve and xylem–sieve 0 (the original: 9.0 × 10⁵)
     - the original's 15 frequencies (``fplxheight_default`` … ``fplxheight_stele_sieve``)
   * - ME-16
     - the high xylem–xylem conductance at every xylem–xylem interface (the original: only between the pieces of one
       vessel)
     - the original's (``xylem_pieces``; none to mature xylem otherwise)
   * - ME-17
     - the original reads ``Fplxheight_peri_peri`` but no radial rule uses it: pericycle–pericycle takes the default
       frequency
     - reproduced
   * - ME-18
     - interfaces added for transfusion tissue (17, 18) and protophloem (23), for needles, unknown to the original
     - kept, with the legacy's frequencies, as 18 parameters

The section's height and the standard problem's boundary
--------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 8 52 40

   * - ID
     - legacy
     - port
   * - ME-19
     - the axial boundary always ``k_xyl`` = 10⁻⁶ per vessel, whatever the height and the original's source (its
       default: Poiseuille through each vessel over the height)
     - corrected: ``standard_axial_source`` 1 (Poiseuille over the section's height, water's viscosity in the vessels,
       phloem sap's in the protophloem); 2 reproduces. kr at 200 µm: +3.6 % (monocot, barrier 1), +1.4 % (barrier 3),
       +3.0 % and +1.3 % (dicot); at 5 mm: +61 % and +53 % at barrier 1
   * - ME-20
     - the viscosity: the original's boundary term uses 1.0 × 10⁻³ Pa s, the legacy's Kx 1.3076 × 10⁻³ (water at 10 °C)
     - 1.3076 × 10⁻³ in both, so that the boundary and Kx agree
   * - ME-21
     - MECHA's height is one cell's length (200 µm in its projects); the plasmodesmata conductance carries no height,
       so with a segment's length as the height kr falls with it (monocot, barrier 3, ``k_xyl`` 10⁻⁶: 3.98 × 10⁻⁵ at
       200 µm, 9.41 × 10⁻⁶ at 1 mm, 1.97 × 10⁻⁶ at 5 mm)
     - kept: the height is the segment's length; per-tissue cell heights are not yet implemented

The osmotic scenarios
---------------------

.. list-table::
   :header-rows: 1
   :widths: 8 52 40

   * - ID
     - legacy
     - port
   * - p
     - the cortex layers: a cell's row found by ``rank in rank_to_row`` (a test on the rows, not the ranks), so every
       cortex cell is in row 0, layer c1; and ``8 − (row_outer − row)`` runs outward
     - corrected: ``presets_rules`` 1 (c1 the outer cortex row, c2 the next inward, … c8 the 8th and deeper; c8 for a
       single-row cortex); 2 reproduces. Standard uptake with the osmotic terms: +10.1 % (monocot, barrier 1) to
       +32.7 % (barrier 3) with ``os_hetero`` 1, +3.8 % to +11.4 % with ``os_hetero`` 2
   * - q
     - ``os_hetero`` 4 reads the endodermis cells' π from ``osmotic_endo``, the stelar apoplast's value
     - corrected: ``presets_rules`` 1, the cells' own ``osmotic_endo_cell``; 2 reproduces
   * - ME-22
     - the relative x of a wall from GRANAP's cell seeds (``_cells_gdf['x']``)
     - the cells' centroids (the DataStructure has no seeds): one unit of rounding with equal sides; it would weigh a
       left–right soil profile
   * - ME-23
     - the soil's left and right potentials (``psi_soil_left``, ``psi_soil_right``), read only by the scenario loop;
       no scenario of the branch sets them apart
     - ported (2026-10-27): ``standard_psi_soil``, ``standard_psi_soil_right`` in the standard problem (the plant
       problem takes a soil potential per node, ``soil_psi``), compared to 10⁻¹² (``test_scenario_boundaries.py``)
   * - ME-24
     - the radial profiles' convection velocity is never computed (0): the "convective" profile is flat
     - reproduced (flat)
   * - ME-25
     - the elongation sink at barrier 0 (``elongation_BC``); a phloem boundary at barrier > 0 (``pressure_sieve`` NaN
       by default)
     - ported (2026-10-27): ``elongation_midpoint_rate``, ``elongation_side_rate_difference``,
       ``water_fraction_apoplast``, ``water_fraction_symplast`` (the sink in the plant problem, and in the standard
       problem with ``standard_elongation`` 1); ``standard_psi_sieve_tube``. The sink per node to 10⁻¹⁵, the
       solutions to 10⁻¹² of their range (``test_scenario_boundaries.py``). The rate is 0 by default (``BCs.xml``'s
       2.8 cm d⁻¹ is maize's, an input: a section's sink does not scale with its height)
   * - ME-26
     - the layer-wise uptake and transmembrane fractions (``standard_transmembrane_fractions``,
       ``create_layer_discretization``); the solutes on the standard problem's flows (``standard_solute_flux``)
     - ported (2026-10-27): ``standard_transmembrane_fraction`` and ``membrane_layer`` per membrane,
       ``postprocess.transmembrane_layers`` and ``transmembrane_cells``, compared per membrane to 10⁻¹⁰ and by layer
       to 10⁻¹² (``test_transmembrane_fractions.py``); MechaSolutes' ``solute_flows`` 2, compared to 10⁻¹¹
       (``test_standard_solutes.py``); the axial flows by layer (``flow_xyl_layer``, ``flow_sieve_layer``):
       ``postprocess.axial_layers``, compared to the solutions' agreement (``test_postprocess.py``)
   * - ME-38
     - a cortex rank with no layer row (an incomplete outermost layer with no layer after it: the layer rows' loop
       skips it): the presets look its row up by value (``rank in rank_to_row``), 0 when absent
     - fixed: the row of the previous assigned cortex rank, as the incomplete layers before it merge (the tomato root,
       rank 49; ``test_simulation.py``); the port raised (NaN row)
   * - ME-39
     - ``elongation_BC`` takes each wall's ``HydraulicWall.thickness``, which GRANAP's path leaves at its default,
       1 µm, while the conductances take the geometry's (1.5 µm)
     - fixed: the walls' thickness (``thickness_source``); the comparison sets the legacy's walls to 1.5 µm
   * - ME-40
     - the transmembrane fractions look a cell's layer row up by value (``rank in rank_to_row``, as ME-38): only the
       membranes whose rank is one of the rows' numbers are counted (on GRANAP's monocot root, the epidermis,
       exodermis, endodermis and pericycle: the cortex and the stele are left out, the fractions sum to −0.19)
     - fixed: every membrane its row (``osmotic.membrane_rows``); the comparison takes the membranes the legacy counts
   * - ME-41
     - the transmembrane fractions compare the cell's id with ``geometry.passage_cell_ids`` and
       ``intercellular_ids``, which hold the cell objects (``cell_manager.passage``, ``.intercellular``): the passage
       cells' row offsets (barrier 2) are never applied, the intercellular spaces' membranes never excluded
     - fixed: the original's rules (``membrane_rows``; ``intercellular=False`` keeps the spaces', for the
       comparison)

Inputs, outputs and scenarios
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 8 52 40

   * - ID
     - legacy
     - port
   * - ME-42
     - the phloem's flow condition (scenario ``flowrate_prox`` at the sieve boundary): split with the protophloem
       cells' area ratios, given by rank to the protophloem (barrier 0) or to the sieve tubes (barrier > 0), so that a
       sieve tube gets a protophloem cell's share, or NaN (GRANAP's roots have no protophloem cells: every right-hand
       side NaN)
     - ported (2026-10-28) and fixed: ``standard_sieve_flow``, split among the conduits as their own areas (the
       protosieve cells, each once, at barrier 0; the sieve tubes at barrier > 0), where the phloem's potential is NaN
       (both given: an error, as the legacy's); kr at barrier 0 through the protophloem's mean potential. Against
       the legacy's scenario 1 given the sieve tubes' shares, to 10⁻¹² (``test_scenario_boundaries.py``)
   * - ME-43
     - ``Kax_source`` 2: the conduits' axial conductances given by cell (the hydraulics file's ``K_xyl`` and
       ``K_sieve`` by id, the same entries as the xylem's and the phloem's boundary penalties), in Kx only (the solve
       keeps ``k_xyl``); at barrier 0 a sieve value read as its text's first character (``get("value")[0]``)
     - ported (2026-10-28) and fixed: ``measured_axial_conductivity`` (a conduit's input, its specific conductance in
       cm⁴ hPa⁻¹ d⁻¹, NaN: Poiseuille's) in Kx and Kx_phloem, the axial boundary (``standard_axial_source`` 1) and
       the plant problem's junctions; the boundary penalties stay ``k_xyl``, ``k_sieve``. Kx against the legacy's to
       10⁻¹⁵ (``test_postprocess.py``); the junctions: ``test_plant_problem.py``
   * - ME-44
     - the cortex's aquaporin gradient (``ratio_cortex`` ≠ 1) on GRANAP's path: ``_calculate_cortex_parameters`` sums
       one-element arrays (``distance_center_grav[wall]``), so that ``a_cortex`` is one, which ``float()`` refuses
       with this environment's numpy (2.5; deprecated since 1.25): TypeError at every membrane of the cortex
     - fixed (scalars, ``conductance.cortex_parameters``); the comparison gives the legacy its value as a float:
       every conductance bit for bit, with the φ-thickenings, the tissues' aquaporin factors, the xylem pieces and
       other conductivities (``test_hydraulic_options.py``)
   * - ME-27
     - its own GRANAR cellset XML builder: cell areas as polygons through the wall midpoints and junctions (0.77 % less
       area than the true outlines), wall positions moving the mean epidermis distance by 0.05 %. On GRANAR's
       ``current_root.xml``: kr 9.111275 × 10⁻⁵ and 3.549032 × 10⁻⁵, Kx 1.026006, against 9.115400 × 10⁻⁵,
       3.550639 × 10⁻⁵, 1.028664 through GRANAP
     - dropped: cellsets are read by GRANAP's ``CellSetOrgan`` (the legacy on the GRANAP organ and the port then agree)
   * - ME-28
     - combinatorial scenario sets (``h`` over ``kw`` × ``kpl`` × ``kaqp`` × barrier values), boundary-condition
       scenarios
     - dropped: one parameter set per plant; scenarios are plants of a scene or separate runs
   * - ME-29
     - the paraview, graphml and plot exports
     - not ported (plots of the DataStructure in ``openalea.mecha.plots``)
   * - ME-30
     - the passage-cell and protophloem branches, the aerenchyma borders, the "no exodermis" and "no pericycle"
       ranks: unreachable from GRANAP's presets
     - reproduced; covered by the code and by tests on stand-in networks (random cgroups, a protophloem stand-in)
   * - ME-31
     - root rules only: no endodermis on stems and leaves (the gravity centre NaN), every epidermis border wall in
       contact with the soil
     - extended: the centre of all the cells without an endodermis; soil contact, perimeter, kr and the standard
       problem on root sections only

The ``reactions`` branch
------------------------

.. list-table::
   :header-rows: 1
   :widths: 8 52 40

   * - ID
     - reactions branch
     - port
   * - r1
     - the osmotic volume flow counted twice in the advection: the edge flows already contain K σ Δπ, and the
       operator adds it again (``_build_osmotic_advection``)
     - fixed: advection by the water's flow, × (1 − σ) on the membranes. On one step of the needle with σ 0.7:
       :math:`\max|\Delta c| / \max c = 4.7`
   * - r2
     - σ read on the wall's end of a membrane stored cell → wall, in the Scharfetter–Gummel operator
     - fixed (one σ per membrane edge); latent on the needle (its graph stores every membrane wall → cell)
   * - r3
     - two independent σ: the water's (presets, by rank) and the solute's (0.7 by cgroup)
     - fixed: one reflection coefficient per membrane, the water's, for both
   * - r4
     - the sucrose–starch reaction in the walls and junctions too, with the raw volumes
     - fixed: in the cells only (0.61 of the starch mass on one needle step)
   * - r5
     - the photosynthetic source a concentration increment per step: its rate scales with 1/Δt
     - fixed: a source flow ``sucrose_source`` (mol d⁻¹); the comparison gives the port V Δc / Δt
   * - r6
     - ``needle.py``'s transpiration sums the ``wall_air`` flows, about 0 by conservation (the pinned walls' outflow
       is the penalty's residual)
     - dropped with the pinned walls: transpiration is the organ's flow at its leaf surface
   * - r7
     - the osmotic baseline is the root presets' (``os_cortex`` −4800 hPa) where the comment says it is zeroed (only
       ``s_factor`` is set)
     - kept as the presets' baseline; ``presets_sections`` 2 applies the presets to the needle, as the branch does
   * - r8
     - the protophloem list mixes cell ids and node ids: the sieve penalties at barrier 0 on wrong rows
     - not ported (the port's own protophloem set, counted once, f)
   * - r9
     - ``kw_meso_hypo`` ("hypodermis–mesophyll") applies to the (2, 4) interface: GRANAP's hypodermis is cgroup 1, so
       it cuts the epidermis–cgroup 4 walls
     - reproduced: ``kw_meso_hypo`` on the (2, 4) walls, their membranes and plasmodesmata (NaN by default: none)
   * - r10
     - a wall's π set per membrane: on 165 needle walls between a cortical-rule and a stelar-rule cell, two apoplastic
       potentials, one per side
     - fixed: one π per wall (the last set); scenario 1 differs by up to 374 hPa (3.7 %) at those walls
   * - r11
     - the vapour side's ``kwa`` and ``k_air`` (1 × 10⁻⁴, labelled cm hPa⁻¹ d⁻¹) multiply a dimensionless vapour
       pressure: placeholders without a physical basis
     - fixed: vapour conductances from diffusion in air (Kelvin walls, air links, stomata's anatomical maximum)
   * - r12
     - the vapour edges carry solute with their flows; the air nodes, without diffusion, are Dirichlet 0 and act as
       sugar sinks
     - fixed: solutes on the liquid nodes only (2 × 10⁻¹² on one needle step)
   * - ME-32
     - ``k_xyl`` and ``k_sieve`` default to 1.0 (10⁻⁶ in the refactored legacy before ``develop``; ``develop`` too has
       1.0)
     - not ported: the Poiseuille boundary (ME-19) holds the vessels (monocot root, barrier 1: kr 1.0619 × 10⁻⁴ against the
       branch's 1.0618 × 10⁻⁴); the comparisons give ``k_xyl`` and ``k_sieve`` 1.0. The legacy returns its matrix after
       adding then removing that 1.0 on the xylem's diagonals, rounded at its scale: compared to 2 ulp of it
   * - ME-33
     - the 10⁻¹⁶ membrane floor only when ``kmb`` is 0 too: air-filled intercellular spaces conduct through ``kmb``
       (5 × 10⁶ times more on the root's, 26 membranes of the needle's air spaces); ``develop`` has it too
     - not adopted: the original's floor (an air-filled space has no water-conducting membrane), ``membrane_floor`` 1;
       ``membrane_floor`` 2 reproduces it (the comparisons: 1196 membranes of the monocot root)
   * - ME-34
     - tissue names (``hypodermis`` → cgroup 2, ``vascular parenchyma``)
     - inert: GRANAP's cells carry their cgroup; the type fallback only applies without one
   * - ME-35
     - the evaporating walls pinned to the air spaces' potential by a penalty (``set_air_wall_bc``); the ``wall_air``
       exchange in series with the flanking membrane
     - ported, compared (the penalty and every coefficient bit-exact, the solution to 1.8 × 10⁻¹²), then removed: the
       pinned walls have no limiting conductance (the seedling's soil–plant iteration diverged); the membrane is
       already an edge of the graph, so no series membrane
   * - ME-36
     - ``k_sieve[0]`` on a scalar default (TypeError with a phloem potential), ``HydraulicCell.RH`` uninitialised,
       ``k_air`` used for two things, ``psi_total`` defined two ways, JFNK with time stepping inconsistent
     - dropped (not ported)
   * - ME-37
     - the steady coupled solver (Picard, JFNK, continuation), the calibration package, the ``apo`` and ``sym`` modes,
       the upwind-only solve, θ ≠ 1: used by no simulation. The capacitance's storage fractions (``C_wall``,
       ``C_cell``) and the single-species degradation (``k_deg``), used by ``develop``'s tutorial 04
     - ported (2026-10-27): the storage fractions and the degradation (``storage_fraction_wall``,
       ``storage_fraction_cell``, ``k_decay``: sucrose only, the starch stores in its whole volume), to 10⁻¹¹
       (``test_solutes.py``); the modes (``solute_pathway`` 2, 3), the upwind scheme (``solute_scheme`` 2) and θ
       (``solute_theta``), to 10⁻¹¹ (``test_standard_solutes.py``; θ with starch, ``test_solutes.py``); the steady
       coupled solve (``solute_solve`` 2: Newton on the coupled residual, instead of Picard, JFNK and continuation),
       against the Picard iteration with the reactions' r1 and r3 removed, the sucrose to 10⁻¹¹ and the water to
       10⁻¹² of their ranges (``test_coupled_steady.py``). The operators' ``'D'`` and ``'A'`` alone (diffusion
       only, advection only: a diagnostic of the operator) are not ported: the water flows at 0 (one potential for
       the soil and the xylem) or the diffusivities at 0 give them, to 10⁻¹¹ and 2 × 10⁻⁵ of the range
       (``test_standard_solutes.py``, ``test_operators_alone``). The calibration package and the scenario builder are
       not ported (F9, F10)

The ``develop`` branch
----------------------

What ``develop`` (945b2be) adds to the refactored legacy, and the behaviours found merging it.

.. list-table::
   :header-rows: 1
   :widths: 8 52 40

   * - ID
     - develop
     - port
   * - d1
     - its merge left ``HydraulicMatrixBuilder._contact_nodes``'s header as a second ``_apply_soil_boundary``: every
       solve raised AttributeError
     - the legacy repaired (46a6f08's header); nothing to port
   * - d2
     - the plasmodesmata factors' keys fixed (ME-09–ME-12): ``develop``'s map applies them. Its phloem-pole factor
       (PPP) on the companion cells' interfaces (stele, xylem, pericycle), not on the phloem-pole pericycle's; with its
       default 0 (and PCC 0), the companion cells are symplastically isolated. The input cannot change it: the loader
       stores PPP under ``pericycle_phloem_pole_factor``, the solver reads ``phloem_pericycle_pole_factor`` (always 0)
     - not reproduced: the comparisons give the legacy PPP 1 (under the key its solver reads); the port's
       ``kpl_PPP_factor`` stays 1 by default
   * - d3
     - tissue groups: 22 the cambium (plasmodesmata frequencies at its interfaces with the endodermis, stele, sieve
       tubes, companion cells, xylem and itself; its walls with the endodermis ``kw_endo_peri``; a stele–cambium factor
       whose key exists nowhere, so 1); 23 labelled "phloem" (still the protosieve list); 24 and 26 labelled companion
       cells (the label only: its tables read the raw cgroup; its ranks read 26 as 12)
     - 22 in both rules (the original rules: develop's frequencies as parameters, ``fplxheight_*_cambium``), its walls'
       ``kw_endo_peri``, ``kpl_cambium_stele_factor`` (rules 2, 1 by default); 24 and 26 companion cells in the
       original rules. GRANAP writes none of them (its cambium is 12)
   * - d4
     - barrier 10 (its comment "Exodermis sub + lignin", its README "suberized exodermis with Casparian strip"; no
       endodermal Casparian strip; absent from ``BARRIER_KEY``) and 11 (barrier 8, ``kw_scler`` = the lignin value on
       every wall of a ``sclerenchyma`` cell, after septa and passage cells, before φ-thickening and the interface
       table)
     - as coded (``test_develop_barriers.py``: bit for bit); the sclerenchyma by GRANAP's tissue (GRANAP gives it
       cgroup 4), its walls after φ-thickening and before passage cells and septa (the legacy's precedence)
   * - d5
     - ``tracheid_as_xylem`` (False by default): the transfusion tracheids' membranes take the xylem's branch
       (kaqp_stele × 10⁴ with barriers, the vessels' apoplastic solute term), never the xylem's boundary
     - ``tracheid_as_xylem`` (0 by default), the same; compared bit for bit on a root whose stele is half retagged
       tracheids
   * - d6
     - ``pd_spatial_config`` (None by default): the plasmodesmata's aperture × a coefficient of their distance to the
       soil-contact nodes along the graph (networkx's Dijkstra), linear or exponential from ``c_wet`` to ``c_dry`` over
       the plasmodesmata's range; its K and its solutes' symplastic term; not the special cases (intercellular spaces,
       xylem–xylem)
     - ``pd_spatial_profile`` (0 by default), ``pd_spatial_c_wet``, ``pd_spatial_c_dry``, ``pd_spatial_length_scale``;
       the distance by scipy's Dijkstra per section (``contact_distance``); compared to 10⁻¹² (both profiles, both
       roots)
   * - d7
     - ``conserve_boundary`` (True by default): the water each node exchanges with the boundary, its edges' flows'
       residual, carries the node's concentration out where it leaves, ``c_boundary`` in where it enters (the
       reactions branch: no such term, a node drained by the boundary concentrates, up to 1.8 × a Dirichlet source)
     - ``solute_boundary_advection`` (1 by default; 0 the reactions branch's, for its comparisons), ``c_boundary``;
       the vapour edges counted in the residual, so that evaporating water leaves its solutes behind (develop has no
       vapour edges in a root); the exchange compared with ``_boundary_flux`` to 10⁻⁹ of its largest
   * - d8
     - views: the radial profile and the pathways' breakdown along the wet→dry transect (``x_axis='signed'``; its
       breakdown averages each ring's position, near 0), the maps of the contact distance and of the plasmodesmata's
       aperture
     - ``plots``: ``axis="signed"`` of ``radial_profile``, ``pathway_flows``, ``pathway_breakdown`` (each layer's wet
       and dry halves apart), ``wetdry_axis``, ``contact_distance_map``; developer tutorial 02
   * - d9
     - ``conserve_boundary`` enters only the upwind operator (``build_advection_matrix``): the Scharfetter–Gummel
       operator, the needle's, has no liquid boundary whatever its value
     - ``solute_boundary_advection`` acts in both schemes; the Scharfetter–Gummel comparisons set it to 0
   * - d10
     - the soil contact by cells (the hormones file's ``Contactrange``, ``hormones.contact``): the border walls of the
       listed protoplasts, whatever ``x_contact``; a border junction when all its cells are listed (``develop``'s
       ``_contact_nodes``; the reactions branch: when any is). Also the source of ``pd_spatial_config``'s distances
     - ported (2026-10-28): ``soil_contact_cell`` (a protoplast's input, 1 in contact), ``contact_junction_rule`` (1
       all, develop's, by default; 2 any); a junction's cells those of the walls it joins (the legacy's
       ``junction_wall_cell``, without its first-three truncation, which only an inner junction of four cells meets).
       Soil terms bit for bit, solution to 10⁻¹² against both legacies (``test_soil_contact_cells.py``)
   * - d11
     - ``SoluteTransport.solve`` with ``boundary_conditions`` (any nodes held at a concentration: its tutorials hold
       the epidermis at 1, with no sink) and without a time step (its steady state, ``T c = −s``), one species
     - ported (2026-10-28): ``fixed_sucrose`` (a node's input: the concentration held, NaN free), ``solute_sinks`` (1
       the reactions' needle sinks at ``c_sink``, by default; 0 none), ``solute_solve`` 3 (the steady state on the
       flows given: the transport, the loss and the liquid boundary; no starch). Its tutorial 04's steady states (no
       loss; ``k_deg`` 48 d⁻¹) to 10⁻¹¹ (``test_standard_solutes.py``)


Constants
=========

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - quantity
     - value
   * - units
     - mm → µm × 1000; µm² → cm² × 10⁻⁸; K in cm³ hPa⁻¹ d⁻¹
   * - junction positions
     - rounded to 6 decimals (µm)
   * - ranks
     - cortex 40–49, stele 50–61 (62 ranks), 12 sweeps
   * - layer rows
     - four endodermis rows, two exodermis rows, a cortex layer merged when it holds fewer than 0.75 of the previous row's cells
   * - relative position
     - −1 at the xylem's 80th-percentile distance, 0 at the endodermis, 1 at the epidermis, clamped at ±10⁻⁵
   * - floors
     - 10⁻¹⁶: barrier walls (default barrier values), septa between xylem pieces, membranes without aquaporin
       activity outside barrier interfaces
   * - xylem
     - membrane aquaporins × 10⁴ in mature vessels; plasmodesmata 10⁴ × the default frequency between xylem pieces
   * - viscosities
     - water 1.3076 × 10⁻³ Pa s (10 °C); phloem sap 1.7 × 10⁻³ Pa s
   * - π in Poiseuille's law
     - 3.141592
   * - plasmodesmata
     - ``kpl`` 5.3 × 10⁻¹² cm³ hPa⁻¹ d⁻¹; default frequency 8.0 × 10⁵ cm⁻¹
   * - graph systems
     - Newton, at most 3 iterations, tolerance 10⁻⁸ cm³ d⁻¹ (water), 10⁻¹⁴ (solutes)
   * - Bernoulli function
     - Taylor series below :math:`|x| = 10^{-10}`, its limits beyond :math:`|x| = 700`
   * - van 't Hoff
     - R = 8.314 × 10⁴ hPa cm³ mol⁻¹ K⁻¹, T 298.15 K
   * - vapour
     - :math:`V_w` 1.8 × 10⁻⁵ m³ mol⁻¹, R 8.314 J mol⁻¹ K⁻¹, Tetens' :math:`e_\text{sat}` (610.78 Pa, 17.27, 237.3)
