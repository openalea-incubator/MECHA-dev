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
   * - not ported
     - ``coupled_solver`` (Picard, JFNK), ``scenario_builder``, ``calibration``, ``elongation_BC``,
       ``standard_transmembrane_fractions``, ``standard_solute_flux``, the legacy's ``SoluteTransport``
     - —

The tutorials are ported in ``example/tutorials/developer/`` (four notebooks on the metafspm components) and
``example/tutorials/tutorial_01_input.py``, ``tutorial_02_bc.py``; ``example/comparison/tutorial_01_input.py`` and
``tutorial_02_bc.py`` run the same sections through the legacy and the port. The diffusion tutorial (03) stays in
``example/legacy/``; the solute transport is compared in ``example/comparison/compare_solutes.py`` and
``compare_solutes_day.py``.


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
     - the scenario loop reads ``scenarios[i_maturity]`` for the elongation; the interface flows take the left soil
       potential even with a left–right profile
     - dropped (the scenario loop, the elongation sink and the left–right soil are not ported)
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
     - not ported; the presets keep the left–right weighting of the soil solution (``osmotic_symmetry_soil`` 1)
   * - ME-24
     - the radial profiles' convection velocity is never computed (0): the "convective" profile is flat
     - reproduced (flat)
   * - ME-25
     - the elongation sink at barrier 0 (``elongation_BC``); a phloem boundary at barrier > 0 (``pressure_sieve`` NaN
       by default)
     - not ported
   * - ME-26
     - the layer-wise uptake and transmembrane fractions (``standard_transmembrane_fractions``,
       ``create_layer_discretization``)
     - not ported (the layer rows are, for the presets)

Inputs, outputs and scenarios
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 8 52 40

   * - ID
     - legacy
     - port
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
       the upwind-only solve, θ ≠ 1, single-species degradation: used by no simulation
     - not ported; implicit Euler only (θ = 1)

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
