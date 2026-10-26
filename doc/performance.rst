===========
Performance
===========

The timings recorded while porting, with the commit that recorded them. They are single runs from October 2026, not benchmarks;
the machine is not recorded, and no profiling script is kept in the repository.


Recorded timings
================

.. list-table::
   :header-rows: 1
   :widths: 44 18 18 20

   * - what
     - legacy
     - port
     - recorded
   * - MECHA's geometry of a section (GRANAP's monocot root): the legacy's ``populate_from_network``, the port's
       extraction from the DataStructure and ``section_geometry``
     - 0.47 s
     - 10 ms (4 ms extraction, 6 ms geometry), once per anatomy template
     - d59f254, 14 October 2026
   * - the legacy on GRANAP's monocot root (seed 0; 4146 walls, 2763 junctions, 1384 cells, 20394 conductances),
       barriers 1 and 3: network, conductances, two solves, kr and Kx
     - 2.8 s
     - —
     - d62d9c1, 14 October 2026
   * - the conductance step on 10 sections (83 000 nodes, 204 000 edges), geometry cached
     - —
     - 0.2 s per call
     - bb271ba, 14 October 2026
   * - one call of ``MechaHydraulics`` on the same 10 sections, with the standard and plant problems as graph systems
     - —
     - 0.5 s (the two systems 0.1 s together)
     - f207211, 14 October 2026
   * - the seedling in the soil–plant–atmosphere scene (48 segments; 524 000 Compartments, 1.29 million
       Connections), stomatal-pore transpiration of that version
     - —
     - built in 35 s; converged in 6 scene steps, 51 s
     - 3e5ed23, 14 October 2026
   * - the ``reactions`` branch's needle (16 195 nodes, 39 189 edges), one dynamic step (two water solves: its
       scenarios 0 and 1, then the transport)
     - about 2.8 s per step
     - —
     - 0644936, 16 October 2026


Changes in metafspm that MECHA needed
=====================================

Two costs of the coupling were in metafspm, and were fixed there:

- **Sparse Jacobians** (metafspm 71340de): a graph system's analytic Jacobian may be a sparse matrix, kept sparse, with
  the boundary sets' terms added sparse. The dense path was out of reach at 10⁵ nodes, and the sparse one failed with
  Robin boundary sets.
- **Anatomy-mode variables read in bulk** (metafspm 4051fba): reading a coarse-scale variable onto the Compartments
  mapped each Compartment to its owner one at a time. On GRANAP's seedling (212 000 Compartments, 526 000
  Connections), constructing GRANAP's anatomy component took 477 s, then 0.6 s; its call 10 s, then 0.8 s.


Where the time goes
===================

- **The geometry** is computed once per distinct anatomy (``MechaSections.section_geometries``): sections copied from
  the same GRANAP template share it, and it is recomputed only when a section's node or edge count changes. Its 10 ms
  per anatomy compare with 0.2 s per call for the conductances.
- **The conductances** are vectorised over every edge of every section at once (``conductance.edge_conductances``,
  one parameter value per edge), 0.2 s per call on 204 000 edges.
- **The solves**: both problems are linear and solved by one Newton step on a sparse analytic Jacobian; on 10 sections
  they take 0.1 s of a 0.5 s call.

No timing separates the remaining steps (the osmotic drives, the axial junctions, the collar, the section results).
The parts of the port written as Python loops for exactness to the legacy are: each wall's distance to the centre
(scalar ``pow``), ``_pow2`` (one ``pow`` per distinct thickness), ``cortex_parameters`` (only when ``ratio_cortex`` is
not 1), ``axial_conductances``, and the osmotic presets (``legacy_rows``, ``relative_positions``, ``wall_counts``,
``section_presets``: one iteration per membrane, per root section, at every call). They have not been timed
separately.


Options affecting speed
=======================

- ``standard_solve`` (per plant, 1 by default): 0 leaves the plant's sections out of the standard problem
  (``in_standard_problem``), and its kr is not computed. The standard problem only runs on root sections.
- **Anatomy templates**: a plant whose sections share GRANAP templates derives MECHA's geometry once per template.
- **Anatomy size**: the Compartments per section follow GRANAP's presets; a 2.5 mm stem section gives 45 000
  Compartments against the seedling's 1 mm stems, which is why the seedling's stems are 1 mm.
- ``MechaOsmoticPresets`` recomputes the presets of every root section at each call; leaving it out of a model removes
  that cost (the osmotic inputs then stay 0).
- ``MechaSolutes`` adds one linear graph system per time step on the liquid nodes, with two unknowns per node (sucrose
  and starch).
