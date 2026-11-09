===========
Performance
===========

Two kinds of figures are given here. The first sections are the timings recorded while porting, with the commit that
recorded them: single runs from October 2026, not benchmarks; the machine is not recorded, and no profiling script is
kept in the repository. They are historical records of the code of their time. The last section, "In metafspm's use
case 1 (November 2026)", gives the figures measured on metafspm's use case 1 (``examples/use_case_1/benchmark.py`` in
metafspm) during metafspm's release optimisation, each with its date, its commits and the scene it was measured on.


Recorded timings (October 2026, historical)
===========================================

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


Where the time goes (October 2026, historical)
==============================================

As recorded in October 2026, on the 10 sections above; see the last section for use case 1's split today.

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


In metafspm's use case 1 (November 2026)
========================================

Measured with metafspm's ``examples/use_case_1/benchmark.py`` (``--variability 1``) during metafspm's release
optimisation, on the development laptop (14 processors), each run alone under an 8 GB memory bound. Every change below
gave the same results as the code before it, bit for bit on every step compared, except where stated. Each phase
measured its before and after on the same day; the machine's speed varied by a few per cent from day to day, so figures
of different phases are not compared with each other. "8 plants resumed" runs start from a checkpoint at hour 264;
day 12 is hours 267 to 288, day 13 hours 289 to 312. The runs of 2026-11-06 used one thread, the default of the time,
and those of O-6 the process pool of the time where stated.

**Kinds by length class** (MECHA ``ae1e519``, metafspm O-4, 2026-11-06). ``plant_reduction``, ``scenario_reduction``
and ``root_system`` take ``length_classes=0.1`` by default: a reduced section's kind is interpolated between the two
neighbouring length classes :math:`(1 + 0.1)^k` (``"exact"`` keeps one kind per exact length; the exactness tests use
it). At 1 plant to day 8, against an earlier run with one kind per exact length (metafspm ``343749c``), the kinds
evaluated went from 1398 (218.5 s) to 470 (106 s) and the day-8 step from 3.83 to 1.81 s. Against ``"exact"`` at
1 plant over 3 days, the total uptake is within 1e-10 of its mean, the roots' and the leaves' water potentials within
2e-6 and 1e-6. Not bit-identical to ``"exact"``, by design.

**Reduced sections batched** (MECHA ``7cc485f``, metafspm O-5, 2026-11-06). The condensed volumes come from the
reduction's ``kept_outputs`` per template; kr, Kx, the uptake and kr_effective of every reduced section are computed at
once (``conduit_axial_conductances`` per section, batched). Kx, Kx_phloem and kr_effective agree with the former
per-section sums at round-off (9e-16, 9e-19 and 3e-18 absolute: bincount's summation order); the water potentials are
identical. With metafspm's vectorised reduction, the last day's step went from 2.05 to 1.73 s at 2 plants, day 8, and
from 1.14 to 0.94 s at 8 plants, day 4.

**The linear solvers** (MECHA ``bde92cf``, metafspm O-6, 2026-11-06; MECHA ``5149d56``, metafspm P3, 2026-11-07).
``_standard_problem`` is declared ``@graph_system(symmetric=True)`` (checked once per pattern). MECHA declares the
property, never a library: the run maps it to a backend, e.g. CHOLMOD with
``Compute(linear_solver={"default": "superlu", "symmetric": "cholmod"})`` (metafspm fix-D: a backend per system, by
its qualified name such as ``"MechaHydraulics._plant_problem"``, else a property it declares, else ``"default"``; the
reductions' Kron factorisations follow the plant problem's). ``_plant_problem`` is not symmetric once the leaves transpire: on use case 1,
pairs of kept shoot nodes at 0.049 one way and 3.6e-8 the other, on rows with a 1.4e-6 diagonal (the vapour path's
explicit terms). metafspm solves the plant problem's independent blocks (one per plant) apart. On the plant system at
8 plants, day 13 (252 k rows, 13.3 M nonzeros, 8 blocks; its pattern changes every hour, so no analysis is ever
reused), one solve took (2026-11-07):

========================================== ========= =========
solver                                     1 thread  4 threads
========================================== ========= =========
SuperLU, the blocks gathered (the default) 0.74 s    0.35 s
PARDISO per block, analysed at each solve  1.43 s    0.60 s
MUMPS, the whole system                    1.49 s    1.52 s
========================================== ========= =========

SuperLU stays the default. PARDISO per block with its analysis kept takes 0.16 s at 8 threads: it pays in a scene whose
pattern does not change. The optional solvers (metafspm's optional extras ``pardiso`` and ``mumps``) differ
from SuperLU by 1 to 3e-12 relative per solve. With BLAS held at one thread inside the scene, the results are the same
bits at any thread count (8 plants resumed at 1, 4 and 12 threads). ``superlu_reuse`` (the ordering kept per pattern)
is not advised on use case 1: at 2 plants over hours 121 to 168 it took 73.9 s against 73.2 s, and its memory grew
about 0.6 GB a day more (2026-11-06, with the process pool of its time).

**The kinds' assembly** (metafspm P2.3, 2026-11-07). metafspm's ``reduction.terms`` writes every group's kinds' blocks
with a compiled kernel: 0.79 to 0.21 s of a profiled step, and the group tables 171 to 62 MB, at 8 plants, day 12,
1 thread. Over metafspm's P2 (8 plants from the sowing to day 13, 1 thread), MechaHydraulics went from 2.81 to 2.30 s
per step on day 13. At day 12 the kinds' evaluation took 0.49 s of a profiled step: the library's groups 0.17 s,
MECHA's components on the library 0.20 s, the Kron reductions (SuperLU) 0.06 s, the condensed volumes 0.04 s.

**threads="auto"** (MECHA README ``1a09fd1``, metafspm P5, 2026-11-07). ``Compute(threads="auto")`` is the default; the
plant problem's blocks take the threads found on its own solves (4 on this laptop). Per day-13 step at 8 plants,
``"auto"``: all of SuperLU's solves 0.69 s, the plant system's about 0.35 s at 4 threads against 0.74 s at 1; the Kron
reductions 0.07 s.

**Generic kernels** (metafspm ``devplan_generic_kernels``, 2026-11-08). Before them, per functor at 8 plants, day 13,
``"auto"``: ``MechaHydraulics.plant_problem`` 1.03 s per step, ``cell_rank`` 0.54 s, ``conductance`` 0.23 s; under the
profiler (hours 290 to 292), metafspm's ``parameter_view`` 0.10 s.

- GK-1 (MECHA ``7540d87``): MECHA's own ``parameter_table`` is gone; the components call metafspm's
  ``FunctionalComponent.parameter_table(names, location)``, gathered through an owner index kept per topology.
- GK-2 (MECHA ``18af37e``): ``section_layout`` and ``per_node`` (``_cell_rank``'s ranks, ``relative_x``,
  ``_root_nodes``) are one ``ds.local_gather`` each. The other per-section loops (the network's φ cells,
  ``cortex_gradient``, ``_kr``, ``_standard_axial_inflow``) stay: in use case 1 every section is reduced, so they run
  on the reduction's library only.
- GK-3 (MECHA ``7142f9e``): the plant and standard problems' Jacobians and the osmotic input Jacobian are
  ``graph_view.laplacian(weights)``, bit-identical to scipy's :math:`B\,\mathrm{diag}(w)\,B^\top`. At 8 plants, day 13,
  its pattern takes 0.027 s to build and 0.005 s a call to fill, against 0.025 s a call for scipy's product.
- Over GK-1 to GK-3, 8 plants resumed: the day-13 step went from 4.93 to 4.46 s at 1 thread and from 4.75 to 4.20 s
  at ``"auto"``.
- GK-4 (MECHA ``5a4cef4``, ``a343048``, with metafspm ``1896fb7``: one library per reduction and time step, every
  signature's kinds in the same batches): the condensed volumes' unit volumes per template are kept per (template,
  thickness parameters) in the component's own caches (``keep(saved=False)``), from one batch of kinds to the next:
  the ``@kind_derived`` time over the 48 resumed steps to day 13 went from 1.71 to 0.35 s. The standard
  transmembrane fractions, which no kind reads, are not computed on a reduction's library (0 and -1 there): they had
  taken 0.9 s over the same steps. At 8 plants to day 8, 1 thread, the kinds' time went from 115.7 to 100.5 s.

**The process pool removed** (MECHA ``fea259b``, metafspm QG-3 ``b69347a``, 2026-11-08). metafspm's process pool for
the kinds (``Compute(workers=)``) is gone; a reduction's kinds are evaluated in the run's threads (metafspm GK-5), with
the same results bit for bit (MECHA's test: ``Compute(threads=4)`` against 1). At 8 plants from the sowing to day 8,
``"auto"``: 292.6 s at a memory peak of 3.15 GB, against 283.1 s and 3.83 GB with the former pool. At 8 plants resumed
from hour 192: day 12 3.59 s, day 13 4.27 s per step at 1 thread; 3.27 and 3.92 s at ``"auto"``. At 16 plants, day 8,
``"auto"`` (metafspm ``86a4a60``, a few hours before the removal, with the pool off): the last day's step 3.26 s at
3.68 GB, against 2.96 s at 4.60 GB with the pool.
