# MECHA as a metafspm component — review and plan

Your request (2026-10-14):
- port MECHA to metafspm on this branch (`to_metafspm`), with the layout used for GRANAP: the original source frozen as legacy, tests and examples on the port, and `comparison/` folders comparing with the legacy during development;
- port first, as part of the MECHA component's logic, the processing around the solve:
  - before it: the properties assigned to nodes and edges, the conductances;
  - after it: the section-scale metrics kr and Kx;
- only then declare a `graph_system` the metafspm way, solving on the DataStructure's graph view, over possibly connected anatomies in the whole multiscale plant;
- the port takes the MPG DataStructure directly: a `FunctionalComponent` that writes its variables on it and extracts the sparse adjacency from it. GRANAP's legacy exports (networkx graph, XML) stay available.

Answer the questions in §5 on the `→ answer:` lines.

## 0. Environment (done)

`conda/environment.yaml` no longer takes `openalea.granap` from conda.
- **GRANAP and metafspm:** you install them from source in the env (`pip install -e metafspm`, `pip install -e GRANAP`), as noted in the file.
- **The env carries their dependencies:**
  - the geometry stack pinned as in GRANAP (geos 3.14.1, shapely 2.1.2), since its goldens depend on it;
  - opencv from pip, for the icu conflict GRANAP documents;
  - openalea.mtg, numba, pyyaml.

`pyproject.toml`: the conda dependency `openalea.granap` is replaced by a comment naming both, to install from source.

## 1. Review: what MECHA computes today (`src/openalea/mecha`, HEAD 2b995b2)

**Entry.**
- `Mecha(InData, network=None)` (`mecha_class.py`, 1733 lines). The network is MECHA's `NetworkBuilder`, a subclass of GRANAP's `AbstractNetwork`, filled in one of two ways:
  - from a GRANAR cellset XML (`build_network`);
  - from a GRANAP organ (`NetworkBuilder(organ).populate_from_network()`, which reads `organ.graph`, `organ._cells_gdf` and `organ.all_cells`, and scales mm → µm).
- **Public calls:** `compute_conductivities(h)` (kr and Kx per maturity stage), `solve_all_W`, `solve_W(h, i_maturity)`, `water_flux(h)` (every stage × every scenario), `standard_solute_flux`.

**Pre-processing — geometry,** done once in `NetworkBuilder`:
- junctions rebuilt from the walls' positions (GRANAP's are deleted);
- cgroups from the cell types (`type_mapper`), and cell ranks (cortex 40–49, stele 50–61);
- the gravity centre (endodermis), and the walls' distances to it;
- the epidermis perimeter (kr's surface);
- per-wall `distance_wall_cell` (the sum of its membranes' `dist`);
- the radial discretization (`rank_to_row`, `r_discret`);
- interface ratios for plasmodesmata (`concentrated_*`);
- border walls and junctions (the soil interface), and the xylem and phloem cells.

Then `HydraulicCellManager` holds one Python object per wall, cell, membrane and plasmodesma. `_count_surrounding_cells` writes counts on the nodes.

**Pre-processing — conductances,** recomputed at each solve, while the matrix is built (`HydraulicMatrixBuilder`):
- **Wall ↔ wall/junction:** `K = kw_key · 1e-4·((lateral_distance + height)·thickness − thickness²)/length`. `kw_key` is chosen by rules:
  - ghost walls (air spaces, joined xylem): 1e-16;
  - passage cells;
  - φ-thickening;
  - otherwise the cgroup pair of the two flanking cells (`interface_kw_key_map`) → plain `kw`, or a barrier value (Casparian strip, suberin, lignin) set by the maturity stage's barrier code 0–9.
- **Membrane:** `K = 1/(1/(kw/(thickness/2)) + 1/(kmb + kaqp_tissue)) · 1e-8·(height + dist)·length`. Aquaporins are set per cgroup; the cortex follows a radial gradient (`a_cortex`, `b_cortex`).
- **Plasmodesmata:** `K = kpl · fplxheight·1e-4·length · aperture factor per interface · a barrier modifier`.
- **Soil boundary** on border walls and junctions: `K = kw·(length/2·height)/(thickness/2)` into the RHS.
- **Xylem / phloem boundary:** `k_xyl` on the diagonal of the xylem (or protophloem) cells, and a pressure or a flow in the RHS.

**Solve.**
- The Doussan matrix is assembled from the networkx graph's adjacency (COO), then `scipy.sparse.linalg.spsolve`.
- One solve per maturity stage (scenario 0); `water_flux` adds one per further scenario (osmotic and BC variants), reusing the matrix with its RHS rebuilt.

**Post-processing.**
- **kr:** `Q_soil / (ψ_soil − ψ_xylem) / perimeter / height`, from the soil-interface flows (`standard_water_flow`).
- **Kx:** Poiseuille over the xylem cells, `Σ area²/(8π·height·η)`, or prescribed values (`calculate_axial_conductance`).
- **Per edge:** flows Q, cross-sections, velocities (`compute_edge_flows`).
- **Layer-wise uptake and transmembrane fractions** (`standard_transmembrane_fractions`).
- **Section scale:** `kr_tot[stage]`, `total_flow[stage, scenario]`, `K_xyl_spec`, and `root_hydraulic_properties` entries `{barrier, height, kr, Kx}`.

**Inputs** (`data_loader.py`, `InData` from XML or defaults):
- **Hydraulics:** `kw`, `kw_septa`, barrier values (Casparian strip, suberin in and out, lignin), `kmb`, `kaqp` and its factors per tissue (cortex, endo, epi, exo, stele), `ratio_cortex`, `kpl` and its factors, `fplxheight` per interface, the axial source, `k_xyl`, `k_sieve`.
- **Geometry:** wall thickness (one scalar), `pd_section`, `k_interc`; the maturity stages (barrier code and height).
- **Boundary conditions:** per scenario, ψ soil left and right, xylem and phloem pressures or flows, osmotic profiles, elongation.
- Hormones and run flags.
- **Combinatorial scenario sets:** `h` indexes combinations of `kw` × `kpl` × `kaqp` × barrier values.

**Tests and examples.**
- **No test compares with stored values.** The only numeric check is "same order of magnitude" (`assert_close_range`), on kr and Kx: GRANAP organ vs the same organ through its XML.
- **The others:**
  - structural checks: every edge has a K and a Q;
  - one physics check: the outer ψ is above the inner ψ;
  - solute mass conservation (`test_advdiff`, minutes).
- **Not real tests:**
  - several run with no assertion (`test_osmotic`, `test_phi_thick`, `test_visu_types`, which catches every error);
  - some block on `plt.show()`;
  - three are `assert True`.
- **Tutorials:** three notebooks (inputs, boundary conditions, diffusion), none run in CI.
- **The README's usage does not run as written,** because of wrong imports.
- **`src_old/` (untracked) holds older monolithic MECHA v4 scripts.** It is not a usable reference.

## 2. Review: problems found (to know before comparing)

The legacy will be compared as it is, so these matter: a port that reproduces the legacy reproduces them too.

**Parameters silently not applied:**
- a. Plasmodesmata factors: the map's keys (`cortex_factor`, `endo_in_factor`, …) do not match the config's (`cortex`, `endo_in`, …), so only the companion-cell and sieve-tube factors apply; the others fall back to 1. The literal also has a duplicate `(11, 11)` key.
- b. The `Fplxheight` values from XML are never applied (the interface map is filled after loading).
- c. Flow boundary conditions are never applied: the code reads `flowrate_prox`, the loader stores `flow_xyl_prox`.
- d. GRANAP's per-wall `wall_thickness` is ignored: one scalar thickness is used everywhere, and the walls' own thickness objects stay at 1.0 on the GRANAP path.

**Crashes and wrong values:**
- e. `calculate_axial_conductance` fails with an unbound variable when a stage has barrier 0, so `compute_conductivities` crashes on any stage without a barrier.
- f. The protophloem cells appear twice on the GRANAP path, so the phloem boundary term is added twice for them.
- g. `k_xyl` and `k_sieve` are lists, used as scalars in some places.
- h. The scenario loop reads `scenarios[i_maturity]` for elongation (instead of the scenario index), and the interface flows use the left soil potential even when the soil boundary is a left–right profile.
- i. GRANAP's `wall_air` and `air_link` edges (leaves, needles) are ignored by the matrix.
- j. The shipped `extdata/Hydraulics.xml` does not load (barrier attributes missing).

**Code health:**
- k. `tag_phi_thick_cells` is called with its arguments swapped (harmless only because of a guard).
- l. Two sources for the same cell sets: cgroup-based in the solver, type-string-based in the class.
- m. Debug prints, dead code, and results lists that grow on every call.
- n. `list_ghostwalls` and `list_ghostjunctions` are created empty and never filled, so the ghost-wall bookkeeping is dead. The "ghost" branch of the wall conductance only depends on the intercellular count and the barrier (`kw_septa` factor when ≥ 2 intercellular neighbours and barrier > 0, else 1e-16 for xylem pieces).

## 3. Design

**3a. Layout, as GRANAP's:**
- **`src/openalea/mecha/legacy/`:** the current source frozen verbatim, imports rewritten. It keeps working on GRANAP's organs through their legacy exports (networkx graph, `_cells_gdf`, XML), which GRANAP keeps.
- **`src/openalea/mecha/` (the main package):** the port.
- **`test/`:** the port's tests, made real (assertions, headless).
- **`test/comparison/`:** the port against the legacy on the same GRANAP section, skipped in CI with `--ignore=comparison` as in GRANAP.
- **`example/`, from the tutorials:** on the port; `example/comparison/` against the legacy.

**3b. The port is a `FunctionalComponent` on the MPG DataStructure** (anatomy mode: Compartments as nodes, Connections as edges, the SubOrgans as their sections):
- **Inputs, by `GranapAnatomy`** (its declared outputs, §54–59 of GRANAP's plan):
  - Compartment: `kind`, `tissue`, `cgroup`, `section_x`/`y`, `wall_length`, `wall_thickness`, `cell_area`;
  - Connection: `network_path`, `edge_length`, `edge_dist`, `lateral_distance`, `distnode_wall_cell`, `membrane_surface`;
  - SubOrgan: `length` (MECHA's "height" becomes the segment's length).
- **Parameters:** MECHA's hydraulic parameters declared with their units (§5 Q-M3 on their form).
- **What it writes on the DataStructure:**
  - the derived geometry (ranks, distances to the centre, the soil interface…);
  - the conductance `K` of every edge, the boundary terms, `Kx`;
  - after the solve, ψ, the edge flows Q, and the section-scale `kr` and `Kx` as SubOrgan variables (mapped up with the DataStructure's mappings).
- **The adjacency for the solve** comes from the DataStructure itself (its incidence over the Compartments and Connections, sparse arrays), never from networkx. `HydraulicCellManager`'s Python objects are replaced by arrays on the DataStructure.

**3c. The steps, in order** (this phase: everything but the solve):
1. **Geometry** (`@stepinit`, or recomputed when the topology changes): cgroup ranks, gravity centre and distances, the soil-interface walls and junctions, the xylem and phloem sets, the plasmodesmata interface ratios. All in bulk, per section, through the owner maps.
2. **Conductances** (`@state`, every step, since the barriers and the lengths change): `K` per edge by type, from the inputs and parameters. The boundary terms per node.
3. **Axial conductance** per section: `Kx` (Poiseuille or prescribed), a SubOrgan state variable.
4. **Post-processing** of a solution (a function of ψ on the nodes):
   - the edges' flows;
   - per section, the soil uptake, `kr = Q_soil / (ψ_soil − ψ_xylem) / perimeter / length`, and the layer-wise uptake, as SubOrgan variables.

**Checked against the legacy on the same section:**
- the conductances, edge by edge;
- the matrix (assembled the legacy's way from the port's `K`, solved with `spsolve`);
- `kr` and `Kx`.

Until §3d exists, the port solves each section with a plain sparse solve, only in the comparison tests.

**3d. Then the `graph_system` (next phase):**
- **The equations:** a node balance `Σ K (ψ_j − ψ_i) = boundary terms` over the plant's graph view (`@graph_system`, `@node_balance`, `@edge_law` Q = K Δψ). The soil and xylem conditions are `boundary_set`s.
- **Over connected anatomies:** sections joined by the axial junctions, with their geometry from GRANAP. So the plant is solved at once, and kr and Kx come out per section from the same solve. See Q-M5 for the "standard" per-section kr.

## 4. Order of work

1. **The legacy copy, and the comparison harness:** the legacy run on a GRANAP section, its conductances, matrix, kr and Kx recorded.
2. **The geometry steps,** compared with the legacy's derived quantities.
3. **The conductances,** compared edge by edge, then the matrix.
4. **Kx and the post-processing,** compared on the legacy's own solution, then on a plain solve of the port's matrix.
5. **The graph system** (§3d), compared with 4 on single sections, then run on connected sections.
6. **Examples** from the tutorials, ported.

## 5. Questions

- **Q-M1 — exactness against the legacy and its problems (§2).** Port the legacy's behaviour exactly first, its problems included (a–i), so that the comparison holds to the last digit, then fix them one by one, each with its reviewed difference? Or fix them while porting?
  - **Recommendation:** exact first, then fix one by one (as GRANAP's Q8). A crash (e, the barrier-0 Kx) is fixed at once in the port, the comparison skipping that case.
  - → answer: yes to Recommendation
- **Q-M2 — the geometry MECHA rebuilds.** The legacy rebuilds GRANAP's junctions from positions, and recomputes its own lateral distance (the sum of the wall's membrane distances, not GRANAP's mean plus offset), a single wall thickness, and its own epidermis perimeter. On the DataStructure, use GRANAP's network as it is? Then the junctions, `lateral_distance` and the per-wall `wall_thickness` are GRANAP's, and the conductances differ from the legacy's. Or compute MECHA's own quantities from GRANAP's columns?
  - **Recommendation:** MECHA's quantities first, computed from GRANAP's columns in bulk, so the comparison holds. Then switch, as a reviewed change, to GRANAP's per-wall thickness and lateral distance, which are the better values.
  - → answer: compute Mecha's approach first, and record ways to optimize it exactly, and then to remove steps not influential to the results in later step.
- **Q-M3 — the parameters' form.**
  - Declare MECHA's scalars as metafspm parameters (`kw`, barrier values, `kmb`, `kaqp`, `kpl`, `pd_section`, `k_interc`, …, numeric, per plant)?
  - And its tables (per-tissue aquaporin factors, per-interface `fplxheight` and plasmodesmata factors) as object parameters? Or flattened into one numeric parameter each (`kaqp_endo_factor`, …)?
  - **Recommendation:** flattened numeric parameters. MECHA's tables are small (5 tissue factors, about 20 interfaces), and flattened parameters can then differ per plant and appear in the documentation like any parameter.
  - → answer: yes to recommendation
- **Q-M4 — maturity and barriers.** In MECHA a "maturity stage" is a (barrier code, height) pair, looped over in one run. In a growing plant, each segment has its own barrier state. Make the barrier code a SubOrgan input (given by a development model, or computed from the distance to the tip), with the height the segment's length?
  - **Recommendation:** yes, a SubOrgan input `barrier` (by `Development`). The legacy's loop over stages becomes one section per stage in the comparison tests.
  - → answer: yes SubOrgan input
- **Q-M5 — the standard kr and Kx.** MECHA's kr is defined by a standard solve: the xylem held at a pressure (Dirichlet) and the soil at another, each section alone. In a connected plant, the solved flows give an effective kr per section instead (uptake / (ψ_soil − ψ_xylem, local) / surface). Keep both: the standard one (each section solved alone, a restricted graph system) as MECHA's definition, and the effective one from the plant's solve?
  - **Recommendation:** both. The standard kr and Kx are the parameters a root architecture model (e.g. a Doussan-type root system model) takes; the effective ones are diagnostics of the coupled solve.
  - → answer: both, and this downstream part should be optional when just wanting to solve flows on the hydraulic graph
- **Q-M6 — what stays out of the first port.** MECHA's scenario loops (`h` over parameter combinations, BC scenarios), the osmotic and elongation terms, solute transport (`SoluteTransport`, `coupled_solver`), and the exports (paraview, graphml, plots). In metafspm, scenarios belong to the scene and to separate runs, not to the component. Port the hydraulics first (one parameter set, one boundary condition set per run), and leave the rest to later phases?
  - **Recommendation:** yes. Osmotic terms come next (they belong to the same balance), then solutes as their own component; the exports become plots of the DataStructure.
  - → answer: Agree with this plan order
- **Q-M7 — MECHA's own cellset XML input** (GRANAR files, without GRANAP). GRANAP's `CellSetOrgan` already reads GRANAR cellsets into the MPG. Drop MECHA's XML network builder from the port (the legacy keeps it), and read cellsets through GRANAP?
  - **Recommendation:** yes. One path into the DataStructure, and GRANAP's reader is the one that keeps the walls' topology.
  - → answer: yes, I want the more direct link through metafspm DataStructure, as intended metafspm, between granap and mecha, all other input and output options stay functional but optional

## 6. Your answers (2026-10-14) and what follows from them

- **Q-M1:** exact first, the legacy's problems included; then fixed one by one, each with its reviewed difference. The barrier-0 Kx crash is fixed at once, with the comparison skipping that case.
- **Q-M2:** MECHA's own geometry first (its junctions, lateral distance, thickness and perimeter, computed from GRANAP's columns). Exact optimisations are recorded as they appear, and the steps that do not influence the results are removed later.
- **Q-M3:** MECHA's parameters as numeric per-plant metafspm parameters, the tables flattened (one parameter per tissue factor and per interface).
- **Q-M4:** `barrier` is a SubOrgan input; the height is the segment's length.
- **Q-M5:** both the standard kr and Kx (each section alone) and the effective ones from the plant's solve. This downstream part is optional: solving the flows on the hydraulic graph needs neither.
- **Q-M6:** the hydraulics first (one parameter set and one boundary-condition set per run), then the osmotic terms, then solutes as their own component; exports as plots of the DataStructure.
- **Q-M7:** the direct link GRANAP → DataStructure → MECHA is the port's path. The other inputs and outputs (cellset XML through GRANAP's `CellSetOrgan`, exports) stay functional but optional.
- **Your additional requests:**
  - **Visual examples at each step,** beside the quantitative checks: the conductance graph (edges coloured by K over the section), then the solved potentials (ψ on the nodes) and the solved flows (Q on the edges), port and legacy side by side, in `example/comparison/`.
  - **A seedling example:** metafspm's soil–plant–atmosphere example (`examples/soil_plant_atmosphere` in metafspm), its anatomical scale populated by GRANAP and solved by the MECHA component. It comes once the graph system exists (step 5).
- **Where the original tests and tutorials go:** they exercise the legacy's own API (`Mecha`, `InData`, `NetworkBuilder`), so they move to `test/legacy/` and `example/legacy/`, run against `openalea.mecha.legacy`, until their ported versions replace them. Like `comparison/`, `test/legacy/` is skipped in CI.

## 7. Step 1 done (2026-10-14): the legacy copy and the harness

- **Layout:** `src/openalea/mecha/legacy/` (frozen at 2b995b2, imports rewritten), `test/legacy/`, `example/legacy/`, `test/comparison/legacy_run.py`.
- **The harness:** `legacy_section(organ, barriers, height)` runs the legacy on a GRANAP organ (main package, through its legacy exports) and records, per stage, the solution, matrix, `Kmb`, rhs, edge `K`, kr and Kx.
- **On GRANAP's monocot root (seed 0, barriers 1 and 3):** 2.8 s; 4146 walls, 2763 junctions, 1384 cells, 20394 conductances; kr 1.025e-4 and 3.975e-5 cm/hPa/d, Kx 1.236.
- **The legacy's junction rebuild (by position) only renumbers GRANAP's junctions:** a bijection, with the same walls on each junction. So the port uses GRANAP's junctions as they are, and the comparison maps them by position.
- **Environment:** the `mecha` env (yours) lacks `cv2`, which GRANAP imports. I ran in a clone of it, `mecha-port`, with `opencv-python-headless` added; `environment.yaml` already lists it.


## 8. Design note: steps 2–4 (geometry, conductances, Kx and post-processing)

**What the legacy computes before and after its solve** (read in `network_builder.populate_from_network`, `hydraulic_solver.HydraulicMatrixBuilder`, `Mecha.standard_water_flow` and `calculate_axial_conductance`):
- **Per section, geometry:**
  - units: mm → µm (×1000), areas ×1e6;
  - the soil-border walls (one membrane, to an epidermis cell) and the aerenchyma borders;
  - the border junctions (two border walls; length Σ wall length/4);
  - the lateral distance of a wall (Σ of its membranes' `dist`);
  - the gravity centre (mean of the endodermis cells' positions) and each wall's distance to it;
  - the epidermis perimeter, 2π·(mean epidermis wall distance);
  - the cell ranks: layers 40–49 in the cortex, 50–61 in the stele, from the plasmodesmata neighbours;
  - the plasmodesmata ratios at the cortex interfaces (`concentrated_*`);
  - the xylem, sieve and protosieve sets (cgroup), the passage and intercellular sets (tissue).
- **Per edge, the conductance K:**
  - wall: `kw_key·1e-4·((lat + h)·t − t²)/length`;
  - membrane: kaqp by cgroup, the cortex gradient on the wall's distance to the centre;
  - plasmodesmata: `kpl`, the interface factor, `fplxheight`, the barrier modifier.
- **Per node, the boundary terms:**
  - the soil Robin terms on the border walls and junctions;
  - `k_xyl` on the xylem cells, or `k_sieve` on the protosieve cells at barrier 0.
- **After the solve:**
  - the soil-interface flows, `kr = ΣQ_soil/(ψ_soil − ψ_xyl)/perimeter/h`;
  - the edge flows;
  - the layer-wise uptake (on the ranks' layers).
- **Kx:** Poiseuille over the xylem cells, `Σ area²/(8π·h·η)·h/1e4`, with η that of water at 10 °C.

**Proposal:**
- **Modules of the port, in `src/openalea/mecha/`:**
  - `geometry.py`: pure array functions, per section, from the DataStructure's arrays;
  - `conductance.py`: the K of every edge and the boundary terms, vectorised over all sections' edges at once;
  - `parameters.py`: the flattened numeric parameters (Q-M3) with their units;
  - `component.py`: `MechaHydraulics(FunctionalComponent)`.
- **Inputs** (by `"Anatomy"`, i.e. `GranapAnatomy`; plus `length` by `"Architecture"`, and `barrier` on the SubOrgan, Q-M4):
  - Compartment: `kind`, `cgroup`, `tissue`, `section_x`/`y`, `wall_length`, `cell_area`;
  - Connection: `network_path`, `edge_length`, `edge_dist`.

  The sections are the Compartments by owner (the DataStructure's `compartments_by_owner`).
- **Steps:**
  - `@postsegmentation _geometry`: runs only for sections whose anatomy changed. The anatomy signature is metafspm's, already used by GRANAP. The geometry is computed once per distinct anatomy and broadcast to the sections that share it, since GRANAP's class sections are identical in 2D.
  - `@state _conductances`: depends on `length` and `barrier`. The legacy's elementwise formulas run over every edge of every section in one numpy expression, so it is exact, in the same operation order.
  - `@state _axial_conductance` (Kx).
  - The post-processing is a function of ψ, run after a solve. It is optional (Q-M5).

  Until step 5, the solve is a plain `spsolve` per section, built from the DataStructure's incidence.
- **Written on the DataStructure** (declared state variables):
  - Compartment: `cell_rank`, `soil_boundary` (the soil term's K), `axial_boundary` (`k_xyl` or `k_sieve`), `psi`;
  - Connection: `conductance` K, `flow` Q;
  - SubOrgan: `perimeter`, `kr`, `Kx`, `uptake`.
- **Kept in the component's per-anatomy cache, not declared:** MECHA's lateral distances, the gravity centre, the wall distances to the centre and the plasmodesmata ratios. Intermediates, readable by the comparison through a function.
- **Exactness:**
  - The sums whose order matters (the gravity centre, the perimeter, the plasmodesmata ratios, the layer distances, Kx's Σ) are taken in the legacy's order. Its node order is GRANAP's MECHA index; its neighbour order is the order of the connections in GRANAP's network. Sequential sums use `np.cumsum` or loops.
  - Two-term sums (lateral distance, border-junction lengths) are order-free.
  - The ranks' sweeps assign within one iteration only cells whose test reads the previous iteration's ranks, so each sweep vectorises exactly.
- **The junctions** are GRANAP's (§7: the legacy's rebuild is a renumbering). The comparison maps the legacy's junction ids by position.
- **Visual examples:**
  - `example/comparison/compare_hydraulics.py`: legacy | port | difference, for K (edges coloured by log K, per path), ψ (nodes) and Q (edges, width and direction), on the monocot root at barriers 1 and 3;
  - `example/hydraulics/monocot_root.py`: the simple one (GRANAP + MECHA components, run, the three plots).

  The seedling SPAC example comes with step 5, since it needs the plant-wide solve.

**Order:**
- (2) geometry, compared quantity by quantity, exact;
- (3) conductances and boundary terms, edge by edge, exact; then the matrix and the solve;
- (4) Kx exact, kr and the edge flows on the legacy's solution (exact), then on the port's solve;
- the two examples.

Each step is committed with its comparison tests and a devlog entry.

**Decisions (with my recommendation):**
- **D1 — the matrix and solve comparison.** K and the boundary terms are bit-exact. The assembled matrix's diagonal is a sum whose order follows the legacy's networkx adjacency, and its node order differs (SuperLU then permutes differently). *Recommendation:* compare the matrix to 4 ulp and ψ, Q and kr to a relative 1e-12, rather than emulating networkx's order in the assembly. Everything before the solve stays bit-exact.
- **D2 — units on the DataStructure.** *Recommendation:* MECHA's own units for now (K in cm³ hPa⁻¹ d⁻¹, ψ in hPa, kr in cm hPa⁻¹ d⁻¹, Kx in cm⁴ hPa⁻¹ d⁻¹), declared in each variable's doc. The conversion to the plant models' units happens at the coupling (step 5), through the translator.
- **D3 — Kx at barrier 0** (the legacy crash, fixed at once per Q-M1). There the legacy fills the protosieve cells' Poiseuille terms (η of phloem sap), then leaves `K_xyl_spec` unbound. *Recommendation:* `Kx` = 0 at barrier 0 (no mature xylem), and a separate `Kx_phloem` = Σ of the protosieve terms. Alternative: `Kx` = that protophloem sum, in the conduit that carries the boundary at barrier 0.
- **D4 — height.** The legacy's stage height (µm) is the segment's `length` (m → µm). The comparison sets the segment length to 200 µm. No decision needed unless you prefer a separate `height` input.

**Your answer (2026-10-14):** D1–D4 agreed as recommended (matrix to 4 ulp, ψ/Q/kr to 1e-12 relative; MECHA's units for now; `Kx` = 0 and a separate `Kx_phloem` at barrier 0; height = the segment's length). opencv now in the `mecha` env, which the port uses (the `mecha-port` clone is removed).


## 9. Step 2 done (2026-10-14): MECHA's geometry from the DataStructure

- **`geometry.py`:**
  - `section_parts(ds)` gives each section's Compartments and Connections;
  - `section_of` reads its GRANAP variables;
  - `section_geometry` returns MECHA's derived geometry, in µm and cm: positions, wall lengths, border walls and junctions, the rebuilt wall–junction edges, lateral distances, gravity centre, distances to the centre, perimeter, ranks and layers, plasmodesmata ratios, and the cell sets (the network's and the cell manager's).
- **Bit-exact against the legacy's `NetworkBuilder`** (`test/comparison/test_geometry.py`, 9 checks × the monocot and dicot roots, 18 passed). The legacy's sum orders are reproduced:
  - cell order for the gravity centre and the layers;
  - wall order for the epidermis distances;
  - for the plasmodesmata ratios, the neighbour order that networkx's `Graph.copy()` produces: earlier nodes first, by node order, then the later ones in insertion order.
- **Exactness lesson:** numpy's scalar `x ** 2` calls libm's `pow` and can differ by 1 ulp from the arrays' `x ** 2` (computed `x*x`). So the walls' distances to the centre are computed per scalar, as the legacy does.
- **The component** `MechaHydraulics`:
  - inputs by `"Anatomy"`;
  - `cell_rank` (Compartment) and `perimeter` (SubOrgan) written by a `@state` step;
  - geometry cached per anatomy template (`test/component`, 3 passed).
- **Cost:** 10 ms per anatomy (4 ms extraction, 6 ms geometry), against 0.47 s for the legacy's `populate_from_network` on the monocot root.
- **Not reachable from GRANAP's presets:** the passage-cell and protophloem branches (no such tissues), the aerenchyma borders (roots' air spaces keep two membranes), and the "no exodermis" and "no pericycle" connection ranks. They are ported, but only covered by the code itself; they will be tested on cellset XML sections (Q-M7's optional path).

## 10. Step 3 done (2026-10-14): conductances, boundary terms, matrix and a plain solve

- **`conductance.py`:**
  - the legacy's rules: kw per barrier code (the 10 configurations), the wall interfaces' keys, the plasmodesmata factor and frequency tables, the cortex aquaporin gradient (`ratio_cortex`);
  - `edge_conductances` (walls, membranes, plasmodesmata), `boundary_terms` (soil; `k_xyl` or `k_sieve`), `section_matrix`.

  Vectorised over all the sections' edges at once, one parameter value per edge (its plant's), so plants can differ.
- **`solve.py`:** a plain per-section solve (`spsolve`) and the edge flows, until the graph system (step 5).
- **The component:**
  - **parameters,** per plant, flattened (Q-M3): kw and the barrier values, thickness, kmb, kaqp and its factors, ratio_cortex, k_interc, kpl, the two plasmodesmata factors the legacy actually applies (§2 a: the others resolve to 1), `fplxheight_default` and one `fplxheight_<tissue>_<tissue>` per interface (41), x_contact, k_xyl, k_sieve, the φ-thickening layers;
  - **inputs:** `barrier` (SubOrgan, by "Development") and `length` (the height);
  - **state variables:** `conductance` (Connection), `soil_conductance` and `axial_boundary_conductance` (Compartment).
- **Against the legacy** (`test/comparison/test_conductance.py`; monocot and dicot roots, barriers 1 and 3; 16 passed):
  - every edge's K and every soil term bit-exact;
  - the matrix to 4 ulp (D1);
  - the solution to 1e-12 of its range.
- **What the comparison showed about the legacy:** `solve_W` returns its matrix after removing the axial terms in place (`remove_xyl_phloem_BC`). It adds them on the network's xylem cells (by cgroup) and removes them on the cell manager's (by tissue): a third case of §2 l. The comparison applies the same removal.
- **Visual check:** `example/comparison/compare_hydraulics.py [dicot]`, legacy | port | difference of log K, ψ and log |Q|. On the monocot and dicot roots: ΔK = 0, |Δψ| ≤ 9e-10 hPa (of 5000), |ΔQ| ≤ 1.3e-15 cm³/d (of 9e-5).
- **Cost:** on 10 sections (83k nodes, 204k edges), the conductance step takes 0.2 s per call; the geometry is cached per template.

## 11. Step 4 done (2026-10-14): the standard problem, kr, Kx and the flows

- **`postprocess.py`:**
  - `soil_uptake` (Σ K_soil (ψ_soil − ψ), border walls then border junctions, in the legacy's junction-id order);
  - `radial_conductivity` (kr);
  - `axial_conductances` (Kx and Kx_phloem by Poiseuille; Kx = 0 at barrier 0, decision D3).
- **The legacy's sums are numpy's.** `mecha_class` star-imports pylab, so its `sum` is numpy's pairwise sum (kr's total uptake, Kx's Σ). The port does the same, on arrays laid out as the legacy's.
- **The legacy's rebuilt junction ids** (in order of first appearance in its wall scan) are reproduced by `geometry.junction_order` (tested).
- **The component's `@state _psi` step:** MECHA's standard problem on each section alone. It writes `psi` (Compartment) and `flow` (Connection), plus `uptake`, `kr`, `Kx` and `Kx_phloem` (SubOrgan). It is optional (`standard_solve = 0`, Q-M5). The standard potentials are parameters (`standard_psi_soil` 0, `standard_psi_xylem` −5000 hPa, `standard_psi_sieve` NaN as the legacy's default: at barrier 0 the section is then at the soil's potential and kr is NaN).
- **Against the legacy** (`test/comparison/test_postprocess.py`; monocot and dicot, barriers 1 and 3; 16 passed):
  - on the legacy's solution, kr and every edge's flow are bit-exact;
  - Kx is bit-exact;
  - the port's own solve gives ψ, kr and the flows to 1e-12.
- **Component tests:** ψ between the soil's and the xylem's potentials, each section's uptake equal to its flow into the xylem, kr falling with suberin, Kx, the optional switch (`test/component`, 11 passed).
- **Examples:**
  - `example/hydraulics/monocot_root.py`: three 5 mm segments at barriers 1, 3 and 4, GRANAP's and MECHA's components, one run, the K, ψ and Q of each section;
  - `example/comparison/compare_hydraulics.py`: now draws the component's own ψ and Q, and prints both kr values.
- **Deferred:** the layer-wise outputs (uptake per cortex/stele layer, transmembrane fractions) need the legacy's `create_layer_discretization` (200 lines). They are part of the optional downstream (Q-M5) and come after step 5.
- **For the later fixes (the legacy's, not the port's):** `k_xyl` is one conductance per vessel, whatever the section's height. In a 5 mm section the membranes conduct 25 times more than in a 200 µm one, so the vessels no longer hold the stele at the xylem's potential. kr then depends on the height: 4.5e-5 at 5 mm against 1.0e-4 cm hPa⁻¹ d⁻¹ at 200 µm (barrier 1). A height-scaled (or Dirichlet) xylem condition would remove this; the graph system (step 5) imposes ψ through the plant's xylem instead.

## 12. Design note: step 5, the graph system over connected sections, and the seedling SPAC example

**Where we are:**
- Each section has MECHA's conductances on its edges (step 3) and is solved alone in MECHA's standard problem (step 4).
- GRANAP joins successive sections by axial junctions (`network_path` 5), between their conducting cells: xylem with xylem, phloem with phloem. Their `edge_length` is the distance between the two sections' centres, and the junction rule gives their geometry.
- MECHA gives these junctions no conductance yet (0).
- metafspm's `@graph_system` solves a node balance with an edge law over the DataStructure's graph view: here all the Compartments and all the Connections, junctions included. Its boundary sets can be `robin` (weight, external value), `dirichlet` or `neumann`.
- metafspm's SPAC example (`PlantWaterTransport`) is such a system for a seedling in MPa and mm³ s⁻¹: the soil and the air as Robin conditions on the root epidermis and the stomatal cavities, coupled to a soil grid by translators.

**Proposal:**
- **5a. The axial junctions' conductance** (`_conductance` step): Poiseuille through each junction's two cells, each over half its own segment's length, in series:
  `1/K = 1/K_a(L_a/2) + 1/K_b(L_b/2)`, with `K(h) = A²/(8π h η)·1e-12` (the legacy's per-vessel term).
  For the xylem, η is water's, and the conductance applies only when both sections have mature xylem (barrier > 0; at barrier 0, Kx = 0, D3). See Q-G1 for the phloem junctions.
- **5b. The plant problem** (`@graph_system`, Newton, linear so one iteration, `schedule_as="state"`, after the conductances):
  - unknown ψ on every Compartment; edge law `Q = K (ψ_a − ψ_b)`; node balance `Σ Q_out = boundary inflows`;
  - **soil:** `boundary_set(kind="robin", weight="soil_conductance", value="soil_psi")`, with `soil_psi` an input by "Soil" (Compartment, hPa), broadcast from the segment or mapped from a soil grid by the scene;
  - **upper boundary:** the collar section's xylem vessels (`is_collar_xylem`), Q-G2.
- **5c. Outputs:**
  - `psi` and `flow` become the plant problem's;
  - the standard problem's (step 4) are renamed `standard_psi` and `standard_flow`;
  - kr and Kx keep MECHA's definition (standard problem, Q-M5);
  - new per section: `uptake` from the plant solve and `kr_effective` = uptake / (ψ_soil − mean ψ of the section's xylem) / surface (Q-M5's effective kr).
- **5d. Checks:**
  - a single section with its xylem held at ψ_xyl through `k_xyl` reproduces step 4's standard problem (same matrix: to 1e-12);
  - two and more connected sections conserve mass (Σ uptake = collar flow);
  - a uniform root behaves like a 1D cable (ψ along the xylem falls as the analytical solution of kr and Kx).
- **5e. The seedling SPAC example** (`example/soil_plant_atmosphere/`, after metafspm's): the same scene (soil grid, water table, air), but the roots' anatomies are GRANAP's and their hydraulics MECHA's. See Q-G3 for the shoot. Plots: the plant at SubOrgan scale, the anatomies coloured by ψ, the soil slice, kr and kr_effective along the roots.
- **Units:** MECHA's (hPa, cm³ d⁻¹) on its variables (D2). The coupling to the SPAC's soil (MPa, mm³ s⁻¹) goes through translator factors (1 MPa = 1e4 hPa; 1 cm³ d⁻¹ = 1/86.4 mm³ s⁻¹).

**Questions (with my recommendation):**
- **Q-G1 — the phloem junctions in the water problem.** In MECHA's water problem the phloem only matters at barrier 0, as the standard problem's axial boundary. *Recommendation:* K = 0 on the phloem junctions for now; water then moves axially in the xylem only, and the phloem's own flow comes with the osmotic phase (Q-M6). Alternative: Poiseuille with η of phloem sap, which would carry water to and from the young (barrier 0) sections through the protophloem.
  answer: Yes to recommendation, but I want the metafspm component logic to be used further then, with axial conductances being defined from a decorated step, working with the filter argument, both of xylem and phloem axial conductances. By the way I don't get why now the solve is not declared as a @graph_system using the results of the Pre-processing
- **Q-G2 — the plant problem's upper boundary.** *Recommendation:* a boundary set on the collar section's xylem vessels, of a kind given by a parameter: Dirichlet `collar_psi` (a root system alone, the standard problem's analogue at plant scale) or Neumann `collar_flow` (a transpiration demand split among the vessels in proportion to their Kx). In the seedling, neither: the collar is joined to the shoot (Q-G3)
answer: yes to the design idea.
- **Q-G3 — the seedling's shoot.** MECHA's rules are those of roots: its only boundary is the soil, at the epidermis. GRANAP generates stems and leaves, but MECHA has no rule for the air (stomata, cuticle) nor for the leaves' air spaces (`wall_air` and `air_link` edges are ignored, §2 i). *Recommendation:*
  - roots: GRANAP anatomies with MECHA's hydraulics;
  - shoot: kept as the SPAC example's (its simplified anatomy, its conductances, its evaporation at the stomatal cavities), joined to the root collar's xylem by axial junctions;
  - the one graph system then covers the whole plant, reading `conductance` on every edge, whoever wrote it.

  Alternatives:
  - (b) GRANAP stems and leaves too, with new MECHA rules for the shoot: cuticle, evaporation from the leaf air spaces, the `wall_air` and `air_link` edges. A model extension, better after the legacy problems are fixed.
  - (c) roots only, the shoot as a collar transpiration demand (Q-G2's Neumann).
  answer: yes to recommendation, Mecha will later be specialized to initialize more precise conductances for the stem and shoot elements. But you can still begin to specialize by transposing parameter from the root for common transmembrane, apoplastic and symplastic pathway to give realistic first estimations, so not just directly from the SPAC example.
- **Q-G4 — where the graph system is declared.** *Recommendation:* in `MechaHydraulics` itself, so the component solves its own plant problem; the SPAC example then uses it in place of `PlantWaterTransport`. For that, the shoot's conductances and the evaporation must reach MECHA's system as inputs: `conductance` on non-MECHA edges, and an `evaporation_conductance` with the air's ψ. Alternative: MECHA only provides the conductances and the soil terms, and metafspm's generic `PlantWaterTransport` solves. That is simpler, but then MECHA has no plant-scale solve of its own.
answer: yes to the recommendation of design, so that answers my question in Q-G1 relative to the graph_system

**Order once agreed:** 5a (junction conductances, tested against Poiseuille and Kx), 5b–5d with a root alone (Dirichlet collar), then 5e the seedling.
yes to order

## 13. Your answers to Q-G1–Q-G4 (2026-10-14), and how I read them

- **Q-G1:**
  - water moves axially in the xylem only;
  - both axial conductances, the xylem's and the phloem's, are computed, each by a decorated step selecting its junctions with `filters=` (an edge descriptor `axial_conduit`: 1 xylem, 2 phloem);
  - the water problem reads the xylem's; the phloem's waits for the osmotic phase.
- **The solves become `@graph_system`s**, using the pre-processing's results (your remark on Q-G1, answered by Q-G4). There are two systems in `MechaHydraulics`:
  - `_standard_problem`: each section alone, the legacy's kr; the junctions carry no conductance in it, so the sections are independent blocks of one system;
  - `_plant_problem`: the sections joined by the xylem junctions.

  The plain `spsolve` of step 4 (`solve.py`) stays only as the comparison's reference path.
- **Scheduling:**
  - the pre-processing (geometry, conductances, junctions) at `@rate`;
  - the two systems at `state`;
  - the post-processing (uptake, kr, kr_effective, Kx) at `@totalstate`.
- **Q-G2:** the collar condition as a boundary set with a per-node `kinds` variable (`collar_kind`: 1 Dirichlet ψ, 2 Neumann flow) on the collar section's xylem vessels.
- **Q-G3:** the shoot's anatomies are GRANAP's, and their conductances MECHA's rules with the roots' parameters as first estimates (walls, membranes, plasmodesmata), to be specialised later. Specifically:
  - no soil contact on stems and leaves;
  - evaporation from the leaves' stomatal air spaces, as the SPAC example's (a Robin condition with the air's ψ), its conductance from GRANAP's stomatal cells.

  I will detail this when I reach 5e.
- **Q-G4:** the graph systems are declared in `MechaHydraulics`.
- **Order:** 5a, then 5b–5d on a root alone, then 5e.

## 14. Steps 5a–5d done (2026-10-14): the junctions, and the two graph systems

- **Pre-processing (`@rate`):**
  - geometry and conductances, as before;
  - `_axial_conduit` (edge descriptor: 1 xylem, 2 phloem junction);
  - `_xylem_axial_conductance` and `_phloem_axial_conductance`, two steps with `filters={"axial_conduit": 1 | 2}`: Poiseuille through the junction's two cells, each over half its segment, in series; the xylem's 0 where either section has no mature xylem (barrier 0);
  - `_collar_kind`: the collar condition on the first root section's vessels (parameters `collar_condition` 1 potential or 2 flow, `collar_psi`, `collar_flow`; the flow split as the vessels' area²).
- **`_standard_problem`** (`@graph_system`, `state`):
  - ψ `standard_psi`; balance `B diag(conductance) Bᵀ ψ`; the junctions carry none, so each section is its own block;
  - robin sets: the soil (`soil_conductance`, `standard_soil_psi`) and the axial boundary (`axial_boundary_conductance`, `standard_axial_psi`; none where the axial potential is NaN);
  - analytic sparse Jacobian; `standard_flow` as a graph output;
  - restricted by `filters={"in_standard_problem": ">0"}` (standard_solve, Q-M5).
- **`_plant_problem`** (`@graph_system`, `state`):
  - ψ `psi`; balance with `conductance + xylem_axial_conductance`;
  - robin soil (`soil_psi`, an input by "Soil"); the collar set with per-node kinds (`collar_kind`, `collar_value`);
  - analytic sparse Jacobian; `flow` as a graph output.
- **Post-processing (`@totalstate _kr`):** `kr`, `standard_uptake` (the legacy's operations), `uptake` and `kr_effective` (plant problem; NaN at barrier 0), `Kx`, `Kx_phloem`.
- **Renamed:** the standard problem's outputs are now `standard_psi`, `standard_flow`, `standard_uptake`; `psi` and `flow` are the plant problem's. The plain-solve module `solve.py` is removed (the comparison builds the legacy's matrix itself).
- **metafspm (71340de):** a `@graph_jacobian` may return a sparse matrix, kept sparse, the boundary sets' terms added sparse. The dense path was out of reach for 10⁵ nodes, and the sparse one failed on robin sets.
- **Tests** (69 passed):
  - the comparison tests now run on the graph-system standard problem (ψ, kr, flows to 1e-12 of the legacy);
  - `test/component/test_plant_problem.py`: Poiseuille in series; Σ junction K between identical sections = Kx·1e4/h to 1e-12; no water through immature xylem; mass balance (collar extraction = Σ uptake) with a potential and with a flow at the collar; ψ between the collar's and the soil's; xylem ψ rising and uptake falling away from the collar.
- **Cost:** 0.5 s per call on 10 sections (83k nodes, 204k edges); the two systems take 0.1 s together.
- **Example `example/hydraulics/root_axis.py`:** six 5 mm sections (barriers 3, 3, 3, 1, 1, 0 from the collar), soil −1000 hPa, collar −5000 hPa. It draws the xylem ψ along the axis, the uptake per section, kr against kr_effective, and three sections coloured by ψ with their flows.
  - **kr_effective (7.3e-5) exceeds MECHA's kr (4.5e-5) in the barrier-1 sections:** in the plant the stele is held by its axial connection, not by the fixed `k_xyl`. This is the height dependence of §11, seen from the plant.

## 15. Design note: step 5e, the seedling in the soil–plant–atmosphere continuum

**What a first run of MECHA on GRANAP's seedling showed** (GRANAP's `example/mpg/seedling.py`: 3 stem elements, 4 leaf elements, 5 root segments; 212k Compartments, 526k Connections):
- **Building it was slow, in metafspm (fixed, 4051fba).** Reading a coarse-scale property onto the Compartments was quadratic (each Compartment mapped to its owner one at a time): constructing GRANAP's component took 477 s, now 0.6 s; its call took 10 s, now 0.8 s.
- **MECHA's rules break on the shoot sections:**
  - stems and leaves have no endodermis, so the gravity centre, the distances to it and the perimeter are NaN, and with them the cortex aquaporin term, and the solve diverges;
  - every epidermis border wall would be in contact with the soil.
- **GRANAP's shoot tissues and MECHA cell groups:** epidermis 2; xylem 13; guard cells 12; parenchyma 5; mesophyll, sclerenchyma, the leaves' sieve elements and companion cells, air spaces and the stomatal pores 4.
- **In the monocot leaf, each of the 20 stomatal pore cells** is joined to its walls by membranes and to its guard cells by plasmodesmata. The 33 air spaces have membranes and plasmodesmata; their 29 `wall_air` edges are extra links (ignored, §2 i).

**Proposal (your Q-G3 answer: the shoot's conductances from MECHA's rules with the root's parameters, as first estimates):**
- **E1 — the shoot sections in MECHA's geometry:**
  - without an endodermis, the gravity centre is the mean of all the cells' positions (root sections unchanged, so the comparison with the legacy holds);
  - kr, kr_effective and the perimeter are root quantities, NaN on stems and leaves;
  - soil contact only on root sections (the SubOrgan label RootSegment): stems and leaves have no exchange through their epidermis (the cuticle).
- **E2 — the shoot's barrier code:** 1, meaning mature xylem (its membranes ×1e4, no plasmodesmata to it). The Casparian rules need an endodermis, which the shoot has none of. The example's development stand-in gives the roots their codes from the distance to the tip: 0 on the tip segment, 1 up to 3 cm, 3 beyond.
- **E3 — evaporation from the leaves, at the stomatal pores** (tissue 'pore'), as the SPAC example's: a robin set, ψ_air (an input by "Atmosphere", default −93.9 MPa = −9.39e5 hPa, 50 % RH at 20 °C) with weight
  `K_vap = g_s · A · vapour_factor`, converted to cm³ d⁻¹ hPa⁻¹ (×8.64e-3 from mm³ s⁻¹ MPa⁻¹).
  `A` is the leaf surface each pore serves: the section's outer epidermis (Σ length of its border walls × height) shared among its pores. The parameter is `stomatal_conductance` (mol m⁻² s⁻¹, 0.2 as the SPAC's). MECHA writes `evaporation_conductance` (node) and the graph output `evaporation`.
- **E4 — the seedling:**
  - the SPAC example's architecture (stem of 3 phytomers with a leaf of 3 elements each, 3 root axes with laterals, their 3D positions), without its anatomies;
  - GRANAP's anatomies: roots by the root preset, its classes by diameter and distance to the tip; stems by the monocot stem preset at a 1 mm radius (2.5 mm gives 45k Compartments per section); leaves by the monocot leaf preset;
  - MECHA's component, collar_condition 0 (the shoot above): the plant problem then goes from the soil through the roots, the stem and the leaves to the air.
- **E5 — the coupling to the soil, as the SPAC scene's:**
  - the same `Soil` grid model, the `root_surface` mask (MECHA's soil-contact nodes), a `CrossMapping` by overlap of the segments' 3D extents (`x1`…`z2`, broadcast to the Compartments);
  - the translators carry units: `soil_psi` (hPa) ← soil `water_potential` (MPa) ×1e4; soil `plant_uptake` (mm³ s⁻¹) ← MECHA's `root_uptake` (cm³ d⁻¹, a new node graph output) ×1000/86400;
  - the scene iterates to the plant–soil fixed point as the SPAC's (`largest_change`).
- **E6 — outputs and plots** (`example/soil_plant_atmosphere/`): the plant at SubOrgan scale coloured by mean ψ, the anatomies (one per organ type) by ψ, the soil slice, transpiration = uptake, kr and kr_effective along the roots.

**Questions (with my recommendation):**
- **Q-E1 — the evaporating sites:** the pores, as above (*recommended*)? Or every leaf epidermis border wall with a cuticular conductance besides (a second robin set)? The latter can come later.
- **Q-E2 — the seedling's size:** the SPAC seedling has about 48 segments, about 0.5–1 M Compartments with GRANAP anatomies. *Recommendation:* keep it whole, if the plant solve stays within a few seconds per scene step (to measure). Else halve the root axes' segments.
