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

## 16. Step 5e done (2026-10-14): the seedling in the soil–plant–atmosphere continuum

Your answer: E1–E6 and the recommendations of Q-E1 (the pores) and Q-E2 (the whole seedling).

- **MECHA:**
  - **E1:** gravity centre of all cells without an endodermis (root sections unchanged: the comparison still passes); soil contact, perimeter, kr, kr_effective and the standard problem on root sections only (labels read once per topology, `section_roles`).
  - **E3:** `evaporation_conductance` on the stomatal pores (g_s × served leaf surface × vapour factor), the robin set `air` (input `air_psi`, by "Atmosphere"); graph outputs `root_uptake` and `evaporation` (node, extensive: the scene exchanges them).
- **`example/soil_plant_atmosphere/`:**
  - `seedling_spac.py`: metafspm's seedling architecture (48 segments, 3D extents), GRANAP's anatomies (stems 1 mm), a development stand-in (barrier 0 on the tip segment, 1 up to 10 cm, 3 beyond), `SeedlingHydraulics` (GRANAP + MECHA), the scene translator with the unit factors, metafspm's `Soil` model reused from its example;
  - `one_plant.py`: the scene;
  - `plots.py`: the plant over the soil, one section per organ by ψ, ψ along the path, the roots' kr.
- **Result:**
  - 524k Compartments, 1.29M Connections; built in 35 s, converged in 6 steps (51 s);
  - transpiration = root uptake = the soil's sink = 0.148 mm³ s⁻¹ (the SPAC example's simplified plant: 0.076);
  - ψ along the path (MPa): soil at the roots −2.11, root xylem −3.44, stem xylem −3.53, leaf xylem −3.59, leaf cells −3.66, stomatal pores −4.44.
- **For your review:**
  - **The leaves are larger than the SPAC's.** GRANAP's monocot leaf section is about 4 mm wide, against the SPAC's 5 mm leaf_width counted as a strip, so transpiration is about twice the SPAC's. The soil around the roots then dries to −2 MPa (its conductivity is the SPAC's).
  - **The tip segments (barrier 0) carry no water axially** (D3): they stay at the soil's potential.
  - **A 3 cm suberin zone made most roots nearly impermeable** (root xylem −5.5 MPa); hence 10 cm in the stand-in.
- **Tests:** `test/component/test_shoot.py` (a stem with a leaf and a root: soil only at the root, air only at the pores, uptake = transpiration, root quantities on roots only); 72 passed.

## 17. Step 6 done (2026-10-14): the tutorials ported

- **`example/tutorials/tutorial_01_input.py [cellset]`:**
  - an anatomy into MECHA through GRANAP's component: the root preset, or a CellSet file (GRANAP's `CellSetOrgan`; the legacy's own XML builder is not ported, Q-M7);
  - kr and Kx at barriers 1 and 3;
  - the tutorial's views from `openalea.mecha.plots`: tissues, ψ map, radial ψ profile, conductances by pathway, the pathways' shares of the radial flow (the legacy's algorithm), flow vectors.
- **`example/tutorials/tutorial_02_bc.py`:** the standard problem with the xylem's potential, then with its uptake as a flow into the xylem. New parameters `standard_axial_condition` (1 potential, 2 flow) and `standard_axial_flow`; a neumann set; the xylem's potential recovered as the legacy does.
- **`example/comparison/tutorial_01_input.py [cellset]` and `tutorial_02_bc.py`:** the same sections through the legacy (on the same GRANAP organ) and the port. The numbers are side by side, and the views are drawn from both by the same functions (`legacy_frame` in `test/comparison/legacy_run.py`).
  - GRANAP root: kr and Kx equal to the printed 7 digits;
  - CellSet: kr 1.470809e-4 and 4.742011e-5, Kx 4.668696e-5, equal;
  - tutorial 02: uptake and kr to ≤ 1.2e-12.
- **Tests** (79 passed):
  - `test/comparison/test_boundary_conditions.py`: the flow case's ψ to 1e-11 of its range (only the soil anchors it, a less well-conditioned system), its uptake and kr to 1e-12 and 1e-10;
  - a component test of the flow condition.
- **A 15th legacy problem (o):** the flow is split by the area ratios of the cell manager's xylem (by tissue: protoxylem, metaxylem, xylem), but given by rank to the network's xylem cells (by cgroup, in cell order). In the monocot root the two orders differ, so vessels get another vessel's share. The port reproduces it, for exactness.
- **Tutorial 03 (diffusion) is not ported:** solute transport is the later solute component (Q-M6). Its legacy notebook stays in `example/legacy/`.
- **Found in GRANAP:** `CellSetOrgan` misreads GRANAR's own XML output (`example/legacy/inputs/current_root.xml`, in mm, `<granardata>`). With `scale=1` it finds no cortex nor endodermis, and 4683 pericycle cells for a 1055-cell file. On it, the legacy MECHA crashes (no cortex–cortex plasmodesmata: a division by zero), and the port gives kr ≈ 0 at barrier 3. The tutorial uses GRANAP's own CellSet test file (a Brassicaceae root) instead.
  - **Fixed in GRANAP (your request):** three causes. (1) GRANAR's `<cellgroups>` give several names per id (4: cortex, then `inter_cellular_space`, `aerenchyma`; 3: endodermis, then `passage_cell`; 11: phloem, then `cambium`), and the reader kept the last one; it now keeps the first. (2) A `<granardata>` file is in mm: the scale now defaults from the root element (1, or 0.001 for `<cellsetdata>`). (3) `regenerate=("stele",)` rebuilt each of the 168 resolved stele cells with a donor stele (the thousands of pericycle cells); it now defaults to `()` for a `<granardata>` file, whose stele is resolved.
  - **`tutorial_01_input.py granar`** (port and comparison) now runs the legacy tutorial's own section through GRANAP: kr 9.115400e-5 and 3.550639e-5, Kx 1.028664, legacy on the GRANAP organ and port equal.
  - The legacy reading the file with its own XML builder (the original tutorial's path) gives kr 9.111275e-5 and 3.549032e-5, Kx 1.026006: the same topology (3162 walls, 2108 junctions, 1055 cells) and wall lengths, but its cell areas are polygons through the wall midpoints and junctions (0.77 % less area than the true outlines GRANAP uses: Kx, by Poiseuille), and its wall positions move the mean epidermis distance by 0.05 % (the perimeter, in kr). A difference between the legacy's two readers, not the port's.

## 18. Design note: the legacy fixes (Q-M1, second half), and what comes after

**Your direction (2026-10-16):**
- Start with the legacy fixes, with this note first.
- Steps 4 and 5 of the later phases (osmotic terms, solutes) will come from a more recent version, your colleague's `reaction` branch. It will be made available and integrated in the port directly, instead of porting the legacy's `SoluteTransport`.

**How a fix is checked.** The legacy stays frozen, so after a fix the port and the legacy no longer agree everywhere. Each fix's comparison test asserts two things:
- outside the fix's scope (the edges, nodes or sections it concerns), everything stays bit-exact;
- inside it, the new value is the one the fix's rule gives.

The difference is then reviewed on the tutorials' sections (kr, Kx, ψ maps), and recorded here.

**The reference for "fixed":** the original MECHA (`src_old/6_MECHAv4_SRS2025.py`, with its GRANAR project inputs `src_old/Projects/GRANAR/in/`), wherever the refactored legacy departs from it.

### 18a. Fixes with no decision to make (in this order)

- **A1 — (f), the protophloem counted twice** in the phloem boundary term (`boundary_terms`). Counted once. It only matters at barrier 0 with a phloem potential given (`standard_psi_sieve`, NaN by default: no effect by default).
- **A2 — (o) and (l), one set of xylem vessels:** the network's (by cgroup, in cell order), wherever the port still uses the cell manager's (by tissue): the flow condition's shares (`_standard_axial_inflow`), and the xylem potential recovered after a flow condition (`_kr`). It only changes the flow-condition case (tutorial 02): each vessel gets its own share.
- **Nothing to fix in the port:**
  - (c) the flow condition read from the input files: the port has its own (`standard_axial_condition`), and the XML inputs are not ported;
  - (e) the barrier-0 Kx: fixed at step 4 (D3);
  - (g), (h), (j), (m): the legacy's code only (lists used as scalars, scenario loop, the shipped XML, debug prints);
  - (k): the port calls `phi_thick_cells` with its arguments in order;
  - (n): the dead ghost-wall bookkeeping was not ported;
  - the two xylem sets of the legacy's boundary-term removal (§10): only in the matrix the legacy returns; the comparison keeps emulating it, the port never removes anything.

### 18b. The plasmodesmata: (a) and (b) are more than key names

Compared with the original's radial plasmodesmata rules (`src_old`, lines 2967–3062), the legacy's tables (`interface_kpl_factor_map`, `fplxheight_map`) differ in:

- **The factors, beyond the mismatched keys:**
  - the original applies `cortex_factor` to outer layer–cortex (harmonic with 1), cortex–cortex (as is) and cortex–endodermis (harmonic). The legacy: never (key not found);
  - `PPP_factor` to the phloem-pole pericycle cells only (pericycle cells joined to a sieve tube by plasmodesmata, `PPP`), at their interfaces with the endodermis, stele, companion cells (harmonic with `PCC_factor`) and sieve tubes. The legacy: never;
  - `PCC_factor` to stele/xylem–companion, companion–companion and companion–sieve. The legacy: only the last two;
  - `PST_factor` to no radial interface in the original. The legacy: to sieve–pericycle, sieve–xylem and stele–sieve;
  - `endo_in_factor` / `endo_out_factor` multiply every plasmodesma by the side of the endodermis its wall is on (`r_rel`). The legacy: never.
- **The factors' defaults:** 1.0 in the original's projects (GRANAR, maize_granar), 0.0 in the legacy without an XML file. The port follows the latter, so its companion–companion, companion–sieve and sieve interfaces have no plasmodesmata at all.
- **The frequencies:**
  - pericycle–pericycle: 7.0e5 in the original's projects, the default 8.0e5 in the legacy;
  - outer layer–cortex: the original takes the actual outer layer (`outercortex_connec_rank`: the epidermis where there is no exodermis), the legacy only exodermis–cortex;
  - stele–sieve and xylem–sieve: 9.0e5 in the original (at barrier 0, where the xylem is not yet a vessel), 0 in the legacy.
- **Xylem–xylem:** the original gives its high conductance only to the walls splitting one vessel (`Xylem_pieces`). The legacy gives it to every xylem–xylem interface.
- **Interfaces the legacy added** (transfusion tissue 17 and 18, protophloem 23, for needles): not in the original, which gives them the default frequency.

**Q-L1 — the plasmodesmata rules.**
- *Recommendation:* the original's rules and factors, as new per-plant parameters:
  - `kpl_cortex_factor`, `kpl_endo_in_factor`, `kpl_endo_out_factor`, `kpl_PPP_factor`;
  - `kpl_PCC_factor` and `kpl_PST_factor`, the latter kept for the 3D (axial) plasmodesmata.
- Their defaults: the original projects' 1.0, which reopens the phloem's plasmodesmata.
- The legacy's added interfaces (17, 18, 23) kept, since GRANAP generates those tissues.
- *Alternative:* fix only the keys and keep the legacy's interface lists and its 0.0 defaults.

### 18c. The section's height: one cell, or a segment of cells (D4 revisited)

**What the probe shows** (monocot root, standard problem, kr in cm hPa⁻¹ d⁻¹; the height is the segment's length, D4):

| height | barrier 1, k_xyl 1e-6 | barrier 1, k_xyl ∞ | barrier 3, k_xyl 1e-6 | barrier 3, k_xyl ∞ |
|---|---|---|---|---|
| 200 µm | 1.025e-4 | 1.062e-4 | 3.98e-5 | 4.03e-5 |
| 1 mm | 6.98e-5 | 7.91e-5 | 9.41e-6 | 9.56e-6 |
| 5 mm | 4.53e-5 | 7.31e-5 | 1.97e-6 | 2.00e-6 |

So `k_xyl` (§11) is the smaller part. Most of the dependence comes from the plasmodesmata: their conductance is `fplxheight × wall length`, with no height in it, while the walls, membranes and soil terms grow with the height. At barrier 3 the endodermis can only be crossed through them, so kr falls as 1/height.

**This is MECHA's intent for one cell.** The original's own comment: `Fplxheight` is "the plasmodesmatal frequency (per cm²) × the cell's axial length when the frequency was counted… these values conserve the quantity of plasmodesmata when cells elongate". In MECHA the height is one cell's length (`Defheight` = 200 µm in its projects), and an elongating cell keeps its plasmodesmata. In the port the height is a segment's length (D4), with many cells stacked along it: their plasmodesmata are missing.

Scaling `kpl` and `k_xyl` by height / 200 µm brings kr back near the 200 µm values (barrier 1: 1.03e-4, 9.73e-5 and 9.63e-5 at 200 µm, 1 mm and 5 mm; barrier 3: 3.98e-5, 3.92e-5, 3.91e-5). The rest comes from the end-wall terms (`lateral distance + height`, `height + dist`), which belong to one cell too.

**Q-L2 — the height.**
- *Recommendation:* a cell length ℓ, separate from the segment's length L. It starts as a per-plant parameter `cell_length` (200 µm, MECHA's `Defheight`), and can later be an input by "Development", for cells elongating near the tip. Each section's rules are then MECHA's for one cell of height ℓ (the legacy's values, exactly, when ℓ = 200 µm). Its radial conductances (walls, membranes, plasmodesmata, soil, the standard problem's axial boundary) are multiplied by n = L / ℓ, the cells stacked along the segment. Then:
  - ψ in the standard problem is that of one cell, its flows × n, and kr is independent of L;
  - the axial junctions keep their Poiseuille over L (§14), and Kx is per unit length already;
  - in the plant problem, radial and axial conductances both follow the segment.
- *Alternative:* keep D4 and scale only the plasmodesmata and `k_xyl` with L / 200 µm.

### 18d. The standard problem's xylem boundary

The original has two sources for the vessels' boundary conductance (`Kax_source`):
- 1, its projects' value: Poiseuille through each vessel over the section's height (area² / 8πηh);
- 2: prescribed `K_xyl` values.

The legacy always uses `k_xyl` = 1e-6, whatever the source. At 200 µm the Poiseuille values are over 10 times larger, which raises kr by up to 3.5 % (the k_xyl ∞ column above).

**Q-L3 — the xylem boundary.**
- *Recommendation:* a parameter `standard_axial_source` with the original's two choices, Poiseuille over ℓ (1) and `k_xyl` (2), and the original's default 1. kr's ψ_xylem is then the vessels' potential, up to the small resistance of one cell's length of vessel.
- *Alternatives:*
  - keep `k_xyl` as the default (the legacy's numbers);
  - impose ψ_xylem on the vessels (a Dirichlet set), the limit of both.

### 18e. The walls' thickness, (d)

MECHA uses one thickness (1.5 µm) for every wall. GRANAP gives each wall its own: 1 to 4 µm on the monocot root, median 2. Every wall and membrane conductance, and the soil terms, depend on it. Q-M2 already decided to switch to GRANAP's values as a reviewed change.

**Q-L4 — the default thickness.**
- *Recommendation:* a parameter `thickness_source`: 1 for MECHA's one value (`thickness`), 2 for GRANAP's per-wall `wall_thickness`. The default is 2, after the change in kr is measured and reviewed on the tutorials' sections, here.
- *Alternative:* keep 1 as the default, GRANAP's thickness as an option.

### 18f. Deferred

- **(i) `wall_air` and `air_link` edges** (the leaves' and needles' air spaces). They need rules MECHA does not have (vapour in the air spaces), so they belong to the shoot's specialisation (Q-G3), with the cuticular evaporation (Q-E1). *Recommendation:* defer them there.
- **Q-M2's other half** (the steps that do not influence the results removed, GRANAP's lateral distances used): after these fixes, since they change what is influential.

**Order:**
1. A1, A2 now.
2. Then, with your answers: Q-L1 (plasmodesmata), Q-L2 (height), Q-L3 (xylem boundary), Q-L4 (thickness). Each is its own commit and reviewed difference, in this order, since each changes the reference of the next.

### 18g. A1 and A2 done (2026-10-16)

- **A1:** `boundary_terms` gives each protosieve cell `k_sieve` once. The geometry keeps the legacy's list, with its duplicates, for the geometry comparison. GRANAP's presets have no protophloem, so the fix is tested on a stand-in network (`test/component/test_legacy_fixes.py`).
- **A2:** the flow condition's shares and the recovered xylem potential use the network's vessels (by cgroup), each with its own area fraction.
  - **Reviewed difference** (tutorial 02's flow case, monocot root, barrier 3): the uptake is unchanged (the given flow, 1.2e-12 from the legacy's); kr is 4.028609e-5 against the legacy's 4.029255e-5 (1.6e-4); ψ differs by up to 177 hPa (3.5 %), in the stele, where the misplaced shares were.
  - The vessels' potentials now spread over 56.5 hPa in the flow case, against 5.9 hPa when their potential is imposed: a flow split by area is not what the section takes at a uniform potential. That is a property of the condition, not of the fix.
  - On the dicot root the two xylem orders agree, and the comparison stays exact. `test_boundary_conditions` keeps the exact checks there, and skips ψ on the monocot (documented).
- **Tests:** 80 passed, 1 skipped (`--ignore=test/legacy`, the frozen legacy's own tests, as in CI).
- **Next:** Q-L1 to Q-L4 wait for your answers.

## 19. Your answers to Q-L1–Q-L4 (2026-10-16), and Q-L1 done

**Answers:**
- **Q-L1, Q-L3, Q-L4:** yes to the recommendations.
- **Q-L2:** no. The height stays MECHA's logic (the segment's length, D4) for now, so that the comparisons do not change. Per-tissue cell heights come later.
- **Q-L3 then reads:** Poiseuille over the section's height (the original's `Kax_source` 1, with its height), not over a cell length.
- **Osmotic potentials:** to be included now. Your colleague works on solute transport and leaf transpiration (the `reaction` branch).
- **Recaps:** each says what changed in metafspm itself.

**How the legacy stays comparable:** each changed default keeps the legacy's choice as an option, which the comparison sets through `legacy_options(mecha)` (`test/comparison/legacy_run.py`). The comparison suite therefore still checks everything else bit-exactly. So far:
- `plasmodesmata_rules` = 2;
- `kpl_PCC_factor` = `kpl_PST_factor` = 0.

**Q-L1 done:**
- **`plasmodesmata_rules`:** 1 the original MECHA's (default), 2 the legacy's.
- **The original's rules** (`_plasmodesmata_original` in `conductance.py`), first match wins:
  - intercellular spaces (barrier > 0): 0;
  - the walls splitting a vessel (`xylem_pieces`): 10000 × default;
  - mature xylem: 0;
  - epidermis–exodermis;
  - outer layer–cortex, cortex–cortex, cortex–endodermis (`kpl_cortex_factor`, the aerenchyma concentration at barrier > 0);
  - endodermis–endodermis;
  - pericycle–endodermis, –stele and –sieve (`kpl_PPP_factor` on the phloem-pole pericycle);
  - stele–companion, pericycle–companion, companion–companion, companion–sieve (`kpl_PCC_factor`);
  - stele–sieve, stele–stele;
  - the tissues the legacy added (17, 18, 23), with its frequencies;
  - the default.

  Then × `kpl_endo_in_factor` (both cells endodermis or stele, or a section without xylem) or `kpl_endo_out_factor`.
- **Parameters:**
  - new: `kpl_cortex_factor`, `kpl_endo_in_factor`, `kpl_endo_out_factor`, `kpl_PPP_factor`, `kpl_PCC_factor`, `kpl_PST_factor` (the legacy's rules only), all 1.0;
  - removed: `kpl_phloem_companion_cell_factor` and `kpl_phloem_sieve_tube_factor`;
  - the 41 per-cgroup-pair frequencies become the original's 15 (`fplxheight_default`, `_epi_exo`, `_outer_cortex`, `_cortex_cortex`, `_cortex_endo`, `_endo_endo`, `_endo_peri`, `_peri_stele`, `_stele_stele`, `_stele_comp`, `_peri_comp`, `_comp_comp`, `_comp_sieve`, `_peri_sieve`, `_stele_sieve`, its GRANAR project's values) plus the 18 of the added tissues.
- **Correction to §18b:** the original reads `Fplxheight_peri_peri` but none of its radial rules uses it, so pericycle–pericycle keeps the default, as in the legacy.
- **Reviewed difference** (standard problem, barriers 0, 1 and 3). Only plasmodesmata edges change:
  - monocot root: stele–sieve (96 edges, 0 in the legacy) opened. kr +0.01 % (1.0253e-4 against 1.0252e-4 at barrier 1; 3.9757e-5 against 3.9753e-5 at barrier 3);
  - dicot root:
    - stele–sieve (91), sieve–sieve (108, the default frequency: the original has no rule closing it) and companion–companion (81) opened;
    - xylem–xylem (52): the stele's frequency at barrier 0 and none at barriers > 0, instead of 10000× everywhere;
    - kr −0.06 % (1.1764e-4 against 1.1771e-4) at barrier 1, −0.03 % at barrier 3.
- **Tests** (`test/component/test_legacy_fixes.py`):
  - every plasmodesmata edge of the dicot root (barriers 0, 1, 3, with and without xylem pieces) against a transcription of the original's code, with factors away from 1;
  - the same on its network with random cgroups, for the branches GRANAP's presets do not reach (phloem-pole pericycle, companion–sieve, pericycle–companion, an epidermis as outer layer, the newer CellSet groups).
  - 86 passed, 1 skipped.

### 19a. Q-L3 done: the standard problem's axial boundary

- **`standard_axial_source`:**
  - 1 (default, the original's `Kax_source` 1): each conduit's Poiseuille conductance over the section's height, water's viscosity in the xylem vessels, phloem sap's in the protosieve cells (barrier 0);
  - 2: the prescribed `k_xyl` and `k_sieve` (the legacy's choice, whatever its source; set by `legacy_options`).
- **The viscosity** is the port's `VISCOSITY_XYLEM` (1.3076e-3 Pa s, water at 10 °C, as the legacy's Kx), not the original's 1.0e-3 in its boundary term, so that the boundary and Kx agree.
- **The vessels' conductances** are 0.005 to 14 cm³ hPa⁻¹ d⁻¹ on the monocot root at 200 µm, against `k_xyl` = 1e-6. They hold the vessels near ψ_xylem, so the standard kr is now nearly the section's own.
- **Reviewed difference** (kr, cm hPa⁻¹ d⁻¹):

| root, height | barrier 1: Poiseuille / k_xyl | barrier 3: Poiseuille / k_xyl |
|---|---|---|
| monocot, 200 µm | 1.0619e-4 / 1.0253e-4 (+3.6 %) | 4.0295e-5 / 3.9757e-5 (+1.4 %) |
| monocot, 5 mm | 7.2992e-5 / 4.5304e-5 (+61 %) | 2.0045e-6 / 1.9714e-6 (+1.7 %) |
| dicot, 200 µm | 1.2120e-4 / 1.1764e-4 (+3.0 %) | 5.4089e-5 / 5.3370e-5 (+1.3 %) |
| dicot, 5 mm | 8.8157e-5 / 5.7670e-5 (+53 %) | 2.9899e-06 / 2.9372e-06 (+1.8 %) |

- **The height dependence that remains** (barrier 3 at 5 mm) is the plasmodesmata's, kept with the height logic (Q-L2: per-tissue cell heights later).
- **Tests:**
  - the boundary is checked against Poiseuille, and `k_xyl` with source 2;
  - the mass balance of the standard problem is now checked to the round-off of the boundary's terms (64 ε Σ K_ax |ψ_xylem|): with conductances of ~10, the flows into the vessels are small differences of large terms (imbalance 2e-8 of the uptake, at a node residual of 4e-11 cm³ d⁻¹, the solver's round-off).

### 19b. Q-L4 done: the walls' thickness

- **`thickness_source`:** 1 MECHA's one value (`thickness`, 1.5 µm; set by `legacy_options`), 2 each wall's own (default).
- **The new input `wall_thickness`** (by "Anatomy", µm) is GRANAP's: the sum of its two cells' wall layers, from the `AnatomyWriter`'s per-tissue table (cortex 1, endodermis 1.5, epidermis 2… µm), plus an outer layer on border walls. On the two roots it ranges from 1 to 4 µm, mean 2.1–2.2.
  - A junction's thickness, for its soil term, is the mean of the walls it joins.
  - Each wall edge and membrane takes its wall's.
- **GRANAP fix (0f6a91f):** its component declared `wall_thickness` and `edge_wall_thickness` in mm. Their values are the exporter's µm; the declarations now say µm.
- **Reviewed difference** (kr, cm hPa⁻¹ d⁻¹; Q-L1 and Q-L3 on):

| root | barrier 1: GRANAP's / one value | barrier 3: GRANAP's / one value |
|---|---|---|
| monocot | 1.1000e-4 / 1.0619e-4 (+3.6 %) | 4.0703e-5 / 4.0295e-5 (+1.0 %) |
| dicot | 1.3175e-4 / 1.2120e-4 (+8.7 %) | 5.5916e-5 / 5.4089e-5 (+3.4 %) |

- **Test:** each wall edge's and soil term's conductance against MECHA's formula with its own thickness, a junction's with its walls' mean.
- **Together, Q-L1, Q-L3 and Q-L4** move the standard kr at 200 µm from the legacy's 1.0252e-4 to 1.1000e-4 on the monocot root at barrier 1 (+7.3 %), and from 3.9753e-5 to 4.0703e-5 at barrier 3 (+2.4 %).

## 20. Design note: the osmotic potentials (Q-M6's next phase)

**Your direction (2026-10-16):** include the osmotic potentials now. Solute transport and leaf transpiration are your colleague's (the `reaction` branch), to be integrated later.

### 20a. What the legacy does (`Mecha.initialize_scenarios`, `water_flux`, `compute_edge_flows`)

- **Scenario 0 is the standard problem** (no osmotic term), which gives MECHA's kr. Each scenario ≥ 1 solves the same section again, its matrix with the xylem boundary re-added, plus:
  - **the osmotic term on the membranes:** `rhs_o[wall] += K σ (π_wall − π_cell)` and the opposite on the cell. The flow of a membrane is then `K [(P_cell − P_wall) + σ (π_cell − π_wall)]`: the unknown is the pressure potential P, and the water potential is P + π. Walls and plasmodesmata have σ = 0;
  - **the soil's potential across the section** (`psi_soil_left`, `psi_soil_right`, by the wall's relative x);
  - **at barrier 0, the elongation sink** (`elongation_BC`), and a phloem boundary (barrier 0 on the protosieve; barrier > 0 on the sieve tubes) when given.
- **The cells' osmotic potentials π_cell, by `os_hetero`:**
  - 0: `os_cortex` everywhere; companion cells at the mean of `os_cortex` and the sieve's;
  - 1, 2: fixed profiles (epidermis, exodermis, endodermis, pericycle, stele, and 8 cortex layers c1–c8 counted from the outer cortex, by the legacy's layer rows);
  - 3: the cortex at `os_cortex`, the stele at −5000 hPa, the endodermis between them;
  - 4: one value per tissue, given.

  The sieve tubes have `osmotic_sieve` (the stele's when NaN). Intercellular spaces take the soil's, with σ = 0. Mature xylem (barrier > 0) takes the xylem's, with σ = 0.
- **The walls' osmotic potentials π_wall** depend on which side of the endodermis the wall is: cortical walls take the soil solution's, stelar walls (barrier > 0) the xylem sap's.
  - The soil's: uniform, or from left to right (`osmotic_symmetry_soil` 1, by x_rel), or radial (2, by `r_rel` with a shape exponent).
  - The xylem's: radial between `osmotic_endo` and `osmotic_xyl`, or their mean.
  - With a diffusivity, an exponential profile in a convection velocity u. u is never computed (always 0), so the profile is flat.
- **The reflection coefficients σ, by `s_hetero` × `s_factor`:** uniform; the stele half; or the cortex half. Per tissue: epidermis, exodermis on its two sides, cortex, endodermis on its two sides, pericycle, stele, companion, sieve.
- **The geometry this needs, not yet in the port:**
  - the legacy's layer rows (`rank_to_row`, `row_outer_cortex`);
  - each wall's radial position relative to the endodermis (`r_rel`, from the rows' distances and the xylem's 80th-percentile distance) and its relative x (`x_rel`).

  To verify while porting: the legacy's `r_rel` reads `distance_from_center` by row, where the original reads its rows' distances (`Layer_dist2`). A possible 16th legacy problem.

### 20b. Proposed design

- **New state variables:**
  - `osmotic_potential` (π, hPa; Compartment: every cell, wall and junction);
  - `reflection_coefficient` (σ; Connection: the membranes, 0 elsewhere).

  Both are written by a `@rate` step `_osmotic_potential`, from:
  - `osmotic_source` 0: π = 0, σ = 0, nothing changes. *Default*: the results stay those of §19;
  - `osmotic_source` 1: MECHA's presets, the legacy's rules above, as per-plant parameters (`os_hetero`, `os_cortex`, `os_sieve`, the tissues' values for 4, the soil's and the xylem's profile parameters, `s_hetero`, `s_factor`);
  - `osmotic_source` 2: an input `solute_osmotic_potential` (by "Solutes", Compartment), where the `reaction` branch's solutes will plug in (π = −c R T, theirs), with σ from the presets.
- **In both problems:** the membranes' term `K σ (π_cell − π_wall)` is added to each node's balance in the graph systems (`@node_balance`; its Jacobian does not change, since π and σ do not depend on the unknown), and to the edge flows.
- **Unknowns and outputs:**
  - `psi` and `standard_psi` stay the unknowns; with π ≠ 0 they are the pressure potential;
  - a new `water_potential` (Compartment) = psi + π, for the plots, the upscaling and the leaves' evaporation (which is driven by the water potential).
  - With `osmotic_source` 0 they are equal.
- **The standard problem:** a parameter `standard_osmotic` (0 by default: MECHA's kr stays the definition without osmotic terms, the legacy's scenario 0; 1: with them, the legacy's scenarios ≥ 1). That gives the per-section comparison with the legacy's `water_flux`.
- **The soil's potential across the section** (`psi_soil_left`/`right`): not ported. In the plant problem the soil's potential is the soil model's, node by node (`soil_psi`). For the comparison, the scenarios are run with left = right.
- **Not in this step:**
  - the elongation sink (barrier 0) and a phloem boundary at barrier > 0: they are other source terms of the same balance, for after this step (or with the `reaction` branch, if it carries them);
  - the convection velocity u of the profiles (never computed in the legacy).

**Checked against the legacy:** `water_flux` scenarios with `os_hetero` 0–4 and `s_hetero` 0–2, at barriers 1 and 3: π, σ, the osmotic right-hand side, the solution and the edge flows, to D1's tolerances.

### 20c. Questions

- **Q-O1 — the source of π.** As above: `osmotic_source` 0 (none, default), 1 (MECHA's presets), 2 (an input by "Solutes"). *Alternative:* only the input, with the presets as an example-side helper writing it.
answer: rather 2, cause we'll have dynamic metabolism models wiring it later, but we must still compare to standard MECHA simulations now for the port.
- **Q-O2 — the unknown.** Keep `psi` as the unknown (the pressure potential once π ≠ 0), with a new `water_potential` = psi + π. *Alternative:* solve for the water potential directly and write the pressure as the derived one. The balance is then written in water potentials, with (1 − σ) π terms on the membranes; this is equivalent, but further from the legacy.
answer: first option
- **Q-O3 — the plant problem's apoplast π.** In the plant, should the cortical walls' π come from the soil model (an input `soil_osmotic_potential`, by "Soil"), and the stelar walls' from the xylem sap, the solutes' later? Or MECHA's presets, as in the standard problem, until the `reaction` branch? *Recommendation:* the presets now (`osmotic_source` 1), and the soil's and the solutes' inputs with `osmotic_source` 2, when the `reaction` branch comes.
answer: the preset now. The inputs from another model can later be provided and apply to nodes with filter and multiscale coupling if that is how the components will be wired.
- **Q-O4 — the leaves' evaporation:** driven by the pores' water potential (psi + π) rather than psi. *Recommendation:* yes. It equals psi while π = 0.
answer: yes to recommendation

## 21. Your answers to Q-O1–Q-O4 (2026-10-16), and the osmotic step done

**Answers** (in §20c):
- **Q-O1:** the input (by "Solutes"), since dynamic metabolism models will be wired to it, with MECHA's presets still available for the comparison now;
- **Q-O2:** `psi` stays the unknown, with `water_potential` = psi + π;
- **Q-O3:** the presets now. Other models' inputs later, applied to the nodes through filters and the multiscale coupling;
- **Q-O4:** evaporation driven by the water potential.

**Done:**
- **`MechaSections`** (`component.py`): the anatomy inputs, the barrier, the length and the per-anatomy geometry cache, shared by MECHA's components. `MechaHydraulics` derives from it.
- **`MechaOsmoticPresets`** (`osmotic.py`), a component on the same sections that writes `osmotic_potential` (nodes) and `reflection_coefficient` (edges), MechaHydraulics' "Solutes" inputs, from the legacy's scenario rules. Its pieces:
  - `legacy_rows` (the legacy's layer rows, transcribed);
  - `relative_positions` (r_rel, x_rel);
  - `wall_counts` (its elif chain);
  - `tissue_values` (os_hetero 0–4, s_hetero 0–2);
  - `section_presets` (per membrane, a cell's π the one its last membrane sets, in the legacy's wall order).

  It applies to root sections only: the presets are root rules, and shoot sections keep π = σ = 0. Its parameters are the legacy's default scenario's values (os_cortex −4800 hPa, sieve −10 000, soil solution −500, xylem sap −1800 with the endodermis' −4800, σ 1).
- **`MechaHydraulics`:**
  - the inputs `osmotic_potential` and `reflection_coefficient` (by "Solutes", 0 by default: nothing changes without a solute model);
  - a rate step `_osmotic_drive`: σ (π_tail − π_head) per edge (`osmotic_drive`), the standard problem's (`standard_osmotic_drive`, with the new parameter `standard_osmotic`, 0 by default), and the pores' `evaporation_psi` = air_psi − π;
  - both node balances and the flows are `K (Δψ + drive)`. The Jacobians do not change, since π and σ do not depend on ψ;
  - the evaporation is `g (ψ − evaporation_psi)`, its water potential's (Q-O4);
  - new outputs `water_potential` = psi + π, and `standard_water_potential` (+ π where standard_osmotic).
- **Against the legacy** (`test/comparison/test_osmotic.py`; monocot and dicot roots, barriers 1 and 3, five scenarios: os_hetero 0–4 with s_hetero 0–2 and s_factor 1, 0.7, 0.9; 80 passed):
  - **σ:** every membrane's, bit-exact;
  - **π:** every cell's and wall's, to 1e-13. The legacy's x range is GRANAP's cell seeds (`_cells_gdf['x']`), which the DataStructure does not carry; the port takes the cells' centroids. With equal left and right soil values this is one ulp of rounding; it would weigh a left–right soil profile;
  - **the solution:** to 1e-12 of its range;
  - **the edge flows** K (Δψ + σΔπ): to 1e-12 of K × the potentials they are differences of;
  - **the uptake:** to 1e-12.
- **Legacy quirks, reproduced and recorded for the fixes:**
  - **(p) the cortex layers of the presets:** the legacy finds a cortex cell's row with `rank in rank_to_row`, a test on the array's values (rows), not its indices (ranks). Every cortex cell gets row 0, hence layer c1. `os_hetero` 1 and 2 give the whole cortex c1's potential instead of their c1–c8 profiles;
  - **(q) os_hetero 4** reads the endodermis cells' potential from `osmotic_endo`, the key that also sets the stelar apoplast's potential at the endodermis. The port has its own `osmotic_endo_cell` (set equal for the comparison).
  - The `r_rel` concern of §20a is unfounded: the legacy's `distance_from_center` is per row, as the original's `Layer_dist2`.
- **Component tests** (`test/component/test_osmotic_component.py`, 5):
  - no solutes, no change;
  - σ on the membranes only;
  - every node's balance with the drives in the plant problem;
  - uniform π cancels;
  - cells more negative than the soil solution increase the uptake.
- **Seedling SPAC example:** the presets run before the hydraulics at each step (the barriers move), and the plots and the upscaling show the water potential. It converges in 6 steps; transpiration = uptake = 0.1483 mm³ s⁻¹ (0.1481 without). Water potentials along the path (MPa):

| soil (matric) | root surface | root xylem | stem xylem | leaf xylem | leaf cells | pores |
|---|---|---|---|---|---|---|
| −2.11 | −2.14 | −3.45 | −3.35 | −3.40 | −3.46 | −4.18 |

  - **The root surface** is below the soil's matric potential by the soil solution's π (−0.05 MPa). The soil model carries no osmotic potential.
  - **The root xylem is below the stem's:** the presets give the roots' xylem sap π = −0.18 MPa and the shoot none. The pressure is continuous through the junctions, but the water potential steps at the collar. It disappears when a solute model gives the sap its π along the whole plant (the `reaction` branch), or with presets for the shoot.
- **Tests:** 172 passed, 1 skipped.
- **Not in this step:** the elongation sink (barrier 0), a phloem boundary at barrier > 0, the profiles' convection velocity.

### 21a. Legacy problems p and q fixed (2026-10-16, your request)

- **`presets_rules`** (MechaOsmoticPresets): 1 the original MECHA's (default), 2 the legacy's (set by the osmotic comparison).
- **p, the cortex layers:**
  - **The original's rule** (`src_old`, its `Os_c1`…`Os_c8`): c1 on the outer cortex row (`row_outercortex`), c2 on the next row inward, …, c8 on the 8th row and deeper, and c8 when the cortex has a single row.
  - **The legacy departs from it twice.** Its lookup tests the rank against the row values, which puts every cortex cell in row 0. Its formula `8 − (row_outer − row)` also runs outward (c8 on the outer row).
  - **The port** now takes each cell's row from its rank, with the original's order.
- **q:** with os_hetero 4, the endodermis cells take their own `osmotic_endo_cell`. The legacy's rules read `osmotic_endo`, the stelar apoplast's value at the endodermis.
- **Reviewed difference** (standard problem with the osmotic terms, default scenario otherwise, the standard uptake in cm³ d⁻¹):

| root, barrier | os_hetero 1: fixed / legacy | os_hetero 2: fixed / legacy |
|---|---|---|
| monocot, 1 | 4.0107e-3 / 3.6444e-3 (+10.1 %) | 4.0183e-3 / 3.8697e-3 (+3.8 %) |
| monocot, 3 | 1.9793e-3 / 1.4919e-3 (+32.7 %) | 1.9328e-3 / 1.7348e-3 (+11.4 %) |
| dicot, 1 | 7.0791e-3 / 6.4316e-3 (+10.1 %) | 7.0890e-3 / 6.8263e-3 (+3.8 %) |
| dicot, 3 | 4.0151e-3 / 3.0268e-3 (+32.6 %) | 3.9147e-3 / 3.5132e-3 (+11.4 %) |

  The fixed profiles take the inner cortex down to −9300 hPa (os_hetero 1). The legacy kept the whole cortex at c1's −6400. The stronger osmotic gradient raises the uptake, most where the endodermis is suberized (barrier 3), since the symplastic path then carries the flow.
- **Tests** (`test_legacy_fixes.py`): every cortex cell's π against its layer (os_hetero 1, the original's order; the legacy's single value with rules 2); the endodermis cells' own value with os_hetero 4.

### 21b. Soil potential gradients around a section: not ported (2026-10-16, your decision)

The legacy reads `psi_soil_left`/`psi_soil_right` (and `osmotic_left`/`osmotic_right`) only in its scenario loop (`water_flux`). No scenario in the branch being ported sets the two sides apart: the default scenario, `extdata/BCs.xml`, the tests and the tutorials all give equal values. So the gradient is not ported until a real scenario needs it. The osmotic presets keep the legacy's left–right weighting of the soil solution (`osmotic_symmetry_soil` 1), which is uniform with equal sides. If one is ported later, the consequences listed in the recap of 2026-10-16 apply:
- the standard problem only;
- kr needs a soil reference;
- net versus gross uptake;
- x_rel from the cells' centroids rather than GRANAP's seeds;
- the anatomy's own x axis.

## 22. Osmotic visual example, and the next steps (2026-10-16)

**Visual example:** `example/hydraulics/osmotic_root.py`. One 200 µm root section at barriers 1 and 3, in three cases: no solutes, MECHA's presets with os_hetero 0, and with os_hetero 1. Its figures:
- the maps of π, the pressure ψ and the water potential ψ + π;
- their radial profiles, the cells drawn without the air spaces (which take the soil solution's π with σ 0);
- the pathways' shares of the radial flow.

The printed uptake and kr per case:

| barrier | case | uptake (cm³ d⁻¹) | kr (cm hPa⁻¹ d⁻¹) |
|---|---|---|---|
| 1 | no solutes | 2.7689e-3 | 1.1000e-4 |
| 1 | os_hetero 0 | 3.4858e-3 | 1.3848e-4 |
| 1 | os_hetero 1 | 4.0107e-3 | 1.5933e-4 |
| 3 | no solutes | 1.0246e-3 | 4.0703e-5 |
| 3 | os_hetero 0 | 1.2869e-3 | 5.1125e-5 |
| 3 | os_hetero 1 | 1.9793e-3 | 7.8630e-5 |

With solutes, the cortex cells are turgid (+4000 to +7000 hPa) and the walls stay near the soil's pressure. `plots.radial_profile` now skips NaN nodes.

**Rule for what is ported next (your decision of §21b):** only what a scenario of the branch being ported uses (its default scenario, tutorials, tests, inputs).

**Next steps, in the order I propose:**
1. **The elongation sink** (barrier 0, `Mecha.elongation_BC`). The legacy's default scenario has an elongation rate (2.8 cm d⁻¹, no side difference), and `water_flux` adds it at every barrier-0 stage, so the branch uses it. It is a node source term in the same balance as the osmotic terms. A design note first: its geometry (the cells' volumes and x_rel) and how a growth model would give it later.
2. **The layer-wise outputs** (`standard_transmembrane_fractions`, `create_layer_discretization`): the uptake and transmembrane fractions per cell layer, as SubOrgan or layer variables. `water_flux` computes them in every scenario. They are the optional downstream of Q-M5, deferred since step 4.
3. **The exports:** the paraview files (tutorial 01's last cell, `test/legacy/test_visu_types.py`). Q-M6: as outputs of the DataStructure, through GRANAP's polygons.
4. **Q-M2's second half:** GRANAP's own lateral distances instead of MECHA's recomputed ones, and the derived steps that do not influence the results removed. Each is a reviewed difference, as for Q-L4.
5. **The `reaction` branch** (solutes, leaf transpiration), when you make it available. Its π, and perhaps σ, plug into MechaHydraulics' "Solutes" inputs, in place of the presets.

**Not ported (no scenario of the branch uses them):**
- the soil's left–right potentials (§21b);
- a phloem boundary at barrier > 0 (`pressure_sieve` is NaN in the default scenario; `extdata/BCs.xml`, which sets it, is loaded by no test or example);
- the profiles' convection velocity.

**Later, with your answers:**
- per-tissue cell heights (Q-L2);
- the shoot's specialisation: `wall_air` and `air_link` edges (problem i), cuticular evaporation (Q-E1), shoot presets for the sap's π (§21).

## 23. The `reactions` branch: snapshot and first findings (2026-10-16)

**Your direction:**
- port the solute transport from your colleague's `reactions` branch, with its simulations in scope (for the comparison tests and the examples);
- use its last commit;
- your colleague works on the port from now on.

**Snapshot (91c49e7):** `origin/reactions` at 4c9c8c2 (2026-10-15), frozen as the second reference beside the legacy:
- `src/openalea/mecha/legacy_reactions/`: its MECHA package, calibration included;
- `example/legacy_reactions/simulations/`: its seven simulation scripts.

Only the imports are rewritten (`openalea.mecha` → `openalea.mecha.legacy_reactions`), and `needle.py`'s insertion of sibling checkouts (`../../MECHA-dev/src`, `../../GRANAP-dev/src`) into `sys.path` is removed. The branch forked at f24c889, before the legacy's freeze point (2b995b2 = f24c889 + PR #55, "retro-compatibility"). Its tests and tutorials are unchanged since the fork, so the solute work is exercised by its simulations only.

**Its water side differs from the legacy's** (same GRANAP sections, standard problem). Two changes found so far:
- **`k_xyl` defaults to 1.0** (1e-6 in the legacy): the vessels are held at the xylem's potential. kr +3.6 % (monocot, barrier 1: 1.0618e-4 against 1.0252e-4; the port's Poiseuille boundary of Q-L3 gives 1.0619e-4), +1.4 % at barrier 3; dicot +3.0 % and +1.3 %.
- **The 1e-16 membrane floor** (a membrane of a cell without aquaporin activity outside the barriers) now applies only when `kmb` is 0 too. Otherwise such a membrane conducts through `kmb`. On the monocot root, the intercellular spaces' membranes at barrier > 0 (cortex, k_interc 0) are 5e6 times more conductive.
- Kx and every other edge are unchanged.

Both are new water behaviour relative to the frozen legacy. Their place in the port is a question of the design note (§23 next), with the solute transport's.

### 23a. What the `reactions` work is (from reading the snapshot and its simulations)

**The package:**
- **`SoluteTransport`:** one mobile solute's concentration per node (mol cm⁻³), the transport `C dc/dt = (D + A − R) c + s`.
  - D: diffusion along the walls (`D_wall`, scaled by the walls' kw ratios, as the barriers do for water) and through the plasmodesmata (`D_pd` × their cross-section and count), and passive diffusion across the membranes (`D_mem`).
  - A: advection by the water flows, the membranes' carried fraction 1 − σ; upwind, or Scharfetter–Gummel (`'sg'`, used by the simulations).
  - R: first-order degradation.
  - θ-method in time; Dirichlet nodes by row override.
- **`MultiSoluteTransport`:** several species coupled by a linear reaction matrix K (d⁻¹), such as sucrose ⇄ starch with an immobile starch, solved as one block system.
- **The coupling with water:** each cell's osmotic potential becomes `baseline − R T c` (van't Hoff, `set_osmotic_from_concentration`), and `water_flux(use_stored_psi_os=True)` re-solves the water. The steady coupled solver (Picard or JFNK) predates the reactions work (byte-identical in the legacy).
- **The water side of the needle,** new (it is the leaf transpiration work):
  - `wall_air` edges, a linearised Kelvin exchange between a wall and an air space, in series with the membrane;
  - `air_link` edges between air spaces;
  - an evaporating-wall Dirichlet penalty (`set_air_wall_bc`);
  - `kw_meso_hypo` (the needle's mesophyll–hypodermis interface, cut);
  - `InData.needle_defaults()`.

**The simulations** (`example/legacy_reactions/simulations/`):
- **The anatomy:** GRANAP's `NeedleAnatomy(seed=42)`: 16 195 nodes, 39 189 edges. GRANAP's main and legacy needle classes give identical networks.
- **The dynamic run** (`needle.py`, its `flux_cdiff` and `flux_pdens` sweeps): a sucrose pulse in the mesophyll drains to sinks (phloem-loading cells and xylem, c = 0), with the osmotic feedback, one water re-solve per step (lagged), SG transport and θ = 1.
- **The diurnal runs:**
  - transpiration (Ψ_air × f(t)) and photosynthesis (a concentration increment c_photo g(t) per step in the mesophyll) forced each hour;
  - starch as an immobile species with synthesis and degradation (`diurnal_starch`);
  - day/night switching rates over 30 days (`diurnal_starch_diel`, 720 steps).
- **Not used by any scenario:** the steady coupled solver (Picard, JFNK, continuation), the calibration package (except `rh_to_water_potential`, for an RH mode no script sets), the `'apo'` and `'sym'` modes, the upwind solve, θ ≠ 1, single-species degradation.
- **Cost:** about 2.8 s per step in the legacy (the water solve: scenarios 0 and 1 each step). A 2–5 step dynamic run (10–17 s plus the anatomy) is a usable comparison test.

**Problems found in it** (to record and fix one by one, as for the legacy):
- **r1 — the osmotic volume flux counted twice in the advection:** the flows Q of `compute_edge_flows` already include K σ Δπ (`mecha_class.py:1474`), and the transport operator adds `K σ (π_wall − π_cell)` again (`solute_transport.py:578–588`, `_build_osmotic_advection`). Verified. Also present in the legacy.
- **r2 — σ looked up on the wrong end of a membrane** in the SG operator (`cell_node = v …` after the ends were reordered, `solute_transport.py:572`): the wall's end whenever the edge is stored cell → wall. Verified.
- **r3 — two independent σ:** the water's (scenario presets, by rank) and the solutes' (by cgroup, 0.7 for cgroups 1–19). Needle cells without a rank in the water's rules get σ = 0 there and 0.7 for the solute.
- **r4 — the reaction runs in the walls too** (`full` mode: sucrose → starch in the apoplast), and uses the raw volumes while the storage uses C × V.
- **r5 — the photosynthetic source is a per-step increment**, so its rate scales with 1/dt.
- **r6 — `needle.py`'s transpiration** sums the `wall_air` flows, probably ~0 by conservation (the evaporating walls are pinned by the penalty, so the outflow is the penalty's residual, as the calibration reads it).
- **r7 — the osmotic baseline** is the root presets' (os_cortex −4800 hPa): the comment says the magnitudes are zeroed; the code only sets `s_factor`.
- **r8 — the protophloem list** mixes cell ids and node ids (a line the legacy removed in PR #55), so the sieve penalties at barrier 0 go on wrong rows.
- **Others:**
  - `k_sieve[0]` on a scalar default (TypeError if a phloem potential is set);
  - `HydraulicCell.RH` uninitialised;
  - the name `k_air` used for two things;
  - `psi_total` defined two ways;
  - JFNK + time stepping inconsistent (unused).

### 23b. Proposed design

- **Scope of the first solute step** (what the scenarios use): the dynamic sucrose transport with its osmotic feedback (`DynamicParams`), then starch (two species, reaction matrix K), then the diurnal forcings as the drivers of examples. Not ported (no scenario): the steady coupled solver, the calibration, the `apo`/`sym` modes, the upwind solve.
- **A component `MechaSolutes`** (on `MechaSections`):
  - a node state variable per species (`sucrose`, `starch`; mol cm⁻³ on the Compartments);
  - its transport as a metafspm `@graph_system`: node unknowns the concentrations, storage C V / dt (implicit Euler, θ = 1, as the simulations), the edge fluxes in `@node_balance` (SG on the water flows of `MechaHydraulics`, diffusion per pathway), Dirichlet sinks as `boundary_set`s, the reaction as node terms, an analytic sparse Jacobian (the system is linear in c);
  - outputs: the solutes' fluxes, and `osmotic_potential` = baseline − R T c on the cells, i.e. it is the "Solutes" model of `MechaHydraulics` (in place of `MechaOsmoticPresets`, which gives the baseline and σ).
- **The coupling as the simulations do it:** one water solve, then one transport step, per time step. In metafspm's schedule: the hydraulics' state, then the solutes' state.
- **Exactness first:** problems r1–r8 included, as for the legacy (Q-M1), each fixed afterwards with its reviewed difference.
- **Comparison tests** on `NeedleAnatomy(seed=42)`, a few steps of the dynamic run, then of the diurnal run with starch (phases shifted so that photosynthesis acts from the first step): concentrations, interior mass, mass to the sinks, water flux.

### 23c. Questions

- **Q-S1 — the needle's water side.** Every simulation needs it: `wall_air` (Kelvin), `air_link`, the evaporating-wall penalty, `kw_meso_hypo`, `needle_defaults`. The port has none of it: the `wall_air` and `air_link` edges are ignored (problem i), and the leaves evaporate at their stomatal pores (E3). Port it now, as the shoot's transpiration model beside the pores (leaves by pores, needles by their air spaces), before the solutes? *Recommendation:* yes, first. It is the water state the solutes are compared on.
answer: ideally, I want to generalize what is done on needles to all leaves so we work on the same basis, but before answering: do we have also air to air nodes and edges that can describe how water leaves the stomatal chamber?
- **Q-S2 — the reactions' other water changes:**
  - the 1e-16 membrane floor only when kmb = 0 (§23);
  - `k_xyl` and `k_sieve` 1.0 (the port's Q-L3 Poiseuille boundary already holds the vessels);
  - the tissue name `vascular parenchyma` (was `vascular_parenchyma`);
  - `hypodermis` mapped to cgroup 2.

  *Recommendation:* the membrane floor as a reviewed fix (an option, the legacy's kept for its comparison); check the tissue names against GRANAP's current ones; `k_xyl` nothing to do.
  answer: yes to recommendation, if the modeler of the reaction branch made unrealistic fixes to make solute work I don't want it to pollute the port but if it just expands a scenario of MECHA's use, I am ok.
- **Q-S3 — GRANAP's needle in metafspm.** The port reads anatomies from the DataStructure, so the needle must come through GRANAP's anatomy component (a needle class or preset in its library). The simulations build it with `NeedleAnatomy(seed=42)` directly. Checked: GRANAP's metafspm component (`granap/metafspm/`) has no needle yet. *Recommendation:* add the needle to GRANAP's library first (a GRANAP step: `NeedleAnatomy` as the organ class of a needle segment).
answer: wrong, look at at the examples again, we generated needles so GRANAP port must already do it no?
- **Q-S4 — the transport as a graph system.** Implicit Euler inside metafspm's graph system (one solve per time step, the step being the simulation's), rather than porting the θ-method as is. θ = 1 is the only value the scenarios use. *Recommendation:* yes, θ = 1 only.
answer: yes
- **Q-S5 — the photosynthetic source:** ported as the simulations do it (a concentration increment before the step) for exactness, then fixed (r5) as a rate source (mol d⁻¹), which a photosynthesis model would give? *Recommendation:* yes.
answer: yes, the intent is rather a photosynthetic source flow.

## 24. Your answers to Q-S1–Q-S5 (2026-10-16), and what I found for them

- **Q-S4:** yes, θ = 1 only (implicit Euler in the graph system).
- **Q-S5:** yes. The photosynthetic input is ported exactly first, then as a source flow (mol d⁻¹), as a photosynthesis model would give it.
- **Q-S3: you were right.**
  - GRANAP's port generates needles: its main `NeedleAnatomy` builds into the MPG (`example/needle/`, `test/test_root_needle_network.py`).
  - The anatomy component's library takes any organ class, as the seedling does for leaves (`AnatomyLibrary(parameters=preset_parameters("for_needle"), organ_class=NeedleAnatomy)`).
  - Checked: `NeedleAnatomy(OrganInputData.for_needle(), seed=42)` gives the simulations' network exactly: 16 195 nodes, 39 189 edges, the same positions.
  - No GRANAP step is needed. My "no needle" came from searching only `granap/metafspm/`.
- **Q-S2, yes to the recommendation, with your condition** (no unrealistic fixes made only to make the solutes work; expansions of MECHA's scenarios are fine):
  - **The membrane floor: not adopted.**
    - The original MECHA sets K = 1e-16 when a cell's aquaporin activity is 0, "kaqp==0 in case of air-filled intercellular space" (`src_old/6_MECHAv4_SRS2025.py:2816`, `:2838`): an air-filled space has no water-conducting membrane.
    - The reactions' rule (floor only when kmb = 0 too) makes those spaces conduct through kmb: 5e6 times more on the root's intercellular spaces, and 26 membranes of the needle's air spaces.
    - It is a workaround against MECHA's intent; the port keeps the original rule.
  - **`k_xyl` and `k_sieve` 1.0:** nothing to port. The port's Poiseuille boundary (Q-L3) already holds the vessels, and is MECHA's original source.
  - **The tissue names** (`hypodermis` → 2, `vascular parenchyma`): inert with GRANAP. Its cells carry their cgroup (its own table maps hypodermis to 1), and it uses neither vascular-parenchyma name. The type fallback they change only applies to cells without a cgroup.
- **Q-S1, your question first: do air-to-air nodes and edges describe how water leaves the stomatal chamber? No.** In GRANAP's anatomies (`NeedleAnatomy(seed=42)`, the monocot leaf preset):
  - **The needle:** 556 air spaces (cells, type `air space`), joined to the walls by 1104 `wall_air` edges and to each other by 872 `air_link` edges (7 components, the largest 525 air spaces). No air space is linked to a stomatal pore or to the outside air. The 12 pore and guard cells are protoplasts: membranes to their walls, plasmodesmata to guard, epidermal, hypodermal and mesophyll cells.
  - **The monocot leaf:** 33 air spaces with 29 `wall_air` edges and no `air_link` at all. Its 20 pores are protoplasts as well.
  - **In the `reactions` model,** water leaves through every wall facing an air space, each pinned to the air's potential by a penalty (`set_air_wall_bc`). The air spaces' nodes (vapour pressures) have no outlet: the vapour network carries no net flow out (problem r6), and the pores play no part.
  - **In the port,** the leaves evaporate at their pores (E3: g_s × the leaf surface they serve, from the pore protoplast's potential), without the air spaces.
  - **Neither describes the path** walls → sub-stomatal air spaces → pore → atmosphere. That path would need:
    - in GRANAP: the pore as an air node (not a protoplast), `air_link` edges from the sub-stomatal air spaces to it, and air links in every leaf preset;
    - in MECHA: vapour conductances along the air links, and the stomatal conductance between the pore and the atmosphere as the only boundary to the air. The evaporating sites are then every wall facing an air space (Kelvin), the same basis for needles and leaves.

  That is the generalisation you describe. Its design, and whether the comparison with the reactions' needle runs (with its pinned walls) comes first, is question Q-S1b.

## 25. Transpiration by the air spaces' walls (your direction, 2026-10-16), and the plan for it

**Your direction:** apply the `reactions` branch's logic to all leaves: pin the potential of all walls of the air spaces. The air transfers out of the leaf come later.

**What the port takes from the `reactions` water side,** exactly first (compared with its needle on `NeedleAnatomy(seed=42)`), each choice that differs from your direction kept as an option for that comparison only:
- **The evaporating walls are pinned** to the air spaces' potential, by the `reactions`' penalty (a robin set of weight `max(1e3 × max |diag W|, 1)`, its value per section).
  - Which walls: all walls bounding an air space (your direction, default); `reactions` pins only the wall ends of its `wall_air` edges (option).
- **`wall_air` edges** (wall ↔ air space, GRANAP's `network_path` 1): the linearised Kelvin exchange in series with the flanking membrane, an asymmetric flux `K_wall ψ_w − K_air x_a + offset`. The air space's unknown is then its vapour pressure (relative to p_sat).
  - Parameters `kwa`, `p_sat`, `psi_ref_wall_air`, `T` and the water's molar mass, at the `reactions`' values.
- **`air_link` edges** (air ↔ air, `network_path` 4): `k_air_link × height × length`, a symmetric vapour diffusion.
- **`kw_meso_hypo`** (NaN: none): the needle's (2, 4) walls (and their plasmodesmata) cut. This is a needle parametrisation, not a fix, so it stays a parameter.
- **The air spaces' potential:** a parameter `air_space_psi`, the `reactions`' default −1e4 hPa (its `psi_atm`, or a relative humidity through Kelvin).
  - It is not the outside air's potential: at 50 % RH that is −935 bar, absurd for a liquid-filled wall.
  - When the air transfers out of the leaf come, the atmosphere will set it through the stomata.
- **The leaves' transpiration model:** `transpiration_model` 2, the air spaces' walls (default). 1 keeps the stomatal pores (E3, the seedling's so far), as an option.
- **For the comparison only:**
  - the `reactions`' membrane floor (`membrane_floor_rules` 2: its 26 air-space membranes conduct);
  - the osmotic presets on all sections (`presets_sections` 2; the needle's cells have the root ranks the presets read);
  - the standard problem on shoot sections (`standard_solve_shoot` 1: the needle alone, its xylem and its pinned walls).

**Steps:**
1. **The needle in the port:** GRANAP's library with `NeedleAnatomy` and `for_needle`. Its sections: tissues, protect_topology, the `wall_air` and `air_link` edges, ranks.
2. **The water model above,** compared with `legacy_reactions`' scenarios 0 (the standard problem: xylem at 0, walls pinned at −1e4 hPa) and 1 (ψ_xyl −200 hPa, its osmotic presets with s_factor 0.7) on the needle, to D1's tolerances.
3. **All leaves:** the defaults above in the plant problem. The seedling SPAC example then transpires through its leaves' air spaces' walls; its new numbers are a reviewed difference.
4. **Then the solutes** (§23–24).

### 25a. Steps 1–2 done; step 3 stopped on a question (2026-10-16)

**The needle in the port:** GRANAP's library with `NeedleAnatomy` and `for_needle`, seed 42 (`test/plants.py`: `needle`, `needle_anatomy_component`): the simulations' network exactly.

**The water model** (`conductance.py`, `component.py`):
- **The `wall_air` exchange:** `wall_air_terms`, the reactions' `_fill_wall_air`, with the flanking membrane's own K. In the balances and flows it is an asymmetric flux `a ψ_tail − b ψ_head + c` (`air_tail_conductance`, `air_head_conductance`, `air_offset`), with its Jacobian.
- **The air links:** `k_air_link` × height × length, in `conductance`.
- **The pinned walls:** `air_wall_conductance` (the penalty), held at `air_wall_psi` / `standard_air_wall_psi` by robin sets in both problems.
- **Which walls:** by default every wall of a leaf section's air spaces, at the water potential (`evaporating_walls` 1). Stems keep theirs free: you said leaves.
- **`kw_meso_hypo`:** the (2, 4) walls, their membranes and plasmodesmata.
- **Options for the comparison:** `membrane_floor_rules`, `standard_solve_shoot`, `presets_sections`.
- **The pores (E3)** as `transpiration_model` 1.
- **The air spaces' water potential** (`water_potential`, `standard_water_potential`) is Kelvin's liquid equivalent of their vapour pressure, ln(x / p_sat) / α. Their ψ holds that vapour pressure (`vapour_node`).
- **The Newton tolerance** is 1e-8 cm³ d⁻¹, from 1e-10: the penalty's round-off is ~1e-9. The systems are linear, solved by the first step.

**Against the reactions' needle** (`test/comparison/test_needle_water.py`, 5 tests):
- the same 1104 pinned walls, the penalty bit-exact (1000.5091151343986);
- every edge's K and every wall_air coefficient bit-exact;
- scenario 0's solution to 1.8e-12 of its range (tested at 1e-11: the penalty's conditioning);
- scenario 1's presets: π to 1e-13, σ exact.

**Problems found in the reactions' needle:**
- **r9:** `kw_meso_hypo` ("hypodermis–mesophyll") applies to the (2, 4) interface. GRANAP's hypodermis is cgroup 1, so it cuts the epidermis–cgroup 4 walls (mesophyll, transfusion, air spaces), not the hypodermis' (1, 4).
- **r10:** the legacy sets a wall's osmotic potential per membrane. On 165 needle walls, between a cortical-rule and a stelar-rule cell (transfusion tissue, cgroup 4 inside the endodermis, against Strasburger and parenchyma cells; the guard cells' walls), its equations carry two apoplastic potentials, one per side. The port keeps one per wall (the last set), the physically consistent choice.
  - With the legacy's per-membrane values as drives, the legacy's scenario-1 solution satisfies the port's equations to 7e-16 of the penalty's scale (the test).
  - With one value per wall it differs by up to 374 hPa (3.7 %), at those walls.
- **Tests:** 180 passed, 1 skipped.

**Step 3, all leaves, is stopped on a question.**
- **The run diverged:** with the seedling's leaves transpiring through their air spaces' walls (pinned at RH 97 %, −4.12 MPa), the SPAC scene went from soil potentials of −2 MPa to −200 MPa, its change per step growing 1.16× each step, the transpiration reversed.
- **One plant alone transpires about as much as with the pores:** at a soil of −0.1 MPa, 0.38 cm³ d⁻¹ against 0.24.
- **The cause:** the pinned walls are a fixed potential with no limiting conductance (the pores had their stomatal one). The scene's plant–soil iteration, without relaxation, amplifies: a drier soil gives a smaller uptake; below −4.12 MPa, the pinned walls feed the soil.
- **For now:** the seedling example keeps its stomatal pores (`transpiration_model` 1, with a note), and gives back its earlier numbers. The question is Q-T1 (recap of 2026-10-16).

## 26. Design note: a transpiration model defined through metafspm (your direction, 2026-10-16)

**Your direction:** the colleague adapts if his approach is unstable. Transpiration is defined through metafspm:
1. MECHA expects a transpiration flow as an input at the Organ scale (constant for now; no stand-in component);
2. the anatomical graph is extended with Connections between the stomatal chambers and the leaf's surface;
3. the cross-scale coupling gives the Neumann boundary condition.

Your questions: how to split the flow between the stomata, whether the Neumann condition belongs outside the leaf (heterogeneous stomatal conductances), and whether two successive edges (stomata → leaf surface → air layer) are the physically right way.

### 26a. The physics, and where the flow is imposed

**The path:**
1. **Evaporating walls → air spaces:** the Kelvin exchange, the `wall_air` edges.
2. **Through the intercellular air to the sub-stomatal chamber:** vapour diffusion, the `air_link` edges.
3. **Through each pore:** its stomatal conductance g_s,i.
4. **Across the boundary layer to the bulk air:** g_b.

Steps 3 and 4 are in series. The usual leaf-scale form is E = g_tot Δe / P, with 1/g_tot = 1/g_s + 1/g_b.

**Imposing the flow at a node outside the stomata is the right place.** One "leaf surface" node, joined to every pore by an edge of that pore's conductance, carries the Neumann condition (the organ's transpiration leaves through it). Its potential is then an unknown that adjusts so that Σ_i K_i (x_pore,i − x_surface) = E. The split among the stomata comes out of their conductances and of the leaf's internal state (which air spaces are wetter, which mesophyll supplies more), not from a rule we would have to choose. Imposing Neumann values on the stomata themselves would need such a rule (by area? by conductance?) and would hide the heterogeneity.

**A second edge (surface → an outer air layer) adds nothing while the flow is prescribed.**
- If the boundary layer is one shared node per organ, every molecule crosses the same edge: it shifts the surface potential but not the split. It is redundant, and its potential is a free value.
- A per-stoma boundary-layer edge would matter only if g_b varied from stoma to stoma. g_b is a property of the leaf surface's regions (edges against the centre of the blade, upper against lower side), not of individual stomata.
- The boundary layer becomes necessary when the atmosphere's state (its humidity, a Dirichlet or robin condition) replaces the prescribed flow: g_b then limits the total. *Recommendation:* the pore → surface edges now; the surface → bulk air edge when the atmosphere's humidity is the boundary.

**The pinned walls go,** in this model: the walls evaporate into the air spaces by the Kelvin exchange, driven by the vapour deficit the prescribed flow creates. The `reactions`' pinned walls stay an option (`transpiration_model` 2) for its comparisons and the colleague's work.

### 26b. Through metafspm

- **The input:** `leaf_transpiration` (cm³ d⁻¹, extensive, Organ scale, by "Transpiration"), initialised constant. A transpiration or atmosphere model will give it later.
- **Down to the anatomy:** metafspm maps a coarse variable to nodes only by broadcasting (`data_api._map`); it has no extensive split. With one surface node per organ, the broadcast value with a `boundary_set(kind="neumann", filters={"leaf_surface": ">0"})` reaches that node alone, and the organ's flow is applied once. No change to metafspm.
- **One surface node per organ, not per section:** a leaf organ has several sections (leaf elements). Per-section surface nodes would need the organ's flow split among the sections, by a rule again. One node per organ joined to every pore of all its sections lets the network split it along the leaf too. The surface node is a Compartment of the organ's first (basal) element.
- **The graph (a GRANAP step):**
  - **The pores become air nodes.** In GRANAP's needle and leaf they are protoplasts today: membranes to their walls, plasmodesmata to the guard cells, no air link. A pore should be joined by `air_link` edges to the sub-stomatal air spaces, its walls to the guard cells, with no membrane of its own.
  - **Air links in every leaf preset:** the monocot leaf has none today (33 air spaces, `wall_air` only).
  - **The surface node, and one stomatal Connection per pore** (a new `network_path`, 6), created with the axial junctions by GRANAP's wiring rules.
- **In MECHA:**
  - the stomatal edges' conductance, from a per-pore `stomatal_conductance` (a node variable: an input by a stomatal model later, the parameter's value by default);
  - the Neumann set on the surface nodes;
  - the outputs: the pores' flows, the surface node's vapour pressure (the leaf surface's humidity), the evaporation per wall.

### 26c. Units of the vapour side (a reactions problem, r11)

The air nodes' unknown is the vapour pressure relative to saturation (dimensionless, `p_sat` = 1). The reactions' `kwa` and `k_air` are labelled "cm hPa⁻¹ d⁻¹" and multiply that dimensionless difference, so their values (1e-4) are placeholders without a physical basis. With prescribed flows, the vapour side's conductances set the leaf's internal potentials, so they need physical values:
- **air links:** vapour diffusivity in air (D_v ≈ 2.4e-5 m² s⁻¹), × area / distance × e_sat V_w / (R T);
- **stomata:** g_s (mol m⁻² s⁻¹) × the pore's share of the leaf area × e_sat V_w / P;
- **the walls' evaporation:** the Kelvin coefficient with the same factor.

These give cm³ d⁻¹ per unit of relative vapour pressure.

### 26d. Questions

- **Q-T1:** one surface node per leaf organ (recommended), or per section?
answer per leaf organ yes, then the solve will attribute heterogeneity between individual stomata
- **Q-T2:** the stomata → surface edges now, and the boundary layer (surface → bulk air) when the atmosphere's humidity becomes the boundary (recommended), or both now?
answer: I don't understand the question
- **Q-T3:** the GRANAP step first:
  - pores as air nodes linked to the sub-stomatal air spaces;
  - air links in all leaf presets;
  - the surface Compartment and the stomatal Connections by the wiring rules.

  Agreed?
answer: agreed
- **Q-T4:** the vapour side's conductances in physical units (§26c) now, as part of this model, with the reactions' placeholders kept for its comparison (`transpiration_model` 2)?
I don't understand the question.
- **Q-T5:** the default model for leaves: this one (`transpiration_model` 3, flow at the leaf surface)? With the pores (1) and the pinned walls (2) as options. The seedling would then transpire its prescribed flow, which also removes the scene's instability: the uptake is fixed by the flow, not by a fixed potential.
answer: Keep only this third model, I don't want several options for now, to keep the component syntax as simple as possible.

### 26e. Your answers (2026-10-16)

- **Q-T1:** one surface node per leaf organ; the solve attributes the heterogeneity among the stomata.
- **Q-T3:** agreed, the GRANAP step first.
- **Q-T5:** only this model (flow at the leaf surface). No options: the stomatal pores' evaporation (E3) and the pinned walls are removed from the component, to keep its syntax simple.
  - Consequence: the needle's water state can no longer be compared exactly with the reactions' (its pinned walls). The solute comparison then feeds both transports the same water flows (the legacy's), which compares the transport itself.
- **Q-T2 and Q-T4:** not understood, so reformulated in the recap of 2026-10-16 (plain-language versions to answer here).

### 26f. Your answers to the reformulated Q-T2 and Q-T4, and the stomatal conductance from the anatomy (2026-10-16)

- **Q-T2:** yes, no boundary layer for now: only the pore → leaf-surface edges, the organ's transpiration imposed at its surface node.
- **Q-T4:** yes, the vapour conductances in physical units:
  - the walls' evaporation by the Kelvin coefficient;
  - the air links by the vapour's diffusivity in air;
  - the stomata as below.

  The reactions' placeholders (`kwa`, `k_air`) go with its pinned walls (Q-T5).
- **The solutes:** the comparison feeds both transports the legacy's water flows.
- **Your question: can the stomatal conductance come from GRANAP's stomata dimensions?** Yes, as the anatomical maximum, the standard way (Brown & Escombe's diffusion through a pore with its end correction; Franks & Beerling 2009, the "anatomical g_max"):
  - **Per pore:** g_max = D_v a / (l + (π/2) √(a/π)), with a the pore's area, l its depth, D_v the vapour's diffusivity.
  - **From the section:** the aperture's width and the depth through the epidermis (the pore polygon). Not in a 2D section, so parameters: the pore's length along the leaf (a = width × length), and how many stomata the section's pores stand for over the segment's length (a stomatal density, or a count per section).
  - **The actual conductance** is regulated by the guard cells (turgor, light, CO2): g_s = aperture fraction × g_max. The aperture fraction is a per-pore input for a stomatal model later, 1 by default.
  - So GRANAP gives the pores' geometry, MECHA computes each pore edge's conductance, and a stomatal model can open or close them one by one.

**Order:**
1. The GRANAP step (Q-T3): pores as air nodes, air links in every leaf preset, the leaf-surface Compartment and the stomatal Connections. A design note in GRANAP's plan first, from the code map.
2. MECHA: the single transpiration model. The organ's flow; the surface nodes' Neumann set; the walls' Kelvin evaporation, the air links' and the stomata's conductances in physical units. The pores (E3) and the pinned walls removed.
3. The solutes, compared on the legacy's water flows.

## 27. The single transpiration model in MECHA: the implementation (2026-10-16)

**GRANAP's graph** (its plan §60–61, 739ccda):
- pores as air nodes (wall_air walls, no plasmodesmata, an air link to their chamber);
- every air space's membrane wall_air;
- one leaf-surface node (kind 3) per leaf organ, joined to every pore by a stomatal Connection (path 6);
- the pores' `pore_width` and `pore_depth`.

No intercellular spaces were added to the monocot leaf (your answer): its air network is its stomata and their chambers (and its 6 lacunae).

**The model:**
- **Unknowns:** the liquid nodes' pressure potential (hPa); the air nodes' (air spaces, pores, the leaf surface) vapour pressure relative to saturation, x.
- **The organ's transpiration** `leaf_transpiration` (cm³ d⁻¹, an input at the Organ scale, by "Transpiration", 0 by default) leaves at its surface node: a Neumann set.
- **The vapour conductances, physical** (Q-T4). They are cm³ d⁻¹ of liquid water per unit of x, from vapour diffusion: G = D_v A / d × e_sat V_w / (R T), with D_v the vapour's diffusivity in air, e_sat at the leaf's temperature, V_w the water's molar volume.
  - **Wall → air space** (`wall_air`): the wall's evaporating face, A = the wall's length × the section's height, over d = its distance to the air space's centre. The wall side by Kelvin, x_wall = exp(α ψ_wall), α = V_w / (R T), linearised at ψ_ref (as the reactions'):
    - flow = G e_ref α ψ_wall − G x_air + G e_ref (1 − α ψ_ref);
    - **no membrane in series:** the cell's membrane to its wall is already an edge of the graph, so the reactions' series membrane counted it twice.
  - **Air space ↔ air space, and pore ↔ chamber** (`air_link`): A = the section's height × the narrower of the two (the pore's aperture width for a pore link; for two air spaces, the smaller one's equivalent diameter), over d = the distance between their centres.
  - **Pore → leaf surface** (path 6): each pore stands for the stomata of its file over the segment's length, n = length / `stomatal_spacing`. Each is a diffusion tube with its end correction (Brown & Escombe; the anatomical g_max): G = D_v a / (l + (π/2) √(a/π)), a = `pore_width` × `pore_length`, l = `pore_depth`. Times the aperture fraction `stomatal_aperture` (a node input by "Stomata", 1 by default: a stomatal model later).
- **Outputs:**
  - `evaporation` (cm³ d⁻¹): each pore's flow to the surface;
  - `water_potential`: the air nodes' by Kelvin, ψ = ln x / α;
  - the leaf surface's relative humidity: its x.
- **Removed (Q-T5):**
  - the stomatal pores' evaporation (E3: `stomatal_conductance`, `vapour_factor`, `evaporation_conductance`, `evaporation_psi`, `air_psi`);
  - the pinned walls (`transpiration_model`, `air_space_psi`, `evaporating_walls`, `air_wall_conductance`, `air_wall_psi`, their robin sets, the penalty);
  - the reactions' placeholders (`kwa`, `p_sat`, `k_air_link`, the series membrane);
  - `standard_solve_shoot` and the needle's pinned-wall comparison (its test). `kw_meso_hypo` stays (the needle's parametrisation).
- **New parameters** (per plant): `vapour_diffusivity` (2.4e-5 m² s⁻¹), `leaf_temperature` (20 °C), `pore_length` (20 µm), `stomatal_spacing` (100 µm).
