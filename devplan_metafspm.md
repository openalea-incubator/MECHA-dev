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
