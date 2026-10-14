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
  - → answer:
- **Q-M2 — the geometry MECHA rebuilds.** The legacy rebuilds GRANAP's junctions from positions, and recomputes its own lateral distance (the sum of the wall's membrane distances, not GRANAP's mean plus offset), a single wall thickness, and its own epidermis perimeter. On the DataStructure, use GRANAP's network as it is? Then the junctions, `lateral_distance` and the per-wall `wall_thickness` are GRANAP's, and the conductances differ from the legacy's. Or compute MECHA's own quantities from GRANAP's columns?
  - **Recommendation:** MECHA's quantities first, computed from GRANAP's columns in bulk, so the comparison holds. Then switch, as a reviewed change, to GRANAP's per-wall thickness and lateral distance, which are the better values.
  - → answer:
- **Q-M3 — the parameters' form.**
  - Declare MECHA's scalars as metafspm parameters (`kw`, barrier values, `kmb`, `kaqp`, `kpl`, `pd_section`, `k_interc`, …, numeric, per plant)?
  - And its tables (per-tissue aquaporin factors, per-interface `fplxheight` and plasmodesmata factors) as object parameters? Or flattened into one numeric parameter each (`kaqp_endo_factor`, …)?
  - **Recommendation:** flattened numeric parameters. MECHA's tables are small (5 tissue factors, about 20 interfaces), and flattened parameters can then differ per plant and appear in the documentation like any parameter.
  - → answer:
- **Q-M4 — maturity and barriers.** In MECHA a "maturity stage" is a (barrier code, height) pair, looped over in one run. In a growing plant, each segment has its own barrier state. Make the barrier code a SubOrgan input (given by a development model, or computed from the distance to the tip), with the height the segment's length?
  - **Recommendation:** yes, a SubOrgan input `barrier` (by `Development`). The legacy's loop over stages becomes one section per stage in the comparison tests.
  - → answer:
- **Q-M5 — the standard kr and Kx.** MECHA's kr is defined by a standard solve: the xylem held at a pressure (Dirichlet) and the soil at another, each section alone. In a connected plant, the solved flows give an effective kr per section instead (uptake / (ψ_soil − ψ_xylem, local) / surface). Keep both: the standard one (each section solved alone, a restricted graph system) as MECHA's definition, and the effective one from the plant's solve?
  - **Recommendation:** both. The standard kr and Kx are the parameters a root architecture model (e.g. a Doussan-type root system model) takes; the effective ones are diagnostics of the coupled solve.
  - → answer:
- **Q-M6 — what stays out of the first port.** MECHA's scenario loops (`h` over parameter combinations, BC scenarios), the osmotic and elongation terms, solute transport (`SoluteTransport`, `coupled_solver`), and the exports (paraview, graphml, plots). In metafspm, scenarios belong to the scene and to separate runs, not to the component. Port the hydraulics first (one parameter set, one boundary condition set per run), and leave the rest to later phases?
  - **Recommendation:** yes. Osmotic terms come next (they belong to the same balance), then solutes as their own component; the exports become plots of the DataStructure.
  - → answer:
- **Q-M7 — MECHA's own cellset XML input** (GRANAR files, without GRANAP). GRANAP's `CellSetOrgan` already reads GRANAR cellsets into the MPG. Drop MECHA's XML network builder from the port (the legacy keeps it), and read cellsets through GRANAP?
  - **Recommendation:** yes. One path into the DataStructure, and GRANAP's reader is the one that keeps the walls' topology.
  - → answer:
