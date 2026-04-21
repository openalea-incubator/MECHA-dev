# Psi and Flow in MECHA

## 0. New Classes in `HydraulicCellManager`

The current `HydraulicCellManager` (in `hydraulic_cell.py`) contains:
- `HydraulicWall` — describes cell walls (apoplastic path, `kw`)
- `HydraulicCell` — describes cells, holds `kpl`, `km`, `kaqp` as scalars

The **solver** (`hydraulic_solver.py`) already handles three edge types: `'wall'`, `'membrane'`, `'plasmodesmata'`. The issue is that the hydraulic properties of **plasmodesmata connections** and **membrane connections** are currently stored inside `HydraulicCell`, not as standalone objects. This makes it hard to query, visualize, or iterate specific connection types.

### TO DO:

- [x] **`HydraulicPlasmodesmata`** — new class describing one plasmodesmata connection (graph edge `path='plasmodesmata'`).
  - **Fields**: `cell_i`, `cell_j` (HydraulicCell refs), `kpl` (conductance, cm³ hPa⁻¹ d⁻¹), `length`, `temp_factor`
  - **Source data**: `eattr['length']` from graph, `kpl` / `temp_factor` populated post-solve
  - **Storage**: `HydraulicCellManager._plasmodesmata: List[HydraulicPlasmodesmata]` + dict lookup `_pd_by_edge: Dict[Tuple[int,int], HydraulicPlasmodesmata]`
  - **Method**: `HydraulicCellManager.sync_plasmodesmata_from_network(network)` ✅

- [x] **`HydraulicMembrane`** — new class describing one membrane connection (graph edge `path='membrane'`).
  - **Fields**: `wall` (HydraulicWall ref), `cell` (HydraulicCell ref), `km`, `kaqp`, `length`, `dist`, `K_computed`
  - **Source data**: `eattr['length']`, `eattr['dist']` from graph; `km`/`kaqp`/`K_computed` populated post-solve
  - **Storage**: `HydraulicCellManager._membranes: List[HydraulicMembrane]` + `HydraulicWall.membranes: List[HydraulicMembrane]`
  - **Method**: `HydraulicCellManager.sync_membranes_from_network(network)` ✅

- [x] **Update `sync_from_network`** — calls `sync_membranes_from_network` and `sync_plasmodesmata_from_network` at end of Pass 3
- [x] **Update `HydraulicCellManager.__repr__`** — now reports cells, walls, membranes, and plasmodesmata counts
- [x] **Update `hydraulic_cell.py` module docstring** — includes all four classes

---

## 1. Psi (Water Potential)

TO DO:
- [ ] **Osmotic Potential (S)**: Information about the osmotic potential stored in the `HydraulicCell` objects.
- [ ] **Matric Potential (M)**: Information about the matric potential stored in the `HydraulicCell` objects.
- [ ] **Pressure Potential (P)**: Information about the pressure potential stored in the `HydraulicCell` objects.
- [ ] **Total Water Potential (W)**: Information about the total water potential stored in the `HydraulicCell` objects.
- [ ] **Node-based Visualization**: Visualizing water potential on the Graph nodes after running a MECHA simulation.
- [ ] **Tissue-specific Potential Profiles**: Plotting average Psi across radial distance for different tissues (cortex, stele).

---

## 2. Flow and Conductance

The solver (`hydraulic_solver.py`) fills `matrix_W` via three fill functions. After solving, a `solution` vector is returned (node potentials). **Flow on each edge** is `K * (Psi_i - Psi_j)` — it is currently never stored in the graph or in `HydraulicWall`/`HydraulicCell` objects.

### TO DO:

- [ ] **Store K per edge in graph**: After each `_fill_*` call in `HydraulicMatrixBuilder.build()`, write the conductance `K` back to the graph edge attribute `eattr['K']` and `eattr['Q']` (to be filled post-solve).
  - Wall edges: `K` from `_fill_wall()`
  - Membrane edges: `K` already returned by `_fill_membrane()` → store as `graph.edges[i,j]['K'] = K_mem`
  - Plasmodesmata edges: `K = kpl * temp_factor` from `_fill_plasmodesmata()`

- [ ] **Post-solve flow computation**: After `Mecha.solve()`, compute `Q = K * (Psi_i - Psi_j)` for every edge and store in `graph.edges[i,j]['Q']`.
  - Implement as `Mecha.compute_edge_flows(solution)` method
  - Store per edge type: `'Q_wall'`, `'Q_membrane'`, `'Q_plasmodesmata'` on node dicts for fast node-level aggregation

- [ ] **`HydraulicWall.kw`**: Populate from solved `K` (apoplastic conductance). Currently set to `None` after `sync_from_network`.
- [ ] **`HydraulicPlasmodesmata.kpl`**: Populate `K` after solver runs (links back to `temp_factor`).
- [ ] **`HydraulicMembrane.km` / `kaqp`**: Populate `K_computed` (from `Kmb` in `build()`) after solver build.
  - `Kmb` array is already built per membrane during `HydraulicMatrixBuilder.build()` — expose it to cell manager.

- [ ] **`HydraulicCell` flow aggregation**: Add `Q_in`, `Q_out` attributes to `HydraulicCell`, summed from all connected membrane edges post-solve.

- [ ] **Conductance (K) visualization**: Visualize conductance on graph edges (colored by type: wall / membrane / PD).
- [ ] **Flow (Q) visualization**: Arrow-based edge visualization showing flow direction and magnitude.
- [ ] **Pathway Contribution**: Identifying and visualizing main water pathways (Apoplastic, Symplastic, Transcellular).

---

## 3. Advanced Visualization & Export

TO DO:
- [ ] **Paraview Integration**: Full support for exporting results to `.vtp` or `.vtk` files for 3D visualization.
- [ ] **Interactive Exploration**: Creating interactive plots (e.g., using Plotly or ipywidgets) to browse potentials and flows across maturity stages.
- [ ] **Temporal Dynamics**: If simulations are time-dependent, visual progress bars or animations of potential changes.
- [ ] **Thick Wall Visualization**: Incorporating wall thickness data into the visual model (representing walls as volumes rather than lines).

