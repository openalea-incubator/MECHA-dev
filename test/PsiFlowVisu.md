# Psi and Flow in MECHA

## 0. New Classes in `HydraulicCellManager`

The current `HydraulicCellManager` (in `hydraulic_cell.py`) contains:
- `HydraulicWall` — describes cell walls (apoplastic path, `kw`)
- `HydraulicCell` — describes cells, holds `kpl`, `km`, `kaqp` as scalars

The **solver** (`hydraulic_solver.py`) already handles three edge types: `'wall'`, `'membrane'`, `'plasmodesmata'`. The issue is that the hydraulic properties of **plasmodesmata connections** and **membrane connections** are currently stored inside `HydraulicCell`, not as standalone objects. This makes it hard to query, visualize, or iterate specific connection types.

### New features implemented:

- **`HydraulicPlasmodesmata`** — new class describing one plasmodesmata connection (graph edge `path='plasmodesmata'`).
  - **Fields**: `cell_i`, `cell_j` (HydraulicCell refs), `kpl` (conductance, cm³ hPa⁻¹ d⁻¹), `length`, `temp_factor`
  - **Source data**: `eattr['length']` from graph, `kpl` / `temp_factor` populated post-solve
  - **Storage**: `HydraulicCellManager._plasmodesmata: List[HydraulicPlasmodesmata]` + dict lookup `_pd_by_edge: Dict[Tuple[int,int], HydraulicPlasmodesmata]`
  - **Method**: `HydraulicCellManager.sync_plasmodesmata_from_network(network)` ✅

- **`HydraulicMembrane`** — new class describing one membrane connection (graph edge `path='membrane'`).
  - **Fields**: `wall` (HydraulicWall ref), `cell` (HydraulicCell ref), `km`, `kaqp`, `length`, `dist`, `K_computed`
  - **Source data**: `eattr['length']`, `eattr['dist']` from graph; `km`/`kaqp`/`K_computed` populated post-solve
  - **Storage**: `HydraulicCellManager._membranes: List[HydraulicMembrane]` + `HydraulicWall.membranes: List[HydraulicMembrane]`
  - **Method**: `HydraulicCellManager.sync_membranes_from_network(network)` ✅

- **`sync_from_network`** — calls  `sync_membranes_from_network` and  `sync_plasmodesmata_from_network` at end of Pass 3
- **`HydraulicCellManager.__repr__`** — now reports cells, walls, membranes, and plasmodesmata counts
- **`hydraulic_cell.py` module docstring** — includes all four classes
- **`polygon` in `HydraulicCell`** — Store the Shapely polygon geometry directly in the `HydraulicCell` object so it is readily available for spatial operations and plotting.
- **`_cells_gdf` in `NetworkBuilder`**: `NetworkBuilder.prep_geo()` builds the GeoDataFrame internally during `build_network()` and stores it as `self._cells_gdf` (or copy it from the GRANAP organ if `populate_from_network` is used).

### Next Round of Implementation: Visuals & Geometry

profile plots: should use the 3 mean radial distances for the nodes in each "row" as x axis.
aggregated all nodes on the outer side of the cell, the ones of the middle and the one on the inner side of the cell layer.

ex:
epidermis layer : outer side (all outer nodes of epidermis)
                  middle side (all middle nodes of epidermis)
                  inner side (all inner nodes of epidermis shared with the ones of the outer nodes of the next layer (exodermis or cortex))
                  ...

  - [x] `_plot_conductance_network` (Conductance K on edges): at the moment, `visualize(obj, visu_type="conductance")` does work. It should use **kwargs to select the maturity stage and scenario.
  - [x] `_plot_flow_network` (Water Flow Q on edges): at the moment, `visualize(obj, visu_type="flow")` does work, but it is slow for large networks. It should use **kwargs to select the maturity stage and scenario. This is slow. use the Q values saved in the node dicts instead.
  - [x] `_plot_flow_pathway_breakdown` (Percentage of flow going through each pathway vs. radial distance - Staked Area Plot). at every discrete set of radial distances, calculate the percentage of flow going through each pathway (apoplast, symplast, transcellular). Top priority!
  - [x] `_plot_psi_radial_profile` (Psi vs radial distance). The average water potential at each radial distance. At the moment, the values are not correctly handled. get inspiration from visu_type="water_potential". Separate symplastic and apoplastic potential distributions in the plot.

- [x] **Update `visualize()` dispatcher** — Integrate the newly transferred plots into the main `visualize(obj, visu_type=...)` function in `visu.py` (e.g., adding `'flow'`, `'conductance'`, `'psi_profile'`).

---

## 1. Psi (Water Potential)

TO DO:
- [ ] **Osmotic Potential (S)**: Information about the osmotic potential stored in the `HydraulicCell` objects.
- [ ] **Pressure Potential (P)**: Information about the pressure potential stored in the `HydraulicCell` objects.
- [ ] **Total Water Potential (W)**: Information about the total water potential stored in the `HydraulicCell` objects.
- [ ] **Node-based Visualization**: Visualizing water potential on the Graph nodes after running a MECHA simulation.
- [ ] **Tissue-specific Potential Profiles**: Plotting average Psi across radial distance for different tissues (cortex, stele).

---

## 2. Flow and Conductance

The solver (`hydraulic_solver.py`) fills `matrix_W` via three fill functions. After solving, a `solution` vector is returned (node potentials). **Flow on each edge** is `K * (Psi_i - Psi_j)` — it is currently never stored in the graph or in `HydraulicWall`/`HydraulicCell` objects.

### TO DO:

- [x] **Store K per edge in graph**: After each `_fill_*` call in `HydraulicMatrixBuilder.build()`, write the conductance `K` back to the graph edge attribute `eattr['K']` and `eattr['Q']` (to be filled post-solve).
  - Wall edges: `K` from `_fill_wall()`
  - Membrane edges: `K` already returned by `_fill_membrane()` → store as `graph.edges[i,j]['K'] = K_mem`
  - Plasmodesmata edges: `K = kpl * temp_factor` from `_fill_plasmodesmata()`

- [x] **Post-solve flow computation**: After `Mecha.solve()`, compute `Q = K * (Psi_i - Psi_j)` for every edge and store in `graph.edges[i,j]['Q']`.
  - Implemented as `Mecha.compute_edge_flows(solution)` method
  - Store per edge type: `'Q_wall'`, `'Q_membrane'`, `'Q_plasmodesmata'` on node dicts for fast node-level aggregation

- [x] **`HydraulicWall.kw`**: Populate from solved `K` (apoplastic conductance). Currently set to `None` after `sync_from_network`.
- [x] **`HydraulicPlasmodesmata.kpl`**: Populate `K` after solver runs (links back to `temp_factor`).
- [x] **`HydraulicMembrane.km` / `kaqp`**: Populate `K_computed` (from `Kmb` in `build()`) after solver build.
  - `Kmb` array is already built per membrane during `HydraulicMatrixBuilder.build()` — expose it to cell manager.

- [ ] **`HydraulicCell` flow aggregation**: Add `Q_in`, `Q_out` attributes to `HydraulicCell`, summed from all connected membrane and plasmodesmata edges post-solve.


---

## 3. Advanced Visualization & Export

TO DO:
- [x] **Paraview Integration**: Full support for exporting results to `.vtk` files for 3D visualization.
  -  [x] **Thick Wall Visualization**: Incorporating wall thickness data into the `prep_geo` logic to make thick walls for visualization (representing walls as volumes rather than lines). Extrude the 2D polygons to 3D. 
  -  [x] **Flow and Pressure Visualization**: 
    - [x] Pressure potential in cell is represented by a single scalar value on cell surfaces (Cell.vtk is empty)
    - [x] **Plasmodesmata**: Plasmodesmata are represented by one 3-point polyline routing through the shared wall midpoint (cell_i → wall midpoint → cell_j). All points sit at z = extrude_z/2 (mid-plane of the section). Use ParaView Tube filter.
    - [x] Cell walls are represented by a surface with thickness from thick wall visualization step.
    - [x] **Membrane**: Membrane is represented as a rectangle lying flat **on the wall surface** (endpoint-A → endpoint-B × [z=0 → z=extrude_z]), using real junction endpoints from `junction_positions`.

- [ ] **Interactive Exploration**: Creating interactive plots (e.g., using Plotly or ipywidgets) to browse potentials and flows across maturity stages.
- [ ] **Temporal Dynamics**: If simulations are time-dependent, visual progress bars or animations of potential changes.
