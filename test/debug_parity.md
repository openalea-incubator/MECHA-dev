# Debug Parity: GRANAP vs. XML Matrix Divergence

This document summarizes the attempts and findings to resolve the numerical discrepancies in the hydraulic matrix (`Matrix_W`) between network construction paths in MECHA:
1.  **`mecha_ganache_1`**: Direct from a GRANAP generated graph (`NetworkExporter`).
2.  **`mecha_ganache_2`**: From an XML file written by `AnatomyWriter.write_to_xml()`.
3.  **`mecha_classic_3`**: From known CellSet XML file.   

The overarching goal is to achieve exact numerical parity between these two pathways to ensure simulations are reproducible regardless of the input method.

## Identified Divergences and Fixes

### 1. Cell ID and Ordering Mismatch
**Observation:** `AnatomyWriter.write_to_xml()` was assigning fresh, sequential cell IDs using `range(len(valid_cells))` when exporting to XML. Conversely, the `NetworkExporter` (and the underlying GRANAP graph) retained the explicit index of the `cells_gdf` GeoDataFrame.
**Consequence:** Because the two pools filtered invalid geometries slightly differently, the cell node IDs in the resulting XML did not map 1-to-1 to the graph node IDs. This caused a shift in the wall-to-cell connectivity map, altering matrix structure.
**Attempted Fix:** Refactored `write_to_xml` to iterate over a filtered `valid_gdf` while strictly preserving and explicitly writing the GDF index as the cell `id`.
**Result:** This fix was unsuccessful in aligning the ordering in the nx.Graph between the two pathways. 

### 2. Spurious Junction Nodes (Precision and Deduplication)
**Observation:** Even with aligned cell IDs, the XML path generated more junction nodes (e.g., 79 extra) compared to directly using the graph.
**Root Cause:** Sub-floating point precision issues during deduplication. Old MECHA generated a string key for a junction using unrounded raw coordinates: `pos_key = "x" + str(x) + "y" + str(y)`. The new MECHA `create_wall_junction_nodes` implementation rounded the floats *before* creating the f-string key. Due to IEEE 754 arithmetic, two mathematically identical junction endpoints sometimes produced slightly different floating-point noise when re-read from XML, which survived rounding differently or resulted in separate keys.
**Attempted Fix:** Updated `NetworkBuilder.create_wall_junction_nodes` to collect raw (unrounded) coordinates specifically for building the deduplication key (mimicking the exact logic of old MECHA), while using the rounded coordinates for actual geometric position assignment.
**Result:** This fix was unsuccessful in aligning the number of junction nodes between the two pathways. 

### 3. Topology Polygon Filtering Mismatch (KD-Tree Snapping Limits)
**Observation:** Even after fixing the string keys, there remained a ~77 junction difference.
**Root Cause:** `CellGenerator._build_topology()` handles vertex snapping across adjacent polygons using a `cKDTree`. The strictness of this snap (`snap_tol`) is dynamically calculated based on the 5th percentile of all edge lengths in the provided polygon set.
- `NetworkExporter.export()` was passing **all** cells to `_build_topology`, including those with `None` geometries.
- `AnatomyWriter.write_to_xml()` explicitly filtered out invalid geometries *before* extracting the polygon set.
This mismatch meant the two pathways computed different `snap_tol` values and generated slightly different KD-tree clusters. Consequently, some vertices snapped together in one context but not the other, leading to mismatched canonical junction coordinates.
**Attempted Fix:** Enforced identical geometry filtering. `NetworkExporter.export` now applies the precise same `valid_mask` filter as `write_to_xml` before calling `_build_topology`, guaranteeing identical snapping behavior and canonical outputs.
**Result:** This fix was unsuccessful in aligning the number of junction nodes between the two pathways. 

### 4. Centroid Calculation Mismatch
**Observation:** Even after fixing the structural and snapping issues, a small numerical difference persisted in the hydraulic matrix.
**Root Cause:** The `NetworkBuilder.create_cell_nodes` method calculated cell centroids by averaging the coordinates of the wall endpoints connected to that cell. In contrast, `AnatomyWriter.write_to_xml` used the `shapely.Polygon.centroid` property.
While mathematically similar, these two methods produce slightly different results due to floating-point arithmetic and the nature of the calculations (mean of vertices vs. geometric centroid of the area). This difference propagated through the hydraulic conductivity calculations, leading to the observed matrix divergence.
**Attempted Fix:** Updated `NetworkBuilder.create_cell_nodes` to optionally use the same `shapely.Polygon.centroid` method as `AnatomyWriter`. A new `centroid_method` parameter was added to `build_network`, defaulting to "shapely" to ensure parity with the XML export method.
**Result:** This fix was successful in aligning the cell centroids between the two pathways. 

### 5. Topological Mismatch in Junction Edges
**Observation:** `plot_edge_and_node_differences` showed topological differences in the edges natively constructed for `Ganache_1` vs `Ganache_2`. `Ganache_1` had incorrectly connected junction nodes.
**Root Cause:** In Ganache 1, `NetworkExporter.export()` built edges between walls and junctions. The node ID assignments for junctions were globally sorted and entirely disconnected from how the XML parser sequentially iterated through walls to discover them. In Ganache 2 (`create_wall_junction_nodes()`), `pos_key` strings mapped sequentially encountered walls. Also, `distnode_wall_cell` was erroneously set to `dist_wall_cell` instead of `dist_junc_wall_node` when Ganache 1 added the apoplastic edges.
**Attempted Fix:** Restructured `NetworkBuilder.populate_from_network()` (Step 4) to strip original `'wall'` edges out and precisely emulate the internal looping behavior of `create_wall_junction_nodes` over `self.n_walls` to produce identically assigned node IDs. Finally, we recalculate edges natively using `build_wall_connections()` to ensure standard metrics like `lateral_distance` match perfectly.
**Result:** Topological alignment between Ganache 1 and Ganache 2 junction connectivity matches exactly.

### 6. Stele Rank Discrepancy (Pandas Int64 Fallback)
**Observation:** `cgroup` and `rank` for stele cells were improperly mapped in Ganache 1 compared to Ganache 2.
**Root Cause:** The `isinstance(cgroup, (int, float))` type-check in `populate_from_network` failed against Pandas/NumPy `numpy.int64` typing. This caused stele layer integers constructed algorithmically by GRANAP to forcefully fall back to `type_mapper` default integers (`5`, `13`, `11`). Meanwhile, Ganache 2 correctly parsed `cgroup` integers right out of the XML using `int(XML_group)`.
**Attempted Fix:** Strengthened validation loop to execute `cgroup = int(cgroup)` immediately, throwing an exception only if non-numeric/empty types arise.
**Result:** Ensures intact GRANAP grouping designations endure to correctly dictate ranks spanning `_rank_cells_from_graph`.

### 7. Cell Type Assignment (Intercellular and Passage Cells)
**Observation:** When intercellular spaces or passage cells were included, numerical divergence appeared in the conductance matrix, even though the total count of cells seemed consistent.
**Root Cause:** The `HydraulicCellManager.sync_from_network()` method had a broken conditional block for determining `cell_type`. It attempted to use `elif` within a parenthesized expression (invalid syntax in Python ternary) and had a logic error in checking the length of `intercellular_cells` (`len(network.intercellular_cells > 0)`). This caused fallback to `node_data.get("cell_type", "")`, which could be empty or inconsistent.
**Attempted Fix:** Fixed the `cell_type` assignment logic in `sync_from_network` by converting it to a standard `if/elif/else` block that correctly prioritizes `network.cell_types`, `network.intercellular_cells`, and `network.passage_cells`.
**Result:** **FULL PARITY ACHIEVED.** Both `Matrix_W` and derived hydraulic properties (`kr`, `Kx`) now match with machine-precision accuracy even when complex tissues like aerenchyma are included.

## Current State
Numerical parity has been fully established across structural, topological, and hydraulic levels! The direct-to-graph (`Ganache_1`) and XML-mediated (`Ganache_2`) pathways now produce matrices and simulation results that are identical to within floating-point epsilon (~1e-15).

## New tools for debugging
- [x] creation of more visualization tools to compare the two networks and identify the source of the numerical discrepancy.
    - [x] create a function to plot the difference between the two networks
    - [x] modify the function to plot the absolute difference between the two matrices (blanck if no difference, large points for differences)
    - [x] create a function to plot the two networks side by side with cgroup coloring
    - [x] create a function to plot the two networks side by side with rank coloring
    - [x] create a function to locate edge and node differences (color code for type of difference junction, wall, cell)
    - [x] create a function to plot where the intercellular spaces are in the two networks


Max difference in Matrix_W: 1.4242e-15
Matrices are practically EXACTLY identical.

--- Ganache network 1 Summary ---
  Cells: 2104
  Walls: 6307
  Xylem cells: 25
  Sieve cells: 50
  Intercellular cells: 338

  {'barrier': 1, 'kr': 8.38849e-05, 'Kx': 1.1849}
  {'barrier': 3, 'kr': 3.53086e-05, 'Kx': 1.1849}

--- Ganache network 2 Summary ---
  Cells: 2104
  Walls: 6307
  Xylem cells: 25
  Sieve cells: 50
  Intercellular cells: 338

  {'barrier': 1, 'kr': 8.38849e-05, 'Kx': 1.1849}
  {'barrier': 3, 'kr': 3.53086e-05, 'Kx': 1.1849}
  