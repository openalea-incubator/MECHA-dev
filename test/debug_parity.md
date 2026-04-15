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
**Fix:** Refactored `write_to_xml` to iterate over a filtered `valid_gdf` while strictly preserving and explicitly writing the GDF index as the cell `id`.

### 2. Spurious Junction Nodes (Precision and Deduplication)
**Observation:** Even with aligned cell IDs, the XML path generated more junction nodes (e.g., 79 extra) compared to directly using the graph.
**Root Cause:** Sub-floating point precision issues during deduplication. Old MECHA generated a string key for a junction using unrounded raw coordinates: `pos_key = "x" + str(x) + "y" + str(y)`. The new MECHA `create_wall_junction_nodes` implementation rounded the floats *before* creating the f-string key. Due to IEEE 754 arithmetic, two mathematically identical junction endpoints sometimes produced slightly different floating-point noise when re-read from XML, which survived rounding differently or resulted in separate keys.
**Fix:** Updated `NetworkBuilder.create_wall_junction_nodes` to collect raw (unrounded) coordinates specifically for building the deduplication key (mimicking the exact logic of old MECHA), while using the rounded coordinates for actual geometric position assignment.

### 3. Topology Polygon Filtering Mismatch (KD-Tree Snapping Limits)
**Observation:** Even after fixing the string keys, there remained a ~77 junction difference.
**Root Cause:** `CellGenerator._build_topology()` handles vertex snapping across adjacent polygons using a `cKDTree`. The strictness of this snap (`snap_tol`) is dynamically calculated based on the 5th percentile of all edge lengths in the provided polygon set.
- `NetworkExporter.export()` was passing **all** cells to `_build_topology`, including those with `None` geometries.
- `AnatomyWriter.write_to_xml()` explicitly filtered out invalid geometries *before* extracting the polygon set.
This mismatch meant the two pathways computed different `snap_tol` values and generated slightly different KD-tree clusters. Consequently, some vertices snapped together in one context but not the other, leading to mismatched canonical junction coordinates.
**Fix:** Enforced identical geometry filtering. `NetworkExporter.export` now applies the precise same `valid_mask` filter as `write_to_xml` before calling `_build_topology`, guaranteeing identical snapping behavior and canonical outputs.

### 4. Centroid Calculation Mismatch
**Observation:** Even after fixing the structural and snapping issues, a small numerical difference persisted in the hydraulic matrix.
**Root Cause:** The `NetworkBuilder.create_cell_nodes` method calculated cell centroids by averaging the coordinates of the wall endpoints connected to that cell. In contrast, `AnatomyWriter.write_to_xml` used the `shapely.Polygon.centroid` property.
While mathematically similar, these two methods produce slightly different results due to floating-point arithmetic and the nature of the calculations (mean of vertices vs. geometric centroid of the area). This difference propagated through the hydraulic conductivity calculations, leading to the observed matrix divergence.
**Fix:** Updated `NetworkBuilder.create_cell_nodes` to optionally use the same `shapely.Polygon.centroid` method as `AnatomyWriter`. A new `centroid_method` parameter was added to `build_network`, defaulting to "shapely" to ensure parity with the XML export method.

## Current State
The structural alignment (number of nodes, walls, junctions, and connectivity mappings) should be steadily unified between the GRANAP-memory export and the XML-file export. Test suites (like `test_ganache.py`) still produce a small difference in the hydraulic matrix, which is under investigation.

## New tools for debugging
- [x] creation of more visualization tools to compare the two networks and identify the source of the numerical discrepancy.
    - [x] create a function to plot the difference between the two networks
    - [x] create a function to plot the absolute difference between the two matrices
    - [x] create a function to plot the two networks side by side with cgroup coloring
    - [x] create a function to plot the two networks side by side with rank coloring

# Next steps