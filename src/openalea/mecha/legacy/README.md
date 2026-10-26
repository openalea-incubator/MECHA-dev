The original MECHA source, frozen: `develop` at 945b2be (2026-10-26; before, `origin/to_metafspm` at 2b995b2),
verbatim but for its internal imports, rewritten (`openalea.mecha.X` → `openalea.mecha.legacy.X`) so that it runs on
its own, and one repair: `develop`'s merge left the header of `HydraulicMatrixBuilder._contact_nodes` as a second
`_apply_soil_boundary` (every solve raised AttributeError); the header is 46a6f08's, `def _contact_nodes(self,
x_contact):`. The calibration package is not copied (it is in `openalea.mecha.legacy_reactions`).

It is the reference the metafspm port is compared with (`test/comparison/`, `example/comparison/`). It reads GRANAP's
organs through their legacy exports (networkx graph, cell GeoDataFrame, XML). It is not developed further: new work goes
into the main package (`openalea.mecha`), which is the port. Its own tests and tutorials are in `test/legacy/` and
`example/legacy/`.
