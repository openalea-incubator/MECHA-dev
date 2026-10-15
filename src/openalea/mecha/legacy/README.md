The original MECHA source, frozen: `origin/to_metafspm` at 2b995b2, verbatim. Only its internal imports are rewritten
(`openalea.mecha.X` → `openalea.mecha.legacy.X`), so that it runs on its own.

It is the reference the metafspm port is compared with (`test/comparison/`, `example/comparison/`). It reads GRANAP's
organs through their legacy exports (networkx graph, cell GeoDataFrame, XML). It is not developed further: new work goes
into the main package (`openalea.mecha`), which is the port. Its own tests and tutorials are in `test/legacy/` and
`example/legacy/`.
