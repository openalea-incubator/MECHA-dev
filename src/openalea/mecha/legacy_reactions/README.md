MECHA's `reactions` branch (solute transport, calibration), frozen: `origin/reactions` at 4c9c8c2 (2026-10-15),
verbatim. Only its internal imports are rewritten (`openalea.mecha.X` → `openalea.mecha.legacy_reactions.X`), so that
it runs on its own beside the legacy (`openalea.mecha.legacy`, `origin/to_metafspm` at 2b995b2).

It is the reference of the solute transport's port (devplan_metafspm.md, §23). Its simulations are in
`example/legacy_reactions/simulations/` (their `sys.path` insertions of sibling checkouts removed). It is not developed
further: the solute work continues in the port.
