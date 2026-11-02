"""
Condensed root sections (GRANAP's condensed_anatomy): a section held by its conduits and its soil node only, joined by
equivalent Connections (network_path 8) whose conductances are the exact reduction (solve.network.kron_reduction) of
the full section's network, computed once per kind of section.

A kind: (the full anatomy's template, the segment's barrier stage, its length or length class, its plant's
parameters). Each kind is evaluated on a library of full sections (a population of its own, one section per kind,
isolated), by MECHA's cell-level code:

  pairs        the equivalent conductances, one per pair of kept nodes (the condensed template's equivalent
               Connections, in their order)
  source       the sources at the kept nodes (the plant problem's: the elongation sink and the osmotic terms, the
               interior's carried to the kept nodes)
  kr, standard_uptake, perimeter
               MECHA's standard problem on the full section (its kr: the kind's)
  reduction    the section's KronReduction, and its interior sources: the cells' potentials recovered on request
               (recover)
  osmotic      the responses of the kept nodes' sources to the osmotic potentials (with the presets, condensed_osmotic
               "presets"): to the kept nodes' own (osmotic_kept, n_kept × n_kept) and to each hidden tissue's
               ({tissue: (response to a uniform potential, response to the kind's)}), and the kind's kept potentials
"""
from dataclasses import dataclass

import numpy as np

EQUIVALENT = 8                    # GRANAP's path of a condensed section's equivalent Connection


@dataclass
class Kind:
    pairs: np.ndarray
    source: np.ndarray
    kr: float
    standard_uptake: float
    perimeter: float
    reduction: object
    interior_source: np.ndarray
    interior_nodes: np.ndarray        # the interior's template nodes (the full section's local indices)
    kept_nodes: np.ndarray            # the kept nodes' template nodes
    osmotic_kept: np.ndarray = None   # the kept sources' response to the kept nodes' osmotic potentials
    osmotic_tissues: dict = None      # {tissue: (response to a uniform potential, response to the kind's)}
    base_kept_osmotic: np.ndarray = None

    def osmotic_source(self, kept_potential=None, tissues=None) -> np.ndarray:
        """The change of the kept nodes' sources for the conduits' *kept_potential* (NaN: the kind's) and the
        hidden tissues' *tissues* ({tissue: π}), from the kind's osmotic potentials (0 without osmotic terms)."""
        out = np.zeros(self.source.size)
        if self.osmotic_kept is None:
            return out
        if kept_potential is not None:
            given = np.asarray(kept_potential, dtype=float)
            delta = np.where(np.isfinite(given), given - self.base_kept_osmotic, 0.)
            out += self.osmotic_kept @ delta
        for tissue, value in (tissues or {}).items():
            response = self.osmotic_tissues.get(tissue)
            if response is not None:
                out += float(value) * response[0] - response[1]
        return out

    def recover(self, kept_values) -> np.ndarray:
        """The interior cells' potentials (in interior_nodes' order) for the kept nodes' *kept_values*."""
        return self.reduction.recover(kept_values, -self.interior_source)


def length_class(length, classes) -> float:
    """The length a kind is evaluated at: the segment's (*classes* "exact"), or its class of relative width
    *classes* ((1 + width)^k, the nearest)."""
    length = float(length)
    if classes == "exact" or length <= 0:
        return length
    base = np.log1p(float(classes))
    return float(np.exp(np.round(np.log(length) / base) * base))


def _library_model(templates):
    """The library's plant model: RootAxis (one segment per kind), each segment its template's full anatomy with its
    soil node (an instance), no junctions."""
    from openalea.granap.metafspm.architectures import RootAxis
    from openalea.granap.metafspm.component import GranapAnatomy
    from openalea.granap.metafspm.library import template_by_index
    from openalea.metafspm.coupling.component import StructuralComponent

    class LibraryAnatomy(StructuralComponent):
        @classmethod
        def initiate_plant(cls, g, plant, parameters):
            from openalea.granap.metafspm.component import _segments
            template_of = g.properties().setdefault("anatomy_template", {})
            for segment, index in zip(_segments(g, plant), templates):
                template_by_index(index).instance(g, segment, GranapAnatomy.add_soil_node, "soil")
                template_of[segment] = float(index)

    def __init__(self, data_structure, time_step):
        self.components = []

    return type("SectionLibrary", (), dict(initiators=(RootAxis, LibraryAnatomy), nodes="Compartment",
                                           from_scale="SubOrgan", wiring=(), __init__=__init__,
                                           run=lambda self: None))


def evaluate(requests: list, parameters: dict, settings: dict, model_class, presets: dict = None) -> list:
    """
    The kinds of *requests* [(template index, barrier, length)], for a plant of *parameters* ({name: value}) and
    the hydraulics model's *settings*: one library population, its sections evaluated by *model_class*
    (MechaHydraulics); with *presets* ({name: value}), MECHA's osmotic presets first (MechaOsmoticPresets: the
    cells' osmotic potentials and the membranes' reflection coefficients), and the kinds' osmotic responses.
    """
    from scipy.sparse import csr_matrix, diags
    from openalea.metafspm.scene.population import Planting, planting, population_data_structure
    from openalea.metafspm.solve.network import kron_reduction
    from openalea.mecha.geometry import SOIL_NODE

    templates = [int(t) for t, _, _ in requests]
    segments = [[1e-4, float(length), 0.] for _, _, length in requests]
    model = _library_model(templates)
    time_step = 3600.                          # the library's sections are steady: any step
    population = population_data_structure(planting([Planting()], model, {"root_segments": segments}), model,
                                           time_step=time_step)
    ds = population.data_structure
    owner = np.asarray(ds.owner("SubOrgan"))
    sections = ds.entity_ids("SubOrgan")
    if sections.size != len(requests):
        raise RuntimeError("condensed.evaluate: the library does not hold one section per kind")
    barrier = np.asarray([b for _, b, _ in requests], dtype=float)
    from openalea.granap.metafspm.component import GranapAnatomy
    GranapAnatomy(data_structure=ds, time_step=time_step)          # the anatomy's declarations (tissue: objects)
    hydraulics = model_class(data_structure=ds, time_step=time_step,
                             **dict(settings, plant_solver="full", soil_boundary="node_dirichlet"))
    for name, value in parameters.items():
        setattr(hydraulics, name, value)
    ds.set("barrier", barrier[owner])
    if presets is not None:
        from openalea.mecha.osmotic import MechaOsmoticPresets
        osmotic = MechaOsmoticPresets(data_structure=ds, time_step=time_step)
        for name, value in presets.items():
            setattr(osmotic, name, value)
        osmotic()
    hydraulics()
    view = ds.to_graph_view()
    K = np.asarray(ds.get("conductance"), dtype=float) + np.asarray(ds.get("xylem_axial_conductance"), dtype=float)
    drive = np.asarray(ds.get("osmotic_drive"), dtype=float)
    sink = np.asarray(ds.get("elongation_sink"), dtype=float)
    kind, nodes_all = np.asarray(ds.get("kind")), ds.entity_ids("node")
    family = ds.structure.values_of("family", nodes_all)
    template_node = ds.structure.values_of("template_node", nodes_all)
    conduit = np.array([family.get(int(v), -1.) in (0., 1.) for v in nodes_all.tolist()]) | (kind == SOIL_NODE)
    kr = np.asarray(ds.get("kr"), dtype=float)
    standard_uptake = np.asarray(ds.get("standard_uptake"), dtype=float)
    perimeter = np.asarray(ds.get("perimeter"), dtype=float)
    sigma = np.asarray(ds.get("reflection_coefficient"), dtype=float) if presets is not None else None
    potential = np.asarray(ds.get("osmotic_potential"), dtype=float) if presets is not None else None
    tissue_names = np.asarray(ds.get("tissue"), dtype=object)
    out = []
    for i in range(sections.size):
        nodes = np.flatnonzero(owner == i)
        local = -np.ones(ds.n_nodes(), dtype=np.int64)
        local[nodes] = np.arange(nodes.size)
        inside = (local[view.tail] >= 0) & (local[view.head] >= 0)
        edges = np.flatnonzero(inside)
        a, b = local[view.tail[edges]], local[view.head[edges]]
        n = nodes.size
        B = csr_matrix((np.r_[np.ones(edges.size), -np.ones(edges.size)],
                        (np.r_[a, b], np.r_[np.arange(edges.size), np.arange(edges.size)])), shape=(n, edges.size))
        laplacian = (B @ diags(K[edges]) @ B.T).tocsr()
        source = np.asarray(B @ (K[edges] * drive[edges])).reshape(-1) + sink[nodes]
        kept = conduit[nodes]
        reduction = kron_reduction(laplacian, kept)
        I, Kn = reduction.interior_nodes, reduction.kept_nodes
        equivalent_source = source[Kn] + reduction.inject(source[I])
        shunts = reduction.shunts()
        if np.abs(shunts).max(initial=0.) > 1e-9 * max(np.abs(reduction.reduced).max(initial=0.), 1e-300):
            raise RuntimeError("condensed.evaluate: a section's reduction has diagonal terms left (a Robin term "
                               "inside the section)")
        nodes_template = np.array([template_node[int(v)] for v in nodes_all[nodes].tolist()], dtype=np.int64)
        kind = Kind(pairs=reduction.pairs()[2], source=equivalent_source, kr=float(kr[i]),
                    standard_uptake=float(standard_uptake[i]), perimeter=float(perimeter[i]),
                    reduction=reduction, interior_source=source[I], interior_nodes=nodes_template[I],
                    kept_nodes=nodes_template[Kn])
        if sigma is not None:                          # the kept sources as linear functions of the potentials
            osmotic = (B @ diags(K[edges] * sigma[edges]) @ B.T).tocsr()      # B (K σ Bᵀ π): the sources' operator
            carry = reduction.injection()                                     # n_kept × n_interior
            response = np.asarray(osmotic[Kn].toarray()) + np.asarray((osmotic[I].T @ carry.T).T)   # n_kept × n
            here = potential[nodes]
            kind.osmotic_kept = response[:, Kn]
            kind.base_kept_osmotic = here[Kn].copy()
            names = tissue_names[nodes]
            kind.osmotic_tissues = {}
            for name in sorted({t for t in names[I].tolist() if t is not None}):
                cells = I[names[I] == name]
                kind.osmotic_tissues[name] = (response[:, cells].sum(axis=1), response[:, cells] @ here[cells])
        out.append(kind)
    return out
