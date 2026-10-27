"""
MECHA's section-scale results from a solution (doc/equations.rst): the soil uptake and the radial
conductivity kr (``Mecha.standard_water_flow``), and the axial conductances by Poiseuille's law
(``Mecha.calculate_axial_conductance``), with the legacy's operations and summation orders, so that they are exact.
The legacy's module imports pylab's names (``from pylab import *``): its ``sum`` is numpy's, a pairwise summation,
emulated here on arrays laid out as the legacy's.
"""
import numpy as np

VISCOSITY_XYLEM = 1.3076E-05          # hPa d ×3600×24: water at 10 °C (1.3076E-3 Pa s), the legacy's
VISCOSITY_PHLOEM = 1.7E-05            # phloem sap (1.7E-3 Pa s, Ross-Elliott et al. 2017), the legacy's


def soil_uptake(geo, soil, psi, psi_soil) -> float:
    """
    The section's water uptake from the soil (cm³ d⁻¹, positive into the root): Σ K_soil (ψ_soil − ψ) over the
    border walls, then the border junctions, in the legacy's order (its wall ids, then its rebuilt junction ids).
    *soil* and *psi*: per local node of the section; *psi_soil*: one value, or one per local node.
    """
    walls = geo.walls[geo.border_wall[geo.walls]]
    junctions = geo.junction_order[geo.border_junction[geo.junction_order]]
    nodes = np.r_[walls, junctions]
    psi_soil = np.asarray(psi_soil, dtype=float)
    q = -soil[nodes] * (psi[nodes] - (psi_soil[nodes] if psi_soil.ndim else psi_soil))
    return float(np.sum(q))


def radial_conductivity(uptake, psi_soil, psi_axial, perimeter, height) -> float:
    """kr (cm hPa⁻¹ d⁻¹): the uptake per unit potential difference and root surface (perimeter (cm) × height (µm))."""
    return uptake / (psi_soil - psi_axial) / perimeter / height / 1.0E-04


def poiseuille(area, height, viscosity):
    """Poiseuille conductances (cm³ hPa⁻¹ d⁻¹) of conduits of cross-section *area* (µm²) over *height* (µm), arrays."""
    return area ** 2 / (8 * 3.141592 * height * viscosity / 3600 / 24) * 1.0E-12


def _poiseuille(area, height, viscosity) -> float:
    """One conduit's axial conductance, as the legacy writes it (Python scalars: area ** 2 is libm's pow)."""
    return float(area) ** 2 / (8 * 3.141592 * height * viscosity / 3600 / 24) * 1.0E-12


def axial_conductances(geo, height, barrier, measured=None) -> tuple:
    """
    (Kx, Kx_phloem) (cm⁴ hPa⁻¹ d⁻¹): Σ of the conduits' Poiseuille conductances × height / 1e4, the xylem vessels'
    (barrier > 0) and the sieve tubes' (barrier > 0) or protosieve cells' (barrier 0), each in cell order; a conduit
    with a *measured* specific conductance (per local node, cm⁴ hPa⁻¹ d⁻¹, NaN: Poiseuille's) takes it (MECHA's
    Kax_source 2). Kx is 0 at barrier 0 (no mature xylem: the legacy fails there, legacy e).
    """
    height = float(height)
    position = np.empty(geo.legacy_order.size, dtype=np.int64)
    position[geo.legacy_order] = np.arange(geo.legacy_order.size)

    def total(cells, viscosity):
        """numpy's sum of the legacy's per-node vector (its node ids: walls, junctions, cells), × height / 1e4."""
        K_axial = np.zeros(geo.legacy_order.size)
        for c in cells:
            given = np.nan if measured is None else float(measured[c])
            K_axial[position[c]] = given / (height / 1.0E04) if np.isfinite(given) else \
                _poiseuille(geo.cell_area[c], height, viscosity)
        return float(np.sum(K_axial)) * height / 1.0E04
    if barrier > 0:
        return total(geo.typed["xylem"], VISCOSITY_XYLEM), total(geo.typed["sieve"], VISCOSITY_PHLOEM)
    return 0., total(geo.typed["protosieve"], VISCOSITY_PHLOEM)


def transmembrane_layers(ds, segment) -> "pd.DataFrame":
    """
    The section *segment*'s standard transmembrane flows by layer row (MECHA's ``uptake_layer_plus``, ``_minus``,
    ``stf_layer_plus``, ``_minus``), from MechaHydraulics' standard_flow, standard_transmembrane_fraction and
    membrane_layer: per row, the membranes' flows into their cells (cm³ d⁻¹) summed where positive and where negative,
    and their fractions likewise.
    """
    import pandas as pd
    from openalea.mecha.geometry import section_parts
    _, edges = section_parts(ds)[segment]
    layer = np.asarray(ds.get("membrane_layer"))[edges].astype(np.int64)
    fraction = np.asarray(ds.get("standard_transmembrane_fraction"))[edges]
    uptake = np.asarray(ds.get("standard_uptake"))[list(ds.entity_ids("SubOrgan")).index(segment)]
    flow = fraction * uptake
    on = layer >= 0
    n = int(layer[on].max()) + 1 if on.any() else 0

    def by_row(values):
        return np.bincount(layer[on], weights=values[on], minlength=n)
    return pd.DataFrame({"layer": np.arange(n), "uptake_plus": by_row(np.maximum(flow, 0.)),
                         "uptake_minus": by_row(np.minimum(flow, 0.)),
                         "fraction_plus": by_row(np.maximum(fraction, 0.)),
                         "fraction_minus": by_row(np.minimum(fraction, 0.))})


def transmembrane_cells(ds, segment) -> "pd.DataFrame":
    """
    The section *segment*'s standard transmembrane fractions by protoplast (MECHA's ``stf_cell_plus``, ``_minus``):
    per cell (its DataStructure node), its membranes' fractions summed where positive and where negative.
    """
    import pandas as pd
    from openalea.mecha.geometry import section_parts
    nodes, edges = section_parts(ds)[segment]
    view = ds.to_graph_view()
    kind = np.asarray(ds.get("kind"))
    on = np.asarray(ds.get("membrane_layer"))[edges] >= 0
    fraction = np.asarray(ds.get("standard_transmembrane_fraction"))[edges][on]
    tail, head = view.tail[edges][on], view.head[edges][on]
    cell = np.where(kind[tail] == 0, tail, head)
    cells = nodes[kind[nodes] == 0]
    index = np.searchsorted(cells, cell)
    return pd.DataFrame({"node": cells,
                         "fraction_plus": np.bincount(index, weights=np.maximum(fraction, 0.), minlength=cells.size),
                         "fraction_minus": np.bincount(index, weights=np.minimum(fraction, 0.), minlength=cells.size)})


def axial_layers(mecha, segment) -> "pd.DataFrame":
    """
    The section *segment*'s standard axial flows by layer row (MECHA's ``flow_xyl_layer``, barrier > 0, and
    ``flow_sieve_layer``, barrier 0 and the sieve tubes' at barrier > 0), from a solved MechaHydraulics *mecha*: per
    row of its conduits' ranks (legacy_rows), the water entering the section from the axial boundary (cm³ d⁻¹;
    negative: leaving it, the xylem's uptake), K (ψ_axial − ψ) and the flow conditions' shares.
    """
    import pandas as pd
    from openalea.mecha.osmotic import legacy_rows
    ds = mecha.data_structure
    nodes, _, geo = mecha.section_geometries()[segment]
    psi = np.asarray(ds.get("standard_psi"), dtype=float)[nodes]
    entering = (np.asarray(ds.get("axial_boundary_conductance"), dtype=float)[nodes]
                * (np.asarray(ds.get("standard_axial_psi"), dtype=float)[nodes] - psi)
                + np.asarray(ds.get("standard_axial_inflow"), dtype=float)[nodes])
    rank_to_row, _, _ = legacy_rows(geo.n_layer, geo.layer_dist)
    n = int(np.nanmax(rank_to_row)) + 1
    barrier = int(np.asarray(ds.get("barrier"))[nodes[0]])
    xylem = geo.xylem_cells if barrier > 0 else np.zeros(0, dtype=np.int64)
    phloem = geo.sieve_cells if barrier > 0 else np.unique(geo.protosieve_cells)

    def by_row(cells):
        rows = rank_to_row[geo.cell_rank[cells]].astype(np.int64)
        return np.bincount(rows, weights=entering[cells], minlength=n)
    return pd.DataFrame({"layer": np.arange(n), "xylem": by_row(xylem), "phloem": by_row(phloem)})
