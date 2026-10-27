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


def axial_conductances(geo, height, barrier) -> tuple:
    """
    (Kx, Kx_phloem) (cm⁴ hPa⁻¹ d⁻¹): Σ of the conduits' Poiseuille conductances × height / 1e4, the xylem vessels'
    (barrier > 0) and the sieve tubes' (barrier > 0) or protosieve cells' (barrier 0), each in cell order. Kx is 0 at
    barrier 0 (no mature xylem: the legacy fails there, legacy e).
    """
    height = float(height)
    position = np.empty(geo.legacy_order.size, dtype=np.int64)
    position[geo.legacy_order] = np.arange(geo.legacy_order.size)

    def total(cells, viscosity):
        """numpy's sum of the legacy's per-node vector (its node ids: walls, junctions, cells), × height / 1e4."""
        K_axial = np.zeros(geo.legacy_order.size)
        for c in cells:
            K_axial[position[c]] = _poiseuille(geo.cell_area[c], height, viscosity)
        return float(np.sum(K_axial)) * height / 1.0E04
    if barrier > 0:
        return total(geo.typed["xylem"], VISCOSITY_XYLEM), total(geo.typed["sieve"], VISCOSITY_PHLOEM)
    return 0., total(geo.typed["protosieve"], VISCOSITY_PHLOEM)
