"""
A plain solve of MECHA's standard problem on each section (devplan_metafspm.md, §8, steps 3–4), until the graph system
(step 5): the section alone, its border walls and junctions at the soil's potential through their soil conductances,
its xylem vessels (barrier > 0) or protophloem cells (barrier 0) at the axial potential through theirs. The matrix
is the legacy's (conductance.section_matrix); the solve, scipy's sparse direct solver.
"""
import numpy as np
from scipy.sparse.linalg import spsolve

from openalea.mecha.conductance import section_matrix


def section_potentials(net, K, in_matrix, soil, axial, axial_rhs, nodes, edges, psi_soil, psi_axial) -> np.ndarray:
    """ψ (hPa) on the section's *nodes* (in that order): W ψ = −soil·ψ_soil − axial·ψ_axial."""
    W = section_matrix(net, K, in_matrix, soil, axial, nodes, edges)
    rhs = -soil[nodes] * psi_soil - axial_rhs[nodes] * psi_axial
    return spsolve(W, rhs)


def edge_flows(net, K, psi, edges) -> np.ndarray:
    """Q = K (ψ_a − ψ_b) (cm³ d⁻¹) on *edges*, from their first node to their second; ψ per DataStructure node."""
    return K[edges] * (psi[net.a[edges]] - psi[net.b[edges]])
