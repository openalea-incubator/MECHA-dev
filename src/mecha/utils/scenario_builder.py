"""
scenario_builder.py

Shared geometry, initial-condition, and boundary-condition helpers for
MECHA scenarios.

All public methods are static; ScenarioBuilder is a logical namespace.
"""

import dataclasses

import numpy as np

from mecha.mecha_class import Mecha

# Tissue groups that are symplastically isolated or not true parenchyma
NONPARENCHYMA_CGROUPS = frozenset({13, 19, 20, 11, 12})


@dataclasses.dataclass
class NodeInfo:
    """Unified representation for an APO wall node or SYM cell node."""

    node_id: int          # full network index
    r: float              # radial distance from root center (µm)
    x: float              # x position (µm)
    y: float              # y position (µm)
    cell_id: int = -1     # node_id − n_wall_junction; −1 for wall nodes
    cgroup: int = -1      # tissue group; −1 for wall nodes
    cell_counts: dict = dataclasses.field(default_factory=dict)
    # Wall nodes: {'cortex': 2, 'stele_overall': 0, 'endo': 0, ...}
    # Cell nodes: {} (cell type encoded in cgroup)


class ScenarioBuilder:
    """
    Shared geometry, IC, and BC helpers for MECHA solute-transport tests.
    """

    # ── Geometry helpers ──────────────────────────────────────────────────────

    @staticmethod
    def root_center(mecha: Mecha) -> tuple:
        """(cx, cy) centroid of all cell nodes in µm."""
        nwj = mecha.network.n_wall_junction
        xs, ys = [], []
        for nd, d in mecha.network.graph.nodes(data=True):
            if mecha.indice[nd] >= nwj and 'position' in d:
                xs.append(d['position'][0])
                ys.append(d['position'][1])
        return float(np.mean(xs)), float(np.mean(ys))

    @staticmethod
    def collect_display_nodes(mecha: Mecha, cx: float, cy: float,
                               key: str = 'apo',
                               filter_fn=None) -> list:
        """
        Collect displayable nodes for the requested transport pathway.

        key='apo'  Wall nodes with positive length (L > 0).
                   NodeInfo.cell_counts populated from all 'count_*' graph
                   attributes (prefix stripped: 'count_cortex' → 'cortex').
        key='sym'  Cell nodes with positive area.
                   NodeInfo.cgroup populated.

        filter_fn  Optional callable(NodeInfo) -> bool.  When None, all
                   qualifying nodes are returned.

        Returns list[NodeInfo].
        """
        nwj    = mecha.network.n_wall_junction
        result = []

        if key == 'apo':
            wl = mecha.network.wall_lengths
            for nd, d in mecha.network.graph.nodes(data=True):
                idx = mecha.indice[nd]
                if idx < nwj and 'position' in d and wl.get(idx, 0.0) > 0:
                    r = np.hypot(d['position'][0] - cx, d['position'][1] - cy)
                    cell_counts = {
                        k[len('count_'):]: int(v)
                        for k, v in d.items()
                        if k.startswith('count_')
                    }
                    node = NodeInfo(
                        node_id=int(idx),
                        r=float(r),
                        x=float(d['position'][0]),
                        y=float(d['position'][1]),
                        cell_counts=cell_counts,
                    )
                    if filter_fn is None or filter_fn(node):
                        result.append(node)

        elif key == 'sym':
            ca = mecha.network.cell_areas
            for nd, d in mecha.network.graph.nodes(data=True):
                idx = mecha.indice[nd]
                if idx < nwj or 'position' not in d:
                    continue
                cell_id = idx - nwj
                area = float(ca[cell_id]) if cell_id < len(ca) else 0.0
                if area <= 0:
                    continue
                cg = int(d.get('cgroup', -1))
                r = np.hypot(d['position'][0] - cx, d['position'][1] - cy)
                node = NodeInfo(
                    node_id=int(idx),
                    r=float(r),
                    x=float(d['position'][0]),
                    y=float(d['position'][1]),
                    cell_id=int(cell_id),
                    cgroup=cg,
                )
                if filter_fn is None or filter_fn(node):
                    result.append(node)

        else:
            raise ValueError(f"key must be 'apo' or 'sym', got {key!r}")

        return result

    # ── Initial conditions ────────────────────────────────────────────────────

    @staticmethod
    def circular_ic(mecha: Mecha, nodes: list,
                    key: str = 'apo',
                    percentile: tuple = (40, 60),
                    filter_fn=None,
                    c_fn=None) -> tuple:
        """
        Concentration IC with a radial percentile ring.

        key='apo'   Candidates filtered by cell_counts.get('cortex', 0) >= 1.
                    c0 shape (n_wall_junction,), indexed by node_id.
        key='sym'   Candidates not in NONPARENCHYMA_CGROUPS
                    (falls back to all nodes when the filter leaves nothing).
                    c0 shape (n_cells,), indexed by cell_id.
        key='full'  All provided nodes are candidates.
                    c0 shape (n_total,), indexed by node_id.

        filter_fn   Callable(NodeInfo) -> bool overriding the key-based default
                    candidate selection.
        c_fn        Callable(r_µm) -> float for graded concentrations.
                    Default: 1.0 everywhere in the percentile ring.

        Returns (c0, r_lo, r_hi) with radii in µm.
        """
        nwj = mecha.network.n_wall_junction

        if filter_fn is not None:
            _filter = filter_fn
        elif key == 'apo':
            _filter = lambda n: n.cell_counts.get('cortex', 0) >= 1
        elif key == 'sym':
            _filter = lambda n: n.cgroup not in NONPARENCHYMA_CGROUPS
        else:
            _filter = lambda n: True   # 'full' and any other key

        if key == 'apo':
            cands = [(n.node_id, n.r) for n in nodes if _filter(n)]
            size  = nwj
        elif key == 'sym':
            cands = [(n.cell_id, n.r) for n in nodes if _filter(n)]
            if not cands:
                cands = [(n.cell_id, n.r) for n in nodes]
            size  = mecha.network.n_cells
        elif key == 'full':
            cands = [(n.node_id, n.r) for n in nodes if _filter(n)]
            size  = mecha.network.graph.number_of_nodes()
        else:
            raise ValueError(f"key must be 'apo', 'sym', or 'full', got {key!r}")

        r_vals = [r for _, r in cands]
        r_lo, r_hi = np.percentile(r_vals, list(percentile))
        c0 = np.zeros(size)
        for ci, r in cands:
            if r_lo <= r <= r_hi:
                c0[ci] = c_fn(r) if c_fn is not None else 1.0
        return c0, float(r_lo), float(r_hi)

    # ── Boundary conditions ───────────────────────────────────────────────────

    @staticmethod
    def solute_bc(nodes: list,
                  key: str = 'apo',
                  c_value: float = 0.0,
                  filter_fn=None,
                  r_percentile: float = 20.0) -> dict:
        """
        Boundary condition at selected nodes.

        key='apo'  Default: wall nodes touching stele but not cortex or endo.
                   Returns {node_id: c_value}.
        key='sym'  Default: cell nodes below r_percentile-th percentile radius.
                   Returns {cell_id: c_value}.

        filter_fn    Callable(NodeInfo) -> bool overriding the key-based default.
        c_value      Concentration at the boundary (default: 0.0 = absorbing).
        r_percentile Used only for the key='sym' default filter.
        """
        if key == 'apo':
            if filter_fn is None:
                _filter = lambda n: (
                    n.cell_counts.get('stele_overall', 0) > 0
                    and n.cell_counts.get('cortex', 0) == 0
                    and n.cell_counts.get('endo', 0) == 0
                )
            else:
                _filter = filter_fn
            return {n.node_id: c_value for n in nodes if _filter(n)}

        elif key == 'sym':
            if filter_fn is None:
                r_thresh = float(np.percentile([n.r for n in nodes], r_percentile))
                _filter = lambda n: n.r < r_thresh
            else:
                _filter = filter_fn
            return {n.cell_id: c_value for n in nodes if _filter(n)}

        else:
            raise ValueError(f"key must be 'apo' or 'sym', got {key!r}")

    # ── Pe scaling and time-stepping ──────────────────────────────────────────

    @staticmethod
    def scale_fluxes_pe1(mecha: Mecha, A_mat, ring_idxs: np.ndarray,
                         D: float, r_lo_cm: float) -> tuple:
        """
        Rescale edge_flux_list[0][0] to achieve global Pe = 1.

        The mean advective speed at ring nodes is
            v_raw = mean(|A_ii|[ring]) / (h × t)   [cm/d]
        and fluxes are multiplied by v_tgt / v_raw where v_tgt = D / r_lo_cm.

        Modifies mecha.edge_flux_list[0][0] in-place.

        Returns (flux_scale, v_tgt [cm/d], T_transit [d]).
        T_transit = r_lo_cm / v_tgt = r_lo_cm² / D.
        """
        height_cm = float(mecha.geometry.maturity_stages[0].get('height')) * 1e-4
        thick_cm  = mecha.geometry.thickness * 1e-4
        Ad_ring   = np.abs(A_mat.diagonal())[ring_idxs]
        valid_v   = Ad_ring > 0
        v_tgt     = D / r_lo_cm
        if valid_v.any():
            v_raw      = float(Ad_ring[valid_v].mean()) / (height_cm * thick_cm)
            flux_scale = v_tgt / v_raw if v_raw > 0 else 1.0
        else:
            flux_scale = 1.0
        for f in mecha.edge_flux_list[0][0]:
            f['flux'] *= flux_scale
        return flux_scale, v_tgt, r_lo_cm / v_tgt

    @staticmethod
    def has_advection(A_mat, ring_ids: np.ndarray) -> bool:
        """True when at least one ring node carries a non-zero advective flux."""
        Ad = np.abs(A_mat.diagonal())
        return bool(Ad.max() > 0 and len(ring_ids) > 0 and Ad[ring_ids].max() > 0)

    @staticmethod
    def cell_timescale(st, D_mat, ring_ids: np.ndarray, nwj: int) -> float:
        """
        Mean cell turnover time τ_cell = V_cell / |D_cell| for ring cells (d).

        Uses the PD diffusion matrix diagonal and node volumes from the solver,
        which accounts for PD pore geometry rather than assuming D_PD directly.
        Falls back to 1.0 d when no valid ring cells are found.
        """
        D_ring   = np.abs(D_mat.diagonal())[ring_ids]
        vols_all = st._compute_node_volumes(0)
        V_ring   = vols_all[nwj + ring_ids]
        mask     = (D_ring > 0) & (V_ring > 0)
        if not mask.any():
            return 1.0
        return float(np.mean(V_ring[mask] / D_ring[mask]))

    @staticmethod
    def auto_dt(tau: float, n_steps_per_tau: int = 10) -> float:
        """
        Round tau / n_steps_per_tau down to 1 significant figure.

        Produces a dt that gives ~n_steps_per_tau steps per characteristic
        timescale, avoiding awkward sub-step precision.
        Minimum value: 1e-4 d.
        """
        dt_raw = tau / n_steps_per_tau
        mag    = 10.0 ** np.floor(np.log10(max(dt_raw, 1e-10)))
        return max(1e-4, round(dt_raw / mag) * mag)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @staticmethod
    def r_cm(c: np.ndarray, nodes: list, key: str = 'apo') -> float:
        """
        Concentration-weighted mean radial position (µm) over displayed nodes.

        key='apo'  Indexes c by node_id.
        key='sym'  Indexes c by cell_id.
        """
        if key == 'apo':
            idxs = np.array([n.node_id for n in nodes], dtype=int)
        elif key == 'sym':
            idxs = np.array([n.cell_id for n in nodes], dtype=int)
        else:
            raise ValueError(f"key must be 'apo' or 'sym', got {key!r}")

        vals = c[idxs]
        r    = np.array([n.r for n in nodes])
        tot  = float(vals.sum())
        if tot < 1e-30:
            return float('nan')
        return float(np.dot(r, vals)) / tot
