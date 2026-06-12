#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#       mecha.utils.hydraulic_cell
#
#       File author(s):
#           Adrien Heymans
#
#       Copyright © by UCLouvain
#       Distributed under the LGPL License.
#       See accompanying file LICENSE.txt or copy at
#           https://www.gnu.org/licenses/lgpl-3.0.en.html
#
# -----------------------------------------------------------------------
"""
HydraulicWall, HydraulicMembrane, HydraulicPlasmodesmata,
HydraulicCell and HydraulicCellManager
=======================================

Standalone descriptors for MECHA's hydraulic network.

``HydraulicWall`` carries:
  - geometry of a cell wall in MECHA's hydraulic network
  - graph-topology identifiers (node_id)
  - hydraulic properties of a cell wall in MECHA's hydraulic network

``HydraulicMembrane`` carries:
  - a single wall ↔ cell membrane connection (graph edge path='membrane')
  - geometry: length, dist (wall-to-cell half-thickness distance)
  - hydraulic properties: km (total membrane), kaqp (aquaporin), K_computed

``HydraulicPlasmodesmata`` carries:
  - a single cell ↔ cell symplastic connection (graph edge path='plasmodesmata')
  - geometry: length between the two cell centroids
  - hydraulic properties: kpl (conductance, cm³ hPa⁻¹ d⁻¹)

``HydraulicCell`` carries:
  - geometry fields mirrored from GRANAP Cell (without inheriting it)
  - graph-topology identifiers (node_id, cgroup, rank, wall_ids)
  - hydraulic configuration / state fields (kw, kpl, km, kaqp, os, psi, psi_p)

``HydraulicCellManager`` holds an ordered list of HydraulicCell objects and
exposes filtered views (xylem, sieve, passage, …) plus O(1) lookups.
It also holds all HydraulicMembrane and HydraulicPlasmodesmata objects.
It communicates with NetworkBuilder exclusively through ``sync_from_network``.

Units (same convention as the rest of MECHA):
  - lengths / diameters : µm
  - areas               : µm²
  - perimeters          : µm
  - conductivities      : cm hPa⁻¹ d⁻¹  (or cm³ hPa⁻¹ d⁻¹ for conductance)
  - potentials          : hPa
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
from shapely.geometry import Polygon

if TYPE_CHECKING:
    # Avoid circular import at runtime; only used for type hints.
    from mecha.utils.network_builder import NetworkBuilder

CGROUP_TO_TYPE = {
    1: "exodermis",
    2: "epidermis",
    3: "endodermis",
    4: "cortex",
    5: "stele",
    16: "pericycle",
    11: "phloem",
    12: "companion",
    13: "xylem",
    19: "xylem",
    20: "xylem",
    17: "transfusion parenchyma",
    18: "transfusion tracheid"
}

# ---------------------------------------------------------------------------
# HydraulicWall
# ---------------------------------------------------------------------------

class HydraulicWall:
    """Standalone descriptor for a single cell wall in MECHA's hydraulic network.

    Parameters
    ----------
    x, y : float
        Midpoint position (µm).
    length : float
        Wall length (µm).
    thickness : float
        Wall half-thickness (µm).
    node_id : int
        Graph node index (0 to n_walls - 1).
    is_border : bool
        Whether this wall is on the exterior border of the organ.
    is_aerenchyma : bool
        Whether this wall borders an aerenchyma / air space.
    """

    __slots__ = (
        "x", "y", "length", "thickness", "node_id", "is_border", "is_aerenchyma",
        "cells", "membranes",
        # Hydraulic properties
        "kw", "Q", "Q_in", "Q_out", "A", "velocity", "psi_os", "psi_total", "psi_p",
    )

    def __init__(
        self,
        *,
        x: float,
        y: float,
        length: float,
        thickness: float,
        node_id: int,
        is_border: bool = False,
        is_aerenchyma: bool = False,
    ) -> None:
        self.x: float = x
        self.y: float = y
        self.length: float = length
        self.thickness: float = thickness
        self.node_id: int = node_id
        self.is_border: bool = is_border
        self.is_aerenchyma: bool = is_aerenchyma

        # Link to connected cells (via membrane edges)
        self.cells: List['HydraulicCell'] = []

        # Link to HydraulicMembrane objects for this wall
        self.membranes: List['HydraulicMembrane'] = []

        # Hydraulic Cell-wall (apoplastic) conductivity [cm hPa⁻¹ d⁻¹]
        self.kw: Optional[float] = None
        self.Q: Optional[float] = None
        self.Q_in: Optional[float] = None
        self.Q_out: Optional[float] = None
        self.A: Optional[float] = None
        self.velocity: Optional[float] = None
        self.psi_os: Optional[float] = None
        self.psi_total: Optional[float] = None
        self.psi_p: Optional[float] = None

    def reset_hydraulics(self) -> None:
        """Reset all hydraulic fields to ``None``."""
        self.kw = None
        self.Q = None
        self.Q_in = None
        self.Q_out = None
        self.A = None
        self.velocity = None
        self.psi_os = None
        self.psi_total = None
        self.psi_p = None

    def get_effective_potential(self, sigma: float = 1.0) -> float:
        """Calculate effective potential P - sigma * os."""
        p = self.psi_p if self.psi_p is not None else 0.0
        os_val = self.psi_os if self.psi_os is not None else 0.0
        return p - sigma * os_val

    def __repr__(self) -> str:
        s = (
            f"HydraulicWall(node_id={self.node_id}, length={self.length:.1f}, "
            f"thickness={self.thickness:.1f}, is_border={self.is_border}, "
            f"is_aerenchyma={self.is_aerenchyma}")
        s += f", kw={self.kw:.1e}" if self.kw is not None else ", kw=None"
        s += f", Q={self.Q:.1e}" if self.Q is not None else ", Q=None"
        if self.Q is not None:
            s += f", velocity={self.velocity:.1e}"
        if self.psi_p is not None:
            s += f", psi_p={self.psi_p:.3f}"
        if self.psi_os is not None:
            s += f", psi_os={self.psi_os:.3f}"
        if self.psi_total is not None:
            s += f", psi_total={self.psi_total:.3f}"
        s += ")"
        return s


# ---------------------------------------------------------------------------
# HydraulicMembrane
# ---------------------------------------------------------------------------

class HydraulicMembrane:
    """Descriptor for a single wall ↔ cell membrane connection.

    Represents one graph edge with ``path='membrane'``, connecting a
    ``HydraulicWall`` node to a ``HydraulicCell`` node.

    Parameters
    ----------
    wall : HydraulicWall
        The wall-side of this membrane connection.
    cell : HydraulicCell
        The cell-side of this membrane connection.
    length : float
        Shared wall length for this connection (µm).
    dist : float
        Half-thickness distance from the wall midpoint to the cell centroid (µm).
    """

    __slots__ = (
        "wall", "cell",
        # Geometry
        "length", "dist",
        # Hydraulic properties — None until solver assigns them
        "km",    # Total membrane conductivity  [cm hPa⁻¹ d⁻¹]
        "kaqp",  # Aquaporin contribution        [cm hPa⁻¹ d⁻¹]
        "K_computed",  # Effective conductance computed by HydraulicMatrixBuilder [cm³ hPa⁻¹ d⁻¹]
        "sigma", # Reflection coefficient
        "Q", "A", "velocity",
        "diffusion_coeff", # Diffusion coefficient
    )

    def __init__(
        self,
        *,
        wall: 'HydraulicWall',
        cell: 'HydraulicCell',
        length: float,
        dist: float,
    ) -> None:
        self.wall: 'HydraulicWall' = wall
        self.cell: 'HydraulicCell' = cell
        self.length: float = length
        self.dist: float = dist

        # Hydraulic fields — None until explicitly assigned
        self.km: Optional[float] = None
        self.kaqp: Optional[float] = None
        self.K_computed: Optional[float] = None
        self.sigma: Optional[float] = None
        self.Q: Optional[float] = None
        self.A: Optional[float] = None
        self.velocity: Optional[float] = None
        self.diffusion_coeff: Optional[float] = None

    def reset_hydraulics(self) -> None:
        """Reset all hydraulic fields to ``None``."""
        self.km = self.kaqp = self.K_computed = self.sigma = self.Q = self.A = self.velocity = None

    def __repr__(self) -> str:
        s = (
            f"HydraulicMembrane(wall={self.wall.node_id}, "
            f"cell={self.cell.node_id}, length={self.length:.1f}, "
            f"dist={self.dist:.1f}"
        )
        s += f", km={self.km:.1f}" if self.km is not None else ", km=None"
        s += f", kaqp={self.kaqp:.1f}" if self.kaqp is not None else ", kaqp=None"
        s += f", K_computed={self.K_computed:.1f}" if self.K_computed is not None else ", K_computed=None"
        s += f", Q={self.Q:.1e}" if self.Q is not None else ", Q=None"
        if self.velocity is not None:
            s += f", velocity={self.velocity:.1e}"
        if self.sigma is not None:
            s += f", sigma={self.sigma:.3f}"
        s += ")"
        return s


# ---------------------------------------------------------------------------
# HydraulicPlasmodesmata
# ---------------------------------------------------------------------------

class HydraulicPlasmodesmata:
    """Descriptor for a single cell ↔ cell symplastic (plasmodesmata) connection.

    Represents one graph edge with ``path='plasmodesmata'``, connecting two
    ``HydraulicCell`` nodes through the symplastic pathway.

    Parameters
    ----------
    cell_i : HydraulicCell
        First cell node of this connection.
    cell_j : HydraulicCell
        Second cell node of this connection.
    length : float
        Shared wall / contact length between the two cells (µm).
    """

    __slots__ = (
        "cell_i", "cell_j",
        # Geometry
        "length",
        # Hydraulic properties — None until solver assigns them
        "kpl",        # Plasmodesmata conductance of a single unit [cm³ hPa⁻¹ d⁻¹]
        "K_computed", # Computed conductance of whole plasmodesmata [cm³ hPa⁻¹ d⁻¹]
        "fplxheight", # plasmodesmata height (default 8.0E5) [] UNITS ?
        "temp_factor",  # Tissue-specific frequency factor (dimensionless × cm)
        "aperture_coef", # interface specific for the aperture of the plasmodesmata
        "sigma",      # Reflection coefficient
        "Q", #Flow rate through plasmodesmata [cm³ d⁻¹]
        "A", "velocity",
        "n_pd", # number of plasmodesmata per interface
    )

    def __init__(
        self,
        *,
        cell_i: 'HydraulicCell',
        cell_j: 'HydraulicCell',
        length: float,
    ) -> None:
        self.cell_i: 'HydraulicCell' = cell_i
        self.cell_j: 'HydraulicCell' = cell_j
        self.length: float = length
        self.n_pd: int = 0

        # Hydraulic fields — None until explicitly assigned
        self.kpl: Optional[float] = 0.0
        self.fplxheight: Optional[float] = 0.0
        self.temp_factor: Optional[float] = 0.0
        self.aperture_coef: Optional[float] = 0.0
        self.sigma: Optional[float] = 0.0
        self.Q: Optional[float] = 0.0
        self.A: Optional[float] = 0.0
        self.velocity: Optional[float] = 0.0
        self.K_computed: Optional[float] = 0.0

    def reset_hydraulics(self) -> None:
        """Reset all hydraulic fields to 0.0."""
        self.kpl = self.fplxheight = self.temp_factor = self.sigma = self.Q = self.A = self.velocity = 0.0
        self.K_computed = self.aperture_coef = self.n_pd = 0.0

    def __repr__(self) -> str:
        s = (
            f"HydraulicPlasmodesmata(cell_i={self.cell_i.node_id}, "
            f"cell_j={self.cell_j.node_id}, length={self.length:.1f}"
        )
        s += f", kpl={self.kpl:.1e}" if self.kpl is not None else ", kpl=None"
        s += f", fplxheight={self.fplxheight:.1e}" if self.fplxheight is not None else ", fplxheight=None"
        s += f", temp_factor={self.temp_factor:.1e}" if self.temp_factor is not None else ", temp_factor=None"
        s += f", Q={self.Q:.1e}" if self.Q is not None else ", Q=None"
        if self.velocity is not None:
            s += f", velocity={self.velocity:.1e}"
        if self.sigma is not None:
            s += f", sigma={self.sigma:.3f}"
        s += ")"
        return s


# ---------------------------------------------------------------------------
# HydraulicCell
# ---------------------------------------------------------------------------

class HydraulicCell:
    """Standalone descriptor for a single cell in MECHA's hydraulic network.

    Geometry fields mirror ``granap.cell_class.Cell`` but this class has
    **no inheritance dependency** on GRANAP — it is constructed purely from
    data already present in the ``NetworkBuilder`` graph.

    Hydraulic fields (``kw``, ``kpl``, ``km``, ``kaqp``, ``psi_os``, ``psi``,
    ``psi_p``) are initialised to ``None`` and are meant to be populated later
    by ``Mecha._set_hydraulic_conductivities`` and post-solve routines.

    Parameters
    ----------
    x, y : float
        Centroid position (µm).
    area : float
        Cross-section area (µm²).
    perimeter : float
        Cell perimeter (µm).
    cell_type : str
        String tissue label, e.g. ``'cortex'``, ``'xylem'``, ``'passage'``.
    node_id : int
        Graph node index in the NetworkBuilder graph
        (equals ``n_wall_junction + cell_id``).
    cell_id : int
        0-based position in the cell array  (``node_id - n_wall_junction``).
    cgroup : int
        MECHA cell-group integer:
        1=exodermis, 2=epidermis, 3=endodermis, 4=cortex, 5=stele,
        11=phloem sieve, 12=companion, 13=xylem, 16=pericycle …
    rank : int, optional
        Layer rank from ``rank_cells`` / ``_rank_cells_from_graph``
        (40–49 = cortex layers, 50–61 = stele layers, etc.).  Default 0.
    walls : list of HydraulicWall, optional
        Wall objects connected to this cell via membrane edges.
    """

    __slots__ = (
        # --- geometry (mirrored from GRANAP Cell) ---
        "x", "y", "area", "perimeter", "cell_type",
        # --- spatial polygon (Shapely Polygon, µm units) ---
        "polygon",
        # --- topology ---
        "node_id", "cell_id", "cgroup", "rank", "walls", "plasmodesmata",
        # --- hydraulic configuration (apoplastic) ---
        "kw",
        # --- hydraulic configuration (symplastic / plasmodesmata) ---
        "kpl",
        # --- hydraulic configuration (membrane) ---
        "km", "kaqp",
        # --- hydraulic state (potentials) ---
        "psi_os", "psi_total", "psi_p",
        # --- flow balance ---
        "Q_in", "Q_out",
        # --- growth ---
        "elongation_rate",
    )

    def __init__(
        self,
        *,
        x: float,
        y: float,
        area: float,
        perimeter: float,
        cell_type: str,
        node_id: int,
        cell_id: int,
        cgroup: int,
        rank: int = 0,
        elongation_rate: Optional[float] = 0.0,
        walls: Optional[List['HydraulicWall']] = None,
        plasmodesmata: Optional[List['HydraulicPlasmodesmata']] = None,
        polygon: Optional[Polygon] = None,
    ) -> None:
        # Geometry
        self.x: float = x
        self.y: float = y
        self.area: float = area
        self.elongation_rate: float = elongation_rate if elongation_rate is not None else 0.0
        self.perimeter: float = perimeter
        self.cell_type: str = cell_type
        self.polygon: Optional[Polygon] = polygon

        # Topology
        self.node_id: int = node_id
        self.cell_id: int = cell_id
        self.cgroup: int = cgroup
        self.rank: int = rank
        self.walls: List['HydraulicWall'] = walls if walls is not None else []
        self.plasmodesmata: List['HydraulicPlasmodesmata'] = plasmodesmata if plasmodesmata is not None else []

        # Hydraulic fields — all None until explicitly assigned
        # -------------------------------------------------------
        # Cell-wall (apoplastic) conductivity  [cm hPa⁻¹ d⁻¹]
        self.kw: Optional[float] = None
        # Plasmodesmata (symplastic) conductance  [cm³ hPa⁻¹ d⁻¹]
        self.kpl: Optional[float] = None
        # Membrane total conductivity  [cm hPa⁻¹ d⁻¹]
        self.km: Optional[float] = None
        # Aquaporin contribution to membrane conductivity
        self.kaqp: Optional[float] = None
        # Osmotic potential  [hPa]
        self.psi_os: Optional[float] = None
        # Total water potential  [hPa]
        self.psi_total: Optional[float] = None
        # Pressure potential  [hPa]
        self.psi_p: Optional[float] = None

        self.Q_in: Optional[float] = None
        self.Q_out: Optional[float] = None

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_effective_potential(self, sigma: float = 1.0) -> float:
        """Calculate effective potential P - sigma * os."""
        p = self.psi_p if self.psi_p is not None else 0.0
        os_val = self.psi_os if self.psi_os is not None else 0.0
        return p - sigma * os_val

    def reset_hydraulics(self) -> None:
        """Reset all hydraulic fields to ``None``."""
        self.kw = self.kpl = self.km = self.kaqp = None
        self.psi_os = self.psi_total = self.psi_p = self.Q_in = self.Q_out = None

    def _polygon(self) -> Polygon:
        
        return Polygon()

    def __repr__(self) -> str:
        s = (f"HydraulicCell(cell_id={self.cell_id}, node_id={self.node_id}, "
             f"cgroup={self.cgroup}, rank={self.rank}, type='{self.cell_type}', "
             f"x={self.x:.1f}, y={self.y:.1f}, area={self.area:.1f}")
        if self.elongation_rate > 0:
            s += f", elongation_rate={self.elongation_rate:.1f}"
        if self.walls:
            s += f", walls={self.walls}"
        if self.plasmodesmata:
            s += f", plasmodesmata={self.plasmodesmata}"
        if self.psi_os is not None:
            s += f", psi_os={self.psi_os:.3f}"
        if self.psi_total is not None:
            s += f", psi_total={self.psi_total:.3f}"
        if self.psi_p is not None:
            s += f", psi_p={self.psi_p:.3f}"
        if self.Q_in is not None:
            s += f", Q_in={self.Q_in:.1e}"
        if self.Q_out is not None:
            s += f", Q_out={self.Q_out:.1e}"
        s += ")"
        return s


# ---------------------------------------------------------------------------
# HydraulicCellManager
# ---------------------------------------------------------------------------

class HydraulicCellManager:
    """Ordered collection of :class:`HydraulicCell` objects.

    The manager is always built **wholesale** from a ``NetworkBuilder`` via
    :meth:`sync_from_network` — there is no public method to append individual
    cells.  Lookups by ``node_id`` or ``cgroup`` are O(1) thanks to internal
    dictionaries that are rebuilt whenever the manager is synchronised.

    Also holds all :class:`HydraulicMembrane` and
    :class:`HydraulicPlasmodesmata` connection objects built from the network
    graph edges (``path='membrane'`` and ``path='plasmodesmata'``).

    Typical usage::

        manager = HydraulicCellManager()
        manager.sync_from_network(network)           # called by NetworkBuilder

        cortex = manager.get_by_cgroup(4)
        xylem  = manager.xylem
        cell   = manager.get_by_node_id(452)

        # Access connection objects
        all_membranes       = manager.membranes
        all_plasmodesmata   = manager.plasmodesmata
        mb = manager.get_membrane_by_edge(wall_id, cell_node_id)
        pd = manager.get_plasmodesmata_by_edge(cell_i_node_id, cell_j_node_id)
    """

    def __init__(self) -> None:
        self._cells: List[HydraulicCell] = []
        self._by_node_id: Dict[int, HydraulicCell] = {}
        self._by_cgroup: Dict[int, List[HydraulicCell]] = {}
        self._by_type: Dict[str, List[HydraulicCell]] = {}

        self._walls: List[HydraulicWall] = []
        self._wall_by_node_id: Dict[int, HydraulicWall] = {}

        # Membrane connections (wall ↔ cell, path='membrane')
        self._membranes: List[HydraulicMembrane] = []
        self._membrane_by_edge: Dict[Tuple[int, int], HydraulicMembrane] = {}

        # Plasmodesmata connections (cell ↔ cell, path='plasmodesmata')
        self._plasmodesmata: List[HydraulicPlasmodesmata] = []
        self._pd_by_edge: Dict[Tuple[int, int], HydraulicPlasmodesmata] = {}

    # ------------------------------------------------------------------
    # Reset hydraulic properties
    # ------------------------------------------------------------------

    def reset_hydraulic_properties(self) -> None:
        """Reset hydraulic properties for all walls, membranes, plasmodesmata and cells."""
        for wall in self._walls:
            wall.reset_hydraulics()
        for membrane in self._membranes:
            membrane.reset_hydraulics()
        for pd in self._plasmodesmata:
            pd.reset_hydraulics()
        for cell in self._cells:
            cell.reset_hydraulics()

    def set_osmotic_from_concentration(
        self,
        c_cells: np.ndarray,
        n_wall_junction: int,
        T: float,
        psi_os_baseline: np.ndarray = None,
    ) -> None:
        """Set cell osmotic potentials from a concentration array via van't Hoff (Ψ_os = −RT·c).

        Parameters
        ----------
        c_cells : ndarray, shape (n_cells,)
            Molar concentration per cell [mol cm⁻³], indexed by cell_id.
        n_wall_junction : int
            Kept for API consistency with SoluteTransport conventions; unused here.
        T : float
            Temperature [K].  R = 8.314e4 hPa cm³ mol⁻¹ K⁻¹ is used internally
            (derived from R = 8.314 J mol⁻¹ K⁻¹ and 1 J = 1e4 hPa cm³).
        psi_os_baseline : ndarray, shape (n_cells,), optional
            Background osmotic potential per cell [hPa] representing unknown /
            non-dynamic solutes.  When provided the dynamic van't Hoff term is
            *added* to this baseline instead of replacing the cell's psi_os
            entirely.  Indexed by cell_id; 0.0 is used for out-of-range entries.
        """
        RT = 8.314e4 * T
        for cell in self._cells:
            if cell.cell_id < len(c_cells):
                dynamic = -RT * float(c_cells[cell.cell_id])
                if psi_os_baseline is not None and cell.cell_id < len(psi_os_baseline):
                    cell.psi_os = float(psi_os_baseline[cell.cell_id]) + dynamic
                else:
                    cell.psi_os = dynamic

    # ------------------------------------------------------------------
    # Collection protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._cells)

    def __iter__(self):
        return iter(self._cells)

    def __getitem__(self, index):
        return self._cells[index]

    def __repr__(self) -> str:
        return (
            f"HydraulicCellManager("
            f"{len(self._cells)} cells, "
            f"{len(self._walls)} walls, "
            f"{len(self._membranes)} membranes, "
            f"{len(self._plasmodesmata)} plasmodesmata)"
        )

    # ------------------------------------------------------------------
    # Lookups — cells / walls
    # ------------------------------------------------------------------

    def get_by_node_id(self, node_id: int) -> Optional[HydraulicCell]:
        """Return the cell whose graph node index is *node_id*, or ``None``."""
        return self._by_node_id.get(node_id)

    def get_wall_by_node_id(self, node_id: int) -> Optional[HydraulicWall]:
        """Return the wall whose graph node index is *node_id*, or ``None``."""
        return self._wall_by_node_id.get(node_id)

    def get_by_cgroup(self, cgroup: int) -> List[HydraulicCell]:
        """Return all cells whose ``cgroup`` matches *cgroup*."""
        return self._by_cgroup.get(cgroup, [])

    def get_by_type(self, type_str: str) -> List[HydraulicCell]:
        """Return all cells whose ``cell_type`` matches *type_str*."""
        return self._by_type.get(type_str, [])

    # ------------------------------------------------------------------
    # Lookups — connection objects
    # ------------------------------------------------------------------

    def get_membrane_by_edge(self, wall_node_id: int, cell_node_id: int) -> Optional[HydraulicMembrane]:
        """Return the membrane connecting *wall_node_id* ↔ *cell_node_id*, or ``None``.

        The key is stored canonically as ``(min, max)`` so the order of
        arguments does not matter.
        """
        key = (min(wall_node_id, cell_node_id), max(wall_node_id, cell_node_id))
        return self._membrane_by_edge.get(key)

    def get_plasmodesmata_by_edge(self, node_i: int, node_j: int) -> Optional[HydraulicPlasmodesmata]:
        """Return the plasmodesmata connecting *node_i* ↔ *node_j*, or ``None``.

        The key is stored canonically as ``(min, max)`` so the order of
        arguments does not matter.
        """
        key = (min(node_i, node_j), max(node_i, node_j))
        return self._pd_by_edge.get(key)

    # ------------------------------------------------------------------
    # Filtered views (properties) — cells
    # ------------------------------------------------------------------

    @property
    def xylem(self, type=["protoxylem", "metaxylem", "xylem"]) -> List[HydraulicCell]:
        """Proto- and meta-xylem cells (cgroup 13, 19, 20)."""
        result = []
        for key in type:
            result.extend(self._by_type.get(key, []))

        return result

    @property
    def sieve(self, type=["sieve", "phloem"] ) -> List[HydraulicCell]:
        """Phloem sieve-tube cells (cgroup 11, 23)."""
        result = []
        for key in type:
            result.extend(self._by_type.get(key, []))
        return result

    # ! same as sieve !  
    @property
    def protosieve(self, type=["protosieve", "phloem"] ) -> List[HydraulicCell]:
        """Phloem sieve-tube cells (cgroup 11, 23)."""
        result = []
        for key in type:
            result.extend(self._by_type.get(key, []))
        return result

    @property
    def epidermis(self, type="epidermis") -> List[HydraulicCell]:
        """Epidermis cells (cgroup 2)."""
        return [c for c in self._cells if c.cell_type == type or c.cgroup == 2]

    @property
    def exodermis(self, type="exodermis") -> List[HydraulicCell]:
        """Exodermis cells (cgroup 1)."""
        return [c for c in self._cells if c.cell_type == type or c.cgroup == 1]

    @property
    def cortex(self, type=["cortex", "outercortex", "innercortex"]) -> List[HydraulicCell]:
        """Cortex cells (cgroup 4)."""
        result = []
        for key in type:
            result.extend(self._by_type.get(key, []))
        return result
    
    @property
    def mesophyll(self, type=["mesophyll", "spongy", "palisade"]) -> List[HydraulicCell]:
        """Mesophyll cells (cgroup 4)."""
        result = []
        for key in type:
            result.extend(self._by_type.get(key, []))
        return result

    @property
    def passage(self) -> List[HydraulicCell]:
        """Passage cells (identified by string type)."""
        return self._by_type.get("passage", [])

    @property
    def endodermis(self) -> List[HydraulicCell]:
        """Endodermis cells."""
        return [c for c in self._cells if c.cgroup == 3]

    @property
    def intercellular(self, type=["intercellular", "air space", "aerenchyma"]) -> List[HydraulicCell]:
        """Intercellular / air-space cells."""
        result = []
        for key in type:
            result.extend(self._by_type.get(key, []))
        return result
    
    @property
    def transfusion_tissue(self, type=["tracheid", "parenchyma"]) -> List[HydraulicCell]:
        """Transfusion tissue cells (cgroup 17, 18)."""
        result = []
        for key in type:
            result.extend(self._by_type.get(key, []))
        return result

    # ------------------------------------------------------------------
    # Filtered views (properties) — connection objects
    # ------------------------------------------------------------------

    @property
    def walls(self) -> List[HydraulicWall]:
        """All cell walls in the network."""
        return self._walls

    @property
    def membranes(self) -> List[HydraulicMembrane]:
        """All membrane (wall ↔ cell) connections in the network."""
        return self._membranes

    @property
    def plasmodesmata(self) -> List[HydraulicPlasmodesmata]:
        """All plasmodesmata (cell ↔ cell) connections in the network."""
        return self._plasmodesmata

    # ------------------------------------------------------------------
    # Communication bridge with NetworkBuilder
    # ------------------------------------------------------------------

    def sync_from_network(self, network: "NetworkBuilder") -> None:
        """Populate the manager from a fully-built ``NetworkBuilder``.

        Reads the graph topology and all MECHA-derived arrays (cell_areas,
        cell_perimeters, cell_ranks, cell_types) directly from *network*.
        Previous contents are discarded entirely.

        Also calls :meth:`sync_membranes_from_network` and
        :meth:`sync_plasmodesmata_from_network` to build the connection
        descriptor objects.

        Parameters
        ----------
        network : NetworkBuilder
            A populated ``NetworkBuilder`` instance.  Must have
            ``n_wall_junction``, ``n_cells``, ``graph``, ``cell_areas``,
            ``cell_perimeters``, ``cell_ranks``, and ``cell_types`` set.
        """
        self._cells = []
        self._by_node_id = {}
        self._by_cgroup = {}
        self._by_type = {}
        self._walls = []
        self._wall_by_node_id = {}
        self._membranes = []
        self._membrane_by_edge = {}
        self._plasmodesmata = []
        self._pd_by_edge = {}

        nwj = network.n_wall_junction
        position = {}
        for node, data in network.graph.nodes(data=True):
            pos = data.get("position")
            if pos is not None:
                position[node] = pos

        # Build polygon lookup from _cells_gdf (id_cell → Shapely Polygon)
        poly_dict: Dict[int, Optional[Polygon]] = {}
        if hasattr(network, '_cells_gdf') and network._cells_gdf is not None:
            gdf = network._cells_gdf
            for _, row in gdf.iterrows():
                if isinstance(row['geometry'], Polygon):
                    poly = row['geometry']
                    if poly.is_empty or not poly.is_valid:
                        continue
                    poly_dict[int(row['id_cell'])] = poly.buffer(0.0)
        else:
            # prep the geometry
            from mecha.utils.visu import prep_section
            gdf = prep_section(network.cellset_data) 
            for _, row in gdf.iterrows():
                if isinstance(row['geometry'], Polygon):
                    poly = row['geometry']
                    if poly.is_empty or not poly.is_valid:
                        continue
                    poly_dict[int(row['id_cell'])] = poly


        # Ensure network.wall_lengths is accessible via standard dict or array
        wall_lengths = getattr(network, "wall_lengths", {})
        if wall_lengths is None:
            wall_lengths = {}
            
        wall_thicknesses = getattr(network, "thickness", {})
        if wall_thicknesses is None:
            wall_thicknesses = {}

        border_walls = set(getattr(network, "border_walls", []))
        border_aerenchyma = set(getattr(network, "border_aerenchyma", []))

        # Pass 1: Build Walls
        for wall_id in range(network.n_walls):
            node_data = network.graph.nodes.get(wall_id, {})
            pos = position.get(wall_id, (0.0, 0.0))
            is_border = wall_id in border_walls
            is_aerenchyma = wall_id in border_aerenchyma
            
            if wall_id in wall_lengths:
                length = float(wall_lengths[wall_id])
            else:
                length = float(node_data.get("length", 0.0))
                
            if wall_id in wall_thicknesses:
                thickness = float(wall_thicknesses[wall_id])
            else:
                thickness = float(node_data.get("thickness", 1.0))

            wall = HydraulicWall(
                x=float(pos[0]),
                y=float(pos[1]),
                length=length,
                thickness=thickness,
                node_id=wall_id,
                is_border=is_border,
                is_aerenchyma=is_aerenchyma,
            )
            self._walls.append(wall)
            self._wall_by_node_id[wall_id] = wall

        # Pass 2: Build Cells and link to Walls
        for cell_id in range(network.n_cells):
            node_id = nwj + cell_id
            node_data = network.graph.nodes[node_id]

            # --- Geometry --------------------------------------------------
            pos = position.get(node_id, (0.0, 0.0))
            x, y = float(pos[0]), float(pos[1])

            area = (
                float(network.cell_areas[cell_id])
                if network.cell_areas is not None else 0.0
            )
            perimeter = (
                float(network.cell_perimeters[cell_id])
                if network.cell_perimeters is not None else 0.0
            )

            # --- Type / group ----------------------------------------------
            cgroup = int(node_data.get("cgroup", 0))
            if (hasattr(network, "cell_types") and network.cell_types is not None
                and cell_id < len(network.cell_types) and network.cell_types[cell_id]):
                cell_type = list(network.cell_types)[cell_id]
            elif (hasattr(network, "intercellular_cells") and network.intercellular_cells is not None
                  and cell_id in network.intercellular_cells):
                cell_type = "air space"
            elif (hasattr(network, "passage_cells") and network.passage_cells is not None
                  and cell_id in network.passage_cells):
                cell_type = "passage"
            else:
                cell_type = node_data.get("cell_type", "")
                if cell_type == "":
                    # use key 'cgroup'
                    cell_type = CGROUP_TO_TYPE[cgroup]
            # --- Rank ------------------------------------------------------
            rank = (
                int(network.cell_ranks[cell_id])
                if network.cell_ranks is not None else 0
            )

            # --- Wall IDs (membrane neighbours that are wall nodes) ---------
            linked_walls = []
            for nbr in network.graph.neighbors(node_id):
                if (network.graph.edges[node_id, nbr].get("path") == "membrane"
                    and nbr < network.n_walls):
                    wall_obj = self._wall_by_node_id.get(nbr)
                    if wall_obj:
                        linked_walls.append(wall_obj)

            # --- Build cell ------------------------------------------------
            cell = HydraulicCell(
                x=x, y=y,
                area=area,
                perimeter=perimeter,
                cell_type=cell_type,
                node_id=node_id,
                cell_id=cell_id,
                cgroup=cgroup,
                rank=rank,
                walls=linked_walls,
                polygon=poly_dict.get(cell_id),
            )

            # Link back from wall to cell
            for w in linked_walls:
                if cell not in w.cells:
                    w.cells.append(cell)

            # --- Register --------------------------------------------------
            self._cells.append(cell)
            self._by_node_id[node_id] = cell

            self._by_cgroup.setdefault(cgroup, []).append(cell)
            self._by_type.setdefault(cell_type, []).append(cell)

        # Pass 3: Build Membrane and Plasmodesmata connection objects
        self.sync_membranes_from_network(network)
        self.sync_plasmodesmata_from_network(network)

    # ------------------------------------------------------------------
    # Connection sync helpers
    # ------------------------------------------------------------------

    def sync_membranes_from_network(self, network: "NetworkBuilder") -> None:
        """Build :class:`HydraulicMembrane` objects from ``path='membrane'`` edges.

        Iterates all membrane edges in the network graph (wall node ↔ cell
        node) and creates one ``HydraulicMembrane`` per edge.  The resulting
        objects are stored in :attr:`_membranes` and indexed in
        :attr:`_membrane_by_edge`.  Back-references are also added to the
        corresponding ``HydraulicWall.membranes`` list.

        Parameters
        ----------
        network : NetworkBuilder
            Must have a populated ``graph`` and a valid ``n_walls`` attribute.
        """
        self._membranes = []
        self._membrane_by_edge = {}

        # Also clear existing membrane back-refs on walls
        for wall in self._walls:
            wall.membranes = []

        for u, v, eattr in network.graph.edges(data=True):
            if eattr.get("path") != "membrane":
                continue

            # Determine which endpoint is the wall, which is the cell
            if u < network.n_walls:
                wall_node_id, cell_node_id = u, v
            elif v < network.n_walls:
                wall_node_id, cell_node_id = v, u
            else:
                # Both endpoints are cells or junctions — not a membrane edge
                continue

            wall_obj = self._wall_by_node_id.get(wall_node_id)
            cell_obj = self._by_node_id.get(cell_node_id)
            if wall_obj is None or cell_obj is None:
                continue

            length = float(eattr.get("length", 0.0))
            dist   = float(eattr.get("dist", eattr.get("distnode_wall_cell", 0.0)))

            mb = HydraulicMembrane(
                wall=wall_obj,
                cell=cell_obj,
                length=length,
                dist=dist,
            )
            self._membranes.append(mb)

            key = (min(wall_node_id, cell_node_id), max(wall_node_id, cell_node_id))
            self._membrane_by_edge[key] = mb

            # Back-reference on the wall
            if mb not in wall_obj.membranes:
                wall_obj.membranes.append(mb)

    def sync_plasmodesmata_from_network(self, network: "NetworkBuilder") -> None:
        """Build :class:`HydraulicPlasmodesmata` objects from ``path='plasmodesmata'`` edges.

        Iterates all plasmodesmata edges in the network graph (cell ↔ cell)
        and creates one ``HydraulicPlasmodesmata`` per edge.  The resulting
        objects are stored in :attr:`_plasmodesmata` and indexed in
        :attr:`_pd_by_edge`.

        Parameters
        ----------
        network : NetworkBuilder
            Must have a populated ``graph`` and valid ``n_wall_junction``.
        """
        self._plasmodesmata = []
        self._pd_by_edge = {}

        # Clear existing plasmodesmata back-refs on cells
        for cell in self._cells:
            cell.plasmodesmata = []

        for u, v, eattr in network.graph.edges(data=True):
            if eattr.get("path") != "plasmodesmata":
                continue

            cell_i = self._by_node_id.get(u)
            cell_j = self._by_node_id.get(v)
            if cell_i is None or cell_j is None:
                continue

            length = float(eattr.get("length", 0.0))

            pd = HydraulicPlasmodesmata(
                cell_i=cell_i,
                cell_j=cell_j,
                length=length,
            )
            self._plasmodesmata.append(pd)

            key = (min(u, v), max(u, v))
            self._pd_by_edge[key] = pd

            # Back-reference on the cells
            if pd not in cell_i.plasmodesmata:
                cell_i.plasmodesmata.append(pd)
            if pd not in cell_j.plasmodesmata:
                cell_j.plasmodesmata.append(pd)
