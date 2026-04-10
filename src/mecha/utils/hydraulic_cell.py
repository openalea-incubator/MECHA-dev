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
HydraulicCell and HydraulicCellManager
=======================================

A standalone cell descriptor for MECHA's hydraulic network.

``HydraulicCell`` carries:
  - geometry fields mirrored from GRANAP Cell (without inheriting it)
  - graph-topology identifiers (node_id, cgroup, rank, wall_ids)
  - hydraulic configuration / state fields (kw, kpl, km, kaqp, os, psi, psi_p)

``HydraulicCellManager`` holds an ordered list of HydraulicCell objects and
exposes filtered views (xylem, sieve, passage, …) plus O(1) lookups.
It communicates with NetworkBuilder exclusively through ``sync_from_network``.

Units (same convention as the rest of MECHA):
  - lengths / diameters : µm
  - areas               : µm²
  - perimeters          : µm
  - conductivities      : cm hPa⁻¹ d⁻¹  (or cm³ hPa⁻¹ d⁻¹ for conductance)
  - potentials          : hPa
"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Avoid circular import at runtime; only used for type hints.
    from mecha.utils.network_builder import NetworkBuilder


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
    node_id : int
        Graph node index (0 to n_walls - 1).
    is_border : bool
        Whether this wall is on the exterior border of the organ.
    is_aerenchyma : bool
        Whether this wall borders an aerenchyma / air space.
    """

    __slots__ = (
        "x", "y", "length", "thickness", "node_id", "is_border", "is_aerenchyma",
        "cells",
        # Hydraulic properties
        "kw",
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

        # Link to connected cells
        self.cells: List['HydraulicCell'] = []

        # Hydraulic Cell-wall (apoplastic) conductivity [cm hPa⁻¹ d⁻¹]
        self.kw: Optional[float] = None

    def reset_hydraulics(self) -> None:
        """Reset all hydraulic fields to ``None``."""
        self.kw = None

    def __repr__(self) -> str:
        return f"HydraulicWall(node_id={self.node_id}, length={self.length:.1f})"

# ---------------------------------------------------------------------------
# HydraulicCell
# ---------------------------------------------------------------------------

class HydraulicCell:
    """Standalone descriptor for a single cell in MECHA's hydraulic network.

    Geometry fields mirror ``granap.cell_class.Cell`` but this class has
    **no inheritance dependency** on GRANAP — it is constructed purely from
    data already present in the ``NetworkBuilder`` graph.

    Hydraulic fields (``kw``, ``kpl``, ``km``, ``kaqp``, ``os``, ``psi``,
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
        # --- topology ---
        "node_id", "cell_id", "cgroup", "rank", "walls",
        # --- hydraulic configuration (apoplastic) ---
        "kw",
        # --- hydraulic configuration (symplastic / plasmodesmata) ---
        "kpl",
        # --- hydraulic configuration (membrane) ---
        "km", "kaqp",
        # --- hydraulic state (potentials) ---
        "os", "psi", "psi_p",
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
        walls: Optional[List['HydraulicWall']] = None,
    ) -> None:
        # Geometry
        self.x: float = x
        self.y: float = y
        self.area: float = area
        self.perimeter: float = perimeter
        self.cell_type: str = cell_type

        # Topology
        self.node_id: int = node_id
        self.cell_id: int = cell_id
        self.cgroup: int = cgroup
        self.rank: int = rank
        self.walls: List['HydraulicWall'] = walls if walls is not None else []

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
        self.os: Optional[float] = None
        # Total water potential  [hPa]
        self.psi: Optional[float] = None
        # Turgor / matric potential  [hPa]  (psi - os)
        self.psi_p: Optional[float] = None

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def reset_hydraulics(self) -> None:
        """Reset all hydraulic fields to ``None``."""
        self.kw = self.kpl = self.km = self.kaqp = None
        self.os = self.psi = self.psi_p = None

    def __repr__(self) -> str:
        return (
            f"HydraulicCell(cell_id={self.cell_id}, node_id={self.node_id}, "
            f"cgroup={self.cgroup}, rank={self.rank}, type='{self.cell_type}', "
            f"x={self.x:.1f}, y={self.y:.1f}, area={self.area:.1f})"
        )


# ---------------------------------------------------------------------------
# HydraulicCellManager
# ---------------------------------------------------------------------------

class HydraulicCellManager:
    """Ordered collection of :class:`HydraulicCell` objects.

    The manager is always built **wholesale** from a ``NetworkBuilder`` via
    :meth:`sync_from_network` — there is no public method to append individual
    cells.  Lookups by ``node_id`` or ``cgroup`` are O(1) thanks to internal
    dictionaries that are rebuilt whenever the manager is synchronised.

    Typical usage::

        manager = HydraulicCellManager()
        manager.sync_from_network(network)           # called by NetworkBuilder

        cortex = manager.get_by_cgroup(4)
        xylem  = manager.xylem
        cell   = manager.get_by_node_id(452)
    """

    def __init__(self) -> None:
        self._cells: List[HydraulicCell] = []
        self._by_node_id: Dict[int, HydraulicCell] = {}
        self._by_cgroup: Dict[int, List[HydraulicCell]] = {}
        self._by_type: Dict[str, List[HydraulicCell]] = {}

        self._walls: List[HydraulicWall] = []
        self._wall_by_node_id: Dict[int, HydraulicWall] = {}

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
        return f"HydraulicCellManager({len(self._cells)} cells)"

    # ------------------------------------------------------------------
    # Lookups
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
    # Filtered views (properties)
    # ------------------------------------------------------------------

    @property
    def xylem(self) -> List[HydraulicCell]:
        """Proto- and meta-xylem cells (cgroup 13, 19, 20)."""
        return [c for c in self._cells if c.cgroup in (13, 19, 20)]

    @property
    def sieve(self) -> List[HydraulicCell]:
        """Phloem sieve-tube cells (cgroup 11, 23)."""
        return [c for c in self._cells if c.cgroup in (11, 23)]

    @property
    def epidermis(self) -> List[HydraulicCell]:
        """Epidermis cells (cgroup 2)."""
        return [c for c in self._cells if c.cgroup == 2]

    @property
    def passage(self) -> List[HydraulicCell]:
        """Passage cells (identified by string type)."""
        return self._by_type.get("passage", [])

    @property
    def intercellular(self) -> List[HydraulicCell]:
        """Intercellular / air-space cells."""
        result = []
        for key in ("intercellular", "air space", "aerenchyma"):
            result.extend(self._by_type.get(key, []))
        return result

    @property
    def walls(self) -> List[HydraulicWall]:
        """All cell walls in the network."""
        return self._walls


    # ------------------------------------------------------------------
    # Communication bridge with NetworkBuilder
    # ------------------------------------------------------------------

    def sync_from_network(self, network: "NetworkBuilder") -> None:
        """Populate the manager from a fully-built ``NetworkBuilder``.

        Reads the graph topology and all MECHA-derived arrays (cell_areas,
        cell_perimeters, cell_ranks, cell_types) directly from *network*.
        Previous contents are discarded entirely.

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

        nwj = network.n_wall_junction
        position = {}
        for node, data in network.graph.nodes(data=True):
            pos = data.get("position")
            if pos is not None:
                position[node] = pos

        # Ensure network.wall_lengths is accessible via standard dict or array
        wall_lengths = getattr(network, "wall_lengths", {})
        if wall_lengths is None:
            wall_lengths = {}
            
        wall_thicknesses = getattr(network, "wall_thickness", {})
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
                thickness = float(node_data.get("wall_thickness", 1.0))

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
            cell_type = (
                network.cell_types[cell_id]
                if (hasattr(network, "cell_types") and network.cell_types is not None
                    and cell_id < len(network.cell_types))
                else node_data.get("cell_type", "")
            )

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
