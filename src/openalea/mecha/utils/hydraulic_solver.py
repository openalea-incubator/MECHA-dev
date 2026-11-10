from openalea.mecha.utils import hydraulic_cell
import numpy as np
import math
import networkx as nx
from scipy.sparse import coo_matrix

class HydraulicMatrixBuilder:
    """
    Builds the Doussan matrix (matrix_W) and solute transport matrix (matrix_C)
    for a specific hydraulic scenario and maturity stage.
    """

    def __init__(self, network, geometry, boundary, hydraulic, hormones, general, geo_props, position, indice):
        self.network = network
        self.geometry = geometry
        self.boundary = boundary
        self.hydraulic = hydraulic
        self.hormones = hormones
        self.general = general
        self.geo_props = geo_props
        self.position = position
        self.indice = indice
        self._pd_spatial_coef = {}
        self._wall_thk = None
        self._junction_thk = {}
        self._scl_pd = {}
        self._wall_air_cids = set()

    def _aqp_factor(self, cell):
        """Optional per-cell aquaporin multiplier (``HydraulicData.aqp_factor`` /
        ``aqp_factor_by_type``), or ``None`` (default: leave kaqp untouched)."""
        h = self.hydraulic
        if cell is None or (getattr(h, 'aqp_factor', None) is None
                            and getattr(h, 'aqp_factor_by_type', None) is None):
            return None
        return h.get_aqp_factor(cell.cell_id, cell.cell_type)

    def _wall_kw_factor(self, wall_id, cells):
        """Optional per-wall kw multiplier (``HydraulicData.kw_wall_factor`` /
        ``kw_wall_factor_fn``), or ``None`` (default: wall untouched)."""
        h = self.hydraulic
        if (getattr(h, 'kw_wall_factor', None) is None
                and getattr(h, 'kw_wall_factor_fn', None) is None):
            return None
        if wall_id >= self.network.n_walls:      # junction node: not a wall
            return None
        return h.get_wall_kw_factor(wall_id, cells)

    def _wall_sigma(self, wall_id, cells, kw_config, kw):
        """``(sigma, band)`` of a wall: the Casparian-band reflection coefficient
        (``HydraulicData.sigma_casp_endo`` / ``sigma_casp_exo``, possibly 0) and the
        band name ('endo' | 'exo') when the wall IS a band at this maturity, else
        ``(0.0, None)``.

        A band is a two-cell wall whose interface resolves, via ``interface_kw_key_map``,
        to ``kw_endo_endo`` / ``kw_exo_exo`` AND whose ``kw_config`` value is below the
        plain ``kw`` (i.e. the band exists at this barrier level; barrier 0 gives
        kw_endo_endo == kw_exo_exo == kw). Independent of the K branch taken in
        ``_fill_wall``: a passage cell or the suberin-lamella model changes the membrane /
        the wall kw but the Casparian band remains. Ghost / air-space walls are never
        endo-endo or exo-exo, junction nodes are not walls."""
        h = self.hydraulic
        se, sx = h.get_casp_sigma('endo'), h.get_casp_sigma('exo')   # validates
        if wall_id >= self.network.n_walls or len(cells) != 2:
            return 0.0, None
        if (cells[0].phi_thick, cells[1].phi_thick) == (1, 1) and cells[0].rank == cells[1].rank:
            return 0.0, None                    # phi-thickening wall (kw_phi_thick branch)
        key = h.interface_kw_key_map.get(tuple(sorted((cells[0].cgroup, cells[1].cgroup))))
        if key == 'kw_endo_endo' and kw_config.get(key, kw) < kw:
            return se, 'endo'
        if key == 'kw_exo_exo' and kw_config.get(key, kw) < kw:
            return sx, 'exo'
        return 0.0, None

    def _build_wall_thickness(self):
        """Per-wall thickness map for ``geometry.use_wall_thickness``.

        Reads the ``wall_thickness`` node attribute written by GRANAP (sum of the
        two flanking cells' ``cell_wall_thickness``, i.e. the *double* wall
        thickness MECHA's ``thickness`` stands for, µm). Walls without a positive
        value are left out and fall back to the global thickness. Junctions get
        the mean of their neighbouring walls. Returns ``None`` when disabled.
        """
        if not getattr(self.geometry, 'use_wall_thickness', False):
            self._junction_thk = {}
            return None
        graph = self.network.graph
        n_walls = self.network.n_walls
        wall_thk = {}
        for n in range(n_walls):
            t = graph.nodes[n].get('wall_thickness') if n in graph else None
            if t is not None and float(t) > 0.0:
                wall_thk[n] = float(t)
        self._junction_thk = {}
        for n in range(n_walls, self.network.n_wall_junction):
            vals = [wall_thk[m] for m in graph.neighbors(n) if m in wall_thk] if n in graph else []
            if vals:
                self._junction_thk[n] = sum(vals) / len(vals)
        return wall_thk

    def _local_thickness(self, default: float, *nodes: int) -> float:
        """Thickness (µm) for the wall/junction *nodes*, or *default* if unset."""
        if self._wall_thk is None:
            return default
        vals = [self._wall_thk[n] for n in nodes if n in self._wall_thk]
        if not vals:
            vals = [self._junction_thk[n] for n in nodes if n in self._junction_thk]
        return sum(vals) / len(vals) if vals else default

    def _build_scl_pd_factors(self, kpl_config: dict, barrier: int) -> dict:
        """Per-cell PD factor of dead / apoptotic sclerenchyma (barrier 11 only).

        A fraction ``scl_dead_fraction`` of the sclerenchyma cells (exact count,
        chosen with a ``scl_seed``-seeded RNG) are dead (factor 0); the others
        are apoptotic and keep ``scl_pd_factor`` of their PD. Returns ``{}`` when
        inactive (other stages, no sclerenchyma, or fraction 0 and factor 1).
        """
        if barrier != 11:
            return {}
        dead_fraction = float(kpl_config.get('scl_dead_fraction', 0.8))
        scl_factor = float(kpl_config.get('scl_pd_factor', 0.05))
        if dead_fraction <= 0.0 and scl_factor == 1.0:
            return {}
        scl_ids = sorted(c.cell_id for c in self.network.cell_manager.get_by_type('sclerenchyma'))
        if not scl_ids:
            return {}
        rng = np.random.default_rng(int(kpl_config.get('scl_seed', 0)))
        n_dead = int(round(min(dead_fraction, 1.0) * len(scl_ids)))
        dead = set(rng.permutation(len(scl_ids))[:n_dead].tolist())
        return {cid: (0.0 if k in dead else scl_factor) for k, cid in enumerate(scl_ids)}

    def build(self, h, i_maturity, hydraulic_conductivities, boundary,
              psi_xyl, psi_sieve, distributed_flow_xyl, distributed_flow_sieve):

        # Unpack properties
        maturity_stages = self.geometry.maturity_stages
        barrier = int(maturity_stages[i_maturity].get("barrier"))
        height = float(maturity_stages[i_maturity].get("height"))
        x_contact = float(self.hydraulic.get_xcontact(h))
        
        hyd_props = hydraulic_conductivities[h, i_maturity, barrier]
        kw_config = hyd_props['kw']
        kpl_config = hyd_props['kpl']
        kaqp_config = hyd_props['kaqp']
        kwa_config = hyd_props['kwa']
        k_air_config = hyd_props['k_air']
        a_cortex = hyd_props['a_cortex']
        b_cortex = hyd_props['b_cortex']

        kw = self.hydraulic.get_kw_value(h)
        kpl = kpl_config['kpl']
        self.hydraulic.set_pd_interface(self.network)
        self._pd_spatial_coef = self._pd_spatial_coefficients(h)

        n_nodes = self.network.graph.number_of_nodes()
        thickness = self.geometry.thickness
        self._wall_thk = self._build_wall_thickness()
        self._scl_pd = self._build_scl_pd_factors(kpl_config, barrier)
        # Needle mesophyll air spaces (protect_topology) are handled by wall_air / air_link
        # edges, not by the root aerenchyma (barrier > 0) rules below.
        self._wall_air_cids = {
            c.cell_id for c in self.network.cell_manager.intercellular
            if self.network.is_wall_air_cell(c.node_id)
        }

        # Initialize COO triplet lists for matrix_W
        self._rows_W = []
        self._cols_W = []
        self._data_W = []

        # Initialize COO triplet lists for matrix_C (only if needed)
        if self.boundary.c_flag:
            self._rows_C = []
            self._cols_C = []
            self._data_C = []

        rhs = np.zeros((n_nodes, 1))
        rhs_C = np.zeros((n_nodes, 1)) if self.boundary.c_flag else None
        rhs_s = np.zeros((n_nodes, 1))
        rhs_x = np.zeros((n_nodes, 1))
        rhs_p = np.zeros((n_nodes, 1))
        # Persistent constant source vector for the linearized wall-air (Kelvin)
        # coupling.  Held in its own vector so it survives the ``rhs[:] = ...``
        # overwrite in ``_apply_xylo_phloem_boundary`` and the per-scenario
        # ``rhs`` re-initialisation in ``Mecha.water_flux``.
        rhs_wa = np.zeros((n_nodes, 1))

        Kmb = np.zeros((self.network.n_membrane, 1))
        jmb = 0

        # 1. Edge loops (wall, membrane, plasmodesmata, wall_air, conductances)
        for node, edges in self.network.graph.adjacency():
            i = self.indice[node]
            for neighboor, eattr in edges.items():
                j = self.indice[neighboor]
                if j > i:  # Only process one way
                    path = eattr['path']
                    if path == 'wall':
                        self._fill_wall(i, j, node, eattr, kw, kw_config, height, thickness, barrier)
                    elif path == 'membrane':
                        K_mem = self._fill_membrane(i, j, node, neighboor, eattr, kw, kw_config, height, thickness, barrier,
                                                   kaqp_config, a_cortex, b_cortex)
                        Kmb[jmb] = K_mem
                        jmb += 1
                    elif path == 'plasmodesmata':
                        self._fill_plasmodesmata(i, j, eattr, kpl_config, thickness, barrier)
                    elif path == 'wall_air':
                        self._fill_wall_air(i, j, node, neighboor, eattr, kwa_config,
                                            height, thickness, barrier, rhs_wa,
                                            kw, kw_config, kaqp_config, a_cortex, b_cortex)
                    elif path == 'air_link':
                        self._fill_air_link(i, j, eattr, k_air_config, height)

        # 2. Add soil-wall connections
        self._apply_soil_boundary(x_contact, height, thickness, kw, barrier, boundary, rhs_s, rhs_C)

        # 3. Add xylem / phloem BC
        self._apply_xylo_phloem_boundary(i_maturity, barrier, psi_xyl, psi_sieve, distributed_flow_xyl, distributed_flow_sieve, boundary, rhs_s, rhs_x, rhs_p, rhs)

        # Build COO sparse matrices from accumulated triplets
        matrix_W = coo_matrix((self._data_W, (self._rows_W, self._cols_W)), shape=(n_nodes, n_nodes))
        if self.boundary.c_flag:
            matrix_C = coo_matrix((self._data_C, (self._rows_C, self._cols_C)), shape=(n_nodes, n_nodes))
        else:
            matrix_C = None

        # Unified solute matrix -> removed ApoC and SymC
        return matrix_W, matrix_C, rhs_C, rhs_p, rhs_x, rhs_s, rhs, rhs_wa, Kmb

    def _add_W(self, i, j, val):
        self._rows_W.append(i)
        self._cols_W.append(j)
        self._data_W.append(val)

    def _add_C(self, i, j, val):
        self._rows_C.append(i)
        self._cols_C.append(j)
        self._data_C.append(val)

    def _get_cells_for_wall_node(
        self, node_id: int
    ) -> list:
        """Return the :class:`HydraulicCell` objects flanking *node_id*.

        Abstracts the two possible states of a wall-path node:

        * **Mid-wall node** (``node_id < network.n_walls``): a
          :class:`HydraulicWall` object exists in :attr:`HydraulicCellManager`;
          ``wall.cells`` already holds the 0–2 flanking cells.
        * **Junction node** (``n_walls ≤ node_id < n_wall_junction``): no
          :class:`HydraulicWall` object exists; the flanking cells are found
          by iterating over the graph neighbours with ``path='membrane'`` and
          looking them up in :attr:`HydraulicCellManager`.

        Parameters
        ----------
        node_id : int
            Graph node index of the wall or junction node.

        Returns
        -------
        list of HydraulicCell
            The cells connected to *node_id* via ``path='membrane'`` edges.
            Typically 2 for an interior mid-wall node, 1 for a border node,
            3–4 for a junction node.
        """
        cm = self.network.cell_manager
        wall = cm.get_wall_by_node_id(node_id)
        if wall is not None:
            # Mid-wall node: use the pre-built HydraulicWall.cells list
            return list(wall.cells)
        # Junction node: query the graph for membrane-connected cell neighbours
        cells = []
        for nbr in self.network.graph.neighbors(node_id):
            edata = self.network.graph.edges[node_id, nbr]
            if edata.get('path') == 'membrane':
                cell = cm.get_by_node_id(nbr)
                if cell is not None:
                    cells.append(cell)
        return cells

    # ── Suberin-lamella model (hydraulic.suberin_model == 'lamella') ──────────
    def _lamella_on(self) -> bool:
        return getattr(self.hydraulic, 'suberin_model', 'none') == 'lamella'

    def _endo_face(self, cell, others):
        """Face of a non-passage endodermal *cell* toward *others*: ``'out'`` (touching
        cortex, resolved first), ``'in'`` (stele / pericycle, any other cgroup > 4),
        ``'rad'`` (toward endodermis) or ``None`` (passage cell / no match)."""
        if cell.cell_id in {c.cell_id for c in self.network.cell_manager.passage}:
            return None
        og = [o.cgroup for o in others]
        if 4 in og:
            return 'out'
        if any(g > 4 for g in og):
            return 'in'
        if 3 in og:
            return 'rad'
        return None

    def _side_layer_kw(self, cell, others, kw_config):
        """kw of *cell*'s secondary layer on the face toward *others*, or ``None``.

        Sclerenchyma: lignified secondary wall (``kw_scler``). Non-passage endodermis
        (``kw_lam_endo_*``): toward cortex -> out, toward stele/pericycle -> in, toward
        endodermis -> rad. Exodermis (``kw_lam_exo_*``): toward epidermis -> out, toward
        cortex/sclerenchyma -> in, toward exodermis -> rad. Any other cell: ``None``.
        """
        if cell.cell_type == 'sclerenchyma':
            return kw_config.get('kw_scler')
        cg = cell.cgroup
        og = [o.cgroup for o in others]
        if cg == 3 and 'kw_lam_endo_out' in kw_config:
            face = self._endo_face(cell, others)
            return None if face is None else kw_config['kw_lam_endo_' + face]
        elif cg == 1 and 'kw_lam_exo_out' in kw_config:
            if 2 in og:
                return kw_config['kw_lam_exo_out']
            if 4 in og:
                return kw_config['kw_lam_exo_in']
            if 1 in og:
                return kw_config['kw_lam_exo_rad']
        return None

    def _prim_thickness(self, t: float) -> float:
        """Primary-wall thickness [um] (one cell pair): the local wall thickness *t*
        (as for any ordinary wall) unless ``prim_wall_thickness`` overrides it."""
        p = getattr(self.hydraulic, 'prim_wall_thickness', None)
        return float(t) if p is None else float(p)

    def _sec_thickness(self, cell, t: float) -> float:
        """Secondary-layer thickness [um] (one side) of *cell*, ADDED to the primary wall;
        an unset ``sec_wall_thickness`` defaults to t/3 of the local wall thickness."""
        s = getattr(self.hydraulic, 'sec_wall_thickness', None)
        if s is None:
            return t / 3.0
        if isinstance(s, dict):
            return float(s.get(cell.cell_type, s.get('default', t / 3.0)))
        return float(s)

    def _tert_thickness(self, cell, others) -> float:
        """Tertiary-layer thickness [um] of the endodermal *cell* on its face toward *others*.

        U-shaped: only on the inner (``'in'``, toward pericycle / stele) and radial
        (``'rad'``, toward endodermis) faces, never on the outer face toward the cortex
        nor in passage cells. The layer lies behind the suberin lamella on the cell side;
        it adds membrane resistance (kw) (PD length unchanged) and is drawn in the
        wall geometry, but does not conduct along the wall. 0 = off."""
        if cell.cgroup != 3:
            return 0.0
        if self._endo_face(cell, others) not in ('in', 'rad'):
            return 0.0
        return float(getattr(self.hydraulic, 'tert_wall_thickness', 0.0) or 0.0)

    def _lamella_wall_kw(self, cells, kw_config, kw, t):
        """Effective kw and thicknesses of a two-cell wall. The conducting layers are
        secondary | primary (middle) | secondary, ADDED to the primary wall (p = local
        thickness *t*); the tertiary layers (U-shaped, see :meth:`_tert_thickness`) are
        excluded from the wall conductance altogether.

        Returns ``(kw_eff, t_c, t_geo)``: ``kw_eff = (sa ka + p kmid + sb kb) / t_c`` over
        the conducting thickness ``t_c = p + sa + sb`` (used for the wall K), and the
        geometric total ``t_geo = t_c + ta + tb`` (stored as ``t_wall``, drawing only).
        ``None`` if neither cell has a secondary layer."""
        a, b = cells
        ka = self._side_layer_kw(a, [b], kw_config)
        kb = self._side_layer_kw(b, [a], kw_config)
        if ka is None and kb is None:
            return None
        if a.cell_type == 'sclerenchyma' or b.cell_type == 'sclerenchyma':
            kmid = kw_config.get('kw_scler', kw)
        elif a.cgroup == b.cgroup == 3:
            kmid = kw_config.get('kw_endo_endo', kw)
        elif a.cgroup == b.cgroup == 1:
            kmid = kw_config.get('kw_exo_exo', kw)
        else:
            kmid = kw
        sa = self._sec_thickness(a, t) if ka is not None else 0.0
        sb = self._sec_thickness(b, t) if kb is not None else 0.0
        ta = self._tert_thickness(a, [b]) if ka is not None else 0.0
        tb = self._tert_thickness(b, [a]) if kb is not None else 0.0
        p = self._prim_thickness(t)
        t_c = p + sa + sb
        if t_c <= 0.0:
            return None
        num = ((sa * ka if ka is not None else 0.0) + p * kmid
               + (sb * kb if kb is not None else 0.0))
        return num / t_c, t_c, t_c + ta + tb

    # Walls are modified by these interface keys only (Casparian-type bands and cortex);
    # suberin (kw_endo_cortex / kw_endo_peri / kw_exo_epi / kw_exo_cortex / kw_passage)
    # acts on the membrane of the owning endodermal / exodermal cell, see
    # _membrane_series_kw. kw_meso_hypo is the needle hypodermis-mesophyll wall.
    _WALL_KW_KEYS = frozenset(('kw_endo_endo', 'kw_exo_exo', 'kw_cortex_cortex', 'kw_meso_hypo'))

    def _membrane_series_kw(self, cgroup, cells_i, kw_config, kw):
        """kw of the wall half in series with the membrane of a cell of tissue *cgroup*.

        Original MECHA semantics: suberin sits in the membrane path of the cell that
        owns it. *cells_i* are all the cells flanking the wall / junction node.
        Endodermis: wall to stele -> kw_endo_peri; endo-endo wall -> min(kw_endo_endo,
        kw_endo_peri); wall to cortex -> kw_endo_cortex (kw_passage with a passage cell,
        except on endo-endo walls). Exodermis: wall to epidermis -> kw_exo_epi; exo-exo
        -> min(kw_exo_exo, kw_exo_epi); otherwise (cortex, sclerenchyma) -> kw_exo_cortex.
        Junction nodes use the same neighbour counts as mid-wall nodes (as the original
        code did). Any other cell: the wall's own kw
        (kw_cortex_cortex band, needle kw_meso_hypo), else kw.
        """
        if cgroup == 3:
            passage_cids = {c.cell_id for c in self.network.cell_manager.passage}
            has_passage = any(c.cell_id in passage_cids for c in cells_i)
            if any(c.cgroup > 4 for c in cells_i):
                return kw_config.get('kw_passage', kw) if has_passage else kw_config.get('kw_endo_peri', kw)
            if sum(1 for c in cells_i if c.cgroup == 3) >= 2:
                return min(kw_config.get('kw_endo_endo', kw), kw_config.get('kw_endo_peri', kw))
            return kw_config.get('kw_passage', kw) if has_passage else kw_config.get('kw_endo_cortex', kw)
        if cgroup == 1:
            if any(c.cgroup == 2 for c in cells_i):
                return kw_config.get('kw_exo_epi', kw)
            if sum(1 for c in cells_i if c.cgroup == 1) >= 2:
                return min(kw_config.get('kw_exo_exo', kw), kw_config.get('kw_exo_epi', kw))
            return kw_config.get('kw_exo_cortex', kw)
        if len(cells_i) == 2:
            # other cells: the wall's own kw (cortex-cortex band, needle hypo-meso)
            kw_key = self.hydraulic.interface_kw_key_map.get(
                tuple(sorted((cells_i[0].cgroup, cells_i[1].cgroup))))
            if kw_key in self._WALL_KW_KEYS:
                return kw_config.get(kw_key, kw)
        return kw

    def _fill_wall(
        self,
        i: int,
        j: int,
        node: int,
        eattr: dict,
        kw: float,
        kw_config: dict,
        height: float,
        thickness: float,
        barrier: int,
    ) -> None:
        """Fill matrix entries for an apoplastic wall–wall (or wall–junction) edge.

        Uses the same interface-between-tissue approach as
        :meth:`_fill_plasmodesmata`:

        1. The flanking :class:`HydraulicCell` objects are retrieved via the
           new :meth:`_get_cells_for_wall_node` helper, which consults
           :attr:`HydraulicCellManager` and handles both mid-wall and junction
           nodes transparently — replacing the previous dependency on
           ``count_*`` graph attributes set by ``_count_surrounding_cells``.
        2. A canonical ``interface = tuple(sorted((cgroup_i, cgroup_j)))`` key
           is built from the two flanking cell cgroups.
        3. The interface is resolved against
           :attr:`HydraulicData.interface_kw_key_map` to select the correct
           ``kw_config`` key string, eliminating the previous long
           ``if/elif`` chain. Only the keys in :attr:`_WALL_KW_KEYS` (Casparian-type
           bands, cortex) modify the wall; suberin keys act on the membrane of the
           owning cell (:meth:`_membrane_series_kw`) and leave the wall at ``kw``.
        4. Named special-case guards (ghost walls, passage cells) are evaluated
           *before* the map lookup and short-circuit the generic path.
        5. ``wall.kw`` is stored on the :class:`HydraulicWall` object for
           post-solve inspection.

        Parameters
        ----------
        i, j : int
            Matrix indices of the two connected wall/junction nodes.
        node : int
            The graph node corresponding to index *i*.
        eattr : dict
            Edge attribute dict (mutated in-place with ``K``).
        kw : float
            Base cell-wall conductivity [cm hPa⁻¹ d⁻¹].
        kw_config : dict
            Wall conductivity configuration dict for the current barrier level,
            as returned by :meth:`HydraulicData.get_wall_conductivities`.
        height : float
            Cross-section height (µm).
        thickness : float
            Cell-wall half-thickness (µm).
        barrier : int
            Maturity barrier level (0 = no strip).
        """
        cm = self.network.cell_manager
        xylem_pieces: bool = getattr(self.geometry, 'xylem_pieces', False)
        thickness = self._local_thickness(thickness, i, j)

        # Shared geometric conductance factor
        def _geom(t_w: float) -> float:
            return (1.0E-04 * ((eattr['lateral_distance'] + height) * t_w - t_w ** 2)
                    / eattr['length'])
        temp: float = _geom(thickness)
        temp_factor: float = 1.0
        kw_val: float = kw

        # Get the flanking cells for this wall node via HydraulicCellManager
        cells_i = self._get_cells_for_wall_node(i)

        # --- Classify flanking cells ----------------------------------------
        intercellular_cids = {c.cell_id for c in cm.intercellular}
        xylem_cgroups = {13, 19, 20}

        n_interc = sum(1 for c in cells_i if c.cell_id in intercellular_cids)
        n_xyl = sum(1 for c in cells_i if c.cgroup in xylem_cgroups)

        # Suberin-lamella model (default off): effective kw of a two-cell wall carrying
        # at least one secondary layer; used below after the ghost / passage guards.
        # The secondary layers are added to the primary wall, so the conducting wall is
        # thicker than ``thickness`` (t_cond, used in the geometric factor); the tertiary
        # layers do not conduct and only enter the geometric total t_wall (drawing).
        lamella_kw = None
        t_cond = t_wall = thickness
        if self._lamella_on() and len(cells_i) == 2:
            res = self._lamella_wall_kw(cells_i, kw_config, kw, thickness)
            if res is not None:
                lamella_kw, t_cond, t_wall = res

        # ── Guard 1: Ghost wall (aerenchyma septa or xylem-piece septa) ──────
        # Both flanking cells are intercellular (aerenchyma) or both are xylem
        # when xylem_pieces is active.  These walls carry near-zero conductance
        # and trigger the ghost-junction book-keeping.
        if (n_interc >= 2 and barrier > 0) or (n_xyl >= 2 and xylem_pieces):
            K: float = 1.0E-16
            temp_factor = 1.0E-16
            # Ghost-junction book-keeping (unchanged from original logic)
            if j not in self.network.list_ghostjunctions:
                fakeJ = True
                for ind in range(int(self.network.n_junction_to_wall[j])):
                    if (
                        self.network.junction_to_wall[j][ind]
                        not in self.network.list_ghostwalls
                    ):
                        fakeJ = False
                if fakeJ:
                    self.network.list_ghostjunctions.append(j)
                    self.network.n_ghost_junction2wall += (
                        int(self.network.n_junction_to_wall[j]) + 2
                    )
                elif n_interc >= 2:  # septa wall between aerenchyma cells
                    K = kw_config.get('kw_septa', kw) * temp
                    if kw > 0:
                        temp_factor = kw_config.get('kw_septa', kw) / kw

        # ── Suberin-lamella model: secondary layer | middle layer | secondary layer
        # (a passage-cell wall no longer gets kw_passage: suberin is on the membrane)
        elif lamella_kw is not None:
            kw_val = lamella_kw
            temp = _geom(t_cond)
            K = kw_val * temp
            if kw > 0:
                temp_factor = kw_val / kw

        # ── Guard 3: Lignified sclerenchyma wall ──────────────────────────────
        # Any wall of a sclerenchyma cell (scl–scl, scl–exo, scl–cortex, and
        # junctions touching one) → kw_scler, for barrier stages defining it.
        elif 'kw_scler' in kw_config and any(c.cell_type == 'sclerenchyma' for c in cells_i):
            kw_val = kw_config['kw_scler']
            K = kw_val * temp
            if kw > 0:
                temp_factor = kw_val / kw

        # ── Interface-based lookup (mirrors _fill_plasmodesmata) ──────────────
        # Two known living cells → build canonical interface tuple, look up map.
        elif len(cells_i) == 2:
            interface = tuple(sorted((cells_i[0].cgroup, cells_i[1].cgroup)))

            # Check for phi_thick walls
            if (tuple(sorted((cells_i[0].phi_thick, cells_i[1].phi_thick))) == (1,1) and cells_i[0].rank == cells_i[1].rank):
                kw_val = kw_config.get('kw_phi_thick', kw)
                K = kw_val * temp
                if kw > 0:
                    temp_factor = kw_val / kw
            else: 
                kw_key = self.hydraulic.interface_kw_key_map.get(interface)
                if kw_key in self._WALL_KW_KEYS:
                    kw_val = kw_config.get(kw_key, kw)
                    K = kw_val * temp
                    if kw > 0:
                        temp_factor = kw_val / kw
                else:
                    # Interface not in map or a suberin key (acts on the membrane)
                    # → plain kw (e.g. endo–cortex, endo–stele, stele–stele)
                    K = kw * temp

        # ── Fallback: border/exterior wall or junction with >2 cells ─────────
        else:
            K = kw * temp

        # ── Air-space / living-cell wall: hydrated cell side + desiccated air side
        # Each half is half the wall thickness, in parallel →
        # k_eff = 0.5 * (kw_val + kw_septa), kw_septa being the desiccated-wall value.
        # Needle mesophyll (wall_air) air spaces are excluded: they keep the plain wall.
        n_interc_root = sum(
            1 for c in cells_i
            if c.cell_id in intercellular_cids and c.cell_id not in self._wall_air_cids
        )
        if n_interc_root == 1 and barrier > 0:
            kw_septa = kw_config.get('kw_septa', kw)
            self.hydraulic.check_kw_septa(kw, kw_septa)
            kw_val = 0.5 * (kw_val + kw_septa)
            K = kw_val * temp
            if kw > 0:
                temp_factor = kw_val / kw

        # Wall thickness actually used (lamella: primary + added layers), for plotting
        if self._lamella_on():
            eattr['t_wall'] = float(t_wall if lamella_kw is not None else thickness)
        else:
            eattr.pop('t_wall', None)

        # Optional per-wall kw multiplier (default None: arithmetic untouched). Applied
        # to K and temp_factor so the diffusive conductance follows.
        wf = self._wall_kw_factor(node, cells_i)
        if wf is not None:
            K = K * wf
            temp_factor = temp_factor * wf

        # Store K on the HydraulicWall object for post-solve inspection
        wall = cm.get_wall_by_node_id(i)
        if wall is not None:
            wall.kw = float(K)

        # Optional Casparian-band reflection coefficient (default 0: no 'sigma_wall' /
        # 'casp' edge attribute). Consumed by Mecha.initialize_scenarios (osmotic term
        # K*sigma*d_pi), Mecha.compute_edge_flows and SoluteTransport (advection
        # factor 1 - sigma). NB: not named 'sigma' because compute_edge_flows
        # overwrites eattr['sigma'] on every edge (membrane / PD reflection).
        sw, band = self._wall_sigma(node, cells_i, kw_config, kw)
        # 'casp' tags every band wall edge (also with sigma 0, so a solute-specific
        # SoluteTransport override can address it); 'sigma_wall' only when sigma != 0.
        if band is not None:
            eattr['casp'] = band
        else:
            eattr.pop('casp', None)
        if sw != 0.0:
            eattr['sigma_wall'] = float(sw)
        else:
            eattr.pop('sigma_wall', None)
        if wall is not None:
            wall.sigma = float(sw)

        self._add_W(i, i, -K)
        self._add_W(i, j,  K)
        self._add_W(j, i,  K)
        self._add_W(j, j, -K)

        # Solute flux (unchanged)
        if self.boundary.c_flag and self.general.c_flag:
            DF = temp * temp_factor * self.hormones.diff1_pw1
            if i not in self.network.apo_wall_zombies0:
                self._add_C(i, i, -DF)
                self._add_C(i, j,  DF)
            if j not in self.network.apo_j_zombies0:
                self._add_C(j, j, -DF)
                self._add_C(j, i,  DF)

        # Store K on graph edge for post-solve flow computation
        eattr['K'] = float(K)

    def _fill_membrane(
        self, i: int, j: int, node: int, neighboor: int, eattr: dict,
        kw: float, kw_config: dict, height: float, thickness: float,
        barrier: int, kaqp_config: dict, a_cortex: float, b_cortex: float
    ) -> float:
        """Fill matrix entries for a wall–cell membrane connection.

        Uses the same interface-between-tissue approach as ``_fill_wall``:
        retrieves flanking cells via ``_get_cells_for_wall_node`` to determine
        the membrane conductivity key from ``interface_kw_key_map`` (with passage
        overrides) and assigns properties directly to the ``HydraulicMembrane``
        connection object.
        """
        cm = self.network.cell_manager
        cgroup = self.network.graph.nodes[neighboor]['cgroup']
        thickness = self._local_thickness(thickness, i)
        n_wall_junction = self.network.n_wall_junction
        intercellular_ids = np.array([c.cell_id for c in cm.intercellular])

        # ── Solute flux logic (unchanged) ─────────────────────────────────────
        if self.boundary.c_flag and self.general.c_flag:
            for carrier in getattr(self.hormones, 'carrier_elems', []):
                if int(carrier.get("tissue")) == cgroup:
                    cid = j - n_wall_junction
                    if cid not in intercellular_ids and not (barrier > 0 and cgroup in [13, 19, 20]):
                        temp_c = float(carrier.get("constant")) * (height + eattr['dist']) * eattr['length']
                        direction = int(carrier.get("direction"))
                        if direction == 1:
                            if cid not in self.hormones.sym_zombie0:
                                self._add_C(j, i,  temp_c)
                            if i not in self.network.apo_wall_zombies0:
                                self._add_C(i, i, -temp_c)
                        elif direction == -1:
                            if cid not in self.hormones.sym_zombie0:
                                self._add_C(j, j, -temp_c)
                            if i not in self.network.apo_wall_zombies0:
                                self._add_C(i, j,  temp_c)

        # ── Aquaporin contribution kaqp_curr (unchanged logic) ───────────────
        kaqp_curr = 0.0
        if cgroup == 1:
            kaqp_curr = kaqp_config['kaqp_exo']
        elif cgroup == 2:
            kaqp_curr = kaqp_config['kaqp_epi']
        elif cgroup == 3:
            kaqp_curr = kaqp_config['kaqp_endo']
        elif cgroup in [13, 19, 20] or (
            cgroup == 18 and getattr(self.hydraulic, 'tracheid_as_xylem', False)
        ):
            # Transfusion tracheids join the xylem branch only on request: a
            # water-filled intercellular space modelled as an apoplastic conduit.
            # They are not in xylem_cells, so they get no xylem boundary condition.
            if barrier > 0:
                kaqp_curr = kaqp_config['kaqp_stele'] * 10000
                if self.boundary.c_flag and self.general.c_flag:
                    temp_c = 1.0E-04 * (self.network.wall_lengths[i] * height) / thickness
                    if i not in self.network.apo_wall_zombies0:
                        self._add_C(i, i, -temp_c * self.hormones.diff1_pw1)
                        self._add_C(i, j,  temp_c * self.hormones.diff1_pw1)
                    if (j - n_wall_junction) not in self.hormones.sym_zombie0:
                        self._add_C(j, j, -temp_c * self.hormones.diff1_pw1)
                        self._add_C(j, i,  temp_c * self.hormones.diff1_pw1)
            else:
                kaqp_curr = kaqp_config['kaqp_stele']
        elif cgroup > 4:
            kaqp_curr = kaqp_config['kaqp_stele']
        elif (j - n_wall_junction in intercellular_ids) and barrier > 0:
            kaqp_curr = getattr(self.geometry, 'k_interc', 0.0)
        elif cgroup == 4:
            kaqp_curr = float(a_cortex * self.network.distance_center_grav[i][0] * 1.0E-04 + b_cortex)
            if kaqp_curr < 0:
                print('Error, negative kaqp in cortical cell, adjust Paqp_cortex')

        # Optional per-cell aquaporin multiplier (default None: untouched; kmb unaffected)
        aqp_f = self._aqp_factor(cm.get_by_node_id(neighboor))
        if aqp_f is not None:
            kaqp_curr = kaqp_curr * aqp_f
            eattr['aqp_factor'] = float(aqp_f)
        else:
            eattr.pop('aqp_factor', None)

        # ── Flanking cells information ────────────────────────────────────────
        cells_i = self._get_cells_for_wall_node(i)

        n_endo = sum(1 for c in cells_i if c.cgroup == 3)
        n_exo = sum(1 for c in cells_i if c.cgroup == 1)
        n_stele = sum(1 for c in cells_i if c.cgroup > 4)

        # ── Series kw: suberin of the cell owning the membrane ───────────────
        kw_val = self._membrane_series_kw(cgroup, cells_i, kw_config, kw)

        # Suberin-lamella model: the wall term is the plain wall (local thickness); a
        # suberized cell (endodermis / exodermis, not sclerenchyma) crosses its primary
        # wall half (p/2) and adds its secondary (+ tertiary on the inner / radial faces of the endodermis) layer in series.
        p_half = thickness / 2
        s_lam, t_lam, kw_lam = 0.0, 0.0, None
        if self._lamella_on():
            kw_val = kw
            cell_c = cm.get_by_node_id(neighboor)
            if cell_c is not None and cell_c.cell_type != 'sclerenchyma':
                others = [c for c in cells_i if c.cell_id != cell_c.cell_id]
                kw_lam = self._side_layer_kw(cell_c, others, kw_config)
                if kw_lam is not None:
                    p_half = self._prim_thickness(thickness) / 2
                    s_lam = self._sec_thickness(cell_c, thickness)
                    t_lam = self._tert_thickness(cell_c, others)

        # Lamella (suberin) layer of this membrane, read by the membrane-PD solute mode
        if kw_lam is not None and s_lam > 0.0:
            eattr['kw_lam'] = float(kw_lam)
            eattr['s_lam'] = float(s_lam)
        else:
            eattr.pop('kw_lam', None)
            eattr.pop('s_lam', None)

        # ── Membrane conductance calculation ──────────────────────────────────
        is_barrier_interface = (n_endo >= 2) or (n_exo >= 2) or (n_stele > 0 and n_endo > 0)

        # Gas-filled intercellular space (barrier > 0, ``k_interc`` == 0): the cell
        # carries no water, so the membrane is cut regardless of the background
        # ``kmb`` (original MECHA: ``kaqp == 0`` -> K = 1e-16). Otherwise cut only
        # when there is genuinely no membrane conductance (``kmb == 0`` *and*
        # ``kaqp_curr == 0``); else the normal series formula (which includes
        # ``kmb``) keeps the membrane permeable (mesophyll cells).
        # Needle mesophyll (wall_air) air spaces are excluded: they keep the kmb path.
        is_air_cell = (
            barrier > 0 and (j - n_wall_junction) in intercellular_ids
            and (j - n_wall_junction) not in self._wall_air_cids
        )
        if kaqp_curr == 0.0 and (
            is_air_cell or ((not is_barrier_interface) and self.hydraulic.kmb == 0.0)
        ):
            K = 1.00E-16
        else:
            if kw_val == 0.0 or kw_lam == 0.0:
                K = 0.0
            elif kw_lam is not None:
                # primary half + secondary lamella + tertiary (kw) + membrane in series;
                # effective wall+layers kw (thickness-weighted harmonic mean) -> mb.km
                K = 1 / ((p_half * 1.0E-04) / kw_val + (s_lam * 1.0E-04) / kw_lam
                         + (t_lam * 1.0E-04) / kw_val
                         + 1 / (self.hydraulic.kmb + kaqp_curr)) * 1.0E-08 * (height + eattr['dist']) * eattr['length']
                kw_val = (p_half + s_lam + t_lam) / (p_half / kw_val + s_lam / kw_lam + t_lam / kw_val)
            else:
                K = 1 / (1 / (kw_val / (p_half * 1.0E-04)) + 1 / (self.hydraulic.kmb + kaqp_curr)) * 1.0E-08 * (height + eattr['dist']) * eattr['length']

        self._add_W(i, i, -K)
        self._add_W(i, j,  K)
        self._add_W(j, i,  K)
        self._add_W(j, j, -K)

        # ── Record on HydraulicMembrane object ────────────────────────────────
        mb = cm.get_membrane_by_edge(i, j)
        if mb is not None:
            mb.km = float(kw_val)
            mb.kaqp = float(kaqp_curr)
            mb.K_computed = float(K)

        eattr['K'] = float(K)
        return K

    def _membrane_conductance(
        self, wall_idx: int, cell_node: int, membrane_length: float,
        membrane_dist: float, kw: float, kw_config: dict, height: float,
        thickness: float, barrier: int, kaqp_config: dict,
        a_cortex: float, b_cortex: float,
    ) -> float:
        r"""Compute the transmembrane conductance ``K_memb`` for one wall↔cell pair.

        It is reused where a membrane resistance is in series with 
        another transport step — a wall ↔ air-space
        interface in :meth:`_fill_wall_air`.

        The pure hydraulic conductance is computed here without the
        solute-transport (C-matrix) side effects of :meth:`_fill_membrane`.

        Parameters
        ----------
        wall_idx : int
            Matrix index of the wall node.
        cell_node : int
            Graph node id of the flanking cell whose membrane is crossed.
        membrane_length : float
            Shared wall length of the membrane (µm).
        membrane_dist : float
            Wall-midpoint → cell-centroid distance (µm).
        kw, kw_config, height, thickness, barrier, kaqp_config, a_cortex, b_cortex
            Same meaning as in :meth:`_fill_membrane`.

        Returns
        -------
        float
            Membrane conductance ``K_memb`` [cm³ hPa⁻¹ d⁻¹].
        """
        cm = self.network.cell_manager
        cgroup = self.network.graph.nodes[cell_node]['cgroup']
        n_wall_junction = self.network.n_wall_junction
        intercellular_ids = np.array([c.cell_id for c in cm.intercellular])

        # ── Aquaporin contribution kaqp_curr (mirrors _fill_membrane) ─────────
        kaqp_curr = 0.0
        if cgroup == 1:
            kaqp_curr = kaqp_config['kaqp_exo']
        elif cgroup == 2:
            kaqp_curr = kaqp_config['kaqp_epi']
        elif cgroup == 3:
            kaqp_curr = kaqp_config['kaqp_endo']
        elif cgroup in [13, 19, 20]:
            kaqp_curr = kaqp_config['kaqp_stele'] * 10000 if barrier > 0 else kaqp_config['kaqp_stele']
        elif cgroup > 4:
            kaqp_curr = kaqp_config['kaqp_stele']
        elif (cell_node - n_wall_junction in intercellular_ids) and barrier > 0:
            kaqp_curr = getattr(self.geometry, 'k_interc', 0.0)
        elif cgroup == 4:
            kaqp_curr = float(a_cortex * self.network.distance_center_grav[wall_idx][0] * 1.0E-04 + b_cortex)
            if kaqp_curr < 0:
                print('Error, negative kaqp in cortical cell, adjust Paqp_cortex')

        aqp_f = self._aqp_factor(cm.get_by_node_id(cell_node))
        if aqp_f is not None:
            kaqp_curr = kaqp_curr * aqp_f

        # ── Series kw of the owning cell (mirrors _fill_membrane) ────────────
        cells_i = self._get_cells_for_wall_node(wall_idx)
        n_endo = sum(1 for c in cells_i if c.cgroup == 3)
        n_exo = sum(1 for c in cells_i if c.cgroup == 1)
        n_stele = sum(1 for c in cells_i if c.cgroup > 4)

        # Suberin of the owning cell in series (same rule as _fill_membrane)
        kw_val = self._membrane_series_kw(cgroup, cells_i, kw_config, kw)

        # ── Membrane conductance calculation (identical to _fill_membrane) ────
        # Cut only when there is no background membrane conductance either 
        # (``kmb == 0``), so needle mesophyll membranes remain permeable through ``kmb``.
        is_barrier_interface = (n_endo >= 2) or (n_exo >= 2) or (n_stele > 0 and n_endo > 0)
        # A gas-filled intercellular cell is cut whatever ``kmb`` is (as in _fill_membrane).
        is_air_cell = (
            barrier > 0 and (cell_node - n_wall_junction) in intercellular_ids
            and (cell_node - n_wall_junction) not in self._wall_air_cids
        )
        if kaqp_curr == 0.0 and (
            is_air_cell or ((not is_barrier_interface) and self.hydraulic.kmb == 0.0)
        ):
            K = 1.00E-16
        elif kw_val == 0.0:
            K = 0.0
        else:
            K = 1 / (1 / (kw_val / (thickness / 2 * 1.0E-04)) + 1 / (self.hydraulic.kmb + kaqp_curr)) \
                * 1.0E-08 * (height + membrane_dist) * membrane_length
        return float(K)

    @staticmethod
    def _get_kpl_factor(kpl_config: dict, factor_key) -> float:
        """Resolve an interface factor key against *kpl_config*.

        Parameters
        ----------
        kpl_config : dict
            Plasmodesmata conductance configuration dictionary returned by
            ``HydraulicData.get_plasmodesmatal_conductance()``.
        factor_key : str | Tuple[str, str] | None
            Value stored in ``HydraulicData.interface_kpl_factor_map``:

            * ``None``           – interface not listed → factor = 1.0.
            * ``str``            – single key look-up in *kpl_config*.
            * ``(str, str)``     – harmonic mean of two *kpl_config* values,
                                   reflecting the series resistance of two
                                   different tissue-side membranes.

        Returns
        -------
        float
            The resolved dimensionless conductance factor.
        """
        if factor_key is None:
            return 1.0
        if isinstance(factor_key, tuple):
            f1 = float(kpl_config.get(factor_key[0], 1.0))
            f2 = float(kpl_config.get(factor_key[1], 1.0))
            denom = f1 + f2
            return 2.0 * f1 * f2 / denom if denom > 0.0 else 0.0
        return float(kpl_config.get(factor_key, 1.0))

    def _fill_plasmodesmata(
        self, i: int, j: int, eattr: dict,
        kpl_config: dict, thickness: float, barrier: int,
    ) -> None:
        """Fill matrix entries for a plasmodesmata connection.

        The per-interface conductance factor is resolved entirely through
        ``HydraulicData.interface_kpl_factor_map`` via the ``_get_kpl_factor``
        helper, eliminating the previous long if/elif chain over tissue-type
        combinations.  This also fixes the bug where ``endo_in_factor`` and
        ``endo_out_factor`` were loaded from XML but never actually consulted.

        Parameters
        ----------
        i, j : int
            Matrix indices for the two connected nodes.
        eattr : dict
            Edge attribute dict (mutated in-place with ``K``, ``fplxheight``,
            and ``temp_factor``).
        kpl_config : dict
            Plasmodesmata conductance config from
            ``HydraulicData.get_plasmodesmatal_conductance()``.
        thickness : float
            Cross-section thickness (µm).
        barrier : int
            Maturity barrier level (0 = no strip).

        Lamella model: PD length is NOT changed by the secondary / tertiary layers
        (length factor = 1). Clarkson, Robards & Sanderson 1971 (Planta 96:292,
        doi 10.1007/BF00386944): endodermal PDs sit in pits where the tertiary wall is
        only 0.05 um thick; measured PD length 0.5 um, shorter than the model primary
        wall (1.5 um). ``kpl`` is a per-PD conductance not scaled by wall thickness.
        Any suberin effect on PDs belongs in the aperture factors, not in length.
        """
        cm = self.network.cell_manager
        pd = cm.get_plasmodesmata_by_edge(i, j)
        kpl = float(kpl_config.get('kpl', 5.3E-12))
        pd.kpl = kpl

        # Cell-type groups and symmetric interface key
        cgroupi: int = pd.cell_i.cgroup
        cgroupj: int = pd.cell_j.cgroup
        interface: tuple = tuple(sorted((cgroupi, cgroupj)))

        # Precompute shared quantities
        n_wall_junction = self.network.n_wall_junction
        intercellular_ids = np.array(
            [c.cell_id for c in self.network.cell_manager.intercellular]
        )
        is_intercellular = (
            (j - n_wall_junction) in intercellular_ids
            or (i - n_wall_junction) in intercellular_ids
        )
        length = pd.length
        fplxheight = self.hydraulic.fplxheight_map.get(interface, 8.0E5)
        pd.fplxheight = fplxheight
        base_area = fplxheight * 1.0E-04 * length

        # --- Per-interface conductance factor (replaces elif chain) --------
        # Looks up the factor key in interface_kpl_factor_map; single str keys
        # are resolved directly, Tuple[str, str] keys yield the harmonic mean.
        factor_key = self.hydraulic.interface_kpl_factor_map.get(interface)
        temp = self._get_kpl_factor(kpl_config, factor_key)

        # Aperture coefficient: same-tissue → temp; cross-tissue → harmonic
        # mean of temp and 1.0, modelling the series resistance of both sides.
        same_tissue = cgroupi == cgroupj
        aperture_coef = temp if same_tissue else 2.0 * temp / (temp + 1.0)

        # Spatially-variable plasmodesmatal aperture (disabled by default; see
        # _pd_spatial_coefficients). Keyed on raw graph node ids, same as
        # cm._pd_by_edge, since self.indice is the identity map on node ids.
        spatial_coef = self._pd_spatial_coef.get((min(i, j), max(i, j)), 1.0)
        aperture_coef *= spatial_coef
        pd.aperture_coef = aperture_coef

        # --- Air-space / cortex barrier modifier (dict lookup) -------------
        # Adjusts PD frequency at cortex interfaces when barrier > 0.
        barrier_modifier = (
            {
                (self.network.outercortex_connec_rank, 4): self.network.concentrated_outer_cortex,
                (4, 4): self.network.concentrated_cortex_cortex,
                (3, 4): self.network.concentrated_cortex_endo,
            }.get(interface, 1.0)
            if barrier > 0 else 1.0
        )

        # --- Special-case temp_factor overrides (priority list + next()) ---
        # Evaluated in order; first matching condition wins.
        # Replaces the previous if/elif block over xylem / intercellular cases.
        # Needle-only: the hypodermis/epidermis–mesophyll (2, 4) wall is
        # lignified (kw_meso_hypo set); cut its symplastic (PD) path too so the
        # outer ring is isolated apoplastically *and* symplastically. Gated on
        # kw_meso_hypo being set, which only needle_defaults() does — roots keep
        # their epidermis–cortex PD intact.
        needle_meso_hypo = (
            self.hydraulic.kw_meso_hypo is not None and interface == (2, 4)
        )
        _special_cases = [
            # Intercellular spaces have no PD
            (is_intercellular and barrier > 0,           0.0),
            # Needle hypodermis/epidermis–mesophyll: no PD across lignified wall
            (needle_meso_hypo,                           0.0),
            # Xylem-xylem connection acts as a high-conductance apoplastic path
            (interface == (13, 13),                      10000.0 * base_area),
            # Any xylem edge blocked by hydrophobic barriers
            (barrier > 0 and 13 in interface,            0.0),
        ]
        temp_factor = next(
            (val for cond, val in _special_cases if cond),
            aperture_coef * base_area * barrier_modifier,  # default
        )

        # Dead / apoptotic sclerenchyma (barrier 11): scale PD touching them
        scl_factor = (
            self._scl_pd.get(pd.cell_i.cell_id, 1.0) * self._scl_pd.get(pd.cell_j.cell_id, 1.0)
            if self._scl_pd else 1.0
        )
        temp_factor *= scl_factor

        K = kpl * temp_factor
        pd.n_pd = int(np.round(base_area * barrier_modifier * scl_factor))   # count: length-independent

        self._add_W(i, i, -K)
        self._add_W(i, j,  K)
        self._add_W(j, i,  K)
        self._add_W(j, j, -K)

        # Store K on graph edge for post-solve flow computation
        eattr['K'] = float(K)
        eattr['fplxheight'] = float(fplxheight)
        eattr['temp_factor'] = float(temp_factor)

        # Solute flux
        if self.boundary.c_flag and getattr(self.general, 'c_flag', False):
            DF = (
                self.geometry.pd_section * temp_factor
                / thickness * 1.0E-04 * self.hormones.diff1_pd1
            )
            if (i - n_wall_junction) not in self.hormones.sym_zombie0:
                self._add_C(i, i, -DF)
                self._add_C(i, j,  DF)
            if (j - n_wall_junction) not in self.hormones.sym_zombie0:
                self._add_C(j, j, -DF)
                self._add_C(j, i,  DF)

    def _fill_wall_air(
        self,
        i: int,
        j: int,
        node: int,
        neighboor: int,
        eattr: dict,
        kwa_config: dict,
        height: float,
        thickness: float,
        barrier: int,
        rhs_wa: np.ndarray,
        kw: float,
        kw_config: dict,
        kaqp_config: dict,
        a_cortex: float,
        b_cortex: float,
    ) -> None:
        r"""Fill matrix / rhs entries for a wall ↔ mesophyll air-space edge.

        Builds the linearized coupled water-vapour transport between an
        apoplastic cell-wall node and a mesophyll air-space node (both flagged
        ``protect_topology`` upstream and joined by a ``path='wall_air'`` edge in
        :meth:`NetworkBuilder.build_wall_air_connections`).

        Physics
        -------
        Water reaching the evaporating wall surface must first cross the
        membrane of the flanking mesophyll cell. The wall ↔ air transport.
        The transport is two steps in series:

            R_tot = R_memb + R_vap        (1/K_tot = 1/K_memb + 1/K_vap)

        1. *Membrane (liquid) step* — identical conductance to
           :meth:`_fill_membrane`, using the full formula with the aquaporin
           contribution and the same membrane exchange area
           ``(height + dist) * length``:

               K_memb = 1 / (1/(kw/(thickness/2)) + 1/(kmb + kaqp)) * A_memb

        2. *Evaporation (vapour) step* — at the wall surface liquid and vapour
           are in local equilibrium. The vapour pressure follows the Kelvin
           equation ``p_vapour = p_sat * exp(alpha * psi_surface)`` with
           ``alpha = M_w / (R * T)`` and constant ``p_sat``. Transport is
           diffusive, ``j = kwa * A * (p_vapour - p_air)``. Linearizing the
           exponential around ``psi_ref`` gives the wall-side slope
           ``K_vap = kwa * p_sat * exp(alpha*psi_ref) * alpha * A``.

        Eliminating the intermediate surface potential ``psi_surface`` from the
        two series steps attenuates every linearized coefficient by the same
        factor ``f = 1 / (1 + K_vap / K_memb)``:

            K_wall  = f * kwa * p_sat * e_ref * alpha * A
            K_air   = f * kwa * A
            offset  = f * kwa * p_sat * A * e_ref * (1 - alpha * psi_ref)

        yielding the linearized flux expression

            j = K_wall * psi_wall - K_air * p_air + offset

        Note that K_wall and K_air are asymmetric,
        and offset is a constant term which is included in the rhs vector.
        The exchange area ``A`` is the membrane exchange area
        ``(height + dist) * length`` (same as the transmembrane path), so that
        both series resistances refer to the same surface.


        Parameters
        ----------
        i, j : int
            Matrix indices of the two connected nodes (wall node and
            air-space cell node, in either order).
        node, neighboor : int
            Graph node ids corresponding to *i* and *j* respectively.
        eattr : dict
            Edge attribute dict (mutated in-place with ``K``, ``K_wall``,
            ``K_air``, ``offset`` and ``A``).
        kwa_config : dict
            Wall-air configuration for the current barrier level, holding the
            conductivity ``kwa`` [cm hPa⁻¹ d⁻¹] plus the linearization
            constants ``p_sat`` , ``psi_ref`` (hPa), 
            ``M_w`` (water molar mass, g mol⁻¹), ``R``(gas constant, hPa cm³ mol⁻¹ K⁻¹) 
            and ``T`` (temperature, K).
        height : float
            Cross-section height (µm).
        thickness : float
            Cell-wall half-thickness (µm).
        barrier : int
            Maturity barrier level (0 = no strip).
        rhs_wa : np.ndarray
            Persistent constant-source vector for the linearized coupling
            (mutated in-place).
        kw, kw_config, kaqp_config, a_cortex, b_cortex
            Membrane-conductance parameters (same meaning as in
            :meth:`_fill_membrane`) used to build the series membrane
            resistance ``R_memb`` of the flanking mesophyll cell.
        """
        cm = self.network.cell_manager

        # Resolve which node is the wall and which is the air-space cell.
        # wall_air edges connect a HydraulicWall node to a HydraulicCell node.
        wall = cm.get_wall_by_node_id(i) or cm.get_wall_by_node_id(j)
        if wall is not None and wall.node_id == i:
            wall_idx, air_idx = i, j
        else:
            wall_idx, air_idx = j, i

        # ── Membrane (liquid) resistance in series with evaporation ──────────
        memb_length = None
        memb_dist = None
        memb_cell_node = None
        for nbr in self.network.graph.neighbors(node if node == wall.node_id else neighboor):
            edata = self.network.graph.edges[wall.node_id, nbr]
            if edata.get('path') == 'membrane':
                memb_cell_node = nbr
                memb_length = float(edata.get('length', eattr['length']))
                memb_dist = float(edata.get('dist', eattr['dist']))
                break

        # ── Geometry: membrane exchange area (cm²) — shared by both steps ────
        length = memb_length if memb_length is not None else float(eattr['length'])
        dist = memb_dist if memb_dist is not None else float(eattr['dist'])
        area = (height + dist) * length * 1.0E-08  # µm² → cm²

        # ── Kelvin-equation parameters ──────────────────────────────────────
        kwa = float(kwa_config['kwa'])
        p_sat = float(kwa_config['p_sat'])
        psi_ref = float(kwa_config['psi_ref'])
        M_w = float(kwa_config['M_w'])
        R = float(kwa_config['R'])
        T = float(kwa_config['T'])

        # Linearization of exp(alpha * psi) around psi_ref (wall side only).
        alpha = M_w / (R * T)
        e_ref = math.exp(alpha * psi_ref)

        # Vapour (evaporation) wall-side slope, before adding membrane series.
        K_vap = kwa * p_sat * e_ref * alpha * area          # cm³ hPa⁻¹ d⁻¹

        # Membrane conductance for the flanking mesophyll cell (full formula
        # with aquaporins).  If no membrane neighbour is found the water path
        # collapses to the pure evaporation step (f = 1).
        if memb_cell_node is not None:
            K_memb = self._membrane_conductance(
                wall.node_id, memb_cell_node, length, dist,
                kw, kw_config, height,
                self._local_thickness(thickness, wall.node_id), barrier,
                kaqp_config, a_cortex, b_cortex,
            )
        else:
            K_memb = float('inf')

        # Series attenuation factor f = 1 / (1 + K_vap / K_memb).
        f = 1.0 / (1.0 + K_vap / K_memb) if K_memb > 0.0 else 0.0

        # Asymmetric conductances (attenuated by the series membrane resistance)
        K_wall = f * K_vap                                  # cm³ hPa⁻¹ d⁻¹
        K_air = f * kwa * area                              # cm³ hPa⁻¹ d⁻¹
        offset = f * kwa * p_sat * area * e_ref * (1.0 - alpha * psi_ref)

        # ── Assemble linear coupling on wall (w) and air (a) nodes ───────────
        self._add_W(wall_idx, wall_idx, -K_wall)
        self._add_W(wall_idx, air_idx,   K_air)
        self._add_W(air_idx,  wall_idx,  K_wall)
        self._add_W(air_idx,  air_idx,  -K_air)

        # Constant Kelvin linearization offset as a persistent source term.
        rhs_wa[wall_idx][0] += offset
        rhs_wa[air_idx][0]  -= offset

        # ── Record on the HydraulicWallAir connection object ────────────────
        wa = cm.get_wall_air_by_edge(wall_idx, air_idx)
        if wa is not None:
            wa.kwa = float(kwa)
            wa.K_computed = float(K_wall)
            wa.K_wall = float(K_wall)
            wa.K_air = float(K_air)
            wa.offset = float(offset)
            wa.A = float(area)

        # Store on the graph edge for post-solve flow computation.  ``K`` keeps
        # the wall-side slope for backward compatibility with generic edge-flow
        # code; ``K_wall``/``K_air``/``offset`` expose the full asymmetric
        # coupling for mecha_class flux reconstruction.
        eattr['K'] = float(K_wall)
        eattr['K_wall'] = float(K_wall)
        eattr['K_air'] = float(K_air)
        eattr['offset'] = float(offset)
        eattr['A'] = float(area)

    def _fill_air_link(
        self,
        i: int,
        j: int,
        eattr: dict,
        k_air_config: dict,
        height: float,
    ) -> None:
        r"""Fill matrix entries for an air-space ↔ air-space vapour edge.

        Builds the diffusive vapour transport between two mesophyll air-space
        nodes joined by a ``path='air_link'`` edge in
        :meth:`NetworkBuilder.build_air_link_connections`.

        Physics
        -------
        The flux follows a symmetric linear diffusion law:

            j_ij = k_air * (psi_i - psi_j)

        where ``psi_i`` and ``psi_j`` are the vapour pressures of the two air
        spaces.

        Parameters
        ----------
        i, j : int
            Matrix indices of the two connected air-space nodes.
        eattr : dict
            Edge attribute dict (mutated in-place with ``K`` and ``A``).
        k_air_config : dict
            Air-link configuration for the current barrier level, holding the
            vapour diffusion conductivity ``k_air`` [cm hPa⁻¹ d⁻¹].
        height : float
            Cross-section height (µm).
        """
        cm = self.network.cell_manager

        # ── Geometry: cross-sectional area of the shared air interface (cm²) ──
        length = float(eattr.get('length', 0.0))
        area = height * length * 1.0E-08  # µm² → cm²

        # ── Diffusive conductance: j = k_air * (psi_i - psi_j) ───────────────
        k_air = float(k_air_config['k_air'])
        K = k_air * area  # cm³ hPa⁻¹ d⁻¹

        # Symmetric Laplacian stencil (vapour conserved, no source term).
        self._add_W(i, i, -K)
        self._add_W(i, j,  K)
        self._add_W(j, i,  K)
        self._add_W(j, j, -K)

        # ── Record on the HydraulicAirLink connection object ─────────────────
        al = cm.get_air_link_by_edge(i, j)
        if al is not None:
            al.k_air = float(k_air)
            al.K_computed = float(K)
            al.A = float(area)

        # Store on the graph edge for post-solve flow computation.
        eattr['K'] = float(K)
        eattr['A'] = float(area)

    def _contact_nodes(self, x_contact):
        """Return the set of node ids in water contact with the soil.

        A border wall is a contact node when its position lies at/beyond
        ``x_contact`` or it belongs to a cell listed in ``self.hormones.contact``.
        A border junction is a contact node under the same positional rule, or
        when all of its real neighbour cells are contact cells. This mirrors
        (and is shared with) the boundary-stamping logic in
        :meth:`_apply_soil_boundary`, so the two can never diverge.
        """
        junction_wall_cell = self.geo_props['junction_wall_cell']
        contact_id_cell = self.hormones.contact
        cm = self.network.cell_manager
        contact_cells = cm.get_cells_by_ids(contact_id_cell)
        contact_walls = [walls for cell in contact_cells for walls in cell.walls]
        contact_wall_id = [wall.node_id for wall in contact_walls if wall.is_border]

        contact_nodes = set()

        for wall_id in self.network.border_walls:
            wall_has_contact = (self.position[wall_id][0] >= x_contact) or (wall_id in contact_wall_id)
            if wall_has_contact:
                contact_nodes.add(wall_id)

        for j_id in self.network.border_junction:
            cells = junction_wall_cell[j_id - self.network.n_walls]
            real_neighbors = [c for c in cells[:3] if not np.isnan(c)]
            has_contact = bool(real_neighbors) and all(
                (c - self.network.n_wall_junction) in contact_id_cell for c in real_neighbors
            )

            junction_has_contact = (self.position[j_id][0] >= x_contact) or has_contact
            if junction_has_contact:
                contact_nodes.add(j_id)

        return contact_nodes

    def _apply_soil_boundary(self, x_contact, height, thickness, kw, barrier, boundary, rhs_s, rhs_C):
        wall_to_cell = self.geo_props['wall_to_cell']
        contact_nodes = self._contact_nodes(x_contact)

        graph = self.network.graph
        for node_id in graph.nodes:
            graph.nodes[node_id]['has_contact'] = False

        for wall_id in self.network.border_walls:
            wall_has_contact = wall_id in contact_nodes
            graph.nodes[wall_id]['has_contact'] = wall_has_contact
            if wall_has_contact:
                thk = self._local_thickness(thickness, wall_id)
                temp = 1.0E-04 * (self.network.wall_lengths[wall_id] / 2 * height) / (thk / 2)
                K = kw * temp
                self._add_W(wall_id, wall_id, -K)
                rhs_s[wall_id][0] = -K
                #if rhs_C is not None:
                #    self._add_C(wall_id, wall_id, -temp * Diff1)
                #    rhs_C[wall_id][0] -= temp * Diff1 * Os_soil[0][0]

        for j_id in self.network.border_junction:
            junction_has_contact = j_id in contact_nodes
            graph.nodes[j_id]['has_contact'] = junction_has_contact
            if junction_has_contact:
                thk = self._local_thickness(thickness, j_id)
                temp = 1.0E-04 * (self.network.wall_lengths[j_id] * height) / (thk / 2)
                K = kw * temp
                self._add_W(j_id, j_id, -K)
                rhs_s[j_id][0] = -K

    def _pd_spatial_coefficients(self, h):
        """Compute a per-plasmodesmata-edge aperture coefficient from graph
        distance to the water-contact nodes.

        Disabled (returns ``{}``) unless ``self.hydraulic.pd_spatial_config``
        is set. When enabled, walks the network graph from the same
        water-contact nodes used by :meth:`_apply_soil_boundary` (via
        :meth:`_contact_nodes`) to get a Euclidean-distance field, then maps
        each ``path == 'plasmodesmata'`` edge's mean endpoint distance to a
        coefficient via a 'linear' or 'exp' profile between ``c_wet`` (d=0)
        and ``c_dry`` (d=d_max). Also stamps ``eattr['contact_distance']``,
        ``eattr['pd_spatial_coef']``, ``self.network.contact_distance`` and
        the ``'contact_distance'`` node attribute for downstream plotting.

        Returns
        -------
        dict
            ``{(min(u, v), max(u, v)): coef}`` for every plasmodesmata edge.
        """
        config = self.hydraulic.pd_spatial_config
        if config is None:
            return {}

        c_wet = config.get('c_wet', 1.0)
        c_dry = config.get('c_dry', 0.6)
        profile = config.get('profile', 'linear')
        length_scale = config.get('length_scale', None)

        x_contact = float(self.hydraulic.get_xcontact(h))
        contact_nodes = self._contact_nodes(x_contact)
        graph = self.network.graph

        if not contact_nodes:
            print("Warning: no water-contact nodes found; plasmodesmatal spatial coefficients disabled.")
            return {}

        def _edge_weight(u, v, _eattr):
            pos_u = self.position.get(u)
            pos_v = self.position.get(v)
            if pos_u is None or pos_v is None:
                return 0.0
            return float(math.hypot(pos_u[0] - pos_v[0], pos_u[1] - pos_v[1]))

        dist = nx.multi_source_dijkstra_path_length(graph, sources=contact_nodes, weight=_edge_weight)

        # Store the full node distance field for downstream plotting.
        self.network.contact_distance = dist
        for n in graph.nodes:
            graph.nodes[n]['contact_distance'] = dist.get(n, float('nan'))

        # First pass: mean endpoint distance of every plasmodesmata edge.
        pd_edges = []
        for u, v, eattr in graph.edges(data=True):
            if eattr.get('path') != 'plasmodesmata':
                continue
            if u in dist and v in dist:
                pd_edges.append((u, v, eattr, 0.5 * (dist[u] + dist[v])))
            else:
                pd_edges.append((u, v, eattr, float('nan')))

        # Normalise on the plasmodesmata themselves, over their distance *range*
        # rather than absolute distance. The coefficient only ever applies to PD
        # edges, and the closest cell-to-cell plasmodesma still sits a cell
        # radius away from the wetted wall nodes, so anchoring on d_min is what
        # makes c_wet and c_dry the exact values at the wettest and driest
        # plasmodesma.
        reachable = [d for _, _, _, d in pd_edges if not math.isnan(d)]
        d_min = min(reachable) if reachable else 0.0
        d_max = max(reachable) if reachable else 0.0
        d_span = d_max - d_min
        if length_scale is None:
            length_scale = d_span / 3.0

        def _coef(d):
            if d_span <= 0:
                return c_wet
            u = min(max((d - d_min) / d_span, 0.0), 1.0)
            if profile == 'exp':
                if length_scale <= 0:
                    return c_dry
                return c_dry + (c_wet - c_dry) * math.exp(-(d - d_min) / length_scale)
            return c_wet - (c_wet - c_dry) * u

        coefficients = {}
        for u, v, eattr, d_e in pd_edges:
            coef = c_dry if math.isnan(d_e) else _coef(d_e)
            eattr['contact_distance'] = float(d_e)
            eattr['pd_spatial_coef'] = float(coef)
            coefficients[(min(u, v), max(u, v))] = coef

        return coefficients

    def _apply_xylo_phloem_boundary(self, i_maturity, barrier, psi_xyl, psi_sieve, distributed_flow_xyl, distributed_flow_sieve, boundary, rhs_s, rhs_x, rhs_p, rhs):
        if barrier > 0:
            if not np.isnan(psi_xyl[1][i_maturity][0]):
                for cid in self.network.xylem_cells:
                    if not isinstance(self.hydraulic.k_xyl, list):
                        k_xyl_value = self.hydraulic.k_xyl
                    else:
                        k_xyl_value = self.hydraulic.k_xyl[0] # not true ?
                    rhs_x[cid][0] = -k_xyl_value
                    self._add_W(cid, cid, -k_xyl_value)
                rhs[:] = rhs_s * boundary.scenarios[0]['psi_soil_left'] + rhs_x * psi_xyl[1][i_maturity][0]

                if not np.isnan(psi_xyl[0][i_maturity][0]):
                    print('Distal xylem pressure BC not accounted for in kr estimation')

            elif not np.isnan(distributed_flow_xyl[1][0][0]):
                for i, cid in enumerate(self.network.xylem_cells):
                    rhs_x[cid][0] = distributed_flow_xyl[1][i+1][0]
                rhs[:] = rhs_s * boundary.scenarios[0]['psi_soil_left'] + rhs_x
            else:
                rhs[:] = rhs_s * boundary.scenarios[0]['psi_soil_left']

        elif barrier == 0:
            if not np.isnan(psi_sieve[1][i_maturity][0]):
                for cid in getattr(self.network, 'protosieve_cells', []):
                    if not isinstance(self.hydraulic.k_sieve, list):
                        k_sieve_value = self.hydraulic.k_sieve
                    else:
                        k_sieve_value = self.hydraulic.k_sieve[0]
                    rhs_p[cid][0] = -k_sieve_value
                    self._add_W(cid, cid, -k_sieve_value)
                rhs[:] = rhs_s * boundary.scenarios[0]['psi_soil_left'] + rhs_p * psi_sieve[1][i_maturity][0]
            elif not np.isnan(distributed_flow_sieve[1][0][0]):
                for i, cid in enumerate(getattr(self.network, 'protosieve_cells', [])):
                    rhs_p[cid][0] = distributed_flow_sieve[1][i+1][0]
                rhs[:] = rhs_s * boundary.scenarios[0]['psi_soil_left'] + rhs_p
            else:
                rhs[:] = rhs_s * boundary.scenarios[0]['psi_soil_left']
