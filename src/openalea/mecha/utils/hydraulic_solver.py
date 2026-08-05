from openalea.mecha.utils import hydraulic_cell
import numpy as np
import math
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

    def build(self, h, i_maturity, hydraulic_conductivities, boundary,
              psi_xyl, psi_sieve, distributed_flow_xyl, distributed_flow_sieve):

        # Unpack properties
        maturity_stages = self.geometry.maturity_stages
        barrier = int(maturity_stages[i_maturity].get("barrier"))
        height = float(maturity_stages[i_maturity].get("height"))
        x_contact = float(self.hydraulic.xcontactrange[h])
        
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

        n_nodes = self.network.graph.number_of_nodes()
        thickness = self.geometry.thickness

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
           ``if/elif`` chain.
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

        # Shared geometric conductance factor
        temp: float = (
            1.0E-04
            * ((eattr['lateral_distance'] + height) * thickness - thickness ** 2)
            / eattr['length']
        )
        temp_factor: float = 1.0

        # Get the flanking cells for this wall node via HydraulicCellManager
        cells_i = self._get_cells_for_wall_node(i)

        # --- Classify flanking cells ----------------------------------------
        intercellular_cids = {c.cell_id for c in cm.intercellular}
        xylem_cgroups = {13, 19, 20}
        passage_cids = {c.cell_id for c in cm.passage}

        n_interc = sum(1 for c in cells_i if c.cell_id in intercellular_cids)
        n_xyl = sum(1 for c in cells_i if c.cgroup in xylem_cgroups)
        has_passage = any(c.cell_id in passage_cids for c in cells_i)

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

        # ── Guard 2: Passage-cell wall ────────────────────────────────────────
        # At least one flanking cell is a passage cell → always use kw_passage.
        elif has_passage:
            kw_val = kw_config.get('kw_passage', kw)
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
                if kw_key is not None:
                    kw_val = kw_config.get(kw_key, kw)
                    K = kw_val * temp
                    if kw > 0:
                        temp_factor = kw_val / kw
                else:
                    # Interface not in map → plain kw (e.g. stele–stele,
                    # pericycle–stele, or any unlisted same-tissue interface)
                    K = kw * temp

        # ── Fallback: border/exterior wall or junction with >2 cells ─────────
        else:
            K = kw * temp

        # Store K on the HydraulicWall object for post-solve inspection
        wall = cm.get_wall_by_node_id(i)
        if wall is not None:
            wall.kw = float(K)

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
        elif cgroup in [13, 19, 20]:
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

        # ── Flanking cells information ────────────────────────────────────────
        cells_i = self._get_cells_for_wall_node(i)
        passage_cids = {c.cell_id for c in cm.passage}

        n_endo = sum(1 for c in cells_i if c.cgroup == 3)
        n_exo = sum(1 for c in cells_i if c.cgroup == 1)
        n_stele = sum(1 for c in cells_i if c.cgroup > 4)
        n_passage = sum(1 for c in cells_i if c.cell_id in passage_cids)

        # ── Select kw_key via interface map ──────────────────────────────────
        kw_key = None
        if len(cells_i) == 2:
            interface = tuple(sorted((cells_i[0].cgroup, cells_i[1].cgroup)))
            kw_key = self.hydraulic.interface_kw_key_map.get(interface, None)

        # Passage-cell override (only if endodermis is present)
        if n_endo > 0 and n_passage > 0:
            kw_key = 'kw_passage'

        kw_val = kw_config.get(kw_key, kw) if kw_key is not None else kw

        # ── Membrane conductance calculation ──────────────────────────────────
        is_barrier_interface = (n_endo >= 2) or (n_exo >= 2) or (n_stele > 0 and n_endo > 0)

        if (not is_barrier_interface) and kaqp_curr == 0.0:
            K = 1.00E-16
        else:
            if kw_val == 0.0:
                K = 0.0
            else:
                K = 1 / (1 / (kw_val / (thickness / 2 * 1.0E-04)) + 1 / (self.hydraulic.kmb + kaqp_curr)) * 1.0E-08 * (height + eattr['dist']) * eattr['length']

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

        # ── Select kw_key via interface map (mirrors _fill_membrane) ──────────
        cells_i = self._get_cells_for_wall_node(wall_idx)
        passage_cids = {c.cell_id for c in cm.passage}
        n_endo = sum(1 for c in cells_i if c.cgroup == 3)
        n_exo = sum(1 for c in cells_i if c.cgroup == 1)
        n_stele = sum(1 for c in cells_i if c.cgroup > 4)
        n_passage = sum(1 for c in cells_i if c.cell_id in passage_cids)

        kw_key = None
        if len(cells_i) == 2:
            interface = tuple(sorted((cells_i[0].cgroup, cells_i[1].cgroup)))
            kw_key = self.hydraulic.interface_kw_key_map.get(interface, None)
        if n_endo > 0 and n_passage > 0:
            kw_key = 'kw_passage'
        kw_val = kw_config.get(kw_key, kw) if kw_key is not None else kw

        # ── Membrane conductance calculation (identical to _fill_membrane) ────
        is_barrier_interface = (n_endo >= 2) or (n_exo >= 2) or (n_stele > 0 and n_endo > 0)
        if (not is_barrier_interface) and kaqp_curr == 0.0:
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
        kpl_config: dict, thickness: float, barrier: int
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
        _special_cases = [
            # Intercellular spaces have no PD
            (is_intercellular and barrier > 0,           0.0),
            # Xylem-xylem connection acts as a high-conductance apoplastic path
            (interface == (13, 13),                      10000.0 * base_area),
            # Any xylem edge blocked by hydrophobic barriers
            (barrier > 0 and 13 in interface,            0.0),
        ]
        temp_factor = next(
            (val for cond, val in _special_cases if cond),
            aperture_coef * base_area * barrier_modifier,  # default
        )

        K = kpl * temp_factor
        pd.n_pd = int(np.round(base_area * barrier_modifier))

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
                kw, kw_config, height, thickness, barrier,
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

    def _apply_soil_boundary(self, x_contact, height, thickness, kw, barrier, boundary, rhs_s, rhs_C):
        wall_to_cell = self.geo_props['wall_to_cell']
        junction_wall_cell = self.geo_props['junction_wall_cell']
        contact_id_cell = self.hormones.contact
        cm = self.network.cell_manager
        contact_cells = cm.get_cells_by_ids(contact_id_cell)
        contact_walls = [walls for cell in contact_cells for walls in cell.walls]
        contact_wall_id = [wall.node_id for wall in contact_walls if wall.is_border]

        for wall_id in self.network.border_walls:
            if (self.position[wall_id][0] >= x_contact) or (wall_id in contact_wall_id):
                temp = 1.0E-04 * (self.network.wall_lengths[wall_id] / 2 * height) / (thickness / 2)
                K = kw * temp
                self._add_W(wall_id, wall_id, -K)
                rhs_s[wall_id][0] = -K
                #if rhs_C is not None:
                #    self._add_C(wall_id, wall_id, -temp * Diff1)
                #    rhs_C[wall_id][0] -= temp * Diff1 * Os_soil[0][0]

        for j_id in self.network.border_junction:
            cells = junction_wall_cell[j_id - self.network.n_walls]
            has_contact = any((c - self.network.n_wall_junction) in contact_id_cell for c in cells[:3] if not np.isnan(c))

            if (self.position[j_id][0] >= x_contact) or has_contact:
                temp = 1.0E-04 * (self.network.wall_lengths[j_id] * height) / (thickness / 2)
                K = kw * temp
                self._add_W(j_id, j_id, -K)
                rhs_s[j_id][0] = -K


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
