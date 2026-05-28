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

        Kmb = np.zeros((self.network.n_membrane, 1))
        jmb = 0

        # 1. Edge loops (wall, membrane, plasmodesmata conductances)
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
        return matrix_W, matrix_C, rhs_C, rhs_p, rhs_x, rhs_s, rhs, Kmb

    def _add_W(self, i, j, val):
        self._rows_W.append(i)
        self._cols_W.append(j)
        self._data_W.append(val)

    def _add_C(self, i, j, val):
        self._rows_C.append(i)
        self._cols_C.append(j)
        self._data_C.append(val)

    def _fill_wall(self, i, j, node, eattr, kw, kw_config, height, thickness, barrier):
        count_interC = self.network.graph.nodes[node].get('count_interC', 0)
        count_xyl = self.network.graph.nodes[node].get('count_xyl', 0)
        count_cortex = self.network.graph.nodes[node].get('count_cortex', 0)
        count_endo = self.network.graph.nodes[node].get('count_endo', 0)
        count_stele_overall = self.network.graph.nodes[node].get('count_stele_overall', 0)
        count_passage = self.network.graph.nodes[node].get('count_passage', 0)
        count_exo = self.network.graph.nodes[node].get('count_exo', 0)

        temp = 1.0E-04 * ((eattr['lateral_distance'] + height) * thickness - thickness**2) / eattr['length']
        temp_factor = 1.0

        xylem_pieces = self.geometry.xylem_pieces if hasattr(self.geometry, 'xylem_pieces') else False

        if (count_interC >= 2 and barrier > 0) or (count_xyl == 2 and xylem_pieces):
            K = 1.0E-16
            temp_factor = 1.0E-16
            # ghost junction logic
            # n_walls = self.network.n_walls
            if j not in self.network.list_ghostjunctions:
                fakeJ = True
                for ind in range(int(self.network.n_junction_to_wall[j])): # why j? and no -n_walls? old code: for ind in range(int(self.network.n_junction_to_wall[j - n_walls])):
                    if self.network.junction_to_wall[j][ind] not in self.network.list_ghostwalls: # why j? and no -n_walls? old code: if self.network.junction_to_wall[j - n_walls][ind] not in self.network.list_ghostwalls:
                        fakeJ = False
                if fakeJ:
                    self.network.list_ghostjunctions.append(j)
                    self.network.n_ghost_junction2wall += int(self.network.n_junction_to_wall[j]) + 2
                elif count_interC >= 2: # septa walls
                    K = kw_config['kw_septa'] * temp
                    if kw > 0: temp_factor = kw_config['kw_septa'] / kw
        elif count_cortex >= 2:
            K = kw_config['kw_cortex_cortex'] * temp
            if kw > 0: temp_factor = kw_config['kw_cortex_cortex'] / kw
        elif count_endo >= 2:
            K = kw_config['kw_endo_endo'] * temp
            if kw > 0: temp_factor = kw_config['kw_endo_endo'] / kw
        elif count_stele_overall > 0 and count_endo > 0:
            if count_passage > 0:
                K = kw_config['kw_passage'] * temp
                if kw > 0: temp_factor = kw_config['kw_passage'] / kw
            else:
                K = kw_config['kw_endo_peri'] * temp
                if kw > 0: temp_factor = kw_config['kw_endo_peri'] / kw
        elif count_stele_overall == 0 and count_endo == 1:
            if count_passage > 0:
                K = kw_config['kw_passage'] * temp
                if kw > 0: temp_factor = kw_config['kw_passage'] / kw
            else:
                K = kw_config['kw_endo_cortex'] * temp
                if kw > 0: temp_factor = kw_config['kw_endo_cortex'] / kw
        elif count_exo >= 2:
            K = kw_config['kw_exo_exo'] * temp
            if kw > 0: temp_factor = kw_config['kw_exo_exo'] / kw
        else:
            K = kw * temp

        self._add_W(i, i, -K)
        self._add_W(i, j,  K)
        self._add_W(j, i,  K)
        self._add_W(j, j, -K)

        # Solute flux
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

    def _fill_membrane(self, i, j, node, neighboor, eattr, kw, kw_config, height, thickness, barrier, kaqp_config, a_cortex, b_cortex):
        count_endo = self.network.graph.nodes[node].get('count_endo', 0)
        count_exo = self.network.graph.nodes[node].get('count_exo', 0)
        count_stele_overall = self.network.graph.nodes[node].get('count_stele_overall', 0)
        count_passage = self.network.graph.nodes[node].get('count_passage', 0)
        count_epi = self.network.graph.nodes[node].get('count_epi', 0)

        cgroup = self.network.graph.nodes[neighboor]['cgroup']
        n_wall_junction = self.network.n_wall_junction

        intercellular_ids = np.array([c.cell_id for c in self.network.cell_manager.intercellular])

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

        kaqp_curr = 0.0
        if cgroup == 1: kaqp_curr = kaqp_config['kaqp_exo']
        elif cgroup == 2: kaqp_curr = kaqp_config['kaqp_epi']
        elif cgroup == 3: kaqp_curr = kaqp_config['kaqp_endo']
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

        # Conductance
        K = 0.0
        kw_endo_endo = kw_config['kw_endo_endo']
        kw_exo_exo = kw_config['kw_exo_exo']
        kw_passage = kw_config['kw_passage']
        kw_endo_peri = kw_config['kw_endo_peri']
        kw_endo_cortex = kw_config['kw_endo_cortex']

        def calc_K(kw_val):
            if kw_val == 0.0: return 0.0
            return 1 / (1 / (kw_val / (thickness / 2 * 1.0E-04)) + 1 / (self.hydraulic.kmb + kaqp_curr)) * 1.0E-08 * (height + eattr['dist']) * eattr['length']

        if count_endo >= 2: K = calc_K(kw_endo_endo)
        elif count_exo >= 2: K = calc_K(kw_exo_exo)
        elif count_stele_overall > 0 and count_endo > 0:
            if count_passage > 0: K = calc_K(kw_passage)
            else: K = calc_K(kw_endo_peri)
        elif count_stele_overall == 0 and count_endo == 1:
            if kaqp_curr == 0.0: K = 1.00E-16
            else:
                if count_passage > 0: K = calc_K(kw_passage)
                else: K = calc_K(kw_endo_cortex)
        else:
            if kaqp_curr == 0.0: K = 1.00E-16
            else: K = calc_K(kw)

        self._add_W(i, i, -K)
        self._add_W(i, j,  K)
        self._add_W(j, i,  K)
        self._add_W(j, j, -K)

        # Store K on graph edge for post-solve flow computation
        eattr['K'] = float(K)

        return K

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

    def _apply_soil_boundary(self, x_contact, height, thickness, kw, barrier, boundary, rhs_s, rhs_C):
        wall_to_cell = self.geo_props['wall_to_cell']
        junction_wall_cell = self.geo_props['junction_wall_cell']
        for wall_id in self.network.border_walls:
            if (self.position[wall_id][0] >= x_contact) or ((wall_to_cell[wall_id][0] - self.network.n_wall_junction) in getattr(self.hormones, 'contact', [])):
                temp = 1.0E-04 * (self.network.wall_lengths[wall_id] / 2 * height) / (thickness / 2)
                K = kw * temp
                self._add_W(wall_id, wall_id, -K)
                rhs_s[wall_id][0] = -K
                #if rhs_C is not None:
                #    self._add_C(wall_id, wall_id, -temp * Diff1)
                #    rhs_C[wall_id][0] -= temp * Diff1 * Os_soil[0][0]

        for j_id in self.network.border_junction:
            cells = junction_wall_cell[j_id - self.network.n_walls]
            contact_nodes = getattr(self.hormones, 'contact', [])
            has_contact = any((c - self.network.n_wall_junction) in contact_nodes for c in cells[:3] if not np.isnan(c))

            if (self.position[j_id][0] >= x_contact) or has_contact:
                temp = 1.0E-04 * (self.network.wall_lengths[j_id] * height) / (thickness / 2)
                K = kw * temp
                self._add_W(j_id, j_id, -K)
                rhs_s[j_id][0] = -K

    def _apply_xylo_phloem_boundary(self, i_maturity, barrier, psi_xyl, psi_sieve, distributed_flow_xyl, distributed_flow_sieve, boundary, rhs_s, rhs_x, rhs_p, rhs):
        if barrier > 0:
            if not np.isnan(psi_xyl[1][i_maturity][0]):
                for cid in self.network.xylem_cells:
                    rhs_x[cid][0] = -self.hydraulic.k_xyl
                    self._add_W(cid, cid, -self.hydraulic.k_xyl)
                rhs[:] = rhs_s * boundary.scenarios[0]['psi_soil_left'] + rhs_x * psi_xyl[1][i_maturity][0]

                if not np.isnan(psi_xyl[0][i_maturity][0]):
                    print('Distal xylem pressure BC not accounted for in kr estimation')

            elif not np.isnan(distributed_flow_xyl[1][1][0]):
                for i, cid in enumerate(self.network.xylem_cells):
                    rhs_x[cid][0] = distributed_flow_xyl[1][i+1][0]
                rhs[:] = rhs_s * boundary.scenarios[0]['psi_soil_left'] + rhs_x
            else:
                rhs[:] = rhs_s * boundary.scenarios[0]['psi_soil_left']

        elif barrier == 0:
            if not np.isnan(psi_sieve[1][i_maturity][0]):
                for cid in getattr(self.network, 'protosieve_list', []):
                    rhs_p[cid][0] = -self.hydraulic.k_sieve
                    self._add_W(cid, cid, -self.hydraulic.k_sieve)
                rhs[:] = rhs_s * boundary.scenarios[0]['psi_soil_left'] + rhs_p * psi_sieve[1][i_maturity][0]
            elif not np.isnan(distributed_flow_sieve[1][1][0]):
                for i, cid in enumerate(getattr(self.network, 'protosieve_list', [])):
                    rhs_p[cid][0] = distributed_flow_sieve[1][i+1][0]
                rhs[:] = rhs_s * boundary.scenarios[0]['psi_soil_left'] + rhs_p
            else:
                rhs[:] = rhs_s * boundary.scenarios[0]['psi_soil_left']
