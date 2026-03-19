import numpy as np
import math

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
        
        n_nodes = self.network.graph.number_of_nodes()
        n_walls = self.network.n_walls
        n_wall_junction = self.network.n_wall_junction
        thickness = self.geometry.thickness
        
        # Initialize matrices
        matrix_W = np.zeros((n_nodes, n_nodes))
        matrix_C = np.zeros((n_nodes, n_nodes)) if self.boundary.c_flag else None
        
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
                        self._fill_wall(i, j, node, eattr, kw, kw_config, height, thickness, barrier, matrix_W, matrix_C)
                    elif path == 'membrane':
                        K_mem = self._fill_membrane(i, j, node, neighboor, eattr, kw, kw_config, height, thickness, barrier,
                                                   kaqp_config, a_cortex, b_cortex, matrix_W, matrix_C)
                        Kmb[jmb] = K_mem
                        jmb += 1
                    elif path == 'plasmodesmata':
                        self._fill_plasmodesmata(i, j, node, neighboor, eattr, kpl, kpl_config, height, thickness, barrier, matrix_W, matrix_C)

        # 2. Add soil-wall connections
        self._apply_soil_boundary(x_contact, height, thickness, kw, barrier, boundary, matrix_W, rhs_s, matrix_C, rhs_C)
        
        # 3. Add xylem / phloem BC
        self._apply_xylo_phloem_boundary(i_maturity, barrier, psi_xyl, psi_sieve, distributed_flow_xyl, distributed_flow_sieve, boundary, matrix_W, rhs_s, rhs_x, rhs_p, rhs)
        
        # Unified solute matrix -> removed ApoC and SymC
        return matrix_W, matrix_C, rhs_C, rhs_p, rhs_x, rhs_s, rhs, Kmb

    def _fill_wall(self, i, j, node, eattr, kw, kw_config, height, thickness, barrier, matrix_W, matrix_C):
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
            n_walls = self.network.n_walls
            if j not in self.network.list_ghostjunctions:
                fakeJ = True
                for ind in range(int(self.network.n_junction_to_wall[j - n_walls])):
                    if self.network.junction_to_wall[j - n_walls][ind] not in self.network.list_ghostwalls:
                        fakeJ = False
                if fakeJ:
                    self.network.list_ghostjunctions.append(j)
                    self.network.n_ghost_junction2wall += int(self.network.n_junction_to_wall[j - n_walls]) + 2
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

        matrix_W[i][i] -= K
        matrix_W[i][j] += K
        matrix_W[j][i] += K
        matrix_W[j][j] -= K
        
        # Solute flux
        if matrix_C is not None and self.general.c_flag:
            DF = temp * temp_factor * self.hormones.diff1_pw1
            if i not in self.network.apo_wall_zombies0:
                matrix_C[i][i] -= DF
                matrix_C[i][j] += DF
            if j not in self.network.apo_j_zombies0:
                matrix_C[j][j] -= DF
                matrix_C[j][i] += DF

    def _fill_membrane(self, i, j, node, neighboor, eattr, kw, kw_config, height, thickness, barrier, kaqp_config, a_cortex, b_cortex, matrix_W, matrix_C):
        count_endo = self.network.graph.nodes[node].get('count_endo', 0)
        count_exo = self.network.graph.nodes[node].get('count_exo', 0)
        count_stele_overall = self.network.graph.nodes[node].get('count_stele_overall', 0)
        count_passage = self.network.graph.nodes[node].get('count_passage', 0)
        count_epi = self.network.graph.nodes[node].get('count_epi', 0)
        
        cgroup = self.network.graph.nodes[neighboor]['cgroup']
        n_wall_junction = self.network.n_wall_junction
        
        intercellular_ids = self.geometry.intercellular_ids if hasattr(self.geometry, 'intercellular_ids') else []

        if matrix_C is not None and self.general.c_flag:
            for carrier in getattr(self.hormones, 'carrier_elems', []):
                if int(carrier.get("tissue")) == cgroup:
                    cid = j - n_wall_junction
                    if cid not in intercellular_ids and not (barrier > 0 and cgroup in [13, 19, 20]):
                        temp_c = float(carrier.get("constant")) * (height + eattr['dist']) * eattr['length']
                        direction = int(carrier.get("direction"))
                        if direction == 1:
                            if cid not in self.hormones.sym_zombie0:
                                matrix_C[j][i] += temp_c
                            if i not in self.network.apo_wall_zombies0:
                                matrix_C[i][i] -= temp_c
                        elif direction == -1:
                            if cid not in self.hormones.sym_zombie0:
                                matrix_C[j][j] -= temp_c
                            if i not in self.network.apo_wall_zombies0:
                                matrix_C[i][j] += temp_c

        kaqp_curr = 0.0
        if cgroup == 1: kaqp_curr = kaqp_config['kaqp_exo']
        elif cgroup == 2: kaqp_curr = kaqp_config['kaqp_epi']
        elif cgroup == 3: kaqp_curr = kaqp_config['kaqp_endo']
        elif cgroup in [13, 19, 20]:
            if barrier > 0:
                kaqp_curr = kaqp_config['kaqp_stele'] * 10000
                if matrix_C is not None and self.general.c_flag:
                    temp_c = 1.0E-04 * (self.network.wall_lengths[i] * height) / thickness
                    if i not in self.network.apo_wall_zombies0:
                        matrix_C[i][i] -= temp_c * self.hormones.diff1_pw1
                        matrix_C[i][j] += temp_c * self.hormones.diff1_pw1
                    if (j - n_wall_junction) not in self.hormones.sym_zombie0:
                        matrix_C[j][j] -= temp_c * self.hormones.diff1_pw1
                        matrix_C[j][i] += temp_c * self.hormones.diff1_pw1
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

        matrix_W[i][i] -= K
        matrix_W[i][j] += K
        matrix_W[j][i] += K
        matrix_W[j][j] -= K

        return K

    def _fill_plasmodesmata(self, i, j, node, neighboor, eattr, kpl, kpl_config, height, thickness, barrier, matrix_W, matrix_C):
        cgroupi = self.network.graph.nodes[node].get('cgroup')
        cgroupj = self.network.graph.nodes[neighboor].get('cgroup')

        def map_cgroup(cg):
            if cg in [19, 20]: return 13
            elif cg == 21: return 16
            elif cg == 23: return 11
            elif cg == 26: return 12
            return cg
        
        cgroupi = map_cgroup(cgroupi)
        cgroupj = map_cgroup(cgroupj)
        
        n_wall_junction = self.network.n_wall_junction
        intercellular_ids = self.geometry.intercellular_ids if hasattr(self.geometry, 'intercellular_ids') else []

        temp_factor = 1.0
        
        if (((j - n_wall_junction) in intercellular_ids) or ((i - n_wall_junction) in intercellular_ids)) and barrier > 0:
            temp_factor = 0.0
        elif cgroupj == 13 and cgroupi == 13:
            temp_factor = 10000 * self.hydraulic.fplxheight * 1.0E-04 * eattr['length']
        elif barrier > 0 and (cgroupj == 13 or cgroupi == 13):
            temp_factor = 0.0
        elif (cgroupi == 2 and cgroupj == 1) or (cgroupj == 2 and cgroupi == 1):
            temp_factor = self.hydraulic.fplxheight_epi_exo * 1.0E-04 * eattr['length']
        elif (cgroupi == self.network.outercortex_connec_rank and cgroupj == 4) or (cgroupj == self.network.outercortex_connec_rank and cgroupi == 4):
            temp = float(kpl_config['cortex_factor'])
            if barrier > 0:
                temp_factor = 2 * temp / (temp + 1) * self.hydraulic.fplxheight_outer_cortex * 1.0E-04 * eattr['length'] * self.network.len_outer_cortex / self.network.cross_section_outer_cortex
            else:
                temp_factor = 2 * temp / (temp + 1) * self.hydraulic.fplxheight_outer_cortex * 1.0E-04 * eattr['length']
        elif (cgroupi == 4 and cgroupj == 4):
            temp = float(kpl_config['cortex_factor'])
            if barrier > 0:
                temp_factor = temp * self.hydraulic.fplxheight_cortex_cortex * 1.0E-04 * eattr['length'] * self.network.len_cortex_cortex / self.network.cross_section_cortex_cortex
            else:
                temp_factor = temp * self.hydraulic.fplxheight_cortex_cortex * 1.0E-04 * eattr['length']
        elif (cgroupi == 3 and cgroupj == 4) or (cgroupj == 3 and cgroupi == 4):
            temp = float(kpl_config['cortex_factor'])
            if barrier > 0:
                temp_factor = 2 * temp / (temp + 1) * self.hydraulic.fplxheight_cortex_endo * 1.0E-04 * eattr['length'] * self.network.len_cortex_endo / self.network.cross_section_cortex_endo
            else:
                temp_factor = 2 * temp / (temp + 1) * self.hydraulic.fplxheight_cortex_endo * 1.0E-04 * eattr['length']
        elif (cgroupi == 3 and cgroupj == 3):
            temp_factor = self.hydraulic.fplxheight_endo_endo * 1.0E-04 * eattr['length']
        elif (cgroupi == 3 and cgroupj == 16) or (cgroupj == 3 and cgroupi == 16):
            if ((i - n_wall_junction) in self.network.plasmodesmata_indice) or ((j - n_wall_junction) in self.network.plasmodesmata_indice):
                temp = float(kpl_config['phloem_pericycle_pole_factor'])
            else: temp = 1
            temp_factor = 2 * temp / (temp + 1) * self.hydraulic.fplxheight_endo_peri * 1.0E-04 * eattr['length']
        elif (cgroupi == 16 and (cgroupj == 5 or cgroupj == 13)) or (cgroupj == 16 and (cgroupi == 5 or cgroupi == 13)):
            if ((i - n_wall_junction) in self.network.plasmodesmata_indice) or ((j - n_wall_junction) in self.network.plasmodesmata_indice):
                temp = float(kpl_config['phloem_pericycle_pole_factor'])
            else: temp = 1
            temp_factor = 2 * temp / (temp + 1) * self.hydraulic.fplxheight_peri_stele * 1.0E-04 * eattr['length']
        elif ((cgroupi == 5 or cgroupi == 13) and cgroupj == 12) or (cgroupi == 12 and (cgroupj == 5 or cgroupj == 13)):
            temp = float(kpl_config['phloem_companion_cell_factor'])
            temp_factor = 2 * temp / (temp + 1) * self.hydraulic.fplxheight_stele_comp * 1.0E-04 * eattr['length']
        elif (cgroupi == 16 and cgroupj == 12) or (cgroupi == 12 and cgroupj == 16):
            temp1 = float(kpl_config['phloem_companion_cell_factor'])
            if ((i - n_wall_junction) in self.network.plasmodesmata_indice) or ((j - n_wall_junction) in self.network.plasmodesmata_indice):
                temp2 = float(kpl_config['phloem_pericycle_pole_factor'])
            else: temp2 = 1
            temp_factor = 2 * temp1 * temp2 / (temp1 + temp2) * self.hydraulic.fplxheight_peri_comp * 1.0E-04 * eattr['length']
        elif (cgroupi == 12 and cgroupj == 12):
            temp = float(kpl_config['phloem_companion_cell_factor'])
            temp_factor = temp * self.hydraulic.fplxheight_comp_comp * 1.0E-04 * eattr['length']
        elif (cgroupi == 12 and cgroupj == 11) or (cgroupi == 11 and cgroupj == 12):
            temp = float(kpl_config['phloem_companion_cell_factor'])
            temp_factor = 2 * temp / (temp + 1) * self.hydraulic.fplxheight_comp_sieve * 1.0E-04 * eattr['length']
        elif (cgroupi == 16 and cgroupj == 11) or (cgroupi == 11 and cgroupj == 16):
            if ((i - n_wall_junction) in self.network.plasmodesmata_indice) or ((j - n_wall_junction) in self.network.plasmodesmata_indice):
                temp = float(kpl_config['phloem_pericycle_pole_factor'])
            else: temp = 1
            temp_factor = 2 * temp / (temp + 1) * self.hydraulic.fplxheight_peri_sieve * 1.0E-04 * eattr['length']
        elif ((cgroupi == 5 or cgroupi == 13) and cgroupj == 11) or (cgroupi == 11 and (cgroupj == 5 or cgroupj == 13)):
            temp_factor = self.hydraulic.fplxheight_stele_sieve * 1.0E-04 * eattr['length']
        elif ((cgroupi == 5 or cgroupi == 13) and (cgroupj == 5 or cgroupj == 13)):
            temp_factor = self.hydraulic.fplxheight_stele_stele * 1.0E-04 * eattr['length']
        else:
            temp_factor = self.hydraulic.fplxheight * 1.0E-04 * eattr['length']

        K = kpl * temp_factor

        matrix_W[i][i] -= K
        matrix_W[i][j] += K
        matrix_W[j][i] += K
        matrix_W[j][j] -= K
        
        # Solute flux
        if matrix_C is not None and getattr(self.general, 'c_flag', False):
            DF = self.geometry.pd_section * temp_factor / thickness * 1.0E-04 * self.hormones.diff1_pd1
            if (i - n_wall_junction) not in self.hormones.sym_zombie0:
                matrix_C[i][i] -= DF
                matrix_C[i][j] += DF
            if (j - n_wall_junction) not in self.hormones.sym_zombie0:
                matrix_C[j][j] -= DF
                matrix_C[j][i] += DF

    def _apply_soil_boundary(self, x_contact, height, thickness, kw, barrier, boundary, matrix_W, rhs_s, matrix_C, rhs_C):
        wall_to_cell = self.geo_props['wall_to_cell']
        junction_wall_cell = self.geo_props['junction_wall_cell']
        
        for wall_id in self.network.border_walls:
            if (self.position[wall_id][0] >= x_contact) or ((wall_to_cell[wall_id][0] - self.network.n_wall_junction) in getattr(self.hormones, 'contact', [])):
                temp = 1.0E-04 * (self.network.wall_lengths[wall_id] / 2 * height) / (thickness / 2)
                K = kw * temp
                matrix_W[wall_id][wall_id] -= K
                rhs_s[wall_id][0] = -K
                #if matrix_C is not None:
                #    matrix_C[wall_id][wall_id] -= temp * Diff1
                #    rhs_C[wall_id][0] -= temp * Diff1 * Os_soil[0][0]
                
        for j_id in self.network.border_junction:
            cells = junction_wall_cell[j_id - self.network.n_walls]
            contact_nodes = getattr(self.hormones, 'contact', [])
            has_contact = any((c - self.network.n_wall_junction) in contact_nodes for c in cells[:3] if not np.isnan(c))
            
            if (self.position[j_id][0] >= x_contact) or has_contact:
                temp = 1.0E-04 * (self.network.wall_lengths[j_id] * height) / (thickness / 2)
                K = kw * temp
                matrix_W[j_id][j_id] -= K
                rhs_s[j_id][0] = -K

    def _apply_xylo_phloem_boundary(self, i_maturity, barrier, psi_xyl, psi_sieve, distributed_flow_xyl, distributed_flow_sieve, boundary, matrix_W, rhs_s, rhs_x, rhs_p, rhs):
        if barrier > 0:
            if not np.isnan(psi_xyl[1][i_maturity][0]):
                for cid in self.network.xylem_cells:
                    rhs_x[cid][0] = -self.hydraulic.k_xyl
                    matrix_W[cid][cid] -= self.hydraulic.k_xyl
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
                    matrix_W[cid][cid] -= self.hydraulic.k_sieve
                rhs[:] = rhs_s * boundary.scenarios[0]['psi_soil_left'] + rhs_p * psi_sieve[1][i_maturity][0]
            elif not np.isnan(distributed_flow_sieve[1][1][0]):
                for i, cid in enumerate(getattr(self.network, 'protosieve_list', [])):
                    rhs_p[cid][0] = distributed_flow_sieve[1][i+1][0]
                rhs[:] = rhs_s * boundary.scenarios[0]['psi_soil_left'] + rhs_p
            else:
                rhs[:] = rhs_s * boundary.scenarios[0]['psi_soil_left']
