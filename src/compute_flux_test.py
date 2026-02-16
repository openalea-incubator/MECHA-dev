




def _initialize_scenarios(self, i_scenario, i_maturity):

    n_nodes = self.network.n_nodes
    #Initializing the connectivity matrix including boundary conditions
    rhs = np.zeros((n_nodes,1))
    rhs_x = np.zeros((n_nodes,1)) #Initializing the right-hand side matrix of xylem pressure potentials
    rhs_p = np.zeros((n_nodes,1)) #Initializing the right-hand side matrix of hydrostatic potentials for phloem BC
    rhs_e = np.zeros((n_nodes,1)) #Initializing the right-hand side matrix of cell elongation
    rhs_o = np.zeros((n_nodes,1)) #Initializing the right-hand side matrix of osmotic potentials
    os_cells = np.zeros((self.network.n_cells,1)) #Initializing the cell osmotic potential vector
    os_walls = np.zeros((self.network.n_walls,1)) #Initializing the wall osmotic potential vector
    s_membranes = np.zeros((self.network.n_membrane,1)) #Initializing the membrane reflection coefficient vector
    os_membranes = np.zeros((self.network.n_membrane,2)) #Initializing the osmotic potential storage side by side of membranes (0 for the wall, 1 for the protoplast)
    #rhs_s invariable between diferent scenarios but can vary for different hydraulic properties
    
    #Apoplastic & symplastic convective direction matrices initialization
    cell_connec_flow=np.zeros((self.network.n_cells,14),dtype=int) #Flow direction across plasmodesmata, positive when entering the cell, negative otherwise
    apo_connec_flow=np.zeros((self.network.n_wall_junction,5),dtype=int) #Flow direction across cell walls, rows correspond to apoplastic nodes, and the listed nodes in each row receive convective flow from the row node
    n_apo_connec_flow=np.zeros((self.network.n_wall_junction,1),dtype=int)
    
    if self.boundary.c_flag:
        s_membranes, os_membranes = self._initialize_scenarios_c(i_scenario, i_maturity)
        
    
def _initialize_scenarios_c(self, i_scenario, i_maturity):
    
    jmb=0 #Index for membrane conductance vector
    passage_cell_ids = self.geometry.passage_cell_ids
    barrier = int(self.geometry.maturity_stages[i_maturity].get("barrier"))
    for node, edges in self.network.graph.adjacency() : #adjacency_iter returns an iterator of (node, adjacency dict) tuples for all nodes. This is the fastest way to look at every edge. For directed graphs, only outgoing adjacencies are included.
        i=self.indice[node] #Node ID number
        
        for neighboor, eattr in edges.items(): #Loop on connections (edges)
            j = (self.indice[neighboor]) #neighbouring node number
            if j > i: #Only treating the information one way to save time
                path = eattr['path'] #eattr is the edge attribute (i.e. connection type)
                if path == "membrane": #Membrane connection
                    #Cell and wall osmotic potentials (cell types: 1=Exodermis;2=epidermis;3=endodermis;4=cortex;5=stele;16=pericycle)
                    rank=int(self.network.cell_ranks[int(j-self.network.n_wall_junction)])
                    row=int(self.network.rank_to_row[rank][0])
                    if rank==1:#Exodermis
                        os_membranes[jmb][1]=self.boundary.osmotic_potentials[i_scenario].get('Os_exo')
                        if self.network.graph.nodes[node]['count_epi']==1: #wall between exodermis and epidermis
                            s_membranes[jmb]=self.boundary.reflection_coefficients[i_scenario].get('s_exo_epi')
                        elif self.network.graph.nodes[node]['count_epi']==0: #wall between exodermis and cortex or between two exodermal cells
                            s_membranes[jmb]=self.boundary.reflection_coefficients[i_scenario].get('s_exo_cortex')
                    elif rank==2:#Epidermis
                        os_membranes[jmb][1]=self.boundary.osmotic_potentials[i_scenario].get('Os_epi')
                        s_membranes[jmb]=self.boundary.reflection_coefficients[i_scenario].get('s_epi')
                    elif rank==3:#Endodermis
                        os_membranes[jmb][1]=self.boundary.osmotic_potentials[i_scenario].get('Os_endo')
                        if self.network.graph.nodes[node]['count_stele_overall']==0: #wall between endodermis and cortex or between two endodermal cells
                            s_membranes[jmb]=self.boundary.reflection_coefficients[i_scenario].get('s_endo_cortex')
                        elif self.network.graph.nodes[node]['count_stele_overall']>0 and self.network.graph.nodes[node]['count_endo']>0: #wall between endodermis and pericycle
                            s_membranes[jmb]=self.boundary.reflection_coefficients[i_scenario].get('s_endo_peri')
                    elif rank>=40 and rank<50:#Cortex
                        if j-self.network.n_wall_junction in self.geometry.intercellular_ids:
                            os_membranes[jmb][1]=0
                            s_membranes[jmb]=0
                        else:
                            if row==self.network.row_outer_cortex-7:
                                os_membranes[jmb][1]=self.boundary.osmotic_potentials[i_scenario].get('Os_c8')
                            elif row==self.network.row_outer_cortex-6:
                                os_membranes[jmb][1]=self.boundary.osmotic_potentials[i_scenario].get('Os_c7')
                            elif row==self.network.row_outer_cortex-5:
                                os_membranes[jmb][1]=self.boundary.osmotic_potentials[i_scenario].get('Os_c6')
                            elif row==self.network.row_outer_cortex-4:
                                os_membranes[jmb][1]=self.boundary.osmotic_potentials[i_scenario].get('Os_c5')
                            elif row==self.network.row_outer_cortex-3:
                                os_membranes[jmb][1]=self.boundary.osmotic_potentials[i_scenario].get('Os_c4')
                            elif row==self.network.row_outer_cortex-2:
                                os_membranes[jmb][1]=self.boundary.osmotic_potentials[i_scenario].get('Os_c3')
                            elif row==self.network.row_outer_cortex-1:
                                os_membranes[jmb][1]=self.boundary.osmotic_potentials[i_scenario].get('Os_c2')
                            elif row==self.network.row_outer_cortex:
                                os_membranes[jmb][1]=self.boundary.osmotic_potentials[i_scenario].get('Os_c1')
                            s_membranes[jmb]=self.boundary.reflection_coefficients[i_scenario].get('s_cortex')
                    elif self.network.graph.nodes[j]['cgroup']==5:#Stelar parenchyma
                        os_membranes[jmb][1]=self.boundary.osmotic_potentials[i_scenario].get('Os_stele')
                        s_membranes[jmb]=self.boundary.reflection_coefficients[i_scenario].get('s_stele')
                    elif rank==16:#Pericycle
                        os_membranes[jmb][1]=self.boundary.osmotic_potentials[i_scenario].get('Os_peri')
                        s_membranes[jmb]=self.boundary.reflection_coefficients[i_scenario].get('s_peri')
                    elif self.network.graph.nodes[j]['cgroup']==11 or self.network.graph.nodes[j]['cgroup']==23:#Phloem sieve tube cell
                        if not isnan(Os_sieve[0][i_scenario]):
                            if barrier>0 or j in self.network.protosieve_list:
                                os_membranes[jmb][1]=float(Os_sieve[0][i_scenario])
                            else:
                                os_membranes[jmb][1]=self.boundary.osmotic_potentials[i_scenario].get('Os_stele')
                        else:
                            os_membranes[jmb][1]=self.boundary.osmotic_potentials[i_scenario].get('Os_stele')
                        s_membranes[jmb]=self.boundary.reflection_coefficients[i_scenario].get('s_sieve')
                    elif self.network.graph.nodes[j]['cgroup']==12 or self.network.graph.nodes[j]['cgroup']==26:#Companion cell
                        if not isnan(Os_sieve[0][i_scenario]):
                            os_membranes[jmb][1]=self.boundary.osmotic_potentials[i_scenario].get('Os_comp')
                        else:
                            os_membranes[jmb][1]=self.boundary.osmotic_potentials[i_scenario].get('Os_stele')
                        s_membranes[jmb]=self.boundary.reflection_coefficients[i_scenario].get('s_comp')
                    elif self.network.graph.nodes[j]['cgroup']==13 or self.network.graph.nodes[j]['cgroup']==19 or self.network.graph.nodes[j]['cgroup']==20:#Xylem cell or vessel
                        if barrier==0:
                            os_membranes[jmb][1]=self.boundary.osmotic_potentials[i_scenario].get('Os_stele')
                            s_membranes[jmb]=self.boundary.reflection_coefficients[i_scenario].get('s_stele')
                        else:
                            os_membranes[jmb][1]=0.0
                            s_membranes[jmb]=0.0
                    jmb+=1
        
    return s_membranes, os_membranes
# if xylem flow rate is given, calculate the total flow rate and the radial flux
def calculate_radial_flux(self, i_scenario, i_maturity):
    #!TODO: check if xylem flux or potential is prox or dist
    height=self.geometry.maturity_stage[i_maturity].get('height')
    #Soil and xylem water potentials
    # if xylem flow rate is given, calculate the total flow rate and the radial flux
    if not isnan(self.flow_xyl[1][i_scenario]):
        if isnan(self.psi_xyl[1][i_maturity][i_scenario]) and isnan(self.dpsi_xyl[i_maturity][i_scenario]):
            tot_flow=self.flow_xyl[1][i_scenario]
            sum_area=0.0
            for i, cid in enumerate(self.network.xylem_cells):
                area=self.network.cell_areas[cid-self.network.n_wall_junction]
                self.flow_xyl[i+1][i_scenario]=tot_flow*area
                sum_area+=area
            for i, cid in enumerate(self.network.xylem_cells):
                self.flow_xyl[i+1][i_scenario]/=sum_area #Total xylem flow rate partitioned proportionnally to xylem cross-section area
            if self.flow_xyl[1][i_scenario]==0.0:
                self.i_equil_xyl=i_scenario
            if self.boundary.c_flag:
                #Estimate the radial distribution of solutes later on from "u"
                #First estimate water radial velocity in the apoplast
                u=zeros((2,1))
                u[0][0]=tot_flow/(height*1.0E-04)/(self.geometry.thickness*1.0E-04)/self.geometry.cell_per_layer[0][0] #Cortex (cm/d)
                u[1][0]=tot_flow/(height*1.0E-04)/(self.geometry.thickness*1.0E-04)/self.geometry.cell_per_layer[1][0] #Stele (cm/d)
        else:
            print('Error: Cannot have both pressure and flow BC at xylem boundary')
    elif not isnan(self.dpsi_xyl[i_maturity][i_scenario]):
        if isnan(self.psi_xyl[1][i_maturity][i_scenario]):
            self.psi_xyl[1][i_maturity][i_scenario]=self.psi_xyl[1][i_maturity][self.i_equil_xyl]+self.dpsi_xyl[i_maturity][i_scenario]
        else:
            print('Error: Cannot have both pressure and pressure change relative to equilibrium as xylem boundary condition')

# if c_flag is True, calculate the flow rate and the radial flux from the xylem pressure BC
def calculate_flow_psi_xyl(self, i_scenario, i_maturity, s_membranes, os_membranes):

    if not self.boundary.c_flag:
        print('Error: c_flag must be True to calculate flow')
        return
    # if pressure is given, calculate the total flow rate and the radial flux
    if not isnan(self.psi_xyl[1][i_maturity][i_scenario]):
        
        #Estimate the radial distribution of solutes
        #First estimate total flow rate (cm^3/d) from BC & kr
        tot_flow1=0.0
        u=zeros((2,1))
        iter=0
        tot_flow2=self.kr_tot[i_maturity][0]*self.network.perimeter*height*1.0E-04*(self.boundary.scenarios[i_scenario]['psi_soil_left']+self.boundary.scenarios[i_scenario]['osmotic_left_soil']-self.psi_xyl[i_maturity][i_scenario]-self.boundary.scenarios[i_scenario]['osmotic_xyl']) 
        tot_flow1=tot_flow2
        print('flow_rate =',tot_flow2,' iter =',iter)
        #Convergence loop of water radial velocity and solute apoplastic convection-diffusion
        while abs(tot_flow1-tot_flow2)/abs(tot_flow2)>0.001 and iter<30:
            iter+=1
            if iter>1 and sign(tot_flow1/tot_flow2)==1:
                tot_flow1=(tot_flow1+tot_flow2)/2
            else:
                tot_flow1=tot_flow1/2
            #Then estimate water radial velocity in the apoplast
            u[0][0]=tot_flow1/(height*1.0E-04)/(self.geometry.thickness*1.0E-04)/self.geometry.cell_per_layer[0][0] #Cortex apoplastic water velocity (cm/d) positive inwards
            u[1][0]=tot_flow1/(height*1.0E-04)/(self.geometry.thickness*1.0E-04)/self.geometry.cell_per_layer[1][0] #Stele apoplastic water velocity (cm/d) positive inwards
            #Then estimate the radial solute distribution from an analytical solution (C(x)=C0+C0*(exp(u*x/D)-1)/(u/D*exp(u*x/D)-exp(u*L/D)+1)
            os_apo_cortex_eq=0.0
            os_apo_stele_eq=0.0
            os_sym_cortex_eq=0.0
            os_sym_stele_eq=0.0
            jmb=0 #Index for membrane vector
            for node, edges in self.network.graph.adjacency() : #adjacency_iter returns an iterator of (node, adjacency dict) tuples for all nodes. This is the fastest way to look at every edge. For directed graphs, only outgoing adjacencies are included.
                i = indice[node] #Node ID number
                if i<network.n_walls: #wall ID
                    for neighboor, eattr in edges.items(): #Loop on connections (edges)
                        if eattr['path'] == 'membrane': #Wall connection
                            if r_rel[i]>=0: #cortical side
                                os_apo=self.boundary.scenarios[i_scenario]['osmotic_left_soil']*exp(u[0][0]*abs(r_rel[i])*L_diff[0]/self.boundary.scenarios[i_scenario]['osmotic_diffusivity_soil'])
                                os_apo_cortex_eq+=self.STFmb[jmb][i_maturity]*(os_apo*s_membranes[jmb])
                                os_sym_cortex_eq+=self.STFmb[jmb][i_maturity]*(os_membranes[jmb][1]*s_membranes[jmb])
                            else: #Stelar side
                                os_apo=self.boundary.scenarios[i_scenario]['osmotic_xyl']*exp(-u[1][0]*abs(r_rel[i])*L_diff[1]/self.boundary.scenarios[i_scenario]['osmotic_diffusivity_xyl'])
                                os_apo_stele_eq-=self.STFmb[jmb][i_maturity]*(os_apo*s_membranes[jmb])
                                os_sym_stele_eq-=self.STFmb[jmb][i_maturity]*(os_membranes[jmb][1]*s_membranes[jmb])
                            os_membranes[jmb][0]=os_apo
                            jmb+=1
            tot_flow2=self.kr_tot[i_maturity][0]*self.network.perimeter*height*1.0E-04*(self.boundary.scenarios[i_scenario]['psi_soil_left']+os_apo_cortex_eq-os_sym_cortex_eq-self.psi_xyl[1][i_maturity][i_scenario]-os_apo_stele_eq+os_sym_stele_eq)
            print('flow_rate =',tot_flow2,' iter =',iter)
        u[0][0]=tot_flow2/(height*1.0E-04)/(self.geometry.thickness*1.0E-04)/self.geometry.cell_per_layer[0][0] #Cortex (cm/d)
        u[1][0]=tot_flow2/(height*1.0E-04)/(self.geometry.thickness*1.0E-04)/self.geometry.cell_per_layer[1][0] #Stele (cm/d)
        ##Then estimate osmotic potentials in radial walls later on: C(x)=C0+C0*(exp(u*x/D)-1)/(u/D*exp(u*x/D)-exp(u*L/D)+1)
        tot_flow=tot_flow2
    return tot_flow, u

def elongation_BC(self, i_scenario: int, i_maturity: int):
    barrier= self.geometry.maturity_stages[i_maturity].get('barrier')
    #Elongation BC
    if barrier==0: #No elongation from the Casparian strip on
        for wall_id in range(self.network.n_walls):
            rhs_e[wall_id][0]=self.network.wall_lengths[wall_id]*self.geometry.thickness/2*1.0E-08*(Elong_cell[0][count]+(x_rel[wall_id]-0.5)*Elong_cell_side_diff[0][count])*self.boundary.water_fraction_apo #cm^3/d Cell wall horizontal surface assumed to be rectangular (junctions are pointwise elements)
        for cid in range(self.network.n_cells):
            if self.network.cell_areas[cid]>self.network.cell_perimeters[cid]*self.geometry.thickness/2:
                rhs_e[self.network.n_wall_junction+cid][0]=(self.network.cell_areas[cid]-self.network.cell_perimeters[cid]*self.geometry.thickness/2)*1.0E-8*(Elong_cell[0][count]+(x_rel[self.network.n_wall_junction+cid]-0.5)*Elong_cell_side_diff[0][count])*self.boundary.water_fraction_sym #cm^3/d Wall geometry.thickness removed from cell horizontal area to obtain protoplast horizontal area
            else:
                rhs_e[self.network.n_wall_junction+cid][0]=0 #The cell elongation virtually does not imply water influx, though its walls do (typically intercellular spaces
    return rhs_e