
def solver():

    #System solving
    Psi_xyl=empty((geometry.n_maturity, boundary.n_scenarios))
    Psi_xyl[:]=np.nan
    dPsi_xyl=empty((geometry.n_maturity, boundary.n_scenarios))
    dPsi_xyl[:]=np.nan
    iEquil_xyl=np.nan #index of the equilibrium root xylem pressure scenario
    Flow_xyl=empty((len(network.xylem_cells)+ 1, boundary.n_scenarios))
    Flow_xyl[:]=np.nan
    Psi_sieve=empty((geometry.n_maturity, boundary.n_scenarios))
    Psi_sieve[:]=np.nan
    dPsi_sieve=empty((geometry.n_maturity, boundary.n_scenarios))
    dPsi_sieve[:]=np.nan
    iEquil_sieve=np.nan #index of the equilibrium root phloem pressure scenario
    Flow_sieve=empty((network.n_sieve+1, boundary.n_scenarios))
    Flow_sieve[:]=np.nan
    Os_sieve=zeros((1, boundary.n_scenarios))
    Os_cortex=zeros((1, boundary.n_scenarios))
    Os_hetero=zeros((1, boundary.n_scenarios))
    s_factor=zeros((1, boundary.n_scenarios))
    s_hetero=zeros((1, boundary.n_scenarios))
    Elong_cell=zeros((1, boundary.n_scenarios))
    Elong_cell_side_diff=zeros((1, boundary.n_scenarios))
    UptakeLayer_plus=zeros((int(network.r_discret[0]), geometry.n_maturity, boundary.n_scenarios))
    UptakeLayer_minus=zeros((int(network.r_discret[0]), geometry.n_maturity, boundary.n_scenarios))
    Q_xyl_layer=zeros((int(network.r_discret[0]), geometry.n_maturity, boundary.n_scenarios))
    Q_sieve_layer=zeros((int(network.r_discret[0]), geometry.n_maturity, boundary.n_scenarios))
    Q_elong_layer=zeros((int(network.r_discret[0]), geometry.n_maturity, boundary.n_scenarios))
    STFmb=zeros((network.n_membrane, geometry.n_maturity))
    STFcell_plus=zeros((network.n_cells, geometry.n_maturity))
    STFcell_minus=zeros((network.n_cells, geometry.n_maturity))
    STFlayer_plus=zeros((int(network.r_discret[0]), geometry.n_maturity))
    STFlayer_minus=zeros((int(network.r_discret[0]), geometry.n_maturity))
    PsiCellLayer=zeros((int(network.r_discret[0]), geometry.n_maturity, boundary.n_scenarios))
    PsiWallLayer=zeros((int(network.r_discret[0]), geometry.n_maturity, boundary.n_scenarios))
    OsCellLayer=zeros((int(network.r_discret[0]), geometry.n_maturity, boundary.n_scenarios))
    nOsCellLayer=zeros((int(network.r_discret[0]), geometry.n_maturity, boundary.n_scenarios))
    OsWallLayer=zeros((int(network.r_discret[0]), geometry.n_maturity, boundary.n_scenarios))
    nOsWallLayer=zeros((int(network.r_discret[0]), geometry.n_maturity, boundary.n_scenarios)) #Used for averaging OsWallLayer
    NWallLayer=zeros((int(network.r_discret[0]), geometry.n_maturity, boundary.n_scenarios))
    #UptakeDistri_plus=zeros((40,3,8))#the size will be adjusted, but won't be more than 40. Dimension 1: radial position, 2: compartment, 3: scenario
    #UptakeDistri_minus=zeros((40,3,8))
    Q_tot=zeros((geometry.n_maturity, boundary.n_scenarios)) #(cm^3/d) Total flow rate at root surface
    kr_tot=zeros((geometry.n_maturity, 1))
    Hydropatterning=empty((geometry.n_maturity, boundary.n_scenarios))
    Hydropatterning[:]=np.nan
    Hydrotropism=empty((geometry.n_maturity, boundary.n_scenarios))
    Hydrotropism[:]=np.nan


    Barrier=int(Maturity.get("Barrier")) #Apoplastic barriers (0: No apoplastic barrier, 1:Endodermis radial walls, 2:Endodermis with passage cells, 3: Endodermis full, 4: Endodermis full and exodermis radial walls)
    height=int(Maturity.get("height")) #Cell length in the axial direction (microns)
    
    #Index for barriers loop
    iMaturity+=1
    print('Maturity #'+str(iMaturity)+' with apoplastic barrier type #'+str(Barrier))
    
    #Soil, xylem, and phloem pressure potentials
    Psi_xyl[iMaturity][0]=float(boundary.bc_xyl_elems[0].get("pressure")) #Xylem pressure potential (hPa)
    dPsi_xyl[iMaturity][0]=float(boundary.bc_xyl_elems[0].get("deltaP")) #Xylem pressure potential change as compared to equilibrium pressure (hPa)
    Flow_xyl[0][0]=float(boundary.bc_xyl_elems[0].get("flowrate")) #Xylem flow rate (cm^3/d)
    if not isnan(Flow_xyl[0][0]):
        if isnan(Psi_xyl[iMaturity][0]) and isnan(dPsi_xyl[iMaturity][0]):
            tot_flow=Flow_xyl[0][0]
            sum_area=0
            i=1
            for cid in network.xylem_cells:
                area=network.cell_areas[cid-network.n_wall_junction]
                Flow_xyl[i][0]=tot_flow*area
                sum_area+=area
                i+=1
            i=1
            for cid in network.xylem_cells:
                Flow_xyl[i][0]/=sum_area #Total xylem flow rate partitioned proportionnally to xylem cross-section area
                i+=1
            if Flow_xyl[0][0]==0.0:
                iEquil_xyl=0
        else:
            print('Error: Cannot have both pressure and flow BC at xylem boundary')
    elif not isnan(dPsi_xyl[iMaturity][0]):
        if isnan(Psi_xyl[iMaturity][0]):
            if not isnan(iEquil_xyl):
                Psi_xyl[iMaturity][0]=Psi_xyl[iMaturity][iEquil_xyl]+dPsi_xyl[iMaturity][0]
            else:
                print('Error: Cannot have xylem pressure change relative to equilibrium without having a prior scenario with equilibrium xylem boundary condition')
        else:
            print('Error: Cannot have both pressure and pressure change relative to equilibrium as xylem boundary condition')
    
    Psi_sieve[iMaturity][0]=float(boundary.bc_sieve_elems[0].get("pressure")) #Phloem sieve element pressure potential (hPa)
    dPsi_sieve[iMaturity][0]=float(boundary.bc_sieve_elems[0].get("deltaP")) #Phloem pressure potential change as compared to equilibrium pressure (hPa)
    Flow_sieve[0][0]=float(boundary.bc_sieve_elems[0].get("flowrate")) #Phloem flow rate (cm^3/d)
    if not isnan(Flow_sieve[0][0]):
        if isnan(Psi_sieve[iMaturity][0]) and isnan(dPsi_sieve[iMaturity][0]):
            tot_flow=Flow_sieve[0][0]
            sum_area=0
            i=1
            for cid in network.protosieve_list:
                area=network.cell_areas[cid-network.n_wall_junction]
                Flow_sieve[i][0]=tot_flow*area
                sum_area+=area
                i+=1
            i=1
            for cid in network.protosieve_list:
                Flow_sieve[i][0]/=sum_area #Total phloem flow rate partitioned proportionnally to phloem cross-section area
                i+=1
            if Flow_sieve[0][0]==0.0:
                iEquil_sieve=0
        else:
            print('Error: Cannot have both pressure and flow BC at phloem boundary')
    elif not isnan(dPsi_sieve[iMaturity][0]):
        if isnan(Psi_sieve[iMaturity][0]):
            if not isnan(iEquil_sieve):
                Psi_sieve[iMaturity][0]=Psi_sieve[iMaturity][iEquil_sieve]+dPsi_sieve[iMaturity][0]
            else:
                print('Error: Cannot have phloem pressure change relative to equilibrium without having a prior scenario with equilibrium phloem boundary condition')
        else:
            print('Error: Cannot have both pressure and pressure change relative to equilibrium as phloem boundary condition')
    
    #Soil - root contact limit
    if hydraulic.n_xcontact == hydraulic.n_hydraulics:
        Xcontact=float(hydraulic.xcontactrange[h].get("value")) #(micrometers) X threshold coordinate of contact between soil and root (lower X not in contact with soil)
    elif hydraulic.n_xcontact == 1:
        Xcontact=float(hydraulic.xcontactrange[0].get("value"))
    else:
        Xcontact=float(hydraulic.xcontactrange[int(h/(hydraulic.n_kaqp*hydraulic.n_kpl*hydraulic.n_kw*hydraulic.n_kw_barrier))].get("value")) #OK
    
    #Cell wall hydraulic conductivity
    if hydraulic.n_kw == hydraulic.n_hydraulics:
        kw = float(hydraulic.kw_elems[h].get("value"))
    elif hydraulic.n_kw == 1:
        kw = float(hydraulic.kw_elems[0].get("value"))
    else:
        kw = float(hydraulic.kw_elems[int(h/(hydraulic.n_kaqp*hydraulic.n_kpl))%hydraulic.n_kw].get("value"))
    if hydraulic.n_kw_barrier == hydraulic.n_hydraulics:
        kw_barrier = float(hydraulic.kw_barrier_elems[h].get("value"))
    elif hydraulic.n_kw_barrier == 1:
        kw_barrier = float(hydraulic.kw_barrier_elems[0].get("value"))
    else:
        kw_barrier = float(hydraulic.kw_barrier_elems[int(h/(hydraulic.n_kaqp*hydraulic.n_kpl*hydraulic.n_kw))%hydraulic.n_kw_barrier].get("value"))
    #kw_barrier = kw/10.0
    if Barrier==0: #No Casparian strip ###Yet to come: Punctured Casparian strip as in Steudle et al. (1993)
        kw_endo_endo=kw
        kw_puncture=kw
        kw_exo_exo=kw #(cm^2/hPa/d) hydraulic conductivity of the suberised walls between exodermis cells
        kw_cortex_cortex=kw
        kw_endo_peri=kw #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
        kw_endo_cortex=kw #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
        kw_passage=kw #(cm^2/hPa/d) hydraulic conductivity of passage cells tangential walls
    elif Barrier==1: #Endodermis radial walls
        kw_endo_endo=kw_barrier
        kw_exo_exo=kw #(cm^2/hPa/d) hydraulic conductivity of the suberised walls between exodermis cells
        kw_cortex_cortex=kw
        kw_endo_peri=kw #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
        kw_endo_cortex=kw #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
        kw_passage=kw #(cm^2/hPa/d) hydraulic conductivity of passage cells tangential walls
    elif Barrier==2: #Endodermis with passage cells
        kw_endo_endo=kw_barrier
        kw_exo_exo=kw #(cm^2/hPa/d) hydraulic conductivity of the suberised walls between exodermis cells
        kw_cortex_cortex=kw
        kw_endo_peri=kw_barrier #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
        kw_endo_cortex=kw_barrier #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
        kw_passage=kw #(cm^2/hPa/d) hydraulic conductivity of passage cells tangential walls
    elif Barrier==3: #Endodermis full
        kw_endo_endo=kw_barrier
        kw_exo_exo=kw #(cm^2/hPa/d) hydraulic conductivity of the suberised walls between exodermis cells
        kw_cortex_cortex=kw
        kw_endo_peri=kw_barrier #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
        kw_endo_cortex=kw_barrier #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
        kw_passage=kw_barrier #(cm^2/hPa/d) hydraulic conductivity of passage cells tangential walls
    elif Barrier==4: #Endodermis full and exodermis radial walls
        kw_endo_endo=kw_barrier
        kw_exo_exo=kw_barrier #(cm^2/hPa/d) hydraulic conductivity of the suberised walls between exodermis cells
        kw_cortex_cortex=kw
        kw_endo_peri=kw_barrier #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
        kw_endo_cortex=kw_barrier #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
        kw_passage=kw_barrier #(cm^2/hPa/d) hydraulic conductivity of passage cells tangential walls
    elif Barrier==5: # Endodermal & exodermal Casparian strips
        kw_endo_endo=kw_barrier[0,0]
        kw_exo_exo=kw_barrier[0,0] #(cm^2/hPa/d) hydraulic conductivity of the suberised walls between exodermis cells
        kw_exo_epi=kw
        kw_exo_cortex=kw
        kw_cortex_cortex=kw
        kw_endo_peri=kw #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
        kw_endo_cortex=kw #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
        kw_passage=kw #(cm^2/hPa/d) hydraulic conductivity of passage cells tangential walls
    elif Barrier==6: #Exodermis full and endodermis radial walls
        kw_endo_endo=kw_barrier[0,0]
        kw_exo_exo=kw_barrier[0,0] #(cm^2/hPa/d) hydraulic conductivity of the suberised walls between exodermis cells
        kw_exo_epi=kw_barrier[0,1]
        kw_exo_cortex=kw_barrier[0,1]
        kw_cortex_cortex=kw
        kw_endo_peri=kw #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
        kw_endo_cortex=kw #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
        kw_passage=kw #(cm^2/hPa/d) hydraulic conductivity of passage cells tangential walls
    elif Barrier==7: #Exodermis radial walls
        kw_endo_endo=kw
        kw_exo_exo=kw_barrier[0,0] #(cm^2/hPa/d) hydraulic conductivity of the suberised walls between exodermis cells
        kw_exo_epi=kw
        kw_exo_cortex=kw
        kw_cortex_cortex=kw
        kw_endo_peri=kw #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
        kw_endo_cortex=kw #(cm^2/hPa/d) hydraulic conductivity of the walls between endodermis and pericycle cells
        kw_passage=kw #(cm^2/hPa/d) hydraulic conductivity of passage cells tangential walls
    
    #Plasmodesmatal hydraulic conductance
    if hydraulic.n_kpl == hydraulic.n_hydraulics:
        iPD=h
    elif hydraulic.n_kpl == 1:
        iPD=0
    else:
        iPD=int(h/hydraulic.n_kaqp)%hydraulic.n_kpl
    Kpl = float(hydraulic.kpl_elems[iPD].get("value"))
    
    #Contribution of aquaporins to membrane hydraulic conductivity
    if hydraulic.n_kaqp == hydraulic.n_hydraulics:
        iAQP=h
    elif hydraulic.n_kaqp == 1:
        iAQP=0
    else:
        iAQP=h%hydraulic.n_kaqp
    kaqp = float(hydraulic.kaqp_elems[iAQP].get("value"))
    kaqp_stele= kaqp*float(hydraulic.kaqp_elems[iAQP].get("stele_factor"))
    kaqp_endo= kaqp*float(hydraulic.kaqp_elems[iAQP].get("endo_factor"))
    kaqp_exo= kaqp*float(hydraulic.kaqp_elems[iAQP].get("exo_factor"))
    kaqp_epi= kaqp*float(hydraulic.kaqp_elems[iAQP].get("epi_factor"))
    kaqp_cortex= kaqp*float(hydraulic.kaqp_elems[iAQP].get("cortex_factor"))
    
    #Calculate parameter a
    if hydraulic.ratio_cortex==1: #Uniform AQP activity in all cortex membranes
        a_cortex=0.0  #(1/hPa/d)
        b_cortex=kaqp_cortex #(cm/hPa/d)
    else:
        tot_surf_cortex=0.0 #Total membrane exchange surface in cortical cells (square centimeters)
        temp=0.0 #Term for summation (cm3)
        for cell_group in network.cellset['cell_to_wall']: #Loop on cells. network.cellset['cell_to_wall'] contains cell wall groups info (one group by cell)
            cell_id = int(cell_group.getparent().get("id")) #Cell ID number
            for r in cell_group: #Loop for wall elements around the cell
                wall_id= int(r.get("id")) #Cell wall ID
                if network.graph.nodes[network.n_wall_junction + cell_id]['cgroup']==4: #Cortex
                    dist_cell=sqrt(square(position[wall_id][0]-position[network.n_wall_junction+cell_id][0])+square(position[wall_id][1]-position[network.n_wall_junction+cell_id][1])) #distance between wall node and cell node (micrometers)
                    surf=(height+dist_cell)*network.wall_lengths[wall_id]*1.0E-08 #(square centimeters)
                    temp+=surf*1.0E-04*(network.distance_center_grav[wall_id]+(hydraulic.ratio_cortex*dmax_cortex-dmin_cortex)/(1-hydraulic.ratio_cortex))
                    tot_surf_cortex+=surf
        a_cortex=kaqp_cortex*tot_surf_cortex/temp  #(1/hPa/d)
        b_cortex=a_cortex*1.0E-04*(hydraulic.ratio_cortex*dmax_cortex-dmin_cortex)/(1-hydraulic.ratio_cortex) #(cm/hPa/d)
    
    ######################
    ##Filling the matrix##
    ######################
    
    matrix_W = np.zeros(((network.graph.number_of_nodes()),network.graph.number_of_nodes())) #Initializes the Doussan matrix
    if general.apo_contagion==2 and general.sym_contagion==2:
        matrix_C = np.zeros(((network.graph.number_of_nodes()),network.graph.number_of_nodes())) #Initializes the matrix of convection diffusion
        rhs_C = np.zeros((network.graph.number_of_nodes(),1)) #Initializing the right-hand side matrix of solute apoplastic concentrations
        for i in range(network.n_walls):
            if i in network.apo_wall_zombies0:
                matrix_C[i][i]=1.0
                rhs_C[i][0]=network.apo_wall_cc[network.apo_wall_zombies0.index(i)] #1.0 #Concentration in source wall i defined in geometry_config
            else: #Decomposition rate (mol decomp/mol-day * cm^3)
                matrix_C[i][i]-=hormones.degrad1*1.0E-12*(network.distance_wall_cell[i][0]*geometry.thickness*network.wall_lengths[i]+height*geometry.thickness*network.wall_lengths[i]/2-square(geometry.thickness)*network.wall_lengths[i])
        for j in range(network.n_walls,network.n_wall_junction):
            if j in Apo_j_Zombies0:
                matrix_C[j][j]=1.0
                rhs_C[j][0]=Apo_j_cc[Apo_j_Zombies0.index(j)] #1.0 #Concentration in source junction j defined in geometry_config
            else: #Decomposition rate (mol decomp/mol-day * cm^3)
                matrix_C[j][j]-=hormones.degrad1*1.0E-12*height*geometry.thickness*network.wall_lengths[j]/2
        for cell_id in range(network.n_cells):
            if cell_id in hormones.sym_zombie0:
                matrix_C[network.n_wall_junction+cell_id][network.n_wall_junction+cell_id]=1.0
                rhs_C[network.n_wall_junction+cell_id][0]=hormones.sym_cc[hormones.sym_zombie0.index(cell_id)] #1.0 #Concentration in source protoplasts defined in geometry_config
            else: #Decomposition rate (mol decomp/mol-day * cm^3)
                matrix_C[network.n_wall_junction+cell_id][network.n_wall_junction+cell_id]-=hormones.degrad1*1.0E-12*network.cell_areas[cell_id]*height
    elif general.apo_contagion==2:
        matrix_ApoC = np.zeros(((network.n_wall_junction),network.n_wall_junction)) #Initializes the matrix of convection
        rhs_ApoC = np.zeros((network.n_wall_junction,1)) #Initializing the right-hand side matrix of solute apoplastic concentrations
        for i in range(network.n_walls):
            if i in network.apo_wall_zombies0:
                matrix_ApoC[i][i]=1.0
                rhs_ApoC[i][0]=network.apo_wall_cc[network.apo_wall_zombies0.index(i)] #1 #Concentration in source wall i equals 1 by default
            else: #Decomposition rate (mol decomp/mol-day * cm^3)
                matrix_ApoC[i][i]-=hormones.degrad1*1.0E-12*(network.distance_wall_cell[i][0]*geometry.thickness*network.wall_lengths[i]+height*geometry.thickness*network.wall_lengths[i]/2-square(geometry.thickness)*network.wall_lengths[i])
        for j in range(network.n_walls,network.n_wall_junction):
            if j in Apo_j_Zombies0:
                matrix_ApoC[j][j]=1.0
                rhs_ApoC[j][0]=Apo_j_cc[Apo_j_Zombies0.index(j)] #1 #Concentration in source junction j equals 1 by default
            else: #Decomposition rate (mol decomp/mol-day * cm^3)
                matrix_ApoC[j][j]-=hormones.degrad1*1.0E-12*height*geometry.thickness*network.wall_lengths[j]/2
    elif general.sym_contagion==2:
        matrix_SymC = np.zeros(((network.n_cells),network.n_cells)) #Initializes the matrix of convection
        rhs_SymC = np.zeros((network.n_cells,1)) #Initializing the right-hand side matrix of solute symplastic concentrations
        for cell_id in range(network.n_cells):
            if cell_id in hormones.sym_zombie0:
                matrix_SymC[cell_id][cell_id]=1.0
                rhs_SymC[cell_id][0]=hormones.sym_cc[hormones.sym_zombie0.index(cell_id)] #1 #Concentration in source protoplasts equals 1 by default
            else: #Decomposition rate (mol decomp/mol-day * cm^3)
                matrix_SymC[cell_id][cell_id]-=hormones.degrad1*1.0E-12*network.cell_areas[cell_id]*height
    
    Kmb=zeros((network.n_membrane,1)) #Stores membranes conductances for the second K loop
    jmb=0 #Index of membrane in Kmb
    K_axial=zeros((network.n_cells + network.n_walls + network.n_junctions,1)) #Vector of apoplastic and plasmodesmatal axial conductances
    if Barrier>0: #K_xyl_spec calculated from Poiseuille law (cm^3/hPa/d)
        for cid in network.xylem_cells:
            K_axial[cid]=network.cell_areas[cid-network.n_wall_junction]**2/(8*3.141592*height*1.0E-05/3600/24)*1.0E-12 #(micron^4/micron)->(cm^3) & (1.0E-3 Pa.s)->(1.0E-05/3600/24 hPa.d) 
        K_xyl_spec=sum(K_axial)*height/1.0E04
        for cid in network.sieve_cells:
            K_axial[cid]=network.cell_areas[cid-network.n_wall_junction]**2/(8*3.141592*height*1.0E-05/3600/24)*1.0E-12 #(micron^4/micron)->(cm^3) & (1.0E-3 Pa.s)->(1.0E-05/3600/24 hPa.d) 
    else:
        K_xyl_spec=0.0
    list_ghostwalls=[] #"Fake walls" not to be displayed
    list_ghostjunctions=[] #"Fake junctions" not to be displayed
    nGhostJunction2Wall=0
    #Adding matrix components at cell-cell, cell-wall, and wall-junction connections
    for node, edges in network.graph.adjacency() : #adjacency_iter returns an iterator of (node, adjacency dict) tuples for all nodes. This is the fastest way to look at every edge. For directed graphs, only outgoing adjacencies are included.
        i=indice[node] #Node ID number
        #Here we count surrounding cell types in order to position apoplastic barriers
        count_endo=0 #total number of endodermis cells around the wall
        count_xyl=0 #total number of xylem cells around the wall
        count_stele_overall=0 #total number of stelar cells around the wall
        count_exo=0 #total number of exodermis cells around the wall
        count_epi=0 #total number of epidermis cells around the wall
        count_cortex=0 #total number of cortical cells around the wall
        count_passage=0 #total number of passage cells around the wall
        count_interC=0 #total number of intercellular spaces around the wall
        if i<network.n_walls: #wall ID
            for neighboor, eattr in edges.items(): #Loop on connections (edges)
                if eattr['path'] == 'membrane': #Wall connection
                    if any(passage_cell_ID==array((indice[neighboor])-network.n_wall_junction)):
                        count_passage+=1
                    if any(geometry.intercellular_ids==array((indice[neighboor])-network.n_wall_junction)):
                        count_interC+=1
                        if count_interC==2 and i not in list_ghostwalls:
                            list_ghostwalls.append(i)
                    if network.graph.nodes[neighboor]['cgroup']==3:#Endodermis
                        count_endo+=1
                    elif network.graph.nodes[neighboor]['cgroup']==13 or network.graph.nodes[neighboor]['cgroup']==19 or network.graph.nodes[neighboor]['cgroup']==20:#Xylem cell or vessel
                        count_xyl+=1
                        if (count_xyl==2 and geometry.xylem_pieces) and i not in list_ghostwalls:
                            list_ghostwalls.append(i)
                    elif network.graph.nodes[neighboor]['cgroup']>4:#Pericycle or stele but not xylem
                        count_stele_overall+=1
                    elif network.graph.nodes[neighboor]['cgroup']==4:#Cortex
                        count_cortex+=1
                    elif network.graph.nodes[neighboor]['cgroup']==1:#Exodermis
                        count_exo+=1
                    elif network.graph.nodes[neighboor]['cgroup']==2:#Epidermis
                        count_epi+=1
        
        for neighboor, eattr in edges.items(): #Loop on connections (edges)
            j = (indice[neighboor]) #neighbouring node number
            if j > i: #Only treating the information one way to save time
                path = eattr['path'] #eattr is the edge attribute (i.e. connection type)
                if path == 'wall': #Wall connection
                    #K = eattr['kw']*1.0E-04*((eattr['lateral_distance']+height)*eattr['geometry.thickness']-square(eattr['geometry.thickness']))/eattr['length'] #Junction-Wall conductance (cm^3/hPa/d)
                    temp=1.0E-04*((eattr['lateral_distance']+height)*geometry.thickness-square(geometry.thickness))/eattr['length'] #Wall section to length ratio (cm)
                    if (count_interC>=2 and Barrier>0) or (count_xyl==2 and geometry.xylem_pieces): #"Fake wall" splitting an intercellular space or a xylem cell in two
                        K = 1.0E-16 #Non conductive
                        if j not in list_ghostjunctions:
                            fakeJ=True
                            for ind in range(int(network.n_junction_to_wall[j-network.n_walls])):
                                if network.junction_to_wall[j-network.n_walls][ind] not in list_ghostwalls:
                                    fakeJ=False #If any of the surrounding walls is real, the junction is real
                            if fakeJ:
                                list_ghostjunctions.append(j)
                                nGhostJunction2Wall+=int(network.n_junction_to_wall[j-network.n_walls])+2 #The first and second thick junction nodes each appear twice in the text file for general.paraview
                    elif count_cortex>=2: #wall between two cortical cells
                        K = kw_cortex_cortex*temp #Junction-Wall conductance (cm^3/hPa/d)
                    elif count_endo>=2: #wall between two endodermis cells
                        K = kw_endo_endo*temp #Junction-Wall conductance (cm^3/hPa/d)  #(height*eattr['geometry.thickness'])/eattr['length']#
                    elif count_stele_overall>0 and count_endo>0: #wall between endodermis and pericycle
                        if count_passage>0:
                            K = kw_passage*temp #(height*eattr['geometry.thickness'])/eattr['length']#
                        else:
                            K = kw_endo_peri*temp #Junction-Wall conductance (cm^3/hPa/d) #(height*eattr['geometry.thickness'])/eattr['length']#
                    elif count_stele_overall==0 and count_endo==1: #wall between endodermis and cortex
                        if count_passage>0:
                            K = kw_passage*temp  #(height*eattr['geometry.thickness'])/eattr['length']#
                        else:
                            K = kw_endo_cortex*temp #Junction-Wall conductance (cm^3/hPa/d)  #(height*eattr['geometry.thickness'])/eattr['length']#
                    elif count_exo>=2: #wall between two exodermis cells
                        K = kw_exo_exo*temp #Junction-Wall conductance (cm^3/hPa/d)  #(height*eattr['geometry.thickness'])/eattr['length']#
                    else: #other walls
                        K = kw*temp #Junction-Wall conductance (cm^3/hPa/d)  #(height*eattr['geometry.thickness'])/eattr['length']#
                    ########Solute fluxes (diffusion across walls and junctions)
                    if general.apo_contagion==2:
                        temp_factor=1.0 #Factor for reduced diffusion across impermeable walls
                        if (count_interC>=2 and Barrier>0) or (count_xyl==2 and geometry.xylem_pieces): #"fake wall" splitting an intercellular space or a xylem cell in two
                            temp_factor=1.0E-16 #Correction
                        elif count_endo>=2:
                            temp_factor=kw_endo_endo/kw
                        elif count_stele_overall>0 and count_endo>0: #wall between endodermis and pericycle
                            if count_passage>0:
                                temp_factor=kw_passage/kw #(height*eattr['geometry.thickness'])/eattr['length']#
                            else:
                                temp_factor=kw_endo_peri/kw #Junction-Wall conductance (cm^3/hPa/d) #(height*eattr['geometry.thickness'])/eattr['length']#
                        elif count_stele_overall==0 and count_endo==1: #wall between endodermis and cortex
                            if count_passage>0:
                                temp_factor=kw_passage/kw  #(height*eattr['geometry.thickness'])/eattr['length']#
                            else:
                                temp_factor=kw_endo_cortex/kw #Junction-Wall conductance (cm^3/hPa/d)  #(height*eattr['geometry.thickness'])/eattr['length']#
                        elif count_exo>=2: #wall between two exodermis cells
                            temp_factor=kw_exo_exo/kw #Junction-Wall conductance (cm^3/hPa/d)  #(height*eattr['geometry.thickness'])/eattr['length']#
                        DF=temp*temp_factor*hormones.diff1_pw1 #"Diffusive flux" (cm^3/d) temp is the section to length ratio of the wall to junction path
                        if general.sym_contagion==2: #Sym & Apo contagion
                            if i not in network.apo_wall_zombies0:
                                matrix_C[i][i] -= DF
                                matrix_C[i][j] += DF #Convection will be dealt with further down
                            if j not in Apo_j_Zombies0:
                                matrix_C[j][j] -= DF #temp_factor is the factor for reduced diffusion across impermeable walls
                                matrix_C[j][i] += DF
                        else: #Only Apo contagion
                            if i not in network.apo_wall_zombies0:
                                matrix_ApoC[i][i] -= DF
                                matrix_ApoC[i][j] += DF
                            if j not in Apo_j_Zombies0:
                                matrix_ApoC[j][j] -= DF #Convection will be dealt with further down
                                matrix_ApoC[j][i] += DF
                elif path == "membrane": #Membrane connection
                    #K = (eattr['hydraulic.kmb']+eattr['kaqp'])*1.0E-08*(height+eattr['dist'])*eattr['length']
                    if general.apo_contagion==2 and general.sym_contagion==2:
                        for carrier in hormones.carrier_elems:
                            if int(carrier.get("tissue"))==network.graph.nodes[j]['cgroup']:
                                #Condition is that the protoplast (j) is an actual protoplast with membranes
                                if j-network.n_wall_junction not in geometry.intercellular_ids and not (Barrier>0 and (network.graph.nodes[j]['cgroup']==13 or network.graph.nodes[j]['cgroup']==19 or network.graph.nodes[j]['cgroup']==20)):
                                    temp=float(carrier.get("constant"))*(height+eattr['dist'])*eattr['length'] #Linear transport constant (Vmax/KM) [liter/day^-1/micron^-2] * membrane surface [micron²]
                                    if int(carrier.get("direction"))==1: #Influx transporter
                                        if j-network.n_wall_junction not in hormones.sym_zombie0: #Concentration not affected if set as boundary condition
                                            matrix_C[j][i] += temp #Increase of concentration in protoplast (j) depends on concentration in cell wall (i)
                                        if i not in network.apo_wall_zombies0: #Concentration not affected if set as boundary condition
                                            matrix_C[i][i] -= temp #Decrease of concentration in apoplast (i) depends on concentration in apoplast (i)
                                    elif int(carrier.get("direction"))==int(-1): #Efflux transporter
                                        if j-network.n_wall_junction not in hormones.sym_zombie0: #Concentration not affected if set as boundary condition
                                            matrix_C[j][j] -= temp #Increase of concentration in protoplast (j) depends on concentration in protoplast (j)
                                        if i not in network.apo_wall_zombies0: #Concentration not affected if set as boundary condition
                                            matrix_C[i][j] += temp #Decrease of concentration in apoplast (i) depends on concentration in protoplast (j)
                                    else:
                                        error('Error, carrier direction is either 1 (influx) or -1 (efflux), please correct in *_Hormones_Carriers_*.xml')
                    if network.graph.nodes[j]['cgroup']==1: #Exodermis
                        kaqp=kaqp_exo
                    elif network.graph.nodes[j]['cgroup']==2: #Epidermis
                        kaqp=kaqp_epi
                    elif network.graph.nodes[j]['cgroup']==3: #Endodermis
                        kaqp=kaqp_endo
                    elif network.graph.nodes[j]['cgroup']==13 or network.graph.nodes[j]['cgroup']==19 or network.graph.nodes[j]['cgroup']==20: #xylem cell or vessel
                        if Barrier>0: #Xylem vessel
                            kaqp=kaqp_stele*10000 #No membrane resistance because no membrane
                            if general.apo_contagion==2 and general.sym_contagion==2:
                                #Diffusion between mature xylem vessels and their walls
                                temp=1.0E-04*(network.wall_lengths[i]*height)/geometry.thickness #Section to length ratio (cm) for the xylem wall
                                if i not in network.apo_wall_zombies0:
                                    matrix_C[i][i] -= temp*hormones.diff1_pw1
                                    matrix_C[i][j] += temp*hormones.diff1_pw1
                                if j-network.n_wall_junction not in hormones.sym_zombie0: #Mature xylem vessels are referred to as cells, so they are on the Sym side even though they are part of the apoplast
                                    matrix_C[j][j] -= temp*hormones.diff1_pw1
                                    matrix_C[j][i] += temp*hormones.diff1_pw1
                        else:
                            kaqp=kaqp_stele
                    elif network.graph.nodes[j]['cgroup']>4: #Stele and pericycle but not xylem
                        kaqp=kaqp_stele
                    elif (j-network.n_wall_junction in geometry.intercellular_ids) and Barrier>0: #the neighbour is an intercellular space "cell". Between j and i connected by a membrane, only j can be cell because j>i
                        kaqp=geometry.k_interc
                        #No carrier
                    elif network.graph.nodes[j]['cgroup']==4: #Cortex
                        kaqp=float(a_cortex*network.distance_center_grav[i][0]*1.0E-04+b_cortex) #AQP activity (cm/hPa/d)
                        if kaqp < 0:
                            error('Error, negative kaqp in cortical cell, adjust Paqp_cortex')
                    #Calculating each conductance
                    if count_endo>=2: #wall between two endodermis cells, in this case the suberized wall can limit the transfer of water between cell and wall
                        if kw_endo_endo==0.00:
                            K=0.00
                        else:
                            K = 1/(1/(kw_endo_endo/(geometry.thickness/2*1.0E-04))+1/(hydraulic.kmb+kaqp))*1.0E-08*(height+eattr['dist'])*eattr['length']
                    elif count_exo>=2: #wall between two exodermis cells, in this case the suberized wall can limit the transfer of water between cell and wall
                        if kw_exo_exo==0.00:
                            K=0.00
                        else:
                            K = 1/(1/(kw_exo_exo/(geometry.thickness/2*1.0E-04))+1/(hydraulic.kmb+kaqp))*1.0E-08*(height+eattr['dist'])*eattr['length']
                    elif count_stele_overall>0 and count_endo>0: #wall between endodermis and pericycle, in this case the suberized wall can limit the transfer of water between cell and wall
                        if count_passage>0:
                            K = 1/(1/(kw_passage/(geometry.thickness/2*1.0E-04))+1/(hydraulic.kmb+kaqp))*1.0E-08*(height+eattr['dist'])*eattr['length']
                        else:
                            if kw_endo_peri==0.00:
                                K=0.00
                            else:
                                K = 1/(1/(kw_endo_peri/(geometry.thickness/2*1.0E-04))+1/(hydraulic.kmb+kaqp))*1.0E-08*(height+eattr['dist'])*eattr['length']
                    elif count_stele_overall==0 and count_endo==1: #wall between cortex and endodermis, in this case the suberized wall can limit the transfer of water between cell and wall
                        if kaqp==0.0:
                            K=1.00E-16
                        else:
                            if count_passage>0:
                                K = 1/(1/(kw_passage/(geometry.thickness/2*1.0E-04))+1/(hydraulic.kmb+kaqp))*1.0E-08*(height+eattr['dist'])*eattr['length']
                            else:
                                if kw_endo_cortex==0.00:
                                    K=0.00
                                else:
                                    K = 1/(1/(kw_endo_cortex/(geometry.thickness/2*1.0E-04))+1/(hydraulic.kmb+kaqp))*1.0E-08*(height+eattr['dist'])*eattr['length']
                    else:
                        if kaqp==0.0:
                            K=1.00E-16
                        else:
                            K = 1/(1/(kw/(geometry.thickness/2*1.0E-04))+1/(hydraulic.kmb+kaqp))*1.0E-08*(height+eattr['dist'])*eattr['length']
                    Kmb[jmb]=K
                    jmb+=1
                elif path == "plasmodesmata": #Plasmodesmata connection
                    cgroupi=network.graph.nodes[i]['cgroup']
                    cgroupj=network.graph.nodes[j]['cgroup']
                    if cgroupi==19 or cgroupi==20:  #Xylem in new Cellset version
                        cgroupi=13
                    elif cgroupi==21: #Xylem Pole Pericyle in new Cellset version
                        cgroupi=16
                    elif cgroupi==23: #Phloem in new Cellset version
                        cgroupi==11
                    elif cgroupi==26: #Companion Cell in new Cellset version
                        cgroupi==12
                    if cgroupj==19 or cgroupj==20:  #Xylem in new Cellset version
                        cgroupj=13
                    elif cgroupj==21: #Xylem Pole Pericyle in new Cellset version
                        cgroupj=16
                    elif cgroupj==23: #Phloem in new Cellset version
                        cgroupj==11
                    elif cgroupj==26: #Companion Cell in new Cellset version
                        cgroupj==12
                    temp_factor=1.0 #Quantity of plasmodesmata (adjusted by relative aperture)
                    if ((j-network.n_wall_junction in geometry.intercellular_ids) or (i-network.n_wall_junction in geometry.intercellular_ids)) and Barrier>0: #one of the connected cells is an intercellular space "cell".
                        temp_factor=0.0
                    elif cgroupj==13 and cgroupi==13: #Fake wall splitting a xylem cell or vessel, high conductance in order to ensure homogeneous pressure within the splitted cell
                        temp_factor=10000*hydraulic.fplxheight*1.0E-04*eattr['length'] #Quantity of PD
                    elif Barrier>0 and (cgroupj==13 or cgroupi==13): #Mature xylem vessels, so no plasmodesmata with surrounding cells
                        temp_factor=0.0 #If Barrier==0, this case is treated like xylem is a stelar parenchyma cell
                    elif (cgroupi==2 and cgroupj==1) or (cgroupj==2 and cgroupi==1):#Epidermis to exodermis cell or vice versa
                        temp_factor=hydraulic.fplxheight_epi_exo*1.0E-04*eattr['length'] #Will not be used in case there is no exodermal layer
                    elif (cgroupi==network.outercortex_connec_rank and cgroupj==4) or (cgroupj==network.outercortex_connec_rank and cgroupi==4):#Exodermis to cortex cell or vice versa
                        temp=float(hydraulic.kpl_elems[iPD].get("cortex_factor")) #Correction for specific cell-type PD aperture
                        if Barrier>0:
                            temp_factor=2*temp/(temp+1)*hydraulic.fplxheight_outer_cortex*1.0E-04*eattr['length']*network.len_outer_cortex /network.cross_section_outer_cortex
                        else: #No aerenchyma
                            temp_factor=2*temp/(temp+1)*hydraulic.fplxheight_outer_cortex*1.0E-04*eattr['length']
                    elif (cgroupi==4 and cgroupj==4):#Cortex to cortex cell
                        temp=float(hydraulic.kpl_elems[iPD].get("cortex_factor")) #Correction for specific cell-type PD aperture
                        if Barrier>0:
                            temp_factor=temp*hydraulic.fplxheight_cortex_cortex*1.0E-04*eattr['length']*network.len_cortex_cortex /network.cross_section_cortex_cortex
                        else: #No aerenchyma
                            temp_factor=temp*hydraulic.fplxheight_cortex_cortex*1.0E-04*eattr['length']
                    elif (cgroupi==3 and cgroupj==4) or (cgroupj==3 and cgroupi==4):#Cortex to endodermis cell or vice versa
                        temp=float(hydraulic.kpl_elems[iPD].get("cortex_factor")) #Correction for specific cell-type PD aperture
                        if Barrier>0:
                            temp_factor=2*temp/(temp+1)*hydraulic.fplxheight_cortex_endo*1.0E-04*eattr['length']*network.len_cortex_endo /network.cross_section_cortex_endo
                        else: #No aerenchyma
                            temp_factor=2*temp/(temp+1)*hydraulic.fplxheight_cortex_endo*1.0E-04*eattr['length']
                    elif (cgroupi==3 and cgroupj==3):#Endodermis to endodermis cell
                        temp_factor=hydraulic.fplxheight_endo_endo*1.0E-04*eattr['length']
                    elif (cgroupi==3 and cgroupj==16) or (cgroupj==3 and cgroupi==16):#Pericycle to endodermis cell or vice versa
                        if (i-network.n_wall_junction in network.plasmodesmata_indice) or (j-network.n_wall_junction in network.plasmodesmata_indice):
                            temp=float(hydraulic.kpl_elems[iPD].get("PPP_factor")) #Correction for specific cell-type PD aperture
                        else:
                            temp=1
                        temp_factor=2*temp/(temp+1)*hydraulic.fplxheight_endo_peri*1.0E-04*eattr['length']
                    elif (cgroupi==16 and (cgroupj==5 or cgroupj==13)) or (cgroupj==16 and (cgroupi==5 or cgroupi==13)):#Pericycle to stele cell or vice versa
                        if (i-network.n_wall_junction in network.plasmodesmata_indice) or (j-network.n_wall_junction in network.plasmodesmata_indice):
                            temp=float(hydraulic.kpl_elems[iPD].get("PPP_factor")) #Correction for specific cell-type PD aperture
                        else:
                            temp=1
                        temp_factor=2*temp/(temp+1)*hydraulic.fplxheight_peri_stele*1.0E-04*eattr['length']
                    elif ((cgroupi==5 or cgroupi==13) and cgroupj==12) or (cgroupi==12 and (cgroupj==5 or cgroupj==13)):#Stele to companion cell
                        temp=float(hydraulic.kpl_elems[iPD].get("PCC_factor")) #Correction for specific cell-type PD aperture
                        temp_factor=2*temp/(temp+1)*hydraulic.fplxheight_stele_comp*1.0E-04*eattr['length']
                    elif (cgroupi==16 and cgroupj==12) or (cgroupi==12 and cgroupj==16):#Pericycle to companion cell
                        temp1=float(hydraulic.kpl_elems[iPD].get("PCC_factor"))
                        if (i-network.n_wall_junction in network.plasmodesmata_indice) or (j-network.n_wall_junction in network.plasmodesmata_indice):
                            temp2=float(hydraulic.kpl_elems[iPD].get("PPP_factor")) #Correction for specific cell-type PD aperture
                        else:
                            temp2=1
                        temp_factor=2*temp1*temp2/(temp1+temp2)*hydraulic.fplxheight_peri_comp*1.0E-04*eattr['length']
                    elif (cgroupi==12 and cgroupj==12):#Companion to companion cell
                        temp=float(hydraulic.kpl_elems[iPD].get("PCC_factor"))
                        temp_factor=temp*hydraulic.fplxheight_comp_comp*1.0E-04*eattr['length']
                    elif (cgroupi==12 and cgroupj==11) or (cgroupi==11 and cgroupj==12):#Companion to phloem sieve tube cell
                        temp=float(hydraulic.kpl_elems[iPD].get("PCC_factor"))
                        temp_factor=2*temp/(temp+1)*hydraulic.fplxheight_comp_sieve*1.0E-04*eattr['length']
                    elif (cgroupi==16 and cgroupj==11) or (cgroupi==11 and cgroupj==16):#Pericycle to phloem sieve tube cell
                        if (i-network.n_wall_junction in network.plasmodesmata_indice) or (j-network.n_wall_junction in network.plasmodesmata_indice):
                            temp=float(hydraulic.kpl_elems[iPD].get("PPP_factor")) #Correction for specific cell-type PD aperture
                        else:
                            temp=1
                        temp_factor=2*temp/(temp+1)*hydraulic.fplxheight_peri_sieve*1.0E-04*eattr['length']
                    elif ((cgroupi==5 or cgroupi==13) and cgroupj==11) or (cgroupi==11 and (cgroupj==5 or cgroupj==13)):#Stele to phloem sieve tube cell
                        temp_factor=hydraulic.fplxheight_stele_sieve*1.0E-04*eattr['length']
                    #elif cgroupi==13 and cgroupj==13: #Fake wall splitting a xylem cell or vessel, high conductance in order to ensure homogeneous pressure within the splitted cell
                    #    temp_factor=10000*hydraulic.fplxheight*1.0E-04*eattr['length']
                    elif ((cgroupi==5 or cgroupi==13) and (cgroupj==5 or cgroupj==13)):#Stele to stele cell
                        temp_factor=hydraulic.fplxheight_stele_stele*1.0E-04*eattr['length']
                    else: #Default plasmodesmatal frequency
                        temp_factor=hydraulic.fplxheight*1.0E-04*eattr['length'] #eattr['kpl']
                    K = Kpl*temp_factor
                    ########Solute fluxes (diffusion across plasmodesmata)
                    if general.sym_contagion==2:
                        DF=geometry.pd_section*temp_factor/geometry.thickness*1.0E-04*hormones.diff1_pd1 #"Diffusive flux": Total PD cross-section area (micron^2) per unit PD length (micron) (tunred into cm) multiplied by solute diffusivity (cm^2/d) (yields cm^3/d)
                        if general.apo_contagion==2: #Sym & Apo contagion
                            if i-network.n_wall_junction not in hormones.sym_zombie0:
                                matrix_C[i][i] -= DF
                                matrix_C[i][j] += DF #Convection will be dealt with further down
                            if j-network.n_wall_junction not in hormones.sym_zombie0:
                                matrix_C[j][j] -= DF
                                matrix_C[j][i] += DF
                        else: #Only Sym contagion
                            if i-network.n_wall_junction not in hormones.sym_zombie0:
                                matrix_SymC[i-network.n_wall_junction][i-network.n_wall_junction] -= DF
                                matrix_SymC[i-network.n_wall_junction][j-network.n_wall_junction] += DF
                            if j-network.n_wall_junction not in hormones.sym_zombie0:
                                matrix_SymC[j-network.n_wall_junction][j-network.n_wall_junction] -= DF #Convection will be dealt with further down
                                matrix_SymC[j-network.n_wall_junction][i-network.n_wall_junction] += DF
                matrix_W[i][i] -= K #Filling the Doussan matrix (symmetric)
                matrix_W[i][j] += K
                matrix_W[j][i] += K
                matrix_W[j][j] -= K
    
    #Adding matrix components at soil-wall and wall-xylem connections & rhs terms
    rhs = np.zeros((network.graph.number_of_nodes(),1))
    rhs_s = np.zeros((network.graph.number_of_nodes(),1)) #Initializing the right-hand side matrix of soil pressure potentials
    rhs_x = np.zeros((network.graph.number_of_nodes(),1)) #Initializing the right-hand side matrix of xylem pressure potentials
    rhs_p = np.zeros((network.graph.number_of_nodes(),1)) #Initializing the right-hand side matrix of hydrostatic potentials for phloem BC
    
    #Adding matrix components at soil-wall connections
    for wall_id in network.border_walls:
        if (position[wall_id][0]>=Xcontact) or (wall_to_cell[wall_id][0]-network.n_wall_junction in hormones.contact): #Wall (not including junctions) connected to soil
            temp=1.0E-04*(network.wall_lengths[wall_id]/2*height)/(geometry.thickness/2)
            K=kw*temp #Half the wall length is used here as the other half is attributed to the junction (Only for connection to soil)
            matrix_W[wall_id][wall_id] -= K #Doussan matrix
            rhs_s[wall_id][0] = -K    #Right-hand side vector, could become Psi_soil[idwall], which could be a function of the horizontal position
            #if boundary.c_flag:
            #    #Diffusion
            #    matrix_C[wall_id][wall_id] -= temp*Diff1
            #    rhs_C[wall_id][0] -= temp*Diff1*Os_soil[0][0]
                
    #Adding matrix components at soil-junction connections
    for network.n_junctions in network.border_junction:
        if (position[network.n_junctions][0]>=Xcontact) or (junction_wall_cell[network.n_junctions-network.n_walls][0]-network.n_wall_junction in hormones.contact) or (junction_wall_cell[network.n_junctions-network.n_walls][1]-network.n_wall_junction in hormones.contact) or (junction_wall_cell[network.n_junctions-network.n_walls][2]-network.n_wall_junction in hormones.contact): #Junction connected to soil
            temp=1.0E-04*(network.wall_lengths[network.n_junctions]*height)/(geometry.thickness/2)
            K=kw*temp
            matrix_W[network.n_junctions][network.n_junctions] -= K #Doussan matrix
            rhs_s[network.n_junctions][0] = -K    #Right-hand side vector, could become Psi_soil[idwall], which could be a function of the horizontal position
            #if boundary.c_flag:
            #    matrix_C[network.n_junctions][network.n_junctions] -= temp*Diff1 #Diffusion BC at soil junction
            #    rhs_C[network.n_junctions][0] -= temp*Diff1*Os_soil[0][0]
    
    #Creating connections to xylem & phloem BC elements for kr calculation (either xylem or phloem flow occurs depending on whether the segment is in the differentiation or elongation zone)
    if Barrier>0:
        if not isnan(Psi_xyl[iMaturity][0]): #Pressure xylem BC
            for cid in network.xylem_cells:
                rhs_x[cid][0] = -hydraulic.k_xyl  #Axial conductance of xylem vessels
                matrix_W[cid][cid] -= hydraulic.k_xyl
                #if boundary.c_flag:
                #    temp=10E-04*((network.cell_perimeters[cid-network.n_wall_junction]/2)**2)/pi/height #Cell approximative cross-section area (cm^2) per length (cm)
                #    matrix_C[cid][cid] -= temp*Diff1*100 #Diffusion BC in xylem open vessels assumed 100 times easier than in walls
                #    rhs_C[cid][0] -= temp*Diff1*100
            rhs = rhs_s*boundary.scenarios[0]['psi_soil_left'] + rhs_x*Psi_xyl[iMaturity][0] #multiplication of rhs components delayed till this point so that rhs_s & rhs_x can be re-used to calculate Q
        elif not isnan(Flow_xyl[0][0]):
            i=1
            for cid in network.xylem_cells:
                rhs_x[cid][0] = Flow_xyl[i][0]
                i+=1
            #    if boundary.c_flag:
            #        temp=10E-04*((network.cell_perimeters[cid-network.n_wall_junction]/2)**2)/pi/height #Cell approximative cross-section area (cm^2) per length (cm)
            #        matrix_C[cid][cid] -= temp*Diff1*100 #Diffusion BC in xylem open vessels assumed 100 times easier than in walls
            #        rhs_C[cid][0] -= temp*Diff1*100
            rhs = rhs_s*boundary.scenarios[0]['psi_soil_left'] + rhs_x #multiplication of rhs components delayed till this point so that rhs_s & rhs_x can be re-used to calculate Q
        else:
            rhs = rhs_s*boundary.scenarios[0]['psi_soil_left']
    elif Barrier==0:
        if not isnan(Psi_sieve[iMaturity][0]):
            for cid in network.protosieve_list:
                rhs_p[cid][0] = -hydraulic.k_sieve  #Axial conductance of phloem sieve tube
                matrix_W[cid][cid] -= hydraulic.k_sieve
            rhs = rhs_s*boundary.scenarios[0]['psi_soil_left'] + rhs_p*Psi_sieve[iMaturity][0] #multiplication of rhs components delayed till this point so that rhs_s & rhs_x can be re-used to calculate Q
        elif not isnan(Flow_sieve[0][0]):
            i=1
            for cid in network.protosieve_list:
                rhs_p[cid][0] = Flow_sieve[i][0]
                i+=1
            rhs = rhs_s*boundary.scenarios[0]['psi_soil_left'] + rhs_p #multiplication of rhs components delayed till this point so that rhs_s & rhs_x can be re-used to calculate Q
        else:
            rhs = rhs_s*boundary.scenarios[0]['psi_soil_left']
    
    
    ##################################################
    ##Solve Doussan equation, results in soln matrix##
    ##################################################
    
    soln = np.linalg.solve(matrix_W,rhs) #Solving the equation to get potentials inside the network
    
    #Verification that computation was correct
    verif1=np.allclose(np.dot(matrix_W,soln),rhs)
    
    
    #Removing xylem and phloem BC terms
    if Barrier>0:
        if not isnan(Psi_xyl[iMaturity][0]): #Pressure xylem BC
            for cid in network.xylem_cells:
                matrix_W[cid][cid] += hydraulic.k_xyl
    elif Barrier==0:
        if not isnan(Psi_sieve[iMaturity][0]):
            for cid in network.protosieve_list:
                matrix_W[cid][cid] += hydraulic.k_sieve
    
    #Flow rates at interfaces
    Q_soil=[]
    for ind in network.border_walls:
        Q_soil.append(rhs_s[ind]*(soln[ind]-boundary.scenarios[0]['psi_soil_left'])) #(cm^3/d) Positive for water flowing into the root
    for ind in network.border_junction:
        Q_soil.append(rhs_s[ind]*(soln[ind]-boundary.scenarios[0]['psi_soil_left'])) #(cm^3/d) Positive for water flowing into the root
    Q_xyl=[]
    Q_sieve=[]
    if Barrier>0:
        if not isnan(Psi_xyl[iMaturity][0]):
            for cid in network.xylem_cells:
                Q=rhs_x[cid][0]*(soln[cid][0]-Psi_xyl[iMaturity][0])
                Q_xyl.append(Q) #(cm^3/d) Negative for water flowing into xylem tubes
                rank=int(network.cell_ranks[cid-network.n_wall_junction])
                row=int(network.rank_to_row[rank])
                Q_xyl_layer[row][iMaturity][0] += Q
        elif not isnan(Flow_xyl[0][0]):
            for cid in network.xylem_cells:
                Q=-rhs_x[cid][0]
                Q_xyl.append(Q) #(cm^3/d) Negative for water flowing into xylem tubes
                rank=int(network.cell_ranks[cid-network.n_wall_junction])
                row=int(network.rank_to_row[rank])
                Q_xyl_layer[row][iMaturity][0] += Q
    elif Barrier==0:
        if not isnan(Psi_sieve[iMaturity][0]):
            for cid in network.protosieve_list:
                Q=rhs_p[cid]*(soln[cid][0]-Psi_sieve[iMaturity][0])
                Q_sieve.append(Q) #(cm^3/d) Negative for water flowing into phloem tubes
                rank=int(network.cell_ranks[cid-network.n_wall_junction][0])
                row=int(network.rank_to_row[rank][0])
                Q_sieve_layer[row][iMaturity][0] += Q
        elif not isnan(Flow_sieve[0][0]):
            for cid in network.protosieve_list:
                Q=-rhs_p[cid]
                Q_sieve.append(Q) #(cm^3/d) Negative for water flowing into xylem tubes
                rank=int(network.cell_ranks[cid-network.n_wall_junction][0])
                row=int(network.rank_to_row[rank][0])
                Q_sieve_layer[row][iMaturity][0] += Q
        
    Q_tot[iMaturity][0]=sum(Q_soil) #Total flow rate at root surface
    if Barrier>0:
        if not isnan(Psi_xyl[iMaturity][0]):
            kr_tot[iMaturity][0]=Q_tot[iMaturity][0]/(boundary.scenarios[0]['psi_soil_left']-Psi_xyl[iMaturity][0])/network.perimeter/height/1.0E-04
        else:
            print('Error: Scenario 0 should have xylem pressure boundary conditions, except for the elongation zone')
    elif Barrier==0:
        if not isnan(Psi_sieve[iMaturity][0]):
            kr_tot[iMaturity][0]=Q_tot[iMaturity][0]/(boundary.scenarios[0]['psi_soil_left']-Psi_sieve[iMaturity][0])/network.perimeter/height/1.0E-04
        else:
            print('Error: Scenario 0 should have phloem pressure boundary conditions in the elongation zone')
    print("Radial conductivity:",kr_tot[iMaturity][0],"cm/hPa/d")#, Barrier:",Barrier,", height: ",height," microns")
    
    if Barrier>0 and isnan(Psi_xyl[iMaturity][0]):
        Psi_xyl[iMaturity][0]=0.0
        for cid in network.xylem_cells:
            Psi_xyl[iMaturity][0]+=soln[cid][0]/len(network.xylem_cells) #Average of xylem water pressures
    elif Barrier==0 and isnan(Psi_sieve[iMaturity][0]):
        Psi_sieve[iMaturity][0]=0.0
        for cid in network.protosieve_list:
            Psi_sieve[iMaturity][0]+=soln[cid][0]/network.n_protosieve #Average of protophloem water pressures
    
    #Calculation of standard transmembrane fractions
    jmb=0 #Index for membrane conductance vector
    for node, edges in network.graph.adjacency() : #adjacency_iter returns an iterator of (node, adjacency dict) tuples for all nodes. This is the fastest way to look at every edge. For directed graphs, only outgoing adjacencies are included.
        i = indice[node] #Node ID number
        if i<network.n_walls: #wall ID
            psi = soln[i][0]    #Node water potential
            psi_o_cell = inf #Opposite cell water potential
            #Here we count surrounding cell types in order to identify in which row of the endodermis or exodermis we are.
            count_endo=0 #total number of endodermis cells around the wall
            count_stele_overall=0 #total number of stelar cells around the wall
            count_exo=0 #total number of exodermis cells around the wall
            count_epi=0 #total number of epidermis cells around the wall
            #count_stele=0 #total number of epidermis cells around the wall
            count_cortex=0 #total number of epidermis cells around the wall
            count_passage=0 #total number of passage cells around the wall
            for neighboor, eattr in edges.items(): #Loop on connections (edges)
                if eattr['path'] == 'membrane': #Wall connection
                    if any(passage_cell_ID==array((indice[neighboor])-network.n_wall_junction)):
                        count_passage+=1
                    if network.graph.nodes[neighboor]['cgroup']==3:#Endodermis
                        count_endo+=1
                    elif network.graph.nodes[neighboor]['cgroup']>4:#Pericycle or stele
                        count_stele_overall+=1
                    elif network.graph.nodes[neighboor]['cgroup']==1:#Exodermis
                        count_exo+=1
                    elif network.graph.nodes[neighboor]['cgroup']==2:#Epidermis
                        count_epi+=1
                    elif network.graph.nodes[neighboor]['cgroup']==4:#Cortex
                        count_cortex+=1
                # if network.graph.nodes[neighboor]['cgroup']==5:#Stele
                #     count_stele+=1
            for neighboor, eattr in edges.items(): #Loop on connections (edges)
                j = indice[neighboor] #Neighbouring node ID number
                path = eattr['path'] #eattr is the edge attribute (i.e. connection type)
                if path == "membrane": #Membrane connection
                    psin = soln[j][0] #Neighbouring node water potential
                    K=Kmb[jmb][0]
                    jmb+=1
                    #Flow densities calculation
                    #Macroscopic distributed parameter for transmembrane flow
                    #Discretization based on cell layers and apoplasmic barriers
                    rank = int(network.cell_ranks[j-network.n_wall_junction])
                    row = int(network.rank_to_row[rank])
                    if rank == 1 and count_epi > 0: #Outer exodermis
                        row += 1
                    if rank == 3 and count_cortex > 0: #Outer endodermis
                        if any(passage_cell_ID==array(j-network.n_wall_junction)) and Barrier==2:
                            row += 2
                        else:
                            row += 3
                    elif rank == 3 and count_stele_overall > 0: #Inner endodermis
                        if any(passage_cell_ID==array(j-network.n_wall_junction)) and Barrier==2:
                            row += 1
                            
                    Flow = K * (psi - psin) #Note that this is only valid because we are in the scenario 0 with no osmotic potentials
                    if ((j-network.n_wall_junction not in geometry.intercellular_ids) and (j not in network.xylem_cells)) or Barrier==0: #Not part of STF if crosses an intercellular space "membrane" or mature xylem "membrane" (that is no membrane though still labelled like one)
                        if Flow > 0 :
                            UptakeLayer_plus[row][iMaturity][0] += Flow #grouping membrane flow rates in cell layers
                        else:
                            UptakeLayer_minus[row][iMaturity][0] += Flow
                        if Flow/Q_tot[iMaturity][0] > 0 :
                            STFlayer_plus[row][iMaturity] += Flow/Q_tot[iMaturity][0] #Cell standard transmembrane fraction (positive)
                            STFcell_plus[j-network.n_wall_junction][iMaturity] += Flow/Q_tot[iMaturity][0] #Cell standard transmembrane fraction (positive)
                            #STFmb[jmb-1][iMaturity] = Flow/Q_tot[iMaturity][0]
                        else:
                            STFlayer_minus[row][iMaturity] += Flow/Q_tot[iMaturity][0] #Cell standard transmembrane fraction (negative)
                            STFcell_minus[j-network.n_wall_junction][iMaturity] += Flow/Q_tot[iMaturity][0] #Cell standard transmembrane fraction (negative)
                            #STFmb[jmb-1][iMaturity] = Flow/Q_tot[iMaturity][0]
                        STFmb[jmb-1][iMaturity] = Flow/Q_tot[iMaturity][0]