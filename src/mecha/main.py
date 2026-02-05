from py_compile import main
import numpy as np 
from numpy import genfromtxt #Load data from a text file, with missing values handled as specified.
from numpy.random import *  # for random sampling
import scipy.linalg as slin #Linear algebra functions
import pylab #Found in the package pyqt
from pylab import *  # for plotting
import networkx as nx
import sys, os
import re
from pylab import *  # for plotting
import geopandas as gpd # not used yet but could be useful for spatial data calculations
import argparse # for command-line argument parsing

from mecha.utils.data_loader import *
from mecha.utils.prepare_paraview import *
from mecha.utils.network_builder import *

def mecha(general_config='../extdata/General.xml',#'Arabido1_General.xml' #'MilletLR3_General.xml' #
          geometry_config='../extdata/Geometry.xml',#'Arabido4_Geometry_BBSRC.xml' #'Maize2_Geometry.xml' #''MilletLR3_Geometry.xml'    #'Wheat1_Nodal_Geometry_aerenchyma.xml' #'Maize1_Geometry.xml' #
          hydraulic_config='../extdata/Hydraulics.xml', #'Arabido1_Hydraulics_ERC.xml' #'MilletLR3_Hydraulics.xml' #'Test_Hydraulics.xml' #
          boundary_condition_config='../extdata/BCs.xml', #'Arabido4_BC_BBSRC2.xml' #'Arabido1_BC_Emily.xml' #'Arabido3_BC_BBSRC.xml' #'Maize_BC_SoluteAna_krOsmo.xml'#'Maize_BC_OSxyl_hetero.xml' #'Arabido1_BC_Emily.xml' #'BC_Test.xml' #'Maize_BC_Plant_phys.xml'
          hormones_config='../extdata/Hormones.xml',
          cellset_file='./extdata/current_root.xml',#present in Geometry.xml
          outdir=os.getcwd()): 
    
    print('[1/5] Importing data')
    general = GeneralData(general_config)
    geometry = GeometryData(geometry_config)
    hormones = HormonesData(hormones_config)
    hydraulic = HydraulicData(hydraulic_config)
    boundary = BoundaryData(boundary_condition_config)

    passage_cell_ID=[]
    
    print('[2/5] Creating the network')
    network = NetworkBuilder()
    # Build network structure
    print(cellset_file)
    cellset_data = parse_cellset(cellset_file= cellset_file)
    network.build_network(general, geometry, cellset_data)

    position=nx.get_node_attributes(network.graph,'position') #Updates nodes XY positions (micrometers)
    indice=nx.get_node_attributes(network.graph,'indice') #Node indices (walls, junctions and cells)

    paraview = prepare_geometrical_properties(general, network, hormones, position, indice)
    locals().update(paraview)

    #Unit changes
    sperd=24.0*3600.0 #(seconds per day)
    cmperm=100.0 #(cm per metre)
    

    print('[3/5] Loop for '+str(hydraulic.n_hydraulics)+' hydraulic scenarios')
    #Start the loop of hydraulic properties
    for h in range(hydraulic.n_hydraulics):
        
        print('Hydraulic scenario #'+str(h))
        newpath=outdir+'/'+geometry.plant_name+'/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        
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
        
        print(f'\n[4/5] Processing {geometry.n_maturity} maturity stage(s)...')

        iMaturity=-1 #Iteration index in the Barriers loop
        for Maturity in geometry.maturity_elems:
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
            
            for count in range(1,boundary.n_scenarios):
                
                #Initializing the connectivity matrix including boundary conditions
                rhs = np.zeros((network.graph.number_of_nodes(),1))
                rhs_x = np.zeros((network.graph.number_of_nodes(),1)) #Initializing the right-hand side matrix of xylem pressure potentials
                rhs_p = np.zeros((network.graph.number_of_nodes(),1)) #Initializing the right-hand side matrix of hydrostatic potentials for phloem BC
                rhs_e = np.zeros((network.graph.number_of_nodes(),1)) #Initializing the right-hand side matrix of cell elongation
                rhs_o = np.zeros((network.graph.number_of_nodes(),1)) #Initializing the right-hand side matrix of osmotic potentials
                Os_cells = np.zeros((network.n_cells,1)) #Initializing the cell osmotic potential vector
                Os_walls = np.zeros((network.n_walls,1)) #Initializing the wall osmotic potential vector
                s_membranes = np.zeros((network.n_membrane,1)) #Initializing the membrane reflection coefficient vector
                Os_membranes = np.zeros((network.n_membrane,2)) #Initializing the osmotic potential storage side by side of membranes (0 for the wall, 1 for the protoplast)
                #rhs_s invariable between diferent scenarios but can vary for different hydraulic properties
                
                #Apoplastic & symplastic convective direction matrices initialization
                Cell_connec_flow=zeros((network.n_cells,14),dtype=int) #Flow direction across plasmodesmata, positive when entering the cell, negative otherwise
                Apo_connec_flow=zeros((network.n_wall_junction,5),dtype=int) #Flow direction across cell walls, rows correspond to apoplastic nodes, and the listed nodes in each row receive convective flow from the row node
                nApo_connec_flow=zeros((network.n_wall_junction,1),dtype=int)
                
                print('   Scenario #'+str(count))
                
                
                #Reflection coefficients of membranes (undimensional)
                s_hetero[0][count]=int(boundary.psi_cell_elems[count].get("s_hetero")) #0:Uniform, 1: non-uniform, stele twice more permeable to solute, 2: non-uniform, cortex twice more permeable to solute
                s_factor[0][count]=float(boundary.psi_cell_elems[count].get("s_factor")) #(undimensional [0 -> 1]) multiplies all sigma values
                Elong_cell[0][count]=float(boundary.elong_cell_elems[count].get("midpoint_rate")) #Cell elongation rate (cm/d)
                Elong_cell_side_diff[0][count]=float(boundary.elong_cell_elems[count].get("side_rate_difference")) #Difference between cell elongation rates on the sides of the root in the EZ (cm/d)
                if s_hetero[0][count]==0:
                    s_epi=s_factor[0][count]*1.0
                    s_exo_epi=s_factor[0][count]*1.0
                    s_exo_cortex=s_factor[0][count]*1.0
                    s_cortex=s_factor[0][count]*1.0
                    s_endo_cortex=s_factor[0][count]*1.0
                    s_endo_peri=s_factor[0][count]*1.0
                    s_peri=s_factor[0][count]*1.0
                    s_stele=s_factor[0][count]*1.0
                    s_comp=s_factor[0][count]*1.0
                    s_sieve=s_factor[0][count]*1.0
                elif s_hetero[0][count]==1:
                    s_epi=s_factor[0][count]*1.0
                    s_exo_epi=s_factor[0][count]*1.0
                    s_exo_cortex=s_factor[0][count]*1.0
                    s_cortex=s_factor[0][count]*1.0
                    s_endo_cortex=s_factor[0][count]*1.0
                    s_endo_peri=s_factor[0][count]*0.5
                    s_peri=s_factor[0][count]*0.5
                    s_stele=s_factor[0][count]*0.5
                    s_comp=s_factor[0][count]*0.5
                    s_sieve=s_factor[0][count]*0.5
                elif s_hetero[0][count]==2:
                    s_epi=s_factor[0][count]*0.5
                    s_exo_epi=s_factor[0][count]*0.5
                    s_exo_cortex=s_factor[0][count]*0.5
                    s_cortex=s_factor[0][count]*0.5
                    s_endo_cortex=s_factor[0][count]*0.5
                    s_endo_peri=s_factor[0][count]*1.0
                    s_peri=s_factor[0][count]*1.0
                    s_stele=s_factor[0][count]*1.0
                    s_comp=s_factor[0][count]*1.0
                    s_sieve=s_factor[0][count]*1.0
                
                #Osmotic potentials (hPa)
                Os_hetero[0][count]=int(boundary.psi_cell_elems[count].get("Os_hetero")) #0:Uniform, 1: non-uniform no KNO3 treatment, 2: non-uniform with KNO3 treatment to help guttation
                Os_cortex[0][count]=float(boundary.psi_cell_elems[count].get("Os_cortex")) # Cortical cell osmotic potential (hPa)
                Os_sieve[0][count]=float(boundary.bc_sieve_elems[count].get("osmotic"))
                if Os_hetero[0][count]==0:
                    #Os_apo=-3000 #-0.3 MPa (Enns et al., 2000) applied stress
                    #-0.80 MPa (Enns et al., 2000) concentration of cortical cells, no KNO3
                    Os_epi=float(Os_cortex[0][count])
                    Os_exo=float(Os_cortex[0][count])
                    Os_c1=float(Os_cortex[0][count])
                    Os_c2=float(Os_cortex[0][count])
                    Os_c3=float(Os_cortex[0][count])
                    Os_c4=float(Os_cortex[0][count])
                    Os_c5=float(Os_cortex[0][count])
                    Os_c6=float(Os_cortex[0][count])
                    Os_c7=float(Os_cortex[0][count])
                    Os_c8=float(Os_cortex[0][count])
                    Os_endo=float(Os_cortex[0][count])
                    Os_peri=float(Os_cortex[0][count])
                    Os_stele=float(Os_cortex[0][count])
                    Os_comp=(float(Os_sieve[0][count])+Os_cortex[0][count])/2 #Average phloem and parenchyma
                    #Os_sieve=float(Os_cortex[0][count])
                elif Os_hetero[0][count]==1:
                    Os_epi=-5000 #(Rygol et al. 1993) #float(Os_cortex[0][count]) #-0.80 MPa (Enns et al., 2000) concentration of cortical cells, no KNO3
                    Os_exo=-5700 #(Rygol et al. 1993) #float(Os_cortex[0][count]) #-0.80 MPa (Enns et al., 2000) concentration of cortical cells, no KNO3
                    Os_c1=-6400 #(Rygol et al. 1993)
                    Os_c2=-7100 #(Rygol et al. 1993)
                    Os_c3=-7800 #(Rygol et al. 1993)
                    Os_c4=-8500 #(Rygol et al. 1993)
                    Os_c5=-9000 #(Rygol et al. 1993)
                    Os_c6=-9300 #(Rygol et al. 1993)
                    Os_c7=-9000 #(Rygol et al. 1993)
                    Os_c8=-8500 #(Rygol et al. 1993)
                    Os_endo=-6200 #-0.62 MPa (Enns et al., 2000) concentration of endodermis cells, no KNO3
                    Os_peri=-5000 #-0.50 MPa (Enns et al., 2000) concentration of pericycle cells, no KNO3
                    Os_stele=-7400 #-0.74 MPa (Enns et al., 2000) concentration of xylem parenchyma cells, no KNO3
                    Os_comp=(float(Os_sieve[0][count])-7400)/2 #Average phloem and parenchyma
                    #Os_sieve=-14200 #-1.42 MPa (Pritchard, 1996) in barley phloem
                elif Os_hetero[0][count]==2:
                    Os_epi=-11200 #(Rygol et al. 1993) #float(Os_cortex[0][count]) #-1.26 MPa (Enns et al., 2000) concentration of cortical cells, with KNO3
                    Os_exo=-11500 #(Rygol et al. 1993) #float(Os_cortex[0][count]) #-1.26 MPa (Enns et al., 2000) concentration of cortical cells, with KNO3
                    Os_c1=-11800 #(Rygol et al. 1993)
                    Os_c2=-12100 #(Rygol et al. 1993)
                    Os_c3=-12400 #(Rygol et al. 1993)
                    Os_c4=-12700 #(Rygol et al. 1993)
                    Os_c5=-12850 #(Rygol et al. 1993)
                    Os_c6=-12950 #(Rygol et al. 1993)
                    Os_c7=-12850 #(Rygol et al. 1993)
                    Os_c8=-12700 #(Rygol et al. 1993)
                    Os_endo=-10500 #-1.05 MPa (Enns et al., 2000) concentration of endodermis cells, with KNO3
                    Os_peri=-9200 #-0.92 MPa (Enns et al., 2000) concentration of pericycle cells, with KNO3
                    Os_stele=-12100 #-1.21 MPa (Enns et al., 2000) concentration of xylem parenchyma cells, with KNO3
                    Os_comp=(float(Os_sieve[0][count])-12100)/2 #Average of phloem and parenchyma
                    #Os_sieve=-14200 #-1.42 MPa (Pritchard, 1996) in barley phloem
                elif Os_hetero[0][count]==3:
                    Os_epi=float(Os_cortex[0][count])
                    Os_exo=float(Os_cortex[0][count])
                    Os_c1=float(Os_cortex[0][count])
                    Os_c2=float(Os_cortex[0][count])
                    Os_c3=float(Os_cortex[0][count])
                    Os_c4=float(Os_cortex[0][count])
                    Os_c5=float(Os_cortex[0][count])
                    Os_c6=float(Os_cortex[0][count])
                    Os_c7=float(Os_cortex[0][count])
                    Os_c8=float(Os_cortex[0][count])
                    Os_endo=float((Os_cortex[0][count]-5000.0)/2.0)
                    Os_peri=-5000.0 #Simple case with no stele pushing water out
                    Os_stele=-5000.0
                    Os_comp=(float(Os_sieve[0][count])-5000.0)/2 #Average phloem and parenchyma
                    #Os_sieve=-5000.0
                
                if boundary.c_flag:
                    jmb=0 #Index for membrane conductance vector
                    for node, edges in network.graph.adjacency() : #adjacency_iter returns an iterator of (node, adjacency dict) tuples for all nodes. This is the fastest way to look at every edge. For directed graphs, only outgoing adjacencies are included.
                        i=indice[node] #Node ID number
                        #Here we count surrounding cell types in order to identify on which side of the endodermis or exodermis we are.
                        count_endo=0 #total number of endodermis cells around the wall
                        count_stele_overall=0 #total number of stelar cells around the wall
                        count_exo=0 #total number of exodermis cells around the wall
                        count_epi=0 #total number of epidermis cells around the wall
                        count_cortex=0 #total number of cortical cells around the wall
                        count_passage=0 #total number of passage cells around the wall
                        if i<network.n_walls: #wall ID
                            for neighboor, eattr in edges.items(): #Loop on connections (edges)
                                if eattr['path'] == 'membrane': #Wall connection
                                    if any(passage_cell_ID==array((indice[neighboor])-network.n_wall_junction)):
                                        count_passage+=1
                                    if network.graph.nodes[neighboor]['cgroup']==3:#Endodermis
                                        count_endo+=1
                                    elif network.graph.nodes[neighboor]['cgroup']>4:#Pericycle or stele
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
                                if path == "membrane": #Membrane connection
                                    #Cell and wall osmotic potentials (cell types: 1=Exodermis;2=epidermis;3=endodermis;4=cortex;5=stele;16=pericycle)
                                    rank=int(network.cell_ranks[int(j-network.n_wall_junction)])
                                    row=int(network.rank_to_row[rank][0])
                                    if rank==1:#Exodermis
                                        Os_membranes[jmb][1]=Os_exo
                                        if count_epi==1: #wall between exodermis and epidermis
                                            s_membranes[jmb]=s_exo_epi
                                        elif count_epi==0: #wall between exodermis and cortex or between two exodermal cells
                                            s_membranes[jmb]=s_exo_cortex
                                    elif rank==2:#Epidermis
                                        Os_membranes[jmb][1]=Os_epi
                                        s_membranes[jmb]=s_epi
                                    elif rank==3:#Endodermis
                                        Os_membranes[jmb][1]=Os_endo
                                        if count_stele_overall==0: #wall between endodermis and cortex or between two endodermal cells
                                            s_membranes[jmb]=s_endo_cortex
                                        elif count_stele_overall>0 and count_endo>0: #wall between endodermis and pericycle
                                            s_membranes[jmb]=s_endo_peri
                                    elif rank>=40 and rank<50:#Cortex
                                        if j-network.n_wall_junction in geometry.intercellular_ids:
                                            Os_membranes[jmb][1]=0
                                            s_membranes[jmb]=0
                                        else:
                                            if row==network.row_outer_cortex-7:
                                                Os_membranes[jmb][1]=Os_c8
                                            elif row==network.row_outer_cortex-6:
                                                Os_membranes[jmb][1]=Os_c7
                                            elif row==network.row_outer_cortex-5:
                                                Os_membranes[jmb][1]=Os_c6
                                            elif row==network.row_outer_cortex-4:
                                                Os_membranes[jmb][1]=Os_c5
                                            elif row==network.row_outer_cortex-3:
                                                Os_membranes[jmb][1]=Os_c4
                                            elif row==network.row_outer_cortex-2:
                                                Os_membranes[jmb][1]=Os_c3
                                            elif row==network.row_outer_cortex-1:
                                                Os_membranes[jmb][1]=Os_c2
                                            elif row==network.row_outer_cortex:
                                                Os_membranes[jmb][1]=Os_c1
                                            s_membranes[jmb]=s_cortex
                                    elif network.graph.nodes[j]['cgroup']==5:#Stelar parenchyma
                                        Os_membranes[jmb][1]=Os_stele
                                        s_membranes[jmb]=s_stele
                                    elif rank==16:#Pericycle
                                        Os_membranes[jmb][1]=Os_peri
                                        s_membranes[jmb]=s_peri
                                    elif network.graph.nodes[j]['cgroup']==11 or network.graph.nodes[j]['cgroup']==23:#Phloem sieve tube cell
                                        if not isnan(Os_sieve[0][count]):
                                            if Barrier>0 or j in network.protosieve_list:
                                                Os_membranes[jmb][1]=float(Os_sieve[0][count])
                                            else:
                                                Os_membranes[jmb][1]=Os_stele
                                        else:
                                            Os_membranes[jmb][1]=Os_stele
                                        s_membranes[jmb]=s_sieve
                                    elif network.graph.nodes[j]['cgroup']==12 or network.graph.nodes[j]['cgroup']==26:#Companion cell
                                        if not isnan(Os_sieve[0][count]):
                                            Os_membranes[jmb][1]=Os_comp
                                        else:
                                            Os_membranes[jmb][1]=Os_stele
                                        s_membranes[jmb]=s_comp
                                    elif network.graph.nodes[j]['cgroup']==13 or network.graph.nodes[j]['cgroup']==19 or network.graph.nodes[j]['cgroup']==20:#Xylem cell or vessel
                                        if Barrier==0:
                                            Os_membranes[jmb][1]=Os_stele
                                            s_membranes[jmb]=s_stele
                                        else:
                                            Os_membranes[jmb][1]=0.0
                                            s_membranes[jmb]=0.0
                                    jmb+=1
                
                #Soil and xylem water potentials
                #boundary.scenarios[count]['psi_soil_left']=float(boundary.psi_soil_elems[count].get("pressure_left")) #Soil pressure potential (hPa)
                Psi_xyl[iMaturity][count]=float(boundary.bc_xyl_elems[count].get("pressure")) #Xylem pressure potential (hPa)
                dPsi_xyl[iMaturity][count]=float(boundary.bc_xyl_elems[count].get("deltaP")) #Xylem pressure potential change as compared to equilibrium pressure (hPa)
                Flow_xyl[0][count]=float(boundary.bc_xyl_elems[count].get("flowrate")) #Xylem flow rate (cm^3/d)
                if not isnan(Flow_xyl[0][count]):
                    if isnan(Psi_xyl[iMaturity][count]) and isnan(dPsi_xyl[iMaturity][count]):
                        tot_flow=Flow_xyl[0][count]
                        sum_area=0.0
                        i=1
                        for cid in network.xylem_cells:
                            area=network.cell_areas[cid-network.n_wall_junction]
                            Flow_xyl[i][count]=tot_flow*area
                            sum_area+=area
                            i+=1
                        i=1
                        for cid in network.xylem_cells:
                            Flow_xyl[i][count]/=sum_area #Total xylem flow rate partitioned proportionnally to xylem cross-section area
                            i+=1
                        if Flow_xyl[0][count]==0.0:
                            iEquil_xyl=count
                        if boundary.c_flag:
                            #Estimate the radial distribution of solutes later on from "u"
                            #First estimate water radial velocity in the apoplast
                            u=zeros((2,1))
                            u[0][0]=tot_flow/(height*1.0E-04)/(geometry.thickness*1.0E-04)/geometry.cell_per_layer[0][0] #Cortex (cm/d)
                            u[1][0]=tot_flow/(height*1.0E-04)/(geometry.thickness*1.0E-04)/geometry.cell_per_layer[1][0] #Stele (cm/d)
                    else:
                        print('Error: Cannot have both pressure and flow BC at xylem boundary')
                elif not isnan(dPsi_xyl[iMaturity][count]):
                    if isnan(Psi_xyl[iMaturity][count]):
                        Psi_xyl[iMaturity][count]=Psi_xyl[iMaturity][iEquil_xyl]+dPsi_xyl[iMaturity][count]
                    else:
                        print('Error: Cannot have both pressure and pressure change relative to equilibrium as xylem boundary condition')
                if not isnan(Psi_xyl[iMaturity][count]):
                    if boundary.c_flag:
                        #Estimate the radial distribution of solutes
                        #First estimate total flow rate (cm^3/d) from BC & kr
                        tot_flow1=0.0
                        u=zeros((2,1))
                        iter=0
                        tot_flow2=kr_tot[iMaturity][0]*network.perimeter*height*1.0E-04*(boundary.scenarios[count]['psi_soil_left']+boundary.scenarios[count]['osmotic_left_soil']-Psi_xyl[iMaturity][count]-boundary.scenarios[count]['osmotic_xyl']) 
                        print('flow_rate =',tot_flow2,' iter =',iter)
                        #Convergence loop of water radial velocity and solute apoplastic convection-diffusion
                        while abs(tot_flow1-tot_flow2)/abs(tot_flow2)>0.001 and iter<30:
                            iter+=1
                            if iter==1:
                                tot_flow1=tot_flow2
                            elif iter>1 and sign(tot_flow1/tot_flow2)==1:
                                tot_flow1=(tot_flow1+tot_flow2)/2
                            else:
                                tot_flow1=tot_flow1/2
                            #Then estimate water radial velocity in the apoplast
                            u[0][0]=tot_flow1/(height*1.0E-04)/(geometry.thickness*1.0E-04)/geometry.cell_per_layer[0][0] #Cortex apoplastic water velocity (cm/d) positive inwards
                            u[1][0]=tot_flow1/(height*1.0E-04)/(geometry.thickness*1.0E-04)/geometry.cell_per_layer[1][0] #Stele apoplastic water velocity (cm/d) positive inwards
                            #Then estimate the radial solute distribution from an analytical solution (C(x)=C0+C0*(exp(u*x/D)-1)/(u/D*exp(u*x/D)-exp(u*L/D)+1)
                            Os_apo_cortex_eq=0.0
                            Os_apo_stele_eq=0.0
                            Os_sym_cortex_eq=0.0
                            Os_sym_stele_eq=0.0
                            #temp1=0.0
                            #temp2=0.0
                            jmb=0 #Index for membrane vector
                            for node, edges in network.graph.adjacency() : #adjacency_iter returns an iterator of (node, adjacency dict) tuples for all nodes. This is the fastest way to look at every edge. For directed graphs, only outgoing adjacencies are included.
                                i = indice[node] #Node ID number
                                if i<network.n_walls: #wall ID
                                    for neighboor, eattr in edges.items(): #Loop on connections (edges)
                                        if eattr['path'] == 'membrane': #Wall connection
                                            if r_rel[i]>=0: #cortical side
                                                Os_apo=boundary.scenarios[count]['osmotic_left_soil']*exp(u[0][0]*abs(r_rel[i])*L_diff[0]/boundary.scenarios[count]['osmotic_diffusivity_soil'])
                                                Os_apo_cortex_eq+=STFmb[jmb][iMaturity]*(Os_apo*s_membranes[jmb])
                                                Os_sym_cortex_eq+=STFmb[jmb][iMaturity]*(Os_membranes[jmb][1]*s_membranes[jmb])
                                                #temp1+=STFmb[jmb][iMaturity]
                                            else: #Stelar side
                                                Os_apo=boundary.scenarios[count]['osmotic_xyl']*exp(-u[1][0]*abs(r_rel[i])*L_diff[1]/boundary.scenarios[count]['osmotic_diffusivity_xyl'])
                                                Os_apo_stele_eq-=STFmb[jmb][iMaturity]*(Os_apo*s_membranes[jmb])
                                                Os_sym_stele_eq-=STFmb[jmb][iMaturity]*(Os_membranes[jmb][1]*s_membranes[jmb])
                                                #temp2+=STFmb[jmb][iMaturity]
                                            Os_membranes[jmb][0]=Os_apo
                                            jmb+=1
                            tot_flow2=kr_tot[iMaturity][0]*network.perimeter*height*1.0E-04*(boundary.scenarios[count]['psi_soil_left']+Os_apo_cortex_eq-Os_sym_cortex_eq-Psi_xyl[iMaturity][count]-Os_apo_stele_eq+Os_sym_stele_eq)
                            print('flow_rate =',tot_flow2,' iter =',iter)
                        u[0][0]=tot_flow2/(height*1.0E-04)/(geometry.thickness*1.0E-04)/geometry.cell_per_layer[0][0] #Cortex (cm/d)
                        u[1][0]=tot_flow2/(height*1.0E-04)/(geometry.thickness*1.0E-04)/geometry.cell_per_layer[1][0] #Stele (cm/d)
                        ##Then estimate osmotic potentials in radial walls later on: C(x)=C0+C0*(exp(u*x/D)-1)/(u/D*exp(u*x/D)-exp(u*L/D)+1)
                
                #Elongation BC
                if Barrier==0: #No elongation from the Casparian strip on
                    for wall_id in range(network.n_walls):
                        rhs_e[wall_id][0]=network.wall_lengths[wall_id]*geometry.thickness/2*1.0E-08*(Elong_cell[0][count]+(x_rel[wall_id]-0.5)*Elong_cell_side_diff[0][count])*boundary.water_fraction_apo #cm^3/d Cell wall horizontal surface assumed to be rectangular (junctions are pointwise elements)
                    for cid in range(network.n_cells):
                        if network.cell_areas[cid]>network.cell_perimeters[cid]*geometry.thickness/2:
                            rhs_e[network.n_wall_junction+cid][0]=(network.cell_areas[cid]-network.cell_perimeters[cid]*geometry.thickness/2)*1.0E-8*(Elong_cell[0][count]+(x_rel[network.n_wall_junction+cid]-0.5)*Elong_cell_side_diff[0][count])*boundary.water_fraction_sym #cm^3/d Wall geometry.thickness removed from cell horizontal area to obtain protoplast horizontal area
                        else:
                            rhs_e[network.n_wall_junction+cid][0]=0 #The cell elongation virtually does not imply water influx, though its walls do (typically intercellular spaces

                Psi_sieve[iMaturity][count]=float(boundary.bc_sieve_elems[count].get("pressure")) #Phloem sieve element pressure potential (hPa)
                dPsi_sieve[iMaturity][count]=float(boundary.bc_sieve_elems[count].get("deltaP")) #Phloem pressure potential change as compared to equilibrium pressure (hPa)
                Flow_sieve[0][count]=float(boundary.bc_sieve_elems[count].get("flowrate")) #Phloem flow rate (cm^3/d)
                if not isnan(Flow_sieve[0][count]):
                    if isnan(Psi_sieve[iMaturity][count]) and isnan(dPsi_sieve[iMaturity][count]):
                        if Barrier==0:
                            if Flow_sieve[0][count]==0:
                                tot_flow=-float(sum(rhs_e)) #"Equilibrium condition" with phloem water fully used for elongation
                            else:
                                tot_flow=Flow_sieve[0][count]
                            sum_area=0
                            i=1
                            for cid in network.protosieve_list:
                                area=network.cell_areas[cid-network.n_wall_junction]
                                Flow_sieve[i][count]=tot_flow*area
                                sum_area+=area
                                i+=1
                            i=1
                            for cid in network.protosieve_list:
                                Flow_sieve[i][count]/=sum_area #Total phloem flow rate partitioned proportionnally to phloem cross-section area
                                i+=1
                        elif Barrier>0:
                            tot_flow=Flow_sieve[0][count]
                            sum_area=0
                            i=1
                            for cid in network.sieve_cells:
                                area=network.cell_areas[cid-network.n_wall_junction]
                                Flow_sieve[i][count]=tot_flow*area
                                sum_area+=area
                                i+=1
                            i=1
                            for cid in network.sieve_cells:
                                Flow_sieve[i][count]/=sum_area #Total phloem flow rate partitioned proportionnally to phloem cross-section area
                                i+=1
                        if Flow_sieve[0][count]==0.0:
                            iEquil_sieve=count
                    else:
                        print('Error: Cannot have both pressure and flow BC at phloem boundary')
                elif not isnan(dPsi_sieve[iMaturity][count]):
                    if isnan(Psi_sieve[iMaturity][count]):
                        if not isnan(iEquil_sieve):
                            Psi_sieve[iMaturity][count]=Psi_sieve[iMaturity][iEquil_sieve]+dPsi_sieve[iMaturity][count]
                        else:
                            print('Error: Cannot have phloem pressure change relative to equilibrium without having a prior scenario with equilibrium phloem boundary condition')
                    else:
                        print('Error: Cannot have both pressure and pressure change relative to equilibrium as phloem boundary condition')
                
                jmb=0 #Index for membrane conductance vector
                for node, edges in network.graph.adjacency() : #adjacency_iter returns an iterator of (node, adjacency dict) tuples for all nodes. This is the fastest way to look at every edge. For directed graphs, only outgoing adjacencies are included.
                    i=indice[node] #Node ID number
                    #Here we count surrounding cell types in order to identify on which side of the endodermis or exodermis we are.
                    count_endo=0 #total number of endodermis cells around the wall
                    count_stele_overall=0 #total number of stelar cells around the wall
                    count_exo=0 #total number of exodermis cells around the wall
                    count_epi=0 #total number of epidermis cells around the wall
                    count_cortex=0 #total number of cortical cells around the wall
                    count_passage=0 #total number of passage cells around the wall
                    if i<network.n_walls: #wall ID
                        if boundary.scenarios[count]['osmotic_symmetry_soil'] == 2: #Central symmetrical gradient for apoplastic osmotic potential
                            if boundary.scenarios[count]['osmotic_diffusivity_soil'] == 0: #Not the analytical solution
                                Os_soil_local=float(boundary.scenarios[count]['osmotic_left_soil']+(boundary.scenarios[count]['osmotic_right_soil']-boundary.scenarios[count]['osmotic_left_soil'])*abs(r_rel[i])**boundary.scenarios[count]['osmotic_shape_soil'])
                            else:
                                if r_rel[i]>=0: #cortical side
                                    Os_soil_local=boundary.scenarios[count]['osmotic_left_soil']*exp(u[0][0]*abs(r_rel[i])*L_diff[0]/boundary.scenarios[count]['osmotic_diffusivity_soil'])
                        elif boundary.scenarios[count]['osmotic_symmetry_soil'] == 1: #Left-right gradient for apoplastic osmotic potential
                            Os_soil_local=float(boundary.scenarios[count]['osmotic_left_soil']*(1-x_rel[i])+boundary.scenarios[count]['osmotic_right_soil']*x_rel[i])
                        if boundary.scenarios[count]['osmotic_symmetry_xyl'] == 2:
                            if boundary.scenarios[count]['osmotic_diffusivity_xyl'] == 0: #Not the analytical solution
                                Os_xyl_local=float(boundary.scenarios[count]['osmotic_endo']+(boundary.scenarios[count]['osmotic_xyl']-boundary.scenarios[count]['osmotic_endo'])*(1-abs(r_rel[i]))**boundary.scenarios[count]['osmotic_shape_xyl'])
                            else:
                                if r_rel[i]<0: #cortical side
                                    Os_xyl_local=boundary.scenarios[count]['osmotic_xyl']*exp(-u[1][0]*abs(r_rel[i])*L_diff[1]/boundary.scenarios[count]['osmotic_diffusivity_xyl'])
                        elif boundary.scenarios[count]['osmotic_symmetry_xyl'] == 1:
                            Os_xyl_local=float((boundary.scenarios[count]['osmotic_xyl']+boundary.scenarios[count]['osmotic_endo'])/2)
                        for neighboor, eattr in edges.items(): #Loop on connections (edges)
                            if eattr['path'] == 'membrane': #Wall connection
                                if any(passage_cell_ID==array((indice[neighboor])-network.n_wall_junction)):
                                    count_passage+=1
                                if network.graph.nodes[neighboor]['cgroup']==3:#Endodermis
                                    count_endo+=1
                                elif network.graph.nodes[neighboor]['cgroup']>4:#Pericycle or stele
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
                            if path == "membrane": #Membrane connection
                                #Cell and wall osmotic potentials (cell types: 1=Exodermis;2=epidermis;3=endodermis;4=cortex;5=stele;16=pericycle)
                                rank=int(network.cell_ranks[int(j-network.n_wall_junction)])
                                row=int(network.rank_to_row[rank][0])
                                if rank==1:#Exodermis
                                    Os_cells[j-network.n_wall_junction]=Os_exo
                                    Os_membranes[jmb][1]=Os_exo
                                    OsCellLayer[row][iMaturity][count]+=Os_exo
                                    nOsCellLayer[row][iMaturity][count]+=1
                                    OsCellLayer[row+1][iMaturity][count]+=Os_exo
                                    nOsCellLayer[row+1][iMaturity][count]+=1
                                    Os_walls[i]=Os_soil_local
                                    if count_epi==1: #wall between exodermis and epidermis
                                        s_membranes[jmb]=s_exo_epi
                                        OsWallLayer[row+1][iMaturity][count]+=Os_soil_local
                                        nOsWallLayer[row+1][iMaturity][count]+=1
                                    elif count_epi==0: #wall between exodermis and cortex or between two exodermal cells
                                        s_membranes[jmb]=s_exo_cortex
                                        OsWallLayer[row][iMaturity][count]+=Os_soil_local
                                        nOsWallLayer[row][iMaturity][count]+=1
                                elif rank==2:#Epidermis
                                    Os_cells[j-network.n_wall_junction]=Os_epi
                                    Os_membranes[jmb][1]=Os_epi
                                    OsCellLayer[row][iMaturity][count]+=Os_epi
                                    nOsCellLayer[row][iMaturity][count]+=1
                                    Os_walls[i]=Os_soil_local
                                    s_membranes[jmb]=s_epi
                                    OsWallLayer[row][iMaturity][count]+=Os_soil_local
                                    nOsWallLayer[row][iMaturity][count]+=1
                                elif rank==3:#Endodermis
                                    Os_cells[j-network.n_wall_junction]=Os_endo
                                    Os_membranes[jmb][1]=Os_endo
                                    OsCellLayer[row][iMaturity][count]+=Os_endo
                                    nOsCellLayer[row][iMaturity][count]+=1
                                    OsCellLayer[row+3][iMaturity][count]+=Os_endo
                                    nOsCellLayer[row+3][iMaturity][count]+=1
                                    if count_stele_overall==0 and count_cortex>0: #wall between endodermis and cortex or between two endodermal cells
                                        Os_walls[i]=Os_soil_local
                                        s_membranes[jmb]=s_endo_cortex
                                        #Not including the osmotic potential of walls that are located at the same place as the casparian strip
                                        OsWallLayer[row+3][iMaturity][count]+=Os_soil_local
                                        nOsWallLayer[row+3][iMaturity][count]+=1
                                    elif count_stele_overall>0 and count_endo>0: #wall between endodermis and pericycle
                                        if Barrier==0: #No apoplastic barrier
                                            Os_walls[i]=Os_soil_local
                                            OsWallLayer[row][iMaturity][count]+=Os_soil_local
                                            nOsWallLayer[row][iMaturity][count]+=1
                                        else:
                                            Os_walls[i]=Os_xyl_local #float(boundary.scenarios[count]['osmotic_xyl'])
                                            OsWallLayer[row][iMaturity][count]+=Os_xyl_local #float(boundary.scenarios[count]['osmotic_xyl'])
                                            nOsWallLayer[row][iMaturity][count]+=1
                                        s_membranes[jmb]=s_endo_peri
                                    else: #Wall between endodermal cells
                                        if Barrier==0: #No apoplastic barrier
                                            Os_walls[i]=Os_soil_local
                                            OsWallLayer[row][iMaturity][count]+=Os_soil_local
                                            nOsWallLayer[row][iMaturity][count]+=1
                                        else:
                                            Os_walls[i]=Os_xyl_local #float(boundary.scenarios[count]['osmotic_xyl'])
                                            OsWallLayer[row][iMaturity][count]+=Os_xyl_local #float(boundary.scenarios[count]['osmotic_xyl'])
                                            nOsWallLayer[row][iMaturity][count]+=1
                                        s_membranes[jmb]=s_endo_peri
                                elif rank>=40 and rank<50:#Cortex
                                    if j-network.n_wall_junction in geometry.intercellular_ids: 
                                        Os_cells[j-network.n_wall_junction]=Os_soil_local
                                        Os_walls[i]=Os_soil_local
                                        Os_membranes[jmb][1]=Os_soil_local
                                        Os_membranes[jmb][0]=Os_soil_local
                                        s_membranes[jmb]=0
                                        OsWallLayer[row][iMaturity][count]+=Os_soil_local
                                        nOsWallLayer[row][iMaturity][count]+=1
                                    else:
                                        if row==network.row_outer_cortex-7:
                                            Os_cells[j-network.n_wall_junction]=Os_c8
                                            Os_membranes[jmb][1]=Os_c8
                                            OsCellLayer[row][iMaturity][count]+=Os_c8
                                            nOsCellLayer[row][iMaturity][count]+=1
                                        elif row==network.row_outer_cortex-6:
                                            Os_cells[j-network.n_wall_junction]=Os_c7
                                            Os_membranes[jmb][1]=Os_c7
                                            OsCellLayer[row][iMaturity][count]+=Os_c7
                                            nOsCellLayer[row][iMaturity][count]+=1
                                        elif row==network.row_outer_cortex-5:
                                            Os_cells[j-network.n_wall_junction]=Os_c6
                                            Os_membranes[jmb][1]=Os_c6
                                            OsCellLayer[row][iMaturity][count]+=Os_c6
                                            nOsCellLayer[row][iMaturity][count]+=1
                                        elif row==network.row_outer_cortex-4:
                                            Os_cells[j-network.n_wall_junction]=Os_c5
                                            Os_membranes[jmb][1]=Os_c5
                                            OsCellLayer[row][iMaturity][count]+=Os_c5
                                            nOsCellLayer[row][iMaturity][count]+=1
                                        elif row==network.row_outer_cortex-3:
                                            Os_cells[j-network.n_wall_junction]=Os_c4
                                            Os_membranes[jmb][1]=Os_c4
                                            OsCellLayer[row][iMaturity][count]+=Os_c4
                                            nOsCellLayer[row][iMaturity][count]+=1
                                        elif row==network.row_outer_cortex-2:
                                            Os_cells[j-network.n_wall_junction]=Os_c3
                                            Os_membranes[jmb][1]=Os_c3
                                            OsCellLayer[row][iMaturity][count]+=Os_c3
                                            nOsCellLayer[row][iMaturity][count]+=1
                                        elif row==network.row_outer_cortex-1:
                                            Os_cells[j-network.n_wall_junction]=Os_c2
                                            Os_membranes[jmb][1]=Os_c2
                                            OsCellLayer[row][iMaturity][count]+=Os_c2
                                            nOsCellLayer[row][iMaturity][count]+=1
                                        elif row==network.row_outer_cortex:
                                            Os_cells[j-network.n_wall_junction]=Os_c1
                                            Os_membranes[jmb][1]=Os_c1
                                            OsCellLayer[row][iMaturity][count]+=Os_c1
                                            nOsCellLayer[row][iMaturity][count]+=1
                                        Os_walls[i]=Os_soil_local
                                        s_membranes[jmb]=s_cortex
                                        OsWallLayer[row][iMaturity][count]+=Os_soil_local
                                        nOsWallLayer[row][iMaturity][count]+=1
                                elif network.graph.nodes[j]['cgroup']==5:#Stelar parenchyma
                                    Os_cells[j-network.n_wall_junction]=Os_stele
                                    Os_membranes[jmb][1]=Os_stele
                                    OsCellLayer[row][iMaturity][count]+=Os_stele
                                    nOsCellLayer[row][iMaturity][count]+=1
                                    if Barrier==0: #No apoplastic barrier
                                        Os_walls[i]=Os_soil_local
                                        OsWallLayer[row][iMaturity][count]+=Os_soil_local
                                        nOsWallLayer[row][iMaturity][count]+=1
                                    else:
                                        Os_walls[i]=Os_xyl_local #float(boundary.scenarios[count]['osmotic_xyl'])
                                        OsWallLayer[row][iMaturity][count]+=Os_xyl_local #float(boundary.scenarios[count]['osmotic_xyl'])
                                        nOsWallLayer[row][iMaturity][count]+=1
                                    s_membranes[jmb]=s_stele
                                elif rank==16:#Pericycle
                                    Os_cells[j-network.n_wall_junction]=Os_peri
                                    Os_membranes[jmb][1]=Os_peri
                                    OsCellLayer[row][iMaturity][count]+=Os_peri
                                    nOsCellLayer[row][iMaturity][count]+=1
                                    if Barrier==0: #No apoplastic barrier
                                        Os_walls[i]=Os_soil_local
                                        OsWallLayer[row][iMaturity][count]+=Os_soil_local
                                        nOsWallLayer[row][iMaturity][count]+=1
                                    else:
                                        Os_walls[i]=Os_xyl_local #float(boundary.scenarios[count]['osmotic_xyl'])
                                        OsWallLayer[row][iMaturity][count]+=Os_xyl_local #float(boundary.scenarios[count]['osmotic_xyl'])
                                        nOsWallLayer[row][iMaturity][count]+=1
                                    s_membranes[jmb]=s_peri
                                elif network.graph.nodes[j]['cgroup']==11 or network.graph.nodes[j]['cgroup']==23:#Phloem sieve tube cell
                                    if not isnan(Os_sieve[0][count]):
                                        if Barrier>0 or j in network.protosieve_list:
                                            Os_cells[j-network.n_wall_junction]=float(Os_sieve[0][count])
                                            Os_membranes[jmb][1]=float(Os_sieve[0][count])
                                            OsCellLayer[row][iMaturity][count]+=float(Os_sieve[0][count])
                                            nOsCellLayer[row][iMaturity][count]+=1
                                        else:
                                            Os_cells[j-network.n_wall_junction]=Os_stele
                                            Os_membranes[jmb][1]=Os_stele
                                            OsCellLayer[row][iMaturity][count]+=Os_stele
                                            nOsCellLayer[row][iMaturity][count]+=1
                                    else:
                                        Os_cells[j-network.n_wall_junction]=Os_stele
                                        Os_membranes[jmb][1]=Os_stele
                                        OsCellLayer[row][iMaturity][count]+=Os_stele
                                        nOsCellLayer[row][iMaturity][count]+=1
                                    if Barrier==0: #No apoplastic barrier
                                        Os_walls[i]=Os_soil_local
                                        OsWallLayer[row][iMaturity][count]+=Os_soil_local
                                        nOsWallLayer[row][iMaturity][count]+=1
                                    else:
                                        Os_walls[i]=Os_xyl_local #float(boundary.scenarios[count]['osmotic_xyl'])
                                        OsWallLayer[row][iMaturity][count]+=Os_xyl_local #float(boundary.scenarios[count]['osmotic_xyl'])
                                        nOsWallLayer[row][iMaturity][count]+=1
                                    s_membranes[jmb]=s_sieve
                                elif network.graph.nodes[j]['cgroup']==12 or network.graph.nodes[j]['cgroup']==26:#Companion cell
                                    if not isnan(Os_sieve[0][count]):
                                        Os_cells[j-network.n_wall_junction]=Os_comp
                                        Os_membranes[jmb][1]=Os_comp
                                        OsCellLayer[row][iMaturity][count]+=Os_comp
                                        nOsCellLayer[row][iMaturity][count]+=1
                                    else:
                                        Os_cells[j-network.n_wall_junction]=Os_stele
                                        Os_membranes[jmb][1]=Os_stele
                                        OsCellLayer[row][iMaturity][count]+=Os_stele
                                        nOsCellLayer[row][iMaturity][count]+=1
                                    if Barrier==0: #No apoplastic barrier
                                        Os_walls[i]=Os_soil_local
                                        OsWallLayer[row][iMaturity][count]+=Os_soil_local
                                        nOsWallLayer[row][iMaturity][count]+=1
                                    else:
                                        Os_walls[i]=Os_xyl_local
                                        OsWallLayer[row][iMaturity][count]+=Os_xyl_local
                                        nOsWallLayer[row][iMaturity][count]+=1
                                    s_membranes[jmb]=s_comp
                                elif network.graph.nodes[j]['cgroup']==13 or network.graph.nodes[j]['cgroup']==19 or network.graph.nodes[j]['cgroup']==20:#Xylem cell or vessel
                                    if Barrier==0:
                                        Os_cells[j-network.n_wall_junction]=Os_stele
                                        Os_membranes[jmb][1]=Os_stele
                                        OsCellLayer[row][iMaturity][count]+=Os_stele
                                        nOsCellLayer[row][iMaturity][count]+=1
                                        Os_walls[i]=Os_soil_local
                                        s_membranes[jmb]=s_stele
                                        OsWallLayer[row][iMaturity][count]+=Os_soil_local
                                        nOsWallLayer[row][iMaturity][count]+=1
                                    else:
                                        Os_cells[j-network.n_wall_junction]=Os_xyl_local
                                        Os_membranes[jmb][0]=Os_xyl_local
                                        Os_membranes[jmb][1]=Os_xyl_local
                                        Os_membranes[jmb][1]=Os_xyl_local
                                        Os_walls[i]=Os_xyl_local
                                        s_membranes[jmb]=0.0
                                        OsWallLayer[row][iMaturity][count]+=Os_xyl_local #float(boundary.scenarios[count]['osmotic_xyl'])
                                        nOsWallLayer[row][iMaturity][count]+=1
                                K=Kmb[jmb][0]
                                rhs_o[i]+= K*s_membranes[jmb]*(Os_walls[i] - Os_cells[j-network.n_wall_junction]) #Wall node
                                rhs_o[j]+= K*s_membranes[jmb]*(Os_cells[j-network.n_wall_junction] - Os_walls[i]) #Cell node 
                                jmb+=1
                for row in range(int(network.r_discret[0][0])):
                    if nOsWallLayer[row][iMaturity][count]>0:
                        OsWallLayer[row][iMaturity][count]=OsWallLayer[row][iMaturity][count]/nOsWallLayer[row][iMaturity][count]
                    if nOsCellLayer[row][iMaturity][count]>0:
                        OsCellLayer[row][iMaturity][count]=OsCellLayer[row][iMaturity][count]/nOsCellLayer[row][iMaturity][count]
                
                #Xylem BC
                if Barrier>0: #No mature xylem before the Casparian strip stage
                    if not isnan(Psi_xyl[iMaturity][count]): #Pressure xylem BC
                        for cid in network.xylem_cells:
                            rhs_x[cid][0] = -hydraulic.k_xyl  #Axial conductance of xylem vessels
                            matrix_W[cid][cid] -= hydraulic.k_xyl
                    elif not isnan(Flow_xyl[0][count]): #Flow xylem BC
                        i=1
                        for cid in network.xylem_cells:
                            rhs_x[cid][0] = Flow_xyl[i][count] #(cm^3/d)
                            i+=1
                
                #Phloem BC
                if Barrier==0: #Protophloem only
                    if not isnan(Psi_sieve[iMaturity][count]):
                        for cid in network.protosieve_list:
                            rhs_p[cid][0] = -hydraulic.k_sieve  #Axial conductance of phloem sieve tube
                            matrix_W[cid][cid] -= hydraulic.k_sieve
                    elif not isnan(Flow_sieve[0][count]):
                        i=1
                        for cid in network.protosieve_list:
                            rhs_p[cid][0] = Flow_sieve[i][count] #(cm^3/d)
                            i+=1
                elif Barrier>0: #Includes mature phloem
                    if not isnan(Psi_sieve[iMaturity][count]): #Then there is a phloem BC in scenarios (assuming that we did not pick scenarios with and others without)
                        for cid in network.sieve_cells: #both proto and metaphloem
                            rhs_p[cid][0] = -hydraulic.k_sieve  #Axial conductance of xylem vessels
                            matrix_W[cid][cid] -= hydraulic.k_sieve
                    elif not isnan(Flow_sieve[0][count]):
                        i=1
                        for cid in network.sieve_cells:
                            rhs_p[cid][0] = Flow_sieve[i][count] #(cm^3/d)
                            i+=1
                
                
                #Adding up all BC
                #Elongation BC
                rhs += rhs_e
                
                #Osmotic BC
                rhs += rhs_o
                
                #Soil BC
                rhs += np.multiply(rhs_s,boundary.scenarios[count]['psi_soil_left']*(1-x_rel)+boundary.scenarios[count]['psi_soil_right']*x_rel)
                
                #Xylem BC
                if not isnan(Psi_xyl[iMaturity][count]): #Pressure xylem BC
                    rhs += rhs_x*Psi_xyl[iMaturity][count]  #multiplication of rhs components delayed till this point so that rhs_s & rhs_x can be re-used
                elif not isnan(Flow_xyl[0][count]): #Flow xylem BC
                    rhs += rhs_x
                
                #Phloem BC
                if not isnan(Flow_sieve[0][count]):
                    rhs += rhs_p
                elif not isnan(Psi_sieve[iMaturity][count]):
                    rhs += rhs_p*Psi_sieve[iMaturity][count]
                
                ##################################################
                ##Solve Doussan equation, results in soln matrix##
                ##################################################
                
                soln = np.linalg.solve(matrix_W,rhs) #Solving the equation to get potentials inside the network
                
                #Verification that computation was correct
                verif1=np.allclose(np.dot(matrix_W,soln),rhs)
                
                #Removing Xylem and phloem BC terms in "matrix" in case they would change in the next scenario
                if Barrier>0:
                    if not isnan(Psi_xyl[iMaturity][count]): #Pressure xylem BC
                        for cid in network.xylem_cells:
                            matrix_W[cid][cid] += hydraulic.k_xyl
                if Barrier==0: #Protophloem only
                    if not isnan(Psi_sieve[iMaturity][count]):
                        for cid in network.protosieve_list:
                            matrix_W[cid][cid] += hydraulic.k_sieve
                elif Barrier>0: #Includes mature phloem
                    if not isnan(Psi_sieve[iMaturity][count]): #Then there is a phloem BC in scenarios (assuming that we did not pick scenarios with and others without)
                        for cid in network.sieve_cells: #both proto and metaphloem
                            matrix_W[cid][cid] += hydraulic.k_sieve
                
                #Flow rates at interfaces
                Q_soil=[]
                for ind in network.border_walls:
                    Q=rhs_s[ind]*(soln[ind]-(boundary.scenarios[count]['psi_soil_left']*(1-x_rel[ind])+boundary.scenarios[count]['psi_soil_right']*x_rel[ind]))
                    Q_soil.append(Q) #(cm^3/d) Positive for water flowing into the root, rhs_s is minus the conductance at the soil root interface
                    if general.apo_contagion==2:
                        if general.sym_contagion==2:
                            if ind not in network.apo_wall_zombies0:
                                if Q<0.0:
                                    matrix_C[ind][ind]+=Q
                        else:
                            if ind not in network.apo_wall_zombies0:
                                if Q<0.0:
                                    matrix_ApoC[ind][ind]+=Q
                            
                for ind in network.border_junction:
                    Q=rhs_s[ind]*(soln[ind]-(boundary.scenarios[count]['psi_soil_left']*(1-x_rel[ind])+boundary.scenarios[count]['psi_soil_right']*x_rel[ind]))
                    Q_soil.append(Q) #(cm^3/d) Positive for water flowing into the root
                    if general.apo_contagion==2:
                        if general.sym_contagion==2:
                            if ind not in Apo_j_Zombies0:
                                if Q<0.0:
                                    matrix_C[ind][ind]+=Q
                        else:
                            if ind not in Apo_j_Zombies0:
                                if Q<0.0:
                                    matrix_ApoC[ind][ind]+=Q
                
                Q_xyl=[]
                if Barrier>0:
                    if not isnan(Psi_xyl[iMaturity][count]): #Xylem pressure BC
                        for cid in network.xylem_cells:
                            Q=rhs_x[cid][0]*(soln[cid][0]-Psi_xyl[iMaturity][count])
                            Q_xyl.append(Q) #(cm^3/d) Negative for water flowing into xylem tubes
                            rank=int(network.cell_ranks[cid-network.n_wall_junction][0])
                            row=int(network.rank_to_row[rank][0])
                            Q_xyl_layer[row][iMaturity][count] += Q
                            
                    elif not isnan(Flow_xyl[0][count]): #Xylem flow BC
                        for cid in network.xylem_cells:
                            Q=-rhs_x[cid][0]
                            Q_xyl.append(Q) #(cm^3/d) Negative for water flowing into xylem tubes
                            rank=int(network.cell_ranks[cid-network.n_wall_junction][0])
                            row=int(network.rank_to_row[rank][0])
                            Q_xyl_layer[row][iMaturity][count] += Q
                            #if boundary.c_flag:
                            #    if Q>0: #Water leaving the cross-section
                            #        matrix_C[cid][cid] -= Q
                            #    else: #Water entering the cross-section through xylem
                            #        rhs_C[cid][0] += Q #Flow rate times concentration BC
                            #    rhs_C[cid][0] *= boundary.scenarios[count]['osmotic_endo']
                
                Q_sieve=[]
                if Barrier==0:
                    if not isnan(Psi_sieve[iMaturity][count]): #Phloem pressure BC
                        for cid in network.protosieve_list: #Q will be 0 for metaphloem if Barrier==0 because rhs_p=0 for these cells
                            Q=rhs_p[cid]*(soln[cid][0]-Psi_sieve[iMaturity][count])
                            Q_sieve.append(Q) #(cm^3/d) Positive for water flowing from sieve tubes
                            rank=int(network.cell_ranks[cid-network.n_wall_junction][0])
                            row=int(network.rank_to_row[rank][0])
                            Q_sieve_layer[row][iMaturity][count] += Q
                    elif not isnan(Flow_sieve[0][count]): #Phloem flow BC
                        for cid in network.protosieve_list:
                            Q=-rhs_p[cid]
                            Q_sieve.append(Q) #(cm^3/d) Negative for water flowing into xylem tubes
                            rank=int(network.cell_ranks[cid-network.n_wall_junction][0])
                            row=int(network.rank_to_row[rank][0])
                            Q_sieve_layer[row][iMaturity][count] += Q
                elif Barrier>0:
                    if not isnan(Psi_sieve[iMaturity][count]): #Phloem pressure BC
                        for cid in network.sieve_cells: #Q will be 0 for metaphloem if Barrier==0 because rhs_p=0 for these cells
                            Q=rhs_p[cid]*(soln[cid][0]-Psi_sieve[iMaturity][count])
                            Q_sieve.append(Q) #(cm^3/d) Positive for water flowing from sieve tubes
                            rank=int(network.cell_ranks[cid-network.n_wall_junction][0])
                            row=int(network.rank_to_row[rank][0])
                            Q_sieve_layer[row][iMaturity][count] += Q
                    elif not isnan(Flow_sieve[0][count]): #Phloem flow BC
                        for cid in network.sieve_cells:
                            Q=-rhs_p[cid]
                            Q_sieve.append(Q) #(cm^3/d) Negative for water flowing into xylem tubes
                            rank=int(network.cell_ranks[cid-network.n_wall_junction][0])
                            row=int(network.rank_to_row[rank][0])
                            Q_sieve_layer[row][iMaturity][count] += Q
                Q_elong=-rhs_e #(cm^3/d) The elongation flux virtually disappears from the cross-section => negative
                for cid in range(network.n_cells):
                    rank=int(network.cell_ranks[cid])
                    row=int(network.rank_to_row[rank][0])
                    Q_elong_layer[row][iMaturity][count] += Q_elong[network.n_wall_junction+cid]
                Q_tot[iMaturity][count]=sum(Q_soil) #(cm^3/d) Total flow rate at root surface
                for ind in range(network.n_wall_junction,len(network.graph.nodes)): #network.n_wall_junction is the index of the first cell
                    cell_id=ind-network.n_wall_junction
                    rank = int(network.cell_ranks[cell_id])
                    row = int(network.rank_to_row[rank][0])
                    if rank == 1: #Exodermis
                        PsiCellLayer[row][iMaturity][count] += soln[ind]*(STFcell_plus[cell_id][iMaturity]+abs(STFcell_minus[cell_id][iMaturity]))/(STFlayer_plus[row][iMaturity]+abs(STFlayer_minus[row][iMaturity])+STFlayer_plus[row+1][iMaturity]+abs(STFlayer_minus[row+1][iMaturity])) #(hPa)
                        PsiCellLayer[row+1][iMaturity][count] += soln[ind]*(STFcell_plus[cell_id][iMaturity]+abs(STFcell_minus[cell_id][iMaturity]))/(STFlayer_plus[row][iMaturity]+abs(STFlayer_minus[row][iMaturity])+STFlayer_plus[row+1][iMaturity]+abs(STFlayer_minus[row+1][iMaturity])) #(hPa)
                    elif rank == 3: #Endodermis
                        if any(passage_cell_ID==array(cell_id)) and Barrier==2: #Passage cell
                            PsiCellLayer[row+1][iMaturity][count] += soln[ind]*(STFcell_plus[cell_id][iMaturity]+abs(STFcell_minus[cell_id][iMaturity]))/(STFlayer_plus[row+1][iMaturity]+abs(STFlayer_minus[row+1][iMaturity])+STFlayer_plus[row+2][iMaturity]+abs(STFlayer_minus[row+2][iMaturity])) #(hPa)
                            PsiCellLayer[row+2][iMaturity][count] += soln[ind]*(STFcell_plus[cell_id][iMaturity]+abs(STFcell_minus[cell_id][iMaturity]))/(STFlayer_plus[row+1][iMaturity]+abs(STFlayer_minus[row+1][iMaturity])+STFlayer_plus[row+2][iMaturity]+abs(STFlayer_minus[row+2][iMaturity])) #(hPa)
                        else:
                            PsiCellLayer[row][iMaturity][count] += soln[ind]*(STFcell_plus[cell_id][iMaturity]+abs(STFcell_minus[cell_id][iMaturity]))/(STFlayer_plus[row][iMaturity]+abs(STFlayer_minus[row][iMaturity])+STFlayer_plus[row+3][iMaturity]+abs(STFlayer_minus[row+3][iMaturity])) #(hPa)
                            PsiCellLayer[row+3][iMaturity][count] += soln[ind]*(STFcell_plus[cell_id][iMaturity]+abs(STFcell_minus[cell_id][iMaturity]))/(STFlayer_plus[row][iMaturity]+abs(STFlayer_minus[row][iMaturity])+STFlayer_plus[row+3][iMaturity]+abs(STFlayer_minus[row+3][iMaturity])) #(hPa)
                            if not Barrier==2:
                                PsiCellLayer[row+1][iMaturity][count] = nan
                                PsiCellLayer[row+2][iMaturity][count] = nan
                    elif (ind not in network.xylem_cells) or Barrier==0: #Not for mature xylem
                        PsiCellLayer[row][iMaturity][count] += soln[ind]*(STFcell_plus[cell_id][iMaturity]+abs(STFcell_minus[cell_id][iMaturity]))/(STFlayer_plus[row][iMaturity]+abs(STFlayer_minus[row][iMaturity])) #(hPa)
                
                if Barrier>0 and isnan(Psi_xyl[iMaturity][count]):
                    Psi_xyl[iMaturity][count]=0.0
                    for cid in network.xylem_cells:
                        Psi_xyl[iMaturity][count]+=soln[cid][0]/len(network.xylem_cells)
                if Barrier>0:
                    if isnan(Psi_sieve[iMaturity][count]):
                        Psi_sieve[iMaturity][count]=0.0
                        for cid in network.sieve_cells:
                            Psi_sieve[iMaturity][count]+=soln[cid][0]/network.n_sieve #Average of phloem water pressures
                elif Barrier==0:
                    if isnan(Psi_sieve[iMaturity][count]):
                        Psi_sieve[iMaturity][count]=0.0
                        for cid in network.protosieve_list:
                            Psi_sieve[iMaturity][count]+=soln[cid][0]/network.n_protosieve #Average of protophloem water pressures
                
                print("Uptake rate per unit root length: soil ",(sum(Q_soil)/height/1.0E-04),"cm^2/d, xylem ",(sum(Q_xyl)/height/1.0E-04),"cm^2/d, phloem ",(sum(Q_sieve)/height/1.0E-04),"cm^2/d, elongation ",(sum(Q_elong)/height/1.0E-04),"cm^2/d")
                if not isnan(sum(Q_sieve)):
                    print("Mass balance error:",(sum(Q_soil)+sum(Q_xyl)+sum(Q_sieve)+sum(Q_elong))/height/1.0E-04,"cm^2/d")
                else:
                    print("Mass balance error:",(sum(Q_soil)+sum(Q_xyl)+sum(Q_elong))/height/1.0E-04,"cm^2/d")
                
                #################################################################
                ##Calul of Fluxes between nodes and Creating the edge_flux_list##
                #################################################################
                
                #Creating a list for the fluxes
                #edge_flux_list=[]
                
                #Filling the fluxes list
                MembraneFlowDensity=[]
                WallFlowDensity=[]
                WallFlowDensity_cos=[]
                PlasmodesmFlowDensity=[]
                Fjw_list=[]
                Fcw_list=[]
                Fcc_list=[]
                jmb=0 #Index for membrane conductance vector
                for node, edges in network.graph.adjacency() : #adjacency_iter returns an iterator of (node, adjacency dict) tuples for all nodes. This is the fastest way to look at every edge. For directed graphs, only outgoing adjacencies are included.
                    i = indice[node] #Node ID number
                    psi = soln[i][0]  #Node water potential
                    psi_o_cell = inf #Opposite cell water potential
                    ind_o_cell = inf #Opposite cell index
                    #Here we count surrounding cell types in order to know if the wall is part of an apoplastic barrier, as well as to know on which side of the exodermis or endodermis the membrane is located
                    count_endo=0 #total number of endodermis cells around the wall
                    count_peri=0 #total number of pericycle cells around the wall
                    count_PPP=0 #total number of network.plasmodesmata_indice cells arount the wall
                    count_exo=0 #total number of exodermis cells around the wall
                    count_epi=0 #total number of epidermis cells around the wall
                    count_stele=0 #total number of stelar parenchyma cells around the wall
                    count_stele_overall=0 #total number of stele cells (of any type) around the wall
                    count_comp=0 #total number of companion cells around the wall
                    count_sieve=0 #total number of stelar parenchyma cells around the wall
                    count_xyl=0 #total number of xylem cells around the wall
                    count_cortex=0 #total number of phloem sieve cells around the wall
                    count_passage=0 #total number of passage cells around the wall
                    count_interC=0 #total number of intercellular spaces around the wall
                    noPD=False #Initializes the flag for wall connected to an intercellular space -> does not have plasmodesmata
                    if i<network.n_walls: #wall ID
                        for neighboor, eattr in edges.items(): #Loop on connections (edges)
                            if eattr['path'] == 'membrane': #Wall connection
                                if any(passage_cell_ID==array((indice[neighboor])-network.n_wall_junction)):
                                    count_passage+=1
                                if any(geometry.intercellular_ids==array((indice[neighboor])-network.n_wall_junction)):
                                    count_interC+=1
                                if network.graph.nodes[neighboor]['cgroup']==3:#Endodermis
                                    count_endo+=1
                                elif network.graph.nodes[neighboor]['cgroup']==13 or network.graph.nodes[neighboor]['cgroup']==19 or network.graph.nodes[neighboor]['cgroup']==20:#Xylem cell or vessel
                                    count_xyl+=1
                                elif network.graph.nodes[neighboor]['cgroup']==16 or network.graph.nodes[neighboor]['cgroup']==21:#Pericycle or stele
                                    count_peri+=1
                                    if neighboor in network.plasmodesmata_indice:
                                        count_PPP+=1
                                elif network.graph.nodes[neighboor]['cgroup']==1:#Exodermis
                                    count_exo+=1
                                elif network.graph.nodes[neighboor]['cgroup']==2:#Epidermis
                                    count_epi+=1
                                elif network.graph.nodes[neighboor]['cgroup']==4:#Cortex
                                    count_cortex+=1
                                elif network.graph.nodes[neighboor]['cgroup']==5:#Stelar parenchyma
                                    count_stele+=1
                                elif network.graph.nodes[neighboor]['cgroup']==11 or network.graph.nodes[neighboor]['cgroup']==23:#Phloem sieve tube
                                    count_sieve+=1
                                elif network.graph.nodes[neighboor]['cgroup']==12 or network.graph.nodes[neighboor]['cgroup']==26:#Companion cell
                                    count_comp+=1
                                if network.graph.nodes[neighboor]['cgroup']>4:#Stele overall
                                    count_stele_overall+=1
                    ijunction=0
                    for neighboor, eattr in edges.items(): #Loop on connections (edges)
                        j = indice[neighboor] #Neighbouring node ID number
                        #if j > i: #Only treating the information one way to save time
                        psin = soln[j][0] #Neighbouring node water potential
                        path = eattr['path'] #eattr is the edge attribute (i.e. connection type)
                        if i<network.n_walls:
                            if general.paraview==1 or general.par_track==1 or general.apo_contagion>0 or general.sym_contagion>0:
                                if path == "wall":
                                    #K = eattr['kw']*1.0E-04*((eattr['lateral_distance']+height)*eattr['geometry.thickness']-square(eattr['geometry.thickness']))/eattr['length'] #Junction-Wall conductance (cm^3/hPa/d)
                                    if (count_interC>=2 and Barrier>0) or (count_xyl==2 and geometry.xylem_pieces): #"Fake wall" splitting an intercellular space or a xylem cell in two
                                        K = 1.0E-16 #Non conductive
                                    elif count_cortex>=2: #wall between two cortical cells
                                        K = kw_cortex_cortex*1.0E-04*((eattr['lateral_distance']+height)*geometry.thickness-square(geometry.thickness))/eattr['length'] #Junction-Wall conductance (cm^3/hPa/d)
                                    elif count_endo>=2: #wall between two endodermis cells
                                        K = kw_endo_endo*1.0E-04*((eattr['lateral_distance']+height)*geometry.thickness-square(geometry.thickness))/eattr['length'] #Junction-Wall conductance (cm^3/hPa/d)
                                    elif count_stele_overall>0 and count_endo>0: #wall between endodermis and pericycle
                                        if count_passage>0:
                                            K = kw_passage*1.0E-04*((eattr['lateral_distance']+height)*geometry.thickness-square(geometry.thickness))/eattr['length']
                                        else:
                                            K = kw_endo_peri*1.0E-04*((eattr['lateral_distance']+height)*geometry.thickness-square(geometry.thickness))/eattr['length'] #Junction-Wall conductance (cm^3/hPa/d)
                                    elif count_stele_overall==0 and count_endo==1: #wall between endodermis and cortex
                                        if count_passage>0:
                                            K = kw_passage*1.0E-04*((eattr['lateral_distance']+height)*geometry.thickness-square(geometry.thickness))/eattr['length']
                                        else:
                                            K = kw_endo_cortex*1.0E-04*((eattr['lateral_distance']+height)*geometry.thickness-square(geometry.thickness))/eattr['length'] #Junction-Wall conductance (cm^3/hPa/d)
                                    elif count_exo>=2: #wall between two exodermis cells
                                        K = kw_exo_exo*1.0E-04*((eattr['lateral_distance']+height)*geometry.thickness-square(geometry.thickness))/eattr['length'] #Junction-Wall conductance (cm^3/hPa/d)
                                    else: #other walls
                                        K = kw*1.0E-04*((eattr['lateral_distance']+height)*geometry.thickness-square(geometry.thickness))/eattr['length'] #Junction-Wall conductance (cm^3/hPa/d)
                                    Fjw = K * (psin - psi) * sign(j-i) #(cm^3/d) Water flow rate positive from junction to wall
                                    Fjw_list.append((i,j,Fjw))
                                    #The ordering in WallFlowDensity will correspond to the one of thick_wall_x, saved for display only
                                    WallFlowDensity.append((i,j, Fjw / (((eattr['lateral_distance']+height)*geometry.thickness-square(geometry.thickness))*1.0E-08))) # (cm/d) Positive towards lower node ID 
                                    cos_angle=(position[i][0]-position[j][0])/(hypot(position[j][0]-position[i][0],position[j][1]-position[i][1])) #Vectors junction1-wall
                                    WallFlowDensity_cos.append((i,j, cos_angle * Fjw / (((eattr['lateral_distance']+height)*geometry.thickness-square(geometry.thickness))*1.0E-08))) # (cm/d) Positive towards lower node ID 
                                    #if boundary.c_flag and Os_soil[5][count]*Os_xyl[5][count]==1:
                                    if general.apo_contagion==2:
                                        if general.sym_contagion==2: # Apo & Sym contagion
                                            if Fjw>0: #Flow from junction to wall
                                                if i not in network.apo_wall_zombies0:
                                                    matrix_C[i][j] += Fjw
                                                if j not in Apo_j_Zombies0:
                                                    matrix_C[j][j] -= Fjw
                                            else: #Flow from wall to junction
                                                if i not in network.apo_wall_zombies0:
                                                    matrix_C[i][i] += Fjw
                                                if j not in Apo_j_Zombies0:
                                                    matrix_C[j][i] -= Fjw
                                        else: #Only Apo contagion
                                            if Fjw>0: #Flow from junction to wall
                                                if i not in network.apo_wall_zombies0:
                                                    matrix_ApoC[i][j] += Fjw
                                                if j not in Apo_j_Zombies0:
                                                    matrix_ApoC[j][j] -= Fjw
                                            else: #Flow from wall to junction
                                                if i not in network.apo_wall_zombies0:
                                                    matrix_ApoC[i][i] += Fjw
                                                if j not in Apo_j_Zombies0:
                                                    matrix_ApoC[j][i] -= Fjw
                                    
                                    if general.apo_contagion==1:
                                        if Fjw>0:
                                            Apo_connec_flow[j][nApo_connec_flow[j]]=i
                                            nApo_connec_flow[j]+=1
                                        elif Fjw<0:
                                            Apo_connec_flow[i][nApo_connec_flow[i]]=j
                                            nApo_connec_flow[i]+=1
                                elif path == "membrane": #Membrane connection
                                    #K = (eattr['hydraulic.kmb']+eattr['kaqp'])*1.0E-08*(height+eattr['dist'])*eattr['length']
                                    if network.graph.nodes[j]['cgroup']==1: #Exodermis
                                        kaqp=kaqp_exo
                                    elif network.graph.nodes[j]['cgroup']==2: #Epidermis
                                        kaqp=kaqp_epi
                                    elif network.graph.nodes[j]['cgroup']==3: #Endodermis
                                        kaqp=kaqp_endo
                                    elif network.graph.nodes[j]['cgroup']==13 or network.graph.nodes[j]['cgroup']==19 or network.graph.nodes[j]['cgroup']==20: #xylem cell or vessel
                                        if Barrier>0: #Xylem vessel
                                            kaqp=kaqp_stele*10000 #No membrane resistance because no membrane
                                            noPD=True
                                        elif Barrier==0: #Xylem cell
                                            kaqp=kaqp_stele
                                            if (count_xyl==2 and geometry.xylem_pieces):
                                                noPD=True
                                    elif network.graph.nodes[j]['cgroup']>4: #Stele and pericycle
                                        kaqp=kaqp_stele
                                    elif (j-network.n_wall_junction in geometry.intercellular_ids) and Barrier>0: #the neighbour is an intercellular space "cell"
                                        kaqp=geometry.k_interc
                                        noPD=True
                                    elif network.graph.nodes[j]['cgroup']==4: #Cortex
                                        kaqp=float(a_cortex*network.distance_center_grav[wall_id]*1.0E-04+b_cortex) #AQP activity (cm/hPa/d)
                                        if kaqp < 0:
                                            error('Error, negative kaqp in cortical cell, adjust Paqp_cortex')
                                    #Calculating conductances
                                    if count_endo>=2: #wall between two endodermis cells, in this case the suberized wall can limit the transfer of water between cell and wall
                                        if kw_endo_endo==0.00:
                                            K=0.00
                                        else:
                                            K = 1/(1/(kw_endo_endo/(geometry.thickness/2*1.0E-04))+1/(hydraulic.kmb+kaqp))*1.0E-08*(height+eattr['dist'])*eattr['length'] #(cm^3/hPa/d)
                                    elif count_exo>=2: #wall between two exodermis cells, in this case the suberized wall can limit the transfer of water between cell and wall
                                        if kw_exo_exo==0.00:
                                            K=0.00
                                        else:
                                            K = 1/(1/(kw_exo_exo/(geometry.thickness/2*1.0E-04))+1/(hydraulic.kmb+kaqp))*1.0E-08*(height+eattr['dist'])*eattr['length'] #(cm^3/hPa/d)
                                    elif count_stele_overall>0 and count_endo>0: #wall between endodermis and pericycle, in this case the suberized wall can limit the transfer of water between cell and wall
                                        if count_passage>0:
                                            K = 1/(1/(kw_passage/(geometry.thickness/2*1.0E-04))+1/(hydraulic.kmb+kaqp))*1.0E-08*(height+eattr['dist'])*eattr['length']
                                        else:
                                            if kw_endo_peri==0.00:
                                                K=0.00
                                            else:
                                                K = 1/(1/(kw_endo_peri/(geometry.thickness/2*1.0E-04))+1/(hydraulic.kmb+kaqp))*1.0E-08*(height+eattr['dist'])*eattr['length']
                                    elif count_stele_overall==0 and count_endo==1: #wall between endodermis and cortex, in this case the suberized wall can limit the transfer of water between cell and wall
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
                                            K = 1/(1/(kw/(geometry.thickness/2*1.0E-04))+1/(hydraulic.kmb+kaqp))*1.0E-08*(height+eattr['dist'])*eattr['length'] #(cm^3/hPa/d)
                                    Fcw = K * (psi - psin + s_membranes[jmb]*(Os_walls[i] - Os_cells[j-network.n_wall_junction])) #(cm^3/d) Water flow rate positive from wall to protoplast
                                    Fcw_list.append((i,j,-Fcw,s_membranes[jmb])) #Water flow rate positive from protoplast to wall
                                    #Flow densities calculation
                                    #The ordering in MembraneFlowDensity will correspond to the one of thick_wall, saved for display only 
                                    MembraneFlowDensity.append(Fcw / (1.0E-08*(height+eattr['dist'])*eattr['length']))
                                    ####Solute convection across membranes####
                                    if general.apo_contagion==2 and general.sym_contagion==2:
                                        if Fcw>0: #Flow from wall to protoplast
                                            if i not in network.apo_wall_zombies0:
                                                if hormones.d2o1==1:#Solute that moves across membranes like water 
                                                    matrix_C[i][i] -= Fcw
                                                else: #Solute that moves across membranes independently of water (the membrane is possibly not one) 
                                                    matrix_C[i][i] -= Fcw*(1-s_membranes[jmb])
                                            if j-network.n_wall_junction not in hormones.sym_zombie0:
                                                if hormones.d2o1==1:#Solute that moves across membranes like water 
                                                    matrix_C[j][i] += Fcw
                                                else: #Solute that moves across membranes independently of water (the membrane is possibly not one) 
                                                    matrix_C[j][i] += Fcw*(1-s_membranes[jmb])
                                        else: #Flow from protoplast to wall
                                            if j-network.n_wall_junction not in hormones.sym_zombie0:
                                                if hormones.d2o1==1:#Solute that moves across membranes like water 
                                                    matrix_C[j][j] += Fcw
                                                else: #Solute that moves across membranes independently of water (the membrane is possibly not one) 
                                                    matrix_C[j][j] += Fcw*(1-s_membranes[jmb])
                                            if i not in network.apo_wall_zombies0:
                                                if hormones.d2o1==1:#Solute that moves across membranes like water 
                                                    matrix_C[i][j] -= Fcw
                                                else: #Solute that moves across membranes independently of water (the membrane is possibly not one) 
                                                    matrix_C[i][j] -= Fcw*(1-s_membranes[jmb])
                                    
                                    #Macroscopic distributed parameter for transmembrane flow
                                    #Discretization based on cell layers and apoplasmic barriers
                                    rank = int(network.cell_ranks[j-network.n_wall_junction])
                                    row = int(network.rank_to_row[rank][0])
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
                                    Flow = K * (psi - psin + s_membranes[jmb]*(Os_walls[i] - Os_cells[j-network.n_wall_junction]))
                                    jmb+=1
                                    if ((j-network.n_wall_junction not in geometry.intercellular_ids) and (j not in network.xylem_cells)) or Barrier==0: #No aerenchyma in the elongation zone
                                        if Flow > 0 :
                                            UptakeLayer_plus[row][iMaturity][count] += Flow #grouping membrane flow rates in cell layers
                                        else:
                                            UptakeLayer_minus[row][iMaturity][count] += Flow
                                    
                                    if K>1.0e-18: #Not an impermeable wall
                                        PsiWallLayer[row][iMaturity][count] += psi
                                        NWallLayer[row][iMaturity][count] += 1
                                    
                                    if psi_o_cell == inf:
                                        psi_o_cell=psin
                                        ind_o_cell=j
                                    else:
                                        if noPD: #No plasmodesmata because the wall i is connected to an intercellular space or xylem vessel
                                            temp=0 #The ordering in PlasmodesmFlowDensity will correspond to the one of thick_wall except for boderline walls, saved for display only                        
                                        elif count_epi==1 and count_exo==1: #wall between epidermis and exodermis
                                            temp=Kpl*hydraulic.fplxheight_epi_exo * (psin - psi_o_cell)
                                        elif (count_exo==1 or count_epi==1) and count_cortex==1: #wall between exodermis and cortex
                                            temp1=float(hydraulic.kpl_elems[iPD].get("cortex_factor"))
                                            temp=Kpl*2*temp1/(temp1+1)*hydraulic.fplxheight_outer_cortex * network.len_outer_cortex / network.cross_section_outer_cortex * (psin - psi_o_cell)
                                        elif count_cortex==2: #wall between cortical cells
                                            temp1=float(hydraulic.kpl_elems[iPD].get("cortex_factor"))
                                            temp=Kpl*temp1*hydraulic.fplxheight_cortex_cortex * network.len_cortex_cortex / network.cross_section_cortex_cortex * (psin - psi_o_cell)
                                        elif count_cortex==1 and count_endo==1: #wall between cortex and endodermis
                                            temp1=float(hydraulic.kpl_elems[iPD].get("cortex_factor"))
                                            temp=Kpl*2*temp1/(temp1+1)*hydraulic.fplxheight_cortex_endo * network.len_cortex_endo / network.cross_section_cortex_endo * (psin - psi_o_cell)
                                        elif count_endo==2: #wall between endodermal cells
                                            temp=Kpl*hydraulic.fplxheight_endo_endo * (psin - psi_o_cell)
                                        elif count_stele_overall>0 and count_endo>0: #wall between endodermis and pericycle
                                            if count_PPP>0:
                                                temp1=float(hydraulic.kpl_elems[iPD].get("PPP_factor"))
                                            else:
                                                temp1=1
                                            temp=Kpl*2*temp1/(temp1+1)*hydraulic.fplxheight_endo_peri * (psin - psi_o_cell)
                                        elif count_stele==2: #wall between stelar parenchyma cells
                                            temp=Kpl*hydraulic.fplxheight_stele_stele * (psin - psi_o_cell)
                                        elif count_peri>0 and count_stele==1: #wall between stele and pericycle
                                            if count_PPP>0:
                                                temp1=float(hydraulic.kpl_elems[iPD].get("PPP_factor"))
                                            else:
                                                temp1=1
                                            temp=Kpl*2*temp1/(temp1+1)*hydraulic.fplxheight_peri_stele * (psin - psi_o_cell)
                                        elif count_comp==1 and count_stele==1: #wall between stele and companion cell
                                            temp1=float(hydraulic.kpl_elems[iPD].get("PCC_factor"))
                                            temp=Kpl*2*temp1/(temp1+1)*hydraulic.fplxheight_stele_comp * (psin - psi_o_cell)
                                        elif count_peri==1 and count_comp==1: #wall between pericycle and companion cell
                                            temp1=float(hydraulic.kpl_elems[iPD].get("PCC_factor"))
                                            if count_PPP>0:
                                                temp2=float(hydraulic.kpl_elems[iPD].get("PPP_factor"))
                                            else:
                                                temp2=1
                                            temp=Kpl*2*temp1*temp2/(temp1+temp2)*hydraulic.fplxheight_peri_comp * (psin - psi_o_cell)
                                        elif count_comp==2: #wall between companion cells 
                                            temp1=float(hydraulic.kpl_elems[iPD].get("PCC_factor"))
                                            temp=Kpl*temp1*hydraulic.fplxheight_comp_comp * (psin - psi_o_cell)
                                        elif count_comp==1 and count_sieve==1: #wall between companion cell and sieve tube
                                            temp1=float(hydraulic.kpl_elems[iPD].get("PCC_factor"))
                                            temp=Kpl*2*temp1/(temp1+1)*hydraulic.fplxheight_comp_sieve * (psin - psi_o_cell)
                                        elif count_peri==1 and count_sieve==1: #wall between stele and sieve tube
                                            temp=Kpl*hydraulic.fplxheight_peri_sieve * (psin - psi_o_cell)
                                        elif count_stele==1 and count_sieve==1: #wall between stele and pericycle
                                            if count_PPP>0:
                                                temp1=float(hydraulic.kpl_elems[iPD].get("PPP_factor"))
                                            else:
                                                temp1=1
                                            temp=Kpl*2*temp1/(temp1+1)*hydraulic.fplxheight_stele_sieve * (psin - psi_o_cell)
                                        else: #Default plasmodesmatal frequency
                                            temp=Kpl*hydraulic.fplxheight * (psin - psi_o_cell)  #The ordering in PlasmodesmFlowDensity will correspond to the one of thick_wall except for boderline walls, saved for display only 
                                        PlasmodesmFlowDensity.append(temp/(1.0E-04*height))
                                        PlasmodesmFlowDensity.append(-temp/(1.0E-04*height))
                                        Fcc=temp*1.0E-04*eattr['length']*sign(j-ind_o_cell)
                                        if ind_o_cell<j:
                                            Fcc_list.append((ind_o_cell,j,Fcc)) #(cm^3/d) Water flow rate positive from high index to low index cell
                                        else:
                                            Fcc_list.append((j,ind_o_cell,Fcc))
                                        #if boundary.c_flag:
                                        if general.sym_contagion==2: #Convection across plasmodesmata
                                            if general.apo_contagion==2: #Apo & Sym Contagion
                                                if Fcc>0: #Flow from high index to low index cell
                                                    if ind_o_cell<j: #From j to ind_o_cell
                                                        if j-network.n_wall_junction not in hormones.sym_zombie0:
                                                            matrix_C[j][j] -= Fcc
                                                        if ind_o_cell-network.n_wall_junction not in hormones.sym_zombie0:
                                                            matrix_C[ind_o_cell][j] += Fcc
                                                    else: #From ind_o_cell to j
                                                        if ind_o_cell-network.n_wall_junction not in hormones.sym_zombie0:
                                                            matrix_C[ind_o_cell][ind_o_cell] -= Fcc
                                                        if j-network.n_wall_junction not in hormones.sym_zombie0:
                                                            matrix_C[j][ind_o_cell] += Fcc
                                                else: #Flow from low index to high index cell
                                                    if ind_o_cell<j: #From ind_o_cell to j
                                                        if ind_o_cell-network.n_wall_junction not in hormones.sym_zombie0:
                                                            matrix_C[ind_o_cell][ind_o_cell] += Fcc
                                                        if j-network.n_wall_junction not in hormones.sym_zombie0:
                                                            matrix_C[j][ind_o_cell] -= Fcc
                                                    else: #From j to ind_o_cell
                                                        if j-network.n_wall_junction not in hormones.sym_zombie0:
                                                            matrix_C[j][j] += Fcc
                                                        if ind_o_cell-network.n_wall_junction not in hormones.sym_zombie0:
                                                            matrix_C[ind_o_cell][j] -= Fcc
                                            else: #Only Sym contagion
                                                if Fcc>0: #Flow from high index to low index cell
                                                    if ind_o_cell<j: #From j to ind_o_cell
                                                        if j-network.n_wall_junction not in hormones.sym_zombie0:
                                                            matrix_SymC[j-network.n_wall_junction][j-network.n_wall_junction] -= Fcc
                                                        if ind_o_cell-network.n_wall_junction not in hormones.sym_zombie0:
                                                            matrix_SymC[ind_o_cell-network.n_wall_junction][j-network.n_wall_junction] += Fcc
                                                    else: #From ind_o_cell to j
                                                        if ind_o_cell-network.n_wall_junction not in hormones.sym_zombie0:
                                                            matrix_SymC[ind_o_cell-network.n_wall_junction][ind_o_cell-network.n_wall_junction] -= Fcc
                                                        if j-network.n_wall_junction not in hormones.sym_zombie0:
                                                            matrix_SymC[j-network.n_wall_junction][ind_o_cell-network.n_wall_junction] += Fcc
                                                else: #Flow from low index to high index cell
                                                    if ind_o_cell<j: #From ind_o_cell to j
                                                        if ind_o_cell-network.n_wall_junction not in hormones.sym_zombie0:
                                                            matrix_SymC[ind_o_cell-network.n_wall_junction][ind_o_cell-network.n_wall_junction] += Fcc
                                                        if j-network.n_wall_junction not in hormones.sym_zombie0:
                                                            matrix_SymC[j-network.n_wall_junction][ind_o_cell-network.n_wall_junction] -= Fcc
                                                    else: #From j to ind_o_cell
                                                        if j-network.n_wall_junction not in hormones.sym_zombie0:
                                                            matrix_SymC[j-network.n_wall_junction][j-network.n_wall_junction] += Fcc
                                                        if ind_o_cell-network.n_wall_junction not in hormones.sym_zombie0:
                                                            matrix_SymC[ind_o_cell-network.n_wall_junction][j-network.n_wall_junction] -= Fcc
                                        
                                        if general.sym_contagion==1:
                                            itemp=0
                                            while not network.cell_connections[ind_o_cell-network.n_wall_junction][itemp] == j-network.n_wall_junction:
                                                itemp+=1
                                            Cell_connec_flow[ind_o_cell-network.n_wall_junction][itemp]=sign(temp)
                                            itemp=0
                                            while not network.cell_connections[j-network.n_wall_junction][itemp] == ind_o_cell-network.n_wall_junction:
                                                itemp+=1
                                            Cell_connec_flow[j-network.n_wall_junction][itemp]=-sign(temp)
                            elif general.paraview==0 and general.par_track==0:
                                if path == "membrane": #Membrane connection
                                    K=Kmb[jmb][0]
                                    #Flow densities calculation
                                    #Macroscopic distributed parameter for transmembrane flow
                                    #Discretization based on cell layers and apoplasmic barriers
                                    rank = int(network.cell_ranks[j-network.n_wall_junction])
                                    row = int(network.rank_to_row[rank][0])
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
                                    Flow = K * (psi - psin + s_membranes[jmb]*(Os_walls[i] - Os_cells[j-network.n_wall_junction]))
                                    jmb+=1
                                    if ((j-network.n_wall_junction not in geometry.intercellular_ids) and (j not in network.xylem_cells)) or Barrier==0:
                                        if Flow > 0 :
                                            UptakeLayer_plus[row][iMaturity][count] += Flow #grouping membrane flow rates in cell layers
                                        else:
                                            UptakeLayer_minus[row][iMaturity][count] += Flow
                                    
                                    if K>1.0e-12: #Not an impermeable wall
                                        PsiWallLayer[row][iMaturity][count] += psi
                                        NWallLayer[row][iMaturity][count] += 1
                
                #if boundary.c_flag: #Calculates stationary solute concentration
                if general.apo_contagion==2 or general.sym_contagion==2: #Sym & Apo contagion
                    if general.apo_contagion==2 and general.sym_contagion==2: #Sym & Apo contagion
                        #Solving apoplastic & symplastic concentrations
                        soln_C = np.linalg.solve(matrix_C,rhs_C) #Solving the equation to get apoplastic relative concentrations
                    elif general.apo_contagion==2:
                        #Solving apoplastic concentrations
                        soln_ApoC = np.linalg.solve(matrix_ApoC,rhs_ApoC) #Solving the equation to get apoplastic & symplastic relative concentrations
                    else: # Only Symplastic contagion
                        #Solving apoplastic concentrations
                        soln_SymC = np.linalg.solve(matrix_SymC,rhs_SymC) #Solving the equation to get symplastic relative concentrations
                    
                #Resets matrix_C and rhs_C to geometrical factor values
                if general.apo_contagion==2:
                    if general.sym_contagion==2: # Apo & Sym contagion
                        for i,j,Fjw in Fjw_list:
                            if Fjw>0: #Flow from junction to wall
                                if i not in network.apo_wall_zombies0:
                                    matrix_C[i][j] -= Fjw #Removing convective term
                                if j not in Apo_j_Zombies0:
                                    matrix_C[j][j] += Fjw #Removing convective term
                            else: #Flow from wall to junction
                                if i not in network.apo_wall_zombies0:
                                    matrix_C[i][i] -= Fjw #Removing convective term
                                if j not in Apo_j_Zombies0:
                                    matrix_C[j][i] += Fjw #Removing convective term
                    else: #Only Apo contagion
                        for i,j,Fjw in Fjw_list:
                            if Fjw>0: #Flow from junction to wall
                                if i not in network.apo_wall_zombies0:
                                    matrix_ApoC[i][j] -= Fjw #Removing convective term
                                if j not in Apo_j_Zombies0:
                                    matrix_ApoC[j][j] += Fjw #Removing convective term
                            else: #Flow from wall to junction
                                if i not in network.apo_wall_zombies0:
                                    matrix_ApoC[i][i] -= Fjw #Removing convective term
                                if j not in Apo_j_Zombies0:
                                    matrix_ApoC[j][i] += Fjw #Removing convective term
                
                if general.sym_contagion==2: #Convection across plasmodesmata
                    if general.apo_contagion==2: #Apo & Sym Contagion
                        for i,j,Fcc in Fcc_list:
                            if Fcc>0: #Flow from j to i
                                if j-network.n_wall_junction not in hormones.sym_zombie0:
                                    matrix_C[j][j] += Fcc #Removing convective term
                                if i-network.n_wall_junction not in hormones.sym_zombie0:
                                    matrix_C[i][j] -= Fcc #Removing convective term
                            else: #Flow from i to j
                                if i-network.n_wall_junction not in hormones.sym_zombie0:
                                    matrix_C[i][i] -= Fcc #Removing convective term
                                if j-network.n_wall_junction not in hormones.sym_zombie0:
                                    matrix_C[j][i] += Fcc #Removing convective term
                    else: #Only Sym contagion
                        for i,j,Fcc in Fcc_list:
                            if Fcc>0: #Flow from j to i
                                if j-network.n_wall_junction not in hormones.sym_zombie0:
                                    matrix_SymC[j-network.n_wall_junction][j-network.n_wall_junction] += Fcc #Removing convective term
                                if ind_o_cell-network.n_wall_junction not in hormones.sym_zombie0:
                                    matrix_SymC[i-network.n_wall_junction][j-network.n_wall_junction] -= Fcc #Removing convective term
                            else: #Flow from i to j
                                if i-network.n_wall_junction not in hormones.sym_zombie0:
                                    matrix_SymC[i-network.n_wall_junction][i-network.n_wall_junction] -= Fcc #Removing convective term
                                if j-network.n_wall_junction not in hormones.sym_zombie0:
                                    matrix_SymC[j-network.n_wall_junction][i-network.n_wall_junction] += Fcc #Removing convective term
                
                if general.apo_contagion==2 and general.sym_contagion==2:
                    for i,j,Fcw,s in Fcw_list:
                        Fcw=-Fcw #Attention, -Fcw was saved
                        if Fcw>0: #Flow from wall to protoplast
                            if i not in network.apo_wall_zombies0:
                                if hormones.d2o1==1:#Solute that moves across membranes like water 
                                    matrix_C[i][i] += Fcw #Removing convective term
                                else: #Solute that moves across membranes independently of water (the membrane is possibly not one) 
                                    matrix_C[i][i] += Fcw*(1-s) #Removing convective term
                            if j-network.n_wall_junction not in hormones.sym_zombie0:
                                if hormones.d2o1==1:#Solute that moves across membranes like water 
                                    matrix_C[j][i] -= Fcw #Removing convective term
                                else: #Solute that moves across membranes independently of water (the membrane is possibly not one) 
                                    matrix_C[j][i] -= Fcw*(1-s) #Removing convective term
                        else: #Flow from protoplast to wall
                            if j-network.n_wall_junction not in hormones.sym_zombie0:
                                if hormones.d2o1==1:#Solute that moves across membranes like water 
                                    matrix_C[j][j] -= Fcw #Removing convective term
                                else: #Solute that moves across membranes independently of water (the membrane is possibly not one) 
                                    matrix_C[j][j] -= Fcw*(1-s) #Removing convective term
                            if i not in network.apo_wall_zombies0:
                                if hormones.d2o1==1:#Solute that moves across membranes like water 
                                    matrix_C[i][j] += Fcw #Removing convective term
                                else: #Solute that moves across membranes independently of water (the membrane is possibly not one) 
                                    matrix_C[i][j] += Fcw*(1-s) #Removing convective term
                
                if general.apo_contagion==2:
                    if general.sym_contagion==2: # Apo & Sym contagion
                        i=0
                        for ind in network.border_walls:
                            if ind not in network.apo_wall_zombies0:
                                Q=Q_soil[i] #(cm^3/d) Positive for water flowing into the root, rhs_s is minus the conductance at the soil root interface
                                if Q<0.0:
                                    matrix_C[ind][ind]-=Q #Removing convective term
                            i+=1
                        for ind in network.border_junction:
                            if ind not in Apo_j_Zombies0:
                                Q=Q_soil[i] #(cm^3/d) Positive for water flowing into the root, rhs_s is minus the conductance at the soil root interface
                                if Q<0.0:
                                    matrix_C[ind][ind]-=Q #Removing convective term
                            i+=1
                    else:
                        i=0
                        for ind in network.border_walls:
                            if ind not in network.apo_wall_zombies0:
                                Q=Q_soil[i] #(cm^3/d) Positive for water flowing into the root, rhs_s is minus the conductance at the soil root interface
                                if Q<0.0:
                                    matrix_ApoC[ind][ind]-=Q #Removing convective term
                            i+=1
                        for ind in network.border_junction:
                            if ind not in Apo_j_Zombies0:
                                Q=Q_soil[i] #(cm^3/d) Positive for water flowing into the root, rhs_s is minus the conductance at the soil root interface
                                if Q<0.0:
                                    matrix_ApoC[ind][ind]-=Q #Removing convective term
                            i+=1
                

                ####################################
                ## Creates .vtk file for general.paraview ##
                ####################################
                

                if general.sym_contagion==1:
                    iZombie=0
                    while not iZombie == size(hormones.sym_zombie0):
                        itemp=0
                        for cid in network.cell_connections[int(hormones.sym_zombie0[iZombie])][0:int(network.n_cell_connections[int(hormones.sym_zombie0[iZombie])])]:
                            if Cell_connec_flow[int(hormones.sym_zombie0[iZombie])][itemp] == -1 and (cid not in hormones.sym_zombie0): #Infection
                                if cid in hormones.sym_immune:
                                    print(cid,': "You shall not pass!"')
                                else:
                                    hormones.sym_zombie0.append(cid)
                                    print(cid,': "Aaargh!"      Zombie count:', size(hormones.sym_zombie0)+1)
                            itemp+=1
                        iZombie+=1
                    print('End of the propagation. Survivor count:', network.n_cells-size(hormones.sym_zombie0)-1)
                    for cid in hormones.sym_target:
                        if cid in hormones.sym_zombie0:
                            print('Target '+ str(cid) +' down. XXX')
                        else:
                            print('Target '+ str(cid) +' missed!')
                    if hormones.sym_target[0] in hormones.sym_zombie0:
                        if hormones.sym_target[1] in hormones.sym_zombie0:
                            Hydropatterning[iMaturity][count]=0 #Both targets reached
                        else:
                            Hydropatterning[iMaturity][count]=1 #Target1 reached only
                    elif hormones.sym_target[1] in hormones.sym_zombie0:
                        Hydropatterning[iMaturity][count]=2 #Target2 reached only
                    else:
                        Hydropatterning[iMaturity][count]=-1 #Not target reached
                    
                    
                    text_file = open(newpath+"Sym_Contagion_bottomb"+str(Barrier)+"_"+str(iMaturity)+"s"+str(count)+".pvtk", "w")
                    with open(newpath+"Sym_Contagion_bottomb"+str(Barrier)+"_"+str(iMaturity)+"s"+str(count)+".pvtk", "a") as myfile:
                        myfile.write("# vtk DataFile Version 4.0 \n")
                        myfile.write("Contaminated symplastic space geometry \n")
                        myfile.write("ASCII \n")
                        myfile.write(" \n")
                        myfile.write("DATASET UNSTRUCTURED_GRID \n")
                        myfile.write("POINTS "+str(len(thick_wall))+" float \n")
                        for ThickWallNode in thick_wall:
                            myfile.write(str(ThickWallNode[3]) + " " + str(ThickWallNode[4]) + " " + str(height/200) + " \n")
                        myfile.write(" \n")
                        myfile.write("CELLS " + str(len(hormones.sym_zombie0)) + " " + str(int(len(hormones.sym_zombie0)+sum(n_cell_to_thick_wall[hormones.sym_zombie0]))) + " \n") #The number of cells corresponds to the number of intercellular spaces
                        Sym_Contagion_order=zeros((network.n_cells,1))
                        temp=0
                        for cid in hormones.sym_zombie0:
                            n=int(n_cell_to_thick_wall[cid]) #Total number of thick wall nodes around the protoplast
                            Polygon=cell_to_thick_wall[cid][:n]
                            ranking=list()
                            ranking.append(int(Polygon[0]))
                            ranking.append(thick_wall[int(ranking[0])][5])
                            ranking.append(thick_wall[int(ranking[0])][6])
                            for id1 in range(1,n):
                                wid1=thick_wall[int(ranking[id1])][5]
                                wid2=thick_wall[int(ranking[id1])][6]
                                if wid1 not in ranking:
                                    ranking.append(wid1)
                                if wid2 not in ranking:
                                    ranking.append(wid2)
                            string=str(n)
                            for id1 in ranking:
                                string=string+" "+str(int(id1))
                            myfile.write(string + " \n")
                            Sym_Contagion_order[cid]=temp
                            temp+=1
                        myfile.write(" \n")
                        myfile.write("CELL_TYPES " + str(len(hormones.sym_zombie0)) + " \n")
                        for i in range(len(hormones.sym_zombie0)):
                            myfile.write("6 \n") #Triangle-strip cell type
                        myfile.write(" \n")
                        myfile.write("POINT_DATA " + str(len(thick_wall)) + " \n")
                        myfile.write("SCALARS Sym_Contagion_order_(#) float \n")
                        myfile.write("LOOKUP_TABLE default \n")
                        for ThickWallNode in thick_wall:
                            cell_id=ThickWallNode[2]-network.n_wall_junction
                            myfile.write(str(int(Sym_Contagion_order[int(cell_id)])) + " \n") #Flow rate from wall (non junction) to cell    min(sath1,max(satl1,  ))
                    myfile.close()
                    text_file.close()
                    
                elif general.sym_contagion==2:
                    text_file = open(newpath+"Sym_Contagion_bottomb"+str(Barrier)+"_"+str(iMaturity)+"s"+str(count)+".pvtk", "w")
                    with open(newpath+"Sym_Contagion_bottomb"+str(Barrier)+"_"+str(iMaturity)+"s"+str(count)+".pvtk", "a") as myfile:
                        myfile.write("# vtk DataFile Version 4.0 \n")
                        myfile.write("Symplastic hormone concentration \n")
                        myfile.write("ASCII \n")
                        myfile.write(" \n")
                        myfile.write("DATASET UNSTRUCTURED_GRID \n")
                        myfile.write("POINTS "+str(len(thick_wall))+" float \n")
                        for ThickWallNode in thick_wall:
                            myfile.write(str(ThickWallNode[3]) + " " + str(ThickWallNode[4]) + " " + str(height/200) + " \n")
                        myfile.write(" \n")
                        myfile.write("CELLS " + str(network.n_cells) + " " + str(int(network.n_cells+sum(n_cell_to_thick_wall))) + " \n") #The number of cells corresponds to the number of intercellular spaces
                        for cid in range(network.n_cells):
                            n=int(n_cell_to_thick_wall[cid]) #Total number of thick wall nodes around the protoplast
                            Polygon=cell_to_thick_wall[cid][:n]
                            ranking=list()
                            ranking.append(int(Polygon[0]))
                            ranking.append(thick_wall[int(ranking[0])][5])
                            ranking.append(thick_wall[int(ranking[0])][6])
                            for id1 in range(1,n):
                                wid1=thick_wall[int(ranking[id1])][5]
                                wid2=thick_wall[int(ranking[id1])][6]
                                if wid1 not in ranking:
                                    ranking.append(wid1)
                                if wid2 not in ranking:
                                    ranking.append(wid2)
                            string=str(n)
                            for id1 in ranking:
                                string=string+" "+str(int(id1))
                            myfile.write(string + " \n")
                        myfile.write(" \n")
                        myfile.write("CELL_TYPES " + str(network.n_cells) + " \n")
                        for i in range(network.n_cells):
                            myfile.write("6 \n") #Triangle-strip cell type
                        myfile.write(" \n")
                        myfile.write("POINT_DATA " + str(len(thick_wall)) + " \n")
                        myfile.write("SCALARS Hormone_Symplastic_Relative_Concentration_(-) float \n")
                        myfile.write("LOOKUP_TABLE default \n")
                        if general.apo_contagion==2:
                            for ThickWallNode in thick_wall:
                                cell_id=ThickWallNode[2]-network.n_wall_junction
                                myfile.write(str(float(soln_C[int(cell_id+network.n_wall_junction)])) + " \n")
                        else:
                            for ThickWallNode in thick_wall:
                                cell_id=ThickWallNode[2]-network.n_wall_junction
                                myfile.write(str(float(soln_SymC[int(cell_id)])) + " \n") #
                    myfile.close()
                    text_file.close()
                
                if general.apo_contagion==1:
                    Apo_w_Zombies=network.apo_wall_zombies0
                    iZombie=0
                    while not iZombie == size(Apo_w_Zombies):
                        id1=Apo_w_Zombies[iZombie]
                        for id2 in Apo_connec_flow[id1][0:nApo_connec_flow[id1]]:
                            if id2 not in Apo_w_Zombies: #Infection
                                if id2 in network.apo_wall_immune:
                                    print(id2,': "You shall not pass!"')
                                else:
                                    Apo_w_Zombies.append(id2)
                                    print(id2,': "Aaargh!"      Zombie count:', size(Apo_w_Zombies))
                        iZombie+=1
                    print('End of the propagation. Survivor count:', network.n_wall_junction-size(Apo_w_Zombies))
                    temp=0
                    for wall_id in network.apo_wall_target:
                        if wall_id in Apo_w_Zombies:
                            temp+=1
                            print('Target '+ str(wall_id) +' down. XXX')
                        else:
                            print('Target '+ str(wall_id) +' missed!')
                    Hydrotropism[iMaturity][count]=float(temp)/size(network.apo_wall_target) #0: No apoplastic target reached; 1: All apoplastic targets reached
                    
                    
                    text_file = open(newpath+"Apo_Contagion_bottomb"+str(Barrier)+"_"+str(iMaturity)+"s"+str(count)+".pvtk", "w")
                    with open(newpath+"Apo_Contagion_bottomb"+str(Barrier)+"_"+str(iMaturity)+"s"+str(count)+".pvtk", "a") as myfile:
                        myfile.write("# vtk DataFile Version 4.0 \n")
                        myfile.write("Contaminated Apoplastic space geometry \n")
                        myfile.write("ASCII \n")
                        myfile.write(" \n")
                        myfile.write("DATASET UNSTRUCTURED_GRID \n")
                        myfile.write("POINTS "+str(len(thick_wall_x))+" float \n")
                        for ThickWallNodeX in thick_wall_x:
                            myfile.write(str(ThickWallNodeX[1]) + " " + str(ThickWallNodeX[2]) + " 0.0 \n")
                        myfile.write(" \n")
                        myfile.write("CELLS " + str(int(network.n_wall_junction+network.n_walls-len(list_ghostwalls)*2-len(list_ghostjunctions))) + " " + str(int(2*network.n_walls*5-len(list_ghostwalls)*10+sum(n_wall_to_wall_x[network.n_walls:])+network.n_wall_junction-network.n_walls+2*len(wall_to_wall_x[network.n_walls:])-nGhostJunction2Wall-len(list_ghostjunctions))) + " \n") #The number of cells corresponds to the number of lines in thick_wall (if no ghost wall & junction)
                        i=0
                        for PolygonX in thick_wall_polygon_x:
                            if floor(i/2) not in list_ghostwalls:
                                myfile.write("4 " + str(int(PolygonX[0])) + " " + str(int(PolygonX[1])) + " " + str(int(PolygonX[2])) + " " + str(int(PolygonX[3])) + " \n")
                            i+=1
                        j=network.n_walls
                        for PolygonX in wall_to_wall_x[network.n_walls:]: #"junction" polygons
                            #Would need to order them based on x or y position to make sure display fully covers the surface (but here we try a simpler not so good solution instead)
                            if j not in list_ghostjunctions:
                                string=str(int(n_wall_to_wall_x[j]+2)) #Added +2 so that the first and second nodes could be added again at the end (trying to fill the polygon better)
                                for id1 in range(int(n_wall_to_wall_x[j])):
                                    string=string+" "+str(int(PolygonX[id1]))
                                string=string+" "+str(int(PolygonX[0]))+" "+str(int(PolygonX[1])) #Adding the 1st and 2nd nodes again to the end
                                myfile.write(string + " \n")
                            j+=1
                        myfile.write(" \n")
                        myfile.write("CELL_TYPES " + str(network.n_wall_junction+network.n_walls-len(list_ghostwalls)*2-len(list_ghostjunctions)) + " \n")
                        i=0
                        for PolygonX in thick_wall_polygon_x:
                            if floor(i/2) not in list_ghostwalls:
                                myfile.write("7 \n") #Polygon cell type (wall)
                            i+=1
                        j=network.n_walls
                        for PolygonX in wall_to_wall_x[network.n_walls:]:
                            if j not in list_ghostjunctions:
                                myfile.write("6 \n") #Triangle-strip cell type (wall junction)
                            j+=1
                        myfile.write(" \n")
                        myfile.write("POINT_DATA " + str(len(thick_wall_x)) + " \n")
                        myfile.write("SCALARS Apo_Contagion_order_(#) float \n")
                        myfile.write("LOOKUP_TABLE default \n")
                        Apo_Contagion_order=zeros((network.n_wall_junction,1))+int(len(Apo_w_Zombies)*1.6)
                        temp=0
                        for wall_id in Apo_w_Zombies:
                            Apo_Contagion_order[wall_id]=temp
                            temp+=1
                        NewApo_Contagion_order=zeros((len(thick_wall_x),1))
                        j=0
                        for PolygonX in wall_to_wall_x:
                            for id1 in range(int(n_wall_to_wall_x[j])):
                                NewApo_Contagion_order[int(PolygonX[id1])]=Apo_Contagion_order[j]
                            j+=1
                        for i in range(len(thick_wall_x)):
                            myfile.write(str(float(NewApo_Contagion_order[i])) + " \n")
                    myfile.close()
                    text_file.close()
                    
                elif general.apo_contagion==2:
                    text_file = open(newpath+"Apo_Contagion_bottomb"+str(Barrier)+"_"+str(iMaturity)+"s"+str(count)+".pvtk", "w")
                    with open(newpath+"Apo_Contagion_bottomb"+str(Barrier)+"_"+str(iMaturity)+"s"+str(count)+".pvtk", "a") as myfile:
                        myfile.write("# vtk DataFile Version 4.0 \n")
                        myfile.write("Apoplastic hormone concentration \n")
                        myfile.write("ASCII \n")
                        myfile.write(" \n")
                        myfile.write("DATASET UNSTRUCTURED_GRID \n")
                        myfile.write("POINTS "+str(len(thick_wall_x))+" float \n")
                        for ThickWallNodeX in thick_wall_x:
                            myfile.write(str(ThickWallNodeX[1]) + " " + str(ThickWallNodeX[2]) + " 0.0 \n")
                        myfile.write(" \n")
                        myfile.write("CELLS " + str(int(network.n_wall_junction+network.n_walls-len(list_ghostwalls)*2-len(list_ghostjunctions))) + " " + str(int(2*network.n_walls*5-len(list_ghostwalls)*10+sum(n_wall_to_wall_x[network.n_walls:])+network.n_wall_junction-network.n_walls+2*len(wall_to_wall_x[network.n_walls:])-nGhostJunction2Wall-len(list_ghostjunctions))) + " \n") #The number of cells corresponds to the number of lines in thick_wall (if no ghost wall & junction)
                        i=0
                        for PolygonX in thick_wall_polygon_x:
                            if floor(i/2) not in list_ghostwalls:
                                myfile.write("4 " + str(int(PolygonX[0])) + " " + str(int(PolygonX[1])) + " " + str(int(PolygonX[2])) + " " + str(int(PolygonX[3])) + " \n")
                            i+=1
                        j=network.n_walls
                        for PolygonX in wall_to_wall_x[network.n_walls:]: #"junction" polygons
                            #Would need to order them based on x or y position to make sure display fully covers the surface (but here we try a simpler not so good solution instead)
                            if j not in list_ghostjunctions:
                                string=str(int(n_wall_to_wall_x[j]+2)) #Added +2 so that the first and second nodes could be added again at the end (trying to fill the polygon better)
                                for id1 in range(int(n_wall_to_wall_x[j])):
                                    string=string+" "+str(int(PolygonX[id1]))
                                string=string+" "+str(int(PolygonX[0]))+" "+str(int(PolygonX[1])) #Adding the 1st and 2nd nodes again to the end
                                myfile.write(string + " \n")
                            j+=1
                        myfile.write(" \n")
                        myfile.write("CELL_TYPES " + str(network.n_wall_junction+network.n_walls-len(list_ghostwalls)*2-len(list_ghostjunctions)) + " \n")
                        i=0
                        for PolygonX in thick_wall_polygon_x:
                            if floor(i/2) not in list_ghostwalls:
                                myfile.write("7 \n") #Polygon cell type (wall)
                            i+=1
                        j=network.n_walls
                        for PolygonX in wall_to_wall_x[network.n_walls:]:
                            if j not in list_ghostjunctions:
                                myfile.write("6 \n") #Triangle-strip cell type (wall junction)
                            j+=1
                        myfile.write(" \n")
                        myfile.write("POINT_DATA " + str(len(thick_wall_x)) + " \n")
                        myfile.write("SCALARS Hormone_Symplastic_Relative_Concentration_(-) float \n")
                        myfile.write("LOOKUP_TABLE default \n")
                        if general.sym_contagion==2:
                            Newsoln_C=zeros((len(thick_wall_x),1))
                            j=0
                            for PolygonX in wall_to_wall_x:
                                for id1 in range(int(n_wall_to_wall_x[j])):
                                    Newsoln_C[int(PolygonX[id1])]=soln_C[j]
                                j+=1
                            for i in range(len(thick_wall_x)):
                                myfile.write(str(float(Newsoln_C[i])) + " \n")
                        else:
                            Newsoln_ApoC=zeros((len(thick_wall_x),1))
                            j=0
                            for PolygonX in wall_to_wall_x:
                                for id1 in range(int(n_wall_to_wall_x[j])):
                                    Newsoln_ApoC[int(PolygonX[id1])]=soln_ApoC[j]
                                j+=1
                            for i in range(len(thick_wall_x)):
                                myfile.write(str(float(Newsoln_ApoC[i])) + " \n")
                    myfile.close()
                    text_file.close()
                
                
                if general.paraview==1:
                    if general.paraview_wp==1: #2D visualization of walls pressure potentials
                        text_file = open(newpath+"Walls2Db"+str(Barrier)+"_"+str(iMaturity)+"s"+str(count)+".pvtk", "w")
                        #sath0=max(soln[0:network.n_wall_junction-1])
                        #satl0=min(soln[0:network.n_wall_junction-1])
                        with open(newpath+"Walls2Db"+str(Barrier)+"_"+str(iMaturity)+"s"+str(count)+".pvtk", "a") as myfile:
                            myfile.write("# vtk DataFile Version 4.0 \n")     #("Purchase Amount: %s" % TotalAmount)
                            myfile.write("Wall geometry 2D \n")
                            myfile.write("ASCII \n")
                            myfile.write(" \n")
                            myfile.write("DATASET UNSTRUCTURED_GRID \n")
                            myfile.write("POINTS "+str(len(network.graph.nodes))+" float \n")
                            for node in network.graph:
                                myfile.write(str(float(position[node][0])) + " " + str(float(position[node][1])) + " " + str(0.0) + " \n")
                            myfile.write(" \n")
                            myfile.write("CELLS " + str(network.n_walls*2-len(list_ghostwalls)*2) + " " + str(network.n_walls*6-len(list_ghostwalls)*6) + " \n") #len(network.graph.nodes)
                            for node, edges in network.graph.adjacency():
                                i=indice[node]
                                if i not in list_ghostwalls:
                                    for neighboor, eattr in edges.items(): #Loop on connections (edges)
                                        j=indice[neighboor]
                                        if j>i and eattr['path']=='wall':
                                            myfile.write(str(2) + " " + str(i) + " " + str(j) + " \n")
                            myfile.write(" \n")
                            myfile.write("CELL_TYPES " + str(network.n_walls*2-len(list_ghostwalls)*2) + " \n") #The number of nodes corresponds to the number of wall to wall connections.... to be checked, might not be generality
                            for node, edges in network.graph.adjacency():
                                i=indice[node]
                                if i not in list_ghostwalls:
                                    for neighboor, eattr in edges.items(): #Loop on connections (edges)
                                        j=indice[neighboor]
                                        if j>i and eattr['path']=='wall':
                                            myfile.write(str(3) + " \n") #Line cell type
                            myfile.write(" \n")
                            myfile.write("POINT_DATA " + str(len(network.graph.nodes)) + " \n")
                            myfile.write("SCALARS Wall_pressure float \n")
                            myfile.write("LOOKUP_TABLE default \n")
                            for node in network.graph:
                                myfile.write(str(float(soln[node])) + " \n") #Line cell type      min(sath0,max(satl0,   ))
                        myfile.close()
                        text_file.close()
                    
                    if general.paraview_wp==1 and general.paraview_cp: #2D visualization of walls & cells osmotic potentials
                        text_file = open(newpath+"WallsOsAndCellsOs2Db"+str(Barrier)+"_"+str(iMaturity)+"s"+str(count)+".pvtk", "w")
                        with open(newpath+"WallsOsAndCellsOs2Db"+str(Barrier)+"_"+str(iMaturity)+"s"+str(count)+".pvtk", "a") as myfile:
                            myfile.write("# vtk DataFile Version 4.0 \n")     #("Purchase Amount: %s" % TotalAmount)
                            myfile.write("Wall geometry 2D \n")
                            myfile.write("ASCII \n")
                            myfile.write(" \n")
                            myfile.write("DATASET UNSTRUCTURED_GRID \n")
                            myfile.write("POINTS "+str(len(network.graph.nodes))+" float \n")
                            for node in network.graph:
                                myfile.write(str(float(position[node][0])) + " " + str(float(position[node][1])) + " " + str(0.0) + " \n")
                            myfile.write(" \n")                                     
                            myfile.write("CELLS " + str(network.n_walls*2-len(list_ghostwalls)*2+network.n_cells) + " " + str(network.n_walls*6-len(list_ghostwalls)*6+network.n_cells*2) + " \n") #
                            for node, edges in network.graph.adjacency():
                                i=indice[node]
                                if i not in list_ghostwalls:
                                    for neighboor, eattr in edges.items(): #Loop on connections (edges)
                                        j=indice[neighboor]
                                        if j>i and eattr['path']=='wall':
                                            myfile.write(str(2) + " " + str(i) + " " + str(j) + " \n")
                                if i>=network.n_wall_junction: #Cell node
                                    myfile.write("1 " + str(i) + " \n")
                            myfile.write(" \n")
                            myfile.write("CELL_TYPES " + str(network.n_walls*2-len(list_ghostwalls)*2+network.n_cells) + " \n") #
                            for node, edges in network.graph.adjacency():
                                i=indice[node]
                                if i not in list_ghostwalls:
                                    for neighboor, eattr in edges.items(): #Loop on connections (edges)
                                        j=indice[neighboor]
                                        if j>i and eattr['path']=='wall':
                                            myfile.write(str(3) + " \n") #Line cell type
                                if i>=network.n_wall_junction: #Cell node
                                    myfile.write("1 \n")
                            myfile.write(" \n")
                            myfile.write("POINT_DATA " + str(len(network.graph.nodes)) + " \n")
                            myfile.write("SCALARS Wall_and_Cell_osmotic_pot float \n")
                            myfile.write("LOOKUP_TABLE default \n")
                            for node, edges in network.graph.adjacency():
                                i=indice[node] #Node ID number
                                if i<network.n_walls: #Wall node
                                    myfile.write(str(float(Os_walls[i])) + " \n")
                                elif i<network.n_wall_junction: #Junction node
                                    myfile.write(str(float(0.0)) + " \n")
                                else: #Cell node
                                    myfile.write(str(float(Os_cells[i-network.n_wall_junction])) + " \n")
                        myfile.close()
                        text_file.close()
                        
    
                    
                    if general.paraview_wp==1 and general.paraview_cp==1: #2D visualization of walls & cells water potentials
                        text_file = open(newpath+"WallsAndCells2Db"+str(Barrier)+"_"+str(iMaturity)+"s"+str(count)+".pvtk", "w")
                        with open(newpath+"WallsAndCells2Db"+str(Barrier)+"_"+str(iMaturity)+"s"+str(count)+".pvtk", "a") as myfile:
                            myfile.write("# vtk DataFile Version 4.0 \n")     #("Purchase Amount: %s" % TotalAmount)
                            myfile.write("Water potential distribution in cells and walls 2D \n")
                            myfile.write("ASCII \n")
                            myfile.write(" \n")
                            myfile.write("DATASET UNSTRUCTURED_GRID \n")
                            myfile.write("POINTS "+str(len(network.graph.nodes))+" float \n")
                            for node in network.graph:
                                myfile.write(str(float(position[node][0])) + " " + str(float(position[node][1])) + " " + str(0.0) + " \n")
                            myfile.write(" \n")
                            myfile.write("CELLS " + str(network.n_walls*2-len(list_ghostwalls)*2+network.n_cells) + " " + str(network.n_walls*6-len(list_ghostwalls)*6+network.n_cells*2) + " \n") #
                            for node, edges in network.graph.adjacency():
                                i=indice[node]
                                if i not in list_ghostwalls:
                                    for neighboor, eattr in edges.items(): #Loop on connections (edges)
                                        j=indice[neighboor]
                                        if j>i and eattr['path']=='wall':
                                            myfile.write(str(2) + " " + str(i) + " " + str(j) + " \n")
                                if i>=network.n_wall_junction: #Cell node
                                    myfile.write("1 " + str(i) + " \n")
                            myfile.write(" \n")
                            myfile.write("CELL_TYPES " + str(network.n_walls*2-len(list_ghostwalls)*2+network.n_cells) + " \n") #
                            for node, edges in network.graph.adjacency():
                                i=indice[node]
                                if i not in list_ghostwalls:
                                    for neighboor, eattr in edges.items(): #Loop on connections (edges)
                                        j=indice[neighboor]
                                        if j>i and eattr['path']=='wall':
                                            myfile.write(str(3) + " \n") #Line cell type
                                if i>=network.n_wall_junction: #Cell node
                                    myfile.write("1 \n")
                            myfile.write(" \n")
                            myfile.write("POINT_DATA " + str(len(network.graph.nodes)) + " \n")
                            myfile.write("SCALARS pressure float \n")
                            myfile.write("LOOKUP_TABLE default \n")
                            for node in network.graph:
                                myfile.write(str(float(soln[node])) + " \n") #Line cell type
                        myfile.close()
                        text_file.close()
                    
                    if general.paraview_cp==1: #2D visualization of cells water potentials
                        text_file = open(newpath+"Cells2Db"+str(Barrier)+"_"+str(iMaturity)+"s"+str(count)+".pvtk", "w")
                        with open(newpath+"Cells2Db"+str(Barrier)+"_"+str(iMaturity)+"s"+str(count)+".pvtk", "a") as myfile:
                            myfile.write("# vtk DataFile Version 4.0 \n")     #("Purchase Amount: %s" % TotalAmount)
                            myfile.write("Pressure potential distribution in cells 2D \n")
                            myfile.write("ASCII \n")
                            myfile.write(" \n")
                            myfile.write("DATASET UNSTRUCTURED_GRID \n")
                            myfile.write("POINTS "+str(len(network.graph.nodes))+" float \n")
                            for node in network.graph:
                                myfile.write(str(float(position[node][0])) + " " + str(float(position[node][1])) + " " + str(0.0) + " \n")
                            myfile.write(" \n")
                            myfile.write("CELLS " + str(network.n_cells) + " " + str(network.n_cells*2) + " \n") #
                            for node, edges in network.graph.adjacency():
                                i=indice[node]
                                if i>=network.n_wall_junction: #Cell node
                                    myfile.write("1 " + str(i) + " \n")
                            myfile.write(" \n")
                            myfile.write("CELL_TYPES " + str(network.n_cells) + " \n") #
                            for node, edges in network.graph.adjacency():
                                i=indice[node]
                                if i>=network.n_wall_junction: #Cell node
                                    myfile.write("1 \n")
                            myfile.write(" \n")
                            myfile.write("POINT_DATA " + str(len(network.graph.nodes)) + " \n")
                            myfile.write("SCALARS Cell_pressure float \n")
                            myfile.write("LOOKUP_TABLE default \n")
                            for node in network.graph:
                                myfile.write(str(float(soln[node])) + " \n") #Line cell type      min(sath01,max(satl01,   ))
                        myfile.close()
                        text_file.close()
                        
                    
                    if general.paraview_mf==1: #3D visualization of membrane fluxes
                        text_file = open(newpath+"Membranes3Db"+str(Barrier)+"_"+str(iMaturity)+"s"+str(count)+".pvtk", "w")
                        with open(newpath+"Membranes3Db"+str(Barrier)+"_"+str(iMaturity)+"s"+str(count)+".pvtk", "a") as myfile:
                            myfile.write("# vtk DataFile Version 4.0 \n")
                            myfile.write("Membranes geometry 3D \n")
                            myfile.write("ASCII \n")
                            myfile.write(" \n")
                            myfile.write("DATASET UNSTRUCTURED_GRID \n")
                            myfile.write("POINTS "+str(len(thick_wall)*2)+" float \n")
                            for ThickWallNode in thick_wall:
                                myfile.write(str(ThickWallNode[3]) + " " + str(ThickWallNode[4]) + " 0.0 \n")
                            for ThickWallNode in thick_wall:
                                myfile.write(str(ThickWallNode[3]) + " " + str(ThickWallNode[4]) + " " + str(height) + " \n")
                            myfile.write(" \n")
                            myfile.write("CELLS " + str(len(thick_wall)-len(list_ghostwalls)*4) + " " + str(len(thick_wall)*5-len(list_ghostwalls)*20) + " \n") #The number of cells corresponds to the number of lines in thick_wall
                            for ThickWallNode in thick_wall:
                                if ThickWallNode[1]>=network.n_walls: #wall that is a junction
                                    if thick_wall[int(ThickWallNode[5])][1] not in list_ghostwalls:
                                        myfile.write("4 " + str(int(ThickWallNode[0])) + " " + str(int(ThickWallNode[5])) + " " + str(int(ThickWallNode[5])+len(thick_wall)) + " " + str(int(ThickWallNode[0])+len(thick_wall)) + " \n") #All network.cellset['points'] were repeated twice (once at z=0 and once at z=height), so adding len(thick_wall) is the same point at z=height
                                    if thick_wall[int(ThickWallNode[6])][1] not in list_ghostwalls:
                                        myfile.write("4 " + str(int(ThickWallNode[0])) + " " + str(int(ThickWallNode[6])) + " " + str(int(ThickWallNode[6])+len(thick_wall)) + " " + str(int(ThickWallNode[0])+len(thick_wall)) + " \n")
                            myfile.write(" \n")
                            myfile.write("CELL_TYPES " + str(len(thick_wall)-len(list_ghostwalls)*4) + " \n")
                            for ThickWallNode in thick_wall:
                                if ThickWallNode[1]>=network.n_walls: #wall that is a junction
                                    if thick_wall[int(ThickWallNode[5])][1] not in list_ghostwalls:
                                        myfile.write("9 \n") #Quad cell type
                                    if thick_wall[int(ThickWallNode[6])][1] not in list_ghostwalls:
                                        myfile.write("9 \n") #Quad cell type
                            myfile.write(" \n")
                            myfile.write("POINT_DATA " + str(len(thick_wall)*2) + " \n")
                            myfile.write("SCALARS TM_flux_(m/s) float \n")
                            myfile.write("LOOKUP_TABLE default \n")
                            for ThickWallNode in thick_wall:
                                if ThickWallNode[0]<len(MembraneFlowDensity):
                                    myfile.write(str(float(MembraneFlowDensity[int(ThickWallNode[0])])/sperd/cmperm) + " \n") #Flow rate from wall (non junction) to cell   min(sath1,max(satl1,  ))
                                else:
                                    myfile.write(str(float((MembraneFlowDensity[int(ThickWallNode[5])]+MembraneFlowDensity[int(ThickWallNode[6])])/2)/sperd/cmperm) + " \n") #Flow rate from junction wall to cell is the average of the 2 neighbouring wall flow rates   min(sath1,max(satl1,  ))
                            for ThickWallNode in thick_wall:
                                if ThickWallNode[0]<len(MembraneFlowDensity):
                                    myfile.write(str(float(MembraneFlowDensity[int(ThickWallNode[0])])/sperd/cmperm) + " \n") #Flow rate from wall (non junction) to cell   min(sath1,max(satl1,  ))
                                else:
                                    myfile.write(str(float((MembraneFlowDensity[int(ThickWallNode[5])]+MembraneFlowDensity[int(ThickWallNode[6])])/2)/sperd/cmperm) + " \n") #Flow rate from junction wall to cell is the average of the 2 neighbouring wall flow rates   min(sath1,max(satl1,  ))
                        myfile.close()
                        text_file.close()
                    
                    if general.paraview_wf==1: #Wall flow density data
                        maxWallFlowDensity=0.0
                        for ir in range(int(len(WallFlowDensity))):
                            maxWallFlowDensity=max(maxWallFlowDensity,abs(WallFlowDensity[ir][2]))
                        sath2=maxWallFlowDensity*general.color_threshold #(1-(1-general.color_threshold)/2)
                        #satl2=0.0
                        text_file = open(newpath+"WallsThick3D_bottomb"+str(Barrier)+"_"+str(iMaturity)+"s"+str(count)+".pvtk", "w")
                        with open(newpath+"WallsThick3D_bottomb"+str(Barrier)+"_"+str(iMaturity)+"s"+str(count)+".pvtk", "a") as myfile:
                            myfile.write("# vtk DataFile Version 4.0 \n")
                            myfile.write("Wall geometry 3D including geometry.thickness bottom \n")
                            myfile.write("ASCII \n")
                            myfile.write(" \n")
                            myfile.write("DATASET UNSTRUCTURED_GRID \n")
                            myfile.write("POINTS "+str(len(thick_wall_x))+" float \n")
                            for ThickWallNodeX in thick_wall_x:
                                myfile.write(str(ThickWallNodeX[1]) + " " + str(ThickWallNodeX[2]) + " 0.0 \n")
                            myfile.write(" \n")
                            myfile.write("CELLS " + str(int(network.n_wall_junction+network.n_walls-len(list_ghostwalls)*2-len(list_ghostjunctions))) + " " + str(int(2*network.n_walls*5-len(list_ghostwalls)*10+sum(n_wall_to_wall_x[network.n_walls:])+network.n_wall_junction-network.n_walls+2*len(wall_to_wall_x[network.n_walls:])-nGhostJunction2Wall-len(list_ghostjunctions))) + " \n") #The number of cells corresponds to the number of lines in thick_wall (if no ghost wall & junction)
                            i=0
                            for PolygonX in thick_wall_polygon_x:
                                if floor(i/2) not in list_ghostwalls:
                                    myfile.write("4 " + str(int(PolygonX[0])) + " " + str(int(PolygonX[1])) + " " + str(int(PolygonX[2])) + " " + str(int(PolygonX[3])) + " \n")
                                i+=1
                            j=network.n_walls
                            for PolygonX in wall_to_wall_x[network.n_walls:]: #"junction" polygons
                                #Would need to order them based on x or y position to make sure display fully covers the surface (but here we try a simpler not so good solution instead)
                                if j not in list_ghostjunctions:
                                    string=str(int(n_wall_to_wall_x[j]+2)) #Added +2 so that the first and second nodes could be added again at the end (trying to fill the polygon better)
                                    for id1 in range(int(n_wall_to_wall_x[j])):
                                        string=string+" "+str(int(PolygonX[id1]))
                                    string=string+" "+str(int(PolygonX[0]))+" "+str(int(PolygonX[1])) #Adding the 1st and 2nd nodes again to the end
                                    myfile.write(string + " \n")
                                j+=1
                            myfile.write(" \n")
                            myfile.write("CELL_TYPES " + str(network.n_wall_junction+network.n_walls-len(list_ghostwalls)*2-len(list_ghostjunctions)) + " \n")
                            i=0
                            for PolygonX in thick_wall_polygon_x:
                                if floor(i/2) not in list_ghostwalls:
                                    myfile.write("7 \n") #Polygon cell type (wall)
                                i+=1
                            j=network.n_walls
                            for PolygonX in wall_to_wall_x[network.n_walls:]:
                                if j not in list_ghostjunctions:
                                    myfile.write("6 \n") #Triangle-strip cell type (wall junction)
                                j+=1
                            myfile.write(" \n")
                            myfile.write("POINT_DATA " + str(len(thick_wall_x)) + " \n")
                            myfile.write("SCALARS Apo_flux_(m/s) float \n")
                            myfile.write("LOOKUP_TABLE default \n")
                            NewWallFlowDensity=zeros((len(thick_wall_x),2))
                            i=0
                            for PolygonX in thick_wall_polygon_x:
                                for id1 in range(4):
                                    if abs(float(WallFlowDensity[i][2]))>min(NewWallFlowDensity[int(PolygonX[id1])]):
                                        NewWallFlowDensity[int(PolygonX[id1])][0]=max(NewWallFlowDensity[int(PolygonX[id1])])
                                        NewWallFlowDensity[int(PolygonX[id1])][1]=abs(float(WallFlowDensity[i][2]))
                                i+=1
                            for i in range(len(thick_wall_x)):
                                myfile.write(str(float(mean(NewWallFlowDensity[i]))/sperd/cmperm) + " \n")  # min(sath2,  )
                        myfile.close()
                        text_file.close()
                        
                        text_file = open(newpath+"WallsThick3Dcos_bottomb"+str(Barrier)+"_"+str(iMaturity)+"s"+str(count)+".pvtk", "w")
                        with open(newpath+"WallsThick3Dcos_bottomb"+str(Barrier)+"_"+str(iMaturity)+"s"+str(count)+".pvtk", "a") as myfile:
                            myfile.write("# vtk DataFile Version 4.0 \n")
                            myfile.write("Wall geometry 3D including geometry.thickness bottom \n")
                            myfile.write("ASCII \n")
                            myfile.write(" \n")
                            myfile.write("DATASET UNSTRUCTURED_GRID \n")
                            myfile.write("POINTS "+str(len(thick_wall_x))+" float \n")
                            for ThickWallNodeX in thick_wall_x:
                                myfile.write(str(ThickWallNodeX[1]) + " " + str(ThickWallNodeX[2]) + " 0.0 \n")
                            myfile.write(" \n")
                            myfile.write("CELLS " + str(int(network.n_wall_junction+network.n_walls-len(list_ghostwalls)*2-len(list_ghostjunctions))) + " " + str(int(2*network.n_walls*5-len(list_ghostwalls)*10+sum(n_wall_to_wall_x[network.n_walls:])+network.n_wall_junction-network.n_walls+2*len(wall_to_wall_x[network.n_walls:])-nGhostJunction2Wall-len(list_ghostjunctions))) + " \n") #The number of cells corresponds to the number of lines in thick_wall (if no ghost wall & junction)
                            i=0
                            for PolygonX in thick_wall_polygon_x:
                                if floor(i/2) not in list_ghostwalls:
                                    myfile.write("4 " + str(int(PolygonX[0])) + " " + str(int(PolygonX[1])) + " " + str(int(PolygonX[2])) + " " + str(int(PolygonX[3])) + " \n")
                                i+=1
                            j=network.n_walls
                            for PolygonX in wall_to_wall_x[network.n_walls:]: #"junction" polygons
                                #Would need to order them based on x or y position to make sure display fully covers the surface (but here we try a simpler not so good solution instead)
                                if j not in list_ghostjunctions:
                                    string=str(int(n_wall_to_wall_x[j]+2)) #Added +2 so that the first and second nodes could be added again at the end (trying to fill the polygon better)
                                    for id1 in range(int(n_wall_to_wall_x[j])):
                                        string=string+" "+str(int(PolygonX[id1]))
                                    string=string+" "+str(int(PolygonX[0]))+" "+str(int(PolygonX[1])) #Adding the 1st and 2nd nodes again to the end
                                    myfile.write(string + " \n")
                                j+=1
                            myfile.write(" \n")
                            myfile.write("CELL_TYPES " + str(network.n_wall_junction+network.n_walls-len(list_ghostwalls)*2-len(list_ghostjunctions)) + " \n")
                            i=0
                            for PolygonX in thick_wall_polygon_x:
                                if floor(i/2) not in list_ghostwalls:
                                    myfile.write("7 \n") #Polygon cell type (wall)
                                i+=1
                            j=network.n_walls
                            for PolygonX in wall_to_wall_x[network.n_walls:]:
                                if j not in list_ghostjunctions:
                                    myfile.write("6 \n") #Triangle-strip cell type (wall junction)
                                j+=1
                            myfile.write(" \n")
                            myfile.write("POINT_DATA " + str(len(thick_wall_x)) + " \n")
                            myfile.write("SCALARS Apo_flux_cosine_(m/s) float \n")
                            myfile.write("LOOKUP_TABLE default \n")
                            NewWallFlowDensity_cos=zeros((len(thick_wall_x),2))
                            i=0
                            for PolygonX in thick_wall_polygon_x:
                                for id1 in range(4):
                                    if abs(float(WallFlowDensity_cos[i][2]))>min(abs(NewWallFlowDensity_cos[int(PolygonX[id1])])):
                                        #Horizontal component of the flux
                                        if abs(NewWallFlowDensity_cos[int(PolygonX[id1])][1])>abs(NewWallFlowDensity_cos[int(PolygonX[id1])][0]): #Keeping the most extreme value
                                            NewWallFlowDensity_cos[int(PolygonX[id1])][0]=NewWallFlowDensity_cos[int(PolygonX[id1])][1]
                                        NewWallFlowDensity_cos[int(PolygonX[id1])][1]=float(WallFlowDensity_cos[i][2])
                                i+=1
                            for i in range(len(thick_wall_x)):
                                myfile.write(str(float(mean(NewWallFlowDensity_cos[i]))/sperd/cmperm) + " \n")  # min(sath2,  )
                        myfile.close()
                        text_file.close()
                    
                        if Barrier>0:
                            text_file = open(newpath+"InterC3D_bottomb"+str(Barrier)+"_"+str(iMaturity)+"s"+str(count)+".pvtk", "w")
                            with open(newpath+"InterC3D_bottomb"+str(Barrier)+"_"+str(iMaturity)+"s"+str(count)+".pvtk", "a") as myfile:
                                myfile.write("# vtk DataFile Version 4.0 \n")
                                myfile.write("Intercellular space geometry 3D \n")
                                myfile.write("ASCII \n")
                                myfile.write(" \n")
                                myfile.write("DATASET UNSTRUCTURED_GRID \n")
                                myfile.write("POINTS "+str(len(thick_wall))+" float \n")
                                for ThickWallNode in thick_wall:
                                    myfile.write(str(ThickWallNode[3]) + " " + str(ThickWallNode[4]) + " " + str(height/200) + " \n")
                                myfile.write(" \n")
                                myfile.write("CELLS " + str(len(geometry.intercellular_ids)) + " " + str(int(len(geometry.intercellular_ids)+sum(n_cell_to_thick_wall[geometry.intercellular_ids]))) + " \n") #The number of cells corresponds to the number of intercellular spaces
                                InterCFlowDensity=zeros((network.n_cells,1))
                                for cid in geometry.intercellular_ids:
                                    n=int(n_cell_to_thick_wall[cid]) #Total number of thick wall nodes around the protoplast
                                    Polygon=cell_to_thick_wall[cid][:n]
                                    ranking=list()
                                    ranking.append(int(Polygon[0]))
                                    ranking.append(thick_wall[int(ranking[0])][5])
                                    ranking.append(thick_wall[int(ranking[0])][6])
                                    for id1 in range(1,n):
                                        wid1=thick_wall[int(ranking[id1])][5]
                                        wid2=thick_wall[int(ranking[id1])][6]
                                        if wid1 not in ranking:
                                            ranking.append(wid1)
                                        if wid2 not in ranking:
                                            ranking.append(wid2)
                                    string=str(n)
                                    for id1 in ranking:
                                        string=string+" "+str(int(id1))
                                    myfile.write(string + " \n")
                                    for twpid in Polygon[:int(n/2)]: #The first half of nodes are wall nodes actually connected to cells
                                        InterCFlowDensity[cid]+=abs(MembraneFlowDensity[int(twpid)])/n #Mean absolute flow density calculation
                                myfile.write(" \n")
                                myfile.write("CELL_TYPES " + str(len(geometry.intercellular_ids)) + " \n")
                                for i in range(len(geometry.intercellular_ids)):
                                    myfile.write("6 \n") #Triangle-strip cell type
                                myfile.write(" \n")
                                myfile.write("POINT_DATA " + str(len(thick_wall)) + " \n")
                                myfile.write("SCALARS Apo_flux_(m/s) float \n")
                                myfile.write("LOOKUP_TABLE default \n")
                                for ThickWallNode in thick_wall:
                                    cell_id=ThickWallNode[2]-network.n_wall_junction
                                    myfile.write(str(float(InterCFlowDensity[int(cell_id)])/sperd/cmperm) + " \n") #Flow rate from wall (non junction) to cell    min(sath1,max(satl1,  ))
                            myfile.close()
                            text_file.close()
                    
                    
                    
                    if general.paraview_pf==1: #Plasmodesmata flow density data disks
                        text_file = open(newpath+"Plasmodesm3Db"+str(Barrier)+"_"+str(iMaturity)+"s"+str(count)+".pvtk", "w")
                        
                        with open(newpath+"Plasmodesm3Db"+str(Barrier)+"_"+str(iMaturity)+"s"+str(count)+".pvtk", "a") as myfile:
                            myfile.write("# vtk DataFile Version 4.0 \n")
                            myfile.write("PD flux disks 3D \n")
                            myfile.write("ASCII \n")
                            myfile.write(" \n")
                            myfile.write("DATASET UNSTRUCTURED_GRID \n")
                            myfile.write("POINTS "+str(len(PlasmodesmFlowDensity)*12)+" float \n")
                            for ThickWallNode in thick_wall:
                                if ThickWallNode[1]<network.n_walls: #selection of new walls (not new junctions)
                                    if ThickWallNode[7]==0: #new walls that are not at the interface with soil or xylem, where there is no plasmodesmata   #if network.graph.nodes[int(ThickWallNode[1])]['borderlink']==0
                                        #calculate the XY slope between the two neighbouring new junctions
                                        twpid1=int(ThickWallNode[5])
                                        twpid2=int(ThickWallNode[6])
                                        if not thick_wall[twpid1][3]==thick_wall[twpid2][3]: #Otherwise we'll get a division by 0 error
                                            slopeNJ=(thick_wall[twpid1][4]-thick_wall[twpid2][4])/(thick_wall[twpid1][3]-thick_wall[twpid2][3]) #slope of the line connecting the new junction nodes neighbouring the new wall
                                        else:
                                            slopeNJ=inf
                                        x0=ThickWallNode[3]
                                        y0=ThickWallNode[4]
                                        z0=general.radius_plasmodesm_disp*3
                                        #Calculate the horizontal distance between XY0 and the cell center, compare it with the distance between the mean position of the new junctions. If the latter is closer to the cell center, it becomes the new XY0 to make sur the disk is visible
                                        xC=position[int(ThickWallNode[2])][0]
                                        yC=position[int(ThickWallNode[2])][1]
                                        xNJ=(thick_wall[twpid1][3]+thick_wall[twpid2][3])/2.0
                                        yNJ=(thick_wall[twpid1][4]+thick_wall[twpid2][4])/2.0
                                        if sqrt(square(x0-xC)+square(y0-yC)) > sqrt(square(xNJ-xC)+square(yNJ-yC)):
                                            x0=xNJ
                                            y0=yNJ
                                        for i in range(12):
                                            x=x0+cos(arctan(slopeNJ))*general.radius_plasmodesm_disp*cos(int(i)*pi/6.0)
                                            y=y0+sin(arctan(slopeNJ))*general.radius_plasmodesm_disp*cos(int(i)*pi/6.0)
                                            z=z0+general.radius_plasmodesm_disp*sin(int(i)*pi/6.0)
                                            myfile.write(str(x) + " " + str(y) + " " + str(z) + " \n")
                                else:
                                    break #interrupts the for loop in case we reached the new junction nodes
                            myfile.write(" \n")
                            myfile.write("CELLS " + str(len(PlasmodesmFlowDensity)) + " " + str(len(PlasmodesmFlowDensity)*13) + " \n") #The number of cells corresponds to the number of lines in thick_wall
                            for i in range(len(PlasmodesmFlowDensity)):
                                if PlasmodesmFlowDensity[i]==0:
                                    myfile.write("12 " + str(i*12+0) + " " + str(i*12+0) + " " + str(i*12+0) + " " + str(i*12+0) + " " + str(i*12+0) + " " + str(i*12+0) + " " + str(i*12+0) + " " + str(i*12+0) + " " + str(i*12+0) + " " + str(i*12+0) + " " + str(i*12+0) + " " + str(i*12+0) + " \n")
                                else:
                                    myfile.write("12 " + str(i*12+0) + " " + str(i*12+1) + " " + str(i*12+2) + " " + str(i*12+3) + " " + str(i*12+4) + " " + str(i*12+5) + " " + str(i*12+6) + " " + str(i*12+7) + " " + str(i*12+8) + " " + str(i*12+9) + " " + str(i*12+10) + " " + str(i*12+11) + " \n")
                            myfile.write(" \n")
                            myfile.write("CELL_TYPES " + str(len(PlasmodesmFlowDensity)) + " \n")
                            for i in range(len(PlasmodesmFlowDensity)):
                                myfile.write("7 \n") #Polygon cell type 
                            myfile.write(" \n")
                            myfile.write("POINT_DATA " + str(len(PlasmodesmFlowDensity)*12) + " \n")
                            myfile.write("SCALARS PD_Flux_(m/s) float \n")
                            myfile.write("LOOKUP_TABLE default \n")
                            for i in range(len(PlasmodesmFlowDensity)):
                                for j in range(12):
                                    myfile.write(str(float(PlasmodesmFlowDensity[i])/sperd/cmperm) + " \n") #min(sath3,max(satl3, ))
                        myfile.close()
                        text_file.close()
                    
                    
                    if general.paraview_mf==1 and general.paraview_pf==1: #Membranes and plasmodesms in the same file
                        text_file = open(newpath+"Membranes_n_plasmodesm3Db"+str(Barrier)+"_"+str(iMaturity)+"s"+str(count)+".pvtk", "w")
                        with open(newpath+"Membranes_n_plasmodesm3Db"+str(Barrier)+"_"+str(iMaturity)+"s"+str(count)+".pvtk", "a") as myfile:
                            myfile.write("# vtk DataFile Version 4.0 \n")
                            myfile.write("Membranes geometry and plasmodesm disks 3D \n")
                            myfile.write("ASCII \n")
                            myfile.write(" \n")
                            myfile.write("DATASET UNSTRUCTURED_GRID \n")
                            myfile.write("POINTS "+str(len(thick_wall)*2+len(PlasmodesmFlowDensity)*12)+" float \n")
                            for ThickWallNode in thick_wall:
                                myfile.write(str(ThickWallNode[3]) + " " + str(ThickWallNode[4]) + " 0.0 \n")
                            for ThickWallNode in thick_wall:
                                myfile.write(str(ThickWallNode[3]) + " " + str(ThickWallNode[4]) + " " + str(height) + " \n")
                            for ThickWallNode in thick_wall:
                                if ThickWallNode[1]<network.n_walls: #selection of new walls (not new junctions)
                                    if ThickWallNode[7]==0: #new walls that are not at the interface with soil or xylem, where there is no plasmodesmata   #if network.graph.nodes[int(ThickWallNode[1])]['borderlink']==0
                                        #calculate the XY slope between the two neighbouring new junctions
                                        twpid1=int(ThickWallNode[5])
                                        twpid2=int(ThickWallNode[6])
                                        if not thick_wall[twpid1][3]==thick_wall[twpid2][3]: #Otherwise we'll get a division by 0 error
                                            slopeNJ=(thick_wall[twpid1][4]-thick_wall[twpid2][4])/(thick_wall[twpid1][3]-thick_wall[twpid2][3]) #slope of the line connecting the new junction nodes neighbouring the new wall
                                        else:
                                            slopeNJ=inf
                                        x0=ThickWallNode[3]
                                        y0=ThickWallNode[4]
                                        z0=general.radius_plasmodesm_disp*3
                                        #Calculate the horizontal distance between XY0 and the cell center, compare it with the distance between the mean position of the new junctions. If the latter is closer to the cell center, it becomes the new XY0 to make sur the disk is visible
                                        xC=position[int(ThickWallNode[2])][0]
                                        yC=position[int(ThickWallNode[2])][1]
                                        xNJ=(thick_wall[twpid1][3]+thick_wall[twpid2][3])/2.0
                                        yNJ=(thick_wall[twpid1][4]+thick_wall[twpid2][4])/2.0
                                        if sqrt(square(x0-xC)+square(y0-yC)) > sqrt(square(xNJ-xC)+square(yNJ-yC)):
                                            x0=xNJ
                                            y0=yNJ
                                        for i in range(12):
                                            x=x0+cos(arctan(slopeNJ))*general.radius_plasmodesm_disp*cos(int(i)*pi/6.0)
                                            y=y0+sin(arctan(slopeNJ))*general.radius_plasmodesm_disp*cos(int(i)*pi/6.0)
                                            z=z0+general.radius_plasmodesm_disp*sin(int(i)*pi/6.0)
                                            myfile.write(str(x) + " " + str(y) + " " + str(z) + " \n")
                                else:
                                    break #interrupts the for loop in case we reached the new junction nodes
                            myfile.write(" \n")
                            myfile.write("CELLS " + str(len(thick_wall)-len(list_ghostwalls)*4+len(PlasmodesmFlowDensity)) + " " + str(len(thick_wall)*5-len(list_ghostwalls)*20+len(PlasmodesmFlowDensity)*13) + " \n") #The number of cells corresponds to the number of lines in thick_wall
                            for ThickWallNode in thick_wall:
                                if ThickWallNode[1]>=network.n_walls: #wall that is a junction
                                    if thick_wall[int(ThickWallNode[5])][1] not in list_ghostwalls:
                                        myfile.write("4 " + str(int(ThickWallNode[0])) + " " + str(int(ThickWallNode[5])) + " " + str(int(ThickWallNode[5])+len(thick_wall)) + " " + str(int(ThickWallNode[0])+len(thick_wall)) + " \n")
                                    if thick_wall[int(ThickWallNode[6])][1] not in list_ghostwalls:
                                        myfile.write("4 " + str(int(ThickWallNode[0])) + " " + str(int(ThickWallNode[6])) + " " + str(int(ThickWallNode[6])+len(thick_wall)) + " " + str(int(ThickWallNode[0])+len(thick_wall)) + " \n")
                            for i in range(len(PlasmodesmFlowDensity)):
                                if PlasmodesmFlowDensity[i]==0:
                                    myfile.write("12 " + str(i*12+0+len(thick_wall)*2) + " " + str(i*12+0+len(thick_wall)*2) + " " + str(i*12+0+len(thick_wall)*2) + " " + str(i*12+0+len(thick_wall)*2) + " " + str(i*12+0+len(thick_wall)*2) + " " + str(i*12+0+len(thick_wall)*2) + " " + str(i*12+0+len(thick_wall)*2) + " " + str(i*12+0+len(thick_wall)*2) + " " + str(i*12+0+len(thick_wall)*2) + " " + str(i*12+0+len(thick_wall)*2) + " " + str(i*12+0+len(thick_wall)*2) + " " + str(i*12+0+len(thick_wall)*2) + " \n")
                                else:
                                    myfile.write("12 " + str(i*12+0+len(thick_wall)*2) + " " + str(i*12+1+len(thick_wall)*2) + " " + str(i*12+2+len(thick_wall)*2) + " " + str(i*12+3+len(thick_wall)*2) + " " + str(i*12+4+len(thick_wall)*2) + " " + str(i*12+5+len(thick_wall)*2) + " " + str(i*12+6+len(thick_wall)*2) + " " + str(i*12+7+len(thick_wall)*2) + " " + str(i*12+8+len(thick_wall)*2) + " " + str(i*12+9+len(thick_wall)*2) + " " + str(i*12+10+len(thick_wall)*2) + " " + str(i*12+11+len(thick_wall)*2) + " \n")
                            myfile.write(" \n")
                            myfile.write("CELL_TYPES " + str(len(thick_wall)-len(list_ghostwalls)*4+len(PlasmodesmFlowDensity)) + " \n")
                            for ThickWallNode in thick_wall:
                                if ThickWallNode[1]>=network.n_walls: #wall that is a junction
                                    if thick_wall[int(ThickWallNode[5])][1] not in list_ghostwalls:
                                        myfile.write("9 \n") #Quad cell type
                                    if thick_wall[int(ThickWallNode[6])][1] not in list_ghostwalls:
                                        myfile.write("9 \n") #Quad cell type
                            for i in range(len(PlasmodesmFlowDensity)):
                                myfile.write("7 \n") #Polygon cell type 
                            myfile.write(" \n")
                            myfile.write("POINT_DATA " + str(len(thick_wall)*2+len(PlasmodesmFlowDensity)*12) + " \n")
                            myfile.write("SCALARS TM_n_PD_flux_(m/s) float \n")
                            myfile.write("LOOKUP_TABLE default \n")
                            for ThickWallNode in thick_wall:
                                if ThickWallNode[0]<len(MembraneFlowDensity):
                                    myfile.write(str(float(MembraneFlowDensity[int(ThickWallNode[0])])/sperd/cmperm) + " \n") #Flow rate from wall (non junction) to cell
                                else:
                                    myfile.write(str(float((MembraneFlowDensity[int(ThickWallNode[5])]+MembraneFlowDensity[int(ThickWallNode[6])])/2)/sperd/cmperm) + " \n") #Flow rate from junction wall to cell is the average of the 2 neighbouring wall flow rates
                            for ThickWallNode in thick_wall:
                                if ThickWallNode[0]<len(MembraneFlowDensity):
                                    myfile.write(str(float(MembraneFlowDensity[int(ThickWallNode[0])])/sperd/cmperm) + " \n") #Flow rate from wall (non junction) to cell
                                else:
                                    myfile.write(str(float((MembraneFlowDensity[int(ThickWallNode[5])]+MembraneFlowDensity[int(ThickWallNode[6])])/2)/sperd/cmperm) + " \n") #Flow rate from junction wall to cell is the average of the 2 neighbouring wall flow rates
                            for i in range(len(PlasmodesmFlowDensity)):
                                for j in range(12):
                                    myfile.write(str(float(PlasmodesmFlowDensity[i])/sperd/cmperm) + " \n")
                        myfile.close()
                        text_file.close()
            
        print('[5/5] Writing outputs...')
        #write down kr_tot and Uptake distributions in matrices
        iMaturity=-1
        kr_tot_saved = []
        for Maturity in geometry.maturity_elems:
            Barrier=int(Maturity.get("Barrier"))
            height=int(Maturity.get("height")) #(microns)
            iMaturity+=1
            kr_tot_saved.append(kr_tot[iMaturity][0])
            text_file = open(newpath+"Macro_prop_"+str(Barrier)+"_"+str(iMaturity)+".txt", "w")
            with open(newpath+"Macro_prop_"+str(Barrier)+"_"+str(iMaturity)+".txt", "a") as myfile:
                myfile.write("Macroscopic root radial hydraulic properties, apoplastic barrier "+str(Barrier)+","+str(iMaturity)+" \n")
                myfile.write("\n")
                myfile.write(str(boundary.n_scenarios-1)+" scenarios \n")
                myfile.write("\n")
                myfile.write("Cross-section height: "+str(height*1.0E-04)+" cm \n")
                myfile.write("\n")
                myfile.write("Cross-section network.perimeter: "+str(network.perimeter)+" cm \n")
                myfile.write("\n")
                myfile.write("Xylem specific axial conductance: "+str(K_xyl_spec)+" cm^4/hPa/d \n")
                myfile.write("\n")
                myfile.write("Cross-section radial conductivity: "+str(kr_tot[iMaturity][0])+" cm/hPa/d \n")
                myfile.write("\n")
                myfile.write("Number of radial discretization boxes: \n")
                r_discret_txt=' '.join(map(str, network.r_discret.T)) 
                myfile.write(r_discret_txt[1:21]+" \n")
                myfile.write("\n")
                myfile.write("Radial distance from stele centre (microns): \n")
                for j in network.distance_from_center:
                    myfile.write(str(float(j.item()))+" \n")
                myfile.write("\n")
                myfile.write("Standard Transmembrane uptake Fractions (%): \n")
                for j in range(int(network.r_discret[0])):
                    myfile.write(str(STFlayer_plus[j][iMaturity]*100)+" \n")
                myfile.write("\n")
                myfile.write("Standard Transmembrane release Fractions (%): \n")
                for j in range(int(network.r_discret[0])):
                    myfile.write(str(STFlayer_minus[j][iMaturity]*100)+" \n")
                for i in range(1,boundary.n_scenarios):
                    myfile.write("\n")
                    myfile.write("\n")
                    myfile.write("Scenario "+str(i)+" \n")
                    myfile.write("\n")
                    myfile.write("h_x: "+str(Psi_xyl[iMaturity][i])+" hPa \n")
                    myfile.write("\n")
                    myfile.write("h_s: "+str(boundary.scenarios[i]['psi_soil_left'])+" to "+str(boundary.scenarios[i]['psi_soil_right'])+" hPa \n")
                    myfile.write("\n")
                    myfile.write("h_p: "+str(Psi_sieve[iMaturity][i])+" hPa \n")
                    myfile.write("\n")
                    myfile.write("O_x: "+str(boundary.scenarios[i]['osmotic_xyl'])+" to "+str(boundary.scenarios[i]['osmotic_endo'])+" hPa \n")
                    myfile.write("\n")
                    myfile.write("O_s: "+str(boundary.scenarios[i]['osmotic_left'])+" to "+str(boundary.scenarios[i]['osmotic_right'])+" hPa \n")
                    myfile.write("\n")
                    myfile.write("O_p: "+str(Os_sieve[0][i])+" hPa \n")
                    myfile.write("\n")
                    myfile.write("Xcontact: "+str(Xcontact)+" microns \n")
                    myfile.write("\n")
                    if Barrier==0:
                        myfile.write("Elong_cell: "+str(Elong_cell[0][i])+" cm/d \n")
                        myfile.write("\n")
                        myfile.write("Elong_cell_side_diff: "+str(Elong_cell_side_diff[0][i])+" cm/d \n")
                        myfile.write("\n")
                    else:
                        myfile.write("Elong_cell: "+str(0.0)+" cm/d \n")
                        myfile.write("\n")
                        myfile.write("Elong_cell_side_diff: "+str(0.0)+" cm/d \n")
                        myfile.write("\n")
                    myfile.write("kw: "+str(kw)+" cm^2/hPa/d \n")
                    myfile.write("\n")
                    myfile.write("Kpl: "+str(Kpl)+" cm^3/hPa/d \n")
                    myfile.write("\n")
                    myfile.write("kAQP: "+str(kaqp_cortex)+" cm/hPa/d \n")
                    myfile.write("\n")
                    myfile.write("s_hetero: "+str(s_hetero[0][count])+" \n")
                    myfile.write("\n")
                    myfile.write("s_factor: "+str(s_factor[0][count])+" \n")
                    myfile.write("\n")
                    myfile.write("Os_hetero: "+str(Os_hetero[0][count])+" \n")
                    myfile.write("\n")
                    myfile.write("Os_cortex: "+str(Os_cortex[0][count])+" hPa \n")
                    myfile.write("\n")
                    myfile.write("q_tot: "+str(Q_tot[iMaturity][i]/height/1.0E-04)+" cm^2/d \n")
                    myfile.write("\n")
                    myfile.write("Stele, cortex, and epidermis uptake distribution cm^3/d: \n")
                    for j in range(int(network.r_discret[0])):
                        myfile.write(str(UptakeLayer_plus[j][iMaturity][i])+" \n")
                    myfile.write("\n")
                    myfile.write("Stele, cortex, and epidermis release distribution cm^3/d: \n")
                    for j in range(int(network.r_discret[0])):
                        myfile.write(str(UptakeLayer_minus[j][iMaturity][i])+" \n")
                    myfile.write("\n")
                    myfile.write("Xylem uptake distribution cm^3/d: \n")
                    for j in range(int(network.r_discret[0])):
                        myfile.write(str(Q_xyl_layer[j][iMaturity][i])+" \n")
                    myfile.write("\n")
                    myfile.write("Phloem uptake distribution cm^3/d: \n")
                    for j in range(int(network.r_discret[0])):
                        myfile.write(str(Q_sieve_layer[j][iMaturity][i])+" \n")
                    myfile.write("\n")
                    myfile.write("Elongation flow convergence distribution cm^3/d: \n")
                    for j in range(int(network.r_discret[0])):
                        myfile.write(str(Q_elong_layer[j][iMaturity][i])+" \n")
                    myfile.write("\n")
                    myfile.write("Cell layers pressure potentials: \n")
                    for j in range(int(network.r_discret[0])):
                        myfile.write(str(PsiCellLayer[j][iMaturity][i])+" \n")
                    myfile.write("\n")
                    myfile.write("Cell layers osmotic potentials: \n")
                    for j in range(int(network.r_discret[0])):
                        myfile.write(str(OsCellLayer[j][iMaturity][i])+" \n")
                    myfile.write("\n")
                    myfile.write("Wall layers pressure potentials: \n")
                    for j in range(int(network.r_discret[0])):
                        if NWallLayer[j][iMaturity][i]>0:
                            myfile.write(str(PsiWallLayer[j][iMaturity][i]/NWallLayer[j][iMaturity][i])+" \n")
                        else:
                            myfile.write("nan \n")
                    myfile.write("\n")
                    myfile.write("Wall layers osmotic potentials: \n")
                    for j in range(int(network.r_discret[0])):
                        myfile.write(str(OsWallLayer[j][iMaturity][i])+" \n")
            myfile.close()
            text_file.close()
        
        return kr_tot_saved
        
        if general.sym_contagion == 1: #write down results of the hydropatterning study
            iMaturity=-1
            for Maturity in geometry.maturity_elems:
                Barrier=int(Maturity.get("Barrier"))
                height=int(Maturity.get("height")) #(microns)
                iMaturity+=1
                text_file = open(newpath+"Hydropatterning_"+str(Barrier)+"_"+str(iMaturity)+".txt", "w")
                with open(newpath+"Hydropatterning_"+str(Barrier)+"_"+str(iMaturity)+".txt", "a") as myfile:
                    myfile.write("Is there symplastic mass flow from source to target cells? Apoplastic barrier "+str(Barrier)+","+str(iMaturity)+" \n")
                    myfile.write("\n")
                    myfile.write(str(boundary.n_scenarios-1)+" scenarios \n")
                    myfile.write("\n")
                    myfile.write("Template: "+path+" \n")
                    myfile.write("\n")
                    myfile.write("Source cell: "+str(hormones.sym_zombie0)+" \n")
                    myfile.write("\n")
                    myfile.write("Target cells: "+str(hormones.sym_target)+" \n")
                    myfile.write("\n")
                    myfile.write("Immune cells: "+str(hormones.sym_immune)+" \n")
                    myfile.write("\n")
                    myfile.write("Cross-section height: "+str(height*1.0E-04)+" cm \n")
                    myfile.write("\n")
                    myfile.write("Cross-section network.perimeter: "+str(network.perimeter)+" cm \n")
                    myfile.write("\n")
                    myfile.write("Xcontact: "+str(Xcontact)+" microns \n")
                    myfile.write("\n")
                    myfile.write("kw: "+str(kw)+" cm^2/hPa/d \n")
                    myfile.write("\n")
                    myfile.write("Kpl: "+str(Kpl)+" cm^3/hPa/d \n")
                    myfile.write("\n")
                    myfile.write("kAQP: "+str(kaqp_cortex)+" cm/hPa/d \n")
                    myfile.write("\n")
                    if Barrier==0:
                        myfile.write("Cell elongation rate: "+str(Elong_cell)+" cm/d \n")
                    else: #No elongation after formation of the Casparian strip
                        myfile.write("Cell elongation rate: "+str(0.0)+" cm/d \n")
                    myfile.write("\n")
                    for i in range(1,boundary.n_scenarios):
                        myfile.write("\n")
                        myfile.write("\n")
                        myfile.write("Scenario "+str(i)+" \n")
                        myfile.write("\n")
                        myfile.write("Expected hydropatterining response (1: Wet-side XPP; -1 to 0: Unclear; 2: Dry-side XPP) \n")
                        myfile.write("Hydropat.: "+str(int(Hydropatterning[iMaturity][i]))+" \n")
                        myfile.write("\n")
                        myfile.write("h_x: "+str(Psi_xyl[iMaturity][i])+" hPa, h_s: "+str(boundary.scenarios[i]['psi_soil_left'])+" to "+str(boundary.scenarios[i]['psi_soil_right'])+" hPa, h_p: "+str(Psi_sieve[iMaturity][i])+" hPa \n")
                        myfile.write("\n")
                        myfile.write("O_x: "+str(boundary.scenarios[i]['osmotic_xyl'])+" to "+str(boundary.scenarios[i]['osmotic_xyl'])+" hPa, O_s: "+str(boundary.scenarios[i]['osmotic_left'])+" to "+str(Os_soil[1][i])+" hPa, O_p: "+str(Os_sieve[0][i])+" hPa \n")
                        myfile.write("\n")
                        myfile.write("Os_cortex: "+str(Os_cortex[0][count])+" hPa, Os_hetero: "+str(Os_hetero[0][count])+", s_hetero: "+str(s_hetero[0][count])+", s_factor: "+str(s_factor[0][count])+" \n")
                        myfile.write("\n")
                        myfile.write("q_tot: "+str(Q_tot[iMaturity][i]/height/1.0E-04)+" cm^2/d \n")
                        myfile.write("\n")
                myfile.close()
                text_file.close()
        
        if general.apo_contagion == 1: #write down results of the hydrotropism study
            iMaturity=-1
            for Maturity in geometry.maturity_elems:
                Barrier=int(Maturity.get("Barrier"))
                height=int(Maturity.get("height")) #(microns)
                iMaturity+=1
                text_file = open(newpath+"Hydrotropism_"+str(Barrier)+"_"+str(iMaturity)+".txt", "w")
                with open(newpath+"Hydrotropism_"+str(Barrier)+"_"+str(iMaturity)+".txt", "a") as myfile:
                    myfile.write("Is there apoplastic mass flow from source to target cells? Apoplastic barrier "+str(Barrier)+","+str(iMaturity)+" \n")
                    myfile.write("\n")
                    myfile.write(str(boundary.n_scenarios-1)+" scenarios \n")
                    myfile.write("\n")
                    myfile.write("Template: "+path+" \n")
                    myfile.write("\n")
                    myfile.write("Source cell: "+str(hormones.apo_zombie0)+" \n")
                    myfile.write("\n")
                    myfile.write("Target cells: "+str(hormones.apo_target)+" \n")
                    myfile.write("\n")
                    myfile.write("Immune cells: "+str(hormones.apo_immune)+" \n")
                    myfile.write("\n")
                    myfile.write("Cross-section height: "+str(height*1.0E-04)+" cm \n")
                    myfile.write("\n")
                    myfile.write("Cross-section network.perimeter: "+str(network.perimeter)+" cm \n")
                    myfile.write("\n")
                    myfile.write("Xcontact: "+str(Xcontact)+" microns \n")
                    myfile.write("\n")
                    myfile.write("kw: "+str(kw)+" cm^2/hPa/d \n")
                    myfile.write("\n")
                    myfile.write("Kpl: "+str(Kpl)+" cm^3/hPa/d \n")
                    myfile.write("\n")
                    myfile.write("kAQP: "+str(kaqp_cortex)+" cm/hPa/d \n")
                    myfile.write("\n")
                    if Barrier==0:
                        myfile.write("Cell elongation rate: "+str(Elong_cell)+" cm/d \n")
                    else: #No elongation after formation of the Casparian strip
                        myfile.write("Cell elongation rate: "+str(0.0)+" cm/d \n")
                    myfile.write("\n")
                    for i in range(1,boundary.n_scenarios):
                        myfile.write("\n")
                        myfile.write("\n")
                        myfile.write("Scenario "+str(i)+" \n")
                        myfile.write("\n")
                        myfile.write("Expected hydrotropism response (1: All cell walls reached by ABA; 0: No target walls reached by ABA) \n")
                        myfile.write("Hydropat.: "+str(int(Hydrotropism[iMaturity][i]))+" \n")
                        myfile.write("\n")
                        myfile.write("h_x: "+str(Psi_xyl[iMaturity][i])+" hPa, h_s: "+str(boundary.scenarios[i]['psi_soil_left'])+" to "+str(boundary.scenarios[i]['psi_soil_right'])+" hPa, h_p: "+str(Psi_sieve[iMaturity][i])+" hPa \n")
                        myfile.write("\n")
                        myfile.write("O_x: "+str(boundary.scenarios[i]['osmotic_xyl'])+" to "+str(boundary.scenarios[i]['osmotic_xyl'])+" hPa, O_s: "+str(boundary.scenarios[i]['osmotic_left'])+" to "+str(Os_soil[1][i])+" hPa, O_p: "+str(Os_sieve[0][i])+" hPa \n")
                        myfile.write("\n")
                        myfile.write("Os_cortex: "+str(Os_cortex[0][count])+" hPa, Os_hetero: "+str(Os_hetero[0][count])+", s_hetero: "+str(s_hetero[0][count])+", s_factor: "+str(s_factor[0][count])+" \n")
                        myfile.write("\n")
                        myfile.write("q_tot: "+str(Q_tot[iMaturity][i]/height/1.0E-04)+" cm^2/d \n")
                        myfile.write("\n")
                myfile.close()
                text_file.close()



def update_xml_attributes(file_path, parent_tag, child_tag, updates, output_path=None):
    """
    Update one or more attributes of an XML element.
    Works for parent+child (e.g. <hydraulic.kaqp_elems><kAQP .../></hydraulic.kaqp_elems>)
    or standalone tags (e.g. <km value="1" />).

    Parameters
    ----------
    file_path : str
        Path to the input XML file.
    parent_tag : str
        Name of the parent element (e.g., "hydraulic.kaqp_elems").
    child_tag : str
        Name of the child element inside the parent (e.g., "kAQP").
        If the tag has no parent, pass the same value for parent_tag and child_tag.
    updates : dict
        Dictionary of attribute updates, e.g. {"value": 0.002, "cortex_factor": 0.9}.
    output_path : str or None
        Path to save the modified XML file. If None, overwrites the input file.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    # handle parent+child or standalone
    if parent_tag == child_tag:
        elem = root.find(f".//{parent_tag}")
    else:
        elem = root.find(f".//{parent_tag}/{child_tag}")

    if elem == None:
        raise ValueError(f"No <{child_tag}> element found (parent={parent_tag}).")

    # apply all updates
    for attr, val in updates.items():
        elem.set(attr, str(val))

    # overwrite or save new file
    if output_path == None:
        output_path = file_path  # overwrite original

    tree.write(output_path, encoding="UTF-8", xml_declaration=True)


def set_hydraulic_scenario(xml_path, barriers):
    """
    Activate one or multiple hydraulic scenarios (Barrier values)
    inside the <geometry.maturity_elems> section of a MECHA XML file.
    Keeps <geometry.maturity_elems> tags intact, adding new barriers if missing.

    Parameters
    ----------
    xml_path : str
        Path to the MECHA XML file
    barriers : int | list[int]
        Barrier value(s) to activate
    """
    if isinstance(barriers, int):
        barriers = [barriers].sort()

    with open(xml_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract the <geometry.maturity_elems> section
    range_match = re.search(r'(<geometry.maturity_elems>)(.*?)(</geometry.maturity_elems>)', content, re.DOTALL)
    if not range_match:
        raise ValueError("No <geometry.maturity_elems> section found in XML.")

    start_tag, inner_text, end_tag = range_match.groups()

    # Match existing <Maturity ... /> lines
    maturity_pattern = re.compile(r'(\s*)(?:<!--\s*)?(<Maturity\s+Barrier="(\d+)"[^>]*\/>)(?:\s*-->)?')

    existing_barriers = {}
    def replacer(match):
        indent, tag, barrier_str = match.groups()
        barrier = int(barrier_str)
        existing_barriers[barrier] = tag  # remember existing tags
        if barrier in barriers:
            return f"{indent}{tag}"  # activate
        else:
            return f"{indent}<!-- {tag} -->"  # deactivate

    # Apply activation/deactivation to existing lines
    new_inner = maturity_pattern.sub(replacer, inner_text)

    # Add missing barriers
    indent_match = re.search(r'(\s*)<Maturity', inner_text)
    indent = indent_match.group(1) if indent_match else '    '  # default 4 spaces
    for barrier in barriers:
        if barrier not in existing_barriers:
            new_inner += f"\n{indent}<Maturity Barrier=\"{barrier}\" height=\"200\" Nlayers=\"1\"/>"

    # Rebuild the <geometry.maturity_elems> section
    new_range_section = f"{start_tag}{new_inner}\n{end_tag}"

    # Replace in the full content
    new_content = content[:range_match.start()] + new_range_section + content[range_match.end():]

    # Write back
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(new_content)

if __name__ == "__main__":
    # This block runs only when the script is executed directly from the terminal.

    parser = argparse.ArgumentParser(
        description="Run the 'mecha' simulation with various configuration files to calculate hydraulic conductance at the anatomical level."
    )

    parser.add_argument(
        "--general-config",
        type=str,
        default='./extdata/General.xml',
        help="Path to the general configuration XML file. Default: './extdata/Maize_General.xml'"
    )
    parser.add_argument(
        "--geometry-config",
        type=str,
        default='./extdata/Geometry.xml',
        help="Path to the geometry configuration XML file. Default: './extdata/Geometry.xml'"
    )
    parser.add_argument(
        "--hydraulic-config",
        type=str,
        default='./extdata/Hydraulics.xml',
        help="Path to the hydraulics configuration XML file. Default: './extdata/Hydraulics.xml'"
    )
    parser.add_argument(
        "--boundary-condition-config",
        type=str,
        default='./extdata/BCs.xml',
        help="Path to the boundary condition XML file. Default: './extdata/BCs.xml'"
    )
    parser.add_argument(
        "--hormones-config",
        type=str,
        default='./extdata/Hormones.xml',
        help="Path to the hormones configuration XML file. Default: '../extdata/Hormones.xml'"
    )
    parser.add_argument(
        "--cellset-file",
        type=str,
        default='./extdata/current_root.xml',
        help="Path to the cellset XML file. Default: '../extdata/current_root.xml'"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.getcwd(), # Use os.getcwd() for the default output directory
        help=f"Directory for output files. Default: Current working directory ({os.getcwd()})"
    )

    args = parser.parse_args()

    print("Script launched from terminal. Parsing arguments...")
    # Call the mecha function with the parsed arguments
    mecha(
        general_config=args.general_config,
        geometry_config=args.geometry_config,
        hydraulic_config=args.hydraulic_config,
        boundary_condition_config=args.boundary_condition_config,
        hormones_config=args.hormones_config,
        cellset_file=args.cellset_file,
        outdir=args.outdir
    )
else:
    # This block executes if the script is imported as a module.
    print("Script imported as a module. Argparse arguments will not be processed.")
    print("You can call 'mecha' function directly from the importing script.")
