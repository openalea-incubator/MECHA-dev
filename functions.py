import xml.etree.ElementTree as ET
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx 
import time 
from numpy import genfromtxt #Load data from a text file, with missing values handled as specified.
from numpy.random import *  # for random sampling
from scipy import sparse
from scipy.sparse import lil_matrix,csr_matrix,spdiags,identity
from scipy.sparse.linalg import spsolve
import scipy.linalg as slin #Linear algebra functions
import math
import pylab #Found in the package pyqt, also needs to be installed for proper use of matlab-python connectivity
from pylab import *  # for plotting
from decimal import Decimal
from lxml import etree #Tree element analysis module
import sys, os 
# =======================
class Input:
    def __init__(self, filename):
        self.tree = ET.parse(filename)
        self.root = self.tree.getroot()
        
    def get_parameter(self, from_ = "", name ="", attribute = "value", type ="null"):
        elem = self.root.find("./{}/{}".format(from_, name))
        if(type == "int"):
            return int(elem.get(attribute))
        elif (type == "float"):
            return float(elem.get(attribute))
        else:
            return elem.get(attribute)

    
    def get_all(self, from_, name, attribute="value"):
        elems = self.root.findall("./{}/{}".format(from_, name))
        return [elem.get(attribute) for elem in elems]
    
    def change_parameter(self, from_ = "", name ="", attribute= "value", value= 0.0):
        elem = self.root.find("./{}/{}".format(from_, name))
        elem.set(attribute, str(value))
    
    def write_xml(self, file_name):
        self.tree.write(file_name)

# =======================
def load_general(InGen):

    # Define Global
    global OS, Output_path, Paraview, ParaviewWF, ParaviewMF, ParaviewPF, ParaviewWP, ParaviewCP, ParTrack, Sym_Contagion, Apo_Contagion, color_threshold, thickness_disp, thicknessJunction_disp, radiusPlasmodesm_disp, UniXwalls, sparseM
    
    OS = InGen.get_parameter(name='OS')
    Output_path=InGen.get_parameter(name='Output', attribute='path')
    Paraview = InGen.get_parameter(name='Paraview', type='int')
    ParaviewWF = InGen.get_parameter(name='Paraview', attribute='WallFlux', type='int')
    ParaviewMF = InGen.get_parameter(name='Paraview', attribute='MembraneFlux', type='int')
    ParaviewPF = InGen.get_parameter(name='Paraview', attribute='PlasmodesmataFlux', type='int')
    ParaviewWP = InGen.get_parameter(name='Paraview', attribute='WallPot', type='int')
    ParaviewCP = InGen.get_parameter(name='Paraview', attribute='CellPot', type='int')
    ParTrack = InGen.get_parameter(name='ParTrack', type='int')
    Sym_Contagion = InGen.get_parameter(name='Sym_Contagion', type='int')
    Apo_Contagion = InGen.get_parameter(name='Apo_Contagion', type='int')
    color_threshold = InGen.get_parameter(name='color_threshold', type='float')
    thickness_disp = InGen.get_parameter(name='thickness_disp', type='float')
    thicknessJunction_disp = InGen.get_parameter(name='thicknessJunction_disp', type='float')
    radiusPlasmodesm_disp = InGen.get_parameter(name='radiusPlasmodesm_disp', type='float')
    UniXwalls = InGen.get_parameter(name='UniXwalls', type='int')
    sparseM = InGen.get_parameter(name='sparse', type='int')
# =======================
def load_geometry(InGeom):
    
    # Define global
    global Plant, path_geom, im_scale, Maturityrange, Printrange, Xwalls, PileUp, passage_cell_ID, InterCid, InterC_perim_search, InterC_perim1, InterC_perim2, InterC_perim3, InterC_perim4, InterC_perim5, kInterC, cell_per_layer, thickness, PD_section, Xylem_pieces
    
    Plant = InGeom.get_parameter(name='Plant')
    path_geom = InGeom.get_parameter(name='path')
    fpath = InGeom.get_parameter(name='path')
    im_scale = InGeom.get_parameter(name='im_scale', type='float')
    Maturityrange = InGeom.get_all(from_='Maturityrange', name='Maturity')
    Printrange = InGeom.get_all(from_='Printrange', name='Print_layer')
    Xwalls = InGeom.get_parameter(name='Xwalls', type='float')
    PileUp = InGeom.get_parameter(name='PileUp', type='int')
    passage_cell_range = InGeom.get_all(from_='passage_cell_range', name='passage_cell', attribute = "id")
    print(passage_cell_range)
    aerenchyma_range = InGeom.get_all(from_='aerenchyma_range', name='aerenchyma', attribute = "id")
    passage_cell_ID=[]
    # for passage_cell in passage_cell_range:
    #    passage_cell_ID.append(int(passage_cell.get("id")))
    InterCid=list() #Aerenchyma is classified as intercellular space
    # for aerenchyma in aerenchyma_range:
    #    if not int(aerenchyma.get("id"))>9E5 and not int(aerenchyma.get("id"))<0:
    #        InterCid.append(int(aerenchyma.get("id"))) #Cell id starting at 0
    #    else:
    #        print('InterCid #'+str(int(aerenchyma.get("id")))+' excluded')
    # InterC_perim <-- for cellSet data
    InterC_perim_search=InGeom.get_parameter(name='InterC_perim_search', type='int')
    if InterC_perim_search==1:
        InterC_perim1=InGeom.get_parameter(name='InterC_perim1', type='float')
        InterC_perim2=InGeom.get_parameter(name='InterC_perim2', type='float')
        InterC_perim3=InGeom.get_parameter(name='InterC_perim3', type='float')
        InterC_perim4=InGeom.get_parameter(name='InterC_perim4', type='float')
        InterC_perim5=InGeom.get_parameter(name='InterC_perim5', type='float')
    kInterC=InGeom.get_parameter(name='kInterC', type='float')
    cell_per_layer=zeros((2,1))
    cell_per_layer[0][0]=InGeom.get_parameter(name='cell_per_layer', attribute = "cortex", type='float')
    cell_per_layer[1][0]=InGeom.get_parameter(name='cell_per_layer', attribute = "stele", type='float')
    thickness=InGeom.get_parameter(name='thickness', type='float') #micron
    PD_section=InGeom.get_parameter(name='PD_section', type='float') #micron^2
    Xylem_pieces=False
    if InGeom.get_parameter(name='Xylem_pieces', attribute = 'flag', type='float')==1:
        Xylem_pieces=True
# =======================
class Macro_hydro_visu:
    def __init__(self, file):
        f = file.split("\n")
        self.scenario = int_elm(f, "scenario")
        self.height = float_elm(f,"height")
        self.perim = float_elm(f, "perimeter")
        self.Kx = float_elm(f, "Xylem specific axial conductance")
        self.kr = float_elm(f, "Cross-section radial conductivity")
        self.boxes = array_elm(f, "radial discretization boxes")
        self.Layer_dist2 = row_elm(f, "Radial distance from stele centre", "Standard Transmembrane uptake Fractions")
        self.STFlayer_plus = row_elm(f, "Standard Transmembrane uptake Fractions", "Standard Transmembrane release Fractions")
        self.STFlayer_minus = row_elm(f, "Standard Transmembrane release Fractions", "Scenario 1")
        self.type= ["stele", "pericycle", "endodermis", "cortex", "exodermis", "epidermis"]
        self.coef_width_symplast=float(4/5)
        self.mpercm=float(0.01)
        self.dpersec=float(1/3600/24)
        
        k = 0
        self.cell_type= []
        for b in self.boxes[1:]:
            for r in range(b):
                self.cell_type.append(self.type[k])
            k+=1

    def poly_table(df):
        new = []
        dist = []
        type = []
        type.append("soil")
        new.append(0.0)
        dist.append(df['dist'][0]-(df['cell_spacing'][1])/2)
        for i in range(len(df)-1):
            if df['type'][i] == 'endodermis':
                if df['type'][i-1] != "endodermis":
                    new.append(new[-1]+df['STUF'][i])
                    dist.append(dist[-1])
                    type.append("endodermis")
                elif df['type'][i+1] != "endodermis":
                    new.append(new[-1]-df['STRF'][i])
                    new.append(new[-1])
                    new.append(new[-1]+df['STUF'][i])
                    dist.append(dist[-1])
                    dist.append(dist[-1]+(df['cell_spacing'][i+1])/5)#+(df['cell_spacing'][i+1]/2)+(df['cell_spacing'][i+1]/20))
                    dist.append(dist[-1])
                    type.append("endodermis")
                    type.append("wall")
                    type.append("wall")
                else:
                    new.append(new[-1])
                    dist.append(df['dist'][i])
                    type.append("endodermis")
            else:
                type.append(df['type'][i])
                type.append(df['type'][i])
                type.append("wall")
                type.append("wall")
                new.append(new[-1]+df['STUF'][i])
                new.append(new[-1])
                new.append(new[-1]-df['STRF'][i])
                new.append(new[-1])
                dist.append(dist[-1])
                dist.append(dist[-1]+(df['cell_spacing'][i+1])-(df['cell_spacing'][i+1]/10))
                dist.append(dist[-1])
                dist.append(dist[-1]+df['cell_spacing'][i+1]/10)#+(df['cell_spacing'][i+1]/2)+(df['cell_spacing'][i+1]/20))

        type.append(df.iloc[-1]['type'])
        type.append(df.iloc[-1]['type'])
        type.append("wall")
        type.append("wall")
        new.append(new[-1]+df.iloc[-1]['STUF'])
        new.append(new[-1])
        new.append(new[-1]-df.iloc[-1]['STRF'])
        new.append(new[-1])
        dist.append(dist[-1])
        dist.append(dist[-1]+(df.iloc[-1]['cell_spacing'])-(df.iloc[-1]['cell_spacing']/10))
        dist.append(dist[-1])
        dist.append(dist[-1]+(df.iloc[-1]['cell_spacing']/10))#+(df.iloc[-1]['cell_spacing']/2))

        new.append(0.0)
        dist.append(dist[-1])
        type.append(type[-1])

        po = {'STF' : new,'dist':dist, 'type':type }        
        poly = pd.DataFrame(data = po)
        return poly

    def graph_apo_symp(df):
        x = df['dist'].to_list()
        y = df['STF'].to_list()

        z = [max(x), max(x), 0,0]
        w = [0, 100, 100, 0]
        plt.figure()
        plt.fill(z, w, color = 'grey', alpha = 0.3)
        plt.fill(x, y, color = 'b')
        plt.show()
# =======================
def get_elm(strings, pattern):
    x = [pattern in i for i in strings]
    res = [i for i, val in enumerate(x) if val]
    return res
# =======================
def int_elm(f, pattern):
    idx = get_elm(f, pattern)
    tmp = str([f[i] for i in idx])
    temp = int(re.findall(r'\d+',tmp)[0])
    return temp
# =======================
def float_elm(f, pattern):
    idx = get_elm(f, pattern)
    tmp = str([f[i] for i in idx])
    temp = float(re.findall(r'[\d]*[.][\d]+',tmp)[0])
    return temp
# =======================
def array_elm(f, pattern):
    idx = get_elm(f, pattern)
    tmp = str([f[i+1] for i in idx])
    temp = [int(s) for s in re.findall(r'\b\d+\b', tmp)]
    return temp
# =======================
def row_elm(f, pattern1, pattern2):
    start = int(get_elm(f, pattern1)[0])+1
    if pattern2 == "Scenario 1":
        end = int(get_elm(f, pattern2)[0])-2
    else:
        end = int(get_elm(f, pattern2)[0])-1
    y = []
    for i in range(start,end,1):
        y.append(float(re.findall(r'[\d]*[.][\d]+',f[i])[0]))
    return y
# =======================
def plot_partition(fl):
    Hydr = Macro_hydro_visu(fl)
    
    info = {'STUF' : Hydr.STFlayer_plus,
            'STRF' : Hydr.STFlayer_minus, 
            'dist' : Hydr.Layer_dist2,
            'type' : Hydr.cell_type}
    df = pd.DataFrame(data = info)
    df = df[::-1].reset_index(drop=True)
    df['STF diff'] = df['STUF']-df['STRF']
    df['STF'] = df['STF diff'].cumsum()
    df['cell_spacing'] = df['dist'].diff()

    poly = Macro_hydro_visu.poly_table(df)
    Macro_hydro_visu.graph_apo_symp(poly)
# =======================    
def initialize_network(points, Walls_loop, Walls_PD, Cells_loop, newpath, im_scale):
    
    G = nx.Graph() #Full network

    #Creates wall & junction nodes
    # print('Creating network nodes')
    t0 = time.perf_counter()
    Nwalls=len(points)
    Ncells=len(Cells_loop)

    # print(Ncells)

    NwallsJun=Nwalls #Will increment at each new junction node
    Junction_pos={}
    Junction2Wall={}
    nJunction2Wall={}
    position_junctions=empty((Nwalls,4)) #Coordinates of junctions associated to each wall
    position_junctions[:]=NAN
    min_x_wall=inf
    max_x_wall=0
    lengths_ini=zeros((Nwalls,1))
    jid=0
    for p in points: #Loop on wall elements (groups of points)
        wid= int((p.getparent().get)("id")) #wid records the current wall id number
        xprev=inf
        yprev=inf
        length=0.0 #Calculating total wall length
        for r in p: #Loop on points within the wall element to calculate their average X and Y coordinates 
            x= im_scale*float(r.get("x")) #X coordinate of the point
            y= im_scale*float(r.get("y")) #Y coordinate of the point
            if xprev==inf: #First point
                pos="x"+str(x)+"y"+str(y) #Position of the first point
                position_junctions[wid][0]=x
                position_junctions[wid][1]=y
                if pos in Junction_pos:
                    ind=Junction_pos[pos]
                    Junction2Wall[ind].append(wid) #Several cell wall ID numbers can correspond to the same X Y coordinate where they meet
                    nJunction2Wall[ind]+=1
                else: #New junction node
                    Junction_pos[pos]=int(jid)
                    Junction2Wall[jid]=[wid] #Saves the cell wall ID number associated to the junction X Y coordinates
                    nJunction2Wall[jid]=1
                    G.add_node(Nwalls+jid, indice=Nwalls+jid, type="apo", position=(float(x),float(y)), length=0) #Nodes are added at walls junctions (previous nodes corresponded to walls middle points). By default, borderlink is 0, but will be adjusted in next loop
                    jid+=1
            else:
                length+=hypot(x-xprev,y-yprev)
            xprev=x
            yprev=y
        #Last point in the wall
        pos="x"+str(x)+"y"+str(y) #Position of the last point
        position_junctions[wid][2]=x
        position_junctions[wid][3]=y
        if pos in Junction_pos: #Get the junction ID
            ind=Junction_pos[pos]
            Junction2Wall[ind].append(wid) #Several cell wall ID numbers can correspond to the same X Y coordinate where they meet
            nJunction2Wall[ind]+=1
        else: #New junction node
            Junction_pos[pos]=int(jid)
            Junction2Wall[jid]=[wid] #Saves the cell wall ID number associated to the junction X Y coordinates
            nJunction2Wall[jid]=1
            G.add_node(Nwalls+jid, indice=Nwalls+jid, type="apo", position=(float(x),float(y)), length=0) #Nodes are added at walls junctions (previous nodes corresponded to walls middle points). By default, borderlink is 0, but will be adjusted in next loop
            jid+=1
        #Second round, identifying the mid-point of the wall
        xprev=inf
        yprev=inf
        length2=0.0 #Calculating the cumulative wall length in order to obtain the exact position of the mid-length of the wall from known total length
        for r in p: #Second loop to catch the true middle position of the wall
            x= im_scale*float(r.get("x")) #X coordinate of the point
            y= im_scale*float(r.get("y")) #Y coordinate of the point
            if not xprev==inf:
                temp1=hypot(x-xprev,y-yprev) #length of the current piece of wall
                if temp1==0:
                    print('Warning null wall segment length! wid:',wid,' x, xprev, y, yprev:',x,xprev,y,yprev)
                    error('error')
                temp2=length2+temp1-length/2 #Cumulative length along the wall
                if temp2>=0: #If beyond the half length of the wall
                    mx=x-(x-xprev)*temp2/temp1 #Middle X coordinate of the wall
                    my=y-(y-yprev)*temp2/temp1 #Middle Y coordinate of the wall
                    break #End the r in p loop
                length2+=temp1
            xprev=x
            yprev=y
        min_x_wall=min(min_x_wall,mx)
        max_x_wall=max(max_x_wall,mx)
        #Creation of the wall node
        G.add_node(wid, indice=wid, type="apo", position=(mx,my), length=length) #Saving wall attributes for graphical display (id, border, type, X and Y coordinates)
        lengths_ini[wid]=length

    NwallsJun=Nwalls+jid
    Ntot=NwallsJun+Ncells
    
    position=nx.get_node_attributes(G,'position') #Nodes XY positions (micrometers)

    #Junction nodes are pointwise by definition so their length is null, except for junctions at root surface, which are attributed a quarter of the length of each surface neighbouring wall for radial transport 
    #lengths=nx.get_node_attributes(G,'length') #Walls lengths (micrometers)
    lengths=zeros((NwallsJun,1))
    lengths[:Nwalls]=lengths_ini

    ##Calculation of the cosine of the trigonometric orientation between horizontal and the junction-wall vector (radian)
    #cos_angle_wall=empty((Nwalls,2))
    #cos_angle_wall[:]=NAN
    #for wid in range(Nwalls):
    #    cos_angle_wall[wid][0]=(position_junctions[wid][0]-position[wid][0])/(hypot(position_junctions[wid][0]-position[wid][0],position_junctions[wid][1]-position[wid][1])) #Vectors junction1-wall
    #    cos_angle_wall[wid][1]=(position_junctions[wid][2]-position[wid][0])/(hypot(position_junctions[wid][2]-position[wid][0],position_junctions[wid][3]-position[wid][1])) #Vectors junction2-wall

    return G, NwallsJun, Ncells, lengths, Junction2Wall, Nwalls, position, position_junctions, min_x_wall, max_x_wall, Ntot
# =======================
def identify_interfaces(NwallsJun, Walls_loop, Cell2Wall_loop, Junction2Wall, Nwalls, lengths):

    Borderlink=2*ones((NwallsJun,1))
    Borderwall=[] #Soil-root interface wall
    Borderaerenchyma=[] #Wall at the surface of aerenchyma
    for w in Walls_loop: #Loop on walls, by cell - wall association, hence a wall can be repeated if associated to two cells
        wid= int(w.get("id")) #Wall id number
        Borderlink[wid]-=1
    for w in Cell2Wall_loop: #Loop on cells. Cell2Wall_loop contains cell wall groups info (one group by cell)
        cgroup=int(w.getparent().get("group")) #Cell type (1=Exodermis;2=epidermis;3=endodermis;4=cortex;5=stele;16=pericycle)
        for r in w: #w points to the cell walls around the current cell
            wid= int(r.get("id")) #Wall id number
            if Borderlink[wid]==1 and cgroup==2: #Wall node at the interface with soil
                if wid not in Borderwall:
                    Borderwall.append(wid)
            elif Borderlink[wid]==1:
                if wid not in Borderaerenchyma:
                    Borderaerenchyma.append(wid)
    #for wid in range(Nwalls):
        
    Borderjunction=[]
    jid=0
    for Junction, Walls in Junction2Wall.items():
        count=0
        length=0
        for wid in Walls:
            if wid in Borderwall:
                count+=1
                length+=lengths[wid]/4.0
        #if count>2: #Should not happen
        #    print('What the count?')
        if count==2:
            Borderjunction.append(jid+Nwalls)
            Borderlink[jid+Nwalls]=1 #Junction node at the interface with soil
            lengths[jid+Nwalls]=length
        else:
            Borderlink[jid+Nwalls]=0
        jid+=1

    return(Borderlink, Borderjunction, Borderaerenchyma, Borderwall)
# =======================
def write_macro(text_file, newpath, b, iMaturity, Nscenarios, Totheight, 
                NWallLayer, PsiWallLayer,
                Nlayers, PileUp, Barr, perimeter, K_xyl_spec, kr_tot,
                Layer_dist2, AxialLayers, STFlayer_plus, TopLayer, STFlayer_minus,
                Os_apo_eq, Os_sym_eq, Os_xyl, Os_soil, Os_sieve, Os_hetero, Os_cortex,
                Xcontacts, Xcontact, 
                Elong_cell, Elong_cell_side_diff, 
                kw, Kpl, kaqp_cortex, s_hetero, s_factor, 
                Q_tot, Q_xyl_layer, Q_sieve_layer, Q_elong_layer, 
                PsiCellLayer, OsCellLayer, OsWallLayer, r_discret,
                Psi_xyl, Psi_soil, Psi_sieve, 
                count, 
                UptakeLayer_plus, UptakeLayer_minus):
    with open(newpath+"Macro_prop_"+str(b)+","+str(iMaturity)+".txt", "a") as myfile:
        myfile.write("Macroscopic root radial hydraulic properties, apoplastic barrier "+str(b)+","+str(iMaturity)+" \n")
        myfile.write("\n")
        myfile.write(str(Nscenarios-1)+" scenarios \n")
        myfile.write("\n")
        myfile.write("Stack height: "+str((Totheight)*1.0E-04)+" cm \n")
        myfile.write("\n")
        myfile.write("Number of zones: "+str(len(Nlayers))+" \n")
        myfile.write("\n")
        temp=str(Nlayers)
        #if len(Nlayers)>1:
        myfile.write("Number of layers: "+temp[1:-1]+" \n")
        #else:
        #    myfile.write("Number of layers: "+temp+" \n")
        myfile.write("\n")
        if PileUp==2:
            temp=str(Barr)
            myfile.write("Type of layers: "+temp[1:-1]+" \n")
        else:
            myfile.write("Type of layers: "+str(b)+" \n")
        myfile.write("\n")
        myfile.write("Cross-section perimeter: "+str(perimeter[0])+" cm \n")
        myfile.write("\n")
        myfile.write("Xylem specific axial conductance: "+str(K_xyl_spec)+" cm^4/hPa/d \n")
        myfile.write("\n")
        myfile.write("Cross-section radial conductivity: "+str(kr_tot[iMaturity][0])+" cm/hPa/d \n")
        myfile.write("\n")
        myfile.write("Number of radial discretization boxes: \n")
        r_discret_txt=' '.join(map(str, r_discret.T)) 
        myfile.write(r_discret_txt[1:21]+" \n")
        myfile.write("\n")
        myfile.write("Radial distance from stele centre (microns): \n")
        for j in Layer_dist2:
            myfile.write(str(float(j))+" \n")
        myfile.write("\n")
        myfile.write("Standard Transmembrane uptake Fractions (%): \n")
        for j in range(int(r_discret[0])):
            if AxialLayers==1:
                myfile.write(str(float(STFlayer_plus[j,TopLayer-AxialLayers:TopLayer]*100))+" \n")
            else:
                temp=str(list(STFlayer_plus[j,TopLayer-AxialLayers:TopLayer]*100))
                myfile.write(temp[1:-1]+" \n")
        myfile.write("\n")
        myfile.write("Standard Transmembrane release Fractions (%): \n")
        for j in range(int(r_discret[0])):
            if AxialLayers==1:
                myfile.write(str(float(STFlayer_minus[j,TopLayer-AxialLayers:TopLayer]*100))+" \n")
            else:
                temp=str(list(STFlayer_minus[j,TopLayer-AxialLayers:TopLayer]*100))
                myfile.write(temp[1:-1]+" \n")
        for i in range(1,Nscenarios):
            myfile.write("\n")
            myfile.write("\n")
            myfile.write("Scenario "+str(i)+" \n")
            myfile.write("\n")
            myfile.write("h_x: "+str(Psi_xyl[1][iMaturity][i])+" hPa \n")
            myfile.write("\n")
            myfile.write("h_s: "+str(Psi_soil[0][i])+" to "+str(Psi_soil[1][i])+" hPa \n")
            myfile.write("\n")
            myfile.write("h_p: "+str(Psi_sieve[1][iMaturity][i])+" hPa \n")
            myfile.write("\n")
            #if AxialLayers>1:
            #    temp=str(list(Os_apo_eq[:,1,i]))
            #    myfile.write("O_apo_stele_eq: "+temp[1:-1]+" hPa \n")
            #    temp=str(list(Os_sym_eq[:,1,i]))
            #    myfile.write("O_sym_stele_eq: "+temp[1:-1]+" hPa \n")
            #    temp=str(list(Os_apo_eq[:,0,i]))
            #    myfile.write("O_apo_cortex_eq: "+temp[1:-1]+" hPa \n")
            #    temp=str(list(Os_sym_eq[:,0,i]))
            #    myfile.write("O_sym_cortex_eq: "+temp[1:-1]+" hPa \n")
            #else:
            myfile.write("O_apo_stele_eq: "+str(Os_apo_eq[iMaturity,1,i])+" hPa \n")
            myfile.write("O_sym_stele_eq: "+str(Os_sym_eq[iMaturity,1,i])+" hPa \n")
            myfile.write("O_apo_cortex_eq: "+str(Os_apo_eq[iMaturity,0,i])+" hPa \n")
            myfile.write("O_sym_cortex_eq: "+str(Os_sym_eq[iMaturity,0,i])+" hPa \n")
            myfile.write("\n")
            myfile.write("O_x: "+str(Os_xyl[0][i])+" to "+str(Os_xyl[1][i])+" hPa \n")
            myfile.write("\n")
            myfile.write("O_s: "+str(Os_soil[0][i])+" to "+str(Os_soil[1][i])+" hPa \n")
            myfile.write("\n")
            myfile.write("O_p: "+str(Os_sieve[0][i])+" hPa \n")
            myfile.write("\n")
            if PileUp==2:
                myfile.write("Xcontact: "+str(Xcontacts)+" microns \n")
            else:
                myfile.write("Xcontact: "+str(Xcontact)+" microns \n")
            myfile.write("\n")
            if b==0:
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
            myfile.write("q_tot: "+str(Q_tot[iMaturity][i]/(Totheight)/1.0E-04)+" cm^2/d \n")
            myfile.write("\n")
            myfile.write("Stele, cortex, and epidermis uptake distribution cm^3/d: \n")
            for j in range(int(r_discret[0])):
                if AxialLayers==1:
                    myfile.write(str(float(UptakeLayer_plus[j,TopLayer-AxialLayers:TopLayer,i]))+" \n")
                else:
                    temp=str(list(UptakeLayer_plus[j,TopLayer-AxialLayers:TopLayer,i]))
                    myfile.write(temp[1:-1]+" \n")
            myfile.write("\n")
            myfile.write("Stele, cortex, and epidermis release distribution cm^3/d: \n")
            for j in range(int(r_discret[0])):
                if AxialLayers==1:
                    myfile.write(str(float(UptakeLayer_minus[j,TopLayer-AxialLayers:TopLayer,i]))+" \n")
                else:
                    temp=str(list(UptakeLayer_minus[j,TopLayer-AxialLayers:TopLayer,i]))
                    myfile.write(temp[1:-1]+" \n")
            myfile.write("\n")
            myfile.write("Xylem uptake distribution cm^3/d: \n")
            for j in range(int(r_discret[0])):
                myfile.write(str(float(Q_xyl_layer[j][iMaturity][i]))+" \n")
            myfile.write("\n")
            myfile.write("Phloem uptake distribution cm^3/d: \n")
            for j in range(int(r_discret[0])):
                myfile.write(str(float(Q_sieve_layer[j][iMaturity][i]))+" \n")
            myfile.write("\n")
            myfile.write("Elongation flow convergence distribution cm^3/d: \n")
            for j in range(int(r_discret[0])):
                myfile.write(str(float(Q_elong_layer[j][iMaturity][i]))+" \n")
            myfile.write("\n")
            myfile.write("Cell layers pressure potentials: \n")
            for j in range(int(r_discret[0])):
                if AxialLayers==1:
                    myfile.write(str(float(PsiCellLayer[j,TopLayer-AxialLayers:TopLayer,i]))+" \n")
                else:
                    temp=str(list(PsiCellLayer[j,TopLayer-AxialLayers:TopLayer,i]))
                    myfile.write(temp[1:-1]+" \n")
            myfile.write("\n")
            myfile.write("Cell layers osmotic potentials: \n")
            if PileUp==2:
                for j in range(int(r_discret[0])):
                    temp=str(list(OsCellLayer[j,:,i]))
                    myfile.write(temp[1:-1]+" \n")
            else:
                for j in range(int(r_discret[0])):
                    myfile.write(str(float(OsCellLayer[j][iMaturity][i]))+" \n")
            myfile.write("\n")
            myfile.write("Wall layers pressure potentials: \n")
            for j in range(int(r_discret[0])):
                if NWallLayer[j][iMaturity][i]>1:
                    if AxialLayers>1:
                        if NWallLayer[j,iMaturity,i]>0:
                            temp=str(list(PsiWallLayer[j,TopLayer-AxialLayers:TopLayer,i]/NWallLayer[j,iMaturity,i]))
                            myfile.write(temp[1:-1]+" \n")
                        else:
                            myfile.write("nan \n")
                    else:
                        if NWallLayer[j,iMaturity,i]>0:
                            myfile.write(str(float(PsiWallLayer[j,TopLayer-AxialLayers:TopLayer,i]/NWallLayer[j,iMaturity,i]))+" \n")
                        else:
                            myfile.write("nan \n")
                else:
                    if AxialLayers>1:
                        temp=str(list(PsiWallLayer[j,TopLayer-AxialLayers:TopLayer,i]))
                        myfile.write(temp[1:-1]+" \n")
                    else:
                        myfile.write(str(float(PsiWallLayer[j,TopLayer-AxialLayers:TopLayer,i]))+" \n")
            myfile.write("\n")
            myfile.write("Wall layers osmotic potentials: \n")
            for j in range(int(r_discret[0])):
                myfile.write(str(float(OsWallLayer[j][iMaturity][i]))+" \n")
        myfile.close()
        text_file.close()
# =======================
def sym_fluxes(dir, Project, inputs, Horm):
    Sym_target_range=etree.parse(dir + Project + inputs + Horm).getroot().xpath('Sym_Contagion/target_range/target')
    Sym_Target=[]
    for target in Sym_target_range:
        Sym_Target.append(int(target.get("id")))
    Sym_immune_range=etree.parse(dir + Project + inputs + Horm).getroot().xpath('Sym_Contagion/immune_range/immune')
    Sym_Immune=[]
    for immune in Sym_immune_range:
        Sym_Immune.append(int(immune.get("id")))
    Apo_source_ini_range=etree.parse(dir + Project + inputs + Horm).getroot().xpath('Apo_Contagion/source_range/Steady-state/source')
    Apo_source_transi_range=etree.parse(dir + Project + inputs + Horm).getroot().xpath('Apo_Contagion/source_range/Transient/source')
    Apo_target_range=etree.parse(dir + Project + inputs + Horm).getroot().xpath('Apo_Contagion/target_range/target')
    Apo_Target=[]
    for target in Apo_target_range:
        Apo_Target.append(int(target.get("id")))
    Apo_immune_range=etree.parse(dir + Project + inputs + Horm).getroot().xpath('Apo_Contagion/immune_range/immune')
    Apo_Immune=[]
    for immune in Apo_immune_range:
        Apo_Immune.append(int(immune.get("id")))

    return(Sym_Target, Sym_Immune, Apo_Target, Apo_Immune)
# =======================x
def get_cell_nodes(G, Cell2Wall_loop, NwallsJun, Apo_Contagion, position):
    listsieve=[]
    listxyl=[]
    listxylwalls=[]
    Apo_w_Target=[]
    Apo_w_Immune=[]
    for w in Cell2Wall_loop: #Loop on cells. Cell2Wall_loop contains cell wall groups info (one group by cell)
        totx=0.0 #Summing up cell walls X positions
        toty=0.0 #Summing up cell walls Y positions
        cellnumber1 = int(w.getparent().get("id")) #Cell ID number
        cgroup=int(w.getparent().get("group")) #Cell type (1=Exodermis;2=epidermis;3=endodermis;4=cortex;5=stele;16=pericycle)
        if cgroup==26:
            error('Please use the label Columella2 for Companion cells in CellSet, MECHA update needed before using the official Companion cell label')
            cgroup=24
        div=float(len(w)) #Total number of walls around the current cell
        for r in w: #w points to the cell walls around the current cell
            wid= int(r.get("id")) #Wall ID number
            totx += position[wid][0] #Contains the walls average X positions
            toty += position[wid][1] #Contains the walls average Y positions
        finalx=totx/div #Average cell X position (from the average position of its walls)
        finaly=toty/div #Average cell Y position (from the average position of its walls)
        G.add_node(NwallsJun + cellnumber1, indice=(NwallsJun) + cellnumber1, type="cell", position = (finalx,finaly), cgroup=cgroup) #Adding cell nodes  borderlink=0,
        if cgroup==23: #Phloem sieve tube
            error('Please use the label "Columella1" for Phloem in CellSet, MECHA update needed in order to use the official CellSet phloem label')
        if cgroup==11 or cgroup==23: #Phloem sieve tube
            listsieve.append(NwallsJun+cellnumber1)
        elif cgroup==13 or cgroup==19 or cgroup==20: #Xylem vessel
            listxyl.append(NwallsJun+cellnumber1)
            for r in w: #w points to the cell walls around the current cell
                wid= int(r.get("id")) #Wall ID number
                listxylwalls.append(wid) #ghost walls crossing xylem vessels will appear twice
        if Apo_Contagion==1:
            if cellnumber1 in Apo_Target:
                for r in w: #w points to the cell walls around the current cell
                    wid= int(r.get("id")) #Wall ID number
                    if wid not in Apo_w_Target:
                        Apo_w_Target.append(wid)
            if cellnumber1 in Apo_Immune:
                for r in w: #w points to the cell walls around the current cell
                    wid= int(r.get("id")) #Wall ID number
                    if wid not in Apo_w_Immune:
                        Apo_w_Immune.append(wid)

    Nxyl=len(listxyl)
    position=nx.get_node_attributes(G,'position') #Updates nodes XY positions (micrometers)
    return Nxyl, position, listxyl, listsieve, Apo_w_Target, Apo_w_Immune
# =======================
def create_network_connections(G, Nwalls, Ncells, Cell2Wall_loop, Walls_loop, 
                               position, NwallsJun, position_junctions,
                               Cell_connec_max, lengths, Junction2Wall):
    t0 = time.perf_counter()
    lat_dists=zeros((Nwalls,1))
    Nmb=0 #Total number of membranes
    cellperimeter=np.linspace(0,0,Ncells)
    cellarea=np.linspace(0,0,Ncells) #(micron^2)
    CellWallsList=[] #Includes both walls & junctions ordered in a consecutive order
    for w in Cell2Wall_loop: #Loop on cells. Cell2Wall_loop contains cell wall groups info (one group by cell)
        cellnumber1 = int(w.getparent().get("id")) #Cell ID number
        i=0
        for r in w: #Loop for wall elements around the cell
            wid= int(r.get("id")) #Cell wall ID
            d_vec=array([position[wid][0]-position[NwallsJun+cellnumber1][0],position[wid][1]-position[NwallsJun+cellnumber1][1]])
            dist_cell=hypot(d_vec[0],d_vec[1]) #distance between wall node and cell node (micrometers)
            d_vec/=dist_cell
            lat_dists[wid]+=dist_cell
            if i==0:
                wid0=wid
                wid1=wid
            else: #This algorithm only works if walls are ordered anti-clockwise around the cell center
                wid2=wid #new wall id
                #Find junction closest to wid1
                dist1=hypot(position[wid1][0]-position_junctions[wid2][0],position[wid1][1]-position_junctions[wid2][1])
                dist2=hypot(position[wid1][0]-position_junctions[wid2][2],position[wid1][1]-position_junctions[wid2][3])
                if dist1<dist2:
                    j=0
                else:
                    j=2
                cellarea[cellnumber1] += (position[wid1][0]+position_junctions[wid2][0+j])*(position[wid1][1]-position_junctions[wid2][1+j]) #Cell area loop (micron^2)
                cellarea[cellnumber1] += (position_junctions[wid2][0+j]+position[wid2][0])*(position_junctions[wid2][1+j]-position[wid2][1]) #Cell area loop (micron^2)
                wid1=wid2
            Nmb+=1
            G.add_edge(NwallsJun + cellnumber1, wid, path='membrane', length=lengths[wid], dist=dist_cell, d_vec=d_vec) #, height=height #Adding all cell to wall connections (edges) #kaqp=kaqp, kw=kw, kmb=kmb, 
            cellperimeter[cellnumber1]+=lengths[wid]
            i+=1
        dist1=hypot(position[wid1][0]-position_junctions[wid0][0],position[wid1][1]-position_junctions[wid0][1])
        dist2=hypot(position[wid1][0]-position_junctions[wid0][2],position[wid1][1]-position_junctions[wid0][3])
        if dist1<dist2:
            j=0
        else:
            j=2
        cellarea[cellnumber1] += (position[wid1][0]+position_junctions[wid0][0+j])*(position[wid1][1]-position_junctions[wid0][1+j]) #Back to the first node
        cellarea[cellnumber1] += (position_junctions[wid0][0+j]+position[wid0][0])*(position_junctions[wid0][1+j]-position[wid0][1]) #Back to the first node
        cellarea[cellnumber1] /= -2.0

    Cell_connec=-ones((Ncells,Cell_connec_max),dtype=int) #Connected cells for further ranking
    nCell_connec=zeros((Ncells,1),dtype=int) #Quantity of cell to cell connections
    for i in range(0, len(Walls_loop)): #Loop on walls, by cell - wall association, hence a wall can be repeated if associated to two cells. Parent structure: Cell/Walls/Wall
                r1 = Walls_loop[i] #Points to the current wall
                cellid1 = r1.getparent().getparent().get("id") #Cell1 ID number
                id1 = r1.get("id") #Wall1 ID number
                for j in range(i + 1, len(Walls_loop) ): #Loop on cell-wall associations that are further down in the list
                    r2 = Walls_loop[j] #Points to the further down wall in the list of cell-wall associations
                    cellid2 = r2.getparent().getparent().get("id") #Cell2 ID number
                    id2 = r2.get("id") #Wall2 ID number
                    if id1 == id2: #If walls 1 and 2 are the same, then cells 1 and 2 are connected by plasmodesmata
                        d_vec=array([position[NwallsJun+int(cellid2)][0]-position[NwallsJun+int(cellid1)][0],position[NwallsJun+int(cellid2)][1]-position[NwallsJun+int(cellid1)][1]])
                        dist_cell=hypot(d_vec[0],d_vec[1]) #distance between wall node and cell node (micrometers)
                        d_vec/=dist_cell
                        G.add_edge(NwallsJun + int(cellid1), NwallsJun + int(cellid2), path='plasmodesmata', length=lengths[int(id1)], d_vec=d_vec) #, height=height #Adding all cell to cell connections (edges) #kpl=kpl, 
                        Cell_connec[int(cellid1)][nCell_connec[int(cellid1)]]=int(cellid2)
                        nCell_connec[int(cellid1)]+=1
                        Cell_connec[int(cellid2)][nCell_connec[int(cellid2)]]=int(cellid1)
                        nCell_connec[int(cellid2)]+=1

    jid=0
    for Junction, Walls in Junction2Wall.items(): #Loop on junctions between walls
        for wid in Walls: #Walls is the list of cell walls ID meeting at the junction pos
            d_vec=array([position[wid][0]-position[jid+Nwalls][0],position[wid][1]-position[jid+Nwalls][1]])
            dist_wall=hypot(d_vec[0],d_vec[1]) #distance between wall node and cell node (micrometers)
            d_vec/=dist_wall #As compared to lat_dist, dist_wall underestimates the actual path length between the wall and the junction. dist_wall is rather a straight distance.
            G.add_edge(jid+Nwalls, int(wid), path='wall', length=lengths[int(wid)]/2, lat_dist=lat_dists[int(wid)][0], d_vec=d_vec, dist_wall=dist_wall) #Adding junction to wall connections (edges)
        jid+=1

    return d_vec, dist_wall, G, Cell_connec, nCell_connec, Nmb, cellarea, cellperimeter
# =======================
def compute_AQP_axial_distribution(G,Ncells, Cell2Wall_loop, position, NwallsJun,
                                   x_grav, y_grav, 
                                   Ncellperimeters,
                                   nCell_connec, Cell_connec, InterCid,
                                   InterC_perim_search, 
                                   listsieve, listxyl):
    
    # =============================================================================================
    # TO DO : for now InterC_perim1 is not taken into account, because InterC_perim_search is always 0
    # =============================================================================================
    
    Cell_rank=zeros((Ncells,1)) #Ranking of cells (1=Exodermis, 2=Epidermis, 3=Endodermis, 4*=Cortex, 5*=Stele, 11=Phloem sieve tube, 12=Companion cell, 13=Xylem, 16=Pericycle), stars are replaced by the ranking within cortical cells and stele cells
    Layer_dist=zeros((62,1)) #Average cell layers distances from center of gravity, by cells ranking 
    nLayer=zeros((62,1)) #Total number of cells in each rank (indices follow ranking numbers)
    xyl_dist=[] #List of distances between xylem and cross-section centre
    #angle_dist_endo_grav=array([-4,0]) #array of distances and angles between endo cells and grav. Initializing the array with values that will eventualy be deleted
    #angle_dist_exo_grav=array([-4,0]) #array of distances and angles between exo cells and grav. Initializing the array with values that will eventualy be deleted
    for w in Cell2Wall_loop: #Loop on cells. Cell2Wall_loop contains cell wall groups info (one group by cell)
        cellnumber1 = int(w.getparent().get("id")) #Cell ID number
        celltype=G.node[NwallsJun + cellnumber1]['cgroup'] #Cell type
        if celltype==19 or celltype==20: #Proto- and Meta-xylem in new Cellset version
            celltype=13
        elif celltype==21: #Xylem pole pericycle in new Cellset version
            celltype=16
        elif celltype==23: #Phloem in new Cellset version
            celltype=11
        elif celltype==26 or celltype==24: #Companion cell in new Cellset version
            celltype=12
        Cell_rank[cellnumber1]=celltype #Later on, cell types 4 and 5 will be updated to account for their ranking within the cortex / stele
        x_cell=position[NwallsJun + cellnumber1][0] #Cell position (micrometers)
        y_cell=position[NwallsJun + cellnumber1][1]
        dist=hypot(x_cell-x_grav,y_cell-y_grav) #(micrometers)
        Layer_dist[celltype]+=dist
        nLayer[celltype]+=1
        if celltype==13:
            xyl_dist.append([dist])
    if len(xyl_dist)>0:
        xyl80_dist=percentile(xyl_dist, 80)
    else:
        xyl80_dist=nan

    if nLayer[16]==0: #If there is no labelled pericycle
        stele_connec_rank=3 #Endodermis connected to stele cells
    else:
        stele_connec_rank=16 #Pericycle connected to stele cells
    if nLayer[1]==0: #If there is no labelled exodermis
        outercortex_connec_rank=2 #Cortex connected to epidermis cells
    else:
        outercortex_connec_rank=1 #Cortex connected to exodermis cells
    if InterC_perim_search==1:
        rank_cellperimeters_in=linspace(nan,nan,Ncellperimeters)
        rank_cellperimeters_out=linspace(nan,nan,Ncellperimeters)
    listprotosieve=[]
    mincid=99999
    Layer_dist[16]=0
    nLayer[16]=0
    for w in Cell2Wall_loop: #Loop on cells. Cell2Wall_loop contains cell wall groups info (one group by cell)
        cellnumber1 = int(w.getparent().get("id")) #Cell ID number #celltype=G.node[NwallsJun + cellnumber1]['cgroup']
        celltype=Cell_rank[cellnumber1] #Cell types 4 and 5 updated to account for their ranking within the cortex / stele
        if celltype==16: #Pericycle
            temp=Cell_rank[Cell_connec[cellnumber1][0:nCell_connec[cellnumber1][0]]]
            if any(temp==5) or any(temp==11) or any(temp==12) or any(temp==13) or any(temp==50): #Cell to cell connection with endodermis
                Cell_rank[cellnumber1]=16 #pericycle ranks now span 16 to 18 instead of 16
                x_cell=position[NwallsJun + cellnumber1][0] #Cell position (micrometers)
                y_cell=position[NwallsJun + cellnumber1][1]
                dist=hypot(x_cell-x_grav,y_cell-y_grav) #(micrometers)
                Layer_dist[16]+=dist
                nLayer[16]+=1
            elif any(temp==3): #Cell to cell connection with endodermis
                Cell_rank[cellnumber1]=18 #pericycle ranks now span 16 to 18 instead of 16
                x_cell=position[NwallsJun + cellnumber1][0] #Cell position (micrometers)
                y_cell=position[NwallsJun + cellnumber1][1]
                dist=hypot(x_cell-x_grav,y_cell-y_grav) #(micrometers)
                Layer_dist[18]+=dist
                nLayer[18]+=1
            elif all(np.logical_or(np.logical_or(temp==16,temp==17),temp==18)):
                Cell_rank[cellnumber1]=17 #pericycle ranks now span 16 to 18 instead of 16
                x_cell=position[NwallsJun + cellnumber1][0] #Cell position (micrometers)
                y_cell=position[NwallsJun + cellnumber1][1]
                dist=hypot(x_cell-x_grav,y_cell-y_grav) #(micrometers)
                Layer_dist[17]+=dist
                nLayer[17]+=1
            else:
                error('Pericycle cell falls in no category')
        elif celltype==4: #Cortex
            if any(Cell_rank[Cell_connec[cellnumber1][0:nCell_connec[cellnumber1][0]]]==3): #Cell to cell connection with endodermis
                Cell_rank[cellnumber1]=25 #Cortex ranks now span 25 to 49 instead of 40 to 49
                x_cell=position[NwallsJun + cellnumber1][0] #Cell position (micrometers)
                y_cell=position[NwallsJun + cellnumber1][1]
                dist=hypot(x_cell-x_grav,y_cell-y_grav) #(micrometers)
                Layer_dist[25]+=dist
                nLayer[25]+=1
                if InterC_perim_search==1: # NB : for now InterC_perim_search is 0 ?
                    rank_cellperimeters_in[int(nLayer[25]-1)]=cellperimeter[cellnumber1]
                    if cellperimeter[cellnumber1]<InterC_perim1:
                        InterCid.append(cellnumber1) #Cell id starting at 0
            elif any(Cell_rank[Cell_connec[cellnumber1][0:nCell_connec[cellnumber1][0]]]==outercortex_connec_rank): #Cell to cell connection with exodermis
                Cell_rank[cellnumber1]=49
                x_cell=position[NwallsJun + cellnumber1][0] #Cell position (micrometers)
                y_cell=position[NwallsJun + cellnumber1][1]
                dist=hypot(x_cell-x_grav,y_cell-y_grav) #(micrometers)
                Layer_dist[49]+=dist
                nLayer[49]+=1
                if InterC_perim_search==1:
                    rank_cellperimeters_out[int(nLayer[49]-1)]=cellperimeter[cellnumber1]
                    if cellperimeter[cellnumber1]<InterC_perim5:
                        InterCid.append(cellnumber1) #Cell id starting at 0
        elif celltype==5 or celltype==11 or celltype==12 or celltype==13: #Stele
            if any(Cell_rank[Cell_connec[cellnumber1][0:nCell_connec[cellnumber1][0]]]==stele_connec_rank): #Cell to cell connection with pericycle
                Cell_rank[cellnumber1]=50
                x_cell=position[NwallsJun + cellnumber1][0] #Cell position (micrometers)
                y_cell=position[NwallsJun + cellnumber1][1]
                dist=hypot(x_cell-x_grav,y_cell-y_grav) #(micrometers)
                Layer_dist[50]+=dist
                nLayer[50]+=1
                if G.node[NwallsJun + cellnumber1]['cgroup']==11 or G.node[NwallsJun + cellnumber1]['cgroup']==23:
                    listprotosieve.append(NwallsJun + cellnumber1)
    Nsieve=len(listsieve)
    Nprotosieve=len(listprotosieve)

    if InterC_perim_search==1:
        cortex_cellperimeters_in=rank_cellperimeters_in #Inner part of the cortex (close to endodermis)
        cortex_cellperimeters_out=rank_cellperimeters_out #Outer part of cortex
    for i in range(12):
        if InterC_perim_search==1:
            rank_cellperimeters_in=linspace(nan,nan,Ncellperimeters)
            rank_cellperimeters_out=linspace(nan,nan,Ncellperimeters)
        for w in Cell2Wall_loop: #Loop on cells. Cell2Wall_loop contains cell wall groups info (one group by cell)
            cellnumber1 = int(w.getparent().get("id")) #Cell ID number
            celltype=Cell_rank[cellnumber1] #Cell types 4 and 5 updated to account for their ranking within the cortex / stele
            if celltype==4 and i<12: #Cortex # if i<12: #Within 12 layers of cortical sides
                temp=Cell_connec[cellnumber1][0:nCell_connec[cellnumber1][0]]
                AloneC=True
                for i_connec in range(len(temp)):
                    if not temp[i_connec] in InterCid:
                        AloneC=False #Isolated cell only in contact with intercellular spaces
                done=False
                for i_connec in range(len(temp)):
                    if Cell_rank[temp[i_connec]]==(25+i) and not done: #Cell to cell connection with endodermis
                        if (not temp[i_connec] in InterCid) or AloneC:    
                            Cell_rank[cellnumber1]=26+i
                            x_cell=position[NwallsJun + cellnumber1][0] #Cell position (micrometers)
                            y_cell=position[NwallsJun + cellnumber1][1]
                            dist=hypot(x_cell-x_grav,y_cell-y_grav) #(micrometers)
                            Layer_dist[26+i]+=dist
                            nLayer[26+i]+=1
                            done=True
                            if InterC_perim_search==1:
                                rank_cellperimeters_in[int(nLayer[26+i]-1)]=cellperimeter[cellnumber1]
                                if i==0 and cellperimeter[cellnumber1]<InterC_perim2:
                                    InterCid.append(cellnumber1)
                                elif i==1 and cellperimeter[cellnumber1]<InterC_perim3:
                                    InterCid.append(cellnumber1)
                                elif i==2 and cellperimeter[cellnumber1]<InterC_perim4:
                                    InterCid.append(cellnumber1)
                                elif i>2 and cellperimeter[cellnumber1]<InterC_perim5:
                                    InterCid.append(cellnumber1)
                    elif Cell_rank[temp[i_connec]]==(49-i) and not done: #Cell to cell connection with exodermis
                        if (not temp[i_connec] in InterCid) or AloneC:
                            Cell_rank[cellnumber1]=48-i
                            x_cell=position[NwallsJun + cellnumber1][0] #Cell position (micrometers)
                            y_cell=position[NwallsJun + cellnumber1][1]
                            dist=hypot(x_cell-x_grav,y_cell-y_grav) #(micrometers)
                            Layer_dist[48-i]+=dist
                            nLayer[48-i]+=1
                            done=True
                            if InterC_perim_search==1:
                                rank_cellperimeters_out[int(nLayer[48-i]-1)]=cellperimeter[cellnumber1]
                                if cellperimeter[cellnumber1]<InterC_perim5:
                                    InterCid.append(cellnumber1)
            elif celltype==5 or celltype==11 or celltype==12 or celltype==13: #Stele
                if i<10:
                    temp=Cell_connec[cellnumber1][0:nCell_connec[cellnumber1][0]]
                    done=False
                    for i_connec in range(len(temp)):
                        if Cell_rank[temp[i_connec]]==(50+i) and not done:
                            if (not temp[i_connec]+NwallsJun in listxyl):
                                #if any(Cell_rank[temp]==(50+i)): #Cell to cell connection with pericycle
                                Cell_rank[cellnumber1]=51+i
                                x_cell=position[NwallsJun + cellnumber1][0] #Cell position (micrometers)
                                y_cell=position[NwallsJun + cellnumber1][1]
                                dist=hypot(x_cell-x_grav,y_cell-y_grav) #(micrometers)
                                Layer_dist[51+i]+=dist
                                nLayer[51+i]+=1
                                done=True
                else: #No more than 11 stele cell layers
                    Cell_rank[cellnumber1]=61
                    x_cell=position[NwallsJun + cellnumber1][0] #Cell position (micrometers)
                    y_cell=position[NwallsJun + cellnumber1][1]
                    dist=hypot(x_cell-x_grav,y_cell-y_grav) #(micrometers)
                    Layer_dist[61]+=dist
                    nLayer[61]+=1
        if i<12:
            if InterC_perim_search==1:
                cortex_cellperimeters_in=vstack((cortex_cellperimeters_in,rank_cellperimeters_in))
                cortex_cellperimeters_out=vstack((rank_cellperimeters_out,cortex_cellperimeters_out))
    if InterC_perim_search==1:
        cortex_cellperimeters=vstack((cortex_cellperimeters_in,cortex_cellperimeters_out))
    #InterCid=InterCid[1:]
        
    return Cell_rank, Layer_dist, nLayer, xyl_dist, Layer_dist, nLayer, InterCid, Nsieve, Nprotosieve, listprotosieve, outercortex_connec_rank, xyl80_dist
# =======================
def compute_cell_surface(G, NwallsJun, InterCid, outercortex_connec_rank):
    # Calculates cell surfaces and tissue interfaces
    indice=nx.get_node_attributes(G,'indice') #Node indices (walls, junctions and cells)
    PPP=list()
    Length_outer_cortex_tot=0.0 #Total cross-section membrane length at the interface between exodermis and cortex
    Length_cortex_cortex_tot=0.0 #Total cross-section membrane length at the interface between cortex and cortex
    Length_cortex_endo_tot=0.0 #Total cross-section membrane length at the interface between cortex and endodermis
    Length_outer_cortex_nospace=0.0 #Cross-section membrane length at the interface between exodermis and cortex not including interfaces with intercellular spaces
    Length_cortex_cortex_nospace=0.0 #Cross-section membrane length at the interface between exodermis and cortex not including interfaces with intercellular spaces
    Length_cortex_endo_nospace=0.0 #Cross-section membrane length at the interface between exodermis and cortex not including interfaces with intercellular spaces

    for node, edges in G.adjacency_iter() :
        i=indice[node] #Node ID number
        if i>=NwallsJun: #Cell
            if G.node[i]['cgroup']==16 or G.node[i]['cgroup']==21:
                for neighbour, eattr in edges.items(): #Loop on connections (edges)
                    if eattr['path'] == "plasmodesmata" and (G.node[indice[neighbour]]['cgroup']==11 or G.node[indice[neighbour]]['cgroup']==23): #Plasmodesmata connection  #eattr is the edge attribute (i.e. connection type)
                        PPP.append(i-NwallsJun)
            elif G.node[i]['cgroup']==outercortex_connec_rank or G.node[i]['cgroup']==4 or G.node[i]['cgroup']==3: #exodermis or cortex or endodermis (or epidermis if there is no exodermis)
                if i-NwallsJun not in InterCid: #The loop focuses on exo, cortex and endodermis cells that are not intercellular spaces
                    for neighbour, eattr in edges.items(): #Loop on connections (edges)
                        if eattr['path'] == "plasmodesmata": #Plasmodesmata connection  #eattr is the edge attribute (i.e. connection type)
                            j = (indice[neighbour]) #neighbouring node number
                            l_membrane=eattr['length']
                            if (G.node[i]['cgroup']==outercortex_connec_rank and G.node[j]['cgroup']==4) or (G.node[j]['cgroup']==outercortex_connec_rank and G.node[i]['cgroup']==4):#Exodermis to cortex cell or vice versa (epidermis if no exodermis exists)
                                Length_outer_cortex_tot+=l_membrane
                                if j-NwallsJun not in InterCid:
                                    Length_outer_cortex_nospace+=l_membrane
                            elif (G.node[i]['cgroup']==4 and G.node[j]['cgroup']==4):#Cortex to cortex cell
                                Length_cortex_cortex_tot+=l_membrane
                                if j-NwallsJun not in InterCid:
                                    Length_cortex_cortex_nospace+=l_membrane
                            elif (G.node[i]['cgroup']==3 and G.node[j]['cgroup']==4) or (G.node[j]['cgroup']==3 and G.node[i]['cgroup']==4):#Cortex to endodermis cell or vice versa
                                Length_cortex_endo_tot+=l_membrane
                                if j-NwallsJun not in InterCid:
                                    Length_cortex_endo_nospace+=l_membrane

    return(G,
           indice, 
           PPP, 
           Length_outer_cortex_tot, 
           Length_cortex_cortex_tot,
           Length_cortex_endo_tot,
           Length_outer_cortex_nospace,
           Length_cortex_cortex_nospace,
           Length_cortex_endo_nospace)
