
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

class Input:
    def __init__(self, filename):
        self.tree = ET.parse(filename)
        self.root = self.tree.getroot()
        
    def get_parameter(self, from_ = "", name ="", attribute = "", type ="null"):
        elem = self.root.find("./{}/{}".format(from_, name))
        if(type == "int"):
            return int(elem.get(attribute))
        elif (type == "float"):
            return float(elem.get(attribute))
        else:
            return elem.get(attribute)

    
    def get_all(self, from_, name, attribute):
        elems = self.root.findall("./{}/{}".format(from_, name))
        return [elem.get(attribute) for elem in elems]
    
    def change_parameter(self, from_ = "", name ="", attribute= "", value= 0.0):
        elem = self.root.find("./{}/{}".format(from_, name))
        elem.set(attribute, str(value))
    
    def write_xml(self, file_name):
        self.tree.write(file_name)

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
        self.STFlayer_minus = row_elm(f, "Standard Transmembrane release Fractions", "end")
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


    def get_elm(strings, pattern):
        x = [pattern in i for i in strings]
        res = [i for i, val in enumerate(x) if val]
        return res

    def int_elm(f, pattern):
        idx = get_elm(f, pattern)
        tmp = str([f[i] for i in idx])
        temp = int(re.findall(r'\d+',tmp)[0])
        return temp

    def float_elm(f, pattern):
        idx = get_elm(f, pattern)
        tmp = str([f[i] for i in idx])
        temp = float(re.findall(r'[\d]*[.][\d]+',tmp)[0])
        return temp

    def array_elm(f, pattern):
        idx = get_elm(f, pattern)
        tmp = str([f[i+1] for i in idx])
        temp = [int(s) for s in re.findall(r'\b\d+\b', tmp)]
        return temp

    def row_elm(f, pattern1, pattern2):
        start = int(get_elm(f, pattern1)[0])+1
        if pattern2 == "end":
            end = len(f)-1
        else:
            end = int(get_elm(f, pattern2)[0])-1
        y = []
        for i in range(start,end,1):
            y.append(float(re.findall(r'[\d]*[.][\d]+',f[i])[0]))
        return y

def plot_partition(file):
    Hydr = Macro_hydro_visu(file)
    
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