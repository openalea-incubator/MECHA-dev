
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