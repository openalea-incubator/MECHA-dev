import numpy as np 
from numpy import genfromtxt #Load data from a text file, with missing values handled as specified.
from numpy.random import *  # for random sampling
from scipy import sparse
from scipy.sparse import lil_matrix,csr_matrix,spdiags,identity
from scipy.sparse.linalg import spsolve
import scipy.linalg as slin #Linear algebra functions
import math
# import pylab #Found in the package pyqt, also needs to be installed for proper use of matlab-python connectivity
from pylab import *  # for plotting
from decimal import Decimal
import networkx as nx 
from lxml import etree
import sys, os 

from Interface import dir, Project, inputs, Gen, Geom, Hydr, BC, Horm, Cell_connec_max, Ncellperimeters, V_modifier, test_mass_balance, maxCell2ThickWalls, matrix_analysis

def Import_Data() : 

    print(Project)
    print(inputs)
    print(Project)
    print(inputs)
    print(Gen)
    print(Geom)
    print(Hydr)
    print(BC)
    print(Horm)
    print(Cell_connec_max)
    print(Ncellperimeters)
    print(V_modifier)
    print(test_mass_balance)
    print(maxCell2ThickWalls)
    print(matrix_analysis)

    #Precision
    dp = np.dtype((np.float64))

    #Import General data
    # print('Importing geometrical data')
    t0 = time.perf_counter()
    OS=etree.parse(dir + Project + inputs + Gen).getroot().xpath('OS')[0].get("value")
    Output_path=etree.parse(dir + Project + inputs + Gen).getroot().xpath('Output')[0].get("path")

    Paraview=int(etree.parse(dir + Project + inputs + Gen).getroot().xpath('Paraview')[0].get("value"))
    ParaviewWF=int(etree.parse(dir + Project + inputs + Gen).getroot().xpath('Paraview')[0].get("WallFlux"))
    ParaviewMF=int(etree.parse(dir + Project + inputs + Gen).getroot().xpath('Paraview')[0].get("MembraneFlux"))
    ParaviewPF=int(etree.parse(dir + Project + inputs + Gen).getroot().xpath('Paraview')[0].get("PlasmodesmataFlux"))
    ParaviewWP=int(etree.parse(dir + Project + inputs + Gen).getroot().xpath('Paraview')[0].get("WallPot"))
    ParaviewCP=int(etree.parse(dir + Project + inputs + Gen).getroot().xpath('Paraview')[0].get("CellPot"))
    ParTrack=int(etree.parse(dir + Project + inputs + Gen).getroot().xpath('ParTrack')[0].get("value"))
    Sym_Contagion=int(etree.parse(dir + Project + inputs + Gen).getroot().xpath('Sym_Contagion')[0].get("value"))
    Apo_Contagion=int(etree.parse(dir + Project + inputs + Gen).getroot().xpath('Apo_Contagion')[0].get("value"))
    color_threshold=float(etree.parse(dir + Project + inputs + Gen).getroot().xpath('color_threshold')[0].get("value"))
    thickness_disp=float(etree.parse(dir + Project + inputs + Gen).getroot().xpath('thickness_disp')[0].get("value"))
    thicknessJunction_disp=float(etree.parse(dir + Project + inputs + Gen).getroot().xpath('thicknessJunction_disp')[0].get("value"))
    radiusPlasmodesm_disp=float(etree.parse(dir + Project + inputs + Gen).getroot().xpath('radiusPlasmodesm_disp')[0].get("value"))
    UniXwalls=int(etree.parse(dir + Project + inputs + Gen).getroot().xpath('UniXwalls')[0].get("value"))
    sparseM=int(etree.parse(dir + Project + inputs + Gen).getroot().xpath('sparse')[0].get("value"))

    #Import Geometrical data
    Plant=etree.parse(dir + Project + inputs + Geom).getroot().xpath('Plant')[0].get("value")
    path_geom=etree.parse(dir + Project + inputs + Geom).getroot().xpath('path')[0].get("value")
    im_scale=float(etree.parse(dir + Project + inputs + Geom).getroot().xpath('im_scale')[0].get("value"))
    Maturityrange=etree.parse(dir + Project + inputs + Geom).getroot().xpath('Maturityrange/Maturity')
    Printrange=etree.parse(dir + Project + inputs + Geom).getroot().xpath('Printrange/Print_layer')
    Xwalls=float(etree.parse(dir + Project + inputs + Geom).getroot().xpath('Xwalls')[0].get("value")) #Transverse walls or not
    PileUp=int(etree.parse(dir + Project + inputs + Geom).getroot().xpath('PileUp')[0].get("value"))
    passage_cell_range=etree.parse(dir + Project + inputs + Geom).getroot().xpath('passage_cell_range/passage_cell')
    aerenchyma_range=etree.parse(dir + Project + inputs + Geom).getroot().xpath('aerenchyma_range/aerenchyma')
    passage_cell_ID=[]
    for passage_cell in passage_cell_range:
        passage_cell_ID.append(int(passage_cell.get("id")))
    PPP=list()
    InterCid=list() #Aerenchyma is classified as intercellular space
    for aerenchyma in aerenchyma_range:
        if not int(aerenchyma.get("id"))>9E5 and not int(aerenchyma.get("id"))<0:
            InterCid.append(int(aerenchyma.get("id"))) #Cell id starting at 0
        else:
            print('InterCid #'+str(int(aerenchyma.get("id")))+' excluded')
    InterC_perim_search=int(etree.parse(dir + Project + inputs + Geom).getroot().xpath('InterC_perim_search')[0].get("value"))
    if InterC_perim_search==1:
        InterC_perim1=float(etree.parse(dir + Project + inputs + Geom).getroot().xpath('InterC_perim1')[0].get("value"))
        InterC_perim2=float(etree.parse(dir + Project + inputs + Geom).getroot().xpath('InterC_perim2')[0].get("value"))
        InterC_perim3=float(etree.parse(dir + Project + inputs + Geom).getroot().xpath('InterC_perim3')[0].get("value"))
        InterC_perim4=float(etree.parse(dir + Project + inputs + Geom).getroot().xpath('InterC_perim4')[0].get("value"))
        InterC_perim5=float(etree.parse(dir + Project + inputs + Geom).getroot().xpath('InterC_perim5')[0].get("value"))
    kInterC=float(etree.parse(dir + Project + inputs + Geom).getroot().xpath('kInterC')[0].get("value"))
    cell_per_layer=zeros((2,1))
    cell_per_layer[0][0]=float(etree.parse(dir + Project + inputs + Geom).getroot().xpath('cell_per_layer')[0].get("cortex"))
    cell_per_layer[1][0]=float(etree.parse(dir + Project + inputs + Geom).getroot().xpath('cell_per_layer')[0].get("stele"))
    thickness=float(etree.parse(dir + Project + inputs + Geom).getroot().xpath('thickness')[0].get("value")) #micron
    PD_section=float(etree.parse(dir + Project + inputs + Geom).getroot().xpath('PD_section')[0].get("value")) #micron^2
    Xylem_pieces=False
    if float(etree.parse(dir + Project + inputs + Geom).getroot().xpath('Xylem_pieces')[0].get("flag"))==1:
        Xylem_pieces=True
    t1 = time.perf_counter()
    # print(t1-t0, "seconds process time")

    #Import hormone properties
    Degrad1=float(etree.parse(dir + Project + inputs + Horm).getroot().xpath('Hormone_movement/Degradation_constant_H1')[0].get("value")) #Hormone 1 degradation constant (mol degraded / mol-day)
    K_MB1=float(etree.parse(dir + Project + inputs + Horm).getroot().xpath('Hormone_movement/MB_H1')[0].get("Partition_coef")) #(-)
    dx_MB1=float(etree.parse(dir + Project + inputs + Horm).getroot().xpath('Hormone_movement/MB_H1')[0].get("Thickness")) #(micron)
    Diff_MB1=float(etree.parse(dir + Project + inputs + Horm).getroot().xpath('Hormone_movement/MB_H1')[0].get("Diffusivity")) #(cm^2/day)
    Diff_PD1=float(etree.parse(dir + Project + inputs + Horm).getroot().xpath('Hormone_movement/Diffusivity_PD_H1')[0].get("value")) #Hormone 1 diffusivity constant (cm^2/day)
    Diff_PW1=float(etree.parse(dir + Project + inputs + Horm).getroot().xpath('Hormone_movement/Diffusivity_PW_H1')[0].get("value")) #Hormone 1 diffusivity constant (cm^2/day)
    Diff_X1=float(etree.parse(dir + Project + inputs + Horm).getroot().xpath('Hormone_movement/Diffusivity_X_H1')[0].get("value")) #Hormone 1 diffusivity constant (cm^2/day)
    D2O1=int(etree.parse(dir + Project + inputs + Horm).getroot().xpath('Hormone_movement/H1_D2O')[0].get("flag")) #Hormone 1 diffusivity constant (cm^2/day)
    Active_transport_range=etree.parse(dir + Project + inputs + Horm).getroot().xpath('Hormone_active_transport/carrier_range/carrier')
    Sym_source_ini_range=etree.parse(dir + Project + inputs + Horm).getroot().xpath('Sym_Contagion/source_range/Steady-state/source')
    Sym_source_transi_range=etree.parse(dir + Project + inputs + Horm).getroot().xpath('Sym_Contagion/source_range/Transient/source')
    contact_range=etree.parse(dir + Project + inputs + Horm).getroot().xpath('Contactrange/Contact')
    Contact=[]
    for contact in contact_range:
        Contact.append(int(contact.get("id")))

    #Import cellset data
    import xml.etree.ElementTree as ET
    #import xml.dom.minidom as DOM
    #xml = xml.dom.minidom.parse(xml_fname) # or xml.dom.minidom.parseString(xml_string)
    #pretty_xml_as_string = xml.toprettyxml()
    tree = etree.parse(dir + 'cellsetdata/' + path_geom) #Maize_Charles\\Maize_pip_cross4.xml') # #Parse literally decrypts the tree element data         SteleOK_high.xml
    rootelt = tree.getroot()
    Cell2Wall_loop = rootelt.xpath('cells/cell/walls') #Cell2Wall_loop contains cell wall groups info (one group by cell), searched by xpath ("Smart" element identifier)