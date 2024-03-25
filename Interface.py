# -*- coding: utf-8 -*-

#TOCONFIRM: rhs_C -> xylemconcentration dist & prox -> sym_cc_flows prox & dist should be ok

#TODO: Fix mass balance estimation
#TODO: Check Role of Defheight vs height in Fpli and Flowdensity through plasmodesmata
#TODO: Double check row for passage cells
#TODO: Read initial condition from xml file
#TODO: Fix 3D_cc* files for simulation initial conditions (currently cc from only one layer saved)
#TODO: cell types 23 and 26
#TODO: Import of multilayered initial data of solute cc
#TODO: turn wallflowdensity rows into full arrays?
#TODO: longitudinal carriers
#TODO: Loop radial water velocity & solute
#TODO: Check/remove? PileUp option

#Directory
dir = './'

#Project
Project ='Projects/granar/' 

#Inputs
inputs='in/'
Gen='General.xml'
Geom='Geometry.xml'
Hydr='Hydraulics.xml'
BC='BC.xml'#'Maize_BC_kr.xml'
Horm='Hormones_Carriers.xml'
Cell_connec_max=50
Ncellperimeters=100
V_modifier=1.0
test_mass_balance=0

maxCell2ThickWalls=101
#DF_axial_factor=1.0
#K_axial_factor=1.0
matrix_analysis=0

# Run MECHA
print("Launching MECHA")

# import subprocess
# subprocess.run(['python', "MECHA_main.py"])
import MECHA_main as Main
from Functions import *

# Import_Data()
Main.MECHA()