# -*- coding: utf-8 -*-

#Directory
dir = './'

#Project
Project ='Projects/granar/' 

#Inputs
inputs='in/'
Gen='General.xml'
Geom='Geometry.xml'
Hydr='Hydraulics.xml'
BC='BC.xml'
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

import main
main.MECHA_run()