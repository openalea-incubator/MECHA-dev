# -*- coding: utf-8 -*-

#Directory
dir = './'

#Project
Project ='Projects/test/' 

#Inputs
inputs='in/'
Gen='General.xml'
Geom='Geometry.xml'
Hydr='Hydraulics.xml'
BC='BC.xml'
Horm='Hormones_Carriers.xml'

# Run MECHA
print("Launching MECHA")

from mecha.main import *
MECHA_run(dir, Project, inputs, Gen, Geom, Hydr, BC, Horm)