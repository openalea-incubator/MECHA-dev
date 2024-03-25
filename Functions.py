
import xml.etree.ElementTree as ET
import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt

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