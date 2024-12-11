# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 11:32:29 2024

Test for HU and DVS
@author: alegn
"""
import numpy as np

# - Heat Units
tsim = np.linspace(0.0, 120.0, 120+1) # [d]
dt = 1 # [d]
T = np.array([tsim, np.full((tsim.size,), 25)]).T
Tavg = T[:,1]

HU = []
DVS = 0
Tmin = 15
Topt = 25
Tmax = 40

cuml_HU = 0
for t in Tavg:    
    if Tmin < t <= Topt:
        HUi = t - Tmin
    elif Topt <t <Tmax:
        HUi = Topt - ((t - Topt)*(Topt-Tmin)/(Tmax-Topt))
    else:
        HUi = 0
    cuml_HU += HUi
    HU.append(cuml_HU)

print("HUvalues: ", HU)

# - Developmental stage (DVS)

cuml_DVS =0
for hu,t in zip(HU,Tavg):
    if hu > 0:
        DVSi = DVS*hu
    else:
        DVSi = 0
    cuml_DVS += DVSi
    DVS=cuml_DVS  

print("DVSvalues: ", DVS)
