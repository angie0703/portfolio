# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:53:33 2024

@author: alegn
"""

import numpy as np
import matplotlib.pyplot as plt
from models.rice import Rice
import pandas as pd

plt.style.use('ggplot')

# Simulation time
t = 120
tsim = np.linspace(0.0, t, t+1) # [d]
dt = 1 # [d]

# Initial conditions
# TODO: define sensible values for the initial conditions
n_rice = 127980 #number of plants in 6000 m2 area
x0 = {'Mrt':0.005,    #[kg DM ha-1 d-1] Dry biomass of root
      'Mst':0.003,    #[kg DM ha-1 d-1] Dry biomass of stems
      'Mlv': 0.002,   #[kg DM ha-1 d-1] Dry biomass of leaves
      'Mpa': 0.0,   #[kg DM ha-1 d-1] Dry biomass of panicles
      'Mgr': 0.0,   #[kg DM ha-1 d-1] Dry biomass of grains
      'HU': 0.0
      }

 

# Model parameters, as provided by Mohtar et al. (1997)
# TODO: define the parameter values in the dictionary p
p = {
     # 'DVSi' : 0.001,
     # 'DVRJ': 0.0013,
     # 'DVRI': 0.001125,
     # 'DVRP': 0.001275,
     # 'DVRR': 0.003,
     'DVSi' : 0.0008753,
     'DVRJ': 0.0008753,
     'DVRI': 0.0007576,
     'DVRP': 0.0007787,
     'DVRR': 0.0015390,
     'Tmax': 40,
     'Tmin': 15,
     'Topt': 33,
     'k': 0.4,
     'mc_rt': 0.01,
     'mc_st': 0.015,
     'mc_lv': 0.02,
     'mc_pa': 0.003,
     'N_lv': 0,
     'Rec_N': 0.5,
     'k_lv_N': 0,
     'k_pa_maxN': 0.0175,
     'M_upN': 8,
     'cr_lv': 1.326,
     'cr_st': 1.326,
     'cr_pa': 1.462,
     'cr_rt': 1.326,
     'IgN': 0.5      
     }

# Disturbances
# PAR [J m-2 d-1], env. temperature [°C], and CO2
# TODO: Specify the corresponding dates to read weather data (see csv file).
# Dictionary of disturbances (2D arrays, with col 1 for time, and col 2 for d)
#disturbances
data_weather = 'C:/Users/alegn/Documents/WUR/Thesis/rfmodel/data/Weather/DIY_202210_202309_Daily.csv'
weather = pd.read_csv(data_weather, header=1, sep=';')
Tmax = weather.iloc[:,2].values #[°C] Maximum daily temperature
Tmin = weather.iloc[:,3].values #[°C] Minimum daily temperature
Tavg = weather.iloc[:,4].values #[°C] Mean daily temperature
Rain = weather.iloc[:,5].values #[mm] Daily precipitation
Igl   = weather.iloc[:,6].values #[MJ m-2] Sum of shortwave radiation daily

I0 = 0.45*Igl*1E6 #Convert [MJ m-2 d-1] to [J m-2 d-1] PAR

d_air = {
      'I0' : np.array([tsim, I0]).T, #[J m-2 d-1] Sum of Global Solar Irradiation (6 AM to 6 PM)
      'CO2': np.array([tsim, np.full((tsim.size,), 400)]).T, #[ppm] CO2 concentration, assume all 400 ppm
      'Tavg':np.array([tsim, Tavg]).T,
      'Rain': np.array([tsim, Rain]).T
      }


d = {'I0':d_air['I0'],
     # 'T':np.array([tsim, np.random.uniform(low=20, high=35, size=tsim.size,)]).T,
     'T': d_air['Tavg'],
     'CO2': np.array([tsim, np.full((tsim.size,), 400)]).T,
     'S_NH4': np.array([tsim, np.full((tsim.size,), 100)]).T,
     'S_NO3': np.array([tsim, np.full((tsim.size,), 100)]).T
     }

# Controlled inputs
#inorganic fertilizer types and concentration used in Indonesia: 
#Nitrogen source: NPK (15%:15%:15%), Urea (46% N)
rice_area = 6000 # m2
NPK_w = 167*rice_area
I_N = (15/100)*NPK_w

#Phosphorus source: SP36 (36%P2O5 ~ 7.85% P)
#P content in SP-36: 7.85% P
Porgf = (7.85/100)

u = {'I_N':I_N}            # [kg m-2 d-1]

# Initialize module
# TODO: Call the module Grass to initialize an instance
rice = Rice(tsim, dt, x0, p)

# Run simulation
# TODO: Call the method run to generate simulation results
tspan = (tsim[0], tsim[-1])
y = rice.run(tspan, d, u)
f_Ph= rice.f['f_Ph']
f_res= rice.f['f_res']
f_gr = rice.f['f_gr']
f_dmv = rice.f['f_dmv']
f_pN = rice.f['f_pN']
f_Nlv = rice.f['f_Nlv']
HU = rice.f['HU']
DVS = rice.f['DVS']
N_lv = rice.f['N_lv']

# Retrieve simulation results
# 1 kg/ha = 0.0001 kg/m2
# 1 kg/ha = 0.1 g/m2
# TODO: Retrieve the simulation results
t = y['t']
Mrt = y['Mrt']
Mst = y['Mst']
Mlv = y['Mlv']
Mpa = y['Mpa']
Mgr = y['Mgr']

# Plot
# # TODO: Make a plot for WsDM, WgDM and grass measurement data.
plt.figure(1)
plt.plot(t, Mrt, label='Roots')
plt.plot(t, Mst, label='Stems')
plt.plot(t, Mlv, label='Leaves')
# plt.plot(t, Mpa, label='Panicles')
plt.plot(t, Mgr, label='Grains')
plt.xlabel(r'time [d]')
plt.ylabel(r'Dry mass [kg DM ha-1 d-1]')
plt.title(r'Accumulative Dry Mass of Rice Crop Organs')
plt.legend()

plt.figure(2)
plt.plot(t, f_Ph, label='f_ph')
plt.plot(t, f_res, label='f_res')
plt.plot(t, f_gr, label='f_gr')
plt.plot(t, f_dmv, label='f_dmv')
plt.xlabel(r'time [d]')
plt.ylabel(r'flow rate [kg CH20 ha-1 d-1]')
plt.title(r'Flow rate in rice plants')
plt.legend()

plt.figure(3)
plt.plot(t, f_Nlv, label='N flow rate in leaves')
plt.plot(t, f_pN, label = 'N flow rate in rice plants')
plt.xlabel(r'time [d]')
plt.ylabel(r'flow rate [kg N ha-1 d-1]')
plt.title(r'Flow rate in rice plants')
plt.legend()

plt.figure(4)
plt.bar('Rice grain yield', Mgr*n_rice)
plt.ylabel(r'Dry mass [kg DM]')
plt.title(r'Rice Grain yield')
plt.legend()

plt.show()