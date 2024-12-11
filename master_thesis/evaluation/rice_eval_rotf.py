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
t = 364
tsim = np.linspace(0.0, t, t+1)
t_weather = np.linspace(0.0, t, t+1)
dt = 1 # [d]

#pond area and volume
area = 10000 #[m2] the total area of the system
rice_area = 6000 #[m2] rice field area
pond_area = area - rice_area #[m2] pond area

# Initial conditions
# TODO: define sensible values for the initial conditions
n_rice = 127980 #number of plants in 6000 m2 area
x0rice = {'Mrt':0.005,    #[kg DM ha-1 d-1] Dry biomass of root
      'Mst':0.003,    #[kg DM ha-1 d-1] Dry biomass of stems
      'Mlv': 0.002,   #[kg DM ha-1 d-1] Dry biomass of leaves
      'Mpa': 0.0,   #[kg DM ha-1 d-1] Dry biomass of panicles
      'Mgr': 0.0,   #[kg DM ha-1 d-1] Dry biomass of grains
      'HU': 0.0,
      'DVS': 0
      }

# Model parameters
# TODO: define the parameter values in the dictionary p
price = {
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
data_weather = 'C:/Users/alegn/Documents/WUR/Thesis/rfmodel/data/Weather/DIY_202210_202309_Daily.csv'
weather = pd.read_csv(data_weather, header=1, sep=';')

##Rice
#First cycle
t_ini4 = '20221001'
t_end4 = '20230129'

#Second Cycle
t_ini5 = '20230305'
t_end5 = '20230703'
weather['Time'] = pd.to_datetime(weather['Time'], format='%Y%m%d')  # Adjust the format if necessary
weather.set_index('Time', inplace=True)
Tavg4 = weather.loc[t_ini4:t_end4,'Tavg'].values #[°C] Mean daily temperature
Igl4 = weather.loc[t_ini4:t_end4, 'I0'].values #[MJ m-2] Sum of shortwave radiation daily

I04 = 0.45*Igl4*1E6 #Convert [MJ m-2 d-1] to [J m-2 d-1] PAR

#second cycle
Tavg5 = weather.loc[t_ini5:t_end5,'Tavg'].values #[°C] Mean daily temperature
Igl5 = weather.loc[t_ini5:t_end5, 'I0'].values #[MJ m-2] Sum of shortwave radiation daily

I05 = 0.45*Igl5*1E6 #Convert [MJ m-2 d-1] to [J m-2 d-1] PAR

ct_r1 = np.linspace(0,120,120+1, dtype=int)
ct_r2= np.linspace(155, 275, 120+1, dtype=int)

# drice1 = {
#     "I0": np.array([t_weather[ct_r1], I04]).T,
#     "T": np.array([t_weather[ct_r1], Tavg4]).T,
#     "CO2": np.array([tsim[ct_r1], np.full((tsim[ct_r1].size,), 400)]).T,
#     "SNO3": np.array([tsim[ct_r1], np.full((tsim[ct_r1].size,), 0.05326087)]).T,
#     "SP": np.array([tsim[ct_r1], np.full((tsim[ct_r1].size,), 0.0327835051546391)]).T,
# }
# drice2 = {
#     "I0": np.array([t_weather[ct_r2], I05]).T,
#     "T": np.array([t_weather[ct_r2], Tavg5]).T,
#     "CO2": np.array([tsim[ct_r2], np.full((tsim[ct_r2].size,), 400)]).T,
#     "SNO3": np.array([tsim[ct_r2], np.full((tsim[ct_r2].size,), 0.05326087)]).T,
#     "SP": np.array([tsim[ct_r2], np.full((tsim[ct_r2].size,), 0.0327835051546391)]).T,
# }

## only t_weather sliced based on
drice1 = {
    "I0": np.array([t_weather[ct_r1], I04]).T,
    "T": np.array([t_weather[ct_r1], Tavg4]).T,
    "CO2": np.array([tsim, np.full((tsim.size,), 400)]).T,
    "SNO3": np.array([tsim, np.full((tsim.size,), 0.05326087)]).T,
    "SP": np.array([tsim, np.full((tsim.size,), 0.0327835051546391)]).T,
}
drice2 = {
    "I0": np.array([t_weather[ct_r2], I05]).T,
    "T": np.array([t_weather[ct_r2], Tavg5]).T,
    "CO2": np.array([tsim, np.full((tsim.size,), 400)]).T,
    "SNO3": np.array([tsim, np.full((tsim.size,), 0.05326087)]).T,
    "SP": np.array([tsim, np.full((tsim.size,), 0.0327835051546391)]).T,
}

# drice1 = {
#     "I0": np.column_stack((t_weather[ct_r1], I04)),
#     "T": np.column_stack((t_weather[ct_r1], Tavg4)),
#     "CO2": np.column_stack((tsim[ct_r1], np.full((121,), 400))),
#     "SNO3": np.column_stack((tsim[ct_r1], np.full((121,), 0.05326087))),
#     "SP": np.column_stack((tsim[ct_r1], np.full((121,), 0.0327835051546391))),
# }
# drice2 = {
#     "I0": np.column_stack((t_weather[ct_r2], I05)),
#     "T": np.column_stack((t_weather[ct_r2], Tavg5)),
#     "CO2": np.column_stack((tsim[ct_r2], np.full((121,), 400))),
#     "SNO3": np.column_stack((tsim[ct_r2], np.full((121,), 0.05326087))),
#     "SP": np.column_stack((tsim[ct_r2], np.full((121,), 0.0327835051546391))),
# }
# Controlled inputs
#inorganic fertilizer types and concentration used in Indonesia: 
#Nitrogen source: NPK (15%:15%:15%), Urea (46% N)
NPK_w = 167*rice_area
I_N = (15/100)*NPK_w

#Phosphorus source: SP36 (36%P2O5 ~ 7.85% P)
#P content in SP-36: 7.85% P
SP36 = 31*rice_area
I_P = (7.85/100)*SP36

urice = {'I_N':I_N, 'I_P': I_P}            # [kg m-2 d-1]

# Initialize module
# TODO: Call the module to initialize an instance
rice1 = Rice(tsim, dt, x0rice, price)
rice2 = Rice(tsim, dt, x0rice, price)
# Run simulation
# TODO: Call the method run to generate simulation results

yrice1 = rice1.run((0,120), drice1, urice)
yrice2 = rice2.run((155, 276), drice2, urice)

# f_Ph= rice.f['f_Ph']
# f_res= rice.f['f_res']
# f_gr = rice.f['f_gr']
# f_dmv = rice.f['f_dmv']
# f_pN = rice.f['f_pN']
# f_Nlv = rice.f['f_Nlv']
# HU = rice.f['HU']
# DVS = rice.f['DVS']
# DVSf = rice.y['DVSf']
# DVSy = rice.y['DVS']
# DVSc = rice.y['DVSc'] 
# N_lv = rice.f['N_lv']

# Retrieve simulation results
# 1 kg/ha = 0.0001 kg/m2
# 1 kg/ha = 0.0011 ton/ha
# 1 kg/ha = 0.1 g/m2
# TODO: Retrieve the simulation results
# t = y['t']
# Mrt = y['Mrt']
# Mst = y['Mst']
# Mlv = y['Mlv']
# Mpa = y['Mpa']
# Mgr = y['Mgr']
# Mgrt = y['Mgr']*n_rice/1000

# # Plot
# # # TODO: Make a plot for WsDM, WgDM and grass measurement data.
# plt.figure(1)
# plt.plot(t, Mrt, label='Roots')
# plt.plot(t, Mst, label='Stems')
# plt.plot(t, Mlv, label='Leaves')
# plt.plot(t, Mpa, label='Panicles')
# plt.xlabel(r'time [d]')
# plt.ylabel(r'Dry mass [kg DM ha-1 d-1]')
# plt.title(r'Accumulative Dry Mass of Rice Crop Organs')
# plt.legend()

# plt.figure(2)
# plt.plot(t, f_Ph, label='f_ph')
# plt.plot(t, f_res, label='f_res')
# plt.plot(t, f_gr, label='f_gr')
# plt.plot(t, f_dmv, label='f_dmv')
# plt.xlabel(r'time [d]')
# plt.ylabel(r'flow rate [kg CH20 ha-1 d-1]')
# plt.title(r'Flow rate in rice plants')
# plt.legend()

# plt.figure(3)
# plt.plot(t, f_Nlv, label='N flow rate in leaves')
# plt.plot(t, f_pN, label = 'N flow rate in rice plants')
# plt.xlabel(r'time [d]')
# plt.ylabel(r'flow rate [kg N ha-1 d-1]')
# plt.title(r'Flow rate in rice plants')
# plt.legend()

# n_rice1 = 127980  # number of plants using 2:1 pattern
# n_rice2 = 153600  # number of plants using 4:1 pattern

# plt.figure(4)
# plt.bar('P1', Mgr*n_rice1, label='Grains P1')
# plt.bar('P2', Mgr*n_rice2, label='Grains P2')
# plt.xlabel(r'time [d]')
# plt.ylabel(r'Dry mass [kg DM ha-1 d-1]')
# plt.title(r'Accumulative Dry Mass of Rice Crop Organs')
# plt.legend()
# plt.show()