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

#pond area and volume
area = 1 #[ha] the total area of the system
rice_area = 0.6 #[ha] rice field area
pond_area = area - rice_area #[ha] pond area
n_rice = 127980 #number of plants in 0.6 ha land

# Initial conditions
# TODO: define sensible values for the initial conditions

x0rice = {
    "Mrt": 0.005,  # [kg DM ] Dry biomass of root
    "Mst": 0.003,  # [kg DM] Dry biomass of stems
    "Mlv": 0.002,  # [kg DM] Dry biomass of leaves
    "Mpa": 0.0,  # [kg DM] Dry biomass of panicles
    "Mgr": 0.0,  # [kg DM] Dry biomass of grains
    "HU": 0.0,
    "DVS": 0,
}

# Model parameters
# TODO: define the parameter values in the dictionary p
price = {
      #Manually calibrated developmental rate
         'DVSi': 0, #[°C d-1]
         'DVRJ': 0.0020, #[°C d-1]
         'DVRI': 0.00195,#[°C d-1]
         'DVRP': 0.00195,#[°C d-1]     
         'DVRR': 0.0024,#[°C d-1]
         "Tmax": 40,#[°C]
         "Tmin": 15,#[°C]
         "Topt": 33,#[°C]
         "k": 0.4,
         "Rm_rt": 0.01,
         "Rm_st": 0.015,
         "Rm_lv": 0.02,
         "Rm_pa": 0.003,
         "N_lv": 0,
         "k_lv_N": 0,
         "k_pa_maxN": 0.0175, #[kg N/kg panicles]
         "cr_lv": 1.326,
         "cr_st": 1.326,
         "cr_pa": 1.462,
         "cr_rt": 1.326,
         'n_rice': n_rice
        }

#%% Disturbances
data_weather = 'C:/Users/alegn/Documents/WUR/Thesis/rfmodel/data/Weather/Magelang_200110_200209_Daily.csv'
weather = pd.read_csv(data_weather, header=0, sep=';')

#FIRST CYCLE
t_ini = '20011001'
t_end = '20020129'

weather['Time'] = pd.to_datetime(weather['Time'], format='%Y%m%d')  # Adjust the format if necessary
weather.set_index('Time', inplace=True)
#first cycle
Tavg = weather.loc[t_ini:t_end,'Tavg'].values #[°C] Mean daily temperature
Rain = weather.loc[t_ini:t_end,'Rain'].values #[mm] Daily precipitation
Igl = weather.loc[t_ini:t_end, 'I0'].values #[MJ m-2] Sum of shortwave radiation daily

I0 = 0.45*Igl*1E6 #Convert [MJ m-2 d-1] to [J m-2 d-1] PAR

drice = {'I0':np.array([tsim, I0]).T,
     'T': np.array([tsim, Tavg]).T,
     'CO2': np.array([tsim, np.full((tsim.size,), 400)]).T,
     'f_N_plt_upt': np.array([tsim, np.full((tsim.size,), 480)]).T
     }

# Initialize module
# TODO: Call the module Grass to initialize an instance
rice = Rice(tsim, dt, x0rice, price)

# Run simulation
# TODO: Call the method run to generate simulation results
tspan = (tsim[0], tsim[-1])
y = rice.run(tspan, drice)
f_Ph= rice.f['f_Ph']
f_res= rice.f['f_res']
kgr = rice.f['f_gr']
f_dmv = rice.f['f_dmv']
f_pN = rice.f['f_pN']
f_Nlv = rice.f['f_Nlv']
f_uptN = rice.f['f_uptN']
HU = rice.f['HU']
DVSf = rice.f['DVS']
DVS = rice.y['DVS'] 
N_lv = rice.f['N_lv']
IgN = np.array([tsim, np.full((tsim.size,), 0.8)]).T

# Retrieve simulation results
# TODO: Retrieve the simulation results
t = y['t']
Mrt = y['Mrt']
Mst = y['Mst']
Mlv = y['Mlv']
Mpa = y['Mpa']
Mgr = y['Mgr']


plt.rcParams.update({
    'figure.figsize': [10,10],
    'font.size': 14,        # Global font size
    'axes.titlesize': 16,   # Title font size
    'axes.labelsize': 14,   # X and Y axis labels font size
    'xtick.labelsize': 14,  # X tick labels font size
    'ytick.labelsize': 14,  # Y tick labels font size
    'legend.fontsize': 14,  # Legend font size
    'figure.titlesize': 18  # Figure title font size
})

plt.figure(1)
plt.plot(t, Mrt, label='$M_{rt}$')
plt.plot(t, Mst, label='$M_{st}$')
plt.plot(t, Mlv, label='$M_{lv}$')
plt.plot(t, Mpa, label='$M_{pa}$')
plt.xlabel(r'time [d]')
plt.ylabel(r'Dry mass $[kg DM d^{-1}]$')
plt.legend()

# plt.figure(2)
# plt.plot(t, f_Ph, label='Photosynthesis')
# plt.plot(t, f_res, label='Maintenance respiration')
# plt.plot(t, f_gr, label='Growth respiration')
# plt.plot(t, f_dmv, label='Leaf death')
# plt.xlabel(r'time [d]')
# plt.ylabel(r'flow rate $[kg CH2O ha^{-1} d^{-1}]$')
# # plt.title(r'Flow rate in rice plants')
# plt.legend()

# plt.figure(3)
# plt.plot(t, f_Nlv, label='N flow rate in leaves')
# plt.plot(t, f_pN, label = 'N flow rate in rice plants')
# plt.xlabel(r'time [d]')
# plt.ylabel(r'flow rate $[kg N ha^{-1} d^{-1}]$')
# # plt.title(r'Flow rate in rice plants')
# plt.legend()

# plt.figure(4)
# plt.plot(t, f_uptN, label = 'Nitrogen uptake from fish pond')
# plt.plot(t, f_Nfert, label = 'Nitrogen uptake from inorganic fertilizer')
# plt.plot(t, IgN[:,1], label = 'Indigenous N uptake from soil')
# plt.xlabel(r'time [d]')
# plt.ylabel(r'flow rate $[kg N ha^{-1} d^{-1}]$')
# plt.legend()

# plt.figure(6)
# plt.plot(t, f_uptN + f_Nfert + IgN[:,1], label = 'Actual Nitrogen supply')
# plt.plot(t, f_pN, label = 'N flow rate in rice plants')
# plt.xlabel(r'time [d]')
# plt.ylabel(r'flow rate $[kg N ha^{-1} d^{-1}]$')
# plt.legend()

# plt.figure(5)
# plt.plot(t, DVSf)
# y_min, y_max = plt.ylim()
# plt.axvline(x=0, color='b', linestyle='--', linewidth=1)
# plt.annotate('DVS=0', xy=(0, y_min), xytext=(0, y_min),
#              fontsize=12, ha='center')
# plt.axvline(x=32.5, color='b', linestyle='--', linewidth=1)
# plt.annotate('DVS=0.4', xy=(32.5, y_min), xytext=(32.5, y_min),
#              fontsize=12, ha='center')
# plt.axvline(x=45.2, color='b', linestyle='--', linewidth=1)
# plt.annotate('DVS=0.65', xy=(45.2, 0), xytext=(45.2, 0.1),
#              fontsize=12, ha='center')
# plt.axvline(x=64, color='b', linestyle='--', linewidth=1)
# plt.annotate('DVS=1', xy=(64, 0), xytext=(64, y_min),
#              fontsize=12, ha='center')
# plt.axvline(x=115.8, color='b', linestyle='--', linewidth=1)
# plt.annotate('DVS=2', xy=(115.8, 0), xytext=(115.8, y_min),
#              fontsize=12, ha='center')
# plt.xlabel(r'time [d]')
# plt.ylabel(r'DVS [-]')
# plt.text(0, 2, 'x=Day 0', ha='center')
# plt.text(32.5, 2, 'x ~ Day 32', ha='center')
# plt.text(45.2, 1.5, 'x ~ Day 45', ha='center')
# plt.text(64, 2, 'x ~ Day 64', ha='center')
# plt.text(115.8, 2, 'x ~ Day 115', ha='center')
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# # plt.title(r'Daily Developmental stage of rice plants')

