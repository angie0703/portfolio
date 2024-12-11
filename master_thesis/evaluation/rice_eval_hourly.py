# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:53:33 2024

@author: alegn
"""

import numpy as np
import matplotlib.pyplot as plt
from models.rice import Rice_hourly
import pandas as pd

plt.style.use('ggplot')

# Simulation time
t = 120
tsim = np.linspace(0.0, t*24, t*24+1) # [d]
dt = 1 # [h]

#pond area and volume
area = 1 #[ha] the total area of the system
rice_area = 0.6 #[ha] rice field area
pond_area = area - rice_area #[ha] pond area
n_rice = 127980 #number of plants in 0.6 ha land

# Initial conditions
# TODO: define sensible values for the initial conditions

x0rice = {
"Mrt": 0.001,  # [kg DM ] Dry biomass of root
"Mst": 0.002,  # [kg DM] Dry biomass of stems
"Mlv": 0.002,  # [kg DM] Dry biomass of leaves
    "Mpa": 0.0,  # [kg DM] Dry biomass of panicles
    "Mgr": 0.0,  # [kg DM] Dry biomass of grains
    "HU": 0.0,   #[ ] hourly Heat Units
    "DVS": 0,    #[ ] hourly developmental stage
}

# Model parameters
# TODO: define the parameter values in the dictionary p
p = {
     # Bouman (2002):
          # 'DVRi': 0, #[°C h-1]
          # 'DVRJ': 0.000773, # hourly development rate during juvenile phase [°C h-1], originally in [°C d-1]
          # 'DVRI': 0.000758,# hourly development rate during photoperiod-sensitive phase [°C h-1], originally in [°C d-1]
           # 'DVRP': 0.000784,# hourly development rate during panicle development phase [°C h-1], originally in [°C d-1]     
           # 'DVRR': 0.001784, # hourly development rate during reproduction (generative) phase [°C h-1], originally in [°C d-1]
     # #Manually calibrated developmental rate OF rice variety IR64
               'DVRi': 0, #[°C h-1]
                'DVRJ': 0.00137, # hourly development rate during juvenile phase [°C h-1] 0 < DVS < 0.4
                'DVRI': 0.00084,# hourly development rate during photoperiod-sensitive phase [°C h-1], 0.4 <= DVS < 0.65
                'DVRP': 0.00117,# hourly development rate during panicle development phase [°C h-1], 0.65 <= DVS < 1    
                'DVRR': 0.00335, # hourly development rate during reproduction (generative) phase [°C h-1], originally in [°C d-1]
     # #Manually calibrated developmental rate
              # 'DVRi': 0, #[°C h-1]
              #   'DVRJ': 0.00209, # hourly development rate during juvenile phase [°C h-1], originally in [°C d-1]
              #  'DVRI': 0.00085,# hourly development rate during photoperiod-sensitive phase [°C h-1], originally in [°C d-1]
              #  'DVRP': 0.00135,# hourly development rate during panicle development phase [°C h-1], originally in [°C d-1]     
              #  'DVRR': 0.00335, # hourly development rate during reproduction (generative) phase [°C h-1], originally in [°C d-1]
         "Tmax": 40,# maximum temperature for development [°C]
         "Tmin": 15,# minimum temperature for development [°C]
         "Topt": 33,# optimum temperature for development [°C]
         "Rm_rt": 0.01,
         "Rm_st": 0.015,
         "Rm_lv": 0.02,
         "Rm_pa": 0.003,
         "k_pa_maxN": 0.0175, #[kg N/kg panicles]
         "cr_lv": 1.326,
         "cr_st": 1.326,
         "cr_pa": 1.462,
         "cr_rt": 1.326,
         'n_rice': n_rice
        }

#%% Disturbances
data_weather = 'C:/Users/alegn/Documents/WUR/Thesis/rfmodel/data/Weather/DIY_202210_202301_Hourly.csv'
weather = pd.read_csv(data_weather, header=0, sep=';')

#FIRST CYCLE
t_ini = '2022-10-01T08:00'
t_end = '2023-01-29T08:00'

weather['Time'] = pd.to_datetime(weather['Time'])
weather.set_index('Time', inplace=True)
#first cycle
T = weather.loc[t_ini:t_end,'Temp'].values #[°C] hourly temperature
Rain = weather.loc[t_ini:t_end,'Rain'].values #[mm] hourly precipitation
Igl = weather.loc[t_ini:t_end, 'I0'].values #[MJ m-2] Sum of shortwave radiation Hourly

I0 = 0.45*Igl*1E6 #Convert [MJ m-2 h-1] to [J m-2 h-1] PAR

drice = {'I0':np.array([tsim, I0]).T,
     'Th': np.array([tsim, T]).T,
     'CO2': np.array([tsim, np.full((tsim.size,), 400)]).T,
     'Nrice': np.array([tsim, np.full((tsim.size,), 4800)]).T
     }

# Initialize module
# TODO: Call the module Grass to initialize an instance
rice = Rice_hourly(tsim, dt, x0rice, p)
flow_rice = rice.f
# Run simulation
# TODO: Call the method run to generate simulation results
tspan = (tsim[0], tsim[-1])
y = rice.run(tspan, drice)
f_Ph= rice.f['f_Ph'] #[kg CO2 h-1]
f_res= rice.f['f_res'] #[kg CH20 h-1]
f_dmv = rice.f['f_dmv'] #[kg DM leaf h-1]
f_Nlv = rice.f['f_Nlv'] #[kg N h-1]
f_uptN = rice.f['f_uptN'] #[kg N h-1]
DVS = rice.y['DVS'] 
N_lv = rice.f['N_lv']

# Retrieve simulation results
# TODO: Retrieve the simulation results
t = y['t']
Mrt = y['Mrt']
Mst = y['Mst']
Mlv = y['Mlv']
Mpa = y['Mpa']
Mgr = y['Mgr']
Mrice = Mgr/600 #ton/ha


x_ticks = np.linspace (1, 2880, 120)
days = np.array([5, 21, 30, 60, 90, 120])
days_grain = np.array([85, 88, 91, 94, 97, 100, 103, 106, 109, 112, 115, 118, 120])
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

one_cycle = np.linspace(1,120,120, dtype=int)
tick_position = np.arange(0, 2881, 24)
day_labels = np.arange(1,121)
# days = np.array([30, 60, 90, 120])

plt.figure(1)
plt.plot(t, Mrt, label='$M_{rt}$')
plt.plot(t, Mst, label='$M_{st}$')
plt.plot(t, Mlv, label='$M_{lv}$')
plt.plot(t, Mpa, label='$M_{pa}$')
plt.plot(t, Mgr, label ='$M_{gr}$')
plt.xlabel(r'time [h]')
# plt.xticks(days*24, labels=[f"{day}" for day in days])
plt.ylabel(r'Mass $[kg]$')
plt.legend()

plt.figure(2)
plt.plot(t, Mrice, label = '$M_{rice}$')
plt.xlabel(r'time [d]')
plt.ylabel(r'Yield $[ton ha^{-1}]$')
# plt.xticks(days_grain*24, labels=[f"{day}" for day in days_grain])
plt.legend()


plt.figure(5)
plt.plot(t, DVS)
plt.legend()
plt.xlabel(r'time [d]')
plt.xticks(days*24, labels=[f"{day}" for day in days])
plt.yticks([0.4, 0.65, 1, 2])
plt.ylabel(r'DVS [-]')
plt.legend()

# Create a figure and a set of subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True)

# Plot in each subplot
# Subplot 1 (top left)
axs[0, 0].plot(t, f_Ph, label = '$\phi_{pgr}$')
axs[0, 0].set_xlabel('time [d]')
# axs[0, 0].set_xticks(days*24, labels=[f"{day}" for day in days])
axs[0, 0].set_ylabel('mass flow rate [$kg CO_2 h^{-1}$]')
axs[0, 0].legend()

# Subplot 2 (top right)
axs[0, 1].plot(t, f_dmv, label = '$\phi_{dmv}$')
axs[0, 1].set_xlabel('time [d]')
# axs[0, 1].set_xticks(days*24, labels=[f"{day}" for day in days])
axs[0, 1].set_ylabel('mass flow rate [$kg DM leaf h^{-1}$]')
axs[0, 1].legend()

# Subplot 3 (bottom left)
axs[1, 0].plot(t, f_res, label ='$\phi_{res}$')
axs[1, 0].set_xlabel('time [d]')
# axs[1, 0].set_xticks(days*24, labels=[f"{day}" for day in days])
axs[1, 0].set_ylabel('mass flow rate [$kg CH_2O h^{-1}$]')
axs[1, 0].legend()

# Subplot 4 (bottom right)
axs[1, 1].plot(t, f_uptN, label = '$\phi_{N,plt,upt}$')
axs[1, 1].set_xlabel('time [d]')
# axs[1, 1].set_xticks(days*24, labels=[f"{day}" for day in days])
axs[1,1].set_ylabel('mass flow rate [$kg N ha^{-1} h^{-1}$]')
axs[1,1].legend()

# Adjust layout so titles and labels don't overlap
plt.tight_layout()