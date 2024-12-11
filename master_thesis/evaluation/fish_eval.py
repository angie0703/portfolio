# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 12:51:08 2023

@author: Angela
"""

from models.fish import Fish
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')

# Simulation time array
# Simulation time array
t = 120 # [d]
tsim = np.linspace(0.0, t, t+1) # [d]
t_weather = np.linspace(0.0, t, t+1) 
dt = 1/24 # [d] # 5 minutes according to Huesemann?
A_sys = 10000 # [m2] the total area of the system
A_rice = 6000  # [m2] rice field area
A_pond = A_sys - A_rice  # [m2] pond area
V_pond = A_pond*0.6 #[m3] Volume of the pond 0.6 m is the pond depth
V_phy = V_pond + A_rice*0.1 #[m3]

#state variables
n_fish = 4000
x0 = {'Mfish': 30*n_fish, #[gDM] 
      'Mdig':1E-6*n_fish, #[gDM]
      'Muri':1E-6*n_fish, #[gDM]
      } #concentration in [mg L-1]
 
#parameters
p= {
    "tau_dig": 4.5/24,  # [d] time constants for digestive *24 for convert into day
    "tau_uri": 20/24,  # [d] time constants for urinary *24  convert into day
    "k_upt": 0.3,  # [-] fraction of nutrient uptake for fish weight
    "k_N_upt": 0.40,  # [-] fraction of N uptake by fish
    "k_P_upt": 0.27,  # [-] fraction of P uptake by fish
    "k_prt": 0.2,  # [-] fraction of particulate matter excreted
    "k_N_prt": 0.05,  # [-] fraction of N particulate matter excreted
    "k_P_prt": 0.27,  # [-] fraction of P particulate matter excreted
    "x_N_fed": 0.05,  # [-] fraction of N in feed
    "x_P_fed": 0.01,  # [-] fraction of P in feed
    "k_DMR": 0.31,
    "k_TAN_sol": 0.15,
    "Ksp": 1,  # [g C m-3] Half-saturation constant for phytoplankton
    "Tmin": 22,
    "Topt": 28,
    "Tmax": 32,
    'V_pond': V_pond
    }


#disturbance
# Disturbances
data_weather = 'C:/Users/alegn/Documents/WUR/Thesis/rfmodel/data/Weather/Magelang_200110_200209_Daily.csv'
weather = pd.read_csv(data_weather, header=0, sep=';')
weather['Time'] = pd.to_datetime(weather['Time'], format='%Y%m%d')  # Adjust the format if necessary
weather.set_index('Time', inplace=True)
#FIRST CYCLE
t_ini = '20011001'
t_end = '20020129'
t_weather = np.linspace(0, 120, 120+1)
Tavg = weather.loc[t_ini:t_end,'Tavg'].values


dfish = {
    "DO": np.array([tsim, np.random.randint(1, 6, size=tsim.size)]).T,
    "T": np.array([t_weather, Tavg]).T,
    "Mphy": np.array([tsim, np.full((tsim.size,), 4.293e-4)]).T,
}

#initialize object
fish = Fish(tsim, dt, x0, p)
flow = fish.f
#run model
tspan = (tsim[0], tsim[-1])
y = fish.run(tspan, dfish)

#retrieve results
t= y['t']
Mfis= y['Mfish']
Mdig= y['Mdig']
Muri= y['Muri']
Mfis_fr= y['Mfish']/p['k_DMR']

# Plot results
plt.figure(figsize=(10,10))
plt.figure(1)
plt.plot(t, Mfis*0.001)
# plt.plot(t, flow['f_fed'], label='total feed intake rate')
# plt.plot(t, flow['f_sol'], label='total soluble excretion')
# plt.plot(t, flow['f_prt'], label='total solid excretion')
# plt.plot(t, Mdig, label='Fish digestive system weight')
# plt.plot(t, Muri, label='Fish urinary system weight')
# plt.plot(t, Mfis_fr, label='Fish fresh weight')
plt.legend()
plt.xlabel(r'$time\ [d]$', fontsize=18)
plt.ylabel(r'biomass accumulation $[kg DM d^{-1}]$', fontsize=18)

plt.figure(3)
plt.plot(t, flow['f_fed'], label='total feed intake rate')
plt.plot(t, flow['f_sol'], label='total soluble excretion')
plt.plot(t, flow['f_prt'], label='total solid excretion')
# plt.plot(t, flow['f_N_sol'], label='N content in soluble excretion')
# plt.plot(t, flow['f_P_sol'], label='P content in soluble excretion')
# plt.plot(t, flow['f_TAN'], label='soluble TAN')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'flow rate $[g d^{-1}]$')