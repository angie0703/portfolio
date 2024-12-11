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
t = 90
tsim = np.linspace(0.0, t, t+1) # [d]
dt = 1 #15 minutes 
tspan = (tsim[0], tsim[-1])

#state variables
n_fish = 60
x0 = {'Mfish': 18.2*n_fish, #[gDM] 
      'Mdig':1E-6*n_fish, #[gDM]
      'Muri':1E-6*n_fish, #[gDM]
      } #concentration in [mg L-1]
 
#parameters
p= {
    'tau_dig': 4.5,  # [h] time constants for digestive
    'tau_uri':20,  # [h] time constants for urinary
    'k_upt':0.3,  # [-] fraction of nutrient uptake for fish weight
    'k_N_upt':0.40,  # [-] fraction of N uptake by fish
    'k_P_upt':0.27,  # [-] fraction of P uptake by fish
    'k_prt':0.2,  # [-] fraction of particulate matter excreted
    'k_N_prt':0.05,  # [-] fraction of N particulate matter excreted
    'k_P_prt':0.27,  # [-] fraction of P particulate matter excreted
    'x_N_fed': 0.05, #[-] fraction of N in feed
    'x_P_fed': 0.01, #[-] fraction of P in feed
    'k_DMR': 0.31,
    'k_TAN_sol': 0.15,
    'Ksp':1,  #[mg C L-1] Half-saturation constant for phytoplankton
    'fT':0,
    'fDO': 0,
    'Tmin': 15,
    'Topt': 25,
    'Tmax': 35
    }


#disturbance
# Disturbances
data_weather = 'C:/Users/alegn/Documents/WUR/Thesis/rfmodel/data/Weather/DIY_202210_202309_Daily.csv'
t_ini = '20221001'
t_end = '20221230'
t_weather = np.linspace(0, 365, 365+1)

weather = pd.read_csv(data_weather, header=1, sep=';')

# Convert the date column to datetime and set as index
weather['Time'] = pd.to_datetime(weather['Time'], format='%Y%m%d')  # Adjust the format if necessary
weather.set_index('Time', inplace=True)
Tavg = weather.loc[t_ini:t_end,'Tavg'].values


d = {'DO':np.array([tsim, np.random.randint(1,6, size=tsim.size)]).T,
     'T':np.array([tsim, Tavg]).T,
     'Mphy':np.array([tsim, np.full((tsim.size,), 1)]).T
     }

#controllable input
u = {'Mfed': 18.2*0.03*n_fish}

#initialize object
# fish = Fish(tsim, dt, x0, p)
# flow = fish.f

# Initialize the Fish model
initial_conditions = {'Mfish': 0, 'Mdig': 0, 'Muri': 0}  # No fish initially
fish = Fish(tsim, dt, x0, p)
#run model

yb = {'t': [], 'Mfish': [], 'Mdig': [], 'Muri': []}
ya = {'t': [], 'Mfish': [], 'Mdig': [], 'Muri': []}

# for ti in tsim:
#     tspan = (tsim[0], tsim[-1])
#     y = fish.run(tspan, d=d, u=u)
#     yb['t'].append(y['t'])
#     yb['Mfish'].append(y['Mfish'])
#     yb['Mdig'].append(y['Mdig'])
#     yb['Muri'].append(y['Muri'])
    
#     if ti == 30:
#         y= fish.restart(x0, tspan,d,u)
#     if ti > 30:
#         ya['t'].append(y['t'])
#         ya['Mfish'].append(y['Mfish'])
#         ya['Mdig'].append(y['Mdig'])
#         ya['Muri'].append(y['Muri'])
for ti in tsim:
    tspan = (ti, ti+1)
    y = fish.run(tspan, d=d, u=u)
    if ti <= 30:
        # Append data to yb until ti reaches 30
        for key in yb:
            yb[key].append(y[key][-1])  # Append the last value of each simulation result
    if ti == 30:
        # Restart the simulation at ti == 30
        y = fish.restart(x0, tspan, d, u)
    if ti > 30:
        # Start appending to ya after the restart
        for key in ya:
            ya[key].append(y[key][-1])  # Append the last value of each simulation result

# Convert lists to arrays for easier handling later on
for key in yb:
    yb[key] = np.array(yb[key])
for key in ya:
    ya[key] = np.array(ya[key])

# #retrieve results
# t= y['t']
# Mfis= y['Mfish']
# Mdig= y['Mdig']
# Muri= y['Muri']
# Mfis_fr= y['Mfish']/p['k_DMR']

# # Plot results
# plt.figure(1)
# plt.plot(t, Mfis, label='Fish dry weight')
# plt.plot(t, Mdig, label='Fish digestive system weight')
# plt.plot(t, Muri, label='Fish urinary system weight')
# plt.plot(t, Mfis_fr, label='Fish fresh weight')
# plt.legend()
# plt.xlabel(r'$time\ [d]$')
# plt.ylabel(r'$biomass\ [g day-1]$')

# plt.figure(2)
# plt.plot(t, flow['f_upt'], label='nutrient uptake flow')
# plt.plot(t, flow['f_N_upt'], label='N uptake flow')
# plt.plot(t, flow['f_P_upt'], label='P uptake flow')
# plt.legend()
# plt.xlabel(r'$time\ [d]$')
# plt.ylabel(r'$flow rate\ [g day-1]$')

# plt.figure(3)
# plt.plot(t, flow['f_sol'], label='total soluble excretion')
# plt.plot(t, flow['f_N_sol'], label='N content in soluble excretion')
# plt.plot(t, flow['f_P_sol'], label='P content in soluble excretion')
# plt.plot(t, flow['f_TAN'], label='soluble TAN')
# plt.legend()
# plt.xlabel(r'$time\ [d]$')
# plt.ylabel(r'$flow rate\ [g day-1]$')

# plt.figure(4)
# plt.plot(t, flow['f_fed'], label='total feed intake rate')
# plt.plot(t, flow['f_prt'], label='total solid excretion')
# plt.plot(t, flow['f_N_prt'], label='N content in solid excretion')
# plt.plot(t, flow['f_P_prt'], label='P content in solid excretion rate')
# plt.legend()
# plt.xlabel(r'$time\ [d]$')
# plt.ylabel(r'$flow rate\ [g day-1]$')
