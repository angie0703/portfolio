# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 12:51:08 2023

@author: Angela
"""

from models.fish import Fish
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# Simulation time array
t = 120
tsim = np.linspace(0.0, t, t+1) # [d]
dt = 1 #[d] 
tspan = (tsim[0], tsim[-1])

#state variables
x0 = {'Mfish': 80, #[gDM] 
      'Mdig':1E-6, #[gDM]
      'Muri':1E-6, #[gDM]
      'Mfis_fr':0, #[g]
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
    'r': 0.02,
    'Tmin': 15,
    'Topt': 25,
    'Tmax': 35,
    'n_fish': 60
    }


#disturbance
DOvalues = 5
Tvalues = 15
d = {'DO':np.array([tsim, np.full((tsim.size,),DOvalues)]).T,
     'T':np.array([tsim, np.full((tsim.size,),Tvalues)]).T
     }

#controllable input
u = {'Mphy': np.array([tsim, np.full((tsim.size,),10)]).T,
     'Mfed': np.array([tsim, np.full((tsim.size,),10)]).T}


 
#initialize object
fish = Fish(tsim, dt, x0, p)
flow = fish.f

#run model
tspan = (tsim[0], tsim[-1])
for i, u_value in enumerate(u['Mphy']):
    u_i = {'Mphy':u_value, 'Mfed' : 4}
    y = fish.run(tspan, d=d, u=u_i)

#retrieve results
t= y['t']
Mfis= y['Mfish']
Muri= y['Muri']
Mfis_fr= y['Mfis_fr']
Mdig= y['Mdig']

# Plot results
plt.figure(1)
plt.plot(t, Mfis, label='Fish dry weight')
plt.plot(t, Mdig, label='Fish digestive system weight')
plt.plot(t, Muri, label='Fish urinary system weight')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$biomass\ [g day-1]$')

plt.figure(2)
plt.plot(t, flow['f_upt'], label='nutrient uptake flow')
plt.plot(t, flow['f_N_upt'], label='N uptake flow')
plt.plot(t, flow['f_P_upt'], label='P uptake flow')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$flow rate\ [g day-1]$')

plt.figure(3)
plt.plot(t, flow['f_sol'], label='total soluble excretion')
plt.plot(t, flow['f_N_sol'], label='N content in soluble excretion')
plt.plot(t, flow['f_P_sol'], label='P content in soluble excretion')
plt.plot(t, flow['f_TAN'], label='soluble TAN')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$flow rate\ [g day-1]$')

plt.figure(4)
plt.plot(t, flow['f_fed'], label='total feed intake rate')
plt.plot(t, flow['f_prt'], label='total solid excretion')
plt.plot(t, flow['f_N_prt'], label='N content in solid excretion')
plt.plot(t, flow['f_P_prt'], label='P content in solid excretion rate')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$flow rate\ [g day-1]$')

plt.figure(5)
plt.plot(t, flow['f_fed'], label='total feed intake rate')
plt.plot(t, flow['f_prt'], label='total solid excretion')
plt.plot(t, flow['f_upt'], label='total soluble excretion')
# plt.plot(t, flow['f_P_prt'], label='P content in solid excretion rate')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$flow rate\ [g day-1]$')

plt.figure(6)
plt.plot(t, Mfis_fr, label='Fish fresh weight')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$biomass\ [g day-1]$')