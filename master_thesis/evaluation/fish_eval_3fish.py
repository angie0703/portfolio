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
t = 365
tsim = np.linspace(0.0, t, t+1) # [d]
dt = 1 #15 minutes 
tspan = (tsim[0], tsim[-1])

#state variables
n_fish = 4000
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
    'Tmin': 22,
    'Topt': 28,
    'Tmax': 32
    }


#disturbance
# Disturbances
data_weather = 'C:/Users/alegn/Documents/WUR/Thesis/rfmodel/data/Weather/DIY_202210_202309_Daily.csv'
t_ini1 = '20221001'
t_end1 = '20221031'

t_ini2 = '20230130'
t_end2 = '20230326'

t_ini3 = '20230703'
t_end3 = '20230827'

t_weather = np.linspace(0, 365, 365+1)

weather = pd.read_csv(data_weather, header=1, sep=';')

# Convert the date column to datetime and set as index
weather['Time'] = pd.to_datetime(weather['Time'], format='%Y%m%d')  # Adjust the format if necessary
weather.set_index('Time', inplace=True)
Tavg1 = weather.loc[t_ini1:t_end1,'Tavg'].values
Tavg2 = weather.loc[t_ini2:t_end2,'Tavg'].values
Tavg3 = weather.loc[t_ini3:t_end3,'Tavg'].values


d1 = {'DO':np.array([tsim, np.random.randint(1,6, size=tsim.size)]).T,
     'T':np.array([tsim[:31], Tavg1]).T,
     'Mphy':np.array([tsim, np.full((tsim.size,), 1)]).T
     }
d2 = {'DO':np.array([tsim, np.random.randint(1,6, size=tsim.size)]).T,
     'T':np.array([tsim[122:178], Tavg2]).T,
     'Mphy':np.array([tsim, np.full((tsim.size,), 1)]).T
     }
d3 = {'DO':np.array([tsim, np.random.randint(1,6, size=tsim.size)]).T,
     'T':np.array([tsim[275:331], Tavg3]).T,
     'Mphy':np.array([tsim, np.full((tsim.size,), 1)]).T
     }

#controllable input
u = {'Mfed': 18.2*0.03*n_fish}

#initialize object
fish1 = Fish(tsim, dt, x0, p)
fish2 = Fish(tsim, dt, x0, p)
fish3 = Fish(tsim, dt, x0, p)

yfish1 = fish1.run((0,31), d1, u)
yfish2 = fish2.run((122,178), d2, u)
yfish3 = fish3.run((275,331), d3, u)

plt.figure(figsize=(10, 6))
plt.plot(yfish1['t'], yfish1['Mfish'], label='Cycle 1: Day 0 to 31', marker='o')
plt.plot(yfish2['t'] + 122, yfish2['Mfish'], label='Cycle 2: Day 122 to 178', marker='o')  # Offset by 122
plt.plot(yfish3['t'] + 275, yfish3['Mfish'], label='Cycle 3: Day 275 to 331', marker='o')  # Offset by 275

# Set the labels and title
plt.xlabel('Day of Year')
plt.ylabel('Fish Mass (gDM)')
plt.title('Fish Mass Over Time Across Different Cycles')
plt.legend()
plt.grid(True)



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
