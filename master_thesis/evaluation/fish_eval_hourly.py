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
t = 55 # [d]
tsim = np.linspace(0.0, t*24, t*24+1) # [d] 
dt = 1 # [d] # 5 minutes according to Huesemann?
A_sys = 10000 # [m2] the total area of the system
A_rice = 6000  # [m2] rice field area
A_pond = A_sys - A_rice  # [m2] pond area
V_pond = A_pond*0.6 #[m3] Volume of the pond 0.6 m is the pond depth
V_phy = V_pond + A_rice*0.1 #[m3]

#state variables
n_fish = 10000 #[fish]
x0 = {'Mfish': 30*n_fish, #[gDM] 
      'Mdig':1E-6*n_fish, #[gDM]
      'Muri':1E-6*n_fish, #[gDM]
      } 
 
#parameters
p= {
    "tau_dig": 4.5,  # [h] time constants for digestive *24 for convert into day
    "tau_uri": 20,  # [h] time constants for urinary *24  convert into day
    "k_upt": 0.3,  # [-] fraction of nutrient uptake for fish weight
    "k_N_upt": 0.40,  # [-] fraction of N uptake by fish
    "k_P_upt": 0.27,  # [-] fraction of P uptake by fish
    "k_prt": 0.2,  # [-] fraction of particulate matter excreted
    "k_N_prt": 0.05,  # [-] fraction of N particulate matter excreted
    "k_P_prt": 0.27,  # [-] fraction of P particulate matter excreted
    "x_N_fed": 0.05,  # [-] fraction of N in feed
    "x_P_fed": 0.01,  # [-] fraction of P in feed
    "k_DMR": 0.31,
    "Ksp": 1,  # [g C m-3] Half-saturation constant for phytoplankton
    "Tmin": 22,
    "Topt": 28,
    "Tmax": 32,
    'V_pond': V_pond
    }


#%% Disturbances
data_weather = 'C:/Users/alegn/Documents/WUR/Thesis/rfmodel/data/Weather/DIY_202210_202301_Hourly.csv'
weather = pd.read_csv(data_weather, header=0, sep=';')

#FIRST CYCLE
t_ini = '2022-10-25T08:00'
t_end = '2022-12-19T08:00'

weather['Time'] = pd.to_datetime(weather['Time'])
weather.set_index('Time', inplace=True)
#first cycle
T = weather.loc[t_ini:t_end,'Temp'].values #[°C] Mean hourly temperature
Rain = weather.loc[t_ini:t_end,'Rain'].values #[mm] Daily precipitation
Igl = weather.loc[t_ini:t_end, 'I0'].values #[MJ m-2] Sum of shortwave radiation daily

I0 = 0.45*Igl*1E6 #Convert [MJ m-2 d-1] to [J m-2 d-1] PAR


dfish = {
    "DO": np.array([tsim, np.random.randint(1, 6, size=tsim.size)]).T,
    "T": np.array([tsim, T]).T,
    # "Mphy": np.array([tsim, np.full((tsim.size,), 4.293e-4*3000)]).T,
    "Mphy": np.array([tsim, np.linspace(4.293e-4*V_phy, 4.55*V_phy, t*24+1)]).T,
}

#%%initialize object
fish = Fish(tsim, dt, x0, p)
flow = fish.f
#run model
tspan = (tsim[0], tsim[-1])
y = fish.run(tspan, dfish)

#retrieve results
t= y['t']
Mfis= y['Mfish']/1000 #[kg]
Mdig= y['Mdig']/1000 #[kg]
Muri= y['Muri']/1000 #[kg]
Mfis_fr= Mfis/p['k_DMR'] #[kg]
Mfis_fr_ton_ha= Mfis_fr/400 #[ton/ha]

x_ticks = np.linspace (1, 2880, 120)

# Plot results
plt.figure(figsize=(10,10))
# plt.figure(1)
# plt.plot(t, Mfis, label = '$M_{fish}$')
# plt.legend()
# plt.xlabel(r'$time\ [h]$', fontsize=18)
# plt.ylabel(r'biomass accumulated $[kg DM h^{-1}]$', fontsize=18)

# plt.figure(2)
# plt.plot(t, Mdig, label='$M_{dig}$')
# plt.legend()
# plt.xlabel(r'$time\ [h]$', fontsize=18)
# plt.ylabel(r'biomass accumulated $[kg DM h^{-1}]$', fontsize=18)

# plt.figure(3)
# plt.plot(t, Muri, label='$M_{uri}$')
# plt.legend()
# plt.xlabel(r'$time\ [h]$', fontsize=18)
# plt.ylabel(r'mass $[kg DM h^{-1}]$', fontsize=18)

# plt.figure(4)
# plt.plot(t, flow['f_upt']/1000, label = '$\phi_{upt}$')
# plt.legend()
# plt.xlabel(r'$time\ [h]$', fontsize=18)
# plt.ylabel(r'mass rate $[kg h^{-1}]$', fontsize=18)

# plt.figure(5)
# plt.plot(t, flow['f_fed']/1000, label = '$\phi_{feed}$')
# plt.plot(t, flow['f_fed_phy']/1000, label = '$\phi_{fed, phy}$')
# plt.plot(t, -flow['f_digout']/1000, label = '$\phi_{digout}$')
# plt.legend()
# plt.xlabel(r'$time\ [h]$', fontsize=18)
# plt.ylabel(r'mass rate $[kg h^{-1}]$', fontsize=18)

# plt.figure(6)
# plt.plot(t, flow['f_diguri']/1000, label = '$\phi_{diguri}$')
# plt.plot(t, -flow['f_sol']/1000, label = '$\phi_{sol}$')
# plt.legend()
# plt.xlabel(r'$time\ [h]$', fontsize=18)
# plt.ylabel(r'mass rate $[kg h^{-1}]$', fontsize=18)

plt.figure(7)
plt.plot(t, Mfis*1000/n_fish, label = '$M_{fish}$')
plt.plot(t, Mfis_fr*1000/n_fish, label = '$M_{fis,fr}$')
plt.legend()
plt.xlabel(r'$time\ [h]$', fontsize=18)
plt.ylabel(r'biomass accumulated $[g DM h^{-1}]$', fontsize=18)
plt.title('Mass of one fish')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax1.plot(t, Mfis, label = '$M_{fish}$')
ax1.set_ylabel('mass [$kg$]')
ax1.legend()
# ax1.set_title('Net phosphorus accumulated')
# # Plot the second subplot
ax2.plot(t, flow['f_upt']/1000, label='$\phi_{upt}$')
ax2.set_ylabel(r'rate $[kg h^{-1}]$')
ax2.set_xticks(x_ticks)
ax2.set_xticklabels(np.arange(1, 120+1))
ax2.set_xlabel('time [d]')
ax2.legend()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax1.plot(t, Mdig, label = '$M_{dig}$')
ax1.set_ylabel('mass [$kg$]')
ax1.legend()

# # Plot the second subplot
ax2.plot(t, flow['f_fed']/1000, label='$\phi_{fed}$')
ax2.plot(t, flow['f_fed_phy']/1000, label='$\phi_{fed,phy}$')
ax2.plot(t, -flow['f_digout']/1000, label='$\phi_{digout}$')
ax2.set_ylabel(r'rate $[kg h^{-1}]$')
ax2.set_xticks(x_ticks)
ax2.set_xticklabels(np.arange(1, 120+1))
ax2.set_xlabel('time [d]')
ax2.legend()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax1.plot(t, Muri, label = '$M_{uri}$')
ax1.set_ylabel('mass [$kg$]')
ax1.legend()

# # Plot the second subplot
ax2.plot(t, flow['f_diguri']/1000, label='$\phi_{diguri}$')
ax2.plot(t, -flow['f_sol']/1000, label='$\phi_{sol}$')
ax2.set_ylabel(r'rate $[kg h^{-1}]$')
ax2.set_xticks(x_ticks)
ax2.set_xticklabels(np.arange(1, 120+1))
ax2.set_xlabel('time [d]')
ax2.legend()