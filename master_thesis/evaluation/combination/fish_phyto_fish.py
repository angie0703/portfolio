# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 09:08:22 2024

@author: alegn
"""

from models.fish import Fish
from models.phytoplankton import Phygrowth
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')

# Simulation time array
t = 120
tsim = np.linspace(0.0, t, t+1) # [d]
t_weather = np.linspace(0.0, t, t+1) 
dt = 1
tspan = (tsim[0], tsim[-1])

# Disturbances
data_weather = 'C:/Users/alegn/Documents/WUR/Thesis/rfmodel/data/Weather/Magelang_200110_200209_Daily.csv'
weather = pd.read_csv(data_weather, header=0, sep=';')

#FIRST CYCLE
t_ini = '20011001'
t_end = '20020129'

#SECOND CYCLE
t_ini2 = '20020130'
t_end2 = '20020530'

weather['Time'] = pd.to_datetime(weather['Time'], format='%Y%m%d')  # Adjust the format if necessary
weather.set_index('Time', inplace=True)
#first cycle
Tavg1 = weather.loc[t_ini:t_end,'Tavg'].values #[°C] Mean daily temperature
Rain1 = weather.loc[t_ini:t_end,'Rain'].values #[mm] Daily precipitation
Igl1 = weather.loc[t_ini:t_end, 'I0'].values #[MJ m-2] Sum of shortwave radiation daily

I01 = 0.45*Igl1*1E6 #Convert [MJ m-2 d-1] to [J m-2 d-1] PAR

#second cycle
Tavg2 = weather.loc[t_ini2:t_end2,'Tavg'].values #[°C] Mean daily temperature
Rain2 = weather.loc[t_ini2:t_end2,'Rain'].values #[mm] Daily precipitation
Igl2 = weather.loc[t_ini2:t_end2, 'I0'].values #[MJ m-2] Sum of shortwave radiation daily

I02 = 0.45*Igl2*1E6 #Convert [MJ m-2 d-1] to [J m-2 d-1] PAR
##phytoplankton model
x0phy = {'Mphy': 4.29e-4, #[g m-3] from Mei et al 2023 concentration of Chla multiply with Chla:phyto mass ratio from Jamu and Piedrahita (2002)
      'NA':   0.55 #[g m-3] TN from Mei et al 2023
      } 

# Model parameters
# TODO: define the parameter values in the dictionary p
pphy = {
      'mu_phy':  1.27, #[d-1] maximum growth rate of phytoplankton
      'mu_Up': 0.005, #[d-1] maximum nutrient uptake coefficient
      'pd': 0.5, #[m] pond depth
      'l_sl': 0.00035, #[m2 mg-1] phytoplankton biomass-specific light attenuation
      'l_bg': 0.77, #[m-1] light attenuation by non-phytoplankton components
      'Kpp': 0.234*60, # [J m-2 s-1] half-saturation constant of phytoplankton production
      'cm': 0.15, #[d-1] phytoplankton mortality constant
      'cl': 4e-6, #[m2 g-1] phytoplankton crowding loss constant
      'c1': 1.57, # [-] temperature coefficients
      'c2': 0.24, # [-] temperature coefficients
      'Topt': 28, #optimum temperature for phytoplankton growth
      'Mp': 0.025 #[g m-3] half saturation constant for nutrient uptake
     }

#initialize object
phyto = Phygrowth(tsim, dt, x0phy, pphy)

##fish model
#state variables
x0fish = {'Mfish': 80, #[gDM] 
      'Mdig':1E-6, #[gDM]
      'Muri':1E-6 #[gDM]
      } 

#parameters
pfish = {
    "tau_dig": 4.5,  # [h] time constants for digestive
    "tau_uri": 20,  # [h] time constants for urinary
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
    "Tmax": 32
}

#initialize object
fish = Fish(tsim, dt, x0fish, pfish)
flow = fish.f

#disturbance
DOvalues = 5
Tvalues = 26
dfish1 = {
    "DO": np.array([tsim, np.random.randint(1, 6, size=tsim.size)]).T,
    "T": np.array([t_weather, Tavg1]).T,
    "Mphy": np.array([tsim, np.full((tsim.size,), 8.586762e-6 / 0.02)]).T,
}
dphy = {
     'I0' :  np.array([tsim, I01]).T, #[J m-2 d-1] Hourly solar irradiation (6 AM to 6 PM)
     'T':np.array([tsim, Tavg1]).T,
     'DVS':np.array([tsim, np.linspace(0, 2.5, 120+1)]).T, #to simulate the information flows from rice growth
     'SNH4': np.array([tsim, np.full((tsim.size,), 0.1)]).T,
     'SNO2': np.array([tsim, np.full((tsim.size,), 0.1)]).T,
     'SNO3': np.array([tsim, np.full((tsim.size,), 0.1)]).T,
     'SP': np.array([tsim, np.full((tsim.size,), 0.1)]).T,
     'Rain': np.array([tsim, Rain1]).T
     }
#run model
tspan = (tsim[0], tsim[-1])
# run phyto model
yphy = phyto.run(tspan, d=dphy)

#retrieve phytoplankton growth for fish model input
#controllable input
ufish = {'Mfed': 80*0.03}
dfish1['Mphy'] = np.array([tsim, yphy['Mphy']]).T

#run fish model
yfish = fish.run(tspan, d=dfish1, u=ufish)

#retrieve results
t= yfish['t']
Mphy= phyto.y['Mphy']
NA = phyto.y['NA']
Mfish= fish.y['Mfish']
Muri= fish.y['Muri']
Mfis_fr= Mfish/pfish['k_DMR']
Mdig= fish.y['Mdig']

# Plot results
plt.figure(1)
plt.plot(t, Mfish, label='Fish dry weight')
plt.plot(t, Mdig, label='Fish digestive system weight')
plt.plot(t, Muri, label='Fish urinary system weight')
plt.plot(t, Mfis_fr, label='Fish fresh weight')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$biomass\ [g day-1]$')
plt.title('Fish biomass accumulation')

plt.figure(2)
plt.plot(t, flow['f_upt'], label='nutrient uptake flow')
plt.plot(t, flow['f_N_upt'], label='N uptake flow')
plt.plot(t, flow['f_P_upt'], label='P uptake flow')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$flow rate\ [g day-1]$')
plt.title('Nutrient uptake rate')

plt.figure(3)
plt.plot(t, flow['f_sol'], label='total soluble excretion')
plt.plot(t, flow['f_N_sol'], label='N content in soluble excretion')
plt.plot(t, flow['f_P_sol'], label='P content in soluble excretion')
plt.plot(t, flow['f_TAN'], label='soluble TAN')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$flow rate\ [g day-1]$')
plt.title('Soluble excretion rate')

plt.figure(4)
plt.plot(t, flow['f_fed'], label='total feed intake rate')
plt.plot(t, flow['f_prt'], label='total solid excretion')
plt.plot(t, flow['f_N_prt'], label='N content in solid excretion')
plt.plot(t, flow['f_P_prt'], label='P content in solid excretion rate')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$flow rate\ [g day-1]$')
plt.title('Particulate excretion rate')