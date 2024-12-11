# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:44:51 2024

@author: alegn
"""

import numpy as np
import matplotlib.pyplot as plt
from models.phytoplankton import Phygrowth
import pandas as pd

plt.style.use('ggplot')

# Simulation time
t = 120 # [d]
tsim = np.linspace(0.0, t, t+1) # [d]
dt = 1 # [d] # 5 minutes according to Huesemann?

# Initial conditions
x0 = {'Mphy': 4.29e-4, #[g m-3] from Mei et al 2023 concentration of Chla multiply with Chla:phyto mass ratio from Jamu and Piedrahita (2002)
      'NA':   0.55 #[g m-3] TN from Mei et al 2023
      } 

p = {
      'mu_phy':  1.27, #[d-1] maximum growth rate of phytoplankton
      'mu_Up': 0.005, #[d-1] maximum nutrient uptake coefficient
      'pd': 0.5, #[m] pond depth
      'l_sl': 3.5e-7, #[m2 g-1] phytoplankton biomass-specific light attenuation
      'l_bg': 0.77, #[m-1] light attenuation by non-phytoplankton components
      'Kpp': 2.82e6, # [J m-2 d-1] half-saturation constant of phytoplankton production, converted from 300 microEinsteins (m2s)-1 to Jm-2d-1 with sunlight duration 12 hours
      'cm': 0.15, #[d-1] phytoplankton mortality rate
      'cl': 0.004, #[m2 g-1] phytoplankton crowding loss constant
      'c1': 1.57, # [-] temperature coefficients
      'c2': 0.24, # [-] temperature coefficients
      'Topt': 28, #optimum temperature for phytoplankton growth
      'Mp': 0.025, #[g m-3] half saturation constant for nutrient uptake
     }

# Disturbances
data_weather = 'C:/Users/alegn/Documents/WUR/Thesis/rfmodel/data/Weather/Magelang_200110_200209_Daily.csv'
weather = pd.read_csv(data_weather, header=0, sep=';')
weather['Time'] = pd.to_datetime(weather['Time'], format='%Y%m%d')  # Adjust the format if necessary
weather.set_index('Time', inplace=True)
#FIRST CYCLE
t_ini = '20011001'
t_end = '20020129'
Tavg = weather.loc[t_ini:t_end,'Tavg'].values #[°C] Mean daily temperature
Rain = weather.loc[t_ini:t_end,'Rain'].values #[mm] Daily precipitation
Igl = weather.loc[t_ini:t_end, 'I0'].values #[MJ m-2] Sum of shortwave radiation daily

I0 = 0.45*Igl*1E6 #Convert [MJ m-2 d-1] to [J m-2 d-1] PAR

d = {
     'I0' :  np.array([tsim, np.full((tsim.size,), I0)]).T, #[J m-2 h-1] Hourly solar irradiation (6 AM to 6 PM)
     'T':np.array([tsim, Tavg]).T,
     'DVS':np.array([tsim, np.linspace(0, 2.5, 120+1)]).T, #to simulate the information flows from rice growth
     'SNO3': np.array([tsim, np.full((tsim.size,), 0.1)]).T,
     'SP': np.array([tsim, np.full((tsim.size,), 0.1)]).T,
     'Rain': np.array([tsim, Rain]).T,
     'd_pond': np.array([tsim, np.full((tsim.size,), 0.6)]).T,
     }

# Initialize module
phyto = Phygrowth(tsim, dt, x0, p)

# Run simulation
# TODO: Call the method run to generate simulation results
tspan = (tsim[0], tsim[-1])
y = phyto.run(tspan, d)


# Retrieve simulation results
# TODO: Retrieve the simulation results
t = y['t'] 

# 1 mg L-1 = 1 g m-3
# to get the value of Mphy and NA in g, multiply it with pond_volume
#pond_area = 4000 m^2
Mphy = y['Mphy']
NA = y['NA']

f1 = phyto.f['f1']
f2 = phyto.f['f2']
f3 = phyto.f['f3']
f4 = phyto.f['f4']
f5 = phyto.f['f5']
f6 = phyto.f['f6']

# Plot
plt.figure(1)
plt.plot(t, Mphy, label='Mphy')
plt.plot(t, NA, label='NA')
plt.xlabel(r'time [d]')
plt.ylabel(r'Growth $[g]$')
# plt.title(r'Accumulative Phytoplankton Growth and Nutrient Availability')
plt.legend()

plt.figure(4)
plt.plot(t, f1, label='phytoplankton growth (f1)')
plt.plot(t, -f2, label= 'natural mortality rate (f2)')
plt.plot(t, -f3, label = 'death by intraspecific competition rate (f3)')
plt.xlabel(r'time [d]')
plt.ylabel(r'Flow rate $[g d^{-1}]$')
# plt.title(r'Flow in Phytoplankton Growth')
plt.legend()

plt.figure(5)
plt.plot(t, f4, label='Nitrate available in pond (f4)')
plt.plot(t, f5, label= 'Dihydrogen phosphate available in pond (f5)')
plt.plot(t, -f6, label = 'Nutrient uptake by phytoplankton (f6)')
plt.xlabel(r'time [d]')
plt.ylabel(r'Flow rate $[g d^{-1}]$')
# plt.title(r'Flow in Phytoplankton Growth')
plt.legend()

