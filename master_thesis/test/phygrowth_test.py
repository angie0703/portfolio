# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:44:51 2024

@author: alegn
"""

import numpy as np
import matplotlib.pyplot as plt
from models.phytoplankton import Phygrowth

plt.style.use('ggplot')

# Simulation time
t = 100 # [d]
tsim = np.linspace(0.0, t, t+1) # [d]
dt = 1 # [d] # 5 minutes according to Huesemann?

# Initial conditions
x0 = {'Mphy': 8.586762e-6/0.02, #[g m-3] from Mei et al 2023 concentration of Chla multiply with Chla:phyto mass ratio from Jamu and Piedrahita (2002)
      'NA':   0.55 #[g m-3] TN from Mei et al 2023
      } 

p = {
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

# Disturbances
d = {
     'I0' : np.array([tsim, np.full((tsim.size,), 288)]).T, #[J m-2 h-1] Hourly solar irradiation (6 AM to 6 PM)
     # 'T':np.array([tsim, np.random.uniform(low=20, high=35, size=tsim.size,)]).T,
     'T':np.array([tsim, np.full((tsim.size,), 25)]).T,
     'DVS':np.array([tsim, np.full((tsim.size,), 0.4)]).T,
     'pd': np.array([tsim, np.full((tsim.size,), 0.55)]).T,
     'S_NH4': np.array([tsim, np.full((tsim.size,), 100)]).T,
     'S_NO2': np.array([tsim, np.full((tsim.size,), 100)]).T,
     'S_NO3': np.array([tsim, np.full((tsim.size,), 100)]).T,
     'Rain': np.array([tsim, np.full((tsim.size,), 100)]).T,
     'S_P': np.array([tsim, np.full((tsim.size,), 5)]).T
     # 'DVS': np.array([tsim, np.linspace(0,2, num = tsim.size)]).T
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
pond_volume = 4000*0.5 # [m3] 
# 1 mg L-1 = 1 g m-3
# to get the value of Mphy and NA in g, multiply it with pond_volume
Mphy = y['Mphy']*pond_volume
NA = y['NA']*pond_volume

f1 = phyto.f['f1']
f2 = phyto.f['f2']
f3 = phyto.f['f3']
f4 = phyto.f['f4']
f5 = phyto.f['f5']

# Plot
plt.figure(1)
plt.plot(t, Mphy, label='Phytoplankton')
plt.plot(t, NA, label='nutrient availability')
plt.xlabel(r'time [d]')
plt.ylabel(r'Accumulation rate [g d-1]')
plt.title(r'Accumulative Phytoplankton Growth and Nutrient Availability')
plt.legend()

plt.figure(2)
plt.plot(t, Mphy, label='Phytoplankton')
plt.xlabel(r'time [d]')
plt.ylabel(r'Phytoplankton growth [g d-1]')
plt.title(r'Accumulative Phytoplankton Growth')
plt.legend()

plt.figure(3)
plt.plot(t, NA, label='nutrient availability')
plt.xlabel(r'time [d]')
plt.ylabel(r'Nutrient availability [g d-1]')
plt.title(r'Accumulative nutrient availability')
plt.legend()

plt.figure(4)
plt.plot(t, f1, label='phyto growth')
plt.plot(t, f2, label= 'natural mortality rate')
plt.plot(t, f3, label = 'death by intraspecific competition rate')
plt.xlabel(r'time [d]')
plt.ylabel(r'Flow rate [g d-1]')
plt.title(r'Flow in Phytoplankton Growth')
plt.legend()

plt.figure(5)
plt.plot(t, f4, label='Nutrient available in Pond')
plt.plot(t, f5, label= 'Nutrient uptake by phytoplankton')
plt.xlabel(r'time [d]')
plt.ylabel(r'Flow rate [g d-1]')
plt.title(r'Flow in Phytoplankton Growth')
plt.legend()