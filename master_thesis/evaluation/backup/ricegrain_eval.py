# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:53:33 2024

@author: alegn
"""

import numpy as np
import matplotlib.pyplot as plt
from models.rice import Rice

plt.style.use('ggplot')

# Simulation time
t = 100
tsim = np.linspace(0.0, t, t+1) # [d]
dt = 1 # [d]

# Initial conditions
# TODO: define sensible values for the initial conditions
n_rice = 1
x0 = {'Mrt':0.05*n_rice,    #[kg DM ha-1 d-1] Dry biomass of root
      'Mst':0.05*n_rice,    #[kg DM ha-1 d-1] Dry biomass of stems
      'Mlv': 0.05*n_rice,   #[kg DM ha-1 d-1] Dry biomass of leaves
      'Mpa': 0.0,   #[kg DM ha-1 d-1] Dry biomass of panicles
      'Mgr': 0.0,   #[kg DM ha-1 d-1] Dry biomass of grains
      'HU': 0.0,
      'DVS': 0,
      'NumGrain':0
      } 

# Model parameters, as provided by Mohtar et al. (1997)
# TODO: define the parameter values in the dictionary p
p = {
     # 'DVSi' : 0.001,
     # 'DVRJ': 0.0013,
     # 'DVRI': 0.001125,
     # 'DVRP': 0.001275,
     # 'DVRR': 0.003,
     'DVSi' : 0.0008753,
     'DVRJ': 0.0008753,
     'DVRI': 0.0007576,
     'DVRP': 0.0007787,
     'DVRR': 0.0015390,
     'Tmax': 40,
     'Tmin': 15,
     'Topt': 33,
     'k': 0.4,
     'mc_rt': 0.01,
     'mc_st': 0.015,
     'mc_lv': 0.02,
     'mc_pa': 0.003,
     'N_lv': 0,
     'Rec_N': 0.5,
     'k_lv_N': 0,
     'k_pa_maxN': 0.0175,
     'M_upN': 8,
     'cr_lv': 1.326,
     'cr_st': 1.326,
     'cr_pa': 1.462,
     'cr_rt': 1.326,
     'IgN': 0.5,
     'gamma': 65      
     }

# Disturbances
# PAR [J m-2 d-1], env. temperature [°C], and CO2
# TODO: Specify the corresponding dates to read weather data (see csv file).
# Dictionary of disturbances (2D arrays, with col 1 for time, and col 2 for d)
d = {'I0':np.array([tsim, np.full((tsim.size,), 1000)]).T,
     # 'T':np.array([tsim, np.random.uniform(low=20, high=35, size=tsim.size,)]).T,
     'T': np.array([tsim, np.full((tsim.size,), 30)]).T,
     'CO2': np.array([tsim, np.full((tsim.size,), 400)]).T,
     'f_uptN': np.array([tsim, np.full((tsim.size,), 100)]).T
     }

# Controlled inputs
u = {'I_N':60}            # [kg N ha-1 d-1]

# Initialize module
# TODO: Call the module Grass to initialize an instance
rice = Rice(tsim, dt, x0, p)

# Run simulation
# TODO: Call the method run to generate simulation results
tspan = (tsim[0], tsim[-1])
y = rice.run(tspan, d, u)
f_Ph= rice.f['f_Ph']
f_res= rice.f['f_res']
f_gr = rice.f['f_gr']
f_dmv = rice.f['f_dmv']
f_pN = rice.f['f_pN']
f_Nlv = rice.f['f_Nlv']
HU = rice.f['HU']
DVS = rice.f['DVS']
N_lv = rice.f['N_lv']
f_spi = rice.f['f_spi']

# Retrieve simulation results
# 1 kg/ha = 0.0001 kg/m2
# 1 kg/ha = 0.1 g/m2
# TODO: Retrieve the simulation results
t = y['t']
Mrt = y['Mrt']
Mst = y['Mst']
Mlv = y['Mlv']
Mpa = y['Mpa']
Mgr = y['Mgr']
HU = y['HU']
DVS = y['DVS']
NumGrain = y['NumGrain']


# Plot
# # TODO: Make a plot for WsDM, WgDM and grass measurement data.
plt.figure(1)
plt.plot(t, Mrt, label='Roots')
plt.plot(t, Mst, label='Stems')
plt.plot(t, Mlv, label='Leaves')
# plt.plot(t, Mpa, label='Panicles')
plt.plot(t, Mgr, label='Grains')
plt.xlabel(r'time [d]')
plt.ylabel(r'Dry mass [kg DM ha-1 d-1]')
plt.title(r'Accumulative Dry Mass of Rice Crop Organs')
plt.legend()

plt.figure(2)
plt.plot(t, f_Ph, label='f_ph')
plt.plot(t, f_res, label='f_res')
plt.plot(t, f_gr, label='f_gr')
plt.plot(t, f_dmv, label='f_dmv')
plt.xlabel(r'time [d]')
plt.ylabel(r'flow rate [kg CH20 ha-1 d-1]')
plt.title(r'Flow rate in rice plants')
plt.legend()

plt.figure(3)
plt.plot(t, f_Nlv, label='N flow rate in leaves')
plt.plot(t, f_pN, label = 'N flow rate in rice plants')
plt.xlabel(r'time [d]')
plt.ylabel(r'flow rate [kg N ha-1 d-1]')
plt.title(r'Flow rate in rice plants')
plt.legend()

plt.figure(4)
plt.plot(t, Mpa, label='Panicles')
plt.plot(t, Mgr, label='Grains')
plt.xlabel(r'time [d]')
plt.ylabel(r'Dry mass [kg DM ha-1 d-1]')
plt.title(r'Accumulative Dry Mass of Rice Crop Organs')
plt.legend()

plt.figure(5)
plt.plot(t, NumGrain, label = 'Number of grains')
plt.plot(t, f_spi, label = 'Number of spikelets')
plt.xlabel(r'time [d]')
plt.ylabel(r'#')
plt.title(r'Number of spikelet and grains formed')
plt.legend()


plt.show()