# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 16:21:34 2023

@author: Angela
"""
import numpy as np
import matplotlib.pyplot as plt

from models.quefts_rice import QUEFTS

plt.style.use('ggplot')

# Define the simulation time array and integration time step
tsim = np.linspace(0, 120, 120+1)
dt = 1

# Initial conditions
# TODO: define the dictionary of initial conditions
x0 = {  'SN':0,    #[kg N/ha] Supplies of crop-available Nitrogen
        'SP':0,    #[kg P/ha] Supplies of crop-available Phosphorus
        'SK':0,    #[kg K/ha] Supplies of crop-available Potassium
        'YU':0    #[kg/ha] Ultimate Yield
      }

# Define the dictonary of parameter values
p = {
         'alpha_N': 6.8, #[-] empirical parameters for Nitrogen
         'alpha_P': 0.014, #[-] empirical parameters for Phosphorus
         'alpha_K': 400, #[-] empirical parameters for Potassium
         'beta_P': 0.5, #[-] empirical parameters for Phosphorus
         'beta_K': 0.9, #[-] empirical parameters for Potassium
         'f_F':1, #[-] flooding factor
         'gamma_K': 2, #[-] empirical parameters for Potassium
         'K_exc': 30,  #[mmol/kg] soil exchangeable Potassium
         'C_org': 70,  #[g/kg] soil organic carbon
         'P_Olsen': 30,#[mg/kg] Soil phosphorus (Olsen P)
         'P_tot': 25*70,  #[mg/kg] Total soil phosphorus
         'R_N': 0.5,    #[-] N maximum recovery fraction
         'R_P': 0.1,   #[-] P maximum recovery fraction
         'R_K': 0.5,    #[-] K maximum recovery fraction
         'rN': 5,    #[kg/ha] minimum N uptake to produce any grain
         'rP': 0.4,    #[kg/ha] minimum P uptake to produce any grain
         'rK': 2,    #[kg/ha] minimum K uptake to produce any grain
         'aN': 30,    #[kg/kg] Physiological efficiency Nitrogen at maximum accumulation of N
         'aP': 200,   #[kg/kg] Physiological efficiency Phosphorus at maximum accumulation of P
         'aK': 30,    #[kg/kg] Physiological efficiency Potassium at maximum accumulation of K
         'dN': 70,    #[kg/kg] Physiological efficiency Nitrogen at maximum dilution of N
         'dP': 600,    #[kg/kg] Physiological efficiency Phosphorus at maximum dilution of P
         'dK': 120,    #[kg/kg] Physiological efficiency Potassium at maximum dilution of K
         'Ymax': 10000  #[kg/ha] potential maize grain yield, at 12% moisture 
     }

# Disturbances (assumed constant for test)
# pH values: 4.5 - 7.4
# TODO: Define sensible constant values for the disturbances
# pHvalues = np.random.uniform(4.5, 7.4, len(tsim))
d = {
     # 'pH':np.array([tsim, np.full((tsim.size,),pHvalues)]).T
     'pH':7
     }

# Controlled inputs
u = {'I_N':0,     #[kg/ha] N fertilizer application
     'I_P':0,   #[kg/ha] P fertilizer application
     'I_K':0    #[kg/ha] K fertilizer application
         }

# Initialize module
rice = QUEFTS(tsim, dt, x0, p)

# Run simulation
tspan = (tsim[0],tsim[-1])
y_rice = rice.run(tspan, d, u)

# Retrieve simulation results
# TODO: Retrieve the arrays from the dictionary of model outputs.
t_rice = y_rice['t']
SN = y_rice['SN'] 
SP = y_rice['SP']
SK = y_rice['SK']
YU = y_rice['YU']

# Plots
plt.figure(1)
plt.plot(t_rice, SN, 'r', label='SN')
plt.plot(t_rice, SP, 'b', label='SP')
plt.plot(t_rice, SK, 'g', label='SK')
plt.xlabel(r'time [d]')
plt.ylabel(r'Nutrient Supply [kg/ha]')
plt.title(r'Nutrient Supply NPK')
plt.legend()

plt.figure(2)
plt.plot(t_rice, YU, 'r', label = 'YU ef')
plt.xlabel(r'time [d]')
plt.ylabel(r'Grain Yield [kg/ha]')
plt.title(r'Rice Grain Yield')
plt.legend()

plt.show()
