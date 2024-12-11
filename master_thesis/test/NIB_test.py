# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 17:05:49 2023

@author: Angela
"""
from models.bacterialgrowth import NIB
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# Simulation time array
tsim = np.linspace(0.0, 24, 24+1) # [d]
dt = 1
tspan = (tsim[0], tsim[-1])

#state variables
x0 = {'S_NH4':5.0,
      'S_NO2':5.0,
      'S_NO3': 2.0,
      'X_AOB': 1.5,
      'X_NOB': 1.5
     } 

#parameters
p= {
    'mu_max_AOB_20': 0.9, #range 0.33 - 1 g/g.d, if b = 0.17, mu_max = 0.9
    'mu_max_NOB_20': 1.33, #range 0.7 - 1.8 g/g.d
    'K_NH4': 5, #range 0.14 - 5 g/m3, 0.6 - 3.6 g/m3, or 0.3 - 0.7 g/m3
    'K_NO2': 0.3, #range 0.05 - 0.3 g/m3
    'K_DO_AOB': 0.5, #range 0.1 - 1 g/m3
    'r': 1,  #r=[2, 2.5, 3]
    'Y_AOB': 10.34,
    'Y_NOB': 10.34, 
    'b_AOB_20': 0.17,#range 0.15 - 0.2 g/g.d
    'b_NOB_20': 0.17, 
    'teta_AOB_mu': 1.072,
    'teta_b': 1.029,
    'teta_NOB_mu': 1.06,
        } 

#Disturbances
d = {'DO':np.array([tsim, np.full((tsim.size,),3)]).T,
     'T':np.array([tsim, np.full((tsim.size,),20)]).T
     }

#initialize object
monod = NIB(tsim, dt, x0, p)

#run model
tspan = (tsim[0], tsim[-1])
y = monod.run(tspan, d=d)

#retrieve results
t= y['t']
S_NH4= y['S_NH4']
S_NO2= y['S_NO2']
S_NO3= y['S_NO3']
X_AOB= y['X_AOB']
X_NOB= y['X_NOB']

# Plot results
plt.figure(1)
plt.plot(t, S_NH4, label='Ammonium')
plt.plot(t, S_NO2, label='Nitrite')
plt.plot(t, S_NO3, label='Nitrate')
plt.plot(t, X_AOB, label='AOB')
plt.plot(t, X_NOB, label='NOB')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$concentration\ [mg L-1]$')

plt.figure(2)
plt.plot(t, S_NH4, label='Ammonium')
plt.plot(t, S_NO2, label='Nitrite')
plt.plot(t, S_NO3, label='Nitrate')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$concentration\ [mg L-1]$')

plt.figure(3)
plt.plot(t, X_AOB, label='AOB')
plt.plot(t, X_NOB, label='NOB')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$concentration\ [mg L-1]$')

plt.figure(4)
plt.plot(t, S_NH4, label='NH4 concentration')
plt.plot(t, X_AOB, label='AOB')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$concentration\ [mg L-1]$')

plt.figure(5)
plt.plot(t, S_NO2, label='NO2 concentration')
plt.plot(t, X_NOB, label='NOB')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$concentration\ [mg L-1]$')

