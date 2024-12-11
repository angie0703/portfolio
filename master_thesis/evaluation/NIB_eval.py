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
tsim = np.linspace(0.0, 48, 48+1) # [d]
dt = 0.1
tspan = (tsim[0], tsim[-1])
x0 = {'S_NH4':2.0,
      'S_NO2':2.0,
      'S_NO3': 2.0,
      'X_AOB': 1.5,
      'X_NOB': 1.5
     } 
p= {
    'mu_max_AOB': 1.15, #maximum rate of substrate use (d-1) 
    'mu_max_NOB': 1.15,
    'K_NH4': 5.14,
    'K_NO2': 5.14,
    'K_DO_AOB': 5.14,
    'K_DO_NOB': 5.14,
    'Y_AOB': 10.34,
    'Y_NOB': 10.34, #bacteria yield (mg bacteria/ mg substrate)
    'b_AOB': 0.021,
    'b_NOB': 0.021 # endogenous bacterial decay (d-1)
    } 

#Modified parameters
p['mu_max_AOB']= 0.76 #maximum rate of substrate use (d-1) 
p['mu_max_NOB']= 1.33
p['K_NH4']= 3.6
p['K_NO2']= 0.3
p['K_DO_AOB']= 2.36
p['K_DO_NOB']= 1.4
p['b_AOB']= 0.65
p['b_NOB']= 0.65 # endogenous bacterial decay (d-1)

#Disturbances
d = {'DO':np.array([tsim, np.full((tsim.size,),3)]).T
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
plt.plot(t, X_AOB, label='NOB')
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
plt.plot(t, X_AOB, label='NOB')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$concentration\ [mg L-1]$')


