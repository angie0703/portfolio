# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 17:05:49 2023

@author: Angela
"""
from models.bacterialgrowth import Monod
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# Simulation time array
t = 1
tsim = np.linspace(0.0, t, t+1) # [d]
dt = 1/60
tspan = (tsim[0], tsim[-1])

DOvalues = 5
Tvalues = 15
d = {'DO':np.array([tsim, np.full((tsim.size,),DOvalues)]).T,
     'T':np.array([tsim, np.full((tsim.size,),Tvalues)]).T
     }

x0NOB = {'S':0.305571392,   #Nitrite concentration 
         'X':0.255,   #NOB concentration
         'P': 0.05326087 #Nitrate concentration
          }          #concentration in [g m-3]

pNOB= {
    'mu_max0': 0.9, #range 0.7 - 1.8 g/g.d
    'Ks': 1.1, #range 0.05 - 0.3 g/m3
    'K_DO': 0.13, #range 0.1 - 1 g/m3
    'Y': 0.08, #0.04 - 0.07 g VSS/g NO2 or 0.08 g VSS/g NO2 
    'b20': 0.04,
    'teta_mu': 1.11,
    'teta_b': 1.029,
    'MrS': 46.01,           #[g/mol] Molecular Weight of NO2
    'MrP': 62.01,           #[g/mol] Molecular Weight of NO3
    'a': 1
    } 


#initialize object
monod = Monod(tsim, dt, x0NOB, pNOB)

#run model
tspan = (tsim[0], tsim[-1])
y = monod.run(tspan, d)

#retrieve results
t= y['t']
S= y['S']
X= y['X']
P = y['P']

# Plot results
plt.figure(1)
plt.plot(t, S, label='Substrate')
plt.plot(t, X, label='Bacteria')
# plt.plot(t, P, label='Product')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$concentration\ [g m-3]$')
