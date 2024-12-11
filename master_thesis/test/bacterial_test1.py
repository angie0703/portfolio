# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 17:05:49 2023

@author: Angela
"""
from models.bacterialgrowth import Monod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# Simulation time array
t = 130
tsim = np.linspace(0.0, t, t+1) # [d]
dt = 1/60
tspan = (tsim[0], tsim[-1])

# #data for interpolation
# data_bacteria = '../data/Bacteria/data_bacteria_tilapia.csv'

# # Total_P =pd.read_csv(data_bacteria, usecols=[0, 1], header=0, sep=';')
# # print(Total_P)
# # print('/')
# # print(Total_P.iloc[:,0])
# # print(Total_P.iloc[:,1])

# NH4 =pd.read_csv(data_bacteria, usecols=[0, 8], header=0, sep=';')
# print(NH4)
# print('/')

# NO2 =pd.read_csv(data_bacteria, usecols=[0, 10], header=0, sep=';')
# print(NO2)
# print('/')

# t_NH4 = NH4['Day']
# yNH4 = NH4['NH4 (g m-3)']
# yNO2 = NO2['NO2 (g m-3)']

# print('t:', t)
# print('yNH4: ', yNH4)
# print('yNO2: ', yNO2)

DOvalues = 5
Tvalues = 15
d = {'DO':np.array([tsim, np.full((tsim.size,),DOvalues)]).T,
     'T':np.array([tsim, np.full((tsim.size,),Tvalues)]).T
     }

x0AOB = {'S':0.08372093,   #Ammonium concentration 
         'X':0.05,   #AOB concentration
         'P': 0.305571392 #Nitrite concentration
         }          #concentration in [g m-3]

pAOB= {
   'mu_max0': 0.057, #range 0.33 - 1 g/g.d, if b = 0.17, mu_max_AOB = 0.9
   'Ks': 12.95, #range 0.14 - 5 g/m3, 0.6 - 3.6 g/m3, or 0.3 - 0.7 g/m3
   'K_DO': 0.17, #range 0.1 - 1 g/m3
   'Y': 0.336, #range 0.10 - 0.15 OR 0.33
   'b20': 0.02,#range 0.15 - 0.2 g/g.d
   'teta_mu': 1.09,
   'teta_b': 1.029,
   'MrS': 18.05,           #[g/mol] Molecular Weight of NH4
   'MrP': 46.01,           #[g/mol] Molecular Weight of NO2
   'a': 1
    } 


#initialize object
monod = Monod(tsim, dt, x0AOB, pAOB)

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

# plt.figure(2)
# plt.plot(t_NH4, yNH4, label = 'NH4', marker = 'o', linestyle = 'None')
# plt.plot(t_NH4, yNO2, label = 'NO2', marker = 'o', linestyle = 'None')
# plt.plot(t, S, label = 'Modelled NH4')
# plt.plot(t, P, label = 'Modelled NO2')
# plt.legend()
# plt.xlabel(r'$time\ [d]$')
# plt.ylabel(r'$concentration\ [g m-3]$')
# plt.title('Measured NH4 and NO2')
