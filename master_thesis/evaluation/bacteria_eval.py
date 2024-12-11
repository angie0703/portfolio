# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 17:05:49 2023

@author: Angela
Evaluation for decomposer bacteria, AOB, and NOB model using modified Monod Kinetics

"""
from models.bacterialgrowth import Monod
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# Simulation time array
tsim = np.linspace(0.0, 24, 24+1) # [d]
dt = 0.1
tspan = (tsim[0], tsim[-1])

#disturbances
# DOvalues = np.random.uniform(0.3, 5, len(tsim))
# Tvalues = np.random.uniform(15, 25, len(tsim))
DOvalues = 5
Tvalues = 15
d = {'DO':np.array([tsim, np.full((tsim.size,),DOvalues)]).T,
     'T':np.array([tsim, np.full((tsim.size,),Tvalues)]).T,
     'S_out': np.array([tsim, np.full((tsim.size,), 0)]).T
     }

##Decomposer Bacteria
x0DB = {'S':5,      #Particulate Matter concentration 
        'X':2.5,    #decomposer bacteria concentration
        'P': 0      #Ammonium concentration
        }           #concentration in [g m-3]

pDB= {
    'mu_max0': 6,  #maximum rate of substrate use (d-1) 
    'Ks':8,         #half velocity constant (g m-3)
    'K_DO': 0.2,    #half velocity constant of DO for the bacteria 
    'Y': 0.45,      #bacteria synthesis yield (g bacteria/ g substrate used)
    'b20': 0.2,     # endogenous bacterial decay (d-1)
    'teta_b': 1.04, #temperature correction factor for b
    'teta_mu': 1.07, #temperature correction factor for mu
    'MrS': 60.07,           #[g/mol] Molecular Weight of urea
    'MrP': 18.05,           #[g/mol] Molecular Weight of NH4
    'a': 2
    } 

#initialize object
db = Monod(tsim, dt, x0DB, pDB)

##Ammonification by AOB 
x0AOB = {'S':2.0,   #Ammonium concentration 
         'X':1.5,   #AOB concentration
         'P':0      #Nitrite concentration
         }          #concentration in [mg L-1]

pAOB= {
   'mu_max0': 0.9, #range 0.33 - 1 g/g.d, if b = 0.17, mu_max_AOB = 0.9
   'Ks': 0.5, #range 0.14 - 5 g/m3, 0.6 - 3.6 g/m3, or 0.3 - 0.7 g/m3
   'K_DO': 0.5, #range 0.1 - 1 g/m3
   'Y': 0.15,
   'b20': 0.17,#range 0.15 - 0.2 g/g.d
   'teta_mu': 1.072,
   'teta_b': 1.029,
    'MrS': 18.05,           #[g/mol] Molecular Weight of NH4
    'MrP': 46.01,           #[g/mol] Molecular Weight of NO2
    'a': 1
    } 

#initialize object
aob = Monod(tsim, dt, x0AOB, pAOB)

# #Nitrification by NOB
x0NOB = {'S':2.0,   #Nitrite concentration 
          'X':1.5,   #NOB concentration
          'P':0      #Nitrate concentration
          }          #concentration in [mg L-1]

pNOB= {
    'mu_max0': 1, #range 0.7 - 1.8 g/g.d
    'Ks': 0.2, #range 0.05 - 0.3 g/m3
    'K_DO': 0.5, #range 0.1 - 1 g/m3
    'Y': 0.05, 
    'b20': 0.17, 
    'teta_b': 1.029,
    'teta_mu': 1.063,
    'MrS': 46.01,           #[g/mol] Molecular Weight of NO2
    'MrP': 62.01,           #[g/mol] Molecular Weight of NO3
    'a': 1
    } 

r = 2 #r=[2, 2.5, 3]
pNOB['K_DO'] = r*pAOB['K_DO']

# #instantiate Model
nob = Monod(tsim, dt, x0NOB, pNOB)

# Iterator
# (stop at second-to-last element, and store index in Fortran order)
it = np.nditer(tsim[:-1], flags=['f_index'])
for ti in it:
    # Index for current time instant
    idx = it.index
    # Integration span
    tspan = (tsim[idx], tsim[idx+1])
    print('Integrating', tspan)
    # Run DB model
    y1 = db.run(tspan,d)
    # Retrieve DB model outputs for AOB model
    x0AOB['S'] = np.array([y1['P']])
  
    # Run AOB model    
    y2 = aob.run(tspan,d)
    # Retrieve AOB model outputs for NOB model
    x0NOB['S']= np.array([y2['P']])
   
    #run NOB model
    y3 = nob.run(tspan,d)
    
# Retrieve simulation results
t = db.t
S_OM = db.y['S']
S_NH4 = db.y['P']
S_NO2 = aob.y['P']
S_NO3 = nob.y['P']
X_DB = db.y['X']
X_AOB = aob.y['X']
X_NOB = nob.y['X']

# Plot results
plt.figure(1)
plt.plot(t, S_OM, label='Organic Matter Substrate')
plt.plot(t, X_DB, label='Decomposer Bacteria')
plt.plot(t, S_NH4, label='P')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$concentration\ [mg L-1]$')
plt.title("Decomposition")

plt.figure(2)
plt.plot(t, S_NH4, label='Ammonium')
plt.plot(t, X_AOB, label='AOB')
plt.plot(t, S_NO2, label='Nitrite')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$concentration\ [mg L-1]$')
plt.title("Nitrite production rate")

plt.figure(3)
plt.plot(t, S_NO2, label='Nitrite Substrate Consumed')
plt.plot(t, X_NOB, label='NOB')
plt.plot(t, S_NO3, label='Nitrate')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$concentration\ [mg L-1]$')
plt.title("Nitrification")

plt.figure(4)
plt.plot(t, S_NO3, label='Nitrate')
plt.plot(t, S_NO2, label='Nitrite')
plt.plot(t, S_NH4, label = 'Ammonium')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$concentration\ [mg L-1]$')
plt.title("Product")