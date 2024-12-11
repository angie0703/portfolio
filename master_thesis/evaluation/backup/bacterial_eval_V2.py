# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 17:05:49 2023

Test code for nutrient cycle subsystem
DB, AOB, NOB, PSB

@author: Angela
"""
from models.bacterialgrowth import NIB, PSB
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

plt.style.use('ggplot')

# Simulation time array
t = 120
tsim = np.linspace(0.0, t, t+1) # [d]
dt = 1/24
tspan = (tsim[0], tsim[-1])
pond_volume = 4000*0.5

# Disturbances
data_weather = 'C:/Users/alegn/Documents/WUR/Thesis/rfmodel/data/Weather/DIY_202210_202309_Daily.csv'
weather = pd.read_csv(data_weather, header=1, sep=';')

t_ini = '20221001'
t_end = '20230129'
weather['Time'] = pd.to_datetime(weather['Time'], format='%Y%m%d')  # Adjust the format if necessary
weather.set_index('Time', inplace=True)
Tavg = weather.loc[t_ini:t_end,'Tavg'].values #[°C] Mean daily temperature
Rain = weather.loc[t_ini:t_end,'Rain'].values #[mm] Daily precipitation
Igl = weather.loc[t_ini:t_end, 'I0'].values #[MJ m-2] Sum of shortwave radiation daily

I0 = 0.45*Igl*1E6 #Convert [MJ m-2 d-1] to [J m-2 d-1] PAR

#t = 14 
x0NIB = {'SNprt':0.323404255,#Particulate Matter concentration 
        'SNH4': 0.08372093,
        'SNO2':0.305571392,
         'XDB':0.2, #decomposer bacteria concentration
         'XAOB':0.07,   #AOB concentration
         'XNOB': 0.02,
         'P_NO3': 0.05326087
        }           #concentration in [g m-3]

#Organic fertilizer used: chicken manure
#N content = 1.23% N/kg fertilizers
#Chicken manure weight = 1000 kg (~500 kg/ha/week) (KangOmbe et al., 2006)
Norgf = (1.23/100)*1000 #N content in organic fertilizers 
uNIB = {'Norgf': Norgf}
   
pNIB= {
    'mu_max0DB': 3.8,  #maximum rate of substrate use (d-1) Kayombo 2003
    'mu_max0AOB': 0.33, #range 0.33 - 1 g/g.d, if b = 0.17, mu_max_AOB = 0.9
    'mu_max0NOB': 1, #range 0.7 - 1.8 g/g.d
    'KsDB':5.14,         #half velocity constant (g m-3)
    'KsAOB': 0.5, #range 0.14 - 5 g/m3, 0.6 - 3.6 g/m3, or 0.3 - 0.7 g/m3
    'KsNOB': 0.2, #range 0.05 - 0.3 g/m3
    'YDB': 0.82,      #bacteria synthesis yield (g bacteria/ g substrate used)
    'YAOB': 0.33, #range 0.10 - 0.15 OR 0.33
    'YNOB': 0.08, #0.04 - 0.07 g VSS/g NO2 or 0.08
    'b20DB': 0.12,     # endogenous bacterial decay (d-1)
    'b20AOB': 0.17,#range 0.15 - 0.2 g/g.d
    'b20NOB': 0.17, 
    'teta_bDB': 1.04, #temperature correction factor for b
    'teta_bAOB': 1.029,
    'teta_bNOB': 1.029,
    'teta_muDB': 1.07, #temperature correction factor for mu
    'teta_muAOB': 1.072,
    'teta_muNOB': 1.063,
    'kN': 0.5
    } 

dNIB = {'DO': np.array([tsim, np.random.randint(1,6, size=tsim.size)]).T, # [mg O2 L-1] random number between 1 - 5 to simulate dissolved oxygen in the pond 
        'T': np.array([tsim, Tavg]).T, #[°C] water temperature, assume similar with air temperature
        'f_N_prt': np.array([tsim, np.random.randint(1,6, size=tsim.size)]).T,
        'f_TAN': np.array([tsim, np.random.randint(1,6, size=tsim.size)]).T,
        'f3':np.array([tsim, np.random.randint(1,6, size=tsim.size)]).T
        }

#initialize object
nib = NIB(tsim, dt, x0NIB, pNIB)

##Phosphorus Solubilizing by PSB 
x0PSB = {'S':0.05781939,   #particulate P concentration (Mei et al 2023)
         'X':0.03,   #PSB concentration
         'P': 0.0327835051546391 #total Soluble Reactive Phosphorus/total P available for uptake (Mei et al 2023)
         }          #concentration in [g m-3]

dPSB = {'DO': np.array([tsim, np.random.randint(1,6, size=tsim.size)]).T, # [mg O2 L-1] random number between 1 - 5 to simulate dissolved oxygen in the pond 
        'T': np.array([tsim, Tavg]).T, #[°C] water temperature, assume similar with air temperature
     'f_P_prt': np.array([tsim, np.random.randint(1,6, size=tsim.size)]).T,
     'f3':np.array([tsim, np.random.randint(1,6, size=tsim.size)]).T
     }


Porgf = 1.39/100*1000
uPSB = {'Porgf': Porgf}

pPSB= {
    'mu_max0': 3.8,  #maximum rate of substrate use (d-1) 
    'Ks':5.14,         #half velocity constant (g m-3)
    'K_DO': 1,    #half velocity constant of DO for the bacteria 
    'Y': 0.45,      #bacteria synthesis yield (g bacteria/ g substrate used)
    'b20': 0.2,     # endogenous bacterial decay (d-1)
    'teta_b': 1.04, #temperature correction factor for b
    'teta_mu': 1.07, #temperature correction factor for mu
    'kP': 0.5
    } 

#initialize object
psb = PSB(tsim, dt, x0PSB, pPSB)

# Run DB model
y1 = nib.run(tspan,dNIB, uNIB)

#run PSB model
y2 = psb.run(tspan,dPSB, uPSB)

# Retrieve simulation results
t = y1['t']
S_N_prt = y1['SNprt']
S_P_prt = y2['S']
S_NH4 = y1['SNH4']
S_NO2 = y1['SNO2']
S_NO3 = y1['P_NO3']
S_P = y2['P']
X_DB = y1['XDB']
X_AOB = y1['XAOB']
X_NOB = y1['XNOB']
X_PSB = y2['X']

# Create a figure and a grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Plot results
axs[0, 0].plot(t, S_N_prt, label='Organic Matter Substrate')
axs[0, 0].plot(t, X_DB, label='Decomposer Bacteria')
axs[0, 0].plot(t, S_NH4, label='Ammonium')
axs[0, 0].legend()
axs[0, 0].set_xlabel(r'$time\ [d]$')
axs[0, 0].set_ylabel(r'$concentration\ [g m-3]$')
axs[0, 0].set_title("Decomposition")

axs[0, 1].plot(t, S_NH4, label='Ammonium')
axs[0, 1].plot(t, X_AOB, label='AOB')
axs[0, 1].plot(t, S_NO2, label='Nitrite')
axs[0, 1].legend()
axs[0, 1].set_xlabel(r'$time\ [d]$')
axs[0, 1].set_ylabel(r'$concentration\ [g m-3]$')
axs[0, 1].set_title("Nitrite production rate")

axs[1, 0].plot(t, S_NO2, label='Nitrite')
axs[1, 0].plot(t, X_NOB, label='NOB')
axs[1, 0].plot(t, S_NO3, label='Nitrate')
axs[1, 0].legend()
axs[1, 0].set_xlabel(r'$time\ [d]$')
axs[1, 0].set_ylabel(r'$concentration\ [g m-3]$')
axs[1, 0].set_title("Nitrate production rate")

axs[1, 1].plot(t, S_P_prt, label='Organic P')
axs[1, 1].plot(t, X_PSB, label='PSB')
axs[1, 1].plot(t, S_P, label='Phosphate')
axs[1, 1].legend()
axs[1, 1].set_xlabel(r'$time\ [d]$')
axs[1, 1].set_ylabel(r'$concentration\ [g m-3]$')
axs[1, 1].set_title("Soluble Phosphorus production rate")

plt.figure(5)
plt.plot(t, S_NH4, label= 'Ammonium')
plt.plot(t, S_NO2, label = 'Nitrite')
plt.plot(t, S_NO3, label = 'Nitrate')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$concentration\ [g m-3]$')
plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()