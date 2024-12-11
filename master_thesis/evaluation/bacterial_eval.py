# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 17:05:49 2023

Test code for nutrient cycle subsystem
DB, AOB, NOB, PSB

@author: Angela
"""
from models.bacterialgrowth1 import Monod, DB, PSB, AOB
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

plt.style.use('ggplot')

# Simulation time array
t = 120
tsim = np.linspace(0.0, t, t+1) # [d]
dt = 1/5
tspan = (tsim[0], tsim[-1])
pond_volume = 4000*0.5

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

#t = 14 
x0DB = {'S':0.323404255,#Particulate Matter concentration 
        'X':0.2, #decomposer bacteria concentration
        'P': 0.08372093      
        }           #concentration in [g m-3]

#Organic fertilizer used: chicken manure
#N content = 1.23% N/kg fertilizers
#Chicken manure weight = 1000 kg (~500 kg/ha/week) (KangOmbe et al., 2006)
Norgf = (1.23/100)*1000 #N content in organic fertilizers 

uDB = {'Norgf': Norgf}
   
pDB= {
    'mu_max0': 3.8,  #maximum rate of substrate use (d-1) Kayombo 2003
    'Ks':5.14,         #half velocity constant (g m-3)
    'K_DO': 1,    #half velocity constant of DO for the bacteria 
    'Y': 0.82,      #bacteria synthesis yield (g bacteria/ g substrate used)
    'b20': 0.12,     # endogenous bacterial decay (d-1)
    'teta_b': 1.04, #temperature correction factor for b
    'teta_mu': 1.07, #temperature correction factor for mu
    'MrS': 60.07,           #[g/mol] Molecular Weight of urea
    'MrP': 18.05,           #[g/mol] Molecular Weight of NH4
    'a': 2,
    'kN': 0.5
    } 

dDB = {'DO': np.array([tsim, np.random.randint(1,6, size=tsim.size)]).T, # [mg O2 L-1] random number between 1 - 5 to simulate dissolved oxygen in the pond 
'T': np.array([tsim, Tavg]).T, #[°C] water temperature, assume similar with air temperature
     'f_N_prt': np.array([tsim, np.zeros(tsim.size)]).T,
     'f3':np.array([tsim, np.zeros(tsim.size)]).T
     }
#initialize object
db = DB(tsim, dt, x0DB, pDB)

##Ammonification by AOB 
x0AOB = {'S':0.08372093,   #Ammonium concentration 
         'X':0.07,   #AOB concentration
         'P': 0.305571392 #Nitrite concentration
         }          #concentration in [g m-3]

pAOB= {
   'mu_max0': 1, #range 0.33 - 1 g/g.d, if b = 0.17, mu_max_AOB = 0.9
   'Ks': 0.5, #range 0.14 - 5 g/m3, 0.6 - 3.6 g/m3, or 0.3 - 0.7 g/m3
   'K_DO': 1, #range 0.1 - 1 g/m3
   'Y': 0.33, #range 0.10 - 0.15 OR 0.33
   'b20': 0.17,#range 0.15 - 0.2 g/g.d
   'teta_mu': 1.072,
   'teta_b': 1.029,
   'MrS': 18.05,           #[g/mol] Molecular Weight of NH4
   'MrP': 46.01,           #[g/mol] Molecular Weight of NO2
   'a': 1
    } 

dAOB = {
     'DO': np.array([tsim, np.random.randint(1,6, size=tsim.size)]).T, # [mg O2 L-1] random number between 1 - 5 to simulate dissolved oxygen in the pond 
     'T': np.array([tsim, Tavg]).T, #[°C] water temperature, assume similar with air temperature
     'f_TAN': np.array([tsim, np.zeros(tsim.size)]).T,
     'SNH4': np.array([tsim, np.zeros(tsim.size)]).T
     }

#initialize object
aob = AOB(tsim, dt, x0AOB, pDB)

# #Nitrification by NOB
x0NOB = {'S':0.305571392,   #Nitrite concentration 
          'X':0.02,   #NOB concentration
          'P': 0.05326087 #Nitrate concentration
          }          #concentration in [g m-3]

pNOB= {
    'mu_max0': 0.7, #range 0.7 - 1.8 g/g.d
    'Ks': 0.2, #range 0.05 - 0.3 g/m3
    'K_DO': 0.5, #range 0.1 - 1 g/m3
    'Y': 0.08, #0.04 - 0.07 g VSS/g NO2 or 0.08 g VSS/g NO2 
    'b20': 0.17, 
    'teta_b': 1.029,
    'teta_mu': 1.063,
    'MrS': 46.01,           #[g/mol] Molecular Weight of NO2
    'MrP': 62.01,           #[g/mol] Molecular Weight of NO3
    'a': 1
    } 

r = 2 #r=[2, 2.5, 3]
pNOB['K_DO'] = r*pAOB['K_DO']

dNOB = {
     'DO': np.array([tsim, np.random.randint(1,6, size=tsim.size)]).T, # [mg O2 L-1] random number between 1 - 5 to simulate dissolved oxygen in the pond 
     'T': np.array([tsim, Tavg]).T, #[°C] water temperature, assume similar with air temperature
     'S_out': np.array([tsim, np.zeros(tsim.size)]).T
     }


# #instantiate Model
nob = Monod(tsim, dt, x0NOB, pDB)

##Phosphorus Solubilizing by PSB 
x0PSB = {'S':0.05781939,   #particulate P concentration (Mei et al 2023)
         'X':0.03,   #PSB concentration
         'P': 0.0327835051546391 #total Soluble Reactive Phosphorus/total P available for uptake (Mei et al 2023)
         }          #concentration in [g m-3]

dPSB = {'DO': np.array([tsim, np.random.randint(1,6, size=tsim.size)]).T, # [mg O2 L-1] random number between 1 - 5 to simulate dissolved oxygen in the pond 
        'T': np.array([tsim, Tavg]).T, #[°C] water temperature, assume similar with air temperature
     'f_P_prt': np.array([tsim, np.zeros(tsim.size)]).T,
     'f_P_sol': np.array([tsim, np.zeros(tsim.size)]).T,
     'f3':np.array([tsim, np.zeros(tsim.size)]).T
     }


Porgf = 1.39/100*1000
uPSB = {'Porgf': Porgf}

pPSB= {
    'mu_max0': 0.11,  #maximum rate of substrate use (d-1) 
    'Ks':5.14,         #half velocity constant (g m-3)
    'K_DO': 1,    #half velocity constant of DO for the bacteria 
    'Y': 0.45,      #bacteria synthesis yield (g bacteria/ g substrate used)
    'b20': 0.021,     # endogenous bacterial decay (d-1)
    'teta_b': 1.04, #temperature correction factor for b
    'teta_mu': 1.07, #temperature correction factor for mu
    'MrS': 647.94,           #[g/mol] Molecular Weight of C6H6O24P6
    'MrP': 98.00,           #[g/mol] Molecular Weight of H2PO4
    'a': 6,
    'kP': 0.5
    } 

#initialize object
psb = PSB(tsim, dt, x0PSB, pPSB)

# Run DB model
y1 = db.run(tspan,dDB, uDB)
# Retrieve DB model outputs for AOB model
dAOB['SNH4'] = np.array([y1['t'], y1['P']])
print(dAOB['SNH4'].shape)  
# Run AOB model    
y2 = aob.run(tspan,dAOB)

# Retrieve AOB model outputs for NOB model
dNOB['S_out']= np.array([y2['t'], y2['P']])

# #run NOB model
y3 = nob.run(tspan,dNOB)

#run PSB model
y4 = psb.run(tspan,dPSB, uPSB)

# Retrieve simulation results
t = db.t
S_N_prt = y1['S']
S_P_prt = y4['S']
S_NH4 = y1['P']
S_NO2 = y2['P']
S_NO3 = y3['P']
S_P = y4['P']
X_DB = y1['X']
X_AOB = y2['X']
X_NOB = y3['X']
X_PSB = y4['X']

# Create a figure and a grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle('Bacterial Growth, Substrate Utilization, and Substrate Production Rate')

# Plot results
axs[0, 0].plot(db.t, S_N_prt, label='Organic Matter Substrate')
axs[0, 0].plot(db.t, X_DB, label='Decomposer Bacteria')
axs[0, 0].plot(db.t, S_NH4, label='Ammonium')
axs[0, 0].legend()
axs[0, 0].set_xlabel(r'$time\ [d]$')
axs[0, 0].set_ylabel(r'concentration $[g m^{-3}]$')
axs[0, 0].set_title("Decomposition")

axs[0, 1].plot(aob.t, S_NH4, label='Ammonium')
axs[0, 1].plot(aob.t, X_AOB, label='AOB')
axs[0, 1].plot(aob.t, S_NO2, label='Nitrite')
axs[0, 1].legend()
axs[0, 1].set_xlabel(r'$time\ [d]$')
axs[0, 1].set_ylabel(r'concentration $[g m^{-3}]$')
axs[0, 1].set_title("Nitrite production rate")

axs[1, 0].plot(nob.t, S_NO2, label='Nitrite')
axs[1, 0].plot(nob.t, X_NOB, label='NOB')
axs[1, 0].plot(nob.t, S_NO3, label='Nitrate')
axs[1, 0].legend()
axs[1, 0].set_xlabel(r'$time\ [d]$')
axs[1, 0].set_ylabel(r'concentration $[g m^{-3}]$')
axs[1, 0].set_title("Nitrate production rate")

axs[1, 1].plot(psb.t, S_P_prt, label='Organic P')
axs[1, 1].plot(psb.t, X_PSB, label='PSB')
axs[1, 1].plot(psb.t, S_P, label='Phosphate')
axs[1, 1].legend()
axs[1, 1].set_xlabel(r'$time\ [d]$')
axs[1, 1].set_ylabel(r'concentration $[g m^{-3}]$')
axs[1, 1].set_title("Soluble Phosphorus production rate")


plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()