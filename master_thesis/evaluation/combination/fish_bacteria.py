# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 09:39:51 2024

Evaluation for connected model Fish and Bacteria
@author: Angela
"""

from models.fish import Fish
from models.bacterialgrowth import Monod
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# Simulation time array
t = 24
tsim = np.linspace(0.0, t, t+1) # [d]
dt = 15/60 #15 minutes 
tspan = (tsim[0], tsim[-1])

#disturbance
DOvalues = 5
Tvalues = 26
d = {'DO':np.array([tsim, np.full((tsim.size,),DOvalues)]).T,
     'T':np.array([tsim, np.full((tsim.size,),Tvalues)]).T,
     'Mphy':np.array([tsim, np.full((tsim.size,), 5)]).T
     }

##fish model
#state variables
x0fish = {'Mfish': 80, #[gDM] 
      'Mdig':1E-6, #[gDM]
      'Muri':1E-6 #[gDM]
      } #concentration in [mg L-1]

#parameters
pfish= {
    'tau_dig': 4.5,  # [h] time constants for digestive
    'tau_uri':20,  # [h] time constants for urinary
    'k_upt':0.3,  # [-] fraction of nutrient uptake for fish weight
    'k_N_upt':0.40,  # [-] fraction of N uptake by fish
    'k_P_upt':0.27,  # [-] fraction of P uptake by fish
    'k_prt':0.2,  # [-] fraction of particulate matter excreted
    'k_N_prt':0.05,  # [-] fraction of N particulate matter excreted
    'k_P_prt':0.27,  # [-] fraction of P particulate matter excreted
    'x_N_fed': 0.05, #[-] fraction of N in feed
    'x_P_fed': 0.01, #[-] fraction of P in feed
    'k_DMR': 0.31,
    'k_TAN_sol': 0.15,
    'Chla':0.02, #[g Chla/g phytoplankton] Phytoplankton biomass density 
    'Ksp':1,  #[mg C L-1] Half-saturation constant for phytoplankton
    'r': 0.02,
    'Tmin': 15,
    'Topt': 25,
    'Tmax': 35,
    'n_fish': 60
    }

#controllable input
ufish = {'Mfed': 80*0.03}

#initialize object
fish = Fish(tsim, dt, x0fish, pfish)
flow = fish.f

##bacteria model
##Decomposer Bacteria
x0DB = {'S':5,      #Particulate Matter concentration 
        'X':2.5,
        'P': 0      #decomposer bacteria concentration
        }           #concentration in [g L-1]

Norgf = 5 #concentration of Nitrogen content in Organic fertilizer
x0DB['S'] = x0DB['S'] + Norgf

pDB= {
    'mu_max0': 0.11,  #maximum rate of substrate use (d-1) 
    'Ks':0.00514,         #half velocity constant (g L-1)
    'K_DO': 1,    #half velocity constant of DO for the bacteria 
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
         'P': 1
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
          'P': 1 
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

##Phosphorus Solubilizing by PSB 
x0PSB = {'S':2.0,   #Ammonium concentration 
         'X':1.5,   #AOB concentration
         'P': 1
         }          #concentration in [mg L-1]

Porgf = 1 #[kg P] P content in organic fertilizer
x0PSB['S'] = x0PSB['S']+ Porgf 

pPSB= {
    'mu_max0': 0.02,  #maximum rate of substrate use (d-1) 
    'Ks':0.00514,         #half velocity constant (mg L-1)
    'K_DO': 1,    #half velocity constant of DO for the bacteria 
    'Y': 0.45,      #bacteria synthesis yield (mg bacteria/ mg substrate used)
    'b20': 0.2,     # endogenous bacterial decay (d-1)
    'teta_b': 1.04, #temperature correction factor for b
    'teta_mu': 1.07, #temperature correction factor for mu
    'MrS': 647.94,           #[g/mol] Molecular Weight of C6H6O24P6
    'MrP': 98.00,           #[g/mol] Molecular Weight of H2PO4
    'a': 6
    } 

#initialize object
psb = Monod(tsim, dt, x0PSB, pPSB)

#run fish model
yfish = fish.run(tspan, d=d, u=ufish)

f_P_prt = fish.f['f_P_prt']
f_N_prt = fish.f['f_N_prt']
f_TAN = fish.f['f_TAN']
f_P_sol = fish.f['f_P_sol']

# Retrieve fish model outputs for DB model
x0DB['S']= f_N_prt

#run DB model for N organic fertilizer
# Run DB model
y1 = db.run(tspan,d)
S_NH4 = y1['P']
# Retrieve DB model outputs for AOB model, Retrieve fish model outputs for AOB model
# Substrate of AOB = product of decomposition + TAN flow rate from fish
x0AOB['S'] = np.array(S_NH4)+f_TAN
  
# Run AOB model    
y2 = aob.run(tspan,d)
# Retrieve AOB model outputs for NOB model
# Substrate of AOB = nitrite from the AOB
x0NOB['S']= np.array(y2['P'])
# #run NOB model
y3 = nob.run(tspan,d)

# Retrieve fish model outputs for PSB model
x0PSB['S']= x0PSB['S']+ Porgf + f_P_prt

# #run PSB model
y4 = psb.run(tspan,d)

# Retrieve simulation results
t = db.t
S_N_prt = db.y['S']
S_P_prt = psb.y['S']
S_NO2 = aob.y['P']
S_NO3 = nob.y['P']
S_P = psb.y['P']+f_P_sol
X_DB = db.y['X']
X_AOB = aob.y['X']
X_NOB = nob.y['X']
X_PSB = psb.y['X']
Mfish= fish.y['Mfish']
Muri= fish.y['Muri']
Mfis_fr= Mfish/pfish['k_DMR']
Mdig= fish.y['Mdig']

#retrieve flows

# Plot results
fig1, axes1 = plt.subplots(nrows=2, ncols=2)
axes1[0,0].plot(t, Mfish, label='Fish dry weight')
axes1[0,0].plot(t, flow['f_fed'], label='total feed intake rate')
axes1[0,0].plot(t, Mfis_fr, label='Fish fresh weight')
axes1[0,0].legend()
axes1[0,0].set_xlabel(r'$time\ [d]$')
axes1[0,0].set_ylabel(r'$biomass\ [g day^{-1}]$')

axes1[0,1].plot(t, S_NH4, label='Ammonium production rate')
axes1[0,1].plot(t, flow['f_P_sol'], label='P content in soluble excretion')
axes1[0,1].plot(t, flow['f_TAN'], label='soluble TAN')
axes1[0,1].legend()
axes1[0,1].set_xlabel(r'$time\ [d]$')
axes1[0,1].set_ylabel(r'$flow rate\ [mg day^{-1}]$')

axes1[1,0].plot(t, flow['f_N_sol'], label='N content in soluble excretion')
axes1[1,0].plot(t, flow['f_P_sol'], label='P content in soluble excretion')
axes1[1,0].plot(t, flow['f_TAN'], label='soluble TAN')
axes1[1,0].legend()
axes1[1,0].set_xlabel(r'$time\ [d]$')
axes1[1,0].set_ylabel(r'$flow rate\ [mg day^{-1}]$')

axes1[1,1].plot(t, flow['f_fed'], label='total feed intake rate')
axes1[1,1].plot(t, flow['f_prt'], label='total solid excretion')
axes1[1,1].plot(t, flow['f_N_prt'], label='N content in solid excretion')
axes1[1,1].plot(t, flow['f_P_prt'], label='P content in solid excretion rate')
axes1[1,1].legend()
axes1[1,1].set_xlabel(r'$time\ [d]$')
axes1[1,1].set_ylabel(r'$flow rate\ [mg day^{-1}]$')


fig2, axes2 = plt.subplots(nrows=2, ncols=2)
axes2[0,0].plot(t, S_N_prt, label='Organic Matter Substrate')
axes2[0,0].plot(t, X_DB, label='Decomposer Bacteria')
axes2[0,0].plot(t, S_NH4, label='Ammonium')
axes2[0,0].legend()
axes2[0,0].set_xlabel(r'$time\ [d]$')
axes2[0,0].set_ylabel(r'$concentration\ [mg L^{-1}]$')
axes2[0,0].set_title("Decomposition")

axes2[0,1].plot(t, S_NH4, label='Ammonium')
axes2[0,1].plot(t, X_AOB, label='AOB')
axes2[0,1].plot(t, S_NO2, label='Nitrite')
axes2[0,1].legend()
axes2[0,1].set_xlabel(r'$time\ [d]$')
axes2[0,1].set_ylabel(r'$concentration\ [mg L^{-1}]$')
axes2[0,1].set_title("Nitrite production rate")

axes2[1,0].plot(t, S_NH4, label='Nitrite')
axes2[1,0].plot(t, X_AOB, label='NOB')
axes2[1,0].plot(t, S_NO2, label='Nitrate')
axes2[1,0].legend()
axes2[1,0].set_xlabel(r'$time\ [d]$')
axes2[1,0].set_ylabel(r'$concentration\ [mg L^{-1}]$')
axes2[1,0].set_title("Nitrate production rate")

axes2[1,1].plot(t, S_P_prt, label='Organic P')
axes2[1,1].plot(t, X_PSB, label='PSB')
axes2[1,1].plot(t, S_P, label='Phosphate')
axes2[1,1].legend()
axes2[1,1].set_xlabel(r'$time\ [d]$')
axes2[1,1].set_ylabel(r'$concentration\ [mg L^{-1}]$')
axes2[1,1].set_title("Soluble Phosphorus production rate")

