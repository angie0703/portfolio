# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 17:13:56 2024

@author: alegn
"""

from models.bacterialgrowth import Monod
from models.phytoplankton import Phygrowth
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# Simulation time array
t = 120
tsim = np.linspace(0.0, t, t+1) # [d]
dt = 1
tspan = (tsim[0], tsim[-1])

#disturbances
# DOvalues = np.random.uniform(0.3, 5, len(tsim))
# Tvalues = np.random.uniform(15, 25, len(tsim))
DOvalues = 5
Tvalues = 15
d = {'DO':np.array([tsim, np.full((tsim.size,),DOvalues)]).T,
     'T':np.array([tsim, np.full((tsim.size,),Tvalues)]).T
     }

##Decomposer Bacteria
x0DB = {'S':5,      #Particulate Matter concentration 
        'X':2.5,
        'P': 0      #decomposer bacteria concentration
        }           #concentration in [g L-1]

Norgf = 5 #N content in organic fertilizers
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

Porgf = 1 #P content in organic fertilizers
x0PSB['S'] = x0PSB['S'] + Porgf

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


# Run DB model
y1 = db.run(tspan,d)
# Retrieve DB model outputs for AOB model
x0AOB['S'] = np.array([y1['t'], y1['P']])
  
# Run AOB model    
y2 = aob.run(tspan,d)
# Retrieve AOB model outputs for NOB model
x0NOB['S']= np.array([y2['t'], y2['P']])
# #run NOB model
y3 = nob.run(tspan,d)

# #run PSB model
y4 = psb.run(tspan,d)

##phytoplankton model
x0phy = {'Mphy': 5, #[g]
      'NA': 5  #[g]
      } 

# Model parameters, as provided by Mohtar et al. (1997)
# TODO: define the parameter values in the dictionary p
pphy = {
      'mu_phy':  1.27, #[d-1] maximum growth rate of phytoplankton
      'Iws':  5.42e-7, #[ ] light intensity at the water surface
      'mu_Up': 0.005, #[d-1] maximum nutrient uptake coefficient
      'pd': 0.5, #[m] pond depth
      'l_sl': 0.00035, #phytoplankton biomass-specific light attenuation
      'l_bg': 0.77, #light attenuation by non-phytoplankton components
      'Kpp': 3.62e-7, #half-saturation constant of phytoplankton production
      'cm': 0.15, #[-] phytoplankton mortality constant
      'cl': 0.004, #[-] phytoplankton crowding loss constant
      'c1': 1.57, # [-] temperature coefficients
      'c2': 0.24, # [-] temperature coefficients
      'Topt': 28, #optimum temperature for phytoplankton growth
      'Mp': 0.0025, #half saturation constant for nutrient uptake
      'a': 0.1, #dilution rate
     }

#initialize object
phyto = Phygrowth(tsim, dt, x0phy, pphy)
flows = phyto.f

#retrieve DB, AOB, AOB, and PSB outputs as phytoplankton inputs
x0phy['NA']= y1['P']+y2['P']+y3['P']+y4['P']

#Run phyto model 
yPhy = phyto.run(tspan, d)

#retrieve Phytoplankton death rate as DB inputs
x0DB['S'] = x0DB['S']+phyto.f['f3'] 

# Retrieve simulation results
t = db.t
S_N_prt = db.y['S']
S_P_prt = psb.y['S']
S_NH4 = db.y['P']
S_NO2 = aob.y['P']
S_NO3 = nob.y['P']
S_P = psb.y['P']
X_DB = db.y['X']
X_AOB = aob.y['X']
X_NOB = nob.y['X']
X_PSB = psb.y['X']
Mphy = phyto.y['Mphy']
NA = phyto.y['NA']

# S_N_prt = y1['S']
# S_P_prt = y4['S']
# S_NH4 = y1['P']
# S_NO2 = y2['P']
# S_NO3 = y3['P']
# S_P = y4['P']
# X_DB = y1['X']
# X_AOB = y2['X']
# X_NOB = y3['X']
# X_PSB = y4['X']
# Mphy = yPhy['Mphy']
# NA = yPhy['NA']

#flows
f1 = phyto.f['f1']
f2 = phyto.f['f2']
f3 = phyto.f['f3']
f4 = phyto.f['f4']
f5 = phyto.f['f5']

# # Plot results
# Plot figures 1-4 in one window
plt.figure(figsize=(12, 8))  # Adjust figure size as needed

plt.figure(1)
plt.plot(t, f1, label= 'f1')
plt.plot(t, f2, label = 'f2')
plt.plot(t, f3, label = 'f3')
plt.legend()

plt.figure(2)
plt.plot(t, f4, label = 'f4')
plt.plot(t, f5, label = 'f5')
plt.legend()

# Subplot 1
plt.subplot(2, 2, 1)
plt.plot(t, S_N_prt, label='Organic Matter Substrate')
plt.plot(t, X_DB, label='Decomposer Bacteria')
plt.plot(t, S_NH4, label='Ammonium')
plt.legend()
plt.xlabel(r'time [d]')
plt.ylabel(r'concentration [mg L$^{-1}$]')
plt.title("Decomposition")

# Subplot 2
plt.subplot(2, 2, 2)
plt.plot(t, S_NH4, label='Ammonium')
plt.plot(t, X_AOB, label='AOB')
plt.plot(t, S_NO2, label='Nitrite')
plt.legend()
plt.xlabel(r'time [d]')
plt.ylabel(r'concentration [mg L$^{-1}$]')
plt.title("Nitrite production rate")

# Subplot 3
plt.subplot(2, 2, 3)
plt.plot(t, S_NH4, label='Nitrite')
plt.plot(t, X_AOB, label='NOB')
plt.plot(t, S_NO2, label='Nitrate')
plt.legend()
plt.xlabel(r'time [d]')
plt.ylabel(r'concentration [mg L$^{-1}$]')
plt.title("Nitrate production rate")

# Subplot 4
plt.subplot(2, 2, 4)
plt.plot(t, S_P_prt, label='Organic P')
plt.plot(t, X_PSB, label='PSB')
plt.plot(t, S_P, label='Phosphate')
plt.legend()
plt.xlabel(r'time [d]')
plt.ylabel(r'concentration [mg L$^{-1}$]')
plt.title("Soluble Phosphorus production rate")

plt.tight_layout()  # Adjust subplot layout to prevent overlap
plt.show()

plt.figure(figsize=(12, 8))  # Adjust figure size as needed

# Subplot 1 (figure 5)
plt.subplot(3, 1, 1)
plt.plot(t, Mphy, label='Phytoplankton')
plt.xlabel(r'time [d]')
plt.ylabel(r'Phytoplankton growth [mg]')
plt.title(r'Accumulative Phytoplankton Growth')
plt.legend()

# Subplot 2 (figure 6)
plt.subplot(3, 1, 2)
plt.plot(t, NA, label='Nutrient availability')
plt.xlabel(r'time [d]')
plt.ylabel(r'Nutrient availability [mg]')
plt.title(r'Accumulative Nutrient Availability')
plt.legend()

# Subplot 3 (Additional plot)
plt.subplot(3, 1, 3)
plt.plot(t, Mphy, label='Phytoplankton growth')
plt.plot(t, NA, label='Nutrient Availability')
plt.xlabel(r'time [d]')
plt.ylabel(r'Concentration [mg]')
plt.title("Phytoplankton Growth and Nutrient Availability")
plt.legend()

plt.tight_layout()  # Adjust subplot layout to prevent overlap
plt.show()