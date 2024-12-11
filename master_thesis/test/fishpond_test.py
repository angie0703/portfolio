# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 23:19:23 2024

@author: Angela
"""

from models.irff import fishpond
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Simulation time array
t = 120
tsim = np.linspace(0.0, t, t+1)
t_weather = np.linspace(0.0, t, t+1) # [d]
dt = 1/5
tspan = (tsim[0], tsim[-1])
pond_volume = 4000*0.5
N_fish = [14.56, 18.2, 21.84]  # [g m-3]
n_fish = [(N * pond_volume) / 18.2 for N in N_fish]

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

#State variables
x0NIB = {'SNprt':0.323404255,#Particulate Matter concentration 
        'SNH4': 0.08372093,
        'SNO2':0.305571392,
         'XDB':0.2, #decomposer bacteria concentration
         'XAOB':0.07,   #AOB concentration
         'XNOB': 0.02,
         'SNO3': 0.05326087
        }           #concentration in [g m-3]
x0PSB = {'S':0.05781939,   #particulate P concentration (Mei et al 2023)
         'X':0.03,   #PSB concentration
         'P': 0.0327835051546391 #total Soluble Reactive Phosphorus/total P available for uptake (Mei et al 2023)
         }          #concentration in [g m-3]
x0phy = {
    "Mphy": 8.586762e-6/0.02,  # [g m-3] from Mei et al 2023 concentration of Chla multiply with Chla:phyto mass ratio from Jamu and Piedrahita (2002)
    "NA": 0.55  # [g m-3] TN from Mei et al 2023
}


# n_fish = [4000, 6000, 8000]
x0fish = {
    "Mfish": 18.2 * n_fish[0],
    "Mdig": 1e-6 * n_fish[0],
    "Muri": 1e-6 * n_fish[0]
    }

#Create dictionary of state variables:
x0 = {
      'NIB': x0NIB,
      'PSB': x0PSB,
      'phy': x0phy,
      'fish': x0fish
      }

#Inputs
Norgf = (1.23/100)*1000 #N content in organic fertilizers 
uNIB = {'Norgf': Norgf}
Porgf = 1.39/100*1000
uPSB = {'Porgf': Porgf}
ufish = {"Mfed": 18.2 * 0.03 * n_fish[0]}

#create dictionary of inputs
u = {
     'NIB': uNIB,
     'PSB': uPSB,
     'fish': ufish
     }

#Parameters
pNIB= {
    'mu_max0DB': 3.8,  #maximum rate of substrate use (d-1) Kayombo 2003
    'mu_max0AOB': 1, #range 0.33 - 1 g/g.d, if b = 0.17, mu_max_AOB = 0.9
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
pphy = {
    "mu_phy": 1.27,  # [d-1] maximum growth rate of phytoplankton
    "mu_Up": 0.005,  # [d-1] maximum nutrient uptake coefficient
    "pd": 0.5,  # [m] pond depth
    "l_sl": 0.00035,  # [m2 mg-1] phytoplankton biomass-specific light attenuation
    "l_bg": 0.77,  # [m-1] light attenuation by non-phytoplankton components
    "Kpp": 0.234*60,  # [J m-2 s-1] half-saturation constant of phytoplankton production
    "cm": 0.15,  # [d-1] phytoplankton mortality constant
    "cl": 4e-6,  # [m2 mg-1] phytoplankton crowding loss constant
    "c1": 1.57,  # [-] temperature coefficients
    "c2": 0.24,  # [-] temperature coefficients
    "Topt": 28,  # optimum temperature for phytoplankton growth
    "Mp": 0.025,  # [g m-3] half saturation constant for nutrient uptake
    "dr": 0.21,  # [d-1] dilution rate (0.21 - 0.45)
}
pfish = {
    "tau_dig": 4.5,  # [h] time constants for digestive
    "tau_uri": 20,  # [h] time constants for urinary
    "k_upt": 0.3,  # [-] fraction of nutrient uptake for fish weight
    "k_N_upt": 0.40,  # [-] fraction of N uptake by fish
    "k_P_upt": 0.27,  # [-] fraction of P uptake by fish
    "k_prt": 0.2,  # [-] fraction of particulate matter excreted
    "k_N_prt": 0.05,  # [-] fraction of N particulate matter excreted
    "k_P_prt": 0.27,  # [-] fraction of P particulate matter excreted
    "x_N_fed": 0.05,  # [-] fraction of N in feed
    "x_P_fed": 0.01,  # [-] fraction of P in feed
    "k_DMR": 0.31,
    "k_TAN_sol": 0.15,
    "Ksp": 1,  # [g C m-3] Half-saturation constant for phytoplankton
    "Tmin": 22,
    "Topt": 28,
    "Tmax": 32,
}
#create dictionary of parameters
p = {
     'NIB': pNIB,
     'PSB': pPSB,
     'phy': pphy,
     'fish': pfish
     }

#Disturbances
dNIB = {'DO': np.array([tsim, np.random.randint(1,6, size=tsim.size)]).T, # [mg O2 L-1] random number between 1 - 5 to simulate dissolved oxygen in the pond 
        'T': np.array([tsim, Tavg]).T, #[°C] water temperature, assume similar with air temperature
        'f_N_prt': np.array([tsim, np.zeros(tsim.size)]).T,
        'f_TAN': np.array([tsim, np.zeros(tsim.size)]).T,
        'f3':np.array([tsim, np.zeros(tsim.size)]).T
        }

dPSB = {'DO': np.array([tsim, np.random.randint(1,6, size=tsim.size)]).T, # [mg O2 L-1] random number between 1 - 5 to simulate dissolved oxygen in the pond 
        'T': np.array([tsim, Tavg]).T, #[°C] water temperature, assume similar with air temperature
     'f_P_prt': np.array([tsim, np.zeros(tsim.size)]).T,
     'f3':np.array([tsim, np.zeros(tsim.size)]).T
     }

dphy = {
    "I0": np.array([t_weather, I0]).T,
    "T": np.array([t_weather, Tavg]).T,
    "Rain": np.array([t_weather, Rain]).T,
    "DVS": np.array([tsim, np.linspace(0, 2.5, tsim.size)]).T,
    "SNH4": np.array([tsim, np.full((tsim.size,), 0.08372093)]).T,
    "SNO2": np.array([tsim, np.full((tsim.size,), 0.305571392)]).T,
    "SNO3": np.array([tsim, np.full((tsim.size,), 0.05326087)]).T,
    "SP": np.array([tsim, np.full((tsim.size,), 0.0327835051546391)]).T,
}

dfish = {
    "DO": np.array([tsim, np.random.randint(1, 6, size=tsim.size)]).T,
    "T": np.array([t_weather, Tavg]).T,
    "Mphy": np.array([tsim, np.full((tsim.size,), 8.586762e-6 / 0.02)]).T,
}

d = {
     'NIB': dNIB,
     'PSB': dPSB,
     'phy': dphy,
     'fish': dfish
     }

# drice = {
#     "I0": np.array([t_weather, I0]).T,
#     "T": np.array([t_weather, Tavg]).T,
#     "CO2": np.array([tsim, np.full((tsim.size,), 400)]).T,
#     "S_NO3": np.array([tsim, np.full((tsim.size,), 0.05326087)]).T,
#     "S_P": np.array([tsim, np.full((tsim.size,), 0.0327835051546391)]).T,
#     "f_P_sol": np.array([tsim, np.full((tsim.size,), 0.0327835051546391)]).T,
# }

#Instantiate object:
fishpond = fishpond(tsim, dt, x0, p)

#Run model
y_fp = fishpond.run_simulation(tspan, d, u)

t = y_fp['Fish']['t']
# SNprt = y_fp['SNprt']
# SPprt = y_fp['SPprt']
# SNH4 = y_fp['SNH4']
# SNO2 = y_fp['SNO2']
SNO3 = y_fp['SNO3']
SP = y_fp['SP']
Mfish_fr = y_fp['Yield']/pfish['k_DMR']
Mphy = y_fp['Phyto']['Mphy']

# NA = y['NA']

plt.figure(1)
plt.plot(t, Mfish_fr, label = 'fresh fish weight')
plt.plot(t, Mphy, label = 'phytoplankton')
plt.xlabel('time (days)')
plt.ylabel('Biomass [g d-1]')
plt.legend()

plt.show()