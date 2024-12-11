# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 11:04:16 2024

@author: alegn
"""

import numpy as np
import matplotlib.pyplot as plt
from models.phytoplankton import Phyto
from models.nutrient import Nutrient
import pandas as pd

plt.style.use('ggplot')

# Simulation time
t = 120 # [d]
tsim = np.linspace(0.0, t, t+1) # [d]
dt = 1 # [d] # 5 minutes according to Huesemann?
A_sys = 1  # [ha] the total area of the system
A_rice = 0.6  # [ha] rice field area
A_pond = A_sys - A_rice  # [m2] pond area
V_pond = A_pond*0.6
V_phy = V_pond + A_rice*0.1

#%% Phytoplankton
# Initial conditions
x0phy = {'Mphy': 4.29e-4, #[g m-3] from Mei et al 2023 concentration of Chla multiply with Chla:phyto mass ratio from Jamu and Piedrahita (2002)
         } 

pphy = {
      'mu_phy':  1.27, #[d-1] maximum growth rate of phytoplankton
      'l_sl': 3.5e-7, #[m2 g-1] phytoplankton biomass-specific light attenuation
      'l_bg': 0.77, #[m-1] light attenuation by non-phytoplankton components
      'Kpp': 2232, # [J m-2 d-1] half-saturation constant of phytoplankton production
      'c_prd': 0.15, #[d-1] phytoplankton mortality rate
      'c_cmp': 0.004, #[m2 (g d)-1] phytoplankton crowding loss constant
      'c1': 1.57, # [-] temperature coefficients
      'c2': 0.24, # [-] temperature coefficients
      'Topt': 28, #optimum temperature for phytoplankton growth
      'K_N_phy': 0.1, #[g m-3] half saturation constant for N uptake (Prats & Llavador, 1994)
      'K_P_phy': 0.02, #[g m-3] half saturation constant for P uptake (Prats & Llavador, 1994)  
      'kNphy': 0.06, #[g N/g biomass] fraction of N from phytoplankton biomass
      'kPphy': 0.01, #[g P/g biomass] fraction of P from phytoplankton biomass
      'V_phy': V_phy
     }

#%% Nutrient
# Initial conditions
x0nut = {'M_N_net': 0.053*V_phy, #[g] (Mei et al. 2023)
         'M_P_net': 0.033*V_phy 
         } #(Mei et al. 2023)

pnut = {
      'mu_Up': 0.005, #[d-1] maximum nutrient uptake coefficient
      'l_sl': 3.5e-7, #[m2 g-1] phytoplankton biomass-specific light attenuation
      'l_bg': 0.77, #[m-1] light attenuation by non-phytoplankton components
      'Kpp': 2232, # [J m-2 d-1] half-saturation constant of phytoplankton production
      'c1': 1.57, # [-] temperature coefficients
      'c2': 0.24, # [-] temperature coefficients
      'Topt': 28, #optimum temperature for phytoplankton growth
      'K_N_phy': 0.1, #[g m-3] half saturation constant for N uptake (Prats & Llavador, 1994)
      'K_P_phy': 0.02, #[g m-3] half saturation constant for P uptake (Prats & Llavador, 1994)  
      'kNdecr': 0.05, #[d-1] decomposition rate (to replace bacteria decomposition rate) (Prats & Llavador, 1994)
      'kPdecr': 0.4, #[d-1] decomposition rate (to replace bacteria decomposition rate) (Prats & Llavador, 1994)    
      'f_N_edg': 0.8,
      'f_P_edg': 0.5,
      'MuptN': 8,
      'MuptP': 5,

     }

#%% Disturbances
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

dphy = {
     'I0' :  np.array([tsim, np.full((tsim.size,), I0)]).T, #[J m-2 h-1] Hourly solar irradiation (6 AM to 6 PM)
     'T':np.array([tsim, Tavg]).T,
     'N_net': np.array([tsim, np.full((tsim.size,), 0.5)]).T,
     'P_net': np.array([tsim, np.full((tsim.size,), 0.5)]).T,
     'Rain': np.array([tsim, Rain]).T
     }

dnut = {
     'I0' :  np.array([tsim, np.full((tsim.size,), I0)]).T, #[J m-2 h-1] Hourly solar irradiation (6 AM to 6 PM)
     'Tw':np.array([tsim, Tavg]).T,
     'Mphy':np.array([tsim, np.full((tsim.size,), 0.5)]).T,
     'f_N_phy_cmp': np.array([tsim, np.full((tsim.size,), 0.5)]).T,
     'f_P_phy_cmp': np.array([tsim, np.full((tsim.size,), 0.5)]).T,
     'f_N_sol': np.array([tsim, np.full((tsim.size,), 0.5)]).T,
     'f_P_sol': np.array([tsim, np.full((tsim.size,), 0.5)]).T,
     'f_N_prt': np.array([tsim, np.full((tsim.size,), 0.5)]).T,
     'f_P_prt': np.array([tsim, np.full((tsim.size,), 0.5)]).T,
     'Rain': np.array([tsim, Rain]).T
     }

#%% Input (u)
#inorganic fertilizer types and concentration used in Indonesia: 
#Nitrogen source: NPK (15%:15%:15%), Urea (46% N)
#NPK fertilizer recommended doses 167 kg/ha (Yassi 2023)
NPK_w = 167*A_rice
I_N1 = (15/100)*NPK_w

#Urea fertilizer recommended doses 100 kg/ha (Yassi 2023)
Urea = 100*A_rice
I_N2 = (46/100)*Urea

#Total N from inorganic fertilizer
I_N = I_N1 + I_N2

#Phosphorus source: SP36 (36%P2O5 ~ 7.85% P)
#P content in SP-36: 7.85% P
#SP fertilizer recommended doses 31 kg/ha (Yassi 2023)
SP36 = 31*A_rice
I_P = (7.85/100)*SP36

Norgf = 12.3
Porgf = 13.9

#%% Initialize module
phyto = Phyto(tsim, dt, x0phy, pphy)
flow_phy = phyto.f
nutrient = Nutrient(tsim, dt, x0nut, pnut)
flow_nut = nutrient.f

#for storing values
yphy_ = {'t': [0, ], 'Mphy': [4.29e-4, ], 'f_N_phy_cmp': [0, ], 'f_P_phy_cmp': [0, ]}
ynut_ = {'t': [0, ], 'M_N_net': [0.053*V_phy, ], 'M_P_net': [0.033*V_phy, ]}
dphy_ = {'N_net': [0, ], 'P_net': [0, ]}
dnut_ = {'Mphy': [4.29e-4, ], 'f_N_phy_cmp': [0, ], 'f_P_phy_cmp': [0, ]}

# Run simulation
# TODO: Call the method run to generate simulation results
#simulation for two different model look at grass-water model
tspan = (tsim[0], tsim[-1])

it = np.nditer(tsim[:-1], flags=['f_index'])
for ti in it:
    # Index for current time instant
    idx = it.index
    # Integration span
    tspan = (tsim[idx], tsim[idx+1])
    print('Integrating', tspan)
    # Controlled inputs
    unut = {'I_N':I_N, 'I_P': I_P, 'Norgf': Norgf, 'Porgf': Porgf}            # [kg d-1]

    # Run phytoplankton model
    yphy = phyto.run(tspan, dphy)
    yphy_['t'].append(yphy['t'][1])
    yphy_['Mphy'].append(yphy['Mphy'][1])
    yphy_['f_N_phy_cmp'].append(yphy['f_N_phy_cmp'][1])
    yphy_['f_P_phy_cmp'].append(yphy['f_P_phy_cmp'][1])
    
    # Retrieve phytoplankton model outputs for nutrient model
    dnut_['Mphy'].append(np.array([yphy['t'][1], yphy['Mphy'][1]]))
    dnut_['f_N_phy_cmp'].append(np.array([yphy['t'][1], yphy['f_N_phy_cmp'][1]]))
    dnut_['f_N_phy_cmp'].append(np.array([yphy['t'][1], yphy['f_P_phy_cmp'][1]]))
    dnut['Mphy'] = np.array([yphy['t'], yphy['Mphy']]).T
    dnut['f_N_phy_cmp'] = np.array([yphy['t'], yphy['f_N_phy_cmp']]).T
    dnut['f_P_phy_cmp'] = np.array([yphy['t'], yphy['f_P_phy_cmp']]).T
    
    # Run nutrient model    
    ynut = nutrient.run(tspan, dnut, unut)
    
    #Retrieve nutrient simulation result for phytoplankton
    dphy_['N_net'].append(np.array([ynut['t'][1], ynut['M_N_net'][1]]))
    dphy_['P_net'].append(np.array([ynut['t'][1], ynut['M_P_net'][1]]))
    dphy['N_net'] = np.array([ynut['t'], ynut['M_N_net']]).T
    dphy['P_net'] = np.array([ynut['t'], ynut['M_P_net']]).T
    
yphy = {key: np.array(value) for key, value in yphy_.items()} 
ynut = {key: np.array(value) for key, value in ynut_.items()}    

#%% Retrieve simulation results
# TODO: Retrieve the simulation results
t = yphy['t'] 

# 1 mg L-1 = 1 g m-3
# to get the value of Mphy and NA in g, multiply it with pond_volume
#pond_area = 4000 m^2
Mphy = yphy['Mphy']
