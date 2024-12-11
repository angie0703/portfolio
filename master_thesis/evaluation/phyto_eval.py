# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:44:51 2024

@author: alegn
"""

import numpy as np
import matplotlib.pyplot as plt
from models.phytoplankton import Phyto
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

dphy = {
     'I0' :  np.array([tsim, np.full((tsim.size,), I0)]).T, #[J m-2 h-1] Hourly solar irradiation (6 AM to 6 PM)
     'T':np.array([tsim, Tavg]).T,
     'N_net': np.array([tsim, np.full((tsim.size,), 0.5)]).T,
     'P_net': np.array([tsim, np.full((tsim.size,), 0.5)]).T,
     'Rain': np.array([tsim, Rain]).T
     }

# Initialize module
phyto = Phyto(tsim, dt, x0phy, pphy)

# Run simulation
# TODO: Call the method run to generate simulation results
tspan = (tsim[0], tsim[-1])
y = phyto.run(tspan, dphy)


# Retrieve simulation results
# TODO: Retrieve the simulation results
t = y['t'] 

# 1 mg L-1 = 1 g m-3
# to get the value of Mphy and NA in g, multiply it with pond_volume
#pond_area = 4000 m^2
Mphy = y['Mphy']

f_phy_grw = phyto.f['f_phy_grw']
f_phy_prd = phyto.f['f_phy_prd']
f_phy_cmp = phyto.f['f_phy_cmp']

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
# Plot the first subplot
ax1.plot(t, Mphy, label = '$M_{phy}$')
ax1.set_ylabel('concentration [$g m^{-3}$]')
ax1.legend()

# Plot the second subplot
ax2.plot(t, f_phy_grw, label ='$\phi_{phy,grw}$')
ax2.plot(t, -f_phy_prd, label ='$\phi_{phy,prd}$')
ax2.plot(t, -f_phy_cmp, label ='$\phi_{phy,cmp}$')
ax2.set_ylabel(r'rate $[g m^{-3} d^{-1}]$')
ax2.set_xlabel('time [day]')
ax2.legend()
