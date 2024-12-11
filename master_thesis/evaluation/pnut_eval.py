# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 11:04:16 2024

@author: alegn
"""

import numpy as np
import matplotlib.pyplot as plt
from models.nutrient import PNut
import pandas as pd

plt.style.use('ggplot')

# Simulation time
t = 120 # [d]
tsim = np.linspace(0.0, t, t+1) # [d]
dt = 1/24 # [d] # 5 minutes according to Huesemann?
A_sys = 10000 # [m2] the total area of the system
A_rice = 0.6  # [ha] rice field area
A_pond = A_sys - A_rice  # [m2] pond area
V_pond = A_pond*0.6 # 0.6 m is the pond depth
V_phy = V_pond + A_rice*0.1 

#%% Nutrient
# Initial conditions
x0_pn = {'M_N_net': 0.053*V_phy, #[g] Net phosphorus in the system (Mei et al. 2023)
         'M_P_net': 0.033*V_phy, #[g] Net phosphorus in the system (Mei et al. 2023)
         'Mphy': 4.29e-4*V_phy, #[g] phytoplankton mass in the system (Mei et al. 2023; Jamu & Prihardita, 2002)
         'Nrice': 0
         } #all multiplied with V_phy because all of the originally in [g/m3]

p_pn = {
      'mu_Up': 0.005, #[d-1] maximum nutrient uptake coefficient
      'mu_phy':  1.27, #[d-1] maximum growth rate of phytoplankton
      'l_sl': 3.5e-7, #[m2 g-1] phytoplankton biomass-specific light attenuation
      'l_bg': 0.77, #[m-1] light attenuation by non-phytoplankton components
      'Kpp': 2232, # [J m-2 d-1] half-saturation constant of phytoplankton production (for 12 hours of daylight per day)
      'c_prd': 0.15, #[d-1] phytoplankton mortality rate
      'c_cmp': 0.004, #[m3 (g d)-1] phytoplankton crowding loss constant
      'c1': 1.57, # [-] temperature coefficients
      'c2': 0.24, # [-] temperature coefficients
      'Topt': 28, #optimum temperature for phytoplankton growth
      'K_N_phy': 0.1, #[g m-3] half saturation constant for N uptake (Prats & Llavador, 1994)
      'K_P_phy': 0.02, #[g m-3] half saturation constant for P uptake (Prats & Llavador, 1994)  
      'kNdecr': 0.05, #[d-1] decomposition rate (to replace bacteria decomposition rate) (Prats & Llavador, 1994)
      'kPdecr': 0.4, #[d-1] decomposition rate (to replace bacteria decomposition rate) (Prats & Llavador, 1994)    
      'f_N_edg': 0.8*0.6*1000, # [g d-1] endogenous N supply from soil [originally kg N ha-1 d-1]
      'f_P_edg': 0.5*0.6*1000, #[g d-1] endogenous P supply from soil [originally kg N ha-1 d-1]
      'MuptN': 8*0.6*1000, #[g d-1] maximum daily N uptake by rice plants, originally in [kg ha-1 d-1], only 0.6 ha of land that is planted with rice
      'MuptP': 5*0.6*1000,#[g d-1] maximum daily P uptake by rice plants, originally in [kg ha-1 d-1], only 0.6 ha of land that is planted with rice
      'kNphy': 0.06, #[g N/g biomass] fraction of N from phytoplankton biomass
      'kPphy': 0.01, #[g P/g biomass] fraction of P from phytoplankton biomass
      'V_phy': V_phy,

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

d_pn = {
     'I0' :  np.array([tsim, np.full((tsim.size,), I0)]).T, #[J m-2 h-1] Hourly solar irradiation (6 AM to 6 PM)
     'Tw':np.array([tsim, Tavg]).T,
     'f_N_sol': np.array([tsim, np.full((tsim.size,), 0.5)]).T,
     'f_P_sol': np.array([tsim, np.full((tsim.size,), 0.5)]).T,
     'f_N_prt': np.array([tsim, np.full((tsim.size,), 0.5)]).T,
     'f_P_prt': np.array([tsim, np.full((tsim.size,), 0.5)]).T,
     'Rain': np.array([tsim, Rain]).T,
     'DVS':np.array([tsim, np.full((tsim.size,), 0.5)]).T
     }

#%% Input (u)
#inorganic fertilizer types and concentration used in Indonesia: 
#Nitrogen source: NPK (15%:15%:15%), Urea (46% N)
#NPK fertilizer recommended doses 167 kg/ha (Yassi 2023)
NPK_w = 167*A_rice #[kg]
NPK = (15/100)*NPK_w

#Urea fertilizer recommended doses 100 kg/ha (Yassi 2023)
Urea_w = 100*A_rice #[kg]
Urea = (46/100)*Urea_w

#Total N from inorganic fertilizer
N_ingf = NPK + Urea

N_ingf_1 = N_ingf/3
N_ingf_2 = N_ingf - N_ingf_1

#Phosphorus source: SP36 (36%P2O5 ~ 7.85% P)
#P content in SP-36: 7.85% P
#SP fertilizer recommended doses 31 kg/ha (Yassi 2023)
SP36 = 31*A_rice
I_P = (7.85/100)*SP36

Norgf = 12.3*0.6/1000 #[kg N]
Porgf = 13.9*0.6/1000 #[kg P]

u_pn = {'N_ingf_1': N_ingf_1, 'N_ingf_2': N_ingf_2, 'I_P': I_P, 'Norgf': Norgf, 'Porgf': Porgf}
#%% Initialize module
pnut = PNut(tsim, dt, x0_pn, p_pn)
flow_pnut = pnut.f

# Run simulation
# TODO: Call the method run to generate simulation results
#simulation for two different model look at grass-water model
tspan = (tsim[0], tsim[-1])
ypn = pnut.run(tspan, d_pn, u_pn)

#%% Retrieve simulation results
# TODO: Retrieve the simulation results
t = ypn['t']
Mphy = ypn['Mphy']/V_phy #[g m-3] 
M_N_net = ypn['M_N_net']/1000 #[kg N]
M_P_net = ypn['M_P_net']/1000 #[kg P]

f_N_edg = np.full((t.size,), pnut.p['f_N_edg'])
f_P_edg = np.full((t.size,), pnut.p['f_P_edg'])
f_N_fert_ing = pnut.f['f_N_fert_ing']
f_P_fert_ing = pnut.f['f_P_fert_ing']
f_N_fert_org = pnut.f['f_N_fert_org']
f_P_fert_org = pnut.f['f_P_fert_org']
f_phy_grw = pnut.f['f_phy_grw']/V_phy
f_phy_prd = pnut.f['f_phy_prd']/V_phy
f_phy_cmp = pnut.f['f_phy_cmp']/V_phy
f_N_fis_sol = pnut.f['f_N_fis_sol']
f_P_fis_sol = pnut.f['f_P_fis_sol']
f_N_fis_prt = pnut.f['f_N_fis_prt']
f_P_fis_prt = pnut.f['f_P_fis_prt']
f_N_phy_cmp = pnut.f['f_N_phy_cmp']
f_P_phy_cmp = pnut.f['f_P_phy_cmp']
f_N_phy_upt = pnut.f['f_N_phy_upt']
f_P_phy_upt = pnut.f['f_P_phy_upt']
f_N_plt_upt = pnut.f['f_N_plt_upt']
f_P_plt_upt = pnut.f['f_P_plt_upt']

#%%Plot results

#Phytoplankton
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
# Plot the first subplot
ax1.plot(t, Mphy, label = '$M_{phy}$')
ax1.set_ylabel('concentration [$g m^{-3}$]')
ax1.legend()
ax1.set_title('Phytoplankton')

# Plot the second subplot
ax2.plot(t, f_phy_grw, label ='$\phi_{phy,grw}$')
ax2.plot(t, -f_phy_prd, label ='$\phi_{phy,prd}$')
ax2.plot(t, -f_phy_cmp, label ='$\phi_{phy,cmp}$')
ax2.set_ylabel(r'rate $[g m^{-3} d^{-1}]$')
ax2.set_xlabel('time [day]')
ax2.legend()

#Net nitrogen
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
# Plot the first subplot
ax1.plot(t, M_N_net, label = '$M_{N,net}$')
ax1.set_ylabel('mass [$kg$]')
ax1.legend()
ax1.set_title('Net nitrogen accumulated')

# # Plot the second subplot
ax2.plot(t, f_N_edg, label = '$\phi_{N,edg}$')
ax2.plot(t, f_N_fert_ing, label = '$\phi_{N,fert,ing}$')
ax2.plot(t, f_N_fert_org, label = '$\phi_{N,fert,org}$')
ax2.plot(t, f_N_phy_cmp, label = '$\phi_{N,phy,cmp}$')
ax2.plot(t, f_N_fis_sol, label = '$\phi_{N,fis,sol}$')
ax2.plot(t, f_N_fis_prt, label = '$\phi_{N,fis,prt}$')
ax2.plot(t, -f_N_phy_upt, label = '$\phi_{N,phy,upt}$')
ax2.plot(t, -f_N_plt_upt, label = '$\phi_{N,plt,upt}$')
ax2.set_ylabel(r'rate $[g d^{-1}]$')
ax2.set_xlabel('time [day]')
ax2.legend()

#Net phosphorus
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
# Plot the first subplot
ax1.plot(t, M_P_net, label = '$M_{P,net}$')
ax1.set_ylabel('mass [$kg$]')
ax1.legend()
ax1.set_title('Net phosphorus accumulated')
# # Plot the second subplot
ax2.plot(t, f_P_edg, label = '$\phi_{P,edg}$')
ax2.plot(t, f_P_fert_ing, label = '$\phi_{P,fert,ing}$')
ax2.plot(t, f_P_fert_org, label = '$\phi_{P,fert, org}$')
ax2.plot(t, f_P_phy_cmp, label = '$\phi_{P,phy,cmp}$')
ax2.plot(t, f_P_fis_sol, label = '$\phi_{P,fis,sol}$')
ax2.plot(t, f_P_fis_prt, label = '$\phi_{P,fis,prt}$')
ax2.plot(t, -f_P_phy_upt, label = '$\phi_{P,phy,upt}$')
ax2.plot(t, -f_P_plt_upt, label = '$\phi_{P,plt,upt}$')
ax2.set_ylabel(r'rate $[g d^{-1}]$')
ax2.set_xlabel('time [day]')
ax2.legend()