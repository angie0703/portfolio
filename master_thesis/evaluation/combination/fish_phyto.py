# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 09:08:22 2024

@author: alegn
"""

from models.fish import Fish
from models.phytoplankton import Phygrowth
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')

# Simulation time array
t = 120
tsim = np.linspace(0.0, t, t+1) # [d]
t_weather = np.linspace(0.0, 120, 120+1) 
dt = 1
tspan = (tsim[0], tsim[-1])
#%%# pond area and volume
area = 10000  # [m2] the total area of the system
rice_area = 6000  # [m2] rice field area
pond_area = area - rice_area  # [m2] pond area
pond_volume = pond_area*0.5
m_fish = 30 #[g] average weight of one fish fry
# N_fish = [60, 120, 180]  # [g m-3] 
# n_fish = [(N * pond_volume) / m_fish for N in N_fish]
n_fish = [4000, 8000, 10000] #[no of fish] according to Pratiwi 2019
n_rice = [127980, 153600] #number of plants in 6000 m2 land
#%% Disturbances

data_pond = 'C:/Users/alegn/Documents/WUR/Thesis/rfmodel/data/Bacteria/Nitrogen_Tilapia_Mei_2023.csv'
pond = pd.read_csv(data_pond, header=0, sep=';')
t_pond = [0, 14, 28, 42, 56, 70] #[d]
data_P = 'C:/Users/alegn/Documents/WUR/Thesis/rfmodel/data/Bacteria/P_Tilapia_Mei_2023.csv'
P_pond = pd.read_csv(data_P, header=0, sep=';')
SNH4_ = pond.loc[t_pond[0]:t_pond[-1], 'NH4 (g m-3)'].values
SNO3_ = pond.loc[t_pond[0]:t_pond[-1], 'NO3 (g m-3)'].values
SNO2_ = pond.loc[t_pond[0]:t_pond[-1], 'NO2 (g m-3)'].values
SP_  = P_pond.loc[t_pond[0]:t_pond[-1], 'SRP'].values

SNH4 =np.interp(tsim, t_pond, SNH4_)
SNO3 =np.interp(tsim, t_pond, SNO3_)
SNO2 =np.interp(tsim, t_pond, SNO2_)
SP =np.interp(tsim, t_pond, SP_)

# Disturbances
data_weather = 'C:/Users/alegn/Documents/WUR/Thesis/rfmodel/data/Weather/Magelang_200110_200209_Daily.csv'
weather = pd.read_csv(data_weather, header=0, sep=';')
weather['Time'] = pd.to_datetime(weather['Time'], format='%Y%m%d')  # Adjust the format if necessary
weather.set_index('Time', inplace=True)
#FIRST CYCLE
t_ini = '20011001'
t_end = '20020129'

#SECOND CYCLE
t_ini2 = '20020130'
t_end2 = '20020530'

#first cycle
Tavg1 = weather.loc[t_ini:t_end,'Tavg'].values #[°C] Mean daily temperature
Rain1 = weather.loc[t_ini:t_end,'Rain'].values #[mm] Daily precipitation
Igl1 = weather.loc[t_ini:t_end, 'I0'].values #[MJ m-2] Sum of shortwave radiation daily

I01 = 0.45*Igl1*1E6 #Convert [MJ m-2 d-1] to [J m-2 d-1] PAR

#second cycle
Tavg2 = weather.loc[t_ini2:t_end2,'Tavg'].values #[°C] Mean daily temperature
Rain2 = weather.loc[t_ini2:t_end2,'Rain'].values #[mm] Daily precipitation
Igl2 = weather.loc[t_ini2:t_end2, 'I0'].values #[MJ m-2] Sum of shortwave radiation daily

I02 = 0.45*Igl2*1E6 #Convert [MJ m-2 d-1] to [J m-2 d-1] PAR
##phytoplankton model
x0phy = {'Mphy': 4.29e-5, #[g m-3] from Mei et al 2023 concentration of Chla multiply with Chla:phyto mass ratio from Jamu and Piedrahita (2002)
      'NA':   0 #[g m-3] TN from Mei et al 2023
      } 

#%% Model parameters
# TODO: define the parameter values in the dictionary p
pphy = {
      'mu_phy':  1.27, #[d-1] maximum growth rate of phytoplankton
      'mu_Up': 0.005, #[d-1] maximum nutrient uptake coefficient
      'pd': 0.5, #[m] pond depth
      'l_sl': 3.5e-7, #[m2 g-1] phytoplankton biomass-specific light attenuation
      'l_bg': 0.77, #[m-1] light attenuation by non-phytoplankton components
      'Kpp': 2812320, # [J m-2 d-1] half-saturation constant of phytoplankton production
      'cm': 0.15, #[d-1] phytoplankton mortality rate
      'cl': 0.004, #[m2 g-1] phytoplankton crowding loss constant
      'c1': 1.57, # [-] temperature coefficients
      'c2': 0.24, # [-] temperature coefficients
      'Topt': 28, #optimum temperature for phytoplankton growth
      'Mp': 0.025, #[g m-3] half saturation constant for nutrient uptake
     'Ksp': 1
        }


#initialize object
phyto = Phygrowth(tsim, dt, x0phy, pphy)

##fish model
#state variables
x0fish = {'Mfish': m_fish*n_fish[0], #[gDM] 
      'Mdig':3.75e-7*n_fish[0], #[gDM]
      'Muri':3.75e-7*n_fish[0] #[gDM]
      } 
# x0fish = {'Mfish': m_fish, #[gDM] 
#       'Mdig':3.75e-7, #[gDM]
#       'Muri':3.75e-7 #[gDM]
#       } 
#parameters
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
    'cl': 0.004,
    'Ksd': 40000 #[g C/m3] 
}

#initialize object
fish = Fish(tsim, dt, x0fish, pfish)
fish0 = Fish(tsim, dt, {key: val*0 for key, val in x0fish.items()}, pfish)
flow = fish.f

#%%disturbance
dfish1 = {
    "DO": np.array([tsim, np.random.randint(4, 6, size=tsim.size)]).T,
    "T": np.array([t_weather, Tavg1]).T,
    "Mphy": np.array([tsim, np.full((tsim.size,), 4.29e-5)]).T,
}
dphy = {
     'I0' :  np.array([t_weather, I01]).T, #[J m-2 d-1] Average solar irradiation (6 AM to 6 PM) or 3,749,760 Joules per square meter per day (J m−2
     'T':np.array([t_weather, Tavg1]).T,
     'DVS':np.array([tsim, np.linspace(0, 2.5, 120+1)]).T, #to simulate the information flows from rice growth
     'SNH4': np.array([tsim, SNH4]).T,
     'SNO2': np.array([tsim, SNO2]).T,
     'SNO3': np.array([tsim, SNO3]).T,
     'SP': np.array([tsim, SP]).T,
     'Rain': np.array([t_weather, Rain1]).T
     }

#%%Simulation
#initialize history storage
dfish1_ = {'Mphy':[]}
dphy_ = {'SNH4': [], 'SP': []} 
yphy_ = {'t': [0, ], 'Mphy': [4.29e-4, ], 'NA':[0.1, ]}
yfish_ = {'t': [0, ], 'Mfish': [0, ], 'Mdig':[0, ], 'Muri':[0, ], 'f_N_prt': [0, ], 'f_P_prt': [0, ], 'f_TAN': [0, ], 'f_P_sol': [0, ] }
#run model
#run model
it = np.nditer(tsim[:-1], flags=['f_index'])
for ti in it:
    # Index for current time instant
    idx = it.index
    # Integration span
    tspan = (tsim[idx], tsim[idx+1])
    print('Integrating', tspan)
    # Controlled inputs
    ufish = {'Mfed': m_fish*0.03*n_fish[0]}
    # run phyto model
    yphy = phyto.run(tspan, d=dphy)
    yphy_['t'].append(yphy['t'][1])
    yphy_['Mphy'].append(yphy['Mphy'][1])
    yphy_['NA'].append(yphy['NA'][1])
    
    # Retrieve phyto model outputs 
    # dfish1['Mphy'] = np.array([yphy['t'], yphy['Mphy']]).T
    dfish1_['Mphy'].append(np.array([yphy['t'][1], yphy['Mphy'][1]]))
    dfish1['Mphy'] = np.array([yphy['t'], yphy['Mphy']*pond_volume])
    
    #run fish model
    if ti < 24:
       #make the fish weight zero to simulate 'no growth'
       yfish0 =  fish0.run((0,23), dfish1, ufish)
       yfish_['t'].append(yfish0['t'][1])
       yfish_['Mfish'].append(0)
       yfish_['Mdig'].append(0)
       yfish_['Muri'].append(0)
       yfish_['f_N_prt'].append(0)
       yfish_['f_P_prt'].append(0)
       yfish_['f_TAN'].append(0)
       yfish_['f_P_sol'].append(0)
    elif ti > 80:
           #make the fish weight zero to simulate 'no growth'
           yfish0 =  fish0.run((80, tsim[-1]), dfish1, ufish)
           yfish_['t'].append(yfish0['t'][1])
           yfish_['Mfish'].append(0)
           yfish_['Mdig'].append(0)
           yfish_['Muri'].append(0)
           yfish_['f_N_prt'].append(0)
           yfish_['f_P_prt'].append(0)
           yfish_['f_TAN'].append(0)
           yfish_['f_P_sol'].append(0)
    else: 
       yfishr =   fish.run((24,79), dfish1, ufish)
       yfish_['t'].append(yfishr['t'][1])
       yfish_['Mfish'].append(yfishr['Mfish'][1])
       yfish_['Mdig'].append(yfishr['Mdig'][1])
       yfish_['Muri'].append(yfishr['Muri'][1])
       yfish_['f_N_prt'].append(yfishr['f_N_prt'][1])
       yfish_['f_P_prt'].append(yfishr['f_P_prt'][1])
       yfish_['f_TAN'].append(yfishr['f_TAN'][1])
       yfish_['f_P_sol'].append(yfishr['f_P_sol'][1])
    
# def replace_values(original, new_values):
#     # Ensure the original has enough elements to replace (at least 121)
#     if len(original) >= 121:
#         # Replace indices 1 to 120
#         original[1:121] = new_values

# # Apply this function to each key in your dictionaries
# for key, val in dphy_.items():
#     dphy_[key] = np.array(val)
# for key, val in dfish1_.items():
#     dfish1_[key] = np.array(val)    
    
# for key in dphy.keys():
#     if key in dphy_:
#         replace_values(dphy[key], dphy_[key])

# for key in dfish1.keys():
#     if key in dfish1_:
#         replace_values(dfish1[key], dfish1_[key])    

for key, val in yfish_.items():
    yfish_[key] = np.array(val)
    
for key, val in yphy_.items():
    yphy_[key] = np.array(val)

yphy = yphy_
yfish = yfish_
#retrieve results
t= yfish['t']
Mphy= phyto.y['Mphy']
NA = phyto.y['NA']
Mfish= fish.y['Mfish']*0.001
Muri= fish.y['Muri']*0.001
Mfis_fr= Mfish/pfish['k_DMR']
Mdig= fish.y['Mdig']*0.001
rphy = fish.f['r_phy']
# Plot results
plt.figure(1)
plt.plot(t, Mfish, label='Fish dry weight')
# plt.plot(t, Mdig, label='Fish digestive system weight')
# plt.plot(t, Muri, label='Fish urinary system weight')
# plt.plot(t, Mfis_fr, label='Fish fresh weight')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$biomass\ [kg day-1]$')
plt.title('Fish biomass accumulation')

plt.figure(2)
plt.plot(t, flow['f_upt'], label='nutrient uptake flow')
plt.plot(t, flow['f_N_upt'], label='N uptake flow')
plt.plot(t, flow['f_P_upt'], label='P uptake flow')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'flow rate $[g day^{-1}]$')
plt.title('Nutrient uptake rate')

plt.figure(3)
plt.plot(t, flow['f_P_sol'], label='P content in soluble excretion')
plt.plot(t, flow['f_TAN'], label='soluble TAN')
plt.plot(t, flow['f_N_prt'], label='N content in solid excretion')
plt.plot(t, flow['f_P_prt'], label='P content in solid excretion rate')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'flow rate $[g day-1]$')
# plt.title('Soluble excretion rate')

plt.figure(4)
plt.plot(t, flow['f_fed'], label='total feed intake rate')
plt.plot(t, Mphy, label='Phytoplankton Biomass')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'flow rate $[g day-1]$')
# plt.title('Particulate excretion rate')

# to get the value of Mphy and NA in g, multiply it with pond_volume
Mphy = yphy['Mphy']
NA = yphy['NA']

f1 = phyto.f['f1']
f2 = phyto.f['f2']
f3 = phyto.f['f3']
f4 = phyto.f['f4']
f5 = phyto.f['f5']

# Plot
plt.figure(5)
plt.plot(t, Mphy, label='Phytoplankton Biomass')
plt.plot(t, NA, label='nutrient availability')
plt.xlabel(r'time [d]')
plt.ylabel(r'Accumulation rate $[g m^{-3} d^{-1}]$')
# plt.title(r'Accumulative Phytoplankton Growth and Nutrient Availability')
plt.legend()

plt.figure(7)
plt.plot(t, f1, label='phyto growth')
plt.plot(t, f2, label= 'natural mortality rate')
plt.plot(t, f3, label = 'death by intraspecific competition rate')
plt.xlabel(r'time [d]')
plt.ylabel(r'Flow rate [g d-1]')
plt.title(r'Flow in Phytoplankton Growth')
plt.legend()

plt.figure(8)
plt.plot(t, f4, label='Nutrient available in Pond')
plt.plot(t, f5, label= 'Nutrient uptake by phytoplankton')
plt.xlabel(r'time [d]')
plt.ylabel(r'Flow rate [g d-1]')
plt.title(r'Flow in Phytoplankton Growth')
plt.legend()

Raff = fish.f['Raff']
Rphy = fish.f['Rphy']
Rdet = fish.f['Rdet']

plt.figure(9)
plt.plot(t[1:],Rphy[1:], label='Phytoplankton')
plt.plot(t[1:], Raff[1:], label='Artifical feed')
plt.plot(t[1:], Rdet[1:], label='Detritus')
plt.xlabel(r'time [d]')
plt.ylabel(r'Feeding rate $[g m^{-3} d^{-1}]$')
plt.title("Feeding Composition")
plt.legend()

r_aff = fish.f['r_aff']
r_det = fish.f['r_det']
r_phy = fish.f['r_phy']
plt.figure(10)
plt.plot(t[1:],r_phy[1:], label='Phytoplankton')
plt.plot(t[1:], r_aff[1:], label='Artifical feed')
plt.plot(t[1:], r_det[1:], label='Detritus')
plt.xlabel(r'time [d]')
plt.ylabel(r'Ratio [-]')
plt.title("Feeding Composition")
plt.legend()