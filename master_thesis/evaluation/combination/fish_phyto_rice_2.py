# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 09:08:22 2024

@author: alegn
"""

from models.fish import Fish
from models.phytoplankton import Phyto
from models.rice import Rice
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')

# Simulation time array
t = 120
tsim = np.linspace(0.0, t, t+1) # [d]
t_weather = np.linspace(0.0, t, t+1) 
dt = 1
tspan = (tsim[0], tsim[-1])

#%%# pond area and volume
area = 1  # [ha] the total area of the system
rice_area = 0.6  # [ha] rice field area
pond_area = (area - rice_area)*1000  # [m2] pond area
pond_volume = pond_area*0.6
m_fish = 30 #[g] average weight of one fish fry
n_fish = 4000 #[no of fish] according to Pratiwi 2019, 10000 fish is just for comparing result with literature
n_rice = 127980 #number of plants in 0.6 ha land

#%% Disturbances
data_weather = 'C:/Users/alegn/Documents/WUR/Thesis/rfmodel/data/Weather/Magelang_200110_200209_Daily.csv'
weather = pd.read_csv(data_weather, header=0, sep=';')

#FIRST CYCLE
t_ini = '20011001'
t_end = '20020129'

weather['Time'] = pd.to_datetime(weather['Time'], format='%Y%m%d')  # Adjust the format if necessary
weather.set_index('Time', inplace=True)
#first cycle
Tavg1 = weather.loc[t_ini:t_end,'Tavg'].values #[°C] Mean daily temperature
Rain1 = weather.loc[t_ini:t_end,'Rain'].values #[mm] Daily precipitation
Igl1 = weather.loc[t_ini:t_end, 'I0'].values #[MJ m-2] Sum of shortwave radiation daily

I01 = 0.45*Igl1*1E6 #Convert [MJ m-2 d-1] to [J m-2 d-1] PAR

##phytoplankton model
x0phy = {'Mphy': 4.29e-4, #[g m-3] from Mei et al 2023 concentration of Chla multiply with Chla:phyto mass ratio from Jamu and Piedrahita (2002)
      'N_deposit':   0.01, #[g m-3] N
      'P_deposit':   0.01, #[g m-3] P
      'Nfert_org': 12.3, #[g m-3] N content of organic fertilizer applied
      'Pfert_org': 13.9 #[g m-3] P content of organic fertilizer applied
      }  

# Model parameters
# TODO: define the parameter values in the dictionary p
pphy = {
      'mu_phy':  1.27, #[d-1] maximum growth rate of phytoplankton
      'mu_Up': 0.005, #[d-1] maximum nutrient uptake coefficient
      'pd': 0.5, #[m] pond depth
      'l_sl': 3.5e-7, #[m2 g-1] phytoplankton biomass-specific light attenuation
      'l_bg': 0.77, #[m-1] light attenuation by non-phytoplankton components
      'Kpp': 2.82e6, # [J m-2 d-1] half-saturation constant of phytoplankton production
      'cm': 0.15, #[d-1] phytoplankton mortality rate
      'cl': 0.004, #[m2 g-1] phytoplankton crowding loss constant
      'c1': 1.57, # [-] temperature coefficients
      'c2': 0.24, # [-] temperature coefficients
      'Topt': 28, #optimum temperature for phytoplankton growth
      'Mp': 0.025, #[g m-3] half saturation constant for nutrient uptake
      # 'Norgf': 1.23, #[g N/m2] N content from organic fertilizer (Kang'Ombe)   
      # 'Porgf': 1.39, #[g P/m2] P content from organic fertilizer (Kang'Ombe)
      'K_N_phy': 0.1, #[g m-3] half saturation constant for N uptake (Prats & Llavador, 1994)
      'K_P_phy': 0.02, #[g m-3] half saturation constant for P uptake (Prats & Llavador, 1994)  
      'kNdecr': 0.05, #[d-1] decomposition rate (to replace bacteria decomposition rate) (Prats & Llavador, 1994)
      'kPdecr': 0.4, #[d-1] decomposition rate (to replace bacteria decomposition rate) (Prats & Llavador, 1994)    
      'kNph': 0.06, #[g N/g biomass] fraction of N from phytoplankton biomass
      'kPph': 0.01, #[g P/g biomass] fraction of P from phytoplankton biomass
      'pond_volume': pond_volume
        }

#initialize object
phy = Phyto(tsim, dt, x0phy, pphy)

##fish model
#state variables
x0fish = {'Mfish': m_fish, #[gDM] 
      'Mdig':1E-6, #[gDM]
      'Muri':1E-6 #[gDM]
      } 

#parameters
pfish = {
    "tau_dig": 4.5,  # [h] time constants for digestive *24 for convert into day
    "tau_uri": 20,  # [h] time constants for urinary *24  convert into day
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
    'pond_area': pond_area
}

#initialize object
fish0 = Fish(tsim, dt, {key: val*0 for key, val in x0fish.items()}, pfish)
fish = Fish(tsim, dt, x0fish, pfish)
flow = fish.f

#disturbance
dfish1 = {
    "DO": np.array([tsim, np.random.randint(1, 6, size=tsim.size)]).T,
    "T": np.array([t_weather, Tavg1]).T,
    "Mphy": np.array([tsim, np.full((tsim.size,), 4.293e-4)]).T,
}
dphy = {
     'I0' :  np.array([tsim, I01]).T, #[J m-2 d-1] Hourly solar irradiation (6 AM to 6 PM)
     'T':np.array([tsim, Tavg1]).T,
     'f_N_sol': np.array([tsim, np.full((tsim.size,), 0.1)]).T,
     'f_P_sol': np.array([tsim, np.full((tsim.size,), 0.1)]).T,
     'Rain': np.array([tsim, Rain1]).T
     }

x0rice = {
    "Mrt": 0.005,  # [kg DM ha-1 d-1] Dry biomass of root
    "Mst": 0.003,  # [kg DM ha-1 d-1] Dry biomass of stems
    "Mlv": 0.002,  # [kg DM ha-1 d-1] Dry biomass of leaves
    "Mpa": 0.0,  # [kg DM ha-1 d-1] Dry biomass of panicles
    "Mgr": 0.0,  # [kg DM ha-1 d-1] Dry biomass of grains
    "HU": 0.0,
    "DVS": 0,
}

price = {
      #Manually calibrated developmental rate
          'DVSi': 0,
           'DVRJ': 0.0020,
           'DVRI': 0.00195,
           'DVRP': 0.00195,     
           'DVRR': 0.0024,
         "Tmax": 40,
         "Tmin": 15,
         "Topt": 33,
         "k": 0.4,
         "Rm_rt": 0.01,
         "Rm_st": 0.015,
         "Rm_lv": 0.02,
         "Rm_pa": 0.003,
         "N_lv": 0,
         "Rec_N": 0.5,
         "k_lv_N": 0,
         "k_pa_maxN": 0.0175,
         "M_upN": 8,
         "cr_lv": 1.326,
         "cr_st": 1.326,
         "cr_pa": 1.462,
         "cr_rt": 1.326,
         "IgN": 0.8, # Indigenous soil N supply during wet season
         # "IgN": 0.5, # Indigenous soil N supply during dry season
         'n_rice': n_rice,
         'gamma': 65000 #[number of grains/kg DM]
     }

drice = {'I0':np.array([tsim, I01]).T,
     # 'T':np.array([tsim, np.random.uniform(low=20, high=35, size=tsim.size,)]).T,
     'T': np.array([tsim, Tavg1]).T,
     'CO2': np.array([tsim, np.full((tsim.size,), 400)]).T,
     'C_NO3': np.array([tsim, np.full((tsim.size,), 0)]).T,
     'C_H2PO4': np.array([tsim, np.full((tsim.size,), 0)]).T,
     }
# inorganic fertilizer types and concentration:
# Nitrogen source: NPK (15%:15%:15%), Urea (46% N)
NPK_w = 167 * rice_area
Urea = 200 * rice_area
I_N = (15 / 100) * NPK_w + (46/100)*Urea

# Phosphorus source: SP36 (36%P2O5 ~ 7.85% P)
# P content in SP-36: 7.85% P
SP36 = 31 * rice_area
I_P = (7.85 / 100) * SP36

#instantiate object
rice = Rice(tsim, dt, x0rice, price)
#run fish model
yfish = {'t': [0, ], 'Mfish': [0, ], 'Mdig':[0, ], 'Muri':[0, ], 'f_N_prt': [0, ], 'f_P_prt': [0, ], 'f_TAN': [0, ], 'f_P_sol': [0, ]}
yrice_ = {'t': [0, ], 'Mrt': [1e-6, ], 'Mst': [1e-6, ], 'Mlv': [1e-6, ], 'Mpa': [0, ], 'Mgr': [0, ], 'HU': [0, ], 'DVS': [0, ]}
yphy_ = {'t': [0, ], 'Mphy': [4.29e-4, ],'N_deposit': [0.01, ], 'P_deposit':[0.01, ], 'Nfert_org': [12.3, ], 'Pfert_org': [13.9, ]}
dphy_ = {'DVS': []}
drice_ = {'C_NO3': [], 'C_H2PO4': []}

it = np.nditer(tsim[:-1], flags=['f_index'])
for ti in it:
    # Index for current time instant
    idx = it.index
    # Integration span
    tspan = (tsim[idx], tsim[idx+1])
    print('Integrating', tspan)
    # Controlled inputs
    urice = {"I_N": I_N, "I_P": I_P}  # [kg m-2 d-1]
    ufish = {'Mfed': m_fish*0.03}
    #Run phyto model
    yphy =  phy.run(tspan, dphy)
    yphy_['t'].append(yphy['t'][1])
    yphy_['Mphy'].append(yphy['Mphy'][1])
    yphy_['N_deposit'].append(yphy['N_deposit'][1])
    yphy_['P_deposit'].append(yphy['P_deposit'][1])
    yphy_['Nfert_org'].append(yphy['Nfert_org'][1])
    yphy_['Pfert_org'].append(yphy['Pfert_org'][1])
    
    #make simulation result of yphy as input for fish
    dfish1['Mphy'] = np.array([yphy["t"], yphy["Mphy"]]).T 
    #run rice model
    yrice =  rice.run(tspan, drice, urice)
    yrice_['t'].append(yrice['t'][1])
    yrice_['Mgr'].append(yrice['Mgr'][1])
    yrice_['Mrt'].append(yrice['Mrt'][1])
    yrice_['Mst'].append(yrice['Mst'][1])
    yrice_['Mlv'].append(yrice['Mlv'][1])
    yrice_['Mpa'].append(yrice['Mpa'][1])
    yrice_['HU'].append(yrice['HU'][1])
    yrice_['DVS'].append(yrice['DVS'][1])
    # if ti>=21:
    #     #retrieve rice simulation result for phytoplankton
    #     dphy['DVS'] = np.array([yrice['t'], yrice['DVS']]).T

    if ti < 24:
       #make the fish weight zero to simulate 'no growth'
       yfish0 =  fish0.run((0,23), dfish1, ufish)
       yfish['t'].append(yfish0['t'][1])
       yfish['Mfish'].append(0)
       yfish['Mdig'].append(0)
       yfish['Muri'].append(0)
       yfish['f_N_prt'].append(0)
       yfish['f_P_prt'].append(0)
       yfish['f_TAN'].append(0)
       yfish['f_P_sol'].append(0)
    elif ti > 80:
           #make the fish weight zero to simulate 'no growth'
           yfish0 =  fish0.run((80, tsim[-1]), dfish1, ufish)
           yfish['t'].append(yfish0['t'][1])
           yfish['Mfish'].append(0)
           yfish['Mdig'].append(0)
           yfish['Muri'].append(0)
           yfish['f_N_prt'].append(0)
           yfish['f_P_prt'].append(0)
           yfish['f_TAN'].append(0)
           yfish['f_P_sol'].append(0)
    else: 
       yfishr =   fish.run((24,79), dfish1, ufish)
       yfish['t'].append(yfishr['t'][1])
       yfish['Mfish'].append(yfishr['Mfish'][1])
       yfish['Mdig'].append(yfishr['Mdig'][1])
       yfish['Muri'].append(yfishr['Muri'][1])
       yfish['f_N_prt'].append(yfishr['f_N_prt'][1])
       yfish['f_P_prt'].append(yfishr['f_P_prt'][1])
       yfish['f_TAN'].append(yfishr['f_TAN'][1])
       yfish['f_P_sol'].append(yfishr['f_P_sol'][1])
       f_N_prt =  yfishr['f_N_prt']
       f_P_prt =  yfishr['f_P_prt']
       f_TAN =  yfishr['f_TAN']
       f_P_sol =  yfishr['f_P_sol']
       #retrieve phy simulation result for rice growth
       drice_['C_NO3'].append(np.array([yphy['t'][1]], yphy['N_deposit'][1]))
       drice_['C_H2PO4'].append(np.array([yphy['t'][1]], yphy['P_deposit'][1]))
       drice['C_NO3'] = np.array([yphy['t'], yphy['N_deposit']]).T
       drice['C_H2PO4'] = np.array([yphy['t'], yphy['P_deposit']]).T
       
yfish = {key: np.array(value) for key, value in yfish.items()} 
yphy = {key: np.array(value) for key, value in yphy_.items()} 
yrice = {key: np.array(value) for key, value in yrice_.items()} 

#retrieve results
t= yphy['t']
Mphy= phy.y['Mphy']
N_deposit = phy.y['N_deposit']
P_deposit = phy.y['P_deposit']
Nfert_org = phy.y['Nfert_org']
Pfert_org = phy.y['Pfert_org']

f1 = phy.f['f1']
f2 = phy.f['f2']
f3 = phy.f['f3']
f4 = phy.f['f4']
f5 = phy.f['f5']
f6 = phy.f['f6']
f7 = phy.f['f7']

Mfish= fish.y['Mfish']
Muri= fish.y['Muri']
Mfis_fr= Mfish/pfish['k_DMR']
Mdig= fish.y['Mdig']

# Plot results
plt.figure(1)
plt.plot(t, Mfish, label='Fish dry weight')
plt.plot(t, Mdig, label='Fish digestive system weight')
plt.plot(t, Muri, label='Fish urinary system weight')
plt.plot(t, Mfis_fr, label='Fish fresh weight')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$biomass\ [g day-1]$')
plt.title('Fish biomass accumulation')

plt.figure(2)
plt.plot(t, flow['f_upt'], label='nutrient uptake flow')
plt.plot(t, flow['f_N_upt'], label='N uptake flow')
plt.plot(t, flow['f_P_upt'], label='P uptake flow')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$flow rate\ [g day-1]$')
plt.title('Nutrient uptake rate')

plt.figure(3)
plt.plot(t, flow['f_sol'], label='total soluble excretion')
plt.plot(t, flow['f_N_sol'], label='N content in soluble excretion')
plt.plot(t, flow['f_P_sol'], label='P content in soluble excretion')
plt.plot(t, flow['f_TAN'], label='soluble TAN')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$flow rate\ [g day-1]$')
plt.title('Soluble excretion rate')

plt.figure(4)
plt.plot(t, flow['f_fed'], label='total feed intake rate')
plt.plot(t, flow['f_prt'], label='total solid excretion')
plt.plot(t, flow['f_N_prt'], label='N content in solid excretion')
plt.plot(t, flow['f_P_prt'], label='P content in solid excretion rate')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$flow rate\ [g day-1]$')
plt.title('Particulate excretion rate')

