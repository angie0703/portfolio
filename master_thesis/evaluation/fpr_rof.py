# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 09:08:22 2024

@author: alegn
"""

from models.fish import Fish
from models.phytoplankton import Phygrowth
from models.rice import Rice
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.style.use('ggplot')

# Simulation time array
t = 120
tsim = np.linspace(0.0, t, t+1) # [d]
t_weather = np.linspace(0.0, t, t+1) 
dt = 1
tspan = (tsim[0], tsim[-1])

#%%# pond area and volume
area = 10000  # [m2] the total area of the system
rice_area = 6000  # [m2] rice field area
pond_area = area - rice_area  # [m2] pond area
pond_volume = pond_area*0.5
m_fish = 30 #[g] average weight of one fish fry
# N_fish = [14.56, 18.2, 21.84]  # [g m-3]
# n_fish = [(N * pond_volume) / m_fish for N in N_fish]
n_fish = [4000, 8000, 10000] #[no of fish] according to Mridha 2014
n_rice = [127980, 153600] #number of plants in 6000 m2 land

#%% Disturbances
data_weather = 'C:/Users/alegn/Documents/WUR/Thesis/rfmodel/data/Weather/Magelang_200110_200209_Daily.csv'
weather = pd.read_csv(data_weather, header=0, sep=';')
weather['Time'] = pd.to_datetime(weather['Time'], format='%Y%m%d')  # Adjust the format if necessary
weather.set_index('Time', inplace=True)
#FIRST CYCLE
t_ini = '20011001'
t_end = '20020129'

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

dfish1 = {
    "DO": np.array([tsim, np.random.randint(1, 6, size=tsim.size)]).T,
    "T": np.array([t_weather, Tavg1]).T,
    "Mphy": np.array([tsim, np.full((tsim.size,), 4.29e-5)]).T,
}
dphy = {
     'I0' :  np.array([tsim, np.full((tsim.size,), I01)]).T, #[J m-2 d-1] Average solar irradiation (6 AM to 6 PM) or 3,749,760 Joules per square meter per day (J m−2
     'T':np.array([tsim, Tavg1]).T,
     'DVS':np.array([tsim, np.linspace(0, 2.5, 120+1)]).T, #to simulate the information flows from rice growth
     'SNH4': np.array([tsim, SNH4]).T,
     'SNO2': np.array([tsim, SNO2]).T,
     'SNO3': np.array([tsim, SNO3]).T,
     'SP': np.array([tsim, SP]).T,
     'Rain': np.array([tsim, Rain1]).T
     }
#Nitrate and available phosphorus for rice
SNO3_rice = SNO3*rice_area
SP_rice = SP*rice_area

drice1 = {
    "I0": np.array([t_weather, I01]).T,
    "T": np.array([t_weather, Tavg1]).T,
    "CO2": np.array([tsim, np.full((tsim.size,), 400)]).T,
    "SNO3": np.array([tsim, SNO3_rice]).T,
    "SP": np.array([tsim, SP_rice]).T,
    
}
#%%state variables
##phytoplankton model
x0phy = {'Mphy': 4.29e-5, #[g m-3] from Mei et al 2023 concentration of Chla multiply with Chla:phyto mass ratio from Jamu and Piedrahita (2002)
      'NA':   0.01 #[g m-3] TN from Mei et al 2023
      } 

# x0fish = {
#     "Mfish": m_fish,
#     "Mdig": 1e-6,
#     "Muri": 1e-6
#     }
x0fish = {
    "Mfish": m_fish*n_fish[0],
    "Mdig": 1e-6*n_fish[0],
    "Muri": 1e-6*n_fish[0]
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
#%%Model parameters
#define the parameter values in the dictionary p
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
    'cl': 0.004,
    'Ksd': 40000 #[g C/m3] 
}

price = {
   #Developmental rate according to Agustiani 2018
       'DVSi' : 0,
       'DVRJ': 0.0008753,
       'DVRI': 0.0007576,
       'DVRP': 0.0007787,
   #Manually calibrated developmental rate
       'DVRR': 0.0028,
    "Tmax": 40,
    "Tmin": 15,
    "Topt": 33,
    "k": 0.4,
    "mc_rt": 0.01,
    "mc_st": 0.015,
    "mc_lv": 0.02,
    "mc_pa": 0.003,
    "N_lv": 0,
    "Rec_N": 0.5,
    "k_lv_N": 0,
    "k_pa_maxN": 0.0175,
    "M_upN": 8,
    "cr_lv": 1.326,
    "cr_st": 1.326,
    "cr_pa": 1.462,
    "cr_rt": 1.326,
    "IgN": 0.8,
}

#initialize object
phyto = Phygrowth(tsim, dt, x0phy, pphy)
fish0 = Fish(tsim, dt, {key: val*0 for key, val in x0fish.items()}, pfish)
fish1 = Fish(tsim, dt, x0fish, pfish)
fish2 = Fish(tsim, dt, x0fish, pfish)
rice = Rice(tsim, dt, x0rice, price)
flow_rice = rice.f
flow_fish = fish.f

#%% Simulations
# inorganic fertilizer types and concentration:
# Nitrogen source: NPK (15%:15%:15%), Urea (46% N)
NPK_w = 167 * rice_area
Urea = 200 * rice_area
I_N = (15 / 100) * NPK_w + (46/100)*Urea

# Phosphorus source: SP36 (36%P2O5 ~ 7.85% P)
# P content in SP-36: 7.85% P
SP36 = 31 * rice_area
I_P = (7.85 / 100) * SP36

#initialize history storage
dfish1_ = {'Mphy':[]}
dphy_ = {'SNH4': [], 'SP': [], 'DVS': []} 
drice_ = {'SNO3': [], 'SP': []}
yphy_ = {'t': [0, ], 'Mphy': [4.29e-5*pond_volume, ], 'NA':[0.1, ]}
yfish_ = {'t': [0, ], 'Mfish': [0, ], 'Mdig':[0, ], 'Muri':[0, ], 'f_N_prt': [0, ], 'f_P_prt': [0, ], 'f_TAN': [0, ], 'f_P_sol': [0, ] }
yrice_ = {'t': [0, ], 'Mrt': [0.005, ], 'Mst': [0.003, ], 'Mlv': [0.002, ], 'Mpa': [0, ], 'Mgr': [0, ], 'HU': [0, ], 'DVS': [0, ]}

#copy of rice disturbances
SNO3_ori = drice1['SNO3'].copy()
SP_ori = drice1['SP'].copy()

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
    urice = {"I_N": I_N, "I_P": I_P} 
    # run phyto model
    yphy = phyto.run(tspan, d=dphy)
    yphy_['t'].append(yphy['t'][1])
    yphy_['Mphy'].append(yphy['Mphy'][1])
    yphy_['NA'].append(yphy['NA'][1])
    
    #run fish model
    yfish = fish.run((0, 20), dfish1, ufish)
    yfish_['t'].append(yfish['t'][1])
    yfish_['Mfish'].append(yfish['Mfish'][1])
    yfish_['Mdig'].append(yfish['Mdig'][1])
    yfish_['Muri'].append(yfish['Muri'][1])
    yfish_['f_N_prt'].append(yfish['f_N_prt'][1])
    yfish_['f_P_prt'].append(yfish['f_P_prt'][1])
    yfish_['f_TAN'].append(yfish['f_TAN'][1])
    yfish_['f_P_sol'].append(yfish['f_P_sol'][1])

    #run rice model
    if t>= 21:
        yrice = rice.run(tspan, drice1, urice)
        yrice_['t'].append(yrice['t'][1])
        yrice_['Mgr'].append(yrice['Mgr'][1])
        yrice_['Mrt'].append(yrice['Mrt'][1])
        yrice_['Mst'].append(yrice['Mst'][1])
        yrice_['Mlv'].append(yrice['Mlv'][1])
        yrice_['Mpa'].append(yrice['Mpa'][1])
        yrice_['HU'].append(yrice['HU'][1])
        yrice_['DVS'].append(yrice['DVS'][1])
    #retrieve yrice results as phytoplankton disturbances
    dphy_['DVS'].append(np.array([yrice['t'][1], rice.f['DVS'][1]]))
    dphy['DVS'] = np.array([yrice['t'], yrice['DVS']])
    
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
        
    
def replace_values(original, new_values):
    # Ensure the original has enough elements to replace (at least 121)
    if len(original) >= 121:
        # Replace indices 1 to 120
        original[1:121] = new_values

# Apply this function to each key in your dictionaries
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
        
# for key in drice1.keys():
#     if key in drice_:
#         replace_values(drice1[key], drice_[key])    

for key, val in yfish_.items():
    yfish_[key] = np.array(val)
    
for key, val in yphy_.items():
    yphy_[key] = np.array(val)
    
for key, val in yrice_.items():
    yrice_[key] = np.array(val)

#%%Simulation
yphy = yphy_
yfish = yfish_
yrice = yrice_

plt.rcParams.update({
    'font.size': 14,        # Global font size
    'axes.titlesize': 16,   # Title font size
    'axes.labelsize': 14,   # X and Y axis labels font size
    'xtick.labelsize': 12,  # X tick labels font size
    'ytick.labelsize': 12,  # Y tick labels font size
    'legend.fontsize': 12,  # Legend font size
    'figure.titlesize': 18  # Figure title font size
})
#retrieve results
t= yphy['t']
Mphy= yphy['Mphy']
NA = yphy['NA']
Mfish= yfish['Mfish']*0.001
Muri= yfish['Muri']*0.001
Mfis_fr= Mfish/pfish['k_DMR']
Mdig= yfish['Mdig']*0.001
rphy = fish.f['r_phy']

#rice
t_rice = yrice['t']
Mgr  = yrice['Mgr']*0.001 #ton
Mrt = yrice['Mrt']
Mlv = yrice['Mlv']
Mst = yrice['Mst']
Mpa = yrice['Mpa']
f_Ph= rice.f['f_Ph']
f_res= rice.f['f_res']
f_gr = rice.f['f_gr']
f_dmv = rice.f['f_dmv']
f_pN = rice.f['f_pN']
f_Nlv = rice.f['f_Nlv']

# Plot results
plt.figure(figsize=(10,10))
plt.figure(1)
plt.plot(t[25:80], Mfish[25:80], label='Fish dry weight')
# plt.plot(t, Mdig, label='Fish digestive system weight')
# plt.plot(t, Muri, label='Fish urinary system weight')
# plt.plot(t, Mfis_fr, label='Fish fresh weight')
plt.legend()
plt.xlabel(r'$time\ [d]$', fontsize = 18)
plt.ylabel(r'biomass accumulation $[kg DM d^{-1}]$', fontsize = 18)

Mphy_pond = Mphy*pond_volume
plt.figure(2)
plt.bar('Fish dry weight', Mfish*2.5*0.001)
plt.bar('Fish fresh weight', Mfis_fr*2.5*0.001)
plt.ylabel('Yield $[kg DM ha^{-1}]$')

fig, ax2 = plt.subplots(figsize=(10,10))
color = 'tab:blue'
ax2.plot(t, flow_fish['f_fed'], label='Total Feed Intake Rate', color=color, linestyle='-', marker='o')
ax2.axhline(ufish['Mfed'], label='Artificial Feed', color='orange', linestyle='--')
ax2.set_xlabel(r'$time\ [d]$')
ax2.set_ylabel(r'$flow rate\ [g \ day^{-1}]$', color=color)
ax2.tick_params(axis='y', labelcolor=color, labelsize=12)
ax2.legend(loc='lower right')
# Creating a secondary y-axis for phytoplankton growth rate
ax3 = ax2.twinx()
color = 'tab:green'
ax3.plot(t, Mphy, label='Phytoplankton', color=color, linestyle='-', marker='s')
ax3.set_ylabel(r'$growth rate\ [g \ m^{-3} \ day^{-1}]$', color=color)
ax3.tick_params(axis='y', labelcolor=color, labelsize=12)
ax3.legend(loc='upper right')


# to get the value of Mphy and NA in g, multiply it with pond_volume
f1 = phyto.f['f1']
f2 = phyto.f['f2']
f3 = phyto.f['f3']
f4 = phyto.f['f4']
f5 = phyto.f['f5']

# Plot
plt.figure(5)
plt.plot(t, Mphy, label='Phytoplankton')
plt.plot(t, NA, label='nutrient availability')
plt.xlabel(r'time [d]')
plt.ylabel(r'Flow rate $[g m^{-3} d^{-1}]$')
# plt.title(r'Accumulative Phytoplankton Growth and Nutrient Availability')
plt.legend()

plt.figure(6)
plt.plot(t, f4, label='Nutrient available in Pond')
plt.plot(t, f5, label= 'Nutrient uptake by phytoplankton')
plt.xlabel(r'time [d]')
plt.ylabel(r'Flow rate $[g m^{-3} d^{-1}]$')
plt.title(r'Flow in Phytoplankton Growth')
plt.legend()

plt.figure(7)
plt.plot(t, yfish['f_P_sol'], label='P content in soluble excretion')
plt.plot(t, yfish['f_TAN'], label='soluble TAN')
plt.plot(t, yfish['f_N_prt'], label='N content in solid excretion')
plt.plot(t, yfish['f_P_prt'], label='P content in solid excretion rate')
plt.xlabel(r'time [d]')
plt.ylabel(r'Flow rate $[g d^{-1}]$')
plt.title(r'Flow in Phytoplankton Growth')
plt.legend()

Raff = fish.f['Raff']
Rphy = fish.f['Rphy']
Rdet = fish.f['Rdet']

plt.figure(8)
plt.plot(t[1:],Rphy[1:], label='Phytoplankton')
plt.plot(t[1:], Raff[1:], label='Artifical feed')
plt.plot(t[1:], Rdet[1:], label='Detritus')
plt.xlabel(r'time [d]')
plt.ylabel(r'Feeding rate $[g m^{-3} d^{-1}]$')
plt.title("Feeding Composition")
plt.title(r'Phytoplankton feeding preferences')
plt.legend()

plt.figure(9)
plt.plot(t, Mrt, label='roots')
plt.plot(t, Mst, label='Stems')
plt.plot(t, Mlv, label='Leaves')
plt.plot(t, Mpa, label='Panicles')
plt.xlabel(r'time [d]')
plt.ylabel(r'Dry mass [kg DM ha-1 d-1]')
plt.legend()

plt.figure(10)
plt.plot(t, f_Ph, label='photosynthesis')
plt.plot(t, f_res, label='maintenance respiration')
plt.plot(t, f_gr, label='growth respiration')
plt.plot(t, f_dmv, label='death leaf rate')
plt.xlabel(r'time [d]')
plt.ylabel(r'flow rate $[kg CH2O ha^-{1} d^{-1}]$')
plt.title(r'Flow rate in rice plants')
plt.legend()

plt.figure(11)
plt.plot(t, f_Nlv, label='N flow rate in leaves')
plt.plot(t, f_pN, label = 'N flow rate in rice plants')
plt.xlabel(r'time [d]')
plt.ylabel(r'flow rate $[kg N ha^{-1} d^{-1}]$')
plt.title(r'Flow rate in rice plants')
plt.legend()

n_rice1 = 127980  # number of plants using 2:1 pattern
n_rice2 = 153600  # number of plants using 4:1 pattern
rice1 = Mgr*0.6/n_rice1 #[kg/plant] productivity per plant if using
rice2 = Mgr*0.6/n_rice2 #[kg/plant]
#productivity in ton/ha
pro_rice1 = rice1*n_rice1/1000
pro_rice2 = rice2*n_rice2/1000

# plt.figure(12)
# plt.bar('Rice', pro_rice1, color='orange')
# plt.bar('Nile tilapia', Mfis_fr/1000, color='blue')
# plt.ylabel(r'Yield $[ton ha^{-1}]$')
# # plt.title(r'Rice and Fish Productivity')
# plt.legend()

# fig, ax12 = plt.subplots()
# ax12.bar([0], Mgr*n_rice1, label='Grains', color='orange')
# ax13 = ax12.twinx()
# ax13.bar([1], Mfis_fr, label='Fish', color='blue')
# ax12.set_xlabel(r'time [d]')
# ax12.set_ylabel(r'Yield $[ton DM ha^{-1}]$')
# ax13.set_ylabel(r'Yield $[kg DM ha^{-1}]$')
# # plt.title(r'Rice and Fish Productivity')
# plt.legend()
plt.show()
