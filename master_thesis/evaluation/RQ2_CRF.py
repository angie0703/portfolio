# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 09:08:22 2024

@author: Angela

The code purpose is to simulate one type of CRF farm, one season (wet/dry), at a time 
 
"""

from models.fishphyto import FishPhy
from models.rice import Rice
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')

# Simulation time array
t = 120
tsim = np.linspace(0, 120, 120+1)
t_weather = np.linspace(0.0, t, t+1) 
dt = 1
tspan = (tsim[0], tsim[-1])
#%%# pond area and volume
area = 10000  # [m2] the total area of the system
rice_area = 6000  # [m2] rice field area
pond_area = area - rice_area  # [m2] pond area
pond_volume = pond_area*0.5
m_fish = 50 #[g] average weight of one fish fry
n_fish = [4000, 8000, 12000] #[no of fish] according to Pratiwi 2019
n_rice = [127980] #number of plants in 0.6 ha land


#%% Disturbances
data_weather = 'C:/Users/alegn/Documents/WUR/Thesis/rfmodel/data/Weather/Magelang_200110_200209_Daily.csv'
weather = pd.read_csv(data_weather, header=0, sep=';')
weather['Time'] = pd.to_datetime(weather['Time'], format='%Y%m%d')
weather.set_index('Time', inplace=True)

data_pond = 'C:/Users/alegn/Documents/WUR/Thesis/rfmodel/data/Bacteria/data_bacteria_tilapia.csv'
pond = pd.read_csv(data_pond, header=0, sep=';')
t_pond = [0, 14, 28, 42, 56, 70] #[d]
SNH4_ = pond.loc[t_pond[0]:t_pond[-1], 'NH4 (g m-3)'].values
SNO3_ = pond.loc[t_pond[0]:t_pond[-1], 'NO3 (g m-3)'].values
SNO2_ = pond.loc[t_pond[0]:t_pond[-1], 'NO2 (g m-3)'].values
SP_  = pond.loc[t_pond[0]:t_pond[-1], 'SRP (g m-3)'].values

SNH4 =np.interp(tsim, t_pond, SNH4_)
SNO3 =np.interp(tsim, t_pond, SNO3_)
SNO2 =np.interp(tsim, t_pond, SNO2_)
SP =np.interp(tsim, t_pond, SP_)

#FIRST CYCLE
t_ini1 = '20011001'
t_end1 = '20020129'

#SECOND CYCLE
t_ini2 = '20020130'
t_end2 = '20020530'

#first cycle
Tavg1 = weather.loc[t_ini1:t_end1,'Tavg'].values #[°C] Mean daily temperature
Rain1 = weather.loc[t_ini1:t_end1,'Rain'].values #[mm] Daily precipitation
Igl1 = weather.loc[t_ini1:t_end1, 'I0'].values #[MJ m-2] Sum of shortwave radiation daily

I01 = 0.45*Igl1*1E6 #Convert [MJ m-2 d-1] to [J m-2 d-1] PAR

#second cycle
Tavg2 = weather.loc[t_ini2:t_end2,'Tavg'].values #[°C] Mean daily temperature
Rain2 = weather.loc[t_ini2:t_end2,'Rain'].values #[mm] Daily precipitation
Igl2 = weather.loc[t_ini2:t_end2, 'I0'].values #[MJ m-2] Sum of shortwave radiation daily

I02 = 0.45*Igl2*1E6 #Convert [MJ m-2 d-1] to [J m-2 d-1] PAR

#For wet season
dfp = {
    "DO": np.array([tsim, np.linspace(4.9, 2.43, 120+1)]).T,
    "T": np.array([t_weather, Tavg1]).T,
    'I0' :  np.array([tsim, np.full((tsim.size,), I01)]).T, #[J m-2 d-1] Average solar irradiation (6 AM to 6 PM) or 3,749,760 Joules per square meter per day (J m−2
    'DVS':np.array([tsim, np.linspace(0, 2.5, 120+1)]).T, #to simulate the information flows from rice growth
    'Rain': np.array([tsim, Rain1]).T
}
#Nitrate and available phosphorus for rice
SNO3_rice = SNO3*rice_area*0.2
SP_rice = SP*rice_area*0.2
drice = {
    "I0": np.array([t_weather, I01]).T,
    "T": np.array([t_weather, Tavg1]).T,
    "CO2": np.array([tsim, np.full((tsim.size,), 400)]).T,
    "SNO3": np.array([tsim, SNO3_rice]).T,
    "SP": np.array([tsim, SP_rice]).T,
    }

#%%state variables
x0fp = {
    "Mfish": m_fish*n_fish[0],
    "Mdig": 1e-6*n_fish[0],
    "Muri": 1e-6*n_fish[0],
    'Mphy': 4.29e-5, #[g m-3] from Mei et al 2023 concentration of Chla multiply with Chla:phyto mass ratio from Jamu and Piedrahita (2002)
          'NA':   0.01 #[g m-3] TN from Mei et al 2023
    }
N_rice = n_rice[0]
x0rice = {
    "Mrt": 1e-6,  # [kg DM ha-1 d-1] Dry biomass of root
    "Mst": 1e-6,  # [kg DM ha-1 d-1] Dry biomass of stems
    "Mlv": 1e-6,  # [kg DM ha-1 d-1] Dry biomass of leaves
    "Mpa": 0.0,  # [kg DM ha-1 d-1] Dry biomass of panicles
    "Mgr": 0.0,  # [kg DM ha-1 d-1] Dry biomass of grains
    "HU": 0.0,
    "DVS": 0,
}
#%%Model parameters
#define the parameter values in the dictionary p
pfp = {
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
    'Ksd': 40000, #[g C/m3]
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

price = {
   # #Developmental rate according to Agustiani 2018
   #     'DVSi' : 0,
        # 'DVRJ': 0.0008753,
        # 'DVRI': 0.0007576,
        # 'DVRP': 0.0007787,
        # 'DVRR': 0.0015390,
   # #Manually calibrated developmental rate
        'DVSi': 0,
         'DVRJ': 0.0020,
         'DVRI': 0.00195,
         'DVRP': 0.00195,     
         'DVRR': 0.0024,
   # developmental rate according to Bouwman
    # 'DVSi' : 0,
    # 'DVRJ': 0.0013,
    # 'DVRI': 0.001125,
    # 'DVRP': 0.001275,
    # 'DVRR': 0.003,    
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
    'n_rice': N_rice,
    'gamma': 65000 #[number of grains/kg DM]
}

#initialize object
fp = FishPhy(tsim, dt, x0fp, pfp)
rice = Rice(tsim, dt, x0rice, price)
flow_rice = rice.f
flow_fp = fp.f

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

dfp_ = {'DVS': []}
drice_ = {'SNO3': [], 'SP': []}
yfp_ = {'t': [0, ], 'Mfish': [0, ], 'Mdig':[0, ], 'Muri':[0, ], 'f_N_prt': [0, ], 'f_P_prt': [0, ], 'f_TAN': [0, ], 'f_P_sol': [0, ], 'f_N_sol': [0, ], 'Mphy': [4.29e-5, ], 'NA':[0.1, ] }
yrice_ = {'t': [0, ], 'Mrt': [1e-6, ], 'Mst': [1e-6, ], 'Mlv': [1e-6, ], 'Mpa': [0, ], 'Mgr': [0, ], 'HU': [0, ], 'DVS': [0, ]}

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
    #run fishphyto model
    yfp = fp.run(tspan, dfp, ufish)
    yfp_['t'].append(yfp['t'][1])
    yfp_['Mfish'].append(yfp['Mfish'][1])
    yfp_['Mdig'].append(yfp['Mdig'][1])
    yfp_['Muri'].append(yfp['Muri'][1])
    yfp_['f_N_prt'].append(yfp['f_N_prt'][1])
    yfp_['f_P_prt'].append(yfp['f_P_prt'][1])
    yfp_['f_TAN'].append(yfp['f_TAN'][1])
    yfp_['f_P_sol'].append(yfp['f_P_sol'][1])
    yfp_['f_N_sol'].append(yfp['f_N_sol'][1])
    yfp_['Mphy'].append(yfp['Mphy'][1])
    yfp_['NA'].append(yfp['NA'][1])
    

    #retrieve simulation result for rice
    drice_['SNO3'].append(np.array([yfp['t'][1], yfp['f_TAN'][1]]))
    drice['SNO3'] = np.array([yfp['t'], yfp['f_TAN']])
    drice_['SP'].append(np.array([yfp['t'][1], yfp['f_P_sol'][1]]))
    drice['SP'] = np.array([yfp['t'], yfp['f_P_sol']])
    #run rice model
    yrice = rice.run(tspan, drice, urice)
    yrice_['t'].append(yrice['t'][1])
    yrice_['Mgr'].append(yrice['Mgr'][1])
    yrice_['Mrt'].append(yrice['Mrt'][1])
    yrice_['Mst'].append(yrice['Mst'][1])
    yrice_['Mlv'].append(yrice['Mlv'][1])
    yrice_['Mpa'].append(yrice['Mpa'][1])
    yrice_['HU'].append(yrice['HU'][1])
    yrice_['DVS'].append(yrice['DVS'][1])
    #retrieve simulation model for fishphyto
    dfp_['DVS'].append(np.array([yrice['t'][1], yrice['DVS'][1]]))
    dfp['DVS'] = np.array([yrice['t'], yrice['DVS']]).T

def replace_values(original, new_values):
    # Ensure the original has enough elements to replace (at least 121)
    if len(original) >= 121:
        # Replace indices 1 to 120
        original[1:121] = new_values

# Apply this function to each key in your dictionaries
for key, val in dfp_.items():
    dfp_[key] = np.array(val)
for key, val in drice_.items():
    drice_[key] = np.array(val)

for key in dfp.keys():
    if key in dfp_:
        replace_values(dfp[key], dfp_[key])    
        
for key in drice.keys():
    if key in drice_:
        replace_values(drice[key], drice_[key])    

for key, val in yfp_.items():
    yfp_[key] = np.array(val)
    
for key, val in yrice_.items():
    yrice_[key] = np.array(val)

yrice = yrice_
yfp = yfp_ 
   
#%%Simulation

plt.rcParams.update({
    'font.size': 14,        # Global font size
    'axes.titlesize': 16,   # Title font size
    'axes.labelsize': 14,   # X and Y axis labels font size
    'xtick.labelsize': 14,  # X tick labels font size
    'ytick.labelsize': 14,  # Y tick labels font size
    'legend.fontsize': 14,  # Legend font size
    'figure.titlesize': 18  # Figure title font size
})
#retrieve results
t= yrice['t']
t_fish = t[24:80]
Mfish= yfp['Mfish']*0.001
Mfis_fr= Mfish[24:80]/pfp['k_DMR']
Mfis_tonha = Mfis_fr*0.001/0.4
Mphy = yfp['Mphy']*4000*0.5 #[g d-1]
NA = yfp['NA']
r_phy = fp.f['r_phy'][24:80]
r_det = fp.f['r_det'][24:80]
r_aff = fp.f['r_aff'][24:80]

#rice
t_rice = yrice['t']
Mgr  = yrice['Mgr']*0.001 #ton
Mgr_tonha = Mgr/0.6 #[ton/ha]
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
LAI = rice.f['LAI']
NumGrain = np.sum(flow_rice['NumGrain'])
#%%
# # # Plot results
plt.figure(figsize=(10,10))
plt.figure(1)
plt.plot(t_fish, Mfis_tonha, label='Fish fresh weight')
plt.legend()
plt.xlabel(r'$time\ [d]$', fontsize = 18)
plt.ylabel(r'biomass accumulation $[ton d^{-1}]$', fontsize = 18)

# # to get the value of Mphy and NA in g, multiply it with pond_volume
f1 = fp.f['f1']
f2 = fp.f['f2']
f3 = fp.f['f3']
f4 = fp.f['f4']
f5 = fp.f['f5']

# Plot
plt.figure(5)
plt.plot(t, Mphy, label='Phytoplankton Biomass')
plt.plot(t, NA, label='nutrient availability')
plt.xlabel(r'time [d]')
plt.ylabel(r'Concentration $[g m^{-3} d^{-1}]$')
# plt.title(r'Accumulative Phytoplankton Growth and Nutrient Availability')
plt.legend()

# plt.figure(5)
# plt.plot(t, f1, label = 'Phytoplankton Growth')
# plt.plot(t, f2, label='mortality by predation')
# plt.plot(t, f3, label='mortality by intraspecific competition')
# plt.xlabel(r'time [d]')
# plt.ylabel(r'Concentration $[g m^{-3} d^{-1}]$')
# # plt.title(r'Accumulative Phytoplankton Growth and Nutrient Availability')
# plt.legend()


# plt.figure(6)
# plt.plot(t, f4, label='Nutrient available in Pond')
# plt.plot(t, f5, label= 'Nutrient uptake by phytoplankton')
# plt.xlabel(r'time [d]')
# plt.ylabel(r'Flow rate $[g m^{-3} d^{-1}]$')
# plt.title(r'Flow in Phytoplankton Growth')
# plt.legend()

# fig, ax2 = plt.subplots(figsize=(10,10))
# color = 'tab:blue'
# ax2.plot(t, flow_fp['f_fed'], label='Total Feed Intake Rate', color=color, linestyle='-', marker='o')
# ax2.axhline(ufish['Mfed'], label='Artificial Feed', color='orange', linestyle='--')
# ax2.set_xlabel(r'$time\ [d]$')
# ax2.set_ylabel(r'$flow rate\ [g \ day^{-1}]$', color=color)
# ax2.tick_params(axis='y', labelcolor=color, labelsize=12)
# ax2.legend(loc='upper left')
# # Creating a secondary y-axis for phytoplankton growth rate
# ax3 = ax2.twinx()
# color = 'tab:green'
# ax3.plot(t, Mphy, label='Phytoplankton', color=color, linestyle='-', marker='s')
# ax3.set_ylabel(r'$Concentration \ [g \ m^{-3} \ day^{-1}]$', color=color)
# ax3.tick_params(axis='y', labelcolor=color, labelsize=12)
# ax3.legend(loc='upper right')

plt.figure(4)
plt.plot(t[25:80], yfp['f_P_sol'][25:80], label='P content in fish urine')
plt.plot(t[25:80], yfp['f_N_sol'][25:80], label='N content in fish urine')
plt.plot(t[25:80], yfp['f_N_prt'][25:80], label='N content in faeces')
plt.plot(t[25:80], yfp['f_P_prt'][25:80], label='P content in faeces')
plt.xlabel(r'time [d]')
plt.ylabel(r'Flow rate $[g d^{-1}]$')
# plt.title(r'Flow from Nile Tilapia Growth')
plt.legend()

# plt.figure(15)
# plt.plot(t_fish,r_phy, label='Phytoplankton')
# plt.plot(t_fish, r_aff, label='Artifical feed')
# # plt.axvline(x=75, color='b', linestyle='--', linewidth=1)
# # plt.annotate('Day 75', xy=(75, 0), xytext=(75, 0),
# #               fontsize=12, ha='center')
# # plt.plot(t_fish, r_det, label='Detritus')
# plt.xlabel(r'time [d]')
# plt.ylabel(r'Feed Composition ratio [-]')
# # plt.title("Feeding Composition")
# # plt.title(r'Phytoplankton feeding preferences')
# plt.legend()

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
# plt.title(r'Flow rate in rice plants')
plt.legend()

plt.figure(11)
plt.plot(t, f_Nlv, label='N flow rate in leaves')
plt.plot(t, f_pN, label = 'N flow rate in rice plants')
plt.xlabel(r'time [d]')
plt.ylabel(r'flow rate $[kg N ha^{-1} d^{-1}]$')
plt.title(r'Flow rate in rice plants')
plt.legend()

plt.figure(12)
plt.plot(t, yrice['DVS'])
y_min, y_max = plt.ylim()
plt.axvline(x=0, color='b', linestyle='--', linewidth=1)
plt.annotate('DVS=0', xy=(0, y_min), xytext=(0, y_min),
              fontsize=12, ha='center')
plt.axvline(x=37.24, color='b', linestyle='--', linewidth=1)
plt.annotate('DVS=0.4', xy=(37.24, y_min), xytext=(37.24, y_min),
              fontsize=12, ha='center')
plt.axvline(x=45.2, color='b', linestyle='--', linewidth=1)
plt.annotate('DVS=0.65', xy=(45.2, 0), xytext=(45.2, 0.1),
              fontsize=12, ha='center')
plt.axvline(x=63.5, color='b', linestyle='--', linewidth=1)
plt.annotate('DVS=1', xy=(63.5, 0), xytext=(63.5, y_min),
              fontsize=12, ha='center')
plt.axvline(x=116.7, color='b', linestyle='--', linewidth=1)
plt.annotate('DVS=2', xy=(116.7, 0), xytext=(116.7, y_min),
              fontsize=12, ha='center')
plt.xlabel(r'time [d]')
plt.ylabel(r'DVS [-]')
plt.text(0, 2, 'x=Day 0', ha='center')
plt.text(37.24, 2, 'x ~ Day 37', ha='center')
plt.text(45.2, 1.4, 'x ~ Day 45', ha='center')
plt.text(63.5, 2, 'x ~ Day 63', ha='center')
plt.text(116.7, 2, 'x ~ Day 116', ha='center')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.figure(13)
plt.bar('Rice', Mgr_tonha, color='orange')
plt.bar('Nile tilapia', Mfis_tonha, color='blue')
plt.ylabel(r'Yield $[ton ha^{-1}]$')
plt.legend()

plt.figure(14)
plt.plot(flow_rice['NumGrain'])
plt.ylabel(r'number grains $[ha^{−1} d^{−1}]$')
plt.legend()
