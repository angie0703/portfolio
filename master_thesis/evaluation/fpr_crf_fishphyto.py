# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 09:08:22 2024

@author: alegn
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
m_fish = 30 #[g] average weight of one fish fry
# N_fish = [14.56, 18.2, 21.84]  # [g m-3]
# n_fish = [(N * pond_volume) / m_fish for N in N_fish]
n_fish = [4000, 8000, 12000] #[no of fish] according to Pratiwi 2019
n_rice = [127980, 153600] #number of plants in 0.6 ha land

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
    "DO": np.array([tsim, np.linspace(4.9, 2.47, 120+1)]).T,
    "T": np.array([t_weather, Tavg1]).T,
    'I0' :  np.array([tsim, np.full((tsim.size,), I01)]).T, #[J m-2 d-1] Average solar irradiation (6 AM to 6 PM) or 3,749,760 Joules per square meter per day (J m−2
    'DVS':np.array([tsim, np.linspace(0, 2.5, 120+1)]).T, #to simulate the information flows from rice growth
    'Rain': np.array([tsim, Rain1]).T
}

drice = {
    "I0": np.array([t_weather, I01]).T,
    "T": np.array([t_weather, Tavg1]).T,
    "CO2": np.array([tsim, np.full((tsim.size,), 400)]).T,
    "SNO3": np.array([tsim, SNO3]).T,
    "SP": np.array([tsim, SP]).T,
    }

# For dry season
# dfish = {
#     "DO": np.array([tsim, np.linspace(4.9, 2.47, 120+1)]).T,
#     "T": np.array([t_weather, Tavg2]).T,
#     "Mphy": np.array([tsim, np.full((tsim.size,), 4.29e-5)]).T,
# }
#n_fish 4000 -> 2.43 mg/L, n_fish 8000 -> 1.47 mg/L, n_fish 12000 -> 1.17 mg/L

# drice = {
#     "I0": np.array([t_weather, I02]).T,
#     "T": np.array([t_weather, Tavg2]).T,
#     "CO2": np.array([tsim, np.full((tsim.size,), 400)]).T,
#     "SNO3": np.array([tsim, SNO3]).T,
#     "SP": np.array([tsim, SP]).T,

# dphy = {
#      'I0' :  np.array([tsim, np.full((tsim.size,), I02)]).T, #[J m-2 d-1] Average solar irradiation (6 AM to 6 PM) or 3,749,760 Joules per square meter per day (J m−2
#      'T':np.array([tsim, Tavg2]).T,
#      'DVS':np.array([tsim, np.linspace(0, 2.5, 120+1)]).T, #to simulate the information flows from rice growth
#      'SNH4': np.array([tsim, SNH4]).T,
#      'SNO2': np.array([tsim, SNO2]).T,
#      'SNO3': np.array([tsim, SNO3]).T,
#      'SP': np.array([tsim, SP]).T,
#      'Rain': np.array([tsim, Rain2]).T
#      }

#%%state variables
##phytoplankton model
x0fp = {
    "Mfish": m_fish*n_fish[0],
    "Mdig": 1e-6*n_fish[0],
    "Muri": 1e-6*n_fish[0],
    'Mphy': 4.29e-5, #[g m-3] from Mei et al 2023 concentration of Chla multiply with Chla:phyto mass ratio from Jamu and Piedrahita (2002)
          'NA':   0.01 #[g m-3] TN from Mei et al 2023
    }
x0rice = {
    "Mrt": 1e-6*n_rice[0],  # [kg DM ha-1 d-1] Dry biomass of root
    "Mst": 1e-6*n_rice[0],  # [kg DM ha-1 d-1] Dry biomass of stems
    "Mlv": 1e-6*n_rice[0],  # [kg DM ha-1 d-1] Dry biomass of leaves
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
        'DVRJ': 0.004,    
       'DVRP': 0.004,     
       'DVRR': 0.0024,
   # developmental rate according to Bouwman
    'DVSi' : 0,
    # 'DVRJ': 0.0013,
    'DVRI': 0.001125,
    # 'DVRP': 0.001275,
    # 'DVRR': 0.003,    
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
    "IgN": 0.8, # Indigenous soil N supply during wet season
    # "IgN": 0.5, # Indigenous soil N supply during dry season
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
urice = {'I_N':I_N, 'I_P': I_P}            # [kg m-2 d-1]
ufish = {'Mfed': m_fish*0.03*n_fish[0]}
yfp = fp.run(tspan, dfp, ufish)
drice
yrice = rice.run(tspan, drice, urice)
#%%Simulation

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
t= yrice['t']
t_fish = t[24:80]
Mfish= (yfp['Mfish']*0.001)-120
Mfis_fr= Mfish[24:80]/pfp['k_DMR']
Mphy = yfp['Mphy']
NA = yfp['NA']
r_phy = fp.f['r_phy'][24:80]
r_det = fp.f['r_det'][24:80]
r_aff = fp.f['r_aff'][24:80]

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

# # Plot results
plt.figure(figsize=(10,10))
plt.figure(1)
plt.plot(t_fish, Mfis_fr, label='Fish fresh weight')
plt.legend()
plt.xlabel(r'$time\ [d]$', fontsize = 18)
plt.ylabel(r'biomass accumulation $[kg DM d^{-1}]$', fontsize = 18)

# # to get the value of Mphy and NA in g, multiply it with pond_volume
f1 = fp.f['f1']
f2 = fp.f['f2']
f3 = fp.f['f3']
f4 = fp.f['f4']
f5 = fp.f['f5']

# # Plot
plt.figure(2)
plt.plot(t, Mphy, label='Phytoplankton')
plt.plot(t, NA, label='nutrient availability')
plt.xlabel(r'time [d]')
plt.ylabel(r'Flow rate $[g m^{-3} d^{-1}]$')
# plt.title(r'Accumulative Phytoplankton Growth and Nutrient Availability')
plt.legend()

# plt.figure(6)
# plt.plot(t, f4, label='Nutrient available in Pond')
# plt.plot(t, f5, label= 'Nutrient uptake by phytoplankton')
# plt.xlabel(r'time [d]')
# plt.ylabel(r'Flow rate $[g m^{-3} d^{-1}]$')
# plt.title(r'Flow in Phytoplankton Growth')
# plt.legend()

plt.figure(3)
plt.plot(t, yfp['f_P_sol'], label='P content in fish urine')
plt.plot(t, yfp['f_N_sol'], label='N content in fish urine')
plt.plot(t, yfp['f_N_prt'], label='N content in faeces')
plt.plot(t, yfp['f_P_prt'], label='P content in faeces')
plt.xlabel(r'time [d]')
plt.ylabel(r'Flow rate $[g d^{-1}]$')
plt.title(r'Flow from Nile Tilapia Growth')
plt.legend()

plt.figure(4)
plt.plot(t_fish,r_phy, label='Phytoplankton')
plt.plot(t_fish, r_aff, label='Artifical feed')
plt.plot(t_fish, r_det, label='Detritus')
plt.xlabel(r'time [d]')
plt.ylabel(r'Feeding rate $[g m^{-3} d^{-1}]$')
plt.title("Feeding Composition")
plt.title(r'Phytoplankton feeding preferences')
plt.legend()

plt.figure(5)
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

# plt.figure(13)
# plt.plot(t, yrice['DVS'])
# y_min, y_max = plt.ylim()
# plt.axvline(x=0, color='b', linestyle='--', linewidth=1)
# plt.annotate('DVS=0', xy=(0, y_min), xytext=(0, y_min),
#              fontsize=12, ha='center')
# plt.axvline(x=29.6419, color='b', linestyle='--', linewidth=1)
# plt.annotate('DVS=0.4', xy=(29.6419, y_min), xytext=(29.6419, y_min),
#              fontsize=12, ha='center')
# plt.axvline(x=40.5469, color='b', linestyle='--', linewidth=1)
# plt.annotate('DVS=0.65', xy=(40.5469, 0), xytext=(40.5469, 0.1),
#              fontsize=12, ha='center')
# plt.axvline(x=55.75, color='b', linestyle='--', linewidth=1)
# plt.annotate('DVS=1', xy=(55.75, 0), xytext=(55.75, y_min),
#              fontsize=12, ha='center')
# plt.axvline(x=101.23, color='b', linestyle='--', linewidth=1)
# plt.annotate('DVS=2', xy=(101.23, 0), xytext=(101.23, y_min),
#              fontsize=12, ha='center')
# plt.xlabel(r'time [d]')
# plt.ylabel(r'DVS [-]')
# plt.text(0, 2.5, 'x=Day 0', ha='center')
# plt.text(29, 2.5, 'x ~ Day 29', ha='center')
# plt.text(40, 2.4, 'x ~ Day 40', ha='center')
# plt.text(55, 2.5, 'x ~ Day 55', ha='center')
# plt.text(101, 2.5, 'x ~ Day 101', ha='center')
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)

# plt.figure(12)
# plt.bar('Rice', Mgr, color='orange')
# plt.bar('Nile tilapia', Mfis_fr/1000, color='blue')
# plt.ylabel(r'Yield $[ton ha^{-1}]$')
# # plt.title(r'Rice and Fish Productivity')
# plt.legend()

# # fig, ax12 = plt.subplots()
# # ax12.bar([0], Mgr*n_rice1, label='Grains', color='orange')
# # ax13 = ax12.twinx()
# # ax13.bar([1], Mfis_fr, label='Fish', color='blue')
# # ax12.set_xlabel(r'time [d]')
# # ax12.set_ylabel(r'Yield $[ton DM ha^{-1}]$')
# # ax13.set_ylabel(r'Yield $[kg DM ha^{-1}]$')
# # # plt.title(r'Rice and Fish Productivity')
# # plt.legend()
# plt.show()
