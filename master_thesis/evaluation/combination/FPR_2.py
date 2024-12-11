# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 09:08:22 2024

@author: Angela
 
"""

from models.fishphyto_backup import FishPhy
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
weather['Time'] = pd.to_datetime(weather['Time'], format='%Y%m%d')
weather.set_index('Time', inplace=True)

data_pond = 'C:/Users/alegn/Documents/WUR/Thesis/rfmodel/data/Bacteria/data_bacteria_tilapia.csv'
pond = pd.read_csv(data_pond, header=0, sep=';')
t_pond = [0, 14, 28, 42, 56, 70] #[d]
C_NH4_ = pond.loc[t_pond[0]:t_pond[-1], 'NH4 (g m-3)'].values
C_NO3_ = pond.loc[t_pond[0]:t_pond[-1], 'NO3 (g m-3)'].values
C_NO2_ = pond.loc[t_pond[0]:t_pond[-1], 'NO2 (g m-3)'].values
C_H2PO4_  = pond.loc[t_pond[0]:t_pond[-1], 'SRP (g m-3)'].values

C_NH4 =np.interp(tsim, t_pond, C_NH4_)
C_NO3 =np.interp(tsim, t_pond, C_NO3_)
C_NO2 =np.interp(tsim, t_pond, C_NO2_)
C_H2PO4 =np.interp(tsim, t_pond, C_H2PO4_)

#FIRST CYCLE
t_ini1 = '20011001'
t_end1 = '20020129'

#first cycle
Tavg1 = weather.loc[t_ini1:t_end1,'Tavg'].values #[°C] Mean daily temperature
Rain1 = weather.loc[t_ini1:t_end1,'Rain'].values #[mm] Daily precipitation
Igl1 = weather.loc[t_ini1:t_end1, 'I0'].values #[MJ m-2] Sum of shortwave radiation daily

I01 = 0.45*Igl1*1E6 #Convert [MJ m-2 d-1] to [J m-2 d-1] PAR

#For wet season
dfp = {
    "DO": np.array([tsim, np.linspace(4.9, 2.43, 120+1)]).T,
    "T": np.array([t_weather, Tavg1]).T,
    'I0' :  np.array([tsim, np.full((tsim.size,), I01)]).T, #[J m-2 d-1] Average solar irradiation (6 AM to 6 PM) or 3,749,760 Joules per square meter per day (J m−2
    'DVS':np.array([tsim, np.linspace(0, 2.5, 120+1)]).T, #to simulate the information flows from rice growth
    'Rain': np.array([tsim, Rain1]).T,
    'd_pond' : np.array([tsim, np.full((tsim.size,), 0.6)]).T, 
}

#Nitrate and available phosphorus for rice
SNO3_rice = C_NO3
SP_rice = C_H2PO4
drice = {
    "I0": np.array([t_weather, I01]).T,
    "T": np.array([t_weather, Tavg1]).T,
    "CO2": np.array([tsim, np.full((tsim.size,), 400)]).T,
    "C_NO3": np.array([tsim, SNO3_rice]).T,
    "C_H2PO4": np.array([tsim, SP_rice]).T,
    }

#%%state variables
x0fp = {
    "Mfish": m_fish*n_fish,
    "Mdig": 1e-6*n_fish,
    "Muri": 1e-6*n_fish,
    'Mphy': 1, #[g m-3] from Mei et al 2023 concentration of Chla multiply with Chla:phyto mass ratio from Jamu and Piedrahita (2002)
          # 'NA':   0.01, #[g m-3] TN from Mei et al 2023
          'N_deposit':   0.01, #[g m-3] N
          'P_deposit':   0.01, #[g m-3] P
          'Nfert_org': 1.23/0.6, #[g m-3] N content of organic fertilizer
          'Pfert_org': 1.39/0.6 #[g m-3] P content of organic fertilizer
    }

x0rice = {
    "Mrt": 1e-6,  # [kg DM ha-1 d-1] Dry biomass of root
    "Mst": 1e-6,  # [kg DM ha-1 d-1] Dry biomass of stems
    "Mlv": 1e-6,  # [kg DM ha-1 d-1] Dry biomass of leaves
    "Mpa": 0.0,  # [kg DM ha-1 d-1] Dry biomass of panicles
    "Mgr": 0.0,  # [kg DM ha-1 d-1] Dry biomass of grains
    "HU": 0.0,
    "DVS": 0,
    'Nfert_in': 0 
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
    "Tmin": 22,
    "Topt": 28,
    "Tmax": 32,
    'cl': 0.004,
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
    'n_rice': n_rice,
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
Urea = 100 * rice_area 
# I_N = 0.15 * NPK_w + 0.46*Urea
I_N1 = 0.15 * NPK_w 
I_N2 = 0.46*Urea
I_N = I_N1+I_N2
# Phosphorus source: SP36 (36%P2O5 ~ 7.85% P)
# P content in SP-36: 7.85% P
SP36 = 31 * rice_area
I_P = (7.85 / 100) * SP36

dfp_ = {'DVS': []}
drice_ = {'SNO3': [], 'SP': []}
yfp_ = {'t': [0, ], 'Mfish': [0, ], 'Mdig':[0, ], 'Muri':[0, ], 'Mphy': [4.29e-5, ], 'N_deposit':[0.1, ], 'P_deposit':[0.1, ], 'Nfert_org': [1.23/0.6, ], 'Pfert_org': [1.39/0.6, ] }
yrice_ = {'t': [0, ], 'Mrt': [1e-6, ], 'Mst': [1e-6, ], 'Mlv': [1e-6, ], 'Mpa': [0, ], 'Mgr': [0, ], 'HU': [0, ], 'DVS': [0, ], 'Nfert_in': [0, ]}

#run model
it = np.nditer(tsim[:-1], flags=['f_index'])
for ti in it:
    # Index for current time instant
    idx = it.index
    # Integration span
    tspan = (tsim[idx], tsim[idx+1])
    print('Integrating', tspan)
    # Controlled inputs
    ufish = {'Mfed': m_fish*0.03*n_fish/pond_volume} #[g m-3]
    urice = {"I_N": I_N, "I_P": I_P} 
    #run fishphyto model
    yfp = fp.run(tspan, dfp, ufish)
    yfp_['t'].append(yfp['t'][1])
    yfp_['Mfish'].append(yfp['Mfish'][1])
    yfp_['Mdig'].append(yfp['Mdig'][1])
    yfp_['Muri'].append(yfp['Muri'][1])
    yfp_['Mphy'].append(yfp['Mphy'][1])
    # yfp_['NA'].append(yfp['NA'][1])
    yfp_['N_deposit'].append(yfp['N_deposit'][1])
    yfp_['P_deposit'].append(yfp['P_deposit'][1])
    yfp_['Nfert_org'].append(yfp['Nfert_org'][1])
    yfp_['Pfert_org'].append(yfp['Pfert_org'][1])

    #retrieve simulation result for rice
    drice_['SNO3'].append(np.array([yfp['t'][1], yfp['N_deposit'][1]]))
    drice['SNO3'] = np.array([yfp['t'], yfp['N_deposit']])

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
    yrice_['Nfert_in'].append(yrice['Nfert_in'][1])
    #retrieve simulation model for fishphyto
    dfp_['DVS'].append(np.array([yrice['t'][1], yrice['DVS'][1]]))
    dfp['DVS'] = np.array([yrice['t'], yrice['DVS']]).T

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
t_fish = t
Mfish= yfp['Mfish']
Mdig = yfp['Mdig']
Muri = yfp['Muri']
Mfis_fr= Mfish[24:80]/pfp['k_DMR']
Mphy = yfp['Mphy']
Nfert_org = yfp['Nfert_org']
Pfert_org = yfp['Pfert_org']

#rice
t_rice = yrice['t']
Mgr  = yrice['Mgr']*0.001 #ton
Mgr_tonha = Mgr/0.6 #[ton/ha]
Mrt = yrice['Mrt']
Mlv = yrice['Mlv']
Mst = yrice['Mst']
Mpa = yrice['Mpa']
Nfert_in = yrice['Nfert_in']
f_Ph= rice.f['f_Ph']
f_res= rice.f['f_res']
f_gr = rice.f['f_gr']
f_dmv = rice.f['f_dmv']
f_pN = rice.f['f_pN']
f_Nlv = rice.f['f_Nlv']
f_uptN = rice.f['f_uptN']
#%%
# # # Plot results
# plt.figure(figsize=(10,10))
# plt.figure(1)
# plt.plot(t_fish, Mfis_tonha, label='Fish fresh weight')
# plt.legend()
# plt.xlabel(r'$time\ [d]$', fontsize = 18)
# plt.ylabel(r'biomass accumulation $[kg DM d^{-1}]$', fontsize = 18)

# # to get the value of Mphy and NA in g, multiply it with pond_volume
f1 = fp.f['f1']
f2 = fp.f['f2']
f3 = fp.f['f3']
f4 = fp.f['f4']
f5 = fp.f['f5']
f6 = fp.f['f6']
f7 = fp.f['f7']
f_fed_phy = fp.f['f_fed_phy']

IgN = np.array([tsim, np.full((tsim.size,), 0.8)]).T

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
# Plot the first subplot
ax1.plot(t, Mfish/pond_volume, label = '$M_{fish}$')
ax1.set_ylabel('concentration [$g m^{-3}$]')
ax1.legend()

# Plot the second subplot
ax2.plot(t, fp.f['f_upt']/pond_volume, label = '$\phi_{upt}$')
ax2.set_ylabel('concentration rate [$g m^{-3} d^{-1}$]')
ax2.set_xlabel('time [day]')
ax2.legend()

plt.figure(4)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
# Plot the first subplot
ax1.plot(t, Mdig/pond_volume, label = '$M_{dig}$')
ax1.set_ylabel('concentration [$g m^{-3}$]')
ax1.legend()

# Plot the second subplot
ax2.plot(t, fp.f['f_fed']/pond_volume, label ='$\phi_{fed}$')
ax2.plot(t, -fp.f['f_digout']/pond_volume, label ='$\phi_{digout}$')
ax2.set_ylabel(r'rate $[g m^{-3} d^{-1}]$')
ax2.set_xlabel('time [day]')
ax2.legend()

plt.figure(5)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
# Plot the first subplot
ax1.plot(t, Muri/pond_volume, label = '$M_{uri}$')
ax1.set_ylabel('concentration [$g m^{-3}$]')
ax1.legend()

# Plot the second subplot
ax2.plot(t, fp.f['f_diguri']/pond_volume, label ='$\phi_{diguri}$')
ax2.plot(t, -fp.f['f_sol']/pond_volume, label ='$\phi_{sol}$')
ax2.set_ylabel(r'rate $[g m^{-3} d^{-1}]$')
ax2.set_xlabel('time [day]')
ax2.legend()

# plt.figure(5)
# modified_f_fed_phy = np.where((t < 24) | (t > 80), 0, fp.f['f_fed_phy'])
# plt.plot(t, fp.f['f_fed'], label='$\phi_{fed}$')
# plt.plot(t, modified_f_fed_phy, label='$\phi_{fed,phy}$')
# plt.xlabel(r'time [d]')
# plt.ylabel(r'rate $[g m^{-3} d^{-1}]$')
# # plt.title(r'Accumulative Phytoplankton Growth and Nutrient Availability')
# plt.legend()

plt.figure(7)
plt.plot(t, f4, label='Nitrate deposition (f4)')
plt.plot(t, -f6, label = 'N uptake by phytoplankton (f6)')
plt.ylabel(r'Flow rate $[g m^{-3} d^{-1}]$')
plt.title(r'Flows of Nutrient in the Pond')
plt.legend()

plt.figure(8)
plt.plot(t, f5, label= 'Dihydrogen phosphate deposition (f5)')
plt.plot(t, -f7, label = 'P uptake by phytoplankton (f7)')
plt.ylabel(r'Flow rate $[g m^{-3} d^{-1}]$')
plt.title(r'Flows of Nutrient in the Pond')
plt.legend()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
# Plot the first subplot
ax1.plot(t, Mphy, label='$M_{phy}$')
ax1.set_ylabel('concentration [$g m^{-3}$]')
ax1.legend()

# Plot the second subplot
ax2.plot(t, f1, label='phytoplankton growth ($\phi_{f1}$)')
ax2.plot(t, -f2, label= 'natural mortality rate ($\phi_{f2}$)')
ax2.plot(t, -f3, label = 'death by intraspecific competition rate ($\phi_{f3}$)')
ax2.set_ylabel(r'Flow rate $[g m^{-3} d^{-1}]$')
ax2.set_xlabel('time [day]')
ax2.legend()

# plt.plot(t, f4, label='Nitrate deposition (f4)')
# # plt.plot(t, f5, label= 'Dihydrogen phosphate deposition (f5)')
# plt.plot(t, -f6, label = 'N uptake by phytoplankton (f6)')
# plt.plot(t, -f7, label = 'P uptake by phytoplankton (f7)')
# plt.ylabel(r'Flow rate $[g m^{-3} d^{-1}]$')
# plt.title(r'Flows of Nutrient in the Pond')
# plt.legend()

# plt.figure(8)
# plt.plot(t[25:80], fp.f['f_P_sol'][25:80], label='P content in fish urine')
# plt.plot(t[25:80], fp.f['f_N_sol'][25:80], label='N content in fish urine')
# plt.plot(t[25:80], fp.f['f_N_prt'][25:80], label='N content in faeces')
# plt.plot(t[25:80], fp.f['f_P_prt'][25:80], label='P content in faeces')
# plt.xlabel(r'time [d]')
# plt.ylabel(r'Flow rate $[g d^{-1}]$')
# # plt.title(r'Flow from Nile Tilapia Growth')
# plt.legend()

# plt.figure(9)
# plt.plot(t, Mrt, label='roots')
# plt.plot(t, Mst, label='Stems')
# plt.plot(t, Mlv, label='Leaves')
# plt.plot(t, Mpa, label='Panicles')
# plt.xlabel(r'time [d]')
# plt.ylabel(r'Dry mass [kg DM ha-1 d-1]')
# plt.legend()

plt.figure(10)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
# Plot the first subplot
ax1.plot(t, rice.f['f_cgr'], label='$\phi_{CGR}$')
ax1.set_ylabel(r'flow rate $[kg C ha^-{1} d^{-1}]$')
ax1.legend()

# Plot the second subplot
ax2.plot(t, f_Ph, label='$\phi_{pgr}$')
ax2.plot(t, -f_res, label='$\phi_{res}$')
ax2.plot(t, -f_gr, label='$\phi_{gr}$')
# ax2.plot(t, f_dmv, label='$\phi_{dmv}$')
ax2.set_ylabel(r'flow rate $[kg CH2O ha^-{1} d^{-1}]$')
ax2.set_xlabel('time [day]')
ax2.legend()

plt.figure(12)
plt.plot(t, f_Nlv, label='N flow rate in leaves')
plt.plot(t, -f_pN, label = 'N flow rate in rice plants')
# plt.plot(t, f_Nfert, label = 'N uptake rate from inorganic fertilizer')
plt.plot(t, -f_uptN, label = 'N uptake rate from pond')
plt.plot(t, -IgN[:,1], label = 'N uptake rate from indigenous soil')
plt.axhline(y=-8, linestyle = '--', label = 'maximum N uptake of rice')
plt.xlabel(r'time [d]')
plt.ylabel(r'flow rate $[kg N ha^{-1} d^{-1}]$')
plt.title(r'Nitrogen dynamics')
plt.legend()

# plt.figure(12)
# plt.plot(t, yrice['DVS'])
# y_min, y_max = plt.ylim()
# plt.axvline(x=0, color='b', linestyle='--', linewidth=1)
# plt.annotate('DVS=0', xy=(0, y_min), xytext=(0, y_min),
#               fontsize=12, ha='center')
# plt.axvline(x=37.24, color='b', linestyle='--', linewidth=1)
# plt.annotate('DVS=0.4', xy=(37.24, y_min), xytext=(37.24, y_min),
#               fontsize=12, ha='center')
# plt.axvline(x=45.2, color='b', linestyle='--', linewidth=1)
# plt.annotate('DVS=0.65', xy=(45.2, 0), xytext=(45.2, 0.1),
#               fontsize=12, ha='center')
# plt.axvline(x=63.5, color='b', linestyle='--', linewidth=1)
# plt.annotate('DVS=1', xy=(63.5, 0), xytext=(63.5, y_min),
#               fontsize=12, ha='center')
# plt.axvline(x=116.7, color='b', linestyle='--', linewidth=1)
# plt.annotate('DVS=2', xy=(116.7, 0), xytext=(116.7, y_min),
#               fontsize=12, ha='center')
# plt.xlabel(r'time [d]')
# plt.ylabel(r'DVS [-]')
# plt.text(0, 2, 'x=Day 0', ha='center')
# plt.text(37.24, 2, 'x ~ Day 37', ha='center')
# plt.text(45.2, 1.4, 'x ~ Day 45', ha='center')
# plt.text(63.5, 2, 'x ~ Day 63', ha='center')
# plt.text(116.7, 2, 'x ~ Day 116', ha='center')
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)

# plt.figure(13)
# plt.plot(t, Mgr_tonha)
# plt.plot(t[24:80], Mfis_tonha)
# plt.ylabel(r'Yield $[ton ha^{-1} d^{-1}]$')
# plt.xlabel(r'time [d]')
# plt.legend()
# plt.show()