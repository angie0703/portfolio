# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 09:08:22 2024

@author: alegn
"""

from models.fish import Fish
from models.nutrient import PNut
from models.rice import Rice
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')

# Simulation time array
t = 120 # [d]
tsim = np.linspace(0.0, t, t+1) # [d]
t_weather = np.linspace(0.0, t, t+1) 
dt = 1 # [d] # 5 minutes according to Huesemann?
A_sys = 10000 # [m2] the total area of the system
A_rice = 6000  # [m2] rice field area
A_pond = A_sys - A_rice  # [m2] pond area
V_pond = A_pond*0.6 #[m3] Volume of the pond 0.6 m is the pond depth
V_phy = V_pond + A_rice*0.1 #[m3]

#%%# mass and number of fish
m_fish = 30 #[g] average weight of one fish fry
n_fish = 10000 #[no of fish] according to Pratiwi 2019, 10000 fish is just for comparing result with literature

#%% Disturbances
data_weather = 'C:/Users/alegn/Documents/WUR/Thesis/rfmodel/data/Weather/Magelang_200110_200209_Daily.csv'
weather = pd.read_csv(data_weather, header=0, sep=';')

#FIRST CYCLE
t_ini = '20011001'
t_end = '20020129'

weather['Time'] = pd.to_datetime(weather['Time'], format='%Y%m%d')  # Adjust the format if necessary
weather.set_index('Time', inplace=True)
#first cycle
Tavg = weather.loc[t_ini:t_end,'Tavg'].values #[°C] Mean daily temperature
Rain = weather.loc[t_ini:t_end,'Rain'].values #[mm] Daily precipitation
Igl = weather.loc[t_ini:t_end, 'I0'].values #[MJ m-2] Sum of shortwave radiation daily

I0 = 0.45*Igl*1E6 #Convert [MJ m-2 d-1] to [J m-2 d-1] PAR

#%% fish model
#state variables
x0fish = {'Mfish': m_fish*n_fish, #[gDM] 
      'Mdig':1E-6*n_fish, #[gDM]
      'Muri':1E-6*n_fish #[gDM]
      } 

#parameters
pfish = {
    "tau_dig": 4.5*24,  # [d] time constants for digestive *24 for convert into day
    "tau_uri": 20*24,  # [d] time constants for urinary *24  convert into day
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
    'V_pond': V_pond
}

#initialize object
fish0 = Fish(tsim, dt, {key: val*0 for key, val in x0fish.items()}, pfish)
fish = Fish(tsim, dt, x0fish, pfish)
flow_fish = fish.f

#disturbance
dfish = {
    "DO": np.array([tsim, np.random.randint(1, 6, size=tsim.size)]).T,
    "T": np.array([t_weather, Tavg]).T,
    "Mphy": np.array([tsim, np.full((tsim.size,), 4.293e-4)]).T,
}
#%% nutrient 
x0_pn = {'M_N_net': 0.053*V_phy, #[g] Net phosphorus in the system (Mei et al. 2023)
         'M_P_net': 0.033*V_phy, #[g] Net phosphorus in the system (Mei et al. 2023)
         'Mphy': 4.29e-4*V_phy, #[g] phytoplankton mass in the system (Mei et al. 2023; Jamu & Prihardita, 2002)
         'Nrice' : 0
         } #all multiplied with V_phy because all of the originally in [g/m3]
p_pn = {
      'mu_Up': 0.005, #[d-1] maximum nutrient uptake coefficient
      'mu_phy':  1.27, #[d-1] maximum growth rate of phytoplankton
      'l_sl': 3.5e-8, #[m2 g-1] phytoplankton biomass-specific light attenuation
      'l_bg': 0.77, #[m-1] light attenuation by non-phytoplankton components
      'Kpp': 2232, # [J m-2 d-1] half-saturation constant of phytoplankton production
      'c_prd': 0.15, #[d-1] phytoplankton mortality rate
      'c_cmp': 0.004, #[m2 (g d)-1] phytoplankton crowding loss constant
      'c1': 1.57, # [-] temperature coefficients
      'c2': 0.24, # [-] temperature coefficients
      'Topt': 28, #optimum temperature for phytoplankton growth
      'K_N_phy': 0.1, #[g m-3] half saturation constant for N uptake (Prats & Llavador, 1994)
      'K_P_phy': 0.02, #[g m-3] half saturation constant for P uptake (Prats & Llavador, 1994)  
      'kNdecr': 0.05, #[d-1] decomposition rate (to replace bacteria decomposition rate) (Prats & Llavador, 1994)
      'kPdecr': 0.4, #[d-1] decomposition rate (to replace bacteria decomposition rate) (Prats & Llavador, 1994)    
      'f_N_edg': 0.8*0.6*1000, # [g d-1] N from endogenous soil [originally kg N ha-1 d-1]
      'f_P_edg': 0.5*0.6*1000, #[g d-1] P from endogenous soil [originally kg N ha-1 d-1]
      'MuptN': 8*0.6*1000, #[g d-1] maximum daily N uptake by rice plants, originally in [kg ha-1 d-1]
      'MuptP': 5*0.6*1000,#[g d-1] maximum daily P uptake by rice plants, originally in [kg ha-1 d-1]
      'kNphy': 0.06, #[g N/g biomass] fraction of N from phytoplankton biomass
      'kPphy': 0.01, #[g P/g biomass] fraction of P from phytoplankton biomass
      'V_phy': V_phy,

     }
d_pn = {
     'I0' :  np.array([tsim, np.full((tsim.size,), I0)]).T, #[J m-2 h-1] Hourly solar irradiation (6 AM to 6 PM)
     'Tw':np.array([tsim, Tavg]).T,
     'f_N_sol': np.array([tsim, np.full((tsim.size,), 0)]).T,
     'f_P_sol': np.array([tsim, np.full((tsim.size,), 0)]).T,
     'f_N_prt': np.array([tsim, np.full((tsim.size,), 0)]).T,
     'f_P_prt': np.array([tsim, np.full((tsim.size,), 0)]).T,
     'Rain': np.array([tsim, Rain]).T,
     # 'DVS': np.array([tsim, np.full((tsim.size,), 0)]).T
     }

pnut = PNut(tsim, dt, x0_pn, p_pn)
flow_pnut = pnut.f

#%% rice
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
         "cr_rt": 1.326
     }

drice = {'I0':np.array([tsim, I0]).T,
     # 'T':np.array([tsim, np.random.uniform(low=20, high=35, size=tsim.size,)]).T,
     'T': np.array([tsim, Tavg]).T,
     'CO2': np.array([tsim, np.full((tsim.size,), 400)]).T,
     'f_N_plt_upt': np.array([tsim, np.full((tsim.size,), 0)]).T,
     # 'f_P_plt_upt': np.array([tsim, np.full((tsim.size,), 0)]).T
     }

#instantiate object
rice = Rice(tsim, dt, x0rice, price)
flow_rice = rice.f

#%% Input (u)
#inorganic fertilizer types and concentration used in Indonesia: 
#Nitrogen source: NPK (15%:15%:15%), Urea (46% N)
#NPK fertilizer recommended doses 167 kg/ha (Yassi 2023)
NPK_w = 167*A_rice #[kg]
I_N1 = (15/100)*NPK_w

#Urea fertilizer recommended doses 100 kg/ha (Yassi 2023)
Urea = 100*A_rice #[kg]
I_N2 = (46/100)*Urea
# I_N2 = 0
#Total N from inorganic fertilizer
# I_N = I_N1 + I_N2
I_N = 0
#Phosphorus source: SP36 (36%P2O5 ~ 7.85% P)
#P content in SP-36: 7.85% P
#SP fertilizer recommended doses 31 kg/ha (Yassi 2023)
SP36 = 31*A_rice
# I_P = (7.85/100)*SP36
I_P = 0
Norgf = 12.3*1000 #[g N]
Porgf = 13.9*1000 #[g P]

u_pn = {'I_N': I_N, 'I_P': I_P, 'Norgf': Norgf, 'Porgf': Porgf}

#%%run model
yfish_ = {'t': [0, ], 'Mfish': [0, ], 'Mdig':[0, ], 'Muri':[0, ], 'f_N_prt': [0, ], 'f_P_prt': [0, ], 'f_P_sol': [0, ], 'f_N_sol':[0, ]}
yrice_ = {'t': [0, ], 'Mrt': [0, ], 'Mst': [0, ], 'Mlv': [0, ], 'Mpa': [0, ], 'Mgr': [0, ], 'HU': [0, ], 'DVS': [0, ]}
ypn_ = {'t': [0, ], 'M_N_net': [0.053*V_phy, ], 'M_P_net': [0.033*V_phy, ],'Mphy': [4.29e-4, ], 'Nrice': [0, ] }

dfish_= {'Mphy': [4.29e-4, ]}
d_pn_ = {'f_N_sol': [ 0, ], 'f_N_prt': [ 0, ], 'f_P_sol': [ 0, ], 'f_P_prt': [ 0, ]}
drice_ = {'f_N_plt_upt': [0, ]}

it = np.nditer(tsim[:-1], flags=['f_index'])
for ti in it:
    # Index for current time instant
    idx = it.index
    # Integration span
    tspan = (tsim[idx], tsim[idx+1])
    print('Integrating', tspan)
    # Controlled inputs
    ufish = {'Mfed': m_fish*0.03}
    
    #Run pnut model
    ypn =  pnut.run(tspan, d_pn, u_pn)
    ypn_['t'].append(ypn['t'][1])
    ypn_['Mphy'].append(ypn['Mphy'][1])
    ypn_['M_N_net'].append(ypn['M_N_net'][1])
    ypn_['M_P_net'].append(ypn['M_P_net'][1])
    ypn_['Nrice'].append(ypn['Nrice'][1])
    #retrieve simulation result for rice growth
    drice_['f_N_plt_upt'].append(np.array([ypn['t'][1], ypn['Nrice'][1]]))
    drice['f_N_plt_upt'] = np.array([ypn['t'], ypn['Nrice']]).T
    #make simulation result of ypn as input for fish
    dfish_['Mphy'].append(np.array([ypn["t"][1], ypn["Mphy"][1]]))
    dfish['Mphy'] = np.array([ypn['t'], ypn['Mphy']]).T
    #run rice model
    yrice =  rice.run(tspan, drice)
    yrice_['t'].append(yrice['t'][1])
    yrice_['Mgr'].append(yrice['Mgr'][1])
    yrice_['Mrt'].append(yrice['Mrt'][1])
    yrice_['Mst'].append(yrice['Mst'][1])
    yrice_['Mlv'].append(yrice['Mlv'][1])
    yrice_['Mpa'].append(yrice['Mpa'][1])
    yrice_['HU'].append(yrice['HU'][1])
    yrice_['DVS'].append(yrice['DVS'][1])

    if ti < 24:
       #make the fish weight zero to simulate 'no growth'
       yfish0 =  fish0.run((0,23), dfish, ufish)
       yfish_['t'].append(yfish0['t'][1])
       yfish_['Mfish'].append(0)
       yfish_['Mdig'].append(0)
       yfish_['Muri'].append(0)
       yfish_['f_N_prt'].append(0)
       yfish_['f_P_prt'].append(0)
       yfish_['f_N_sol'].append(0)
       yfish_['f_P_sol'].append(0)
       d_pn_['f_N_sol'].append(np.array([yfish0['t'][1], np.zeros_like(yfish0['t'])[1]]))
       d_pn_['f_P_sol'].append(np.array([yfish0['t'][1], np.zeros_like(yfish0['t'])[1]]))
       d_pn_['f_N_prt'].append(np.array([yfish0['t'][1], np.zeros_like(yfish0['t'])[1]]))
       d_pn_['f_P_prt'].append(np.array([yfish0['t'][1], np.zeros_like(yfish0['t'])[1]]))
       d_pn['f_N_sol'] = np.array([yfish0['t'], np.zeros_like(yfish0['t'])]).T
       d_pn['f_N_prt'] = np.array([yfish0['t'], np.zeros_like(yfish0['t'])]).T
       d_pn['f_P_sol'] = np.array([yfish0['t'], np.zeros_like(yfish0['t'])]).T
       d_pn['f_P_prt'] = np.array([yfish0['t'], np.zeros_like(yfish0['t'])]).T
    
    elif ti > 80:
           #make the fish weight zero to simulate 'no growth'
           yfish0 =  fish0.run((80, tsim[-1]), dfish, ufish)
           yfish_['t'].append(yfish0['t'][1])
           yfish_['Mfish'].append(0)
           yfish_['Mdig'].append(0)
           yfish_['Muri'].append(0)
           yfish_['f_N_prt'].append(0)
           yfish_['f_P_prt'].append(0)
           yfish_['f_N_sol'].append(0)
           yfish_['f_P_sol'].append(0)
           d_pn_['f_N_sol'].append(np.array([yfish0['t'][1], np.zeros_like(yfish0['t'][1])]))
           d_pn_['f_P_sol'].append(np.array([yfish0['t'][1], np.zeros_like(yfish0['t'][1])]))
           d_pn_['f_N_prt'].append(np.array([yfish0['t'][1], np.zeros_like(yfish0['t'][1])]))
           d_pn_['f_P_prt'].append(np.array([yfish0['t'][1], np.zeros_like(yfish0['t'][1])]))
           d_pn['f_N_sol'] = np.array([yfish0['t'], np.zeros_like(yfish0['t'])])
           d_pn['f_N_prt'] = np.array([yfish0['t'], np.zeros_like(yfish0['t'])])
           d_pn['f_P_sol'] = np.array([yfish0['t'], np.zeros_like(yfish0['t'])])
           d_pn['f_P_prt'] = np.array([yfish0['t'], np.zeros_like(yfish0['t'])])
           
    else: 
       yfishr = fish.run((24,79), dfish, ufish)
       yfish_['t'].append(yfishr['t'][1])
       yfish_['Mfish'].append(yfishr['Mfish'][1])
       yfish_['Mdig'].append(yfishr['Mdig'][1])
       yfish_['Muri'].append(yfishr['Muri'][1])
       yfish_['f_N_prt'].append(yfishr['f_N_prt'][1])
       yfish_['f_P_prt'].append(yfishr['f_P_prt'][1])
       yfish_['f_N_sol'].append(yfishr['f_N_sol'][1])
       yfish_['f_P_sol'].append(yfishr['f_P_sol'][1])
       
       d_pn_['f_N_sol'].append(np.array([yfishr['t'][1], yfishr['f_N_sol'][1]]))
       d_pn_['f_P_sol'].append(np.array([yfishr['t'][1], yfishr['f_P_sol'][1]]))
       d_pn_['f_N_prt'].append(np.array([yfishr['t'][1], yfishr['f_N_prt'][1]]))
       d_pn_['f_P_prt'].append(np.array([yfishr['t'][1], yfishr['f_P_prt'][1]]))
       d_pn['f_N_sol'] = np.array([yfishr['t'], yfishr['f_N_sol']])
       d_pn['f_N_prt'] = np.array([yfishr['t'], yfishr['f_N_prt']])
       d_pn['f_P_sol'] = np.array([yfishr['t'], yfishr['f_P_sol']])
       d_pn['f_P_prt'] = np.array([yfishr['t'], yfishr['f_P_prt']])
       
yfish = {key: np.array(value) for key, value in yfish_.items()} 
ypn = {key: np.array(value) for key, value in ypn_.items()} 
yrice = {key: np.array(value) for key, value in yrice_.items()}
# d_pn = {key if key not in d_pn_ else key: (np.array(d_pn_[key]) if key in d_pn_ else np.array(value)) for key, value in d_pn.items()} 
# drice = {key if key not in drice_ else key: (np.array(drice_[key]) if key in drice_ else np.array(value)) for key, value in drice.items()} 
# dfish = {key if key not in dfish_ else key: (np.array(dfish_[key]) if key in dfish_ else np.array(value)) for key, value in dfish.items()} 

# drice.update({key: drice_[key] for key in drice_ if key in drice})
# dfish.update({key: dfish_[key] for key in dfish_ if key in dfish})

#retrieve results
t = ypn['t']
Mphy = ypn['Mphy']/V_phy #[g m-3] 
M_N_net = ypn['M_N_net']/1000 #[kg N]
M_P_net = ypn['M_P_net']/1000 #[kg P]

f_N_edg = np.full((tsim.size,), pnut.p['f_N_edg'])
f_P_edg = np.full((tsim.size,), pnut.p['f_P_edg'])
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

Mfish= fish.y['Mfish']
Muri= fish.y['Muri']
Mfis_fr= Mfish/pfish['k_DMR']
Mdig= fish.y['Mdig']
f_fed_phy = fish.f['f_fed_phy']
f_fed = fish.f['f_fed']

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
f_uptN = rice.f['f_uptN']

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
plt.plot(t, f_fed, label='$\phi_{fed}$')
plt.plot(t, f_fed_phy, label = '$\phi_{fed,phy}$')
plt.xlabel(r'time [d]')
plt.ylabel(r'flow rate $[g d^{-1}]$')
plt.title(r'Feeding dynamics')
plt.legend()

plt.figure(3)
plt.plot(t, Mrt, label='roots')
plt.plot(t, Mst, label='Stems')
plt.plot(t, Mlv, label='Leaves')
plt.plot(t, Mpa, label='Panicles')
plt.xlabel(r'time [d]')
plt.ylabel(r'Dry mass [kg DM ha-1 d-1]')
plt.legend()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
# Plot the first subplot
ax1.plot(t, rice.f['f_cgr'], label='$\phi_{CGR}$')
ax1.set_ylabel(r'flow rate $[kg C ha^-{1} d^{-1}]$')
ax1.legend()

# Plot the second subplot
ax2.plot(t, f_Ph, label='$\phi_{pgr}$')
ax2.plot(t, f_res, label='$\phi_{res}$')
ax2.plot(t, f_gr, label='$\phi_{gr}$')
# ax2.plot(t, f_dmv, label='$\phi_{dmv}$')
ax2.set_ylabel(r'flow rate $[kg CH2O ha^-{1} d^{-1}]$')
ax2.set_xlabel('time [day]')
ax2.legend()



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

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
# Plot the first subplot
ax1.plot(t, M_N_net, label = '$M_{N,net}$')
ax1.set_ylabel('mass [$kg$]')
ax1.legend()

# # Plot the second subplot
ax2.plot(t, f_N_edg, label='$\phi_{N,edg}$', color='red')
ax2.plot(t, f_N_fert_ing, label='$\phi_{N,fert,ing}$', color='green')
ax2.plot(t, f_N_fert_org, label='$\phi_{N,fert,org}$', color='orange')
ax2.plot(t, f_N_phy_cmp, label='$\phi_{N,phy,cmp}$', color='purple')
ax2.plot(t, f_N_fis_sol, label='$\phi_{N,fis,sol}$', color='brown')
ax2.plot(t, f_N_fis_prt, label='$\phi_{N,fis,prt}$', color='pink')
ax2.plot(t, -f_N_phy_upt, label = '$\phi_{N,phy,upt}$', color = 'gray')
ax2.plot(t, -f_N_plt_upt, label = '$\phi_{N,plt,upt}$', color = 'cyan')
ax2.set_ylabel(r'rate $[g d^{-1}]$')
ax2.set_xlabel('time [day]')
ax2.legend()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
# Plot the first subplot
ax1.plot(t, M_P_net, label = '$M_{P,net}$')
ax1.set_ylabel('mass [$kg$]')
ax1.legend()

# # Plot the second subplot
ax2.plot(t, f_P_edg, label = '$\phi_{P,edg}$', color = 'red')
ax2.plot(t, f_P_fert_ing, label = '$\phi_{P,fert,ing}$', color = 'green')
ax2.plot(t, f_P_fert_org, label = '$\phi_{P,fert, org}$', color = 'orange')
ax2.plot(t, f_P_phy_cmp, label = '$\phi_{P,phy,cmp}$', color = 'purple')
ax2.plot(t, f_P_fis_sol, label = '$\phi_{P,fis,sol}$', color = 'brown')
ax2.plot(t, f_P_fis_prt, label = '$\phi_{P,fis,prt}$', color = 'pink')
ax2.plot(t, -f_P_phy_upt, label = '$\phi_{P,phy,upt}$', color = 'gray')
ax2.plot(t, -f_P_plt_upt, label = '$\phi_{P,plt,upt}$', color = 'cyan')
ax2.set_ylabel(r'rate $[g d^{-1}]$')
ax2.set_xlabel('time [day]')
ax2.legend()

