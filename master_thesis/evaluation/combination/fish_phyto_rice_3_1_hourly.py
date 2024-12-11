# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 09:08:22 2024

@author: alegn
"""

from models.fish import Fish
from models.nutrient import PNut
from models.rice import Rice_hourly
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')

# Simulation time array
t = 120 # [d]
tsim = np.linspace(0.0, t*24, t*24+1) # [d]
dt = 1 # [h] hourly
A_sys = 10000 # [m2] the total area of the system
A_rice = 6000  # [m2] rice field area
Arice_ha = A_rice/A_sys #[ha] rice field area in ha
A_pond = A_sys - A_rice  # [m2] pond area
V_pond = A_pond*0.6 #[m3] Volume of the pond 0.6 m is the pond depth
V_phy = V_pond + A_rice*0.1 #[m3]

#%%# Mass and number of fish
m_fish = 30 #[g] average weight of one fish fry
n_fish = 10000 #[no of fish] according to Pratiwi 2019, 10000 fish is just for comparing result with literature
n_rice = 127980 #number of plants in 0.6 ha land

#%% Disturbances
data_weather = 'C:/Users/alegn/Documents/WUR/Thesis/rfmodel/data/Weather/DIY_202210_202301_Hourly.csv'
weather = pd.read_csv(data_weather, header=0, sep=';')

#FIRST CYCLE
t_ini = '2022-10-01T08:00'
t_end = '2023-01-29T08:00'

weather['Time'] = pd.to_datetime(weather['Time'])
weather.set_index('Time', inplace=True)
#first cycle
T = weather.loc[t_ini:t_end,'Temp'].values #[°C] Mean daily temperature
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
    'V_pond': V_pond
}

#initialize object
fish0 = Fish(tsim, dt, {key: val*0 for key, val in x0fish.items()}, pfish)
tsim_fish = np.linspace(24*24, 79*24+1, 55*24+2)
fish = Fish(tsim_fish, dt, x0fish, pfish)
flow = fish.f

#disturbance
dfish = {
    "DO": np.array([tsim, np.random.randint(1, 6, size=tsim.size)]).T,
    "T": np.array([tsim, T]).T,
    "Mphy": np.array([tsim, np.full((tsim.size,), 4.293e-4)]).T,
}
#%% nutrient 
x0_pn = {'M_N_net': 0.053*V_phy, #[g] Net phosphorus in the system (Mei et al. 2023)
         'M_P_net': 0.033*V_phy, #[g] Net phosphorus in the system (Mei et al. 2023)
         'Mphy': 4.29e-4*V_phy, #[g] phytoplankton Mass in the system (Mei et al. 2023; Jamu & Prihardita, 2002)
         'Nrice' : 0, # equals to f_N_plt_upt
         } #all multiplied with V_phy because all of the values are originally in [g/m3]
p_pn = {
      'mu_Up': 0.005/24, #[h-1] maximum nutrient uptake coefficient
      'mu_phy':  1.27/24, #[h-1] maximum growth rate of phytoplankton
      'l_sl': 3.5e-7, #[m2 g-1] phytoplankton bioMass-specific light attenuation
      'l_bg': 0.77, #[m-1] light attenuation by non-phytoplankton components
      'Kpp': 2232/12, # [J m-2 h-1] half-saturation constant of phytoplankton production (for 12 hours of daylight per day)
      'c_prd': 0.15/24, #[h-1] phytoplankton mortality rate
      'c_cmp': 0.004/24, #[m3 (g d)-1] phytoplankton crowding loss constant
      'c1': 1.57, # [-] temperature coefficients
      'c2': 0.24, # [-] temperature coefficients
      'Topt': 28, #optimum temperature for phytoplankton growth
      'K_N_phy': 0.1, #[g m-3] half saturation constant for N uptake (Prats & Llavador, 1994)
      'K_P_phy': 0.02, #[g m-3] half saturation constant for P uptake (Prats & Llavador, 1994)  
      'kNdecr': 0.05/24, #[h-1] decomposition rate (to replace bacteria decomposition rate) (Prats & Llavador, 1994)
      'kPdecr': 0.4/24, #[h-1] decomposition rate (to replace bacteria decomposition rate) (Prats & Llavador, 1994)    
      'f_N_edg': 0.8*Arice_ha*1000/24, # [g h-1] endogenous N supply from soil [originally kg N ha-1 d-1]
      'f_P_edg': 0.5*Arice_ha*1000/24, #[g h-1] endogenous P supply from soil [originally kg N ha-1 d-1]
      'MuptN': 8*Arice_ha*1000/24, #[g h-1] maximum daily N uptake by rice plants, originally in [kg ha-1 d-1], only 0.6 ha of land that is planted with rice
      'MuptP': 8*Arice_ha*1000/24,#[g h-1] maximum daily P uptake by rice plants, originally in [kg ha-1 d-1], only 0.6 ha of land that is planted with rice
      'kNphy': 0.06, #[g N/g bioMass] fraction of N from phytoplankton biomass
      'kPphy': 0.01, #[g P/g biomass] fraction of P from phytoplankton biomass
      'V_phy': V_phy,

     }
d_pn = {
     'I0' :  np.array([tsim, np.full((tsim.size,), I0)]).T, #[J m-2 h-1] Hourly solar irradiation (6 AM to 6 PM)
     'Tw':np.array([tsim, T]).T,
     'f_N_sol': np.array([tsim, np.full((tsim.size,), 0)]).T,
     'f_P_sol': np.array([tsim, np.full((tsim.size,), 0)]).T,
     'f_N_prt': np.array([tsim, np.full((tsim.size,), 0)]).T,
     'f_P_prt': np.array([tsim, np.full((tsim.size,), 0)]).T,
     'Rain': np.array([tsim, Rain]).T,
     'DVS': np.array([tsim, np.full((tsim.size,), 0)]).T
     }

pnut = PNut(tsim, dt, x0_pn, p_pn)
flow_pnut = pnut.f

#%% rice
x0rice = {
    "Mrt": 0.001,  # [kg DM ] Dry biomass of root
    "Mst": 0.002,  # [kg DM] Dry biomass of stems
    "Mlv": 0.002,  # [kg DM] Dry biomass of leaves
    "Mpa": 0.0,  # [kg DM] Dry biomass of panicles
    "Mgr": 0.0,  # [kg DM] Dry biomass of grains
    "HU": 0.0,
    "DVS": 0,
}

price = {
      #Manually calibrated developmental rate
          'DVRi': 0, #[°C h-1]
          'DVRJ': 0.00137, # hourly development rate during juvenile phase [(°C d)-1] 0 < DVS < 0.4
          'DVRI': 0.00084,# hourly development rate during photoperiod-sensitive phase [(°C d)-1], 0.4 <= DVS < 0.65
          'DVRP': 0.00117,# hourly development rate during panicle development phase [(°C d)-1], 0.65 <= DVS < 1    
          'DVRR': 0.00335, # hourly development rate during reproduction (generative) phase [(°C d)-1], originally in [°C d-1]
         "Tmax": 40,#[°C]
         "Tmin": 15,#[°C]
         "Topt": 33,#[°C]
         "Rm_rt": 0.01,
         "Rm_st": 0.015,
         "Rm_lv": 0.02,
         "Rm_pa": 0.003,
         "k_pa_maxN": 0.0175, #[kg N/kg panicles]
         "cr_lv": 1.326,
         "cr_st": 1.326,
         "cr_pa": 1.462,
         "cr_rt": 1.326,
         'n_rice': n_rice
     }

drice = {'I0':np.array([tsim, I0]).T,
     'Th': np.array([tsim, T]).T,
     'CO2': np.array([tsim, np.full((tsim.size,), 400)]).T,
     'Nrice': np.array([tsim, np.full((tsim.size,), 0)]).T,
     }

#instantiate object
rice = Rice_hourly(tsim, dt, x0rice, price)
#%% Input (u)
#inorganic fertilizer types and concentration used in Indonesia: 
#Nitrogen source: NPK (15%:15%:15%), Urea (46% N)
#NPK fertilizer recommended doses 167 kg/ha (Yassi 2023)
# NPK_w = 167*Arice_ha #[kg]
# NPK = (15/100)*NPK_w
NPK = 0
#Urea fertilizer recommended doses 100 kg/ha (Yassi 2023)
#Urea fertilizer recommended doses 200 kg/ha (Bedriyetti, 2000)
Urea_w = 200*Arice_ha #[kg]
Urea = (46/100)*Urea_w

#Total N from inorganic fertilizer
# N_ingf = (NPK + Urea)*1000 # [g N]
N_ingf = (Urea)*1000 # [g N]

N_ingf_1 = N_ingf/2
N_ingf_2 = N_ingf - N_ingf_1

#Phosphorus source: SP36 (36%P2O5 ~ 7.85% P)
#P content in SP-36: 7.85% P
#P fertilizer recommended doses 31 kg/ha (Yassi 2023)
#P fertilizer recommended doses 75 kg/ha (Bedriyetti 2000)
SP36 = 31*Arice_ha
I_P = (7.85/100)*SP36*1000 #[g P]

#Organic fertilizer (Kang'ombe 2006), to replace the organic fertilizer suggested by Yassi 2023
Norgf = (1.23/100)*500*1000 #[g N]
Porgf = (1.39/100)*500*1000 #[g P]

#experiment 1: no fertilizers
u_pn = {'N_ingf_1': 0, 'N_ingf_2': 0, 'N_ingf_3':0, 'P_ingf': 0, 'Norgf': 0, 'Porgf': 0}
#experiment 2: using only organic fertilizers
# u_pn = {'N_ingf_1': 0, 'N_ingf_2': 0, 'N_ingf_3':0, 'P_ingf': 0, 'Norgf': Norgf, 'Porgf': Porgf}
#experiment 3: using only inorganic fertilizers
# u_pn = {'N_ingf_1': N_ingf_1, 'N_ingf_2': N_ingf_2, 'N_ingf_3':0, 'P_ingf': I_P, 'Norgf': 0, 'Porgf': 0}
#experiment 4: using only inorganic and organic fertilizers
# u_pn = {'N_ingf_1': N_ingf_1, 'N_ingf_2': N_ingf_2, 'N_ingf_3':0, 'P_ingf': I_P, 'Norgf': Norgf, 'Porgf': Porgf}
#%%run model

it = np.nditer(tsim[:-1], flags=['f_index'])
for ti in it:
    # Index for current time instant
    idx = it.index
    # Integration span
    tspan = (tsim[idx], tsim[idx+1])
    print('Integrating', tspan)
    
    #Run pnut model
    ypn =  pnut.run(tspan, d_pn, u_pn)
    # retrieve simulation result for rice growth
    drice['Nrice'] = np.array([ypn['t'], ypn['Nrice']])
    #make simulation result of ypn as input for fish
    dfish['Mphy'] = np.array([ypn['t'], ypn['Mphy']])
    #run rice model
    yrice =  rice.run(tspan, drice)
    
    
    if ti>=21*24:
        #retrieve rice simulation result for pnutrient
        d_pn['DVS'] = np.array([yrice['t'], yrice['DVS']])

    # if ti <= 24*24 or ti > 79*24:
    #     d_pn['f_N_sol'] = np.array([tsim.T, np.zeros_like(tsim)])
    #     d_pn['f_N_prt'] = np.array([tsim.T, np.zeros_like(tsim)])
    #     d_pn['f_P_sol'] = np.array([tsim.T, np.zeros_like(tsim)])
    #     d_pn['f_P_prt'] = np.array([tsim.T, np.zeros_like(tsim)])
       
    # else: 
    #    yfish = fish.run(tspan, dfish)
    #    d_pn['f_N_sol'] = np.array([yfish['t'], yfish['f_N_sol']])
    #    d_pn['f_N_prt'] = np.array([yfish['t'], yfish['f_N_prt']])
    #    d_pn['f_P_sol'] = np.array([yfish['t'], yfish['f_P_sol']])
    #    d_pn['f_P_prt'] = np.array([yfish['t'], yfish['f_P_prt']])
       
# yfish_all = fish.y 
ypn_all = pnut.y 
yrice_all = rice.y

#%%retrieve results
t = pnut.t

days = np.array([30, 60, 90, 120])
Mphy = ypn_all['Mphy'] #[g m-3] 
M_N_net = ypn_all['M_N_net']/1000 #[kg N]
M_P_net = ypn_all['M_P_net']/1000 #[kg P]

f_N_edg = np.full((tsim.size,), pnut.p['f_N_edg']/1000)
f_P_edg = np.full((tsim.size,), pnut.p['f_P_edg']/1000)
MuptN = np.full((tsim.size,), pnut.p['MuptN']/1000)
MuptP = np.full((tsim.size,), pnut.p['MuptP']/1000)
f_N_fert_ing = pnut.f['f_N_fert_ing']/1000
f_P_fert_ing = pnut.f['f_P_fert_ing']/1000
f_N_fert_org = pnut.f['f_N_fert_org']/1000
f_P_fert_org = pnut.f['f_P_fert_org']/1000
f_phy_grw = pnut.f['f_phy_grw']
f_phy_prd = pnut.f['f_phy_prd']
f_phy_cmp = pnut.f['f_phy_cmp']
f_N_fis_sol = pnut.f['f_N_fis_sol']/1000
f_P_fis_sol = pnut.f['f_P_fis_sol']/1000
f_N_fis_prt = pnut.f['f_N_fis_prt']/1000
f_P_fis_prt = pnut.f['f_P_fis_prt']/1000
f_N_phy_cmp = pnut.f['f_N_phy_cmp']/1000
f_P_phy_cmp = pnut.f['f_P_phy_cmp']/1000
f_N_phy_upt = pnut.f['f_N_phy_upt']/1000
f_P_phy_upt = pnut.f['f_P_phy_upt']/1000
f_N_plt_upt = pnut.f['f_N_plt_upt']/1000
f_P_plt_upt = pnut.f['f_P_plt_upt']/1000

# Mfish= fish.y['Mfish']/1000 #[kg]
# Muri= fish.y['Muri']/1000 #[kg]
# Mfis_fr= fish.y['Mfish']/pfish['k_DMR'] #[g]
# Mdig= fish.y['Mdig']/1000 #[kg]

Mrt = yrice_all['Mrt']
Mst = yrice_all['Mst']
Mlv = yrice_all['Mlv']
Mpa = yrice_all['Mpa']
Mgr = yrice_all['Mgr']
DVS = yrice_all['DVS']
drice_all = rice.d
f_Ph= rice.f['f_Ph'] #[kg CO2 h-1]
f_res= rice.f['f_res'] #[kg CH20 h-1]
f_dmv = rice.f['f_dmv'] #[kg DM leaf h-1]
f_Nlv = rice.f['f_Nlv'] #[kg N h-1]
f_uptN = rice.f['f_uptN'] #[kg N h-1]
#yield
# Mff= Mfis_fr/400000 #[ton/ha]
Mrice = Mgr/600 #ton/ha


#%% Plot results

x_ticks = np.linspace (1, 2880, 120)
days = np.array([21, 30, 60, 90, 120])
days_fish = np.array([24, 79])

#for emphasize time of fish enter the system and harvested, and monthly 
days_all = np.array([21, 24, 30, 60, 79, 90, 120]) #for experiment without apply any fertilizer
days_exp2 = np.array([0, 7, 21, 24, 30, 60, 79, 90, 120]) #for experiment that apply only organic fertilizer 
days_expN = np.array([0, 7, 21, 24, 31, 61, 79, 90, 120]) #for experiment that apply inorganic & organic fertilizer like Indonesian farmers
days_expP = np.array([0, 7, 21, 24, 30, 41, 60, 79, 90, 120]) #for experiment that apply inorganic & organic fertilizer like Indonesian farmers

plt.rcParams.update({
    'figure.figsize': [10,10],
    'font.size': 14,        # Global font size
    'axes.titlesize': 16,   # Title font size
    'axes.labelsize': 16,   # X and Y axis labels font size
    'xtick.labelsize': 14,  # X tick labels font size
    'ytick.labelsize': 14,  # Y tick labels font size
    'legend.fontsize': 16,  # Legend font size
    'figure.titlesize': 18  # Figure title font size
}) 
# plt.figure(1)
# plt.plot(t, Mrt, label='$M_{rt}$')
# plt.plot(t, Mst, label='$M_{st}$')
# plt.plot(t, Mlv, label='$M_{lv}$')
# plt.plot(t, Mpa, label='$M_{pa}$')
# plt.plot(t, Mgr, label ='$M_{gr}$')
# plt.xlabel(r'time [d]')
# plt.xticks(days*24, labels=[f"{day}" for day in days])
# plt.ylabel(r'Mass $[kg]$')
# plt.legend()

# plt.figure(2)
# plt.plot(t, DVS)
# plt.legend()
# plt.xlabel(r'time [d]')
# plt.xticks(days*24, labels=[f"{day}" for day in days])
# # plt.ylabel(r'DVS [-]')
# plt.legend()

# plt.figure(3)
# plt.plot(tsim_fish, Mfis_fr/n_fish, label = '$M_{ff}$')
# plt.xlabel(r'time [d]')
# plt.xticks(x_ticks, labels=np.arange(1,120+1))
# # plt.xticklabels(np.arange(1, 120+1))
# plt.ylabel(r'Fresh mass [$g$]')
# plt.legend()

# # #Fish
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
# # Plot the first subplot
# ax1.plot(tsim_fish, Mff, color = 'b', label = '$M_{ff}$')
# ax1.set_ylabel('Yield [$ton ha^{-1}$]')
# ax1.legend()
# # Plot the second subplot
# ax2.plot(tsim_fish, fish.f['f_upt']/1000000*0.4, label ='$\phi_{upt}$')
# ax2.set_ylabel(r'rate $[ton ha^{-1} h^{-1}]$')
# ax2.set_xticks(x_ticks)
# ax2.set_xticklabels(np.arange(1, 120+1))
# ax2.set_xlabel('time [d]')
# ax2.legend()

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
# # Plot the first subplot
# ax1.plot(tsim_fish, Mdig, label = '$M_{dig}$')
# ax1.set_ylabel('Mass [$kg$]')
# ax1.legend()

# # Plot the second subplot
# ax2.plot(tsim_fish, fish.f['f_fed']/1000, label ='$\phi_{fed}$')
# ax2.plot(tsim_fish, fish.f['f_fed_phy']/1000, label ='$\phi_{fed,phy}$')
# ax2.plot(tsim_fish, -fish.f['f_digout']/1000, label ='$\phi_{digout}$')
# ax2.set_ylabel(r'Mass flow rate $[kg h^{-1}]$')
# ax2.set_xticks(x_ticks)
# ax2.set_xticklabels(np.arange(1, 120+1))
# ax2.set_xlabel('time [d]')
# ax2.legend()

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
# # Plot the first subplot
# ax1.plot(tsim_fish, Muri, label = '$M_{uri}$')
# ax1.set_ylabel('Mass [$kg$]')
# ax1.legend()

# # Plot the second subplot
# ax2.plot(tsim_fish, fish.f['f_diguri']/1000, label ='$\phi_{diguri}$')
# ax2.plot(tsim_fish, -fish.f['f_sol']/1000, label ='$\phi_{sol}$')
# ax2.set_ylabel(r'Mass flow rates $[g h^{-1}]$')
# ax2.set_xticks(x_ticks)
# ax2.set_xticklabels(np.arange(1, 120+1))
# ax2.set_xlabel('time [d]')
# ax2.legend()


# # #Phytoplankton
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
# # Plot the first subplot
# ax1.plot(t, Mphy, label = '$M_{phy}$')
# ax1.set_ylabel('Mass [$g$]')
# ax1.legend()
# # ax1.set_title('Phytoplankton')

# # Plot the second subplot
# ax2.plot(t, f_phy_grw, label ='$\phi_{phy,grw}$')
# ax2.plot(t, -f_phy_prd, label ='$\phi_{phy,prd}$')
# ax2.plot(t, -f_phy_cmp, label ='$\phi_{phy,cmp}$')
# ax2.set_ylabel(r'Mass flow rate $[g h^{-1}]$')
# ax2.set_xticks(days*24, labels=[f"{day}" for day in days])
# ax2.set_xlabel('time [d]')
# ax2.legend()

#Net nitrogen
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
# Plot the first subplot
ax1.plot(t, M_N_net, label = '$M_{N,net}$')
ax1.set_ylabel('Mass [$kg$]')
ax1.legend()
# ax1.set_title('Net nitrogen accumulated')

# # Plot the second subplot
ax2.plot(t, f_N_edg, label = '$\phi_{N,edg}$', color ='blue')
ax2.plot(t, -MuptN, label = '$N_{max}$', color ='cyan')
ax2.plot(t, f_N_fert_ing, label = '$\phi_{N,fert,ing}$', color = 'orange')
ax2.plot(t, f_N_fert_org, label = '$\phi_{N,fert,org}$', color='green')
ax2.plot(t, f_N_phy_cmp, label = '$\phi_{N,phy,cmp}$', color='red')
ax2.plot(t, f_N_fis_sol, label = '$\phi_{N,fis,sol}$', color='purple')
ax2.plot(t, f_N_fis_prt, label = '$\phi_{N,fis,prt}$', color = 'brown')
ax2.plot(t, -f_N_phy_upt, label = '$\phi_{N,phy,upt}$', color='pink')
ax2.plot(t, -f_N_plt_upt, label = '$\phi_{N,plt,upt}$', color = 'gray')
ax2.set_ylabel(r'Mass flow rate $[kg h^{-1}]$')
ax2.set_xlabel('time [d]')
# ax2.set_xticks(days_all*24, labels=[f"{day}" for day in days_all])
# ax2.set_xticks(days_exp2*24, labels=[f"{day}" for day in days_exp2])
ax2.set_xticks(days_expN*24, labels=[f"{day}" for day in days_expN], rotation = 45)
ax2.legend(loc='upper right')

#Net phosphorus
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
# Plot the first subplot
ax1.plot(t, M_P_net, label = '$M_{P,net}$')
ax1.set_ylabel('Mass [$kg$]')
ax1.legend()
# ax1.set_title('Net phosphorus accumulated')
# # Plot the second subplot
ax2.plot(t, f_P_edg, label = '$\phi_{P,edg}$', color = 'blue')
ax2.plot(t, -MuptP, label = '$P_{max}$', color ='cyan')
ax2.plot(t, f_P_fert_ing, label = '$\phi_{P,fert,ing}$', color = 'orange')
ax2.plot(t, f_P_fert_org, label = '$\phi_{P,fert, org}$', color = 'green')
ax2.plot(t, f_P_phy_cmp, label = '$\phi_{P,phy,cmp}$', color ='red')
ax2.plot(t, f_P_fis_sol, label = '$\phi_{P,fis,sol}$', color = 'purple')
ax2.plot(t, f_P_fis_prt, label = '$\phi_{P,fis,prt}$', color ='brown')
ax2.plot(t, -f_P_phy_upt, label = '$\phi_{P,phy,upt}$', color = 'pink')
ax2.plot(t, -f_P_plt_upt, label = '$\phi_{P,plt,upt}$', color = 'gray')
ax2.set_ylabel(r'Mass flow rate $[kg h^{-1}]$')
ax2.set_xlabel('time [d]')
# ax2.set_xticks(days_all*24, labels=[f"{day}" for day in days_all])
# ax2.set_xticks(days_exp2*24, labels=[f"{day}" for day in days_exp2])
ax2.set_xticks(days_expP*24, labels=[f"{day}" for day in days_expP])
ax2.legend(loc='upper right')

# Create a figure and a set of subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex = True)

# Plot in each subplot
# Subplot 1 (top left)
axs[0, 0].plot(t, f_Ph, label = '$\phi_{pgr}$')
axs[0, 0].set_xlabel('time [d]')
axs[0, 0].set_xticks(days*24, labels=[f"{day}" for day in days])
axs[0, 0].set_ylabel('mass flow rate [$kg CO_2 h^{-1}$]')
axs[0, 0].legend()

# Subplot 2 (top right)
axs[0, 1].plot(t, f_dmv, label = '$\phi_{dmv}$')
axs[0, 1].set_xlabel('time [d]')
axs[0, 1].set_xticks(days*24, labels=[f"{day}" for day in days])
axs[0, 1].set_ylabel('mass flow rate [$kg DM leaf h^{-1}$]')
axs[0, 1].legend()

# Subplot 3 (bottom left)
axs[1, 0].plot(t, f_res, label ='$\phi_{res}$')
axs[1, 0].set_xlabel('time [d]')
axs[1, 0].set_xticks(days*24, labels=[f"{day}" for day in days])
axs[1, 0].set_ylabel('mass flow rate [$kg CH_2O h^{-1}$]')
axs[1, 0].legend()

# Subplot 4 (bottom right)
axs[1, 1].plot(t, f_uptN, label = '$\phi_{N,plt,upt}$')
axs[1, 1].set_xlabel('time [d]')
axs[1, 1].set_xticks(days*24, labels=[f"{day}" for day in days])
axs[1,1].set_ylabel('mass flow rate [$kg N ha^{-1} h^{-1}$]')
axs[1,1].legend()

# Adjust layout so titles and labels don't overlap
plt.tight_layout()