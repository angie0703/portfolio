# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 23:19:23 2024

@author: Angela
"""

from models.fish import Fish
from models.rice import Rice
from models.bacterialgrowth import Monod, DB, AOB, PSB
from models.phytoplankton import Phygrowth
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Simulation time array
t = 364
tsim = np.linspace(0.0, t, t+1)
t_weather = np.linspace(0.0, t, t+1) # [d]
dt = 1

tspan = (tsim[0], tsim[-1])
# pond area and volume
area = 10000  # [m2] the total area of the system
rice_area = 6000  # [m2] rice field area
pond_area = area - rice_area  # [m2] pond area
pond_volume = pond_area*0.5
m_fish = 18.2 #[g] average weight of one fish fry
N_fish = [14.56, 18.2, 21.84]  # [g m-3]
n_fish = [(N * pond_volume) / m_fish for N in N_fish]
n_rice = [127980, 153600] #number of plants

# Disturbances
data_weather = 'C:/Users/alegn/Documents/WUR/Thesis/rfmodel/data/Weather/DIY_202210_202309_Daily.csv'
weather = pd.read_csv(data_weather, header=1, sep=';')
weather['Time'] = pd.to_datetime(weather['Time'], format='%Y%m%d')  # Adjust the format if necessary
weather.set_index('Time', inplace=True)

#Disturbances for bacteria and phytoplankton
t_ini = '20221001'
t_end = '20230930'

#first cycle
Tavg = weather.loc[t_ini:t_end,'Tavg'].values #[°C] Mean daily temperature
Rain = weather.loc[t_ini:t_end,'Rain'].values #[mm] Daily precipitation
Igl = weather.loc[t_ini:t_end, 'I0'].values #[MJ m-2] Sum of shortwave radiation daily

I0 = 0.45*Igl*1E6 #Convert [MJ m-2 d-1] to [J m-2 d-1] PAR

#%% External disturbances for rice and fish
## Fish
#FIRST CYCLE
t_ini1 = '20221001'
t_end1 = '20221031'

#Second Cycle
t_ini2 = '20230129'
t_end2 = '20230324'

#Third cycle
t_ini3 = '20230703'
t_end3 = '20230826'

##Rice
#First cycle
t_ini4 = '20221001'
t_end4 = '20230129'

#Second Cycle
t_ini5 = '20230305'
t_end5 = '20230703'

#Fish
#first cycle
Tavg1 = weather.loc[t_ini1:t_end1,'Tavg'].values #[°C] Mean daily temperature

#second cycle
Tavg2 = weather.loc[t_ini2:t_end2,'Tavg'].values #[°C] Mean daily temperature

#third cycle
Tavg3 = weather.loc[t_ini3:t_end3,'Tavg'].values #[°C] Mean daily temperature

#Rice
#first cycle
Tavg4 = weather.loc[t_ini4:t_end4,'Tavg'].values #[°C] Mean daily temperature
Igl4 = weather.loc[t_ini4:t_end4, 'I0'].values #[MJ m-2] Sum of shortwave radiation daily

I04 = 0.45*Igl4*1E6 #Convert [MJ m-2 d-1] to [J m-2 d-1] PAR

#second cycle
Tavg5 = weather.loc[t_ini5:t_end5,'Tavg'].values #[°C] Mean daily temperature
Igl5 = weather.loc[t_ini5:t_end5, 'I0'].values #[MJ m-2] Sum of shortwave radiation daily

I05 = 0.45*Igl5*1E6 #Convert [MJ m-2 d-1] to [J m-2 d-1] PAR
#%% State variables
x0DB = {'S':0.323404255,#Particulate Matter concentration 
        'X':0.2, #decomposer bacteria concentration
        'P': 0.08372093      
        }           #concentration in [g m-3]

x0AOB = {'S':0.08372093,   #Ammonium concentration 
         'X':0.07,   #AOB concentration
         'P': 0.305571392 #Nitrite concentration
         }          #concentration in [g m-3]

x0NOB = {'S':0.305571392,   #Nitrite concentration 
          'X':0.02,   #NOB concentration
          'P': 0.05326087 #Nitrate concentration
          }          #concentration in [g m-3]

x0PSB = {'S':0.05781939,   #particulate P concentration (Mei et al 2023)
         'X':0.03,   #PSB concentration
         'P': 0.0327835051546391 #total Soluble Reactive Phosphorus/total P available for uptake (Mei et al 2023)
         }          #concentration in [g m-3]
x0phy = {
    "Mphy": 8.586762e-6/0.02,  # [g m-3] from Mei et al 2023 concentration of Chla multiply with Chla:phyto mass ratio from Jamu and Piedrahita (2002)
    "NA": 0.55  # [g m-3] TN from Mei et al 2023
}

x0fish = {
    "Mfish": m_fish * n_fish[0],
    "Mdig": 1e-6 * n_fish[0],
    "Muri": 1e-6 * n_fish[0]
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

#%%Inputs
Norgf = (1.23/100)*1000 #N content in organic fertilizers 
uDB = {'Norgf': Norgf}
Porgf = 1.39/100*1000
uPSB = {'Porgf': Porgf}
ufish = {"Mfed": 18.2 * 0.03 * n_fish[0]}

# inorganic fertilizer types and concentration:
# Nitrogen source: NPK (15%:15%:15%), Urea (46% N)
NPK_w = 167 * rice_area
Urea = 100 * rice_area
I_N = (15 / 100) * NPK_w + (46/100)*Urea

# Phosphorus source: SP36 (36%P2O5 ~ 7.85% P)
# P content in SP-36: 7.85% P
SP36 = 31 * rice_area
I_P = (7.85 / 100) * SP36

urice = {"I_N": I_N, "I_P": I_P}  # [kg m-2 d-1]

#%%Parameters
pDB= {
    'mu_max0': 3.8,  #maximum rate of substrate use (d-1) Kayombo 2003
    'Ks':5.14,         #half velocity constant (g m-3)
    'K_DO': 1,    #half velocity constant of DO for the bacteria 
    'Y': 0.82,      #bacteria synthesis yield (g bacteria/ g substrate used)
    'b20': 0.12,     # endogenous bacterial decay (d-1)
    'teta_b': 1.04, #temperature correction factor for b
    'teta_mu': 1.07, #temperature correction factor for mu
    'MrS': 60.07,           #[g/mol] Molecular Weight of urea
    'MrP': 18.05,           #[g/mol] Molecular Weight of NH4
    'a': 2,
    'kN': 0.5
    } 

pAOB= {
   'mu_max0': 1, #range 0.33 - 1 g/g.d, if b = 0.17, mu_max_AOB = 0.9
   'Ks': 0.5, #range 0.14 - 5 g/m3, 0.6 - 3.6 g/m3, or 0.3 - 0.7 g/m3
   'K_DO': 0.5, #range 0.1 - 1 g/m3
   'Y': 0.33, #range 0.10 - 0.15 OR 0.33
   'b20': 0.17,#range 0.15 - 0.2 g/g.d
   'teta_mu': 1.072,
   'teta_b': 1.029,
   'MrS': 18.05,           #[g/mol] Molecular Weight of NH4
   'MrP': 46.01,           #[g/mol] Molecular Weight of NO2
   'a': 1
    } 
pNOB= {
    'mu_max0': 1, #range 0.7 - 1.8 g/g.d
    'Ks': 0.2, #range 0.05 - 0.3 g/m3
    'K_DO': 0.5, #range 0.1 - 1 g/m3
    'Y': 0.08, #0.04 - 0.07 g VSS/g NO2 or 0.08 g VSS/g NO2 
    'b20': 0.17, 
    'teta_b': 1.029,
    'teta_mu': 1.063,
    'MrS': 46.01,           #[g/mol] Molecular Weight of NO2
    'MrP': 62.01,           #[g/mol] Molecular Weight of NO3
    'a': 1
    } 

pPSB= {
    'mu_max0': 3.8,  #maximum rate of substrate use (d-1) 
    'Ks':5.14,         #half velocity constant (g m-3)
    'K_DO': 1,    #half velocity constant of DO for the bacteria 
    'Y': 0.45,      #bacteria synthesis yield (g bacteria/ g substrate used)
    'b20': 0.2,     # endogenous bacterial decay (d-1)
    'teta_b': 1.04, #temperature correction factor for b
    'teta_mu': 1.07, #temperature correction factor for mu
    'kP': 0.5
    } 
pphy = {
    "mu_phy": 1.27,  # [d-1] maximum growth rate of phytoplankton
    "mu_Up": 0.005,  # [d-1] maximum nutrient uptake coefficient
    "pd": 0.5,  # [m] pond depth
    "l_sl": 0.00035,  # [m2 mg-1] phytoplankton biomass-specific light attenuation
    "l_bg": 0.77,  # [m-1] light attenuation by non-phytoplankton components
    "Kpp": 0.234*60,  # [J m-2 s-1] half-saturation constant of phytoplankton production
    "cm": 0.15,  # [d-1] phytoplankton mortality constant
    "cl": 4e-6,  # [m2 mg-1] phytoplankton crowding loss constant
    "c1": 1.57,  # [-] temperature coefficients
    "c2": 0.24,  # [-] temperature coefficients
    "Topt": 28,  # optimum temperature for phytoplankton growth
    "Mp": 0.025,  # [g m-3] half saturation constant for nutrient uptake
    "dr": 0.21,  # [d-1] dilution rate (0.21 - 0.45)
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
}

price = {
    "DVSi": 0.001,
    "DVRJ": 0.0013,
    "DVRI": 0.001125,
    "DVRP": 0.001275,
    "DVRR": 0.003,
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
    "IgN": 0.5,
}

#%%Disturbances
dDB = {'DO': np.array([tsim, np.random.randint(1,6, size=tsim.size)]).T, # [mg O2 L-1] random number between 1 - 5 to simulate dissolved oxygen in the pond 
'T': np.array([t_weather, Tavg]).T, #[°C] water temperature, assume similar with air temperature
     'f_N_prt': np.array([tsim, np.zeros(tsim.size)]).T,
     'f3':np.array([tsim, np.zeros(tsim.size)]).T
     }

dAOB = {
     'DO': np.array([tsim, np.random.randint(1,6, size=tsim.size)]).T, # [mg O2 L-1] random number between 1 - 5 to simulate dissolved oxygen in the pond 
     'T': np.array([t_weather, Tavg]).T, #[°C] water temperature, assume similar with air temperature
     'f_TAN': np.array([tsim, np.zeros(tsim.size)]).T,
     'SNH4': np.array([tsim, np.zeros(tsim.size)]).T
     }

dNOB = {
     'DO': np.array([tsim, np.random.randint(1,6, size=tsim.size)]).T, # [mg O2 L-1] random number between 1 - 5 to simulate dissolved oxygen in the pond 
     'T': np.array([t_weather, Tavg]).T, #[°C] water temperature, assume similar with air temperature
     'S_out': np.array([tsim, np.zeros(tsim.size)]).T
     }

dPSB = {'DO': np.array([tsim, np.random.randint(1,6, size=tsim.size)]).T, # [mg O2 L-1] random number between 1 - 5 to simulate dissolved oxygen in the pond 
        'T': np.array([t_weather, Tavg]).T, #[°C] water temperature, assume similar with air temperature
     'f_P_prt': np.array([tsim, np.zeros(tsim.size)]).T,
     'f3':np.array([tsim, np.zeros(tsim.size)]).T,
     "f_P_sol": np.array([tsim, np.full((tsim.size,), 0.0327835051546391)]).T
     }

dphy= {
    "I0": np.array([t_weather, I0]).T,
    "T": np.array([t_weather, Tavg]).T,
    "Rain": np.array([t_weather, Rain]).T,
    "DVS": np.array([tsim, np.zeros(tsim.size)]).T,
    "SNH4": np.array([tsim, np.full((tsim.size,), 0.08372093)]).T,
    "SNO2": np.array([tsim, np.full((tsim.size,), 0.305571392)]).T,
    "SNO3": np.array([tsim, np.full((tsim.size,), 0.05326087)]).T,
    "SP": np.array([tsim, np.full((tsim.size,), 0.0327835051546391)]).T,
}

#%%Disturbances for fish and rice
# TODO: Cycle Time
ct_f1 = np.linspace(0,30,30+1, dtype=int)
ct_f2 = np.linspace(122,176,54+1, dtype=int)
ct_f3 = np.linspace(276, 331, 54+1, dtype = int)
ct_r1 = np.linspace(0,120,120+1, dtype=int)
ct_r2= np.linspace(155, 275, 120+1, dtype=int)

dfish1 = {
    "DO": np.array([tsim[ct_f1], np.random.randint(1, 6, size=tsim[ct_f1].size)]).T,
    "T": np.array([t_weather[ct_f1], Tavg1]).T,
    "Mphy": np.array([tsim[ct_f1], np.full((tsim[ct_f1].size,), 8.586762e-6 / 0.02)]).T,
}
dfish2 = {
    "DO": np.array([tsim[ct_f2], np.random.randint(1, 6, size=tsim[ct_f2].size)]).T,
    "T": np.array([t_weather[ct_f2], Tavg2]).T,
    "Mphy": np.array([tsim[ct_f2], np.full((tsim[ct_f2].size,), 8.586762e-6 / 0.02)]).T,
}
dfish3 = {
    "DO": np.array([tsim[ct_f3], np.random.randint(1, 6, size=tsim[ct_f3].size)]).T,
    "T": np.array([t_weather[ct_f3], Tavg3]).T,
    "Mphy": np.array([tsim[ct_f3], np.full((tsim[ct_f3].size,), 8.586762e-6 / 0.02)]).T,
}

drice1 = {
    "I0": np.array([t_weather[ct_r1], I04]).T,
    "T": np.array([t_weather[ct_r1], Tavg4]).T,
    "CO2": np.array([tsim, np.full((tsim.size,), 400)]).T,
    "SNO3": np.array([tsim, np.full((tsim.size,), 0.05326087)]).T,
    "SP": np.array([tsim, np.full((tsim.size,), 0.0327835051546391)]).T,
}
drice2 = {
    "I0": np.array([t_weather[ct_r2], I05]).T,
    "T": np.array([t_weather[ct_r2], Tavg5]).T,
    "CO2": np.array([tsim, np.full((tsim.size,), 400)]).T,
    "SNO3": np.array([tsim, np.full((tsim.size,), 0.05326087)]).T,
    "SP": np.array([tsim, np.full((tsim.size,), 0.0327835051546391)]).T,
}
#%% Object and run model
#Instantiate object:
fish1 = Fish(tsim, dt, x0fish, pfish)
fish2 = Fish(tsim, dt, x0fish, pfish)
fish3 = Fish(tsim, dt, x0fish, pfish)

rice1 = Rice(tsim, dt, x0rice, price)
rice2 = Rice(tsim, dt, x0rice, price)

db = DB(tsim, dt, x0DB, pDB)
aob = AOB(tsim, dt, x0AOB, pAOB)
nob = Monod(tsim, dt, x0NOB, pNOB)
psb = PSB(tsim, dt, x0PSB, pPSB)
phy = Phygrowth(tsim, dt, x0phy, pphy)

for day in range(len(tsim)):
    # Define today's timespan as a single day, or potentially a rolling window
    tspan = (tsim[day], tsim[min(day+1, len(tsim)-1)])  # this might need adjustment based on how models interpret tspan

    print(f"Simulating day {day+1} / {len(tsim)}")
    #Run bacteria and phytoplankton model
    ydb = db.run(tspan, dDB, uDB)
    #retrieve DB result for AOB
    dDB['SNH4'] = np.array([ydb["t"], ydb["P"]]).T
    #Run aob model
    yaob =aob.run(tspan, dAOB)
    dNOB['S_out'] = np.array([yaob["t"], yaob["P"]]).T
    #Run nob model
    ynob = nob.run(tspan, dNOB)
    
    ypsb = psb.run(tspan, dPSB, uPSB)
    
    #make simulation result of y1 and y2 as Phygrowth state variable of x0['phy']['NA']
    dphy['SNH4'] = np.array([ydb["t"], ydb["P"]]).T
    dphy['SNO2'] = np.array([yaob["t"], yaob["P"]]).T
    dphy['SNO3'] = np.array([ynob["t"], ynob["P"]]).T 
    dphy['SP'] = np.array([ypsb["t"], ypsb["P"]]).T 
    
    yphy = phy.run(tspan, dphy)
    
    dfish1['Mphy'] = np.array([tsim[int(t)], yphy["Mphy"][int(t)]]).T
    dfish2['Mphy'] = np.array([tsim[ct_f2], yphy["Mphy"][ct_f2]]).T 
    dfish3['Mphy'] = np.array([tsim[ct_f3], yphy["Mphy"][ct_f3]]).T  
    
    #Run fish model
    yfish1 = fish1.run((0,31), dfish1, ufish)
    flows_fish1 = {key: np.array([x for x in value if not np.isnan(x)]) for key, value in fish1.f.items()}
    f_N_prt1 = flows_fish1['f_N_prt']
    f_TAN1 = flows_fish1['f_TAN']
    f_P_prt1 = flows_fish1['f_P_prt']
    f_P_sol1 = flows_fish1['f_P_sol']
    rphy1 = flows_fish1['rphy']
    
    if 0 <= day < 30: 
        dDB['f3'] = np.array([yphy["t"], phy.f['f3']]).T 
        dDB['f_N_prt'][:31,:]= np.array([yfish1["t"][ct_f1], f_N_prt1]).T
        dAOB['f_TAN'][:31,:] = np.array([yfish1["t"][ct_f1], f_TAN1]).T
        dPSB['f3'] = np.array([yphy["t"], phy.f['f3']]).T
        dPSB['f_P_prt'][:31,:] = np.array([yfish1["t"][ct_f1], f_P_prt1]).T
        dPSB['f_P_sol'][:31,:] = np.array([yfish1["t"][ct_f1], f_P_sol1]).T
    
    yfish2 = fish2.run((122,177), dfish2, ufish)
    flows_fish2 = {key: np.array([x for x in value if not np.isnan(x)]) for key, value in fish2.f.items()}
    f_N_prt2 = fish2.f['f_N_prt'][ct_f2]
    f_TAN2 = fish1.f['f_TAN'][ct_f2]
    f_P_prt2 = fish1.f['f_P_prt'][ct_f2]
    f_P_sol2 = fish1.f['f_P_sol'][ct_f2]
    if 122 <=day < 177:
        dDB['f3'] = np.array([yphy["t"], phy.f['f3']]).T 
        dDB['f_N_prt'] = np.array([yfish2["t"], fish2.f['f_N_prt']]).T
        dAOB['f_TAN'] = np.array([yfish2["t"], fish2.f['f_TAN']]).T
        dPSB['f3'] = np.array([yphy["t"], phy.f['f3']]).T
        dPSB['f_P_prt'] = np.array([yfish2["t"], fish2.f['f_P_prt']]).T
        dPSB['f_P_sol'] = np.array([yfish2["t"], fish2.f['f_P_sol']]).T
    
    yfish3 = fish3.run((276,332), dfish3, ufish)
    flows_fish3 = {key: np.array([x for x in value if not np.isnan(x)]) for key, value in fish3.f.items()}
    f_N_prt3 = fish2.f['f_N_prt'][ct_f3]
    f_TAN3 = fish1.f['f_TAN'][ct_f3]
    f_P_prt3 = fish1.f['f_P_prt'][ct_f3]
    f_P_sol3 = fish1.f['f_P_sol'][ct_f3]
    if 276<=day <332:
        dDB['f3'] = np.array([yphy["t"], phy.f['f3']]).T 
        dDB['f_N_prt'] = np.array([yfish1["t"], fish1.f['f_N_prt']]).T
        dAOB['f_TAN'] = np.array([yfish1["t"], fish1.f['f_TAN']]).T
        dPSB['f3'] = np.array([yphy["t"], phy.f['f3']]).T
        dPSB['f_P_prt'] = np.array([yfish1["t"], fish1.f['f_P_prt']]).T
        dPSB['f_P_sol'] = np.array([yfish1["t"], fish1.f['f_P_sol']]).T
    
    if 30 <= day < 121:
        drice1['SNO3'] = np.array([ynob["t"][ct_r1], ynob["P"][ct_r1]]).T 
        drice1['SP'] = np.array([ypsb["t"][ct_r1], ypsb["P"][ct_r1]]).T 
    if 185 <= day < 276:
        drice2['SNO3'] = np.array([ynob["t"][ct_r2], ynob["P"][ct_r2]]).T 
        drice2['SP'] = np.array([ypsb["t"][ct_r2], ypsb["P"][ct_r2]]).T
    
    #Run rice model
    yrice1 = rice1.run((0,121), drice1, urice)
    #Retrieve DVS1 as input for dphy
    DVS1 = rice1.f['DVS']
    dphy['DVS'][ct_r1,:] = np.array([yrice1['t'][ct_r1], DVS1]).T
    yrice2 = rice2.run((155, 276), drice2, urice)
    #Retrieve DVS2 as input for dphy
    DVS2 = rice2.f['DVS']
    dphy['DVS'][ct_r2,:] = np.array([yrice2['t'][:121], DVS2]).T

#Retrieve simulation results
#Fish weights, stocking density 1
flows_rice1 = {key: np.array([x for x in value if not np.isnan(x)]) for key, value in rice1.f.items()}
f_Ph1 = flows_rice1['f_Ph']
f_gr1 = flows_rice1['f_gr']
f_Nlv1 = flows_rice1['f_Nlv']
f_pN1 = flows_rice1['f_pN']
N_lv1 = flows_rice1['N_lv']


flows_rice2 = {key: np.array([x for x in value if not np.isnan(x)]) for key, value in rice2.f.items()}
f_Ph2 = flows_rice2['f_Ph']
f_gr2 = flows_rice2['f_gr']
f_Nlv2 = flows_rice2['f_Nlv']
f_pN2 = flows_rice2['f_pN']
N_lv2 = flows_rice2['N_lv']
DVS2 = flows_rice2['DVS']

Mfish_fr1 = yfish1['Mfish']/pfish['k_DMR']
Mfish_fr2 = yfish2['Mfish']/pfish['k_DMR']
Mfish_fr3 = yfish3['Mfish']/pfish['k_DMR']

#Fish weights, stocking density 2
Mfish_fr1 = yfish1['Mfish']/pfish['k_DMR']
Mfish_fr2 = yfish2['Mfish']/pfish['k_DMR']
Mfish_fr3 = yfish3['Mfish']/pfish['k_DMR']

#Fish weights, stocking density 3
Mfish_fr1 = yfish1['Mfish']/pfish['k_DMR']
Mfish_fr2 = yfish2['Mfish']/pfish['k_DMR']
Mfish_fr3 = yfish3['Mfish']/pfish['k_DMR']

#Rice grain weight
Mgr11 = yrice1['Mgr']*n_rice[0]
Mgr21 = yrice2['Mgr']*n_rice[0]
Mgr12 = yrice1['Mgr']*n_rice[1]
Mgr22 = yrice2['Mgr']*n_rice[1]

fig, ax1 = plt.subplots()

# Plotting on the first y-axis
ax1.plot(yfish1['t'], Mfish_fr1, 'g-', label='Fish cycle 1')
ax1.plot(yfish2['t'] + 122, Mfish_fr2, 'g-', label='Fish cycle 2')
ax1.plot(yfish3['t'] + 275, Mfish_fr3, 'g-', label='Fish cycle 3')  # Corrected label
ax1.set_xlabel('Days')
ax1.set_ylabel('Fish Mass (kg)', color='g')
ax1.tick_params(axis='y', labelcolor='g')

# Create the second y-axis
ax2 = ax1.twinx()
ax2.plot(yrice1['t'], yrice1['Mgr'], 'r-', label='Rice Cycle 1')
ax2.plot(yrice2['t'] + 155, yrice2['Mgr'], 'r-', label='Rice Cycle 2')
ax2.set_ylabel('Rice Growth Rate (kg/day)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Optionally add legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.show()

#%% Visualization

#rice 1 flows 


#phytoplankton
Mphy = yphy['Mphy']
NA = yphy['NA']
