# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 23:19:23 2024

@author: Angela
"""

from models.fish import Fish
from models.rice import Rice
from models.phytoplankton import Phygrowth
from models.bacterialgrowth import Monod, DB, AOB, PSB
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Simulation time array
t = 120
tsim = np.linspace(0.0, t, t+1)
t_weather = np.linspace(0.0, t, t+1) # [d]
dt = 1
tspan = (tsim[0], tsim[-1])
# pond area and volume
area = 10000  # [m2] the total area of the system
rice_area = 6000  # [m2] rice field area
pond_area = area - rice_area  # [m2] pond area
pond_volume = pond_area*0.5
N_fish = [14.56, 18.2, 21.84]  # [g m-3]
n_fish = [(N * pond_volume) / 18.2 for N in N_fish]

# Disturbances
data_weather = 'C:/Users/alegn/Documents/WUR/Thesis/rfmodel/data/Weather/DIY_202210_202309_Daily.csv'
weather = pd.read_csv(data_weather, header=1, sep=';')

#FIRST CYCLE
t_ini = '20221001'
t_end = '20230129'

#SECOND CYCLE
t_ini2 = '20230130'
t_end2 = '20230530'

weather['Time'] = pd.to_datetime(weather['Time'], format='%Y%m%d')  # Adjust the format if necessary
weather.set_index('Time', inplace=True)
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

#State variables
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
    "Mfish": 18.2 * n_fish[0],
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

#Inputs
Norgf = (1.23/100)*1000 #N content in organic fertilizers 
uDB = {'Norgf': Norgf}
Porgf = 1.39/100*1000
uPSB = {'Porgf': 0}
ufish = {"Mfed": 18.2 * 0.03 * n_fish[0]}

# inorganic fertilizer types and concentration:
# Nitrogen source: NPK (15%:15%:15%), Urea (46% N)
NPK_w = 167 * rice_area
I_N = (15 / 100) * NPK_w

# Phosphorus source: SP36 (36%P2O5 ~ 7.85% P)
# P content in SP-36: 7.85% P
SP36 = 31 * rice_area
I_P = (7.85 / 100) * SP36

urice = {"I_N": I_N, "I_P": I_P}  # [kg m-2 d-1]

#Parameters
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

#create dictionary of parameters
p = {
     'DB': pDB,
     'AOB': pAOB,
     'NOB': pNOB,
     'PSB': pPSB,
     'phy': pphy,
     'fish': pfish,
     'rice': price
     }

#Disturbances
#First cycle

dDB1 = {'DO': np.array([tsim, np.random.randint(1,6, size=tsim.size)]).T, # [mg O2 L-1] random number between 1 - 5 to simulate dissolved oxygen in the pond 
'T': np.array([tsim, Tavg1]).T, #[°C] water temperature, assume similar with air temperature
     'f_N_prt': np.array([tsim, np.zeros(tsim.size)]).T,
     'f3':np.array([tsim, np.zeros(tsim.size)]).T
     }

dAOB1 = {
     'DO': np.array([tsim, np.random.randint(1,6, size=tsim.size)]).T, # [mg O2 L-1] random number between 1 - 5 to simulate dissolved oxygen in the pond 
     'T': np.array([tsim, Tavg1]).T, #[°C] water temperature, assume similar with air temperature
     'f_TAN': np.array([tsim, np.zeros(tsim.size)]).T,
     'SNH4': np.array([tsim, np.zeros(tsim.size)]).T
     }

dNOB1 = {
     'DO': np.array([tsim, np.random.randint(1,6, size=tsim.size)]).T, # [mg O2 L-1] random number between 1 - 5 to simulate dissolved oxygen in the pond 
     'T': np.array([tsim, Tavg1]).T, #[°C] water temperature, assume similar with air temperature
     'S_out': np.array([tsim, np.zeros(tsim.size)]).T
     }

dPSB1 = {'DO': np.array([tsim, np.random.randint(1,6, size=tsim.size)]).T, # [mg O2 L-1] random number between 1 - 5 to simulate dissolved oxygen in the pond 
        'T': np.array([tsim, Tavg1]).T, #[°C] water temperature, assume similar with air temperature
     'f_P_prt': np.array([tsim, np.zeros(tsim.size)]).T,
     'f3':np.array([tsim, np.zeros(tsim.size)]).T
     }

dphy1 = {
    "I0": np.array([t_weather, I01]).T,
    "T": np.array([t_weather, Tavg1]).T,
    "Rain": np.array([t_weather, Rain1]).T,
    "DVS": np.array([tsim, np.zeros(tsim.size)]).T,
    "SNH4": np.array([tsim, np.full((tsim.size,), 0.08372093)]).T,
    "SNO2": np.array([tsim, np.full((tsim.size,), 0.305571392)]).T,
    "SNO3": np.array([tsim, np.full((tsim.size,), 0.05326087)]).T,
    "SP": np.array([tsim, np.full((tsim.size,), 0.0327835051546391)]).T,
}

dfish1 = {
    "DO": np.array([tsim, np.random.randint(1, 6, size=tsim.size)]).T,
    "T": np.array([t_weather, Tavg1]).T,
    "Mphy": np.array([tsim, np.full((tsim.size,), 8.586762e-6 / 0.02)]).T,
}

drice1 = {
    "I0": np.array([t_weather, I01]).T,
    "T": np.array([t_weather, Tavg1]).T,
    "CO2": np.array([tsim, np.full((tsim.size,), 400)]).T,
    "SNO3": np.array([tsim, np.full((tsim.size,), 0.05326087)]).T,
    "SP": np.array([tsim, np.full((tsim.size,), 0.0327835051546391)]).T,
    "f_P_sol": np.array([tsim, np.full((tsim.size,), 0.0327835051546391)]).T,
}

d1 = {
     'DB': dDB1,
     'AOB': dAOB1,
     'NOB': dNOB1,
     'PSB': dPSB1,
     'phy': dphy1,
     'fish': dfish1,
     'rice': drice1
     }

#Second cycle

dDB2 = {'DO': np.array([tsim, np.random.randint(1,6, size=tsim.size)]).T, # [mg O2 L-1] random number between 1 - 5 to simulate dissolved oxygen in the pond 
'T': np.array([tsim, Tavg2]).T, #[°C] water temperature, assume similar with air temperature
     'f_N_prt': np.array([tsim, np.zeros(tsim.size)]).T,
     'f3':np.array([tsim, np.zeros(tsim.size)]).T
     }

dAOB2 = {
     'DO': np.array([tsim, np.random.randint(1,6, size=tsim.size)]).T, # [mg O2 L-1] random number between 1 - 5 to simulate dissolved oxygen in the pond 
     'T': np.array([tsim, Tavg2]).T, #[°C] water temperature, assume similar with air temperature
     'f_TAN': np.array([tsim, np.zeros(tsim.size)]).T,
     'SNH4': np.array([tsim, np.zeros(tsim.size)]).T
     }

dNOB2 = {
     'DO': np.array([tsim, np.random.randint(1,6, size=tsim.size)]).T, # [mg O2 L-1] random number between 1 - 5 to simulate dissolved oxygen in the pond 
     'T': np.array([tsim, Tavg2]).T, #[°C] water temperature, assume similar with air temperature
     'S_out': np.array([tsim, np.zeros(tsim.size)]).T
     }

dPSB2 = {'DO': np.array([tsim, np.random.randint(1,6, size=tsim.size)]).T, # [mg O2 L-1] random number between 1 - 5 to simulate dissolved oxygen in the pond 
        'T': np.array([tsim, Tavg2]).T, #[°C] water temperature, assume similar with air temperature
     'f_P_prt': np.array([tsim, np.zeros(tsim.size)]).T,
     'f3':np.array([tsim, np.zeros(tsim.size)]).T
     }

dphy2 = {
    "I0": np.array([t_weather, I02]).T,
    "T": np.array([t_weather, Tavg2]).T,
    "Rain": np.array([t_weather, Rain2]).T,
    "DVS": np.array([tsim, np.zeros(tsim.size)]).T,
    "SNH4": np.array([tsim, np.full((tsim.size,), 0.08372093)]).T,
    "SNO2": np.array([tsim, np.full((tsim.size,), 0.305571392)]).T,
    "SNO3": np.array([tsim, np.full((tsim.size,), 0.05326087)]).T,
    "SP": np.array([tsim, np.full((tsim.size,), 0.0327835051546391)]).T,
}

dfish2 = {
    "DO": np.array([tsim, np.random.randint(1, 6, size=tsim.size)]).T,
    "T": np.array([t_weather, Tavg2]).T,
    "Mphy": np.array([tsim, np.full((tsim.size,), 8.586762e-6 / 0.02)]).T,
}

drice2 = {
    "I0": np.array([t_weather, I02]).T,
    "T": np.array([t_weather, Tavg2]).T,
    "CO2": np.array([tsim, np.full((tsim.size,), 400)]).T,
    "SNO3": np.array([tsim, np.full((tsim.size,), 0.05326087)]).T,
    "SP": np.array([tsim, np.full((tsim.size,), 0.0327835051546391)]).T,
    "f_P_sol": np.array([tsim, np.full((tsim.size,), 0.0327835051546391)]).T,
}

d2 = {
     'DB': dDB2,
     'AOB': dAOB2,
     'NOB': dNOB2,
     'PSB': dPSB2,
     'phy': dphy2,
     'fish': dfish2,
     'rice': drice2
     }

#Instantiate object:
db = DB(tsim, dt, x0DB, pDB)
aob = AOB(tsim, dt, x0AOB, pAOB)
nob = Monod(tsim, dt, x0NOB, pNOB)
psb = PSB(tsim, dt, x0PSB, pPSB)
phyto = Phygrowth(tsim, dt, x0phy, pphy)
fish = Fish(tsim, dt, x0fish, pfish)
rice = Rice(tsim, dt, x0rice, price)

#First cycle
y11 = db.run(tspan, dDB1, uDB)
y21 = aob.run(tspan, dAOB1)
y31 = nob.run(tspan,dNOB1)
y41 = psb.run(tspan, dPSB1, uPSB)
yphy1 = phyto.run(tspan, dphy1)
yfish1 = fish.run(tspan, dfish1, ufish)
yrice1 = rice.run(tspan, drice1, urice)

# class SimulationController:
#     def __init__(self):
#         self.t_start = 0
#         self.t_end = 120
#         self.t_fish_start = 24
#         self.t_fish_end = 55
#         self.t_rice_start = 21

#     def run(self, db, aob, nob, psb, phyto, fish, rice):
#         for t in range(self.t_start, self.t_end + 1):
#             dDB1, dAOB1, dNOB1, dPSB1, dphy1, dfish1, drice1 = {}, {}, {}, {}, {}, {}, {}
#             uDB, uPSB, ufish, urice = {}, {}, {}, {}
            
#             model_outputs = {}
#             # Run all models except fish based on t
#             y11 = db.run(t, dDB1, uDB)
#             y21 = aob.run(t, dAOB1)
#             y31 = nob.run(t, dNOB1)
#             y41 = psb.run(t, dPSB1, uPSB)
#             yphy1 = phyto.run(t, dphy1)
#             yrice1 = rice.run(t, drice1, urice)
            
#             for t in range(self.t_start, self.t_end + 1):
#                 # Assume each model's run method returns a result that is needed by the next model
#                 model_outputs['db'] = 
#                 model_outputs['aob'] = aob.run(t, model_outputs['db'])  # aob uses output from db
#                 model_outputs['nob'] = nob.run(t, model_outputs['aob'])  # nob uses output from aob
#                 model_outputs['psb'] = psb.run(t, model_outputs['nob'])  # psb uses output from nob
#                 model_outputs['phyto'] = phyto.run(t, model_outputs['psb'])  # phyto uses output from psb
                
#             # The rice model uses outputs from db, aob, nob, psb when t >= 21
#             if t >= self.t_rice_start:
#                 # Collect required inputs from previous models
#                 inputs_for_rice = {
#                     'from_db': model_outputs['db'],
#                     'from_aob': model_outputs['aob'],
#                     'from_nob': model_outputs['nob'],
#                     'from_psb': model_outputs['psb']
#                 }
#                 model_outputs['rice'] = rice.run(t, inputs_for_rice)


#             # Run fish model only between t_fish_start and t_fish_end
#             if self.t_fish_start <= t <= self.t_fish_end:
#                 yfish1 = fish.run(t, dfish1)
                
#             return{
#                 'y1': y11,
#                 'y2': y21,
#                 'y3': y31,
#                 'y4': y41,
#                 'yphy': yphy1,
#                 'yfish': yfish1,
#                 'yrice': yrice1
#                 }
        
# control = SimulationController()
# yrcf = control.run(db, aob, nob, psb, phyto, fish, rice)        


#Second cycle
y12 = db.run(tspan, dDB2, uDB)
y22 = aob.run(tspan, dAOB2)
y32 = nob.run(tspan,dNOB2)
y42 = psb.run(tspan, dPSB2, uPSB)
yphy2 = phyto.run(tspan, dphy2)
yfish2 = fish.run(tspan, dfish2, ufish)
yrice2 = rice.run(tspan, drice2, urice)

# #retrieve fishpond simulation results
# t_fp = y_fp['Fish']['t']
# # SNprt = y_fp['SNprt']
# # SPprt = y_fp['SPprt']
# # SNH4 = y_fp['SNH4']
# # SNO2 = y_fp['SNO2']
# SNO3 = y_fp['SNO3']
# SP = y_fp['SP']
# Mfish_fr = y_fp['Yield']/pfish['k_DMR']
# Mphy = y_fp['Phyto']['Mphy']
# # NA = y['NA']

# #retrieve rice simulation results
# t_rice = yrice['t']
# f_Ph= rice.f['f_Ph']
# f_res= rice.f['f_res']
# f_gr = rice.f['f_gr']
# f_dmv = rice.f['f_dmv']
# f_pN = rice.f['f_pN']
# f_Nlv = rice.f['f_Nlv']
# HU = rice.f['HU']
# DVS = rice.f['DVS']
# N_lv = rice.f['N_lv']

# Mrt = yrice['Mrt']
# Mst = yrice['Mst']
# Mlv = yrice['Mlv']
# Mpa = yrice['Mpa']
# Mgr = yrice['Mgr']

# plt.figure(1)
# plt.plot(t_fp, Mfish_fr, label = 'fresh fish weight')
# plt.plot(t_fp, Mphy, label = 'phytoplankton')
# plt.xlabel('time (days)')
# plt.ylabel('Biomass [g d-1]')
# plt.legend()

# plt.figure(2)
# plt.plot(t_rice, Mrt, label='Roots')
# plt.plot(t_rice, Mst, label='Stems')
# plt.plot(t_rice, Mlv, label='Leaves')
# plt.plot(t_rice, Mpa, label='Panicles')
# plt.xlabel(r'time [d]')
# plt.ylabel(r'Dry mass [kg DM ha-1 d-1]')
# plt.title(r'Accumulative Dry Mass of Rice Crop Organs')
# plt.legend()

# plt.figure(3)
# plt.plot(t_rice, f_Ph, label='f_ph')
# plt.plot(t_rice, f_res, label='f_res')
# plt.plot(t_rice, f_gr, label='f_gr')
# plt.plot(t_rice, f_dmv, label='f_dmv')
# plt.xlabel(r'time [d]')
# plt.ylabel(r'flow rate [kg CH20 ha-1 d-1]')
# plt.title(r'Flow rate in rice plants')
# plt.legend()

# plt.figure(4)
# plt.plot(t_rice, f_Nlv, label='N flow rate in leaves')
# plt.plot(t_rice, f_pN, label = 'N flow rate in rice plants')
# plt.xlabel(r'time [d]')
# plt.ylabel(r'flow rate [kg N ha-1 d-1]')
# plt.title(r'Flow rate in rice plants')
# plt.legend()

# n_rice1 = 127980  # number of plants using 2:1 pattern
# n_rice2 = 153600  # number of plants using 4:1 pattern

# plt.figure(5)
# plt.bar('P1', Mgr*n_rice1, label='Grains P1')
# plt.bar('P2', Mgr*n_rice2, label='Grains P2')
# plt.ylabel(r'Dry mass [kg DM ha-1 d-1]')
# plt.legend()
# plt.show()

# plt.show()