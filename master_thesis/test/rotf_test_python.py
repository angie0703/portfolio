# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 23:19:23 2024

@author: Angela
"""
from models.rotf_python import ROTF
import numpy as np
import pandas as pd 

# Simulation time array
t = 364
tsim = np.linspace(0.0, t, t+1)
t_weather = pd.date_range(start='20221001', end = '20230930', freq='D')
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

#%%Weather data extraction
# Function to extract weather data for specified dates and columns
def extract_weather_data(weather, start_date, end_date, columns):
    # Filter data for the given date range and columns
    filtered_data = weather.loc[start_date:end_date, columns]
    # Convert the data to a dictionary of numpy arrays for easy access
    return {col: filtered_data[col].values for col in columns}

def setup_weather_cycles(weather_df, cycles_info):
    cycle_data = {}
    for cycle_name, (start_date, end_date) in cycles_info.items():
        cycle_data[cycle_name] = extract_weather_data(weather_df, start_date, end_date, ['Tavg', 'Rain', 'I0'])
    return cycle_data

# d
data_weather = 'C:/Users/alegn/Documents/WUR/Thesis/rfmodel/data/Weather/DIY_202210_202309_Daily.csv'
weather = pd.read_csv(data_weather, header=1, sep=';')
weather['Time'] = pd.to_datetime(weather['Time'], format='%Y%m%d')  # Adjust the format if necessary
weather.set_index('Time', inplace=True)

#%% External d for rice and fish
cultivation_cycles ={
    'all': ('20221001', '20230930'),
    'f1':('20221001','20221030'),
    'f2': ('20230129','20230324'),
    'f3': ('20230703', '20230826'),
    'r1': ('20221001', '20230129'),
    'r2': ('20230305', '20230703')
    }

def get_cycle_weather(weather_df, cultivation_cycle, elements):
    # Prepare a dictionary to store weather data for all cycles
    cycle_weather_data = {}
    for cycle, dates in cultivation_cycle.items():
        start_date, end_date = dates
        cycle_weather_data[cycle] = extract_weather_data(weather_df, start_date, end_date, elements)
    return cycle_weather_data

# Elements to extract from the weather data
elements = ['Tavg', 'Rain', 'I0']
weather_data = get_cycle_weather(weather, cultivation_cycles, elements)
Tavg = weather_data['all']['Tavg']

# Convert Igl for photosynthetic active radiation
for cycle, data in weather_data.items():
    data['I0'] = 0.45 * data['I0'] * 1E6  # Conversion factor
        
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
    "Mfish": m_fish,
    "Mdig": 1e-6,
    "Muri": 1e-6
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
x0 = {'DB': x0DB, 'AOB': x0AOB, 'NOB': x0NOB, 'PSB': x0PSB, 'phy': x0phy, 'fish':x0fish, 'rice': x0rice}
#%%Inputs
Norgf = (1.23/100)*1000 #N content in organic fertilizers 
uDB = {'Norgf': Norgf}
Porgf = 1.39/100*1000
uPSB = {'Porgf': Porgf}
ufish = {"Mfed": m_fish * 0.03}

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
    "Tmax": 32
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
p = {'DB': pDB, 'AOB': pAOB, 'NOB': pNOB, 'PSB': pPSB, 'phy': pphy, 'fish': pfish, 'rice': price}

#%%d
dDB = {'DO': np.array([tsim, np.random.randint(3,6, size=tsim.size)]).T, # [mg O2 L-1] random number between 1 - 5 to simulate dissolved oxygen in the pond 
       'T': np.array([t_weather, Tavg]).T, #[°C] water temperature, assume similar with air temperature
     'f_N_prt': np.array([tsim, np.zeros(tsim.size)]).T,
     'f3':np.array([tsim, np.zeros(tsim.size)]).T
     }

dAOB = {
     'DO': np.array([tsim, np.random.randint(3,6, size=tsim.size)]).T, # [mg O2 L-1] random number between 1 - 5 to simulate dissolved oxygen in the pond 
     'T': np.array([t_weather, Tavg]).T, #[°C] water temperature, assume similar with air temperature
     'f_TAN': np.array([tsim, np.zeros(tsim.size)]).T,
     'SNH4': np.array([tsim, np.zeros(tsim.size)]).T
     }

dNOB = {
     'DO': np.array([tsim, np.random.randint(3,6, size=tsim.size)]).T, # [mg O2 L-1] random number between 1 - 5 to simulate dissolved oxygen in the pond 
     'T': np.array([t_weather, Tavg]).T, #[°C] water temperature, assume similar with air temperature
     'S_out': np.array([tsim, np.zeros(tsim.size)]).T
     }

dPSB = {'DO': np.array([tsim, np.random.randint(3,6, size=tsim.size)]).T, # [mg O2 L-1] random number between 1 - 5 to simulate dissolved oxygen in the pond 
        'T': np.array([t_weather, Tavg]).T, #[°C] water temperature, assume similar with air temperature
     'f_P_prt': np.array([tsim, np.zeros(tsim.size)]).T,
     'f3':np.array([tsim, np.zeros(tsim.size)]).T,
     "f_P_sol": np.array([tsim, np.full((tsim.size,), 0.0327835051546391)]).T
     }

dphy= {
    "I0": np.array([t_weather, weather_data['all']['I0']]).T,
    "T": np.array([t_weather, weather_data['all']['Tavg']]).T,
    "Rain": np.array([t_weather, weather_data['all']['Rain']]).T,
    "DVS": np.array([tsim, np.zeros(tsim.size)]).T,
    "SNH4": np.array([tsim, np.full((tsim.size,), 0.08372093)]).T,
    "SNO2": np.array([tsim, np.full((tsim.size,), 0.305571392)]).T,
    "SNO3": np.array([tsim, np.full((tsim.size,), 0.05326087)]).T,
    "SP": np.array([tsim, np.full((tsim.size,), 0.0327835051546391)]).T,
}
dfish = {
    "DO": np.array([tsim, np.random.randint(3, 6, size=tsim.size)]).T,
    "T": np.array([t_weather, weather_data['all']['Tavg']]).T,
    "Mphy": np.array([tsim, np.full((tsim.size,), 8.586762e-6/0.02)]).T,
}
drice = {
    "I0": np.array([t_weather, weather_data['all']['I0']]).T,
    "T": np.array([t_weather, weather_data['all']['Tavg']]).T,
    "CO2": np.array([tsim, np.full((tsim.size,), 400)]).T,
    "SNO3": np.array([tsim, np.full((tsim.size,), 0.05326087)]).T,
    "SP": np.array([tsim, np.full((tsim.size,), 0.0327835051546391)]).T,
}
d = {'DB': dDB, 'AOB': dAOB, 'NOB': dNOB, 'PSB': dPSB, 'phy': dphy, 'fish':dfish, 'rice': drice}

#%% Instantiate the object and run simulation
rotf00 = ROTF(tsim, dt,x0, p, cultivation_cycles, n_fish[0], n_rice[0])
rotf01 = ROTF(tsim, dt,x0, p, cultivation_cycles, n_fish[0], n_rice[1])
rotf10 = ROTF(tsim, dt,x0, p, cultivation_cycles, n_fish[1], n_rice[0])
rotf11 = ROTF(tsim, dt,x0, p, cultivation_cycles, n_fish[1], n_rice[1])
rotf20 = ROTF(tsim, dt,x0, p, cultivation_cycles, n_fish[2], n_rice[0])
rotf21 = ROTF(tsim, dt,x0, p, cultivation_cycles, n_fish[2], n_rice[1])

y00 = rotf00.run_simulation()