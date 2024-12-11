# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 10:11:48 2024

@author: Angela

Research Question 1: “Which combination of fish stocking density 
and rice planting density that optimizes rice and fish productivity?”

"""

from models.bacterialgrowth import Monod
from models.fish import Fish
from models.phytoplankton import Phygrowth
from models.rice import Rice
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('ggplot')

# Simulation time array
t = 120
tsim = np.linspace(0.0, t, t+1) # [d]
dt = 1
tspan = (tsim[0], tsim[-1])

#disturbances
data_weather = 'C:/Users/alegn/Documents/WUR/Thesis/rfmodel/data/Weather/DIY_202210_202309_Daily.csv'
weather = pd.read_csv(data_weather, header=1, sep=';')

t_ini = '20221001'
t_end = '20230129'
weather['Time'] = pd.to_datetime(weather['Time'], format='%Y%m%d')  # Adjust the format if necessary
weather.set_index('Time', inplace=True)
Tavg = weather.loc[t_ini:t_end,'Tavg'].values #[°C] Mean daily temperature
Rain = weather.loc[t_ini:t_end,'Rain'].values #[mm] Daily precipitation
Igl = weather.loc[t_ini:t_end, 'I0'].values #[MJ m-2] Sum of shortwave radiation daily

I0 = 0.45*Igl*1E6 #Convert [MJ m-2 d-1] to [J m-2 d-1] PAR

d_pond = {
    'DO': np.array([tsim, np.random.randint(1,6, size=tsim.size)]).T, # [mg O2 L-1] random number between 1 - 5 to simulate dissolved oxygen in the pond 
    'T': np.array([tsim, Tavg]).T #[°C] water temperature, assume similar with air temperature
    }

dt_bacteria = 1 #[d]
#pond area and volume
area = 10000 #[m2] the total area of the system
rice_area = 6000 #[m2] rice field area
pond_area = area - rice_area #[m2] pond area

##Decomposer Bacteria
x0DB = {'S':0.323404255,#Particulate Matter concentration 
        'X':0.27, #decomposer bacteria concentration
        'P': 0.08372093      
        }           #concentration in [g m-3]

#Organic fertilizer used: chicken manure
#N content = 1.23% N/kg fertilizers
#Chicken manure weight = 1000 kg (~500 kg/ha/week) (KangOmbe et al., 2006)
Norgf = (1.23/100)*1000 #N content in organic fertilizers 
if t > 0: 
    Norgf = 0
    
x0DB['S'] = x0DB['S'] + Norgf 

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
    'a': 2
    } 

#initialize object
db = Monod(tsim, dt_bacteria, x0DB, pDB)

##Ammonification by AOB 
x0AOB = {'S':0.08372093,   #Ammonium concentration 
         'X':0.5,   #AOB concentration
         'P': 0.305571392 #Nitrite concentration
         }          #concentration in [g m-3]
pAOB= {
   'mu_max0': 0.9, #range 0.33 - 1 g/g.d, if b = 0.17, mu_max_AOB = 0.9
   'Ks': 0.5, #range 0.14 - 5 g/m3, 0.6 - 3.6 g/m3, or 0.3 - 0.7 g/m3
   'K_DO': 0.5, #range 0.1 - 1 g/m3
   'Y': 0.15,
   'b20': 0.17,#range 0.15 - 0.2 g/g.d
   'teta_mu': 1.072,
   'teta_b': 1.029,
    'MrS': 18.05,           #[g/mol] Molecular Weight of NH4
    'MrP': 46.01,           #[g/mol] Molecular Weight of NO2
    'a': 1
    } 

#initialize object
aob = Monod(tsim, dt_bacteria, x0AOB, pAOB)

# #Nitrification by NOB
x0NOB = {'S':0.305571392,   #Nitrite concentration 
          'X':0.5,   #NOB concentration
          'P': 0.05326087 #Nitrate concentration
          }          #concentration in [g m-3]

pNOB= {
    'mu_max0': 1, #range 0.7 - 1.8 g/g.d
    'Ks': 0.2, #range 0.05 - 0.3 g/m3
    'K_DO': 0.5, #range 0.1 - 1 g/m3
    'Y': 0.05, 
    'b20': 0.17, 
    'teta_b': 1.029,
    'teta_mu': 1.063,
    'MrS': 46.01,           #[g/mol] Molecular Weight of NO2
    'MrP': 62.01,           #[g/mol] Molecular Weight of NO3
    'a': 1
    } 

r = 2 #r=[2, 2.5, 3]
pNOB['K_DO'] = r*pAOB['K_DO']

# #instantiate Model
nob = Monod(tsim, dt_bacteria, x0NOB, pNOB)

##Phosphorus Solubilizing by PSB 
x0PSB = {'S':0.05781939,   #particulate P concentration (Mei et al 2023)
         'X':0.5,   #PSB concentration
         'P': 0.0327835051546391 #total Soluble Reactive Phosphorus/total P available for uptake (Mei et al 2023)
         }          #concentration in [g m-3]

#P content in organic fertilizers
Porgf = (1.39/100)*1000
if t > 0: 
    Porgf = 0
    
x0PSB['S'] = x0PSB['S'] + Porgf


pPSB= {
    'mu_max0': 0.02,  #maximum rate of substrate use (d-1) 
    'Ks':0.00514,         #half velocity constant (mg L-1)
    'K_DO': 1,    #half velocity constant of DO for the bacteria 
    'Y': 0.45,      #bacteria synthesis yield (mg bacteria/ mg substrate used)
    'b20': 0.2,     # endogenous bacterial decay (d-1)
    'teta_b': 1.04, #temperature correction factor for b
    'teta_mu': 1.07, #temperature correction factor for mu
    'MrS': 647.94,           #[g/mol] Molecular Weight of C6H6O24P6
    'MrP': 98.00,           #[g/mol] Molecular Weight of H2PO4
    'a': 6
    } 

#initialize object
psb = Monod(tsim, dt_bacteria, x0PSB, pPSB)


##phytoplankton model
x0phy = {'Mphy': 8.586762e-6/0.02, #[g m-3] from Mei et al 2023 concentration of Chla multiply with Chla:phyto mass ratio from Jamu and Piedrahita (2002)
      'NA':   0.55 #[g m-3] TN from Mei et al 2023
      } 


# Model parameters
pphy = {
      'mu_phy':  1.27, #[d-1] maximum growth rate of phytoplankton
      'mu_Up': 0.005, #[d-1] maximum nutrient uptake coefficient
      'pd': 0.5, #[m] pond depth
      'l_sl': 0.00035, #[m2 mg-1] phytoplankton biomass-specific light attenuation
      'l_bg': 0.77, #[m-1] light attenuation by non-phytoplankton components
      'Kpp': 0.234*60, # [J m-2 s-1] half-saturation constant of phytoplankton production
      'cm': 0.15, #[d-1] phytoplankton mortality constant
      'cl': 4e-6, #[m2 mg-1] phytoplankton crowding loss constant
      'c1': 1.57, # [-] temperature coefficients
      'c2': 0.24, # [-] temperature coefficients
      'Topt': 28, #optimum temperature for phytoplankton growth
      'Mp': 0.025, #[g m-3] half saturation constant for nutrient uptake
      'dr': 0.21, #[d-1] dilution rate (0.21 - 0.45)
     }

#initialize object
phyto = Phygrowth(tsim, dt, x0phy, pphy)
flows_phyto = phyto.f

##fish model
#state variables
N_fish = [14.56, 18.2, 21.84] #[g m-3]
pond_volume = pond_area * 0.5
n_fish = [(N*pond_volume)/18.2 for N in N_fish]


x0fish = {
        'Mfish': 18.2 * n_fish[0],
        'Mdig': 1E-6 * n_fish[0],
        'Muri': 1E-6 * n_fish[0]
    }

ufish = {'Mfed': 18.2 * 0.03 * n_fish[0]}
 
#parameters
pfish= {
    'tau_dig': 4.5,  # [h] time constants for digestive
    'tau_uri':20,  # [h] time constants for urinary
    'k_upt':0.3,  # [-] fraction of nutrient uptake for fish weight
    'k_N_upt':0.40,  # [-] fraction of N uptake by fish
    'k_P_upt':0.27,  # [-] fraction of P uptake by fish
    'k_prt':0.2,  # [-] fraction of particulate matter excreted
    'k_N_prt':0.05,  # [-] fraction of N particulate matter excreted
    'k_P_prt':0.27,  # [-] fraction of P particulate matter excreted
    'x_N_fed': 0.05, #[-] fraction of N in feed
    'x_P_fed': 0.01, #[-] fraction of P in feed
    'k_DMR': 0.31,
    'k_TAN_sol': 0.15,
    'Ksp':1,  #[g C m-3] Half-saturation constant for phytoplankton
    'Tmin': 15,
    'Topt': 25,
    'Tmax': 35
    }

#initialize object
fish = Fish(tsim, dt, x0fish, pfish)
flows_fish = fish.f

# Rice model
n_rice1 = 127980 #number of plants using 2:1 pattern
n_rice2 = 153600 #number of plants using 4:1 pattern 

x0rice = {'Mrt':0.001,    #[kg DM ha-1 d-1] Dry biomass of root
      'Mst':0.001,    #[kg DM ha-1 d-1] Dry biomass of stems
      'Mlv': 0.001,   #[kg DM ha-1 d-1] Dry biomass of leaves
      'Mpa': 0.0,   #[kg DM ha-1 d-1] Dry biomass of panicles
      'Mgr': 0.0,   #[kg DM ha-1 d-1] Dry biomass of grains
      'HU': 0.0
      } 

# Model parameters
price = {
     'DVSi' : 0.001,
     'DVRJ': 0.0013,
     'DVRI': 0.001125,
     'DVRP': 0.001275,
     'DVRR': 0.003,
     'Tmax': 40,
     'Tmin': 15,
     'Topt': 33,
     'k': 0.4,
     'mc_rt': 0.01,
     'mc_st': 0.015,
     'mc_lv': 0.02,
     'mc_pa': 0.003,
     'N_lv': 0,
     'Rec_N': 0.5,
     'k_lv_N': 0,
     'k_pa_maxN': 0.0175,
     'M_upN': 8,
     'cr_lv': 1.326,
     'cr_st': 1.326,
     'cr_pa': 1.462,
     'cr_rt': 1.326,
     'IgN': 0.5      
     }

# Initialize module
rice = Rice(tsim, dt, x0rice, price)

# Iterator
fish_active = False
rice_active = False

#initialized accumulated flows
f_P_sol_accumulated = 0
f_TAN_accumulated = 0

it = np.nditer(tsim[:-1], flags=['f_index'])

for ti in it:
    # Index for current time instant
    idx = it.index
    # Integration span
    tspan = (tsim[idx], tsim[idx+1])
    print('Integrating', tspan)
    
    # Controlled inputs
    #inorganic fertilizer types and concentration:
    #Nitrogen source: NPK (15%:15%:15%), Urea (46% N)
    NPK_w = 167*rice_area
    I_N = (15/100)*NPK_w
    
    #Phosphorus source: SP36 (36%P2O5 ~ 7.85% P)
    #P content in SP-36: 7.85% P
    Porgf = (7.85/100)
    urice = {'I_N':I_N}            # [kg m-2 d-1]
    
    # Activate fish model at t = 15 and re-activate at t = 61
    if ti == 15 or ti == 61:
        fish_active = True

    # Start rice model at t = 21
    if ti == 21:
        rice_active = True

    # Harvest fish at t = 65 and t = 115, and deactivate fish model
    if ti == 65 or ti == 115:
        fish_active = False

    # Harvest rice at t = 120 and deactivate rice model
    if ti == 120:
        rice_active = False
    
    # Run DB model
    y1 = db.run(tspan,d_pond)

    # Retrieve DB model outputs for AOB model
    S_NH4 =  y1['P']
    x0AOB['S'] = np.array([y1['t'], S_NH4])

    # Run AOB model    
    y2 = aob.run(tspan,d_pond)

    # Retrieve AOB model outputs for NOB model
    x0NOB['S']= np.array([y2['t'], y2['P']])

    # #run NOB model
    y3 = nob.run(tspan,d_pond)

    # #run PSB model
    y4 = psb.run(tspan,d_pond)
    
    #retrieve PSB outputs as phytoplankton and rice input (disturbances)    
    S_P = y4['P']

    #retrieve output of DB, AOB, and NOB as phytoplankton inputs (disturbances)
    dphy = {
    'I0': np.array([tsim, I0]).T,
    'T': np.array([tsim, Tavg]).T,
    'Rain': np.array([tsim, Rain]).T,
    'DVS': np.array([tsim, rice.f['DVS'] if not rice_active else np.zeros(tsim.size)]).T,
    'pd': np.array([tsim, np.full((tsim.size,), 0.501)]).T,
    'S_NH4': np.array([y1['t'], y1['P']]) if not fish_active else np.array([y1['t'], S_NH4]).T,
    'S_NO2': np.array([y2['t'], y2['P']]).T,
    'S_NO3': np.array([y3['t'], y3['P']]).T,
    'S_P': np.array([y4['t'], S_P]).T
    }

    #Run phyto model 
    yPhy = phyto.run(tspan, dphy)
    
    #retrieve Mphy as Fish inputs (disturbances)
    dfish = {'DO':np.array([tsim, np.random.randint(1,6, size=tsim.size)]).T,
         'T':np.array([tsim, Tavg]).T,
         'Mphy':np.array([yPhy['t'], yPhy['Mphy']]).T
         }
    
    # Run fish model
    if fish_active:
        yfish = fish.run(tspan, d=dfish, u=ufish)
        # Retrieve f_N_prt from fish and phytoplankton dead bodies as DB input
        f_N_prt = flows_fish['f_N_prt'] if fish_active else 0
        x0DB['S'] += f_N_prt + 0.5 * flows_phyto['f3']
    
        # Retrieve fish outputs and phyto outputs as PSB inputs
        f_P_prt = flows_fish['f_P_prt'] if fish_active else 0
        x0PSB['S'] += f_P_prt + 0.5 * flows_phyto['f3']
    
        # Retrieve DB model outputs for AOB model
        f_TAN = fish.f['f_TAN']
        f_TAN_accumulated += f_TAN
        # Reshape y1['P'] to match the shape of f_TAN_accumulated
        S_NH4 = y1['P'] + np.repeat(f_TAN_accumulated[-1], len(y1['P']))
        x0AOB['S'] = np.array([y1['t'], S_NH4])
    
        # Retrieve PSB outputs as phytoplankton and rice input (disturbances)
        f_P_sol = flows_fish['f_P_sol']
        f_P_sol_accumulated += f_P_sol
        last_f_P_sol = f_P_sol_accumulated[-1]
        S_P = f_P_sol_accumulated[:-1] + np.repeat(last_f_P_sol, len(y4['P']))
    
    #retrieve bacteria outputs as rice plant inputs
    drice = {'I0':np.array([tsim, I0]).T,
         'T': np.array([tsim, Tavg]).T,
         'CO2': np.array([tsim, np.full((tsim.size,), 400)]).T,
         'S_NH4': np.array([y1['t'], y1['P']]).T if ti > 20 else 0,
         'S_NO3': np.array([y3['t'], y3['P']]).T if ti > 20 else 0,
         'S_P': np.array([y4['t'], S_P]).T if ti > 20 else 0
         }
    
    #run rice model
    if rice_active==True:
        yrice = rice.run(tspan, drice, urice)
    
# Retrieve simulation results
# 1 kg/ha = 0.0001 kg/m2
# 1 kg/ha = 0.1 g/m2
t = yrice['t']

#fish
Mfish= fish.y['Mfish']/1000 #convert g DM to kg DM
Muri= fish.y['Muri']
Mfis_fr= Mfish/pfish['k_DMR']
Mdig= fish.y['Mdig']

#fish flows
f_fed = fish.f['f_fed']
f_N_prt = fish.f['f_N_prt']
f_P_prt = fish.f['f_P_prt']
f_TAN = fish.f['f_TAN']
f_P_sol = fish.f['f_P_sol']

#bacteria
S_N_prt = db.y['S']
S_P_prt = psb.y['S']
S_NH4 = db.y['P']
S_NO2 = aob.y['P']
S_NO3 = nob.y['P']
S_P = psb.y['P']
X_DB = db.y['X']
X_AOB = aob.y['X']
X_NOB = nob.y['X']
X_PSB = psb.y['X']

#phytoplankton
Mphy = phyto.y['Mphy']
NA = phyto.y['NA']

# phyto flows
f1 = phyto.f['f1']
f2 = phyto.f['f2']
f3 = phyto.f['f3']
f4 = phyto.f['f4']
f5 = phyto.f['f5']

#rice
Mrt = yrice['Mrt']
Mst = yrice['Mst']
Mlv = yrice['Mlv']
Mpa = yrice['Mpa']
Mgr = yrice['Mgr']

f_Ph= rice.f['f_Ph']
f_res= rice.f['f_res']
f_gr = rice.f['f_gr']
f_dmv = rice.f['f_dmv']
f_pN = rice.f['f_pN']
f_Nlv = rice.f['f_Nlv']
HU = rice.f['HU']
DVS = rice.f['DVS']


plt.figure(1)
plt.bar('Fish Fresh Weight', Mfis_fr)
plt.bar('Grain weight 2:1', Mgr*n_rice1)
plt.bar('Grain Weight 4:1', Mgr*n_rice2)
plt.legend()
plt.ylabel(r'$biomass\ [kg DM]$')

plt.figure(2)
plt.plot(t, Mrt, label='Roots')
plt.plot(t, Mst, label='Stems')
plt.plot(t, Mlv, label='Leaves')
plt.plot(t, Mpa, label='Panicles')
plt.plot(t, Mgr, label='Grains')
plt.xlabel(r'time [d]')
plt.ylabel(r'Dry mass [kg DM m-2 d-1]')
plt.title(r'Accumulative Dry Mass of Rice Crop Organs')
plt.legend()

# Create a figure and a grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Plot results
axs[0, 0].plot(t, S_N_prt, label='Organic Matter Substrate')
axs[0, 0].plot(t, X_DB, label='Decomposer Bacteria')
axs[0, 0].plot(t, S_NH4, label='Ammonium')
axs[0, 0].legend()
axs[0, 0].set_xlabel(r'$time\ [d]$')
axs[0, 0].set_ylabel(r'$concentration\ [mg L-1]$')
axs[0, 0].set_title("Decomposition")

axs[0, 1].plot(t, S_NH4, label='Ammonium')
axs[0, 1].plot(t, X_AOB, label='AOB')
axs[0, 1].plot(t, S_NO2, label='Nitrite')
axs[0, 1].legend()
axs[0, 1].set_xlabel(r'$time\ [d]$')
axs[0, 1].set_ylabel(r'$concentration\ [mg L-1]$')
axs[0, 1].set_title("Nitrite production rate")

axs[1, 0].plot(t, S_NH4, label='Nitrite')
axs[1, 0].plot(t, X_AOB, label='NOB')
axs[1, 0].plot(t, S_NO2, label='Nitrate')
axs[1, 0].legend()
axs[1, 0].set_xlabel(r'$time\ [d]$')
axs[1, 0].set_ylabel(r'$concentration\ [mg L-1]$')
axs[1, 0].set_title("Nitrate production rate")

axs[1, 1].plot(t, S_P_prt, label='Organic P')
axs[1, 1].plot(t, X_PSB, label='PSB')
axs[1, 1].plot(t, S_P, label='Phosphate')
axs[1, 1].legend()
axs[1, 1].set_xlabel(r'$time\ [d]$')
axs[1, 1].set_ylabel(r'$concentration\ [mg L-1]$')
axs[1, 1].set_title("Soluble Phosphorus production rate")

plt.show()