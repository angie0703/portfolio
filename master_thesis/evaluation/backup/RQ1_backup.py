# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 10:11:48 2024

@author: alegn
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
t = 365
tsim = np.linspace(0.0, t, t+1) # [d]
dt = 1
tspan = (tsim[0], tsim[-1])

#disturbances
data_weather = 'C:/Users/alegn/Documents/WUR/Thesis/rfmodel/data/Weather/DIY_202210_202309_Daily.csv'
weather = pd.read_csv(data_weather, header=1, sep=';')
Tmax = weather.iloc[:,2].values #[°C] Maximum daily temperature
Tmin = weather.iloc[:,3].values #[°C] Minimum daily temperature
Tavg = weather.iloc[:,4].values #[°C] Mean daily temperature
Rain = weather.iloc[:,5].values #[mm] Daily precipitation
Igl   = weather.iloc[:,6].values #[MJ m-2] Sum of shortwave radiation daily

I0 = 0.45*Igl*1E6 #Convert [MJ m-2 d-1] to [J m-2 d-1] PAR

d_air = {
      'I0' : np.array([tsim, I0]).T, #[J m-2 d-1] Sum of Global Solar Irradiation (6 AM to 6 PM)
      'CO2': np.array([tsim, np.full((tsim.size,), 400)]).T, #[ppm] CO2 concentration, assume all 400 ppm
      'Tavg':np.array([tsim, Tavg]).T,
      'Rain': np.array([tsim, Rain]).T
      }

d_pond = {
    # 'DO': np.array([tsim, np.full((tsim.size,), 5)]).T,
    'DO': np.array([tsim, np.random.randint(1,6, size=tsim.size)]).T, # [mg O2 L-1] random number between 1 - 5 to simulate dissolved oxygen in the pond 
    'T': d_air['Tavg'] #[°C] water temperature, assume similar with air temperature
    }

#pond area and volume
area = 10000 #[m2] the total area of the system
rice_area = 6000 #[m2] rice field area
pond_area = area - rice_area #[m2] pond area

DOvalues = 5
Tvalues = 26

##Decomposer Bacteria
x0DB = {'S':0.323404255,#Particulate Matter concentration 
        'X':0.27, #decomposer bacteria concentration
        'P': 0.08372093      
        }           #concentration in [g m-3]

#Organic fertilizer used: chicken manure
#N content = 1.23% N/kg fertilizers
#Chicken manure weight = 1000 kg (~500 kg/ha/week) (KangOmbe et al., 2006)
Norgf = (1.23/100)*1000 #N content in organic fertilizers 
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
db = Monod(tsim, dt, x0DB, pDB)

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
aob = Monod(tsim, dt, x0AOB, pAOB)

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
nob = Monod(tsim, dt, x0NOB, pNOB)

##Phosphorus Solubilizing by PSB 
x0PSB = {'S':0.05781939,   #particulate P concentration (Mei et al 2023)
         'X':0.5,   #PSB concentration
         'P': 0.0327835051546391 #total Soluble Reactive Phosphorus/total P available for uptake (Mei et al 2023)
         }          #concentration in [g m-3]

#P content in organic fertilizers
Porgf = (1.39/100)*1000

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
psb = Monod(tsim, dt, x0PSB, pPSB)


##phytoplankton model
x0phy = {'Mphy': 8.586762e-6/0.02, #[g m-3] from Mei et al 2023 concentration of Chla multiply with Chla:phyto mass ratio from Jamu and Piedrahita (2002)
      'NA':   0.55 #[g m-3] TN from Mei et al 2023
      } 


# Model parameters, as provided by Mohtar et al. (1997)
# TODO: define the parameter values in the dictionary p
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
N_fish = [4000, 8000, 12000]

x0fish = {'Mfish': 80*N_fish[0], #[gDM] 
      'Mdig':1E-6*N_fish[0], #[gDM]
      'Muri':1E-6*N_fish[0] #[gDM]
      }
 
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

#controllable input
ufish = {'Mfed': 80*0.03*N_fish[0]}

#initialize object
fish = Fish(tsim, dt, x0fish, pfish)
flows_fish = fish.f

# Rice model
# TODO: define sensible values for the initial conditions
n_rice1 = 127980 #number of plant using 2:1 pattern
n_rice2 = 153600 #number of plants using 4:1 pattern 

x0rice = {'Mrt':0.001,    #[kg DM ha-1 d-1] Dry biomass of root
      'Mst':0.001,    #[kg DM ha-1 d-1] Dry biomass of stems
      'Mlv': 0.001,   #[kg DM ha-1 d-1] Dry biomass of leaves
      'Mpa': 0.0,   #[kg DM ha-1 d-1] Dry biomass of panicles
      'Mgr': 0.0,   #[kg DM ha-1 d-1] Dry biomass of grains
      'HU': 0.0
      } 

# Model parameters, as provided by Mohtar et al. (1997)
# TODO: define the parameter values in the dictionary p
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

# Controlled inputs
#inorganic fertilizer types and concentration used in Indonesia: 
#Nitrogen source: NPK (15%:15%:15%), Urea (46% N)
NPK_w = 167*rice_area
I_N = (15/100)*NPK_w

#Phosphorus source: SP36 (36%P2O5 ~ 7.85% P)
#P content in SP-36: 7.85% P
Porgf = (7.85/100)


urice = {'I_N':I_N}            # [kg m-2 d-1]

# Initialize module
# TODO: Call the module Grass to initialize an instance
rice = Rice(tsim, dt, x0rice, price)

# Run DB model
#retrieve f_N_prt from fish as DB input
f_N_prt = flows_fish['f_N_prt']
y1 = db.run(tspan,d_pond)

# Retrieve DB model outputs for AOB model
DB_NH4 =  y1['P']

f_TAN = flows_fish['f_TAN']
x0AOB['S'] = np.array([y1['t'], y1['P']])

# Run AOB model    
y2 = aob.run(tspan,d_pond)

# Retrieve AOB model outputs for NOB model
x0NOB['S']= np.array([y2['t'], y2['P']])

# #run NOB model
y3 = nob.run(tspan,d_pond)

# #run PSB model
y4 = psb.run(tspan,d_pond)

#retrieve Phytoplankton death rate as DB inputs
x0DB['S'] = x0DB['S']+phyto.f['f3'] 

#retrieve output of DB and NOB as rice inputs
f_uptN = nob.y['P']

#retrieve DB, AOB, AOB, and PSB outputs as phytoplankton inputs
x0phy['NA']= y1['P']+y2['P']+y3['P']+y4['P']

dphy = {
     'I0':d_air['I0'],
     'T':d_air['Tavg'],
     'Rain': d_air['Rain'],
     'DVS':np.array([tsim, rice.f['DVS']]).T,
     'pd':np.array([tsim, np.full((tsim.size,),0.501)]).T,
     'S_NH4': np.array([tsim, y1['P']]).T,
     'S_NO2': np.array([tsim, y2['P']]).T,
     'S_NO3': np.array([tsim, y3['P']]).T
     }

#Run phyto model 
yPhy = phyto.run(tspan, dphy)

dfish = {'DO':d_pond['DO'],
     'T':d_pond['T'],
     'Mphy':np.array([tsim, yPhy['Mphy']]).T
     }


#run fish model
yfish = fish.run(tspan, d=dfish, u=ufish)

drice = {'I0':d_air['I0'],
     'T': d_air['Tavg'],
     'CO2': np.array([tsim, np.full((tsim.size,), 400)]).T,
     'S_NH4': np.array([tsim, y1['P']]).T,
     'S_NO3': np.array([tsim, y3['P']]).T
     }

#run fish model
yrice = rice.run(tspan, drice, urice)

# Iterator
# (stop at second-to-last element, and store index in Fortran order)
it = np.nditer(tsim[:-1], flags=['f_index'])
for ti in it:
    # Index for current time instant
    idx = it.index
    # Integration span
    tspan = (tsim[idx], tsim[idx+1])
    print('Integrating', tspan)
    # Controlled inputs
    u_grs = {'f_Gr':0, 'f_Hr':0}   # [kgDM m-2 d-1]
    u_wtr = {'f_Irg':0}            # [mm d-1]
    # Run grass model
    y_grs = grass.run(tspan, d_grs, u_grs)
    # Retrieve grass model outputs for water model
    d_wtr['LAI'] = np.array([y_grs['t'], y_grs['LAI']])
    # Run water model    
    y_wtr = water.run(tspan, d_wtr, u_wtr)
    # Retrieve water model outputs for grass model
    d_grs['WAI'] = np.array([y_wtr['t'], y_wtr['WAI']])

# Run simulation
# TODO: Call the method run to generate simulation results
tspan = (tsim[0], tsim[-1])

f_Ph= rice.f['f_Ph']
f_res= rice.f['f_res']
f_gr = rice.f['f_gr']
f_dmv = rice.f['f_dmv']
f_pN = rice.f['f_pN']
f_Nlv = rice.f['f_Nlv']
HU = rice.f['HU']
DVS = rice.f['DVS']

# Retrieve simulation results
# 1 kg/ha = 0.0001 kg/m2
# 1 kg/ha = 0.1 g/m2
# TODO: Retrieve the simulation results
t = yrice['t']

# Retrieve simulation results
t = db.t

#fish
Mfish= fish.y['Mfish'] #convert g DM to kg DM
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