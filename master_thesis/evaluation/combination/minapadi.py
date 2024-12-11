# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 10:11:48 2024

@author: alegn
"""

from models.bacterialgrowth import Monod, DB, AOB, PSB
from models.fish import Fish
from models.phytoplankton import Phygrowth
from models.rice import Rice
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')

# Simulation time array
t = 120
tsim = np.linspace(0.0, t, t+1) # [d]
dt = 1/10
tspan = (tsim[0], tsim[-1])

# pond area and volume
area = 10000  # [m2] the total area of the system
rice_area = 6000  # [m2] rice field area
pond_area = area - rice_area  # [m2] pond area

# disturbances
data_weather = "C:/Users/alegn/Documents/WUR/Thesis/rfmodel/data/Weather/DIY_202210_202309_Daily.csv"
weather = pd.read_csv(data_weather, header=1, sep=";")

t_ini = "20221001"
t_end = "20230129"
weather["Time"] = pd.to_datetime(
    weather["Time"], format="%Y%m%d"
)  # Adjust the format if necessary
weather.set_index("Time", inplace=True)
Tavg = weather.loc[t_ini:t_end, "Tavg"].values  # [°C] Mean daily temperature
Rain = weather.loc[t_ini:t_end, "Rain"].values  # [mm] Daily precipitation
Igl = weather.loc[t_ini:t_end, "I0"].values  # [MJ m-2] Sum of shortwave radiation daily

I0 = 0.45 * Igl * 1e6  # Convert [MJ m-2 d-1] to [J m-2 d-1] PAR

##fish model
#state variables
n_fish = 4000
x0fish = {
    "Mfish": 18.2 * n_fish,
    "Mdig": 1e-6 * n_fish,
    "Muri": 1e-6 * n_fish,
    "f_N_prt": 0,
    "f_P_prt": 0,
    "f_TAN": 0,
    "f_P_sol": 0,
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
    'Chla':0.02, #[g Chla/g phytoplankton] Phytoplankton biomass density 
    'Ksp':1,  #[mg C L-1] Half-saturation constant for phytoplankton
    'r': 0.02,
    'Tmin': 15,
    'Topt': 25,
    'Tmax': 35,
    'n_fish': 60
    }

##Decomposer Bacteria
x0DB = {
    "S": 0.323404255,  # Particulate Matter concentration
    "X": 0.2,  # decomposer bacteria concentration
    "P": 0.08372093,
}  # concentration in [g m-3]

pDB = {
    "mu_max0": 3.8,  # maximum rate of substrate use (d-1) Kayombo 2003
    "Ks": 5.14,  # half velocity constant (g m-3)
    "K_DO": 1,  # half velocity constant of DO for the bacteria
    "Y": 0.82,  # bacteria synthesis yield (g bacteria/ g substrate used)
    "b20": 0.12,  # endogenous bacterial decay (d-1)
    "teta_b": 1.04,  # temperature correction factor for b
    "teta_mu": 1.07,  # temperature correction factor for mu
    "MrS": 60.07,  # [g/mol] Molecular Weight of urea
    "MrP": 18.05,  # [g/mol] Molecular Weight of NH4
    "a": 2,
    "kN": 0.5,
}

##Ammonification by AOB 
x0AOB = {
    "S": 0.08372093,  # Ammonium concentration
    "X": 0.07,  # AOB concentration
    "P": 0.305571392,  # Nitrite concentration
}  # concentration in [g m-3]

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

# #Nitrification by NOB
x0NOB = {
    "S": 0.305571392,  # Nitrite concentration
    "X": 0.2,  # NOB concentration
    "P": 0.05326087,  # Nitrate concentration
}  # concentration in [g m-3]

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

##Phosphorus Solubilizing by PSB 
x0PSB = {
    "S": 0.05781939,  # particulate P concentration (Mei et al 2023)
    "X": 0.03,  # PSB concentration
    "P": 0.0327835051546391,  # total Soluble Reactive Phosphorus/total P available for uptake (Mei et al 2023)
}  # concentration in [g m-3]

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
    'a': 6,
    'kP': 0.5
    } 


##phytoplankton model
x0phy = {
    "Mphy": 8.586762e-6/0.02,  # [g m-3] from Mei et al 2023 concentration of Chla multiply with Chla:phyto mass ratio from Jamu and Piedrahita (2002)
    "NA": 0.55,  # [g m-3] TN from Mei et al 2023
    "f3": 0,
}


# Model parameters, as provided by Mohtar et al. (1997)
# TODO: define the parameter values in the dictionary p
pphy = {
      'mu_phy':  1.27, #[d-1] maximum growth rate of phytoplankton
      'Iws':  5.42e-7, #[ ] light intensity at the water surface
      'mu_Up': 0.005, #[d-1] maximum nutrient uptake coefficient
      'pd': 0.5, #[m] pond depth
      'l_sl': 0.00035, #phytoplankton biomass-specific light attenuation
      'l_bg': 0.77, #light attenuation by non-phytoplankton components
      'Kpp': 3.62e-7, #half-saturation constant of phytoplankton production
      'cm': 0.15, #[-] phytoplankton mortality constant
      'cl': 0.004, #[-] phytoplankton crowding loss constant
      'c1': 1.57, # [-] temperature coefficients
      'c2': 0.24, # [-] temperature coefficients
      'Topt': 28, #optimum temperature for phytoplankton growth
      'Mp': 0.0025, #half saturation constant for nutrient uptake
      'a': 0.1, #dilution rate
     }



# Rice model
# TODO: define sensible values for the initial conditions
n_rice1 = 127980 #number of plant using 2:1 pattern
n_rice2 = 153600 #number of plants using 4:1 pattern 

x0rice = {'Mrt':0.001,    #[kg DM ha-1 d-1] Dry biomass of root
      'Mst':0.001,    #[kg DM ha-1 d-1] Dry biomass of stems
      'Mlv': 0.001,   #[kg DM ha-1 d-1] Dry biomass of leaves
      'Mpa': 0.0,   #[kg DM ha-1 d-1] Dry biomass of panicles
      'Mgr': 0.0,   #[kg DM ha-1 d-1] Dry biomass of grains
      'HU': 0.0,
      'DVS': 0
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

# Disturbances
# PAR [J m-2 d-1], env. temperature [°C], and CO2
# TODO: Specify the corresponding dates to read weather data (see csv file).
# Dictionary of disturbances (2D arrays, with col 1 for time, and col 2 for d)
dDB = {
    "DO": np.array(
        [tsim, np.random.randint(1, 6, size=tsim.size)]
    ).T,  # [mg O2 L-1] random number between 1 - 5 to simulate dissolved oxygen in the pond
    "T": np.array(
        [tsim, Tavg]
    ).T,  # [°C] water temperature, assume similar with air temperature
    "f_N_prt": np.array([tsim, np.zeros(tsim.size)]).T,
    "f3": np.array([tsim, np.zeros(tsim.size)]).T,
}

dAOB = {
    "DO": np.array(
        [tsim, np.random.randint(1, 6, size=tsim.size)]
    ).T,  # [mg O2 L-1] random number between 1 - 5 to simulate dissolved oxygen in the pond
    "T": np.array(
        [tsim, Tavg]
    ).T,  # [°C] water temperature, assume similar with air temperature
    "f_TAN": np.array([tsim, np.zeros(tsim.size)]).T,
}

dNOB = {
    "DO": np.array(
        [tsim, np.random.randint(1, 6, size=tsim.size)]
    ).T,  # [mg O2 L-1] random number between 1 - 5 to simulate dissolved oxygen in the pond
    "T": np.array(
        [tsim, Tavg]
    ).T,  # [°C] water temperature, assume similar with air temperature
    "S_out": np.array([tsim, np.zeros(tsim.size)]).T,
}

dPSB = {
    "DO": np.array(
        [tsim, np.random.randint(1, 6, size=tsim.size)]
    ).T,  # [mg O2 L-1] random number between 1 - 5 to simulate dissolved oxygen in the pond
    "T": np.array(
        [tsim, Tavg]
    ).T,  # [°C] water temperature, assume similar with air temperature
    "f_P_prt": np.array([tsim, np.zeros(tsim.size)]).T,
    "f3": np.array([tsim, np.zeros(tsim.size)]).T,
}

dphy = {
    "I0": np.array([tsim, I0]).T,
    "T": np.array([tsim, Tavg]).T,
    "Rain": np.array([tsim, Rain]).T,
    "DVS": np.array([tsim, np.zeros(tsim.size)]).T,
    "pd": np.array([tsim, np.full((tsim.size,), 0.501)]).T,
    "S_NH4": np.array([tsim, np.full((tsim.size,), 0.08372093)]).T,
    "S_NO2": np.array([tsim, np.full((tsim.size,), 0.305571392)]).T,
    "S_NO3": np.array([tsim, np.full((tsim.size,), 0.05326087)]).T,
    "S_P": np.array([tsim, np.full((tsim.size,), 0.0327835051546391)]).T,
}

dfish = {
    "DO": np.array([tsim, np.random.randint(1, 6, size=tsim.size)]).T,
    "T": np.array([tsim, Tavg]).T,
    "Mphy": np.array([tsim, np.full((tsim.size,), 8.586762e-6 / 0.02)]).T,
}

drice = {
    "I0": np.array([tsim, I0]).T,
    "T": np.array([tsim, Tavg]).T,
    "CO2": np.array([tsim, np.full((tsim.size,), 400)]).T,
    "S_NO3": np.array([tsim, np.full((tsim.size,), 0.05326087)]).T,
    "S_P": np.array([tsim, np.full((tsim.size,), 0.0327835051546391)]).T,
    "f_P_sol": np.array([tsim, np.full((tsim.size,), 0.0327835051546391)]).T,
}

# Controlled inputs
# Chicken manure weight = 1000 kg (~500 kg/ha/week) (KangOmbe et al., 2006)
# DB
Norgf = (1.23 / 100) * 1000  # N content in organic fertilizers
uDB = {"Norgf": Norgf}

#PSB
Porgf = 1.39 / 100 * 1000
uPSB = {"Porgf": Porgf}

#controllable input
ufish = {'Mfed': 80*0.03*n_fish}

#RICE
# Nitrogen source: NPK (15%:15%:15%), Urea (46% N)
NPK_w = 167 * rice_area
I_N = (15 / 100) * NPK_w

# Phosphorus source: SP36 (36%P2O5 ~ 7.85% P)
# P content in SP-36: 7.85% P
SP36 = 31 * rice_area
I_P = (7.85 / 100) * SP36
urice = {'I_N':60, 'I_P':0}            # [kgDM m-2 d-1]

# Initialize object
# TODO: Call the modules to initialize an instance
#initialize object
db = DB(tsim, dt, x0DB, pDB)
aob = AOB(tsim, dt, x0AOB, pAOB)
nob = Monod(tsim, dt, x0NOB, pNOB)
psb = PSB(tsim, dt, x0PSB, pPSB)
fish = Fish(tsim, dt, x0fish, pfish)
phyto = Phygrowth(tsim, dt, x0phy, pphy)
rice = Rice(tsim, dt, x0rice, price)

#initialize flows
flows_fish = fish.f
flows_phyto = phyto.f
flows_rice = rice.f

# Run simulation
# TODO: Call the method run to generate simulation results

# Run DB model
y1 = db.run(tspan,dDB, uDB)

# Retrieve DB model outputs for AOB model
dAOB["S_NH4_DB"] = np.array([y1["t"], y1["P"]]).T
  
# Run AOB model    
y2 = aob.run(tspan,dAOB)

# Retrieve AOB model outputs for NOB model
dNOB["S_out"] = np.array([y2["t"], y2["P"]]).T

# #run NOB model
y3 = nob.run(tspan,dNOB)

# #run PSB model
y4 = psb.run(tspan,dPSB, uPSB)

#run #retrieve DB, AOB, AOB, and PSB outputs as phytoplankton inputs
# retrieve output of DB, AOB, and NOB as phytoplankton inputs (disturbances)
dphy["S_NH4"] = np.array([y1["t"], y1["P"]]).T
dphy["S_NO2"] = np.array([y2["t"], y2["P"]]).T
dphy["S_NO3"] = np.array([y3["t"], y3["P"]]).T
dphy["S_P"] = np.array([y4["t"], y4["P"]]).T

#Run phyto model 
yPhy = phyto.run(tspan, dphy)

#retrieve Phytoplankton death rate as DB inputs
dDB["f3"] = np.array([yPhy["t"], yPhy["f3"]]).T
dPSB["f3"] = np.array([yPhy["t"], yPhy["f3"]]).T

#fish model
yfish = fish.run(tspan, d=dfish, u=ufish)

# Retrieve f_N_prt from fish as DB and AOB input
dDB["f_N_prt"] = np.array((yfish["t"], yfish["f_N_prt"])).T
dAOB["f_TAN"] = np.array([yfish["t"], yfish["f_TAN"]]).T

# retrieve bacteria outputs as rice plant inputs
drice["S_NO3"] = np.array([y3["t"], y3["P"]]).T
drice["S_P"] = np.array([y4["t"], y4["P"]]).T
drice["f_P_sol"] = np.array([yfish["t"], yfish["f_P_sol"]]).T

#rice model
yrice = rice.run(tspan, drice, urice)

# retrieve rice output as phytoplankton inputs
dphy["DVS"] = np.array([yrice["t"], yrice["DVS"]]).T

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
plt.bar('Grain weight', Mgr*127980)
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
axs[0, 0].plot(db.t, S_N_prt, label='Organic Matter Substrate')
axs[0, 0].plot(db.t, X_DB, label='Decomposer Bacteria')
axs[0, 0].plot(db.t, S_NH4, label='Ammonium')
axs[0, 0].legend()
axs[0, 0].set_xlabel(r'$time\ [d]$')
axs[0, 0].set_ylabel(r'$concentration\ [mg L-1]$')
axs[0, 0].set_title("Decomposition")

axs[0, 1].plot(aob.t, S_NH4, label='Ammonium')
axs[0, 1].plot(aob.t, X_AOB, label='AOB')
axs[0, 1].plot(aob.t, S_NO2, label='Nitrite')
axs[0, 1].legend()
axs[0, 1].set_xlabel(r'$time\ [d]$')
axs[0, 1].set_ylabel(r'$concentration\ [mg L-1]$')
axs[0, 1].set_title("Nitrite production rate")

axs[1, 0].plot(nob.t, S_NO2, label='Nitrite')
axs[1, 0].plot(nob.t, X_NOB, label='NOB')
axs[1, 0].plot(nob.t, S_NO3, label='Nitrate')
axs[1, 0].legend()
axs[1, 0].set_xlabel(r'$time\ [d]$')
axs[1, 0].set_ylabel(r'$concentration\ [mg L-1]$')
axs[1, 0].set_title("Nitrate production rate")

axs[1, 1].plot(psb.t, S_P_prt, label='Organic P')
axs[1, 1].plot(psb.t, X_PSB, label='PSB')
axs[1, 1].plot(psb.t, S_P, label='Phosphate')
axs[1, 1].legend()
axs[1, 1].set_xlabel(r'$time\ [d]$')
axs[1, 1].set_ylabel(r'$concentration\ [mg L-1]$')
axs[1, 1].set_title("Soluble Phosphorus production rate")

#plot
plt.figure(5)
plt.plot(rice.t, Mrt, label='Roots')
plt.plot(rice.t, Mst, label='Stems')
plt.plot(rice.t, Mlv, label='Leaves')
plt.plot(rice.t, Mpa, label='Panicles')
plt.xlabel(r'time [d]')
plt.ylabel(r'Dry mass [kg DM ha-1 d-1]')
plt.title(r'Accumulative Dry Mass of Rice Crop Organs')
plt.legend()

plt.figure(4)
plt.plot(phyto.t, Mphy, label='Phytoplankton')
plt.plot(phyto.t, NA, label = 'Nutrient Availability')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$biomass\ [g m-3 day-1]$')

plt.figure(6)
plt.plot(t, f_Nlv, label='N flow rate in leaves')
plt.plot(t, f_pN, label = 'N flow rate in rice plants')
plt.xlabel(r'time [d]')
plt.ylabel(r'flow rate [kg N ha-1 d-1]')
plt.title(r'Flow rate in rice plants')
plt.legend()


plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()