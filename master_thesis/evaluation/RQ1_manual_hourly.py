# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 10:11:48 2024

@author: Angela

Research Question 1: “Which combination of fish stocking density 
and rice planting density that can minimize the input artifical fertilizers?”

"""

from models.bacterialgrowth import Monod, DB, PSB, AOB
from models.fish import Fish
from models.phytoplankton import Phygrowth
from models.rice_hourly import Rice
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")

# Simulation time array
t = 120*24
tsim = np.linspace(0.0, t, t + 1)  # [d]
dt = 1

# disturbances
data_weather = 'C:/Users/alegn/Documents/WUR/Thesis/rfmodel/data/Weather/DIY_202210_202309_Hourly.csv'
weather = pd.read_csv(data_weather, header=3, sep=';')

t_ini = '20221001T00:00'
t_end = '20230129T00:00'
# t_end = '20230930T23:00'
t_weather = np.linspace(0, 120*24, 120*24+1)
weather['time'] = pd.to_datetime(weather['time'], format='ISO8601')  # Adjust the format if necessary
weather.set_index('time', inplace=True)
Tavg = weather.loc[t_ini:t_end,'temperature'].values #[°C] Mean daily temperature
Rain = weather.loc[t_ini:t_end,'rain'].values #[mm] Daily precipitation
Igl = weather.loc[t_ini:t_end, 'shortwave_radiation'].values #[W/m2] Hourly shortwave radiation

I0 = I0 = Igl*3600  # Convert [W m-2] to [J m-2 h-1] PAR

dt_bacteria = 1 # [d]
# pond area and volume
area = 10000  # [m2] the total area of the system
rice_area = 6000  # [m2] rice field area
pond_area = area - rice_area  # [m2] pond area

##Decomposer Bacteria
x0DB = {
    "S": 0.323404255,  # Particulate Matter concentration
    "X": 0.2,  # decomposer bacteria concentration
    "P": 0.08372093,
}  # concentration in [g m-3]

# Organic fertilizer used: chicken manure
# N content = 1.23% N/kg fertilizers
# Chicken manure weight = 1000 kg (~500 kg/ha/week) (KangOmbe et al., 2006)
# DB
Norgf = (1.23 / 100) * 1000  # N content in organic fertilizers
uDB = {"Norgf": Norgf}

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

# initialize object
db = DB(tsim, dt_bacteria, x0DB, pDB)

##Ammonification by AOB
x0AOB = {
    "S": 0.08372093,  # Ammonium concentration
    "X": 0.07,  # AOB concentration
    "P": 0.305571392,  # Nitrite concentration
}  # concentration in [g m-3]

pAOB = {
    "mu_max0": 1,  # range 0.33 - 1 g/g.d, if b = 0.17, mu_max_AOB = 0.9
    "Ks": 0.5,  # range 0.14 - 5 g/m3, 0.6 - 3.6 g/m3, or 0.3 - 0.7 g/m3
    "K_DO": 0.5,  # range 0.1 - 1 g/m3
    "Y": 0.33,  # range 0.10 - 0.15 OR 0.33
    "b20": 0.17,  # range 0.15 - 0.2 g/g.d
    "teta_mu": 1.072,
    "teta_b": 1.029,
    "MrS": 18.05,  # [g/mol] Molecular Weight of NH4
    "MrP": 46.01,  # [g/mol] Molecular Weight of NO2
    "a": 1,
}

dAOB = {
    "DO": np.array(
        [tsim, np.random.randint(1, 6, size=tsim.size)]
    ).T,  # [mg O2 L-1] random number between 1 - 5 to simulate dissolved oxygen in the pond
    "T": np.array(
        [tsim, Tavg]
    ).T,  # [°C] water temperature, assume similar with air temperature
    "f_TAN": np.array([tsim, np.zeros(tsim.size)]).T,
    "S_NH4_DB": np.array([tsim, np.zeros(tsim.size)]).T,
}


# initialize object
aob = AOB(tsim, dt_bacteria, x0AOB, pAOB)

# #Nitrification by NOB
x0NOB = {
    "S": 0.305571392,  # Nitrite concentration
    "X": 0.02,  # NOB concentration
    "P": 0.05326087,  # Nitrate concentration
}  # concentration in [g m-3]

pNOB = {
    "mu_max0": 1,  # range 0.7 - 1.8 g/g.d
    "Ks": 0.2,  # range 0.05 - 0.3 g/m3
    "K_DO": 0.5,  # range 0.1 - 1 g/m3
    "Y": 0.08,  # 0.04 - 0.07 g VSS/g NO2 or 0.08 g VSS/g NO2
    "b20": 0.17,
    "teta_b": 1.029,
    "teta_mu": 1.063,
    "MrS": 46.01,  # [g/mol] Molecular Weight of NO2
    "MrP": 62.01,  # [g/mol] Molecular Weight of NO3
    "a": 1,
}

r = 2  # r=[2, 2.5, 3]
pNOB["K_DO"] = r * pAOB["K_DO"]

dNOB = {
    "DO": np.array(
        [tsim, np.random.randint(1, 6, size=tsim.size)]
    ).T,  # [mg O2 L-1] random number between 1 - 5 to simulate dissolved oxygen in the pond
    "T": np.array(
        [tsim, Tavg]
    ).T,  # [°C] water temperature, assume similar with air temperature
    "S_out": np.array([tsim, np.zeros(tsim.size)]).T,
}

# #instantiate Model
nob = Monod(tsim, dt_bacteria, x0NOB, pNOB)

##Phosphorus Solubilizing by PSB
x0PSB = {
    "S": 0.05781939,  # particulate P concentration (Mei et al 2023)
    "X": 0.03,  # PSB concentration
    "P": 0.0327835051546391,  # total Soluble Reactive Phosphorus/total P available for uptake (Mei et al 2023)
}  # concentration in [g m-3]

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

#PSB
Porgf = 1.39 / 100 * 1000
uPSB = {"Porgf": Porgf}


pPSB = {
    "mu_max0": 3.8,  # maximum rate of substrate use (d-1)
    "Ks": 5.14,  # half velocity constant (g m-3)
    "K_DO": 1,  # half velocity constant of DO for the bacteria
    "Y": 0.45,  # bacteria synthesis yield (g bacteria/ g substrate used)
    "b20": 0.2,  # endogenous bacterial decay (d-1)
    "teta_b": 1.04,  # temperature correction factor for b
    "teta_mu": 1.07,  # temperature correction factor for mu
    "MrS": 647.94,  # [g/mol] Molecular Weight of C6H6O24P6
    "MrP": 98.00,  # [g/mol] Molecular Weight of H2PO4
    "a": 6,
    "kP": 0.5,
}

# initialize object
psb = PSB(tsim, dt_bacteria, x0PSB, pPSB)

##phytoplankton model
x0phy = {
    "Mphy": 8.586762e-6/0.02,  # [g m-3] from Mei et al 2023 concentration of Chla multiply with Chla:phyto mass ratio from Jamu and Piedrahita (2002)
    "NA": 0.55,  # [g m-3] TN from Mei et al 2023
    "f3": 0,
}


# Model parameters
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

# initialize object
phyto = Phygrowth(tsim, dt, x0phy, pphy)
flows_phyto = phyto.f

##fish model
# state variables
N_fish = [14.56, 18.2, 21.84]  # [g m-3]
pond_volume = pond_area * 0.5
n_fish = [(N * pond_volume) / 18.2 for N in N_fish]


x0fish = {
    "Mfish": 18.2 * n_fish[0],
    "Mdig": 1e-6 * n_fish[0],
    "Muri": 1e-6 * n_fish[0],
    "f_N_prt": 0,
    "f_P_prt": 0,
    "f_TAN": 0,
    "f_P_sol": 0,
}

ufish = {"Mfed": 18.2 * 0.03 * n_fish[0]}

# parameters
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
    "Tmin": 15,
    "Topt": 25,
    "Tmax": 35,
}

# initialize object
fish = Fish(tsim, dt, x0fish, pfish)
flows_fish = fish.f

# Rice model
n_rice1 = 127980  # number of plants using 2:1 pattern
n_rice2 = 153600  # number of plants using 4:1 pattern

x0rice = {
    "Mrt": 0.005,  # [kg DM ha-1 d-1] Dry biomass of root
    "Mst": 0.003,  # [kg DM ha-1 d-1] Dry biomass of stems
    "Mlv": 0.002,  # [kg DM ha-1 d-1] Dry biomass of leaves
    "Mpa": 0.0,  # [kg DM ha-1 d-1] Dry biomass of panicles
    "Mgr": 0.0,  # [kg DM ha-1 d-1] Dry biomass of grains
    "HU": 0.0,
    "DVS": 0,
}

# Model parameters
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

# inorganic fertilizer types and concentration:
# Nitrogen source: NPK (15%:15%:15%), Urea (46% N)
NPK_w = 167 * rice_area
I_N = (15 / 100) * NPK_w

# Phosphorus source: SP36 (36%P2O5 ~ 7.85% P)
# P content in SP-36: 7.85% P
SP36 = 31 * rice_area
I_P = (7.85 / 100) * SP36

# Initialize module
rice = Rice(tsim, dt, x0rice, price)
flows_rice = rice.f

# Iterator
# Initial disturbance
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

#initialize variables to store the values from iterator
y1_results = {
    't': [0,],
    "S": [0.323404255, ],  # Particulate Matter concentration
    "X": [0.2, ],  # decomposer bacteria concentration
    "P": [0.08372093, ],
    }

y2_results = {
    't': [0, ],
    "S": [0.08372093, ],  # Ammonium concentration
    "X": [0.07, ],  # AOB concentration
    "P": [0.305571392, ]  # Nitrite concentration
    }

y3_results = {
    't': [0, ],
    "S": [0.305571392, ],  # Nitrite concentration
    "X": [0.02, ],  # NOB concentration
    "P": [0.05326087, ]  # Nitrate concentration
    }

y4_results = {
    't': [0, ],
    "S": [0.05781939, ],  # particulate P concentration (Mei et al 2023)
    "X": [0.03, ],  # PSB concentration
    "P": [0.0327835051546391, ] # total Soluble Reactive Phosphorus/total P available for uptake (Mei et al 2023)
    }

yphy_results = {
    't': [0, ],
    "Mphy": [8.586762e-6/0.02, ],  # [g m-3] from Mei et al 2023 concentration of Chla multiply with Chla:phyto mass ratio from Jamu and Piedrahita (2002)
    "NA": [0.55, ]  # [g m-3] TN from Mei et al 2023
    }

yfish_results = {
    't': [0, ],
    "Mfish": [18.2 * n_fish[0], ],
    "Mdig": [1e-6 * n_fish[0], ],
    "Muri": [1e-6 * n_fish[0], ]
    }

yrice_results = {
    't': [0, ],
    "Mrt": [0.005, ],  # [kg DM ha-1 d-1] Dry biomass of root
    "Mst": [0.003, ],  # [kg DM ha-1 d-1] Dry biomass of stems
    "Mlv": [0.002, ], # [kg DM ha-1 d-1] Dry biomass of leaves
    "Mpa": [0.0, ], # [kg DM ha-1 d-1] Dry biomass of panicles
    "Mgr": [0.0, ],  # [kg DM ha-1 d-1] Dry biomass of grains
    "HU": [0.0, ]
    }

it = np.nditer(tsim[:-1], flags=["f_index"])
for ti in it:
    # Index for current time instant
    idx = it.index
    # Integration span
    tspan = (tsim[idx], tsim[idx + 1])
    print("Integrating", tspan)

    # Controlled inputs
    urice = {"I_N": I_N, "I_P": I_P}  # [kg m-2 d-1]

    # Run DB model
    y1 = db.run(tspan, dDB, uDB)
    y1_results['t'].append(y1['t'][1])
    y1_results['S'].append(y1['S'][1])
    y1_results['X'].append(y1['X'][1])
    y1_results['P'].append(y1['P'][1])

    # Retrieve DB model outputs for AOB model
    dAOB["S_NH4_DB"] = np.array([y1["t"], y1["P"]]).T

    # Run AOB model
    y2 = aob.run(tspan, dAOB)
    y2_results['t'].append(y2['t'][1])
    y2_results['S'].append(y2['S'][1])
    y2_results['X'].append(y2['X'][1])
    y2_results['P'].append(y2['P'][1])

    # Retrieve AOB model outputs for NOB model
    dNOB["S_out"] = np.array([y2["t"], y2["P"]]).T

    # #run NOB model
    y3 = nob.run(tspan, dNOB)
    y3_results['t'].append(y3['t'][1])
    y3_results['S'].append(y3['S'][1])
    y3_results['X'].append(y3['X'][1])
    y3_results['P'].append(y3['P'][1])

    # run PSB model
    y4 = psb.run(tspan, dPSB, uPSB)
    y4_results['t'].append(y4['t'][1])
    y4_results['S'].append(y4['S'][1])
    y4_results['X'].append(y4['X'][1])
    y4_results['P'].append(y4['P'][1])

    # retrieve output of DB, AOB, and NOB as phytoplankton inputs (disturbances)
    dphy["S_NH4"] = np.array([y1["t"], y1["P"]]).T
    dphy["S_NO2"] = np.array([y2["t"], y2["P"]]).T
    dphy["S_NO3"] = np.array([y3["t"], y3["P"]]).T
    dphy["S_P"] = np.array([y4["t"], y4["P"]]).T

    # Run phyto model
    yphy = phyto.run(tspan, dphy)
    yphy_results['t'].append(yphy['t'][1])
    yphy_results['Mphy'].append(yphy['Mphy'][1])
    yphy_results['NA'].append(yphy['NA'][1])

    # retrieve Mphy as Fish inputs (disturbances)
    dfish["Mphy"] = np.array([yphy["t"], yphy["Mphy"]]).T

    # Run fish model
    yfish = fish.run(tspan, d=dfish, u=ufish)
    yfish_results['t'].append(yfish['t'][1])
    yfish_results['Mfish'].append(yfish['Mfish'][1])
    yfish_results['Mdig'].append(yfish['Mdig'][1])
    yfish_results['Muri'].append(yfish['Muri'][1])

    # retrieve phyto ouput as DB and PSB inputs
    dDB["f3"] = np.array([yphy["t"], yphy["f3"]]).T
    dPSB["f3"] = np.array([yphy["t"], yphy["f3"]]).T

    # Retrieve f_N_prt from fish and phytoplankton dead bodies as DB input
    dDB["f_N_prt"] = np.array((yfish["t"], yfish["f_N_prt"])).T

    # Retrieve fish outputs and phyto outputs as PSB inputs
    dPSB["f_P_prt"] = np.array([yfish["t"], yfish["f_P_prt"]]).T

    # Retrieve DB model outputs for AOB model
    dAOB["f_TAN"] = np.array([yfish["t"], yfish["f_TAN"]]).T

    # retrieve bacteria outputs as rice plant inputs
    drice["S_NO3"] = np.array([y3["t"], y3["P"]]).T
    drice["S_P"] = np.array([y4["t"], y4["P"]]).T
    drice["f_P_sol"] = np.array([yfish["t"], yfish["f_P_sol"]]).T

    # run rice model
    yrice = rice.run(tspan, drice, urice)
    yrice_results['t'].append(yrice['t'][1])
    yrice_results['Mrt'].append(yrice['Mrt'][1])
    yrice_results['Mst'].append(yrice['Mst'][1])
    yrice_results['Mlv'].append(yrice['Mlv'][1])
    yrice_results['Mpa'].append(yrice['Mpa'][1])
    yrice_results['Mgr'].append(yrice['Mgr'][1])
    yrice_results['HU'].append(yrice['HU'][1])

    # retrieve rice output as phytoplankton inputs
    dphy["DVS"] = np.array([yrice["t"], yrice["DVS"]]).T

#change dictionary of lists to be dictionary of arrays
y1 = {key: np.array(value) for key, value in y1_results.items()}
y2 = {key: np.array(value) for key, value in y2_results.items()}
y3 = {key: np.array(value) for key, value in y3_results.items()}
y4 = {key: np.array(value) for key, value in y4_results.items()}
yphy = {key: np.array(value) for key, value in yphy_results.items()}
yfish = {key: np.array(value) for key, value in yfish_results.items()}
yrice = {key: np.array(value) for key, value in yrice_results.items()}

# Retrieve simulation results
# 1 kg/ha = 0.0001 kg/m2
# 1 kg/ha = 0.1 g/m2

# fish flows
f_fed = fish.f["f_fed"]
f_N_prt = fish.f["f_N_prt"]
f_P_prt = fish.f["f_P_prt"]
f_TAN = fish.f["f_TAN"]
f_P_sol = fish.f["f_P_sol"]

# fish
Mfish = flows_fish['f_upt']  # convert g DM to kg DM
Mfis_fr = (Mfish / pfish["k_DMR"])*0.001

# bacteria
S_N_prt = y1["S"]
S_P_prt = y4["S"]
S_NH4 = y1["P"]
S_NO2 = y2["P"]
S_NO3 = y3["P"]
S_P = y4["P"]
X_DB = y1["X"]
X_AOB = y2["X"]
X_NOB = y3["X"]
X_PSB = y4["X"]

# phytoplankton
Mphy = yphy["Mphy"]
NA = yphy["NA"]

# phyto flows
f1 = phyto.f["f1"]
f2 = phyto.f["f2"]
f3 = phyto.f["f3"]
f4 = phyto.f["f4"]
f5 = phyto.f["f5"]

#rice flows
f_Ph = rice.f["f_Ph"]
f_res = rice.f["f_res"]
f_gr = rice.f["f_gr"]
f_dmv = rice.f["f_dmv"]
f_pN = rice.f["f_pN"]
f_Nlv = rice.f["f_Nlv"]
HU = rice.f["HU"]
DVS = rice.f["DVS"]

# rice
Mrt = yrice["Mrt"]
Mst = yrice["Mst"]
Mlv = yrice["Mlv"]
Mpa = yrice["Mpa"]
Mgr = yrice['Mgr'] #[kg/m2]

# Visualization
# Plot results
plt.figure(1)
plt.plot(yfish['t'], Mfis_fr, label='Fish fresh weight')
plt.plot(yrice['t'], Mgr, label = 'P1')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$biomass\ [g day-1]$')

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
plt.figure(4)
plt.plot(phyto.t, Mphy, label='Phytoplankton')
plt.plot(phyto.t, NA, label = 'Nutrient Availability')
plt.legend()
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$biomass\ [g m-3 day-1]$')

plt.figure(5)
plt.plot(yrice['t'], Mrt, label='Roots')
plt.plot(yrice['t'], Mst, label='Stems')
plt.plot(yrice['t'], Mlv, label='Leaves')
plt.plot(yrice['t'], Mpa, label='Panicles')
plt.xlabel(r'time [d]')
plt.ylabel(r'Dry mass [kg DM ha-1 d-1]')
plt.title(r'Accumulative Dry Mass of Rice Crop Organs')
plt.legend()

plt.figure(6)
plt.plot(yrice['t'], f_Nlv, label='N flow rate in leaves')
plt.plot(yrice['t'], f_pN, label = 'N flow rate in rice plants')
plt.plot(yrice['t'], S_NO3, label = 'nitrate production rate')
plt.xlabel(r'time [d]')
plt.ylabel(r'flow rate [kg N ha-1 d-1]')
plt.title(r'Flow rate in rice plants')
plt.legend()

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()