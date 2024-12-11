# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 19:46:50 2024

@author: alegn
"""

import numpy as np
from models.bacterialgrowth import NIB, PSB, DB, AOB, Monod
from models.fish import Fish
from models.phytoplankton import Phygrowth
from models.rice import Rice
import pandas as pd
from copy import deepcopy

class ModelScheduler:
    def __init__(self, model, start_time, end_time=None):
        self.model = model
        self.start_time = start_time
        self.end_time = end_time

    def run(self, current_time, inputs):
        if current_time >= self.start_time and (self.end_time is None or current_time <= self.end_time):
            return self.model.run(inputs)
        else:
            return self.model.idle_state()  # Return an idle state with no contribution

# # Use:
# fish_scheduler = ModelScheduler(self.fish, start_time=24, end_time=80)

class fishpond():
    """ Module for all process in the fishpond, consists of
    - Bacteria (NIB, PSB)
    - Phytoplankton
    - Fish
    
    The output would be the N and P available for the rice plants
    The external disturbances involve 
    - light intensity
    - water temperature
    - Dissolved Oxygen level
    
    The controllable inputs:
    - N and P content of organic fertilizers
    - Artificial fish feed
    
    Flow rates inside the system:
           
    Parameters
    ----------

        
    Returns
    -------
    y : dictionary
        Model outputs as 1D arrays ('S', 'X'),
        and the evaluation time 't'.
    """
    # Initialize object
    def __init__(self,tsim,dt,x0,p):
        try:    
            self.nib = NIB(tsim, dt, x0['NIB'], p['NIB'])   
            self.psb = PSB(tsim, dt, x0['PSB'], p['PSB'])
            self.phy = Phygrowth(tsim, dt, x0['phy'], p['phy'])
            self.fish = Fish(tsim, dt, x0['fish'], p['fish'])
            
            self.tsim = tsim
            self.dt = dt
        except KeyError as e:
            print(f"Error initializing FishPond: missing configuration for {e}")

    # Define simulations with inputs and disturbances
    def run_simulation(self, tspan, d, u):
        #Disturbances and inputs
        local_d = {key: dict(val) for key, val in d.items()}
        local_u = {key: dict(val) for key, val in u.items()}
        
        dnib = local_d['NIB']
        dpsb = local_d['PSB']
        dphy = local_d['phy']
        dfish = local_d['fish']
        unib = local_u['NIB']
        upsb = local_u['PSB']
        ufish = local_u['fish']
        
        #Run the model
        y1 = self.nib.run(tspan, dnib, unib)
        y2 = self.psb.run(tspan, dpsb, upsb)
        
        #make simulation result of y1 and y2 as Phygrowth state variable of x0['phy']['NA']
        dphy['SNH4'] = np.array([y1["t"], y1["SNH4"]]).T
        dphy['SNO2'] = np.array([y1["t"], y1["SNO2"]]).T
        dphy['SNO3'] = np.array([y1["t"], y1["SNO3"]]).T 
        dphy['SP'] = np.array([y2["t"], y2["P"]]).T 
        
        #Run phyto model
        yphy = self.phy.run(tspan, dphy)
        
        #retrieve simulation result of Phygrowth
        f3 = self.phy.f['f3']
        print('f3 shape: ', f3.shape)
        #make simulation result of yphy as input for fish
        dfish['Mphy'] = np.array([yphy["t"], yphy["Mphy"]]).T 
        
        #run fish model
        yfish = self.fish.run(tspan, dfish, ufish)
        print('Fish:', yfish['Mfish'])
        #retrieve simulation result of fish
        f_N_prt = self.fish.f['f_N_prt']
        f_P_prt = self.fish.f['f_P_prt']
        f_TAN = self.fish.f['f_TAN']
        f_P_sol = self.fish.f['f_P_sol']
        
        #make simulation result of Phygrowth and fish as disturbances of NIB and PSB
        dnib['f3'] = np.array([yphy["t"], f3]).T 
        dnib['f_N_prt'] = np.array([yfish["t"], f_N_prt]).T
        dnib['f_TAN'] = np.array([yfish["t"], f_TAN]).T
        dpsb['f3'] = np.array([yphy["t"], f3]).T
        dpsb['f_P_prt'] = np.array([yfish["t"], f_P_prt]).T
        dpsb['f_P_sol'] = np.array([yfish["t"], f_P_sol]).T
        
        #Simulation result
        
        return {
                'NIB': y1,
                'PSB': y2,
                'Phyto': yphy,
                'Fish': yfish,
                'SNO3': y1['SNO3'],
                'SP': y2['P'],
                'Yield': yfish['Mfish'],
                'f_P_sol': f_P_sol
                }
    
class Fishpond():
    """ Module for all process in the fishpond, consists of
    - Bacteria (BD, AOB, NOB (Monod), PSB)
    - Phytoplankton
    - Fish
    
    The output would be the N and P available for the rice plants
    The external disturbances involve 
    - light intensity
    - water temperature
    - Dissolved Oxygen level
    
    The controllable inputs:
    - N and P content of organic fertilizers
    - Artificial fish feed
    
    Flow rates inside the system:
           
    Parameters
    ----------

        
    Returns
    -------
    y : dictionary
        Model outputs as 1D arrays ('S', 'X'),
        and the evaluation time 't'.
    """
    # Initialize object
    def __init__(self,tsim,dt,x0,p):
        try:    
            self.db = DB(tsim, dt, x0['DB'], p['DB'])
            self.aob = AOB(tsim, dt, x0['AOB'], p['AOB'])
            self.nob = Monod(tsim, dt, x0['NOB'], p['NOB'])
            self.psb = PSB(tsim, dt, x0['PSB'], p['PSB'])
            self.phy = Phygrowth(tsim, dt, x0['phy'], p['phy'])
            self.fish = Fish(tsim, dt, x0['fish'], p['fish'])
            
            self.tsim = tsim
            self.dt = dt
        except KeyError as e:
            print(f"Error initializing FishPond: missing configuration for {e}")

    # Define simulations with inputs and disturbances
    def run_simulation(self, tspan, d, u):
        #Disturbances and inputs
        local_d = {key: dict(val) for key, val in d.items()}
        local_u = {key: dict(val) for key, val in u.items()}
        
        ddb = local_d['DB']
        daob = local_d['AOB']
        dnob = local_d['NOB']
        dpsb = local_d['PSB']
        dphy = local_d['phy']
        dfish = local_d['fish']
        udb = local_u['DB']
        upsb = local_u['PSB']
        ufish = local_u['fish']
        
        #Run DB model
        y1 = self.db.run(tspan, ddb, udb)
        
        #retrieve DB result for AOB
        daob['SNH4'] = np.array([y1["t"], y1["SNH4"]]).T
    
        #run AOB model
        y2 = self.aob.run(tspan, daob)
        
        #retrieve AOB result for NOB
        dnob['S_out'] = np.array([y2["t"], y2["P"]]).T
        
        #run NOB model
        y3 = self.nob.run(tspan, dnob)
        
        #run psb model
        y4 = self.psb.run(tspan, dpsb, upsb)
        
        #make simulation result of y1 and y2 as Phygrowth state variable of x0['phy']['NA']
        dphy['SNH4'] = np.array([y1["t"], y1["P"]]).T
        dphy['SNO2'] = np.array([y2["t"], y2["P"]]).T
        dphy['SNO3'] = np.array([y3["t"], y3["P"]]).T 
        dphy['SP'] = np.array([y4["t"], y4["P"]]).T 
        
        #Run phyto model
        yphy = self.phy.run(tspan, dphy)
        
        #retrieve simulation result of Phygrowth
        f3 = self.phy.f['f3']
        
        #make simulation result of yphy as input for fish
        dfish['Mphy'] = np.array([yphy["t"], yphy["Mphy"]]).T 
        
        #run fish model
        yfish = self.fish.run(tspan, dfish, ufish)
        
        #retrieve simulation result of fish
        f_N_prt = self.fish.f['f_N_prt']
        f_P_prt = self.fish.f['f_P_prt']
        f_TAN = self.fish.f['f_TAN']
        f_P_sol = self.fish.f['f_P_sol']
        
        #make simulation result of Phygrowth and fish as disturbances of DB and PSB
        ddb['f3'] = np.array([yphy["t"], f3]).T 
        ddb['f_N_prt'] = np.array([yfish["t"], f_N_prt]).T
        daob['f_TAN'] = np.array([yfish["t"], f_TAN]).T
        dpsb['f3'] = np.array([yphy["t"], f3]).T
        dpsb['f_P_prt'] = np.array([yfish["t"], f_P_prt]).T
        dpsb['f_P_sol'] = np.array([yfish["t"], f_P_sol]).T
        
        #Simulation result
        
        return {
                'NIB': y1,
                'PSB': y2,
                'Phyto': yphy,
                'Fish': yfish,
                'SNO3': y1['SNO3'],
                'SP': y2['P'],
                'Yield': yfish['Mfish'],
                'f_P_sol': f_P_sol
                }

class RCF():
    """ 
    Module for modelling IRFF: Rice-Cum-Fish Method for one cycle
    Fish were grown with rice in the field. Fish-fry were stocked at 3-7 days
    after rice transplanting.
   
    References: 
        Abdulrachman et al. (2015)
        Dwiyana and Mendoza (2006)      

    """
    # Initialize object
    def __init__(self,tsim,dt,x0,p, n_fish, n_rice):
        try:    
            self.db = DB(tsim, dt, x0['DB'], p['DB'])
            self.aob = AOB(tsim, dt, x0['AOB'], p['AOB'])
            self.nob = Monod(tsim, dt, x0['NOB'], p['NOB'])   
            self.psb = PSB(tsim, dt, x0['PSB'], p['PSB'])
            self.phy = Phygrowth(tsim, dt, x0['phy'], p['phy'])
            self.fish = Fish(tsim, dt, {key: val*n_fish for key, val in x0['fish'].items()}, p['fish'])
            self.fish0 = Fish(tsim, dt, {key: val*0 for key, val in x0['fish'].items()}, p['fish'])
            self.rice = Rice(tsim, dt,x0['rice'], p['rice'])
            self.tsim = tsim
            self.dt = dt
            self.n_fish = n_fish
            self.n_rice = n_rice
        except KeyError as e:
            print(f"Error initializing FishPond: missing configuration for {e}")
            
    # Define simulations with inputs and disturbances
    def run_simulation(self, tspan, d, u, pond_volume):
        self.pond_volume = pond_volume
        flow_rice = self.rice.f
        local_d = {key: dict(val) for key, val in d.items()}
        local_u = {key: dict(val) for key, val in u.items()}
        
        #Disturbances and inputs
        ddb = local_d['DB']
        daob = local_d['AOB']
        dnob = local_d['NOB']
        dpsb = local_d['PSB']
        dphy = local_d['phy']
        dfish = local_d['fish']
        drice = local_d['rice']
        udb = local_u['DB']
        upsb = local_u['PSB']
        ufish = {key: val*self.n_fish for key, val in local_u['fish'].items()}
        urice = local_u['rice']
        
        #Run DB model
        y1 = self.db.run(tspan, ddb, udb)
        
        #retrieve DB result for AOB
        daob['SNH4'] = np.array([y1["t"], y1["P"]*self.pond_volume]).T
    
        #run AOB model
        y2 = self.aob.run(tspan, daob)
        dnob['S_out'] = np.array([y2["t"], y2["P"]]).T
        
        #run NOB model
        y3 = self.nob.run(tspan, dnob)
        
        #run psb model
        y4 = self.psb.run(tspan, dpsb, upsb)
        
        #make simulation result of y1 and y2 as Phygrowth state variable of x0['phy']['NA']
        dphy['SNH4'] = np.array([y1["t"], y1["P"]]).T
        dphy['SNO2'] = np.array([y2["t"], y2["P"]]).T
        dphy['SNO3'] = np.array([y3["t"], y3["P"]]).T 
        dphy['SP'] = np.array([y4["t"], y4["P"]]).T 
        
        #Run phyto model
        yphy = self.phy.run(tspan, dphy)
        
        #retrieve simulation result of Phygrowth
        f3 = self.phy.f['f3']
        
        #run rice model
        yrice = self.rice.run(tspan, drice, urice)

        #make simulation result of yphy as input for fish
        dfish['Mphy'] = np.array([yphy["t"], yphy["Mphy"]]).T 
        
        #run fish model
        yfish = {'t': [], 'Mfish':[], 'Mdig': [], 'Muri': [], 'f_N_prt': [], 'f_P_prt': [], 'f_TAN': [], 'f_P_sol': []}

        
        for t in self.tsim:
            if t>=21:
                drice['SNO3'] = np.array([y3["t"], y3["P"]]).T 
                drice['SP'] = np.array([y4["t"], y4["P"]]).T  
                #retrieve rice simulation result for phytoplankton
                dphy['DVS'] = np.array([yrice['t'], yrice['DVS']]).T
    
            if t < 24:
               #make the fish weight zero to simulate 'no growth'
               yfish0 = self.fish0.run((0,23), dfish, ufish)
               yfish['t'].append(yfish0['t'][1])
               yfish['Mfish'].append(0)
               yfish['Mdig'].append(0)
               yfish['Muri'].append(0)
               yfish['f_N_prt'].append(0)
               yfish['f_P_prt'].append(0)
               yfish['f_TAN'].append(0)
               yfish['f_P_sol'].append(0)
            elif t > 80:
                   #make the fish weight zero to simulate 'no growth'
                   yfish0 = self.fish0.run((80,self.tsim[-1]), dfish, ufish)
                   yfish['t'].append(yfish0['t'][1])
                   yfish['Mfish'].append(0)
                   yfish['Mdig'].append(0)
                   yfish['Muri'].append(0)
                   yfish['f_N_prt'].append(0)
                   yfish['f_P_prt'].append(0)
                   yfish['f_TAN'].append(0)
                   yfish['f_P_sol'].append(0)
            else: 
               yfishr =  self.fish.run((24,79), dfish, ufish)
               yfish['t'].append(yfishr['t'][1])
               yfish['Mfish'].append(yfishr['Mfish'][1])
               yfish['Mdig'].append(yfishr['Mdig'][1])
               yfish['Muri'].append(yfishr['Muri'][1])
               yfish['f_N_prt'].append(yfishr['f_N_prt'][1])
               yfish['f_P_prt'].append(yfishr['f_P_prt'][1])
               yfish['f_TAN'].append(yfishr['f_TAN'][1])
               yfish['f_P_sol'].append(yfishr['f_P_sol'][1])
               f_N_prt =  yfishr['f_N_prt']
               f_P_prt =  yfishr['f_P_prt']
               f_TAN =  yfishr['f_TAN']
               f_P_sol =  yfishr['f_P_sol']
               #make simulation result of Phygrowth and fish as disturbances of DB and PSB
               ddb['f3'] = np.array([yphy["t"], f3]).T 
               ddb['f_N_prt'] = np.array([yfishr["t"], f_N_prt]).T
               daob['f_TAN'] = np.array([yfishr["t"], f_TAN]).T
               dpsb['f3'] = np.array([yphy["t"], f3]).T
               dpsb['f_P_prt'] = np.array([yfishr["t"], f_P_prt]).T
               dpsb['f_P_sol'] = np.array([yfishr["t"], f_P_sol]).T    
               #Retrieve simulation results of bacteria and fish as disturbances for rice
               drice['f_P_sol'] = np.array([yfishr["t"], f_P_sol]).T
        yfish = {key: np.array(value) for key, value in yfish.items()}          
        return {
            'Fish': yfish,    
            'Rice': {key: val*self.n_rice for key, val in yrice.items()},
            'riceflow': flow_rice,
            'DB': y1,
            'AOB': y2,
            'NOB': y3,
            'PSB': y4,
            'Phyto': yphy,
                }

class CRF():
    """ 
    Module for modelling IRFF: Rice-Cum-Fish Method for one cycle
    Fish were grown with rice in the field. Fish-fry were stocked at 3-7 days
    after rice transplanting.
   
    References: 
        Abdulrachman et al. (2015)
        Dwiyana and Mendoza (2006)      

    """
    # Initialize object
    def __init__(self,tsim,dt,x0,p, n_fish, n_rice):
        try:    
            self.db = DB(tsim, dt, x0['DB'], p['DB'])
            self.aob = AOB(tsim, dt, x0['AOB'], p['AOB'])
            self.nob = Monod(tsim, dt, x0['NOB'], p['NOB'])   
            self.psb = PSB(tsim, dt, x0['PSB'], p['PSB'])
            self.phy = Phygrowth(tsim, dt, x0['phy'], p['phy'])
            self.fish = Fish(tsim, dt, {key: val*n_fish for key, val in x0['fish'].items()}, p['fish'])

            self.rice = Rice(tsim, dt,x0['rice'], p['rice'])
            self.tsim = tsim
            self.dt = dt
            self.n_fish = n_fish
            self.n_rice = n_rice
        except KeyError as e:
            print(f"Error initializing FishPond: missing configuration for {e}")

    # Define simulations with inputs and disturbances
    def run_simulation(self, tspan, d, u, pond_volume):
        self.pond_volume = pond_volume
        flow_rice = self.rice.f
        local_d = {key: dict(val) for key, val in d.items()}
        local_u = {key: dict(val) for key, val in u.items()}
        
        #Disturbances and inputs
        ddb = local_d['DB']
        daob = local_d['AOB']
        dnob = local_d['NOB']
        dpsb = local_d['PSB']
        dphy = local_d['phy']
        dfish = local_d['fish']
        drice = local_d['rice']
        udb = local_u['DB']
        upsb = local_u['PSB']
        ufish = {key: val*self.n_fish for key, val in local_u['fish'].items()}
        urice = local_u['rice']
        
        it = np.nditer(self.tsim[:-1], flags=['f_index'])
        for ti in it:
            # Index for current time instant
            idx = it.index
            # Integration span
            tspan = (self.tsim[idx], self.tsim[idx+1])
            
            y1 = self.db.run(tspan, ddb, udb)
            
            #retrieve DB result for AOB
            daob['SNH4'] = np.array([y1["t"], y1["P"]*self.pond_volume]).T
        
            #run AOB model
            y2 = self.aob.run(tspan, daob)
            dnob['S_out'] = np.array([y2["t"], y2["P"]]).T
            
            #run NOB model
            y3 = self.nob.run(tspan, dnob)
            
            #run psb model
            y4 = self.psb.run(tspan, dpsb, upsb)
            
            #make simulation result of y1 and y2 as Phygrowth state variable of x0['phy']['NA']
            dphy['SNH4'] = np.array([y1["t"], y1["P"]]).T
            dphy['SNO2'] = np.array([y2["t"], y2["P"]]).T
            dphy['SNO3'] = np.array([y3["t"], y3["P"]]).T 
            dphy['SP'] = np.array([y4["t"], y4["P"]]).T 
            
            #Run phyto model
            yphy = self.phy.run(tspan, dphy)
            
            #retrieve simulation result of Phygrowth
            f3 = yphy['f3']
            print('yphy: ', yphy['t'].shape)
            print('f3:', f3.shape)
            ddb['f3']=np.array([yphy['t'], yphy['f3']]).T
            dpsb['f3']=np.array([yphy['t'], yphy['f3']]).T
            
            #run rice model
            yrice = self.rice.run(tspan, drice, urice)
            if ti>=21:
                drice['SNO3'] = np.array([y3["t"], y3["P"]]).T 
                drice['SP'] = np.array([y4["t"], y4["P"]]).T  
                #retrieve rice simulation result for phytoplankton
                dphy['DVS'] = np.array([yrice['t'], yrice['DVS']]).T
            
            if ti >= 24 or ti <80:
                # Run fish model normally
                dfish['Mphy'] = np.array([yphy["t"], yphy["Mphy"]]).T
                yfish = self.fish.run(tspan, dfish, ufish)
                f_N_prt = yfish['f_N_prt']
                f_P_prt = yfish['f_P_prt']
                f_TAN = yfish['f_TAN']
                f_P_sol = yfish['f_P_sol']
                
                #make simulation result of Phygrowth and fish as disturbances of DB and PSB
                ddb['f3'] = np.array([yphy["t"], f3]).T 
                ddb['f_N_prt'] = np.array([yfish["t"], f_N_prt]).T
                daob['f_TAN'] = np.array([yfish["t"], f_TAN]).T
                dpsb['f3'] = np.array([yphy["t"], f3]).T
                dpsb['f_P_prt'] = np.array([yfish["t"], f_P_prt]).T
                dpsb['f_P_sol'] = np.array([yfish["t"], f_P_sol]).T
            else:
                # Before day 24, fish model computations are skipped
                yfish = {'Mfish': np.zeros_like(tspan), 'Mdig': np.zeros_like(tspan), 'Muri': np.zeros_like(tspan)}
            
        return {
            'Fish': yfish,    
            'Rice': {key: val*self.n_rice for key, val in yrice.items()},
            'riceflow': flow_rice,
            'DB': y1,
            'AOB': y2,
            'NOB': y3,
            'PSB': y4,
            'Phyto': yphy,
                }
class ROTF:
    """ 
    Module for modelling IRFF: Rotational Fish-Rice Method for one year (365 days)
    Fish were grown before and after the rice cultivation. The dikes and water level
    were made higher during fish cultivation and lowered when the rice cultivation started.
    Fish were stocked by farmers 3-7 days after land preparations and 
    animal manure application.Fish farvesting was conducted after 30-70 days, 
    then the field are prepared for growing rice for the next season.
    In this simulation, fish is being harvested after 55 days of cultivation.
    
    References: 
        Abdulrachman et al. (2015)
        Dwiyana and Mendoza (2006)      
        
    Returns
    -------
    y : dictionary
        Model outputs as 1D arrays ('S', 'X'),
        and the evaluation time 't'.
    """
    def __init__(self, tsim, dt, x0, p, cultivation_cycles, n_fish, n_rice):
        self.n_fish = n_fish #[no. of fish] Fish stocking density
        self.n_rice = n_rice #[no. of plants] Number of rice plants on the field
        self.tsim = pd.to_datetime(tsim)
        self.dt = dt
        self.models = {
            'fish': Fish(tsim, dt, {key:val*n_fish for key, val in x0['fish'].items()}, p['fish']),
            'rice': Rice(tsim, dt, {key: val*n_rice for key, val in x0['rice'].items()}, p['rice']),
            'db': DB(tsim, dt, x0['DB'], p['DB']),
            'aob': AOB(tsim, dt, x0['AOB'], p['AOB']),
            'nob': Monod(tsim, dt, x0['NOB'], p['NOB']),
            'psb': PSB(tsim, dt, x0['PSB'], p['PSB']),
            'phy': Phygrowth(tsim, dt, x0['phy'], p['phy'])
        }
        self.cultivation_cycles = cultivation_cycles
        
    def date_to_index(self, date_str):
        """Converts a date string to an index in the tsim array."""
        date = pd.to_datetime(date_str)
        return np.where(self.tsim == date)[0][0]
    
    def extract_output(self, result, key):
        """Extracts output data from results for use as input in another model."""
        return np.array([self.tsim, result[key]])
    
    def run_model(self, model_name, cycle, d=None, u=None):
        """Runs a specific model for a defined cycle time span."""
        self.d = {}  # Initialize or load d
        start, end = self.cultivation_cycles[cycle]
        tspan = (self.date_to_index(start), self.date_to_index(end))
        return self.models[model_name].run(tspan, self.d[model_name], self.u[model_name])

    def run_simulation(self):
        # Follow the specified sequence of model runs and interactions
        results = {}
        # Run DB, AOB, NOB, and PSB model for the full year
        results['db'] = self.run_model('db', 'all')
        # Update disturbances for AOB based on DB output
        self.d['aob']['SNH4'] = self.extract_output(results['db'], 'SNH4')
        results['aob'] = self.run_model('aob', 'all')
        # Update disturbances for NOB based on AOB output
        self.d['nob']['S_out'] = self.extract_output(results['aob'], 'SNO2')
        results['nob'] = self.run_model('nob', 'all')
        # Update disturbances for Phy based on DB, AOB, NOB, PSB output
        results['psb'] = self.run_model('psb', 'all')
        
        # Update disturbances for Phy based on DB, AOB, NOB, PSB output
        self.d['phy']['SNH4'] = self.extract_output(results['db'], 'SNH4')
        self.d['phy']['SNO2'] = self.extract_output(results['aob'], 'SNO2')
        self.d['phy']['SNO3'] = self.extract_output(results['nob'], 'SNO3')
        self.d['phy']['SP'] = self.extract_output(results['psb'], 'P')
        
        #run phytoplankton model
        results['phy'] = self.run_model('phy', 'all')
        #update disturbances of DB and PSB based on phy output
        self.d['db']['f3'] = self.extract_output(self.models['phy'].f['f3'], 'f3')

       # Run fish and rice models according to their specific cycles
        for i, model_name in enumerate(['fish', 'rice']):
            for j, model in enumerate(self.models[model_name]):
                cycle_key = f'{model_name[0]}{j+1}'  # f1, f2, f3 for fish, r1, r2 for rice
                results[cycle_key] = self.run_model(model_name, cycle_key, model)

        return results
    
    def update_model_flows(self, model_name, flows):
        """Update or store the flows after each model run."""
        # This could be storing or logging flows, depending on your needs
        # Example: storing flows in a dictionary
        if model_name in self.models_flow:
            self.models_flow[model_name].append(flows)
        else:
            self.models_flow[model_name] = [flows]