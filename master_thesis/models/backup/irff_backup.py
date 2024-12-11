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
from copy import deepcopy

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
    def __init__(self,tsim,dt,x0,p):
        try:    
            self.db = DB(tsim, dt, x0['DB'], p['DB'])
            self.aob = AOB(tsim, dt, x0['AOB'], p['AOB'])
            self.nob = Monod(tsim, dt, x0['NOB'], p['NOB'])   
            self.psb = PSB(tsim, dt, x0['PSB'], p['PSB'])
            self.phy = Phygrowth(tsim, dt, x0['phy'], p['phy'])
            self.fish = Fish(tsim, dt, x0['fish'], p['fish'])
            self.rice = Rice(tsim, dt, x0['rice'], p['rice'])
            self.tsim = tsim
            self.dt = dt
            
        except KeyError as e:
            print(f"Error initializing FishPond: missing configuration for {e}")
            
        #Set initial active states
        self.db.active = True
        self.aob.active = True
        self.nob.active = True
        self.psb.active = True
        self.phy.active = True
        self.rice.active = True
        self.fish.active = True

    # Define simulations with inputs and disturbances
    def run_simulation(self, tspan, d, u):
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
        ufish = local_u['fish']
        urice = local_u['rice']
        
        #Run DB model
        y1 = self.db.run(tspan, ddb, udb)
        
        #retrieve DB result for AOB
        daob['SNH4'] = np.array([y1["t"], y1["P"]]).T
    
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
        
        yfish = self.fish.run(tspan, dfish, ufish)
        f_N_prt = self.fish.f['f_N_prt']
        f_P_prt = self.fish.f['f_P_prt']
        f_TAN = self.fish.f['f_TAN']
        f_P_sol = self.fish.f['f_P_sol']
        
        for t in self.tsim:
            if t>=21:
                drice['SNO3'] = np.array([y3["t"], y3["P"]]).T 
                drice['SP'] = np.array([y4["t"], y4["P"]]).T  
                #retrieve rice simulation result for phytoplankton
                dphy['DVS'] = np.array([yrice['t'], yrice['DVS']]).T
    
            if t < 24:
               #make the fish weight zero to simulate 'no growth'
               yfish['Mfish'][int(t)] = 0
               print('time:',t)
               print('Fish: ', yfish['Mfish'])
               yfish['Mdig'][int(t)] = 0
               yfish['Muri'][int(t)] = 0 
            else: 
               #retrieve simulation result of fish
               print('t:else', t)
               #make simulation result of Phygrowth and fish as disturbances of DB and PSB
               ddb['f3'] = np.array([yphy["t"], f3]).T 
               ddb['f_N_prt'] = np.array([yfish["t"], f_N_prt]).T
               daob['f_TAN'] = np.array([yfish["t"], f_TAN]).T
               dpsb['f3'] = np.array([yphy["t"], f3]).T
               dpsb['f_P_prt'] = np.array([yfish["t"], f_P_prt]).T
               dpsb['f_P_sol'] = np.array([yfish["t"], f_P_sol]).T    
               #Retrieve simulation results of bacteria and fish as disturbances for rice
               drice['f_P_sol'] = np.array([yfish["t"], f_P_sol]).T
                           
            if t > 80:
                self.fish.active = True
                #make the fish weight zero to simulate 'no growth'
                yfish['Mfish'][int(t)] = 0
                yfish['Mdig'][int(t)] = 0
                yfish['Muri'][int(t)] = 0 
                   
        return {
            'Fish': yfish,    
            'Rice': yrice,
            'riceflow': flow_rice,
            'DB': y1,
            'AOB': y2,
            'NOB': y3,
            'PSB': y4,
            'Phyto': yphy,
                }

class ROTF():
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
    # Initialize object
    def __init__(self,tsim,dt,x0,p):
        try:    
            self.db = DB(tsim, dt, x0['DB'], p['DB'])
            self.aob = AOB(tsim, dt, x0['AOB'], p['AOB'])
            self.nob = Monod(tsim, dt, x0['NOB'], p['NOB'])   
            self.psb = PSB(tsim, dt, x0['PSB'], p['PSB'])
            self.phy = Phygrowth(tsim, dt, x0['phy'], p['phy'])
            
            #3 cycles of fish
            self.fish1 = Fish(tsim, dt, x0['fish'], p['fish'])
            self.fish2 = Fish(tsim, dt, x0['fish'], p['fish'])
            self.fish3 = Fish(tsim, dt, x0['fish'], p['fish'])
            
            #2 cycles of rice
            self.rice1 = Rice(tsim, dt, x0['rice'], p['rice'])
            self.rice2 = Rice(tsim, dt, x0['rice'], p['rice'])
            
            self.tsim = tsim
            self.dt = dt
            
            self.x0 = deepcopy(x0)
        except KeyError as e:
            print(f"Error initializing ROTF: missing configuration for {e}")

    # Define simulations with inputs and disturbances
    def run_simulation(self, tspan, d, u):

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
        ufish = local_u['fish']
        urice = local_u['rice']
        
        #Run DB model
        y1 = self.db.run(tspan, ddb, udb)
        
        #retrieve DB result for AOB
        daob['SNH4'] = np.array([y1["t"], y1["P"]]).T
    
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
        
        #make simulation result of yphy as input for fish
        dfish['Mphy'] = np.array([yphy["t"], yphy["Mphy"]]).T 
        #run fish model
        yfish1 = self.fish1.run(tspan, dfish, ufish)
        f_N_prt1 = self.fish1.f['f_N_prt']
        f_P_prt1 = self.fish1.f['f_P_prt']
        f_TAN1 = self.fish1.f['f_TAN']
        f_P_sol1 = self.fish1.f['f_P_sol']
        
        yfish2 = self.fish2.run(tspan, dfish, ufish)
        f_N_prt2 = self.fish2.f['f_N_prt']
        f_P_prt2 = self.fish2.f['f_P_prt']
        f_TAN2 = self.fish2.f['f_TAN']
        f_P_sol2 = self.fish2.f['f_P_sol']
        
        yfish3 = self.fish3.run(tspan, dfish, ufish)
        f_N_prt3 = self.fish3.f['f_N_prt']
        f_P_prt3 = self.fish3.f['f_P_prt']
        f_TAN3 = self.fish3.f['f_TAN']
        f_P_sol3 = self.fish3.f['f_P_sol']
        
        #run rice model
        yrice1 = self.rice1.run(tspan, drice, urice)
        f_Ph1 = self.rice1.f['f_Ph']
        f_gr1 = self.rice1.f['f_gr']
        f_dmv1 = self.rice1.f['f_dmv']
        f_res1 = self.rice1.f['f_res']
        f_Nlv1 = self.rice1.f['f_Nlv']
        f_pN1 = self.rice1.f['f_pN']
        DVS1 = self.rice1.f['DVS']
        
        yrice2 = self.rice2.run(tspan, drice, urice)
        f_Ph2 = self.rice2.f['f_Ph']
        f_gr2 = self.rice2.f['f_gr']
        f_dmv2 = self.rice2.f['f_dmv']
        f_res2 = self.rice2.f['f_res']
        f_Nlv2 = self.rice2.f['f_Nlv']
        f_pN2 = self.rice2.f['f_pN']
        DVS2 = self.rice2.f['DVS']
        
        #for loop
        '''
        fish running from t=0 to t=30
        fish harvested at t=30 -> fish.active = False
        rice running from t = 31 to t=51 without disturbances
        t>=52 rice running with disturbances
        
        '''
        fish1 = {'t': [],'Mfish': [],'Mdig': [],'Muri':[]}
        fish2 = {'t': [],'Mfish': [],'Mdig': [],'Muri':[]}
        fish3 = {'t': [],'Mfish': [],'Mdig': [],'Muri':[]}
        rice1 = {'t': [],'Mrt': [],'Mst': [],'Mlv':[], 'Mpa': [],'Mgr': [], 'DVS':[], 
                 'f_Ph':[], 'f_dmv':[], 'f_gr':[], 'f_res':[], 'f_Nlv':[], 'f_pN':[]}
        rice2 = {'t': [],'Mrt': [],'Mst': [],'Mlv':[], 'Mpa': [],'Mgr': [], 'DVS':[],
                 'f_Ph':[], 'f_dmv':[], 'f_gr':[], 'f_res':[], 'f_Nlv':[], 'f_pN':[]}
        
        # List of keys to set to zero at index int(t), excluding 't'
        keys_fish = ['Mfish', 'Mdig', 'Muri']
        keys_rice = ['Mrt','Mst','Mlv', 'Mpa','Mgr',
                     # 'DVS','f_Ph', 'f_dmv', 'f_gr', 'f_res', 'f_Nlv', 'f_pN'
                     ]
        
        for t in self.tsim:
            print('tsim.index: ', self.tsim[int(t)])
            idx = self.tsim[int(t)]
            tspan = (idx, idx+1)
            print(tspan)
            #make rice2, fish2, and fish3 value 0 to simulate 'no growth'
            for key in keys_rice:
                yrice2[key][int(t)] = 0
                
            for key in keys_fish:
                yfish2[key][int(t)] = 0
                yfish3[key][int(t)] = 0
                
            #Initial rice and fish (rice1 and fish1)
            rice1['t'].append(yrice1['t'])
            rice1['Mrt'].append(yrice1['Mrt'])
            rice1['Mst'].append(yrice1['Mst'])
            rice1['Mlv'].append(yrice1['Mlv'])
            rice1['Mpa'].append(yrice1['Mpa'])
            rice1['Mgr'].append(yrice1['Mgr'])
            rice1['DVS'].append(DVS1)
            rice1['f_Ph'].append(f_Ph1)
            rice1['f_gr'].append(f_gr1)
            rice1['f_dmv'].append(f_dmv1)
            rice1['f_res'].append(f_res1)
            rice1['f_Nlv'].append(f_Nlv1)
            rice1['f_pN'].append(f_pN1)
            
            #fish1
            fish1['t'].append(yfish1['t'])
            fish1['Mfish'].append(yfish1['Mfish'])
            fish1['Mdig'].append(yfish1['Mdig'])
            fish1['Muri'].append(yfish1['Muri'])
            
            ddb['f3'][int(t)] = np.array([yphy["t"][int(t)], f3[int(t)]]).T 
            ddb['f_N_prt'][int(t)] = np.array([yfish1["t"][int(t)], f_N_prt1[int(t)]]).T
            daob['f_TAN'][int(t)] = np.array([yfish1["t"][int(t)], f_TAN1[int(t)]]).T
            dpsb['f3'][int(t)] = np.array([yphy["t"][int(t)], f3[int(t)]]).T
            dpsb['f_P_prt'][int(t)] = np.array([yfish1["t"][int(t)], f_P_prt1[int(t)]]).T
             
                
            if t>30:
                # Set each specified key's value at index int(t) to zero
                for key in keys_fish:
                    yfish1[key][int(t)] = 0
            
            if t>=30:
                
                #rice transplanted to field = got disturbances from ponds
                drice['SNO3'][int(t)] = np.array([y3["t"][int(t)], y3["P"][int(t)]]).T 
                drice['SP'][int(t)] = np.array([y4["t"][int(t)], y4["P"][int(t)]]).T  
                drice['f_P_sol'][int(t)] = np.array([yfish1["t"][int(t)], f_P_sol1[int(t)]]).T
                #retrieve rice simulation result for phytoplankton
                dphy['DVS'][int(t)] = np.array([yrice1['t'][int(t)], DVS1[int(t)]]).T
                
            if t == 121:
                self.fish2.reset()
                
            if t>= 121:
                # Set each specified key's value at index int(t) to zero
                for key in keys_rice:
                    yrice1[key][int(t)] = 0

                #fish2
                fish2['t'].append(yfish2['t'])
                fish2['Mfish'].append(yfish2['Mfish'])
                fish2['Mdig'].append(yfish2['Mdig'])
                fish2['Muri'].append(yfish2['Muri'])
                
                ddb['f3'][int(t)] = np.array([yphy["t"][int(t)], f3[int(t)]]).T 
                ddb['f_N_prt'][int(t)] = np.array([yfish2["t"][int(t)], f_N_prt2[int(t)]]).T
                daob['f_TAN'][int(t)] = np.array([yfish2["t"][int(t)], f_TAN2[int(t)]]).T
                dpsb['f3'][int(t)] = np.array([yphy["t"][int(t)], f3[int(t)]]).T
                dpsb['f_P_prt'][int(t)] = np.array([yfish1["t"][int(t)], f_P_prt2[int(t)]]).T
                
            if 121 <= t < 152:
                rice2['t'].append(yrice2['t'])
                rice2['Mrt'].append(yrice2['Mrt'])
                rice2['Mst'].append(yrice2['Mst'])
                rice2['Mlv'].append(yrice2['Mlv'])
                rice2['Mpa'].append(yrice2['Mpa'])
                rice2['Mgr'].append(yrice2['Mgr'])
                rice2['DVS'].append(DVS2)
                rice2['f_Ph'].append(f_Ph2)
                rice2['f_gr'].append(f_gr2)
                rice2['f_dmv'].append(f_dmv2)
                rice2['f_res'].append(f_res2)
                rice2['f_Nlv'].append(f_Nlv2)
                rice2['f_pN'].append(f_pN2)
                
                 
            
            if t>= 151:
                drice['SNO3'][int(t)] = np.array([y3["t"][int(t)], y3["P"][int(t)]]).T 
                drice['SP'][int(t)] = np.array([y4["t"][int(t)], y4["P"][int(t)]]).T  
                drice['f_P_sol'][int(t)] = np.array([yfish1["t"][int(t)], f_P_sol2[int(t)]]).T
                #retrieve rice simulation result for phytoplankton
                dphy['DVS'][int(t)] = np.array([yrice1['t'][int(t)], DVS2[int(t)]]).T
                
                
            if t == 242:
                for key in keys_rice:
                    yrice2[key][int(t)] = 0
            
            if 242 < t < 297:    
                #Fish
                fish3['t'].append(yfish3['t'])
                fish3['Mfish'].append(yfish3['Mfish'])
                fish3['Mdig'].append(yfish3['Mdig'])
                fish3['Muri'].append(yfish3['Muri'])
                ddb['f3'][int(t)] = np.array([yphy["t"][int(t)], f3[int(t)]]).T 
                ddb['f_N_prt'][int(t)] = np.array([yfish3["t"][int(t)], f_N_prt3[int(t)]]).T
                daob['f_TAN'][int(t)] = np.array([yfish3["t"][int(t)], f_TAN3[int(t)]]).T
                dpsb['f3'][int(t)] = np.array([yphy["t"][int(t)], f3[int(t)]]).T
                dpsb['f_P_prt'][int(t)] = np.array([yfish3["t"][int(t)], f_P_prt3[int(t)]]).T
                drice['f_P_sol'][int(t)] = np.array([yfish3["t"][int(t)], 0]).T 
            else:
                #make the fish weight zero to simulate 'no growth'
                yfish3['Mfish'][int(t)] = 0
                yfish3['Mdig'][int(t)] = 0
                yfish3['Muri'][int(t)] = 0 
        
        return {
                'Fish1': yfish1,    
                'Fish2': yfish2,
                'Fish3': yfish3,
                'Rice1': yrice1,
                'Rice2': yrice2,
                'DB': y1,
                'AOB': y2,
                'NOB': y3,
                'PSB': y4,
                'Phyto': yphy,
              
                }                