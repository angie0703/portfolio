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
import pandas as pd 

class ROTF:
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