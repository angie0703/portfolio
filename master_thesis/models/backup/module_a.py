#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   Daniel Reyes Lastiri, Stefan Maranus,
            Rachel van Ooteghem, Tim Hoogstad

Basic class for a model module
"""
    
import copy
import numpy as np
import pandas as pd

class Module():
    def __init__(self,tsim,dt,x0,p):
        # Simulation time array
        self.tsim = tsim
        # Module time-step
        self.dt = dt
        # Module time array
        len_t = int((tsim[-1]-tsim[0])/dt) + 1
        self.t = np.linspace(tsim[0],tsim[-1],len_t)
        # Initial conditions
        self.x0 = copy.deepcopy(x0)
        # Parameters
        self.p = copy.deepcopy(p)
        # Logs of simulation results
        self.y = {}
    
    def run(self,tspan,d=None,u=None):
        """ Run the attribute 'model' of Module instance
        
        Parameters
        ----------
        tspan : 2-element array-like
            initial and final time for the model run
        d : dictionary
            disturbances
        u : dictionary
            controlled inputs
        """
        # Assign disturbances and control input
        self.d, self.u = d, u
        # Call model
        print('run',self.u)
        y = self.output(tspan)
        # Update model output logs
        # (if first simulation, initialize logs)
        if len(self.y) == 0:
            self.y_keys = [yk for yk in y.keys() if yk!='t']
            for k in self.y_keys:
                self.y[k] = np.full((len(self.t),),np.nan)       
        for k in self.y_keys:
            idxs = np.isin(self.t,y['t'])
            self.y[k][idxs] = y[k]
        # Update initial conditions
        for k in self.x0.keys():
            self.x0[k] = y[k][-1]
        return y

    def ns(self,x0,p_ref,d=None,u=None,y_keys=None):
        # Reset intial conditions for reference module
        self.x0 = copy.deepcopy(x0)
        # Initialize instances for -/+
        instance_mns = copy.deepcopy(self)
        instance_pls = copy.deepcopy(self)
        # Run module for reference parameters
        tspan = (self.t[0],self.t[-1])
        y_ref = self.run(tspan,d,u)
        # Initialize pandas dataframe for normalized sensitivities (with nan)
        # - create MultiIndex
        if not y_keys:
            y_keys = [yk for yk in self.y_keys]
        p_keys = [pk for pk in p_ref.keys()]
        pm = ['-','+']
        iterables = [y_keys,p_keys,pm]
        midx = pd.MultiIndex.from_product(iterables, names=['y', 'p', '-/+'])
        # - initialize nan DataFrame
        nan_arr = np.full((self.t.size, 2*len(y_keys)*len(p_keys)),0.)
        ns_df = pd.DataFrame(nan_arr, index=self.t, columns=midx)
        # Iterate over model parameters
        for kp in p_ref.keys():
            # Reset initial conditions and parameters for mns and pls modules
            instance_mns.x0 = copy.deepcopy(x0)
            instance_pls.x0 = copy.deepcopy(x0)
            p_mns = copy.deepcopy(p_ref)
            p_pls = copy.deepcopy(p_ref)
            instance_mns.p = p_mns
            instance_pls.p = p_pls
            # Modify parameter kp
            instance_mns.p[kp] = 0.95*p_ref[kp]
            instance_pls.p[kp] = 1.05*p_ref[kp]
            if kp[0] == 'T':
                instance_mns.p[kp] = p_ref[kp]-1.0
                instance_pls.p[kp] = p_ref[kp]+1.0
            # Run model
            y_mns = instance_mns.run(tspan,d,u)
            y_pls = instance_pls.run(tspan,d,u)
            # Compute normalized sensitivities per model output
            for ky in y_keys:
                # Compute sensitivity
                s_mns = (y_mns[ky]-y_ref[ky])/(p_mns[kp]-p_ref[kp])
                s_pls = (y_pls[ky]-y_ref[ky])/(p_pls[kp]-p_ref[kp])
                # Compute normalized sensitivity
                ns_mns = s_mns*p_ref[kp]/np.average(y_ref[ky])
                ns_pls = s_pls*p_ref[kp]/np.average(y_ref[ky])
                # Store in corresponding column in DataFrame
                # (tuple to access MultiIndex column name)
                ns_df.loc[:,(ky,kp,'-')] = ns_mns
                ns_df.loc[:,(ky,kp,'+')] = ns_pls
        # Add columns for reference model outputs
        for yk in y_keys:
            ns_df[yk,'ref','ref'] = y_ref[yk]
        # Results
        return ns_df
