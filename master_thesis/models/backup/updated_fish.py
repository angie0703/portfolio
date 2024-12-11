
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 14:30:32 2023

@author: Angela

Class for Fish Growth Model, with disturbances factor from Yi (1999). Environmental factor
such as Temperature, DO, photoperiod, could affect the feeding intake rate, hence the 
overall growth of Nile Tilapia.

"""
import numpy as np
from models.module import Module
from models.integration import fcn_euler_forward, fcn_rk4

class Fish(Module):
    '''
    Fish growth and Nutrient Balances model by Reyes Lastiri (2016)
    Feeding intake rate (f_fed) is based on Jamu et al (2000) that consider
    Nile Tilapia feed intake preferences (phytoplankton>organic matter>artificial fish feed)
    '''

    def __init__(self, tsim, dt, x0, p):
        Module.__init__(self, tsim, dt, x0, p)
        super().__init__(tsim, dt, x0, p)
        # Initialize dictionary of flows
        self.f = {}
        self.f_keys = ('f_upt', 'f_prt', 'f_sol', 'f_digout', 'f_N_sol', 'f_P_sol',
                       'f_diguri', 'f_fed', 'f_N_upt', 'f_P_upt', 'f_N_prt', 'f_P_prt', 'f_TAN', 'r_phy')
        for k in self.f_keys:
            self.f[k] = np.full((self.t.size,), np.nan)
        self.tsim = tsim
        self.dt = dt
        self.x0 = x0
        self.p = p
        self.is_active = True
        self.status = 'inactive'

    def diff(self, _t, _x0):
        if not self.is_active:
            return np.zeros_like(_x0)

        # -- Initial conditions
        Mfish = _x0[0]
        Mdig = _x0[1]
        Muri=  _x0[2]
        
        # -- Model parameteres
        
        # -- Physical constants
        DOmin = 0.3            # [mg/l] DO minimal for open-pond Nile Tilapia
        DOcrit = 1             # [mg/l] DO critical for open-pond Nile Tilapia
        
        # - fish growth
        tau_dig = self.p['tau_dig']  # [h] time constants for digestive
        tau_uri = self.p['tau_uri']  # [h] time constants for urinary
        k_upt = self.p['k_upt']  # [-] fraction of nutrient uptake for fish weight
        k_N_upt = self.p['k_N_upt']  # [-] fraction of N uptake by fish
        k_P_upt = self.p['k_P_upt']  # [-] fraction of P uptake by fish
        k_prt = self.p['k_prt']  # [-] fraction of particulate matter excreted
        k_N_prt = self.p['k_N_prt']  # [-] fraction of N particulate matter excreted
        k_P_prt = self.p['k_P_prt']  # [-] fraction of P particulate matter excreted
        x_N_fed = self.p['x_N_fed'] #[-] fraction of N in feed
        x_P_fed = self.p['x_P_fed'] #[-] fraction of P in feed
        k_DMR = self.p['k_DMR']
        k_TAN_sol = self.p['k_TAN_sol']
        Ksp = self.p['Ksp'] #[g m-3] Half-saturation constant for phytoplankton feeding 
        k_N_sol = 1 - k_N_upt - k_N_prt # [-] fraction of N in soluble matter
        k_P_sol = 1 - k_P_upt - k_P_prt # [-] fraction of P in soluble matter
        Tmin = self.p['Tmin']  # [°C] Minimum Temperature for Nile Tilapia
        Topt = self.p['Topt']  # [°C] Optimum Temperature for Nile Tilapia
        Tmax = self.p['Tmax']  # [°C] Maximum Temperature for Nile Tilapia

        # -- Disturbances at instant _t
        DO = self.d['DO']
        T = self.d['T']
        Mphy = self.d['Mphy']
        _DO = np.interp(_t,DO[:,0],DO[:,1])     # [mg l-1] Dissolved Oxygen
        _T = np.interp(_t,T[:,0],T[:,1])        # [°C] Environment temperature
        _Mphy = np.interp(_t,Mphy[:,0],Mphy[:,1]) #[g d-1] Phytoplankton biomass 
        
        # -- Controlled input
        Mfed = self.u['Mfed']
        
        #-feed intake rate with disturbances factor
        r_phy = _Mphy/(Ksp+_Mphy) #[-] ratio of phytoplankton eaten by fish
        
        fT = 0
        if _T >= Topt:
            fT = np.exp(-4.6*(((_T-Topt)/(Tmax-Topt))**4))
        else:
            fT = np.exp(-4.6*(((Topt-_T)/(Topt-Tmin))**4))
        
        fDO = 0
        if DOmin <= _DO <= DOcrit:
            fDO = (_DO-DOmin)/(DOcrit - DOmin)
        else:
            fDO = 1
        
        #Mfed = Mfish*0.03 #recommended amount of feed is 3% of the biomass weight
        fed = _Mphy + Mfed
        f_fed = fT*fDO*(r_phy*fed + (1-r_phy)*fed)
        
        # - Flows
        
        #Fish Model
        # [gDM h-1] total mass flow rate leaving the digestive system
        f_digout = (1/tau_dig)*Mdig
        # [gDM h-1] nutrient uptake rate
        f_upt = k_upt*f_digout
        # [gDM h-1] particulate matter excretion rate
        f_prt = k_prt*f_digout
        # [gDM h-1] soluble waste excretion rate
        f_sol = (1/tau_uri)*Muri
        # [gDM h-1] digestive to urinary rate
        f_diguri = f_digout - f_upt - f_prt
        
        #Nutrient Balances
        #N uptake flow rate by fish during growth
        f_N_upt = k_N_upt*x_N_fed*f_digout
        #N excretion flow rate by fish during growth
        f_N_prt = k_N_prt*x_N_fed*f_digout
        #P uptake flow rate by fish during growth
        f_P_upt = k_P_upt*x_P_fed*f_digout
        #P excretion flow rate by fish during growth
        f_P_prt = k_P_prt*x_P_fed*f_digout
        #N flow rate in soluble matter
        f_N_sol = k_N_sol*x_N_fed*(f_digout/f_diguri)*f_sol
        #P flow rate in soluble matter
        f_P_sol = k_P_sol*x_P_fed*(f_digout/f_diguri)*f_sol    
        
        # -- Supporting equations
        f_TAN = k_TAN_sol*f_N_sol
        
        # -- Differential equations [gDM day-1]
        dMfish_dt = f_upt
        dMdig_dt = f_fed - f_digout
        dMuri_dt = f_diguri - f_sol

        # -- Store flows [kgC m-2 d-1]
        idx = np.isin(self.t, _t)
        self.f['f_digout'][idx] = f_digout
        self.f['f_fed'][idx] = f_fed
        self.f['f_upt'][idx] = f_upt
        self.f['f_prt'][idx] = f_prt
        self.f['f_sol'][idx] = f_sol
        self.f['f_diguri'][idx] = f_diguri
        self.f['f_N_prt'][idx] = f_N_prt
        self.f['f_P_prt'][idx] = f_P_prt
        self.f['f_N_upt'][idx] = f_N_upt
        self.f['f_P_upt'][idx] = f_P_upt
        self.f['f_N_sol'][idx] = f_N_sol
        self.f['f_P_sol'][idx] = f_P_sol
        self.f['f_TAN'][idx] = f_TAN
        self.f['r_phy'][idx] = r_phy
        
        return np.array([dMfish_dt,
                         dMdig_dt, 
                         dMuri_dt
                         ])
    
    def pause_growth(self):
        self.status = 'inactive'
        print("Growth paused.")

    def resume_growth(self, new_conditions=None):
        self.status = 'active'
        if new_conditions is not None:
            self.x0.update(new_conditions)  # Optionally reset conditions when resuming
        print("Growth resumed.")

    def harvest_fish(self):
        self.status = 'harvested'
        print("Fish harvested.")
        
    def output(self, tspan):
        dt = self.dt
        diff = self.diff
        y0 = np.array([self.x0['Mfish'], self.x0['Mdig'], self.x0['Muri']])

        y_int = fcn_euler_forward(diff, tspan, y0, h=dt)
        t = y_int['t']
        Mfish = y_int['y'][0,:]
        Mdig = y_int['y'][1,:]
        Muri = y_int['y'][2,:]

        return {
            't': t,
            'Mfish': Mfish,
            'Mdig': Mdig,
            'Muri': Muri
        }

    def reset(self):
        self.is_active = False

    def activate(self):
        self.is_active = True
        # Reset initial conditions if necessary

# Example usage:
# fish_model = Fish(tsim=[0, 1, 2], dt=1, x0={'Mfish': 1, 'Mdig': 0.5, 'Muri': 0.2}, p={'param1': 0.1})
# fish_model.reset()
# fish_model.activate()
