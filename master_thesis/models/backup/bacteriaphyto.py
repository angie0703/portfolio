# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 19:46:50 2024

@author: alegn
"""

import numpy as np

from models.module import Module
from models.integration import fcn_euler_forward, fcn_rk4
from models.bacterialgrowth import NIB, PSB
from models.fish import Fish
from models.phytoplankton import Phygrowth

class fishpond(Module):
    """ Module for all process in the fishpond, consists of
    - Bacteria
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
    tsim : array
        Sequence of time points for the simulation
    dt : float
        Time step size [d]
    x0 : dictionary
        Initial conditions of the state variables \n
        * S : substrate concentration [g m-3 d-1]
        * X : bacteria concentration [g m-3]
        * P : product concentration [g m-3]
    p : dictionary of scalars
        Model parameters \n
        * mu_max_S20 : maximum rate of substrate use at 20 C (g substrate/g microorganisms.d)
        * Ks : half velocity constant (g L-1)
        * Y : bacteria synthesis yield (g bacteria biomass/ g substrate used)
        * b20 : specific endogenous decay coefficient of bacteria at 20 C[g VSS/g VSS.d]
        * teta_mu : temperature correction factor for mu [-]
        * teta_b : temperature correction factor for b [-]
        * Mr : molar Mass of the substrate (g/mol)
        
    Returns
    -------
    y : dictionary
        Model outputs as 1D arrays ('S', 'X'),
        and the evaluation time 't'.
    """
    # Initialize object. Inherit methods from object Module
    def __init__(self,tsim,dt,x0,p):
        Module.__init__(self,tsim,dt,x0,p)
        # Initialize dictionary of flows
        self.f = {}
        self.f_keys = ('')
        for k in self.f_keys:
            self.f[k] = np.full((self.t.size,), np.nan)
            
        self.nib = NIB(self, tsim, dt, x0NIB, pNIB)
        self.psb = PSB(self, tsim, dt, x0PSB, pPSB)
        self.phy = Phygrowth(self, tsim, dt, x0phy, pphy)
        self.fish = Fish(self, tsim, dt, x0fish, pfish)

    # Define system of differential equations of the model
    def diff(self,_t,_x0):
        # State variables
        
        SNprt = _x0[0]
        SNH4 = _x0[1]
        SNO2 = _x0[2]
        XDB = _x0[3]
        XAOB = _x0[4]
        XNOB = _x0[5]
        XPSB = _x0[6]
        SNO3 = _x0[7]
        SP   = _x0[8]
        Mphy = _x0[9]
        Mfish = _x0[10]
        
        # -- Physical constants
        DOmin = 0.3            # [mg/l] DO minimal for open-pond Nile Tilapia
        DOcrit = 1             # [mg/l] DO critical for open-pond Nile Tilapia
        Tminf = 22             # [] min temperature for fish growth
        Toptf = 28             # [] optimal temperature for fish growth
        Tmaxf = 32             # [] max temperature for fish growth
        
        # Parameters DB
        mu_max0DB = self.p['mu_max0DB']
        KsDB = self.p['KsDB']
        # K_DO = self.p['K_DO']
        b20DB = self.p['b20DB']
        teta_muDB = self.p['teta_muDB']
        teta_bDB = self.p['teta_bDB']
        YDB = self.p['YDB']
        
        # Parameters AOB
        mu_max0AOB = self.p['mu_max0AOB']
        KsAOB = self.p['KsAOB']
        # K_DO = self.p['K_DO']
        b20AOB = self.p['b20AOB']
        teta_muAOB = self.p['teta_muAOB']
        teta_bAOB = self.p['teta_bAOB']
        YAOB = self.p['YAOB']
        
        #Parameters NOB
        mu_max0NOB = self.p['mu_max0NOB']
        KsNOB = self.p['KsNOB']
        # K_DO = self.p['K_DO']
        b20NOB = self.p['b20NOB']
        teta_muNOB = self.p['teta_muNOB']
        teta_bNOB = self.p['teta_bNOB']
        YNOB = self.p['YNOB']
        
        kN = self.p['kN'] #[-] fraction of N content in phytoplankton dead bodies
        
        # -- Disturbances at instant _t
        DO = self.d['DO']
        T = self.d['T'] 
        I0 = self.d['I0']
        DVS = self.d['DVS']
        
        _DO = np.interp(_t,DO[:,0],DO[:,1])
        _T = np.interp(_t,T[:,0],T[:,1])
        _I0 = np.interp(_t,I0[:,0],I0[:,1])
        _DVS = np.interp(_t, DVS[:,0], DVS[:,1])
        
        #supporting equations
        # Controllable inputs
        Norgf = self.u['Norgf']
        if _t ==0: 
            SNprt = _f_N_prt + _f3*kN + Norgf
        elif _t>0:
            SNprt = _f_N_prt + _f3*kN
        else:
            SNprt = 0
        
        Porgf = self.u['Porgf'] 
        if _t == 0:
            SPprt += Porgf + _f3*kP +_f_P_prt    
        elif _t>0:
            SPprt += _f3*kP +_f_P_prt
        else:
            SPprt = 0
        
        #Mfed = Mfish*0.03 #recommended amount of feed is 3% of the biomass weight
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
        Mfed = self.u['Mfed']
        fed = Mphy + Mfed
        f_fed = fT*fDO*(r_phy*fed + (1-r_phy)*fed)
        
        # Differential equations

        # -- Store flows [g m-3 d-1]
        idx = np.isin(self.t, _t)
        
        
        return np.array([dSNprt_dt, dSNH4_dt, dSNO2_dt, 
                         dXDB_dt, dXAOB_dt, dXNOB_dt, P_NO3])

    # Define model outputs from numerical integration of differential equations.
    # This function is called by the Module method 'run'.
    def output(self,tspan):
        # Retrieve object properties
        dt = self.dt        # integration time step size
        diff = self.diff    # function with system of differential equations
        SNprt0 = self.x0['SNprt'] # initial condition of Substrate
        SNH40 = self.x0['SNH4'] # initial condition 
        SNO20 = self.x0['SNO2']
        XDB0 = self.x0['XDB']
        XAOB0 = self.x0['XAOB']
        XNOB0 = self.x0['XNOB']
        P_NO30 = self.x0['P_NO3']

        # Numerical integration
        # (for numerical integration, y0 must be numpy array)
        y0 = np.array([SNprt0, SNH40, SNO20, XDB0, XAOB0, XNOB0, P_NO30])
        # y_int = fcn_euler_forward(diff,tspan,y0,h=dt)
        y_int = fcn_rk4(diff, tspan, y0, h=dt)
        
        # Retrieve results from numerical integration output
        t = y_int['t']              # time
              # first output (row 0, all columns)
        SNH4 = y_int['y'][1,:]      # second output (row 1, all columns)
        SNO2 = y_int['y'][2,:]
        XDB = y_int['y'][3,:]
        XAOB = y_int['y'][4,:]
        XNOB = y_int['y'][5,:]
        XPSB = y_int['y'][,:]
        P_NO3 = y_int['y'][6,:]
        Mphy = y_int['y']
        
        return {'t':t, 
                'SNprt':SNprt,
                'SNH4':SNH4,
                'SNO2':SNO2,
                'XDB':XDB,
                'XAOB':XAOB,
                'XNOB':XNOB,
                'P_NO3':P_NO3
                }