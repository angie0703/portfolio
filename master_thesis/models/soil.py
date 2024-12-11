# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 19:24:46 2024

@author: Angela
"""
import numpy as np
from models.module import Module
from models.integration import fcn_euler_forward 

class SoilN(Module):
   """ 
   Model framework:
      Soil N available = N mineralization + Net Nertilizer N Supply - (Leaves N Uptake + Stem N Uptake)
      N mineralization = N from nitrification process (S_NO3)
      Net fertilizer N supply = Daily fertilizer N application * N recovery fraction
      
      Soil P available = P mineralization + Net Fertilizer P Supply - (leaves P uptake + Stem P uptake)
   Parameters
   ----------
   tsim : array
       Sequence of time points for the simulation
   dt : float
       Time step size [d]
   x0 : dictionary
       Initial conditions of the state variables \n
       
       ======    =============================================================
       key       meaning
       ======    =============================================================
       'SN'      [kg N/ha] Supplies of crop-available Nitrogen
       'SP'      [kg P/ha] Supplies of crop-available Phosphorus
       ======  =============================================================
       
   p : dictionary of scalars
       Model parameters \n
       
       Step 1:
        ======  =============================================================
        key     meaning
        ======  =============================================================
        'R_N'    [-] N maximum recovery fraction
        'R_P'    [-] P maximum recovery fraction
        'R_K'    [-] K maximum recovery fraction
        ======  =============================================================
           

   d : dictionary of floats or arrays
       Model disturbances (required for method 'run'),
       of shape (len(t_d),2) for time and disturbance.
       
       =======  ============================================================
       key      meaning
       =======  ============================================================
        'pH'     [-] pH (water)
       =======  ============================================================

   u : dictionary of 2D arrays
       Controlled inputs (required for method 'run'),
       of shape (len(t_d),2) for time and controlled input.
       Fertilizer application should be provided as an input table, consists of the day after emergence,
       and 
       
       =======  ============================================================
       key      meaning
       =======  ============================================================
        'I_N'    [kg/ha] N concentration in fertilizer
        'I_P'    [kg/ha] P concentration in fertilizer
       =======  ============================================================ 

   Returns
   -------
   y : dictionary
       Model outputs as 1D arrays ('SN', 'SP', 'B_RP', 'B_RO', 'N_ST', 'N_LE', 'N_PA')
       and the evaluation time 't'.
   """
   def __init__(self, tsim, dt, x0, p):
       Module.__init__(self, tsim, dt, x0, p)
       # Initialize dictionary of flows
       self.f = {}
       self.f_keys = ('f_P', 'f_SR', 'f_G', 'f_MR',
                      'f_R', 'f_S', 'f_Hr', 'f_Gr')
       for k in self.f_keys:
           self.f[k] = np.full((self.t.size,), np.nan)
   
   def diff(self, _t, _x0):
       # -- Initial conditions
       # -- 
       # -- Soil N available
       SN, SP = _x0[0], _x0[1]
       
       # -- Physical constants

       
       # -- Model parameteres
       R_N = self.p['R_N']    #[-] N maximum recovery fraction
       R_P = self.p['R_P']    #[-] P maximum recovery fraction
       
       # -- Disturbances at instant _t
       pH = self.d['pH']     #[-] pH (water)
       _pH = np.interp(_t,pH[:,0],pH[:,1])     # [-] pH (water)
       
       # -- Controlled inputs
       I_N = self.u['I_N']    #[kg/ha] N fertilizer application
       I_P = self.u['I_P']    #[kg/ha] P fertilizer application
       
       # -- Supporting equations
       # - Mass

       
       # - Temperature index [-]

       
       # - Photosynthesis
       LAI = a*Wg                      # [m2 m-2] Leaf area index
       if TI==0 and _I0==0:
           P = 0.0
       else:
           Pm = P0*TI                      # [kgCO2 m-2 d-1] Max photosynthesis
           C1 = alpha*k*_I0/(1-m)          # [kgCO2 m-2 d-1]
           C2 = (C1+Pm)/(C1*np.exp(-k*LAI)+Pm) # [-]
           P = Pm/k*np.log(C2)             # [kgCO2 m-2 d-1] Photosynthesis rate
       
        # - Flows
       # Photosynthesis [kgC m-2 d-1]
       #f_P = _WAI*phi*theta*P
       # Shoot respiration [kgC m-2 d-1]
       #f_SR = (1-Y)/Y * mu_m*Ws*Wg/W
       # Maintenance respiration [kgC m-2 d-1]
       #f_MR = M*Wg
       # Growth [kgC m-2 d-1]
       #f_G = mu_m*Ws*Wg/W
       # Senescence [kgC m-2 d-1]
       #f_S = beta*Wg
       # Recycling (can solve Ws<0, remove for student version)
       # if Ws<1E-5:
       #     f_R = 0.10*Wg
       # else:
       #     f_R = 0
       f_R = 0
       
       # -- Differential equations [kgC m-2 d-1]
       dSN_dt = 
       dSP_dt = 
       
       # -- Store flows [kgC m-2 d-1]
       # idx = np.isin(self.t, _t)
       # self.f['f_P'][idx] = f_P
       # self.f['f_SR'][idx] = f_SR
       # self.f['f_G'][idx] = f_G
       # self.f['f_MR'][idx] = f_MR
       # self.f['f_R'][idx] = f_R
       # self.f['f_S'][idx] = f_S
       # self.f['f_Hr'][idx] = f_Hr
       # self.f['f_Gr'][idx] = f_Gr
       
       return np.array([dSN_dt,dSP_dt])
   
   def output(self, tspan):
       # Retrieve object properties
       dt = self.dt        # integration time step size
       diff = self.diff    # function with system of differential equations
       SN0 = self.x0['Ws'] # initial condition
       SP0 = self.x0['Wg'] # initial condiiton
       
       # Numerical integration
       y0 = np.array([SN0,SP0])
       y_int = fcn_euler_forward(diff,tspan,y0,h=dt)
       
       # Model results
       # assuming 0.4 kgC/kgDM (Mohtar et al. 1997, p. 1492)
       t = y_int['t']
       SN = y_int['y'][0,:]
       SP = y_int['y'][1,:]
       #LAI = a*Wg
       return {
           't':t,          # [d] Integration time
           'SN':SN,        #  
           'SP':SP,        #  
           #'LAI':LAI,      # [-] Leaf area index
       }