# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:35:31 2024

@author: Angela

Model for Lowland rice with Nitrogen-limited and Phosphorus-limited conditions
The model use general structure of ORYZA2000,  

"""

import numpy as np
import pandas as pd
from models.module import Module
from models.integration import fcn_euler_forward, fcn_rk4 
from scipy.interpolate import interp1d


class Rice(Module):
   """    
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
       'Mrt'    [kg C/ha] roots weight
       'Mst'    [kg C/ha] stems weight
       'Mlv'    [kg C/ha] leaves weight
       'Mpa'    [kg C/ha] panicles weight
       'Mgr'    [kg C/ha] grains weight
       ======  =============================================================
       
   p : dictionary of scalars
       Model parameters \n
       
        ======  =============================================================
        key     meaning
        ======  =============================================================
        For developmental stage:
        Tmin    [°C] T minimum for rice growth
        Topt    [°C] T optimum for rice growth
        Tmax    [°C] T maximum for rice growth
        HU      [-] Heat unit
        DVS     [-] Developmental stage value
        SLA     [-] Specific Leaf Area
        LAI     [-] Leaf Area Index
        
        Calculate photosynthetic rate
        Pg      [???] Photosynthesis rate
        Pgmax   [???] Maximum photosynthesis rate
        LUE       [???] light use efficiency factor
        LUE340    [???] LUE when 340ppm
        Iabs    [???] Light intensity absorbed by plants
        Rec_N    [-] fraction of N fertilizer recovery
        Rec_P    [-] fraction of P fertilizer recovery
        
        ======  =============================================================
           

   d : dictionary of floats or arrays
       Model disturbances (required for method 'run'),
       of shape (len(t_d),2) for time and disturbance.
       
       =======  ============================================================
       key      meaning
       =======  ============================================================
        'I0'     [...] sun irradiance
        'T'      [°C] Air temperature
        'CO2'    [...] carbon dioxide concentration 
       =======  ============================================================

   u : dictionary of 2D arrays
       Controlled inputs (required for method 'run'),
       of shape (len(t_d),2) for time and controlled input.
       
       =======  ============================================================
       key      meaning
       =======  ============================================================
        'I_N'    [kg/ha] N concentration in fertilizer
        'I_P'    [kg/ha] P concentration in fertilizer
       =======  ============================================================ 

   Returns
   -------
   y : dictionary
       Model outputs as 1D arrays (Mrt, Mst, Mlv, Mpa, Mgr)
       and the evaluation time 't'.
   """ 
   def __init__(self, tsim, dt, x0, p):
       Module.__init__(self, tsim, dt, x0, p)
       # Initialize dictionary of flows
       self.f = {}
       self.f_keys = ('f_Ph', 'f_Nlv', 'f_cgr', 'f_res', 'f_gr', 
                      'f_dmv', 'f_pN', 'HU', 'DVS', 'N_lv', 'Mrt', 'Mst', 
                      'Mlv', 'Mpa', 'Mgr')
       for k in self.f_keys:
           self.f[k] = np.full((self.t.size,), np.nan)           
   
   def diff(self, _t, _x0):
       # -- Initial conditions
 
       Mrt = _x0[0]
       Mst = _x0[1]
       Mlv = _x0[2]
       Mpa = _x0[3]
       Mgr = _x0[4]
       HU = _x0[5]
       DVS = _x0[6]
       
       # -- Model parameters
       #physical parameters
       Tref = 25                # [°C] reference temperature
       # - rice plants
       DVSi = self.p['DVSi']    # [°Cd-1] initial DVS
       DVRJ = self.p['DVRJ']    # [°Cd-1] Development rate in Juvenile Phase ( 0 <=DVS < 4)
       DVRI = self.p['DVRI']    # [°Cd-1] Development rate in Active tillering Phase ( 0.4 <=DVS < 0.65)
       DVRP = self.p['DVRP']    # [°Cd-1] Development rate in Panicle Development Phase ( 0.65 <=DVS < 1)
       DVRR = self.p['DVRR']    # [°Cd-1] Development rate in Reproductive Phase ( DVS >= 1)
       Tmax = self.p['Tmax']    # [°C] maximum temperature for growth
       Tmin = self.p['Tmin']    # [°C] minimum temperature for growth
       Topt = self.p['Topt']    # [°C] optimum temperature for growth
       k = self.p['k']          # [-] leaf extinction coefficient (value= 0.4)
       mc_rt = self.p['mc_rt']  # [-] maintenance respiration coefficient of roots (0.01)
       mc_st = self.p['mc_st']  # [-] maintenance respiration coefficient of stems (0.015)
       mc_lv = self.p['mc_lv']  # [-] maintenance respiration coefficient of leaves (0.02)
       mc_pa = self.p['mc_pa']  # [-] maintenance respiration coefficient of panicles (0.003)
       
       #carbohydrate requirements allocated
       cr_lv = self.p['cr_lv'] #leaves
       cr_st = self.p['cr_st'] #stems
       cr_pa = self.p['cr_pa'] #panicles
       cr_rt = self.p['cr_rt'] #roots
       
       #-N dynamics
       N_lv = self.p['N_lv']    # [kg N ha-1 d-1] amount of N in leaves
       IgN = self.p['IgN']      #[kg N ha-1 d-1] Indigenous Soil N supply
       Rec_N = self.p['Rec_N']    # [-] N fertilizers recovery coefficient
       fac = 0      # [-] fertilizer application coefficient, default 0 (no application)
       k_pa_maxN = self.p['k_pa_maxN'] #[kg N kg DM-1] maximum N content in panicles (0.0175)
       M_upN = self.p['M_upN'] #[kg N ha-1 d-1] maximum N uptake by plants (8, range 8 - 12)
       
       # -- Disturbances at instant _t
       I0 = self.d['I0']
       T = self.d['T']
       CO2 = self.d['CO2']
       
       #disturbances within the system
       S_NO3 = self.d['S_NO3']
       S_P = self.d['S_P']
       f_P_sol = self.d['f_P_sol']
       
       _I0 = np.interp(_t,I0[:,0],I0[:,1])     # [J m-2 d-2] PAR
       _T = np.interp(_t,T[:,0],T[:,1])        # [°C] Environment temperature
       _CO2 = np.interp(_t,CO2[:,0],CO2[:,1])  # [ppm] Atmospheric CO2 concentration
       _S_NO3 = np.interp(_t, S_NO3[:,0], S_NO3[:,1]) #[g d-1] Nitrate concentration
       _f_P_sol = np.interp(_t, f_P_sol[:,0], f_P_sol[:,1])
       _S_P = np.interp(_t, S_P[:,0], S_P[:,1])
       
       f_uptN = _S_NO3
       f_uptP = _f_P_sol + _S_P
       
       # -- Controlled inputs
       I_N = self.u['I_N'] #N concentration in fertilizer
       I_P = self.u['I_P'] #N concentration in fertilizer
       
       #data for interpolation
       data_rice = 'C:/Users/alegn/Documents/WUR/Thesis/rfmodel/data/data_rice.csv'
       
       #Data of N fraction in leaves [g N m-2] as a function of DVS
       DVS_k_lv_N_area =pd.read_csv(data_rice, usecols=[0, 1], header=1, sep=';')

       #data of SLA values based on DVS
       DVS_SLA = pd.read_csv(data_rice, usecols=[2,3], header=1, sep=';')
       
       #data of LUE values based on Temperature
       T_LUE = pd.read_csv(data_rice, usecols=[4,5], header=1, sep=';')
       
       #data of fraction of max N in leaves based on weight basis as a function of DVS
       DVS_k_lv_maxN = pd.read_csv(data_rice, usecols=[6,7], header=1, sep=';')
              
       # - fertilizer application, monoculture
       # if _t == 12 or _t==30 or _t==67:
       #     fac = 1
       # elif _t== 95:
       #     fac = 1
       #     I_N = 45
       # else:
       #     fac = 0
           
       # - fertilizer application, minapadi
       if _t == 12:
           fac = 1
       else:
           fac = 0
           
       # - Heat Units
       #Tavg = (np.min(_T)+np.max(_T))/2
       Tavg = _T
       
       # Previous HU value
       
       if Tmin < Tavg <= Topt:
           dHU_dt= (Tavg - Tmin)/24
       elif Tavg <= Tmin:
           dHU_dt = 0
       else:
          dHU_dt = (Topt - ((Tavg - Topt)*(Topt-Tmin)/(Tmax-Topt)))/24 
           
       # # - Developmental stage (DVS)
       # Calculate DVS based on previous value and HU
       
       # dDVS_dt = DVSi*HU
       DVS = DVSi*HU
       
       DVSi = DVRJ if 0 <= DVS < 0.4 else DVRI if 0.4 <= DVS < 0.65 else DVRP if 0.65 <= DVS < 1 else DVRR
        
       # - Dry Matter Partitioning
       # according to Bouwman 2001
       # m_rt = 0.5 if 0 <= DVS < 0.4 else 0.25 if 0.4 <= DVS < 1 else 0 if DVS >= 1 else 0
       # m_st = 0.3 if 0 <= DVS < 1 else 0.4 if 1 <= DVS <2 else 0.3
       # m_lv = 0.2 if 0 <= DVS < 0.4 else 0.3 if 0.4 <= DVS < 1 else 0
       # m_pa = 0 if DVS < 1 else 0.6 if 1<= DVS < 2 else 1.0
       
       # According to Agustiani et al 2019
       m_rt = 0.39 if 0 <= DVS < 0.4 else 0.26 if 0.4 <= DVS < 0.65 else 0.0 
       m_st = 0.2 if 0 <= DVS < 0.4 else 0.43 if  0.4 <= DVS < 0.65 else 0.0
       m_lv = 0.41 if 0 <= DVS < 0.4 else 0.31 if 0.4 <= DVS < 0.65 else 0
       m_pa = 0 if 0 <= DVS < 0.65 else 1 
       
       k = 0.4 if 0 <= DVS < 1 else 0.6 
       
       Rec_N = 0.3 if 0 <= DVS < 0.4 else 0.5 if 0.4 <= DVS < 1 else 0.75
       
       d_lv = 0 if 0 <=DVS < 0.4 else 0.015 if 1<= DVS < 2 else 0.05  
                 
       print("DVS: ", DVS, " DVSi :", DVSi, " m_rt :", m_rt, ' m_st: ', m_st, ' m_lv: ', m_lv, 
             " m_pa: ", m_pa, ' k: ', k, 'Rec_N: ', Rec_N, 'd_lv: ', d_lv)
       
       # Convert dictionary to lists
       DVS_values = DVS_SLA.iloc[:,0].values
       SLA_values = DVS_SLA.iloc[:,1].values
        
       # Create an interpolation function
       SLA_func = interp1d(DVS_values, SLA_values, kind='linear')
             
       # Interpolate to get the actual SLA based on current DVS
       SLA = SLA_func(DVS) if DVS <=2.5 else 0.0
       
       # - Leaf Area Index (LAI)
       LAI = SLA*Mlv   # [m2 m-2] Leaf area index
       
       # -- Supporting equations
       # -- Nitrogen Dynamics
       
       # - to calculate N uptake flow by leaves
       # - maximum N demand of leaves
       DVS_k_lv_maxN_values = DVS_k_lv_maxN.iloc[:, 0].values
       k_lv_maxN_values = DVS_k_lv_maxN.iloc[:, 1].values
        
       # Create an interpolation function for 
       k_lv_maxN_func = interp1d(DVS_k_lv_maxN_values, k_lv_maxN_values, kind='linear')
             
       # Interpolate to get the actual k_lv_maxN based on current DVS
       k_lv_maxN = k_lv_maxN_func(DVS) if DVS <=2.5 else 0.0
       
       MD_lv_N = k_lv_maxN*Mlv
       if MD_lv_N <=0:
           MD_lv_N = 0
       
       # - maximum N demand of stems
       MD_st_N = k_lv_maxN*0.5*Mst
       if MD_st_N <=0:
           MD_st_N = 0
           
       # - maximum N demand of panicles
       MD_pa_N = k_pa_maxN*Mpa
             
       # - N uptake by crop organs
       f_pN = min(M_upN, (f_uptN + I_N*Rec_N*fac + IgN))
       
       #max of total potential N demand from leaves, stems, and panicles
       MDcropN = MD_lv_N + MD_st_N + MD_pa_N 
       
       #N uptake of the leaves
       f_Nlv = max(0, min(MD_lv_N, f_pN*MD_lv_N/MDcropN))
      
       #Actual N content in leaves
       N_lv = f_Nlv/Mlv
       
       # calculate Maximum N content of leaves on a leaf area basis
       DVS_k_lv_N_area_values = DVS_k_lv_N_area.iloc[:,0].values
       k_lv_N_area_values = DVS_k_lv_N_area.iloc[:,1].values
       
       # Create an interpolation function for fraction of N in leaves
       k_lv_N_area_func = interp1d(DVS_k_lv_N_area_values, k_lv_N_area_values, kind='linear')
             
       # Interpolate to get the actual k_lv_N_area based on current DVS [g N m-2 leaf]
       k_lv_N_area = k_lv_N_area_func(DVS) if DVS <=2.5 else 0.0
              
       PgmaxN = (N_lv/k_lv_maxN)*k_lv_N_area
            
       #- - Photosynthesis                      
       # - Maximum photosynthesis rate
       
       PgmaxCO2 = (49.57/34.26)*(1-np.exp(-0.208*(_CO2-60)/49.57))
       
       Pgmax = 9.5+(22*0.314*PgmaxN*PgmaxCO2)
       #0.314 is converter of g N/m2 to kg CO2/ha
       
       # - Initial LUE
       LUE340 = (1- np.exp(-0.00305*_CO2-0.222))/(1- np.exp(-0.00305*340-0.222))

       # Convert dictionary to lists
       T_values = T_LUE.iloc[:,0].values
       LUE_values = T_LUE.iloc[:,1].values

       # Create an interpolation function
       LUE_func = interp1d(T_values, LUE_values, kind='linear')

       # Interpolate the values
       LUE = LUE_func(Tavg)*LUE340
       
       # - Solar radiation absorbed (Iabs)
       Iabs = 0.50*(_I0*(1-np.exp(-k*LAI)))
       
       # - Photosynthesis rate
       f_Ph = Pgmax*(1-np.exp(-LUE*(Iabs/Pgmax)))
       
       # - maintenance respiration
       Teff = 2**((Tavg-Tref)/10)
       f_res = (mc_rt*Mrt+mc_st*Mst+mc_lv*Mlv+mc_pa*Mpa)*Teff
       
       # - growth respiration
       f_gr = (m_lv+m_st+m_pa)*(cr_lv*m_lv+cr_st*m_st+cr_pa*m_pa)+cr_rt*m_rt
       
       # - leaves death rate
       f_dmv = d_lv*Mlv
       
       # - Crop growth rate
       f_cgr = (f_Ph*(30/44)-f_res)/f_gr
            
       
       # -- Differential equations [kgDM ha-1 d-1]
       
       dMrt_dt = m_rt*f_cgr  
       dMst_dt = m_st*f_cgr 
       dMlv_dt = m_lv*f_cgr
       
       # if DVS>=1:
       #     dMlv_dt = m_lv*m_sh*f_cgr - f_dmv 
       # dMpa_dt = 0
       # if DVS>= 0.65:
       #     dMpa_dt = m_pa*m_sh*f_cgr
       # dMgr_dt = 0
       # if DVS >= 0.95:
       #     dMgr_dt = dMpa_dt
       
       if DVS>=1:
           dMlv_dt = m_lv*f_cgr - f_dmv 
       dMpa_dt = 0
       if DVS>= 0.65:
           dMpa_dt = m_pa*f_cgr
       dMgr_dt = 0
       if DVS >= 0.95:
           dMgr_dt = Mpa
       
       dDVS_dt = DVS
       Mrt = dMrt_dt 
       Mst = dMst_dt
       Mlv = dMlv_dt
       Mpa = dMpa_dt
       
       # -- Store flows [kgC m-2 d-1]
       idx = np.isin(self.t, _t)
       self.f['f_Ph'][idx] = f_Ph
       self.f['f_res'][idx] = f_res
       self.f['f_gr'][idx] = f_gr
       self.f['f_cgr'][idx] = f_cgr
       self.f['f_pN'][idx] = f_pN
       self.f['f_Nlv'][idx] = f_Nlv
       self.f['f_dmv'][idx] = f_dmv
       self.f['HU'][idx] = HU
       self.f['DVS'][idx] = DVS
       self.f['N_lv'][idx] = N_lv
       self.f['Mrt'][idx] = Mrt
       self.f['Mst'][idx] = Mst
       self.f['Mlv'][idx] = Mlv
       self.f['Mpa'][idx] = Mpa
       self.f['Mgr'][idx] = Mgr
       
       return np.array([dMrt_dt,
                        dMst_dt, 
                        dMlv_dt, 
                        dMpa_dt, 
                        dMgr_dt,
                        dHU_dt,
                        dDVS_dt
                        ])
   
   def output(self, tspan):
       # Retrieve object properties
       dt = self.dt        # integration time step size
       diff = self.diff    # function with system of differential equations
       Mrt0 = self.x0['Mrt'] # initial condition
       Mst0 = self.x0['Mst'] # initial condiiton
       Mlv0 = self.x0['Mlv']
       Mpa0 = self.x0['Mpa']
       Mgr0 = self.x0['Mgr']
       HU0 = self.x0['HU']
       DVS0 = self.x0['DVS']
       
       # Numerical integration
       y0 = np.array([Mrt0, 
                      Mst0, 
                      Mlv0, 
                      Mpa0, 
                      Mgr0,
                      HU0,
                      DVS0
                      ])
       
       # y_int = fcn_euler_forward(diff,tspan,y0,h=dt)
       y_int = fcn_rk4(diff, tspan, y0, h=dt)
       
       # Model results
       t = y_int['t']
       Mrt = y_int['y'][0,:]
       Mst = y_int['y'][1,:]
       Mlv = y_int['y'][2,:]
       Mpa = y_int['y'][3,:]
       Mgr = y_int['y'][4,:]
       HU =  y_int['y'][5,:]
       DVS = y_int['y'][6,:]
       
       return {
           't':t,          # [d] Integration time
           'Mrt': Mrt,
           'Mst': Mst,
           'Mlv': Mlv,
           'Mpa': Mpa,
           'Mgr': Mgr,
           'HU': HU,
           'DVS': DVS
       }
