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
from models.integration import fcn_euler_forward
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
       self.f_keys = ('f_Ph', 'f_Nlv', 'f_cgr', 'f_res', 'f_gr', 'f_uptN', 
                      'f_dmv', 'f_pN', 'HU', 'N_lv', 'DVS', 'LAI', 'f_spi', 'NumGrain')
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
       # Nfert_in = _x0[7] #[kg/ha] N content in inoganic fertilizers       
       
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
       Rm_rt = self.p['Rm_rt']  # [-] maintenance respiration coefficient of roots (0.01)
       Rm_st = self.p['Rm_st']  # [-] maintenance respiration coefficient of stems (0.015)
       Rm_lv = self.p['Rm_lv']  # [-] maintenance respiration coefficient of leaves (0.02)
       Rm_pa = self.p['Rm_pa']  # [-] maintenance respiration coefficient of panicles (0.003)
       n_rice = self.p['n_rice'] #[plants m-2] number of plants per square meter
       # gamma = self.p['gamma']  # [-] Spikelet growth factor (65 number of spikelet kg-1, 45 - 70 depend on the varieties)
       
       #carbohydrate requirements allocated
       cr_lv = self.p['cr_lv'] #leaves
       cr_st = self.p['cr_st'] #stems
       cr_pa = self.p['cr_pa'] #panicles
       cr_rt = self.p['cr_rt'] #roots
           
       # -- Disturbances at instant _t
       I0 = self.d['I0']
       T = self.d['T']
       CO2 = self.d['CO2']
       
       #disturbances within the system
       f_N_plt_upt = self.d['f_N_plt_upt']
       # f_P_plt_upt = self.d['f_P_plt_upt']
       
       _I0 = np.interp(_t,I0[:,0],I0[:,1])     # [J m-2 d-2] PAR
       _T = np.interp(_t,T[:,0],T[:,1])        # [°C] Environment temperature
       _CO2 = np.interp(_t,CO2[:,0],CO2[:,1])  # [ppm] Atmospheric CO2 concentration
      
       
       #-N dynamics
       
       if _t >= 21*24:
           N_lv = self.p['N_lv']    # [kg N ha-1 d-1] amount of N in leaves
           k_pa_maxN = self.p['k_pa_maxN'] #[kg N kg DM-1] maximum N content in panicles (0.0175)
           _f_N_plt_upt = np.interp(_t, f_N_plt_upt[:,0], f_N_plt_upt[:,1]) #[g d-1] N uptake by rice plants
           # _f_P_plt_upt = np.interp(_t, f_P_plt_upt[:,0], f_P_plt_upt[:,1]) #[g d-1] P uptake by rice plants
           
       
       else:
           N_lv = self.p['N_lv']   # [kg N ha-1 d-1] amount of N in leaves
           k_pa_maxN = self.p['k_pa_maxN']
           _f_N_plt_upt = 0 #[g m-3 d-1] Nitrate concentration
           # _f_P_plt_upt = 0
           _I0 = _I0*0.50
           _T = _T*0.80
       
       print('NO3: ', _f_N_plt_upt)
       f_uptN = _f_N_plt_upt/1000 #[kg d-1] convert the nutrient source to kg
                   
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
       
       # Previous HU value
       Tavg = _T
       if Tmin < Tavg <= Topt:
           dHU_dt=Tavg - Tmin
       elif Tavg <= Tmin:
           dHU_dt = 0
       else:
          dHU_dt = Topt - ((Tavg - Topt)*(Topt-Tmin)/(Tmax-Topt)) 
           
       # # - Developmental stage (DVS)
       # Calculate DVS based on previous value and HU
       DVSi = DVRJ if 0 <= DVS < 0.4 else DVRI if 0.4 <= DVS < 0.65 else DVRP if 0.65 <= DVS < 1 else DVRR
       DVS = DVSi*HU
       dDVS_dt = DVS
       
       # - Dry Matter Partitioning
        # according to Bouwman 2001
       m_rt = 0.5 if 0 <= DVS < 0.4 else 0.25 if 0.4 <= DVS < 1 else 0 if DVS >= 1 else 0
       m_st = 0.3 if 0 <= DVS < 1 else 0.4 if 1 <= DVS <2 else 0.3
       m_lv = 0.2 if 0 <= DVS < 0.4 else 0.3 if 0.4 <= DVS < 1 else 0
       m_pa = 0 if DVS < 1 else 0.6 if 1<= DVS < 2 else 1.0
       
       # According to Agustiani et al 2019
       # m_rt = 0.39 if 0 <= DVS < 0.4 else 0.26 if 0.4 <= DVS < 0.65 else 0.0 
       # m_st = 0.2 if 0 <= DVS < 0.4 else 0.43 if  0.4 <= DVS < 0.65 else 0.0
       # m_lv = 0.41 if 0 <= DVS < 0.4 else 0.31 if 0.4 <= DVS < 0.65 else 0
       # m_pa = 0 if 0 <= DVS < 0.65 else 1 
       
       k = 0.4 if 0 <= DVS < 1 else 0.6 
       
       Rec_N = 0.3 if 0 <= DVS < 0.4 else 0.5 if 0.4 <= DVS < 1 else 0.75
       
       d_lv = 0 if 0 <=DVS < 1 else 0.015 if 1<= DVS < 2 else 0.05  
       
       # spi = 0 if 0 <=DVS <0.65 or DVS >=1 else 1
       
       # ACDD = 0 #Accumulated Cold Degree Days
       # SPF = 0
       # if 0.75 <= DVS < 1.2:
       #     #grain formation
       #     CDD = max(0, 22-Tavg)
       #     ACDD = ACDD + CDD
       #     SF1 = 1 - (4.6+ 0.054*(ACDD)**1.56)/100
       #     SF1 = min(1, max(0,SF1))
       #     SF2 = 1/(1+np.exp(0.853*(Tavg-36.6)))
       #     SF2 = min(1,max(0,SF2))
       #     SPF = min(SF1, SF2)
                 
       # print("DVS: ", DVS, " DVSi :", DVSi, " m_rt :", m_rt, ' m_st: ', m_st, ' m_lv: ', m_lv, 
       #       " m_pa: ", m_pa, ' k: ', k, 'Rec_N: ', Rec_N, 'd_lv: ', d_lv)
       
       # Convert dictionary to lists
       DVS_values = DVS_SLA.iloc[:,0].values
       SLA_values = DVS_SLA.iloc[:,1].values
        
       # Create an interpolation function
       SLA_func = interp1d(DVS_values, SLA_values, kind='linear')
             
       # Interpolate to get the actual SLA based on current DVS
       SLA = SLA_func(DVS) if DVS <=2.5 else 0.0
       print('SLA: ', SLA)
       # - Leaf Area Index (LAI)
       LAI = SLA*Mlv*n_rice  # [m2 m-2] Leaf area index
       
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
       f_pN = f_uptN*Rec_N
       
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
              
       PgmaxN = (N_lv/k_lv_maxN)*k_lv_N_area #[g N m-2 leaf]
            
       #- - Photosynthesis                      
       # - Maximum photosynthesis rate
       
       PgmaxCO2 = (49.57/34.26)*(1-np.exp(-0.208*(_CO2-60)/49.57)) #[]
       
       Pgmax = 9.5+(22*PgmaxN*0.314*PgmaxCO2) #[kg CO2/ha/d]
       #0.314 is converter of g N/m2 to kg CO2/ha
       
       # - Initial LUE
       LUE340 = (1- np.exp(-0.00305*_CO2-0.222))/3600*(1- np.exp(-0.00305*340-0.222))

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
       if 0 <= _t <=5*24:
           f_Ph = 0
       else:
           f_Ph = Pgmax*(1-np.exp(-LUE*(Iabs/Pgmax)))
       
       # - maintenance respiration
       Teff = 2**((Tavg-Tref)/10)
       f_res = (Rm_rt*Mrt+Rm_st*Mst+Rm_lv*Mlv+Rm_pa*Mpa)*Teff
       
       # - growth respiration
       f_gr = (m_lv+m_st+m_pa)*(cr_lv*m_lv+cr_st*m_st+cr_pa*m_pa)+cr_rt*m_rt
       
       # - leaves death rate
       f_dmv = d_lv*Mlv
       
       # - Crop growth rate
       f_cgr = (f_Ph*(30/44)-f_res)/f_gr
            
       # #spikelet formation
       # f_spi = f_cgr*gamma*spi #[number of spikelet formed d-1]
       # NumGrain = f_spi*SPF
       
       # -- Differential equations [kgDM ha-1 d-1]
       
       dMrt_dt = m_rt*f_cgr  
       dMst_dt = m_st*f_cgr 
       dMlv_dt = m_lv*f_cgr
      
       if DVS>=1:
           dMlv_dt = m_lv*f_cgr - f_dmv 
       dMpa_dt = 0
       if DVS>= 0.65:
            dMpa_dt = m_pa*f_cgr
       dMgr_dt = 0
       if DVS >= 0.95:
           dMgr_dt = Mpa
       
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
       self.f['N_lv'][idx] = N_lv
       self.f['DVS'][idx] = DVS
       self.f['LAI'][idx] = LAI
       self.f['f_uptN'][idx] = f_uptN

       return np.array([dMrt_dt,
                        dMst_dt, 
                        dMlv_dt, 
                        dMpa_dt, 
                        dMgr_dt,
                        dHU_dt,
                        dDVS_dt,

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

       DVSi = self.p['DVSi']
       DVRJ = self.p['DVRJ']
       DVRI = self.p['DVRI']
       DVRP = self.p['DVRP']
       DVRR = self.p['DVRR']
       
       
       # Numerical integration
       y0 = np.array([Mrt0, 
                      Mst0, 
                      Mlv0, 
                      Mpa0, 
                      Mgr0,
                      HU0,
                      DVS0,
                      # Nfert_in0
                      ])
       
       y_int = fcn_euler_forward(diff,tspan,y0,h=dt)

       # Model results
       t = y_int['t']
       Mrt = y_int['y'][0,:]
       Mst = y_int['y'][1,:]
       Mlv = y_int['y'][2,:]
       Mpa = y_int['y'][3,:]
       Mgr = y_int['y'][4,:]
       HU =  y_int['y'][5,:]
       DVSc = y_int['y'][6,:]
       DVSi = np.where(DVSc < 0.4, DVRJ, 
                np.where(DVSc < 0.65, DVRI, 
                         np.where(DVSc < 1, DVRP, DVRR)))
       DVS = DVSi*HU
       if DVS[-1].any() < DVS[-2]:
           DVS[-1] = DVS[-2]
           
       return {
           't':t,          # [d] Integration time
           'Mrt': Mrt,
           'Mst': Mst,
           'Mlv': Mlv,
           'Mpa': Mpa,
           'Mgr': Mgr,
           'HU': HU,
           'DVS': DVS,

       }

class Rice_hourly(Module):
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
       self.f_keys = ('f_Ph', 'f_Nlv', 'f_cgr', 'f_res', 'f_gr', 'f_uptN', 
                      'f_dmv', 'f_pN', 'HU', 'N_lv', 'DVS', 'LAI', 'f_spi', 'NumGrain')
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
       # Nfert_in = _x0[7] #[kg/ha] N content in inoganic fertilizers       
       
       # -- Model parameters
       #physical parameters
       Tref = 25                # [°C] reference temperature
       # - rice plants
       DVRi = self.p['DVRi']    # [°Cd-1] initial DVS
       DVRJ = self.p['DVRJ']    # [°Cd-1] Development rate in Juvenile Phase ( 0 <=DVS < 4)
       DVRI = self.p['DVRI']    # [°Cd-1] Development rate in Active tillering Phase ( 0.4 <=DVS < 0.65)
       DVRP = self.p['DVRP']    # [°Cd-1] Development rate in Panicle Development Phase ( 0.65 <=DVS < 1)
       DVRR = self.p['DVRR']    # [°Cd-1] Development rate in Reproductive Phase ( DVS >= 1)
       Tmax = self.p['Tmax']    # [°C] maximum temperature for growth
       Tmin = self.p['Tmin']    # [°C] minimum temperature for growth
       Topt = self.p['Topt']    # [°C] optimum temperature for growth
       k = self.p['k']          # [-] leaf extinction coefficient (value= 0.4)
       Rm_rt = self.p['Rm_rt']  # [-] maintenance respiration coefficient of roots (0.01)
       Rm_st = self.p['Rm_st']  # [-] maintenance respiration coefficient of stems (0.015)
       Rm_lv = self.p['Rm_lv']  # [-] maintenance respiration coefficient of leaves (0.02)
       Rm_pa = self.p['Rm_pa']  # [-] maintenance respiration coefficient of panicles (0.003)
       n_rice = self.p['n_rice'] #[plants m-2] number of plants per square meter
       # gamma = self.p['gamma']  # [-] Spikelet growth factor (65 number of spikelet kg-1, 45 - 70 depend on the varieties)
       
       #carbohydrate requirements allocated
       cr_lv = self.p['cr_lv'] #leaves
       cr_st = self.p['cr_st'] #stems
       cr_pa = self.p['cr_pa'] #panicles
       cr_rt = self.p['cr_rt'] #roots
           
       # -- Disturbances at instant _t
       I0 = self.d['I0']
       Th = self.d['Th']
       # Tday = self.d['Tday']
       # Tavg = self.d['Tavg']
       CO2 = self.d['CO2']
       
       #disturbances within the system
       f_N_plt_upt = self.d['f_N_plt_upt']
       # f_P_plt_upt = self.d['f_P_plt_upt']
       
       _I0 = np.interp(_t,I0[:,0],I0[:,1])     # [J m-2 d-2] PAR
       _Th = np.interp(_t,Th[:,0],Th[:,1])        # [°C h-1] Hourly temperature
       # _Tday = np.interp(_t,Tday[:,0],Tday[:,1])        # [°C] Average daytime temperature
       # _Tavg = np.interp(_t,Tavg[:,0],Tavg[:,1])        # [°C] Average daily temperature
       _CO2 = np.interp(_t,CO2[:,0],CO2[:,1])  # [ppm] Atmospheric CO2 concentration
      
       
       #-N dynamics
       if _t >= 21*24:
           N_lv = self.p['N_lv']    # [kg N ha-1 d-1] amount of N in leaves
           k_pa_maxN = self.p['k_pa_maxN'] #[kg N kg DM-1] maximum N content in panicles (0.0175)
           _f_N_plt_upt = np.interp(_t, f_N_plt_upt[:,0], f_N_plt_upt[:,1]) #[g d-1] N uptake by rice plants
           # _f_P_plt_upt = np.interp(_t, f_P_plt_upt[:,0], f_P_plt_upt[:,1]) #[g d-1] P uptake by rice plants
           
       
       else:
           N_lv = 0   # [kg N ha-1 d-1] amount of N in leaves
           k_pa_maxN = 0
           _f_N_plt_upt = 0 #[g m-3 d-1] Nitrate concentration
           # _f_P_plt_upt = 0
           _I0 = _I0*0.50
           _Th = _Th*0.80
       
       print('NO3: ', _f_N_plt_upt)
       f_uptN = _f_N_plt_upt/1000 #[kg d-1] convert the nutrient source to kg
                   
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
       
       # Previous HU value
       #include Tavg, Tmax, and Tmin of each hour? or day?
       # T_high = self.d['T_high'] #[C] daily minimum temperature
       # T_low = self.d['T_low'] #[C] dailty maximum temperature
       # h = self.d['h'] #[h] time of the day
       #Th = (T_low+T_high)/2 + (T_high+Tmin)*np.cos(0.2618*(h-14))/2
       
       if Tmin < Th <= Topt:
           dHU_dt= (Th - Tmin)/24 #[h] hourly HU
       elif Th <= Tmin or Th > Tmax:
           dHU_dt = 0
       else:
          dHU_dt = (Topt - ((Th - Topt)*(Topt-Tmin)/(Tmax-Topt)))/24 #[h] hourly HU 
           
       # # - Developmental stage (DVS)
       # Calculate DVS based on previous value and HU
       DVRi = DVRJ if 0 <= DVS < 0.4 else DVRI if 0.4 <= DVS < 0.65 else DVRP if 0.65 <= DVS < 1 else DVRR
       DVS = DVRi*HU
       dDVS_dt = DVS
       
       # - Dry Matter Partitioning
        # according to Bouwman 2001
       m_rt = 0.5 if 0 <= DVS < 0.4 else 0.25 if 0.4 <= DVS < 1 else 0 if DVS >= 1 else 0
       m_st = 0.3 if 0 <= DVS < 1 else 0.4 if 1 <= DVS <2 else 0.3
       m_lv = 0.2 if 0 <= DVS < 0.4 else 0.3 if 0.4 <= DVS < 1 else 0
       m_pa = 0 if DVS < 1 else 0.6 if 1<= DVS < 2 else 1.0
       
       # According to Agustiani et al 2019
       # m_rt = 0.39 if 0 <= DVS < 0.4 else 0.26 if 0.4 <= DVS < 0.65 else 0.0 
       # m_st = 0.2 if 0 <= DVS < 0.4 else 0.43 if  0.4 <= DVS < 0.65 else 0.0
       # m_lv = 0.41 if 0 <= DVS < 0.4 else 0.31 if 0.4 <= DVS < 0.65 else 0
       # m_pa = 0 if 0 <= DVS < 0.65 else 1 
       
       k = 0.4 if 0 <= DVS < 1 else 0.6 
       
       Rec_N = 0.3 if 0 <= DVS < 0.4 else 0.5 if 0.4 <= DVS < 1 else 0.75
       
       d_lv = 0 if 0 <=DVS < 1 else 0.015 if 1<= DVS < 2 else 0.05  
       
       # spi = 0 if 0 <=DVS <0.65 or DVS >=1 else 1
       
       # ACDD = 0 #Accumulated Cold Degree Days
       # SPF = 0
       # if 0.75 <= DVS < 1.2:
       #     #grain formation
       #     CDD = max(0, 22-Tavg)
       #     ACDD = ACDD + CDD
       #     SF1 = 1 - (4.6+ 0.054*(ACDD)**1.56)/100
       #     SF1 = min(1, max(0,SF1))
       #     SF2 = 1/(1+np.exp(0.853*(Tavg-36.6)))
       #     SF2 = min(1,max(0,SF2))
       #     SPF = min(SF1, SF2)
                 
       # print("DVS: ", DVS, " DVSi :", DVSi, " m_rt :", m_rt, ' m_st: ', m_st, ' m_lv: ', m_lv, 
       #       " m_pa: ", m_pa, ' k: ', k, 'Rec_N: ', Rec_N, 'd_lv: ', d_lv)
       
       # Convert dictionary to lists
       DVS_values = DVS_SLA.iloc[:,0].values
       SLA_values = DVS_SLA.iloc[:,1].values
        
       # Create an interpolation function
       SLA_func = interp1d(DVS_values, SLA_values, kind='linear')
             
       # Interpolate to get the actual SLA based on current DVS
       SLA = SLA_func(DVS) if DVS <=2.5 else 0.0
       print('SLA: ', SLA)
       # - Leaf Area Index (LAI)
       LAI = SLA*Mlv*n_rice  # [m2 m-2] Leaf area index
       
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
       f_pN = f_uptN*Rec_N
       
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
              
       PgmaxN = (N_lv/k_lv_maxN)*k_lv_N_area #[g N m-2 leaf]
            
       #- - Photosynthesis  
       #Differentiate between average daily temperature (Tavg) and average daytime temperature (Tday)
       # - Maximum photosynthesis rate
       
       PgmaxCO2 = (49.57/34.26)*(1-np.exp(-0.208*(_CO2-60)/49.57)) #[]
       
       Pgmax = 9.5+(22*PgmaxN*0.314*PgmaxCO2) #[kg CO2/ha/d]
       #0.314 is converter of g N/m2 to kg CO2/ha
       
       # - Initial LUE
       LUE340 = (1- np.exp(-0.00305*_CO2-0.222))/3600*(1- np.exp(-0.00305*340-0.222))

       # Convert dictionary to lists
       T_values = T_LUE.iloc[:,0].values
       LUE_values = T_LUE.iloc[:,1].values

       # Create an interpolation function
       LUE_func = interp1d(T_values, LUE_values, kind='linear')

       # Interpolate the values
       # LUE = LUE_func(_Tday)*LUE340
       LUE = LUE_func(_Th)*LUE340 
       
       # - Solar radiation absorbed (Iabs)
       Iabs = 0.50*(_I0*(1-np.exp(-k*LAI)))
       
       # - Photosynthesis rate
       # photosynthesis hasn't happened until rice seedlings develop their first true leaves, generally on day 5
       if 0 <= _t <=5*24:
           f_Ph = 0
       else:
           f_Ph = Pgmax*(1-np.exp(-LUE*(Iabs/Pgmax)))
       
       # - maintenance respiration
       # Teff = 2**((_Tavg-Tref)/10)
       Teff = 2**((_Th-Tref)/10)
       f_res = (Rm_rt*Mrt+Rm_st*Mst+Rm_lv*Mlv+Rm_pa*Mpa)*Teff
       
       # - growth respiration
       f_gr = (m_lv+m_st+m_pa)*(cr_lv*m_lv+cr_st*m_st+cr_pa*m_pa)+cr_rt*m_rt
       
       # - leaves death rate
       f_dmv = d_lv*Mlv
       
       # - Crop growth rate
       f_cgr = (f_Ph*(30/44)-f_res)/f_gr
            
       # #spikelet formation
       # f_spi = f_cgr*gamma*spi #[number of spikelet formed d-1]
       # NumGrain = f_spi*SPF
       
       # -- Differential equations [kgDM ha-1 d-1]
       
       dMrt_dt = m_rt*f_cgr  
       dMst_dt = m_st*f_cgr 
       dMlv_dt = m_lv*f_cgr
      
       if DVS>=1:
           dMlv_dt = m_lv*f_cgr - f_dmv 
       dMpa_dt = 0
       if DVS>= 0.65:
            dMpa_dt = m_pa*f_cgr
       dMgr_dt = 0
       if DVS >= 0.95:
           dMgr_dt = Mpa
       
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
       self.f['N_lv'][idx] = N_lv
       self.f['DVS'][idx] = DVS
       self.f['LAI'][idx] = LAI
       self.f['f_uptN'][idx] = f_uptN

       return np.array([dMrt_dt,
                        dMst_dt, 
                        dMlv_dt, 
                        dMpa_dt, 
                        dMgr_dt,
                        dHU_dt,
                        dDVS_dt,

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

       DVRi = self.p['DVRi']
       DVRJ = self.p['DVRJ']
       DVRI = self.p['DVRI']
       DVRP = self.p['DVRP']
       DVRR = self.p['DVRR']
       
       
       # Numerical integration
       y0 = np.array([Mrt0, 
                      Mst0, 
                      Mlv0, 
                      Mpa0, 
                      Mgr0,
                      HU0,
                      DVS0,
                      # Nfert_in0
                      ])
       
       y_int = fcn_euler_forward(diff,tspan,y0,h=dt)

       # Model results
       t = y_int['t']
       Mrt = y_int['y'][0,:]
       Mst = y_int['y'][1,:]
       Mlv = y_int['y'][2,:]
       Mpa = y_int['y'][3,:]
       Mgr = y_int['y'][4,:]
       HU =  y_int['y'][5,:]
       DVSc = y_int['y'][6,:]
       DVRi = np.where(DVSc < 0.4, DVRJ, 
                np.where(DVSc < 0.65, DVRI, 
                         np.where(DVSc < 1, DVRP, DVRR)))
       DVS = DVRi*HU
       if DVS[-1].any() < DVS[-2]:
           DVS[-1] = DVS[-2]
           
       return {
           't':t,          # [d] Integration time
           'Mrt': Mrt,
           'Mst': Mst,
           'Mlv': Mlv,
           'Mpa': Mpa,
           'Mgr': Mgr,
           'HU': HU,
           'DVS': DVS,

       }
