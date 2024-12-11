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
       self.f_keys = ('f_Ph', 'f_Nlv', 'f_Nst', 'f_cgr', 'f_res', 'f_gr', 'f_dmv')
       for k in self.f_keys:
           self.f[k] = np.full((self.t.size,), np.nan)           
   
   def diff(self, _t, _x0):
       # -- Initial conditions
 
       Mrt = _x0[0]
       Mst = _x0[1]
       Mlv = _x0[2]
       Mpa = _x0[3]
       Mgr = _x0[4]
             
       # -- Model parameters
       #physical parameters
       Tref = 25                # [°C] reference temperature
       # - rice plants
       HU = self.p['HU']
       DVS = self.p['DVS']
       Tmax = self.p['Tmax']    # [°C] maximum temperature for growth
       Tmin = self.p['Tmin']    # [°C] minimum temperature for growth
       Topt = self.p['Topt']    # [°C] optimum temperature for growth
       # Tavg = self.p['Tavg']    # [°C] average daily temperature
       # SLA = self.p['SLA']      # [-] Specific Leaf Area
       k = self.p['k']          # [-] leaf extinction coefficient (value= 0.4)
       mc_rt = self.p['mc_rt']  # [-] maintenance respiration coefficient of roots (0.01)
       mc_st = self.p['mc_st']  # [-] maintenance respiration coefficient of stems (0.015)
       mc_lv = self.p['mc_lv']  # [-] maintenance respiration coefficient of leaves (0.02)
       mc_pa = self.p['mc_pa']  # [-] maintenance respiration coefficient of panicles (0.003)
       SN = self.p['SN'] #[kg N ha-1 d-1] Nitrogen supply from nutrient cycle subsystems
       # SP = self.p['SP'] #[kg P ha-1 d-1] Phosphorus supply from nutrient cycle subsystems
       
       #carbohydrate requirements allocated
       cr_lv = self.p['cr_lv'] #leaves
       cr_st = self.p['cr_st'] #stems
       cr_pa = self.p['cr_pa'] #panicles
       cr_rt = self.p['cr_rt'] #roots
       
       
       #-N and P dynamics
       N_lv = self.p['N_lv']    # [kg N ha-1 d-1] amount of N in leaves
       N_st = self.p['N_st']    # [kg N ha-1 d-1] amount of N in stems
       N_pa = self.p['N_pa']    # [kg N ha-1 d-1] amount of N in storage organs (panicles)
       IgN = self.p['IgN']      #[kg N ha-1 d-1] Indigenous Soil N supply
       # IgP = self.p['IgP']      #[kg P ha-1 d-1] Indigenous Soil P supply
       # P_lv = self.p['P_lv']    # [kg P ha-1 d-1] amount of P in leaves
       # P_st = self.p['P_st']    # [kg P ha-1 d-1] amount of P in stems
       # P_pa = self.p['P_pa']    # [kg P ha-1 d-1] amount of P in storage organs (panicles)
       Rec_N = self.p['Rec_N']    # [-] N fertilizers recovery coefficient
       # Rec_P = self.p['Rec_P']    # [-] P fertilizers recovery coefficient
       fac = 0      # [-] fertilizer application coefficient, default 0 (no application)
       k_lv_N = self.p['k_lv_N']  # [kg N kg leaves-1] fraction of N in leaves on weight basis
       k_Nst = 0.5*k_lv_N        # [kg N kg stems-1] fraction of N in stems on weight basis
       # k_Plv = self.p['k_Plv']  # [-] fraction of P in leaves
       # k_Pst = self.p['k_Pst']  # [-] fraction of P in stems (0.0011)
       k_pa_maxN = self.p['k_pa_maxN'] #[kg N kg DM-1] maximum N content in panicles (0.0175)
       # k_pa_maxP = self.p['k_pa_maxP'] #[kg P kg DM-1] maximum P content in panicles (0.0026)
       M_upN = self.p['M_upN'] #[kg N ha-1 d-1] maximum N uptake by plants (8, range 8 - 12)
       # M_upP = self.p['M_upP'] #[kg P ha-1 d-1] maximum P uptake by plants (8, range 8 - 12)
       d_lv = 0 #[-] fraction of dead leaf, default at 0 during vegetative growth
       
       # -- Disturbances at instant _t
       I0 = self.d['I0']
       T = self.d['T']
       CO2 = self.d['CO2']
       _I0 = np.interp(_t,I0[:,0],I0[:,1])     # [J m-2 d-2] PAR
       _T = np.interp(_t,T[:,0],T[:,1])        # [°C] Environment temperature
       _CO2 = np.interp(_t,CO2[:,0],CO2[:,1])  # [ppm] Atmospheric CO2 concentration
       
       # -- Controlled inputs
       I_N = self.u['I_N'] #N concentration in fertilizer
       # I_P = self.u['I_P'] #P concentration in fertilizer
       
       #data for interpolation
       data_rice = '../data/data_rice.csv'
       
       #Data of N fraction in leaves [g N m-2] as a function of DVS
       DVS_k_lv_N_area =pd.read_csv(data_rice, usecols=[0, 1], header=1, sep=';')

       #data of SLA values based on DVS
       DVS_SLA = pd.read_csv(data_rice, usecols=[2,3], header=1, sep=';')
       
       #data of LUE values based on Temperature
       T_LUE = pd.read_csv(data_rice, usecols=[4,5], header=1, sep=';')
       
       #data of fraction of max N in leaves based on weight basis as a function of DVS
       DVS_k_lv_maxN = pd.read_csv(data_rice, usecols=[6,7], header=1, sep=';')
       
       # #data of fraction of max N in leaves based on weight basis as a function of DVS
       # DVS_k_lv_P_area = pd.read_csv(data_rice, usecols=[7,8], header=1, sep=';')
       
       # #data of fraction of max N in leaves based on weight basis as a function of DVS
       # DVS_k_lv_maxP = pd.read_csv(data_rice, usecols=[9,10], header=1, sep=';')
       
       # - fertilizer application
       if _t == 12 or _t==30 or _t==67:
           fac = 1
       elif _t== 95:
           fac = 1
           I_N = 45
       else:
           fac = 0
           
       # - Heat Units
       #Tavg = (np.min(_T)+np.max(_T))/2
       Tavg = _T
       
       HU = 0
       if Tmin < Tavg <= Topt:
           HUi = Tavg - Tmin
       elif Topt <Tavg <Tmax:
           HUi = Topt - ((Tavg - Topt)*(Topt-Tmin)/(Tmax-Topt))
       else:
           HUi = 0
       
       HU+=HUi
       HUvalues=HU
       print("HUvalues: ", HUvalues)
       
       # - Developmental stage (DVS)
       if HU > 0:
           DVSi = (Tavg - Tmin)/HU
       else:
           DVSi = 0
       
       DVS+=DVSi
       DVSvalues=DVS
       print("DVSvalues: ", DVSvalues)
       

       # def calculate_HU(Tavg):
       #       if Tmin < Tavg <= Topt:
       #           HUi = Tavg - Tmin
       #       elif Topt < Tavg < Tmax:
       #           HUi = Topt - ((Tavg - Topt)*(Topt-Tmin)/(Tmax-Topt))
       #       else:
       #           HUi = 0
       #       return HUi
            
       # def calculate_DVS(DVS, HU):
       #       return DVS*HU
        
        
       # cHU = calculate_HU(Tavg)
       # HU += cHU
       # cDVS = calculate_DVS(DVS, HU)
       # DVS += cDVS
    
       # print("HU: ", HU)
       # print("DVS: ", DVS)
       
        # - Dry Matter Partitioning
        
       if 0 <= DVS < 0.4: #from seed emergence until active tillering
             m_rt = 0.5 #[-] biomass partitioning coefficient for roots
             m_st = 0.3 #[-] biomass partitioning coefficient for stems
             m_lv = 0.2 #[-] biomass partitioning coefficient for leaves
             m_pa = 0 #[-] biomass partitioning coefficient for panicles
             m_sh = m_st + m_lv + m_pa #total biomass partitioning for shoots
             k = 0.4
             Rec_N = 0.3
       
       elif 0.4<= DVS < 0.65: #from active tillering until panicle initiation 
             m_rt = 0.25
             m_st = 0.45
             m_lv = 0.3
             m_pa = 0
             m_sh = m_st + m_lv + m_pa #total biomass partitioning for shoots
             k= 0.4
             Rec_N = 0.5
            
       # elif 0.75 <= DVS < 1.2: #spikelet formation phase/initial flowering phase
       #       m_rt = 0.0
       #       m_st = 0.7
       #       m_lv = 0.3
       #       m_pa = 0
       #       m_sh = m_st + m_lv + m_pa #total biomass partitioning for shoots
       #       k=0.6
       #       Rec_N = 0.75
            
       elif 0.65 <= DVS < 1: #from panicle initiation until 50% flowering
             m_rt = 0.25
             m_st = 0.45
             m_lv = 0.3
             m_pa = 0.0
             m_sh = m_st + m_lv + m_pa #total biomass partitioning for shoots
             k = 0.6
             Rec_N = 0.5

          
       # elif DVS >= 1.2:
       #      m_rt = 0.0
       #      m_st = 0.0
       #      m_lv = 0.0
       #      m_pa = 1
       #      m_sh = m_st + m_lv + m_pa #total biomass partitioning for shoots
       #      k = 0.6
       #      Rec_N = 0.75
       #      d_lv = 0.025

           
       elif 1 <= DVS < 2: #from 50% flowering until maturity
            m_rt = 0.0
            m_st = 0.4
            m_lv = 0.0
            m_pa = 0.6
            m_sh = m_st + m_lv + m_pa #total biomass partitioning for shoots
            k = 0.6
            Rec_N = 0.75
            d_lv = 0.015
           
       elif DVS >= 2: #maturity stage
            m_rt = 0
            m_st = 0
            m_lv = 0
            m_pa = 1
            m_sh = m_st + m_lv + m_pa #total biomass partitioning for shoots
            k = 0.6
            Rec_N = 0.75
            d_lv = 0.05
   
       # if DVS >= 2: #maturity stage
       #      m_rt = 0
       #      m_st = 0
       #      m_lv = 0
       #      m_pa = 1
       #      m_sh = m_st + m_lv + m_pa #total biomass partitioning for shoots
       #      k = 0.6
       #      Rec_N = 0.75
       #      d_lv = 0.05
            
       # elif 1 <= DVS < 2: #from 50% flowering until maturity
       #      m_rt = 0.0
       #      m_st = 0.4
       #      m_lv = 0.0
       #      m_pa = 0.6
       #      m_sh = m_st + m_lv + m_pa #total biomass partitioning for shoots
       #      k = 0.6
       #      Rec_N = 0.75
       #      d_lv = 0.015
            
       # # elif DVS >= 1.2:
       # #      m_rt = 0.0
       # #      m_st = 0.0
       # #      m_lv = 0.0
       # #      m_pa = 1
       # #      m_sh = m_st + m_lv + m_pa #total biomass partitioning for shoots
       # #      k = 0.6
       # #      Rec_N = 0.75
       # #      d_lv = 0.025
        
       # elif 0.65 <= DVS < 1: #from panicle initiation until 50% flowering
       #      m_rt = 0.25
       #      m_st = 0.45
       #      m_lv = 0.3
       #      m_pa = 0.0
       #      m_sh = m_st + m_lv + m_pa #total biomass partitioning for shoots
       #      k = 0.6
       #      Rec_N = 0.5
        
       # # elif 0.75 <= DVS < 1.2: #spikelet formation phase/initial flowering phase
       # #      m_rt = 0.0
       # #      m_st = 0.7
       # #      m_lv = 0.3
       # #      m_pa = 0
       # #      m_sh = m_st + m_lv + m_pa #total biomass partitioning for shoots
       # #      k = 0.6
       # #      Rec_N = 0.75
        
       # elif 0.4 <= DVS < 0.65: #from active tillering until panicle initiation 
       #      m_rt = 0.25
       #      m_st = 0.45
       #      m_lv = 0.3
       #      m_pa = 0
       #      m_sh = m_st + m_lv + m_pa #total biomass partitioning for shoots
       #      k= 0.4
       #      Rec_N = 0.5
        
       # elif 0 <= DVS < 0.4: #from seed emergence until active tillering
       #      m_rt = 0.5 #[-] biomass partitioning coefficient for roots
       #      m_st = 0.3 #[-] biomass partitioning coefficient for stems
       #      m_lv = 0.2 #[-] biomass partitioning coefficient for leaves
       #      m_pa = 0 #[-] biomass partitioning coefficient for panicles
       #      m_sh = m_st + m_lv + m_pa #total biomass partitioning for shoots
       #      k = 0.4
       #      Rec_N = 0.3    
       
        # Convert dictionary to lists
       DVS_values = DVS_SLA.iloc[:,0].values
       SLA_values = DVS_SLA.iloc[:,1].values
        
       # Create an interpolation function
       SLA_func = interp1d(DVS_values, SLA_values, kind='linear')
             
       # Interpolate to get the actual SLA based on current DVS
       SLA = SLA_func(DVS)
       
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
       k_lv_maxN = k_lv_maxN_func(DVS)
       
       MD_lv_N = k_lv_maxN*Mlv
       if MD_lv_N <=0:
           MD_lv_N = 0
                     
       # - to calculate N uptake flow by stems
       # - maximum N demand of stems
       MD_lv_st_N = k_lv_maxN*0.5*Mst - N_st
       if MD_lv_st_N <=0:
           MD_lv_st_N = 0
           
       # - to calculate N uptake flow by panicles
       
       # - maximum N demand of panicles
       MD_pa_N = k_pa_maxN*Mpa
             
       # - N uptake by crop organs
       f_uppN = min(M_upN, (SN + I_N*Rec_N*fac + IgN))
       
       #max of total potential N demand from leaves, stems, and panicles
       MDcropN = MD_lv_N + MD_lv_st_N + MD_pa_N 
       
       #N uptake of the leaves
       f_Nlv = max(0, min(MD_lv_N, f_uppN*MD_lv_N/MDcropN))
      
       #Actual N content in leaves
       N_lv = f_Nlv/Mlv
       
       # calculate Maximum N content of leaves on a leaf area basis
       DVS_k_lv_N_area_values = DVS_k_lv_N_area.iloc[:,0].values
       k_lv_N_area_values = DVS_k_lv_N_area.iloc[:,1].values
       
       # Create an interpolation function for fraction of N in leaves
       k_lv_N_area_func = interp1d(DVS_k_lv_N_area_values, k_lv_N_area_values, kind='linear')
             
       # Interpolate to get the actual k_lv_N_area based on current DVS [g N m-2 leaf]
       k_lv_N_area = k_lv_N_area_func(DVS)
              
       PgmaxN = (N_lv/k_lv_maxN)*k_lv_N_area
       
       # # -- Phosphorus Dynamics
       # # SP = Psupp +I_P*Rec_P*fac - f_Plv - fPst
       
       # # - to calculate P uptake flow by leaves
       # # - maximum P demand of leaves
       # DVS_k_lv_maxP_values = DVS_k_lv_maxP.iloc[:, 0].values
       # k_lv_maxP_values = DVS_k_lv_maxP.iloc[:, 1].values
        
       # # Create an interpolation function for 
       # k_lv_maxP_func = interp1d(DVS_k_lv_maxP_values, k_lv_maxP_values, kind='linear')
             
       # # Interpolate to get the actual k_lv_maxN based on current DVS
       # k_lv_maxP = k_lv_maxP_func(DVS)
       
       # MD_lv_P = k_lv_maxP*Mlv
       # if MD_lv_P <=0:
       #     MD_lv_P = 0
                     
       # # - to calculate P uptake flow by stems
       # # - maximum P demand of stems
       # MD_st_P = k_lv_maxP*0.5*Mst - P_st
       # if MD_st_P <=0:
       #     MD_st_P = 0
           
       # # - to calculate P uptake flow by panicles
       
       # # - maximum P demand of panicles
       # MD_pa_P = k_pa_maxP*Mpa
             
       # # - P uptake by crop organs
       # f_uppP = min(M_upP, (SP + I_P*Rec_P*fac + IgP))
       
       # #max of total potential P demand from leaves, stems, and panicles
       # MDcropP = MD_lv_P + MD_st_P + MD_pa_P 
       
       # #P uptake of the leaves
       # f_Plv = max(0, min(MD_lv_P, f_uppP*MD_lv_P/MDcropP))
      
       # #Actual P content in leaves
       # P_lv = f_Plv/Mlv
       
       # # calculate Maximum P content of leaves on a leaf area basis
       # DVS_k_lv_P_area_values = DVS_k_lv_P_area.iloc[:,0].values
       # k_lv_P_area_values = DVS_k_lv_P_area.iloc[:,1].values
       
       # # Create an interpolation function for fraction of N in leaves
       # k_lv_P_area_func = interp1d(DVS_k_lv_P_area_values, k_lv_P_area_values, kind='linear')
             
       # # Interpolate to get the actual k_lv_N_area based on current DVS [g N m-2 leaf]
       # k_lv_P_area = k_lv_P_area_func(DVS)
              
       # PgmaxP = (P_lv/k_lv_maxP)*k_lv_P_area
       
       
       #- - Photosynthesis                      
       # - Maximum photosynthesis rate
       
       PgmaxCO2 = (49.57/34.26)*(1-np.exp(-0.208*(_CO2-60)/49.57))
       
       # Pgmax = 9.5+(22*0.314*PgmaxN*PgmaxCO2*0.142*PgmaxP)
       Pgmax = 9.5+(22*0.314*PgmaxN*PgmaxCO2)
       
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
       
       m_sh = m_st+m_lv+m_pa       
       
       # -- Differential equations [kgC m-2 d-1]
       dMrt_dt = m_rt*m_sh*f_cgr  
       dMst_dt = m_st*m_sh*f_cgr 
       dMlv_dt = m_lv*m_sh*f_cgr
       
       
       if DVS>=1:
           dMlv_dt = m_lv*m_sh*f_cgr - f_dmv 
       dMpa_dt = 0
       if DVS>= 0.65:
           dMpa_dt = m_pa*m_sh*f_cgr
       dMgr_dt = 0
       if DVS >= 0.95:
           dMgr_dt = dMpa_dt
       
       # -- Store flows [kgC m-2 d-1]
       idx = np.isin(self.t, _t)
       self.f['f_Ph'][idx] = f_Ph
       self.f['f_res'][idx] = f_res
       self.f['f_gr'][idx] = f_gr
       self.f['f_cgr'][idx] = f_cgr
       self.f['f_dmv'][idx] = f_dmv
       
       return np.array([dMrt_dt,
                        dMst_dt, 
                        dMlv_dt, 
                        dMpa_dt, 
                        dMgr_dt
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
       
       # Numerical integration
       y0 = np.array([Mrt0, 
                      Mst0, 
                      Mlv0, 
                      Mpa0, 
                      Mgr0
                      ])
       y_int = fcn_euler_forward(diff,tspan,y0,h=dt)
       
       # Model results
       t = y_int['t']
       Mrt = y_int['y'][0,:]
       Mst = y_int['y'][1,:]
       Mlv = y_int['y'][2,:]
       Mpa = y_int['y'][3,:]
       Mgr = y_int['y'][4,:]
       
       return {
           't':t,          # [d] Integration time
           'Mrt': Mrt,
           'Mst': Mst,
           'Mlv': Mlv,
           'Mpa': Mpa,
           'Mgr': Mgr
       }
