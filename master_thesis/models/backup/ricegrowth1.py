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
        Nrec    [-] fraction of N fertilizer recovery
        Prec    [-] fraction of P fertilizer recovery
        
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
       self.f_keys = ('f_Ph', 'f_spi', 'f_Nlv', 'f_Nst', 'f_cgr', 'f_res', 'f_gr', 'f_dmv')
       for k in self.f_keys:
           self.f[k] = np.full((self.t.size,), np.nan)
   
   def diff(self, _t, _x0):
       # -- Initial conditions
 
       Mrt = _x0[0]
       Mst = _x0[1]
       Mlv = _x0[2]
       Mpa = _x0[3]
       Mgr = _x0[4]
       SN  = _x0[5]     #nitrogen supply from nutrient cycle model
       # SP  = _x0[6]     #phosphorus supply from nutrient cycle model
             
       # -- Model parameters
       #physical parameters
       Tref = 25                # [°C] reference temperature
       # - rice plants
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
       gamma = self.p['gamma']  # [-] Spikelet growth factor (65 number of spikelet kg-1, 45 - 70 depend on the varieties)
       
       #carbohydrate requirements allocated
       cr_lv = self.p['cr_lv'] #leaves
       cr_st = self.p['cr_st'] #stems
       cr_pa = self.p['cr_pa'] #panicles
       cr_rt = self.p['cr_rt'] #roots
       
       
       #-N and P dynamics
       N_lv = self.p['N_lv']    # [] amount of N in leaves
       N_st = self.p['N_st']    # [] amount of N in stems
       N_pa = self.p['N_pa']    # [] amount of N in storage organs (panicles)
       #P_lv = self.p['P_lv']    # [] amount of P in leaves
       #P_st = self.p['P_st']    # [] amount of P in stems
       #P_pa = self.p['P_pa']    # [] amount of P in storage organs (panicles)
       Nrec = self.p['Nrec']    # [-] N fertilizers recovery coefficient
       #Prec = self.p['Prec']    # [-] P fertilizers recovery coefficient
       fac = 0      # [-] fertilizer application coefficient, default 0 (no application)
       k_Nlv = self.p['k_Nlv']  # [kg N kg leaves-1] fraction of N in leaves on weight basis
       # k_Nst = 0.5*k_Nlv        # [kg N kg stems-1] fraction of N in stems on weight basis
       #k_Plv = self.p['k_Plv']  # [-] fraction of P in leaves
       #k_Pst = self.p['k_Pst']  # [-] fraction of P in stems
       k_Npa_max = self.p['k_Npa_max'] #[kg N kg DM-1] maximum N content in panicles (0.0175)
       k_Npa_min = self.p['k_Npa_min'] #[kg N kg DM-1] minimum N content in panicles
       spi = 0 #[-] spikelet coefficient, if DVS > 1 the value become 1, default 0 (no application)
       ACDD = 0 #[°Cd] Accumulated Cold Degree Days
       Nmaxupt = self.p['Nmaxupt'] #[kg N ha-1 d-1] maximum N uptake by plants (8, range 8 - 12)
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
       #I_P = self.u['I_P'] #P concentration in fertilizer
       
       #data for interpolation
       data_rice = '../data/data_rice.csv'
       
       #Data of N fraction in leaves [g N m-2] as a function of DVS
       DVS_k_Nlv =pd.read_csv(data_rice, usecols=[0, 1], header=1, sep=';')

       #data of SLA values based on DVS
       DVS_SLA = pd.read_csv(data_rice, usecols=[2,3], header=1, sep=';')
       
       #data of LUE values based on Temperature
       T_LUE = pd.read_csv(data_rice, usecols=[4,5], header=1, sep=';')
       
       #data of fraction of max N in leaves based on weight basis as a function of DVS
       DVS_k_Nlv_max = pd.read_csv(data_rice, usecols=[6,7], header=1, sep=';')
       
       #data of fraction of min N in leaves based on weight basis as a function of DVS
       DVS_k_Nlv_min = pd.read_csv(data_rice, usecols=[8,9], header=1, sep=';')
       
       #data of minimum N concentration in storage organs (kg N kgDM-1) as a function of the amount of N in the crop until flowering
       #Ncrop = N_lv+N_st + N_pa
       Ncrop_k_Npa_min = pd.read_csv(data_rice, usecols=[10,11], header=1, sep=';')
       
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
           HU = Tavg - Tmin
       elif Topt <Tavg <Tmax:
           HU = Topt - ((Tavg - Topt)*(Topt-Tmin)/(Tmax-Topt))
       else:
           HU = 0
       HU+=HU
       HUvalues=HU
       print(HUvalues)
       
       # - Developmental stage (DVS)
       if HU > 0:
           DVS = (Tavg - Tmin)/HU
       else:
           DVS = 0
       
       DVS += DVS
       DVSvalues=DVS
       print(DVSvalues)

        # - Dry Matter Partitioning
       if 0 <= DVS < 0.4: #from seed emergence until active tillering
            m_rt = 0.5 #[-] biomass partitioning coefficient for roots
            m_st = 0.3 #[-] biomass partitioning coefficient for stems
            m_lv = 0.2 #[-] biomass partitioning coefficient for leaves
            m_pa = 0 #[-] biomass partitioning coefficient for panicles
            m_sh = m_st + m_lv + m_pa #total biomass partitioning for shoots
            k = 0.4
            Nrec = 0.3
       
       elif 0.4<= DVS < 0.65: #from active tillering until panicle initiation 
            m_rt = 0.25
            m_st = 0.45
            m_lv = 0.3
            m_pa = 0
            m_sh = m_st + m_lv + m_pa #total biomass partitioning for shoots
            k= 0.4
            Nrec = 0.5
            
       elif 0.75 <= DVS < 1.2: #spikelet formation phase
            m_rt = 0.0
            m_st = 0.7
            m_lv = 0.3
            m_pa = 0
            m_sh = m_st + m_lv + m_pa #total biomass partitioning for shoots
            k=0.6
            Nrec = 0.75
            spi = 1

            #cold degree days calculation
            CDD = max(0, 22-Tavg)
            ACDD = ACDD + CDD
            
       elif 0.65 <= DVS < 1: #from panicle initiation until flowering
            m_rt = 0.25
            m_st = 0.45
            m_lv = 0.3
            m_pa = 0.0
            m_sh = m_st + m_lv + m_pa #total biomass partitioning for shoots
            k = 0.6
            Nrec = 0.5
            spi = 1
          
       elif DVS >= 1.2:
           m_rt = 0.0
           m_st = 0.0
           m_lv = 0.0
           m_pa = 1
           m_sh = m_st + m_lv + m_pa #total biomass partitioning for shoots
           k = 0.6
           Nrec = 0.75
           d_lv = 0.025
           spi = 1
           
           #grain formation
           CDD = max(0, 22-Tavg)
           ACDD = ACDD + CDD
           SF1 = 1 - (4.6+ 0.054*(ACDD)**1.56)/100
           SF1 = min(1, max(0,SF1))
           SF2 = 1/(1+np.exp(0.853*(Tavg-36.6)))
           SF2 = min(1,max(0,SF2))
           SPF = min(SF1, SF2)
           
       elif 1 <= DVS < 2: #from flowering until maturity
           m_rt = 0.0
           m_st = 0.4
           m_lv = 0.0
           m_pa = 0.6
           m_sh = m_st + m_lv + m_pa #total biomass partitioning for shoots
           k = 0.6
           Nrec = 0.75
           d_lv = 0.015
           spi = 1
        
           #grain formation
           CDD = max(0, 22-Tavg)
           ACDD = ACDD + CDD
           SF1 = 1 - (4.6+ 0.054*(ACDD)**1.56)/100
           SF1 = min(1, max(0,SF1))
           SF2 = 1/(1+np.exp(0.853*(Tavg-36.6)))
           SF2 = min(1,max(0,SF2))
           SPF = min(SF1, SF2)
           
       elif DVS >= 2: #maturity stage
           m_rt = 0
           m_st = 0
           m_lv = 0
           m_pa = 1
           m_sh = m_st + m_lv + m_pa #total biomass partitioning for shoots
           k = 0.6
           Nrec = 0.75
           d_lv = 0.05
           spi = 1 
   
           #grain formation
           CDD = max(0, 22-Tavg)
           ACDD = ACDD + CDD
           SF1 = 1 - (4.6+ 0.054*(ACDD)**1.56)/100
           SF1 = min(1, max(0,SF1))
           SF2 = 1/(1+np.exp(0.853*(Tavg-36.6)))
           SF2 = min(1,max(0,SF2))
           SPF = min(SF1, SF2)
           
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
       # - general dynamics
       # SN = Nsupp + I_N*Nrec*fac - f_Nlv - f_Nst
       
       # - to calculate N uptake flow by leaves
             
       # - maximum N demand of leaves
       DVS_k_Nlv_max_values = DVS_k_Nlv_max.iloc[:, 0].values
       k_Nlv_max_values = DVS_k_Nlv_max.iloc[:, 1].values
        
       # Create an interpolation function for 
       k_Nlv_max_func = interp1d(DVS_k_Nlv_max_values, k_Nlv_max_values, kind='linear')
             
       # Interpolate to get the actual k_Nlv_max based on current DVS
       k_Nlv_max = k_Nlv_max_func(DVS)
       
       MaxND_lv = k_Nlv_max*Mlv - N_lv
       if MaxND_lv <=0:
           MaxND_lv = 0
          
       # - minimum N demand of leaves (for translocation purpose)
       #- get the value from the dataset
       DVS_k_Nlv_min_values = DVS_k_Nlv_min.iloc[:,0].values
       k_Nlv_min_values = DVS_k_Nlv_min.iloc[:,1].values
        
       # Create an interpolation function for fraction of N in leaves
       k_Nlv_min_func = interp1d(DVS_k_Nlv_min_values, k_Nlv_min_values, kind='linear')
             
       # Interpolate to get the actual k_Nlv_min based on current DVS
       k_Nlv_min = k_Nlv_min_func(DVS)
       
       MinND_lv = k_Nlv_min*Mlv - N_lv
       if MinND_lv < 0:
           MinND_lv = 0
           
       # - to calculate N uptake flow by stems
       # - maximum N demand of stems
       MaxND_st = k_Nlv_max*0.5*Mst - N_st
       if MaxND_st <=0:
           MaxND_st = 0
           
       # - minimum N demand of stems (for translocation purpose, may not used)
       MinND_st = k_Nlv_min*0.5*Mst - N_st
       if MinND_st <=0:
           MinND_st = 0
           
       # - to calculate N uptake flow by panicles
       
       # - maximum N demand of panicles
       MaxND_pa = k_Npa_max*Mpa
       
       # - minimum N demand of panicles (for translocation purpose)
       Ncrop_values = Ncrop_k_Npa_min.iloc[:,0].values
       k_Npa_min_values = Ncrop_k_Npa_min.iloc[:,1].values
       
       # Create an interpolation function for fraction of N in leaves
       k_Npa_min_func = interp1d(Ncrop_values, k_Npa_min_values, kind='linear')
             
       # Interpolate to get the actual k_Nlv_min based on current Total N content in crop organs
       Ncrop = N_lv + N_st + N_pa
       k_Npa_min = k_Npa_min_func(Ncrop)
       
       MinND_pa = k_Npa_min*Mpa
       
       # - N uptake by crop organs
       NUPP = min(Nmaxupt, (SN + I_N*Nrec*fac))
       
       #max of total potential N demand from leaves, stems, and panicles
       MNDcrop = MaxND_lv + MaxND_st + MaxND_pa 
       
       #N uptake of the leaves
       f_Nlv = max(0, min(MaxND_lv, NUPP*MaxND_lv/MNDcrop))
      
       #N content in leaves
       N_lv = f_Nlv/Mlv
       
       # calculate Maximum N content of leaves on a leaf area basis
       DVS_k_Nlv_values = DVS_k_Nlv.iloc[:,0].values
       k_Nlv_values = DVS_k_Nlv.iloc[:,1].values
       
       # Create an interpolation function for fraction of N in leaves
       k_Nlv_func = interp1d(DVS_k_Nlv_values, k_Nlv_values, kind='linear')
             
       # Interpolate to get the actual k_Nlv_min based on current DVS
       k_Nlv = k_Nlv_func(DVS)
              
       PgmaxN = (N_lv/MaxND_lv)*k_Nlv
       
       # -- Phosphorus Dynamics
       # SP = Psupp +I_P*Prec*fac - f_Plv - fPst
       
       #- - Photosynthesis                      
       # - Maximum photosynthesis rate
       
       PgmaxCO2 = (49.57/34.26)*(1-np.exp(-0.208*(_CO2-60)/49.57))
       
       Pgmax = 9.5+(22*PgmaxN*PgmaxCO2)
       
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
       
       #spikelet formation
       f_spi = f_Ph*gamma*spi
       
       #number of grain formation
       # f_gr = f_spi*SPF*spi
       
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
       self.f['f_spi'][idx] = f_spi
       self.f['f_res'][idx] = f_res
       self.f['f_gr'][idx] = f_gr
       self.f['f_cgr'][idx] = f_cgr
       self.f['f_dmv'][idx] = f_dmv
       
       return np.array([dMrt_dt,
                        dMst_dt, 
                        dMlv_dt, 
                        dMpa_dt, 
                        dMgr_dt,
                        SN
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
       SN0 = self.x0['SN']
       
       # Numerical integration
       y0 = np.array([Mrt0, 
                      Mst0, 
                      Mlv0, 
                      Mpa0, 
                      Mgr0,
                      SN0
                      ])
       y_int = fcn_euler_forward(diff,tspan,y0,h=dt)
       
       # Model results
       t = y_int['t']
       Mrt = y_int['y'][0,:]
       Mst = y_int['y'][1,:]
       Mlv = y_int['y'][2,:]
       Mpa = y_int['y'][3,:]
       Mgr = y_int['y'][4,:]
       SN  = y_int['y'][5,:]
       
       return {
           't':t,          # [d] Integration time
           'Mrt': Mrt,
           'Mst': Mst,
           'Mlv': Mlv,
           'Mpa': Mpa,
           'Mgr': Mgr,
           'SN': SN
       }
