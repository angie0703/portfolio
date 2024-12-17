# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:35:31 2024

@author: Angela

Model for Lowland rice with Nitrogen-limited and Phosphorus-limited conditions
The model use general structure of ORYZA2000.  

"""

import numpy as np
from models.module import Module
from models.integration import fcn_euler_forward
from scipy.interpolate import interp1d


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
       self.f_keys = ('f_Ph', 'f_Nlv', 'f_cgr', 'f_res', 'Cr_tot', 'f_uptN', 
                      'f_dmv', 'HU', 'N_lv', 'DVS', 'LAI')
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
       DVRi = self.p['DVRi']    # [(°Ch)-1] initial DVS
       DVRJ = self.p['DVRJ']    # [(°Ch)-1] Development rate in Juvenile Phase ( 0 <=DVS < 4)
       DVRI = self.p['DVRI']    # [(°Ch)-1] Development rate in Active tillering Phase ( 0.4 <=DVS < 0.65)
       DVRP = self.p['DVRP']    # [(°Ch)-1] Development rate in Panicle Development Phase ( 0.65 <=DVS < 1)
       DVRR = self.p['DVRR']    # [(°Ch)-1] Development rate in Reproductive Phase ( DVS >= 1)
       Tmax = self.p['Tmax']    # [°C] maximum temperature for growth
       Tmin = self.p['Tmin']    # [°C] minimum temperature for growth
       Topt = self.p['Topt']    # [°C] optimum temperature for growth       
       Rm_rt = self.p['Rm_rt']  # [-] maintenance respiration coefficient of roots (0.01)
       Rm_st = self.p['Rm_st']  # [-] maintenance respiration coefficient of stems (0.015)
       Rm_lv = self.p['Rm_lv']  # [-] maintenance respiration coefficient of leaves (0.02)
       Rm_pa = self.p['Rm_pa']  # [-] maintenance respiration coefficient of panicles (0.003)
       n_rice = self.p['n_rice'] #[plants] number of plants
       
       #carbohydrate requirements allocated
       cr_lv = self.p['cr_lv'] #[kg CH2O/kg DM] leaves
       cr_st = self.p['cr_st'] #[kg CH2O/kg DM] stems
       cr_pa = self.p['cr_pa'] #[kg CH2O/kg DM] panicles
       cr_rt = self.p['cr_rt'] #[kg CH2O/kg DM] roots
           
       # -- Disturbances at instant _t
       I0 = self.d['I0'] 
       Th = self.d['Th'] # [°C h-1]
       CO2 = self.d['CO2']
       
       #disturbances within the system
       Nrice = self.d['Nrice']
       
       _I0 = np.interp(_t,I0[:,0],I0[:,1])     # [J m-2 h-1] hourly PAR
       _Th = np.interp(_t,Th[:,0],Th[:,1])        # [°C h-1] Hourly temperature
       _CO2 = np.interp(_t,CO2[:,0],CO2[:,1])  # [ppm] Atmospheric CO2 concentration
      
       #-N dynamics
       #to simulate rice seedlings planted in the nursery and transplanted on day 21
       if _t >= 21*24:
           _Nrice = np.interp(_t, Nrice[:,0], Nrice[:,1]) #[g h-1] N uptake by rice plants           
       else:
           _Nrice = 0 #[g h-1] N uptake by rice plants
       
       print('Nrice: ', _Nrice)
       
       # Previous HU value
      #HU units [ C d/d], DVR = [1/Cd], DVS = [-]
       if Tmin < _Th <= Topt:
            dHU_dt= (_Th - Tmin)/24 #[Cd/h] hourly HU
       elif _Th <= Tmin or _Th > Tmax:
            dHU_dt = 0 #[Cd/h]
       else:
           dHU_dt = (Topt - ((_Th - Topt)*(Topt-Tmin)/(Tmax-Topt)))/24 #[Cd/h] hourly HU 
       print('Temperature: ', _Th)
       
       # # - Developmental stage (DVS)
       # Calculate DVS based on previous value and HU 
       # DVRi units = [1/Cd]
       DVRi = DVRJ if 0 <= DVS < 0.4 else DVRI if 0.4 <= DVS < 0.65 else DVRP if 0.65 <= DVS < 1 else DVRR
       dDVS_dt = DVRi*dHU_dt # [h-1] 
       
       # - Dry Matter Partitioning
        # according to Bouwman 2001
       # m_rt = 0.5 if 0 <= DVS < 0.4 else 0.25 if 0.4 <= DVS < 1 else 0 if DVS >= 1 else 0
       # m_st = 0.3 if 0 <= DVS < 1 else 0.4 if 1 <= DVS <2 else 0.3
       # m_lv = 0.2 if 0 <= DVS < 0.4 else 0.3 if 0.4 <= DVS < 1 else 0
       # m_pa = 0 if DVS < 0.65 else 0.6 if 0.65 <= DVS < 1 else 1.0
       
       # According to Agustiani et al 2019
       m_rt = 0.39 if 0 <= DVS < 0.4 else 0.26 if 0.4 <= DVS < 0.65 else 0.0 
       m_st = 0.2 if 0 <= DVS < 0.4 else 0.43 if  0.4 <= DVS < 0.65 else 0.0
       m_lv = 0.41 if 0 <= DVS < 0.4 else 0.31 if 0.4 <= DVS < 0.65 else 0
       m_pa = 0 if 0 <= DVS < 0.65 else 1 
       
       # k : # [-] leaf extinction coefficient (value= 0.4)
       k = 0.4 if 0 <= DVS< 1 else 0.9 
       
       #death leaf coefficient
       # d_lv = 0 if 0 <=DVS < 1 else 0.015 if 1<= DVS < 2 else 0.05
       d_lv = 0 if 0 <=DVS < 1 else 0.015 if 1<= DVS < 1.6 else 0.025 if 1.6 <= DVS < 2.1 else 0.05

       # Convert dictionary to lists
       DVS_values = [0, 0.16, 0.33, 0.65, 0.79, 2.1, 2.5] #[-]
       SLA_values = [0.0045, 0.0045, 0.0033, 0.0028, 0.0024, 0.0023,0.0023] #[???]
       # m = 3
       # SLA_values = [0.0045*m, 0.0045*m, 0.0033*m, 0.0028*m, 0.0024*m, 0.0023*m,0.0023*m] #[???]
       
       # Create an interpolation function
       SLA_func = interp1d(DVS_values, SLA_values, kind='linear')
             
       # Interpolate to get the actual SLA based on current DVS
       SLA = SLA_func(DVS) if DVS <=2.5 else 0.0
       print('dDVS_dt: ', dDVS_dt)
       print('SLA: ', SLA)
       
       # - Leaf Area Index (LAI)
       LAI = SLA*Mlv*n_rice  # [ha ha-1 soil] Leaf area index
       
       # -- Supporting equations
       # -- Nitrogen Dynamics
       
       # - to calculate N uptake flow by leaves
       N_lv = 0    # [kg N ha-1 h-1] amount of N in leaves
       k_pa_maxN = 0 #[kg N kg DM-1] maximum N content in panicles (0.0175)
       # - maximum N demand of leaves
       DVS_k_lv_maxN_values = [0, 0.4, 0.75, 1, 2, 2.5]
       k_lv_maxN_values = [0.053, 0.053, 0.04, 0.028, 0.022, 0.015]
       
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
       MuptN = 8*0.6*1000/24 #[kg h-1] maximum daily N uptake by rice plants, originally in [kg ha-1 d-1], only 0.6 ha of land that is planted with rice
       Rec = 0.35 if 0 <= DVS < 0.4 else 0.5 if 0.4 <= DVS < 1 else 0.75
       f_N_plt_upt = np.maximum(0, np.minimum(MuptN, _Nrice/1000*Rec)) #[kg h-1]
       f_uptN = f_N_plt_upt/0.6 #[kg ha-1 h-1] convert the nutrient source to kg ha-1 h-1

       print('N uptake of rice: ', f_uptN, 'kg')
       
       
       #max of total potential N demand from leaves, stems, and panicles
       MDcropN = MD_lv_N + MD_st_N + MD_pa_N 
       
       #N uptake of the leaves
       f_Nlv = max(0, min(MD_lv_N, f_uptN*MD_lv_N/MDcropN))
      
       #Actual N content in leaves per kg DM leaf
       N_lv = f_Nlv/Mlv
       
       # calculate Maximum N content of leaves on a leaf area basis
       DVS_k_lv_N_area_values = [0, 0.16, 0.33, 0.65, 0.79, 1, 1.46, 2.02, 2.5] #[ ] hourly developmental stages
       k_lv_N_area_values = [0.54, 0.54, 1.53, 1.22, 1.56, 1.29, 1.37, 0.83, 0.83]
       
       # Create an interpolation function for fraction of N in leaves
       k_lv_N_area_func = interp1d(DVS_k_lv_N_area_values, k_lv_N_area_values, kind='linear')
             
       # Interpolate to get the actual k_lv_N_area based on current DVS [g N m-2 leaf]
       k_lv_N_area = k_lv_N_area_func(DVS) if DVS <=2.5 else 0.0
              
       PgmaxN = (N_lv/k_lv_maxN)*k_lv_N_area #[g N m-2 leaf]
       print('PgmaxN: ', PgmaxN)    
       
       #- - Photosynthesis  
       # - Maximum photosynthesis rate
       PgmaxCO2 = (49.57/34.26)*(1-np.exp(-0.208*(_CO2-60)/49.57)) #[]
       
       Pgmax = 9.5+(22*PgmaxN*PgmaxCO2) #[kg CO2/ha/d]
       #0.314 is converter of g N/m2 to kg CO2/ha
       
       # - Initial LUE
       LUE340 = (1- np.exp(-0.00305*_CO2-0.222))/3600*(1- np.exp(-0.00305*340-0.222))
 
       T_values = [10, 40] #[C]
       LUE_values = [0.54, 0.36] #[-]

       # Create an interpolation function
       LUE_func = interp1d(T_values, LUE_values, kind='linear')

       # Interpolate the values
       LUE = LUE_func(_Th)*LUE340 
       print("LUE: ", LUE)
       # - Solar radiation absorbed (Iabs)
       Iabs = 0.50*(_I0*(1-np.exp(-k*LAI)))
       
       # - Photosynthesis rate
       # photosynthesis hasn't happened until rice seedlings develop their first true leaves, generally on day 5
       if 0 <= _t < 5*24:
           f_Ph = 0
       else:
           f_Ph = Pgmax*(1-np.exp(-LUE*(Iabs/Pgmax)))
       
       # - maintenance respiration
       # Teff = 2**((_Tavg-Tref)/10)
       Teff = 2**((_Th-Tref)/10)
       f_res = (Rm_rt*Mrt+Rm_st*Mst+Rm_lv*Mlv+Rm_pa*Mpa)*Teff
       
       # - growth respiration
       Cr_tot = (m_lv+m_st+m_pa)*(cr_lv*m_lv+cr_st*m_st+cr_pa*m_pa)+cr_rt*m_rt
       
       # - leaves death rate
       f_dmv = d_lv*Mlv
       
       # - Crop growth rate
       f_cgr = (f_Ph*(30/44)-f_res)/Cr_tot
       
       # -- Differential equations [kgDM ha-1 d-1]
       
       dMrt_dt = m_rt*f_cgr  
       dMst_dt = m_st*f_cgr 
       dMlv_dt = m_lv*f_cgr
      
       if DVS>=1:
           dMlv_dt = m_lv*f_cgr - f_dmv 
       
       dMpa_dt = 0
       if DVS>= 0.65:
            dMpa_dt = np.maximum(0, m_pa*f_cgr)
            # dMpa_dt = m_pa*f_cgr
       
       dMgr_dt = 0
       if DVS >= 0.95:
           dMgr_dt = dMpa_dt
       
       # -- Store flows [kgC m-2 d-1]
       idx = np.isin(self.t, _t)
       self.f['f_Ph'][idx] = f_Ph
       self.f['f_res'][idx] = f_res
       self.f['Cr_tot'][idx] = Cr_tot
       self.f['f_cgr'][idx] = f_cgr
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

       # Numerical integration
       y0 = np.array([Mrt0, 
                      Mst0, 
                      Mlv0, 
                      Mpa0, 
                      Mgr0,
                      HU0,
                      DVS0,
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
       DVS = y_int['y'][6,:]
       # DVRi = np.where(DVSc < 0.4, DVRJ, 
       #          np.where(DVSc < 0.65, DVRI, 
       #                   np.where(DVSc < 1, DVRP, DVRR)))
       # DVS = DVRi*HU
       # if DVS[-1].any() < DVS[-2]:
       #     DVS[-1] = DVS[-2]
           
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
