# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 19:39:18 2024

@author: alegn
"""

import numpy as np
from models.module import Module
from models.integration import fcn_euler_forward

class Nutrient(Module):
    """Module for calculate net nitrogen in the pond
        This module basically gather all flows from fish, phytoplankton, and rice
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
       'N_net'    [kg m^-3] Net nitrogen in the system
       ======  =============================================================        
          
    p : dictionary of scalars
        Model parameters \n

        =======  ============================================================
        key      meaning
        =======  ============================================================
        'mu_phy'     [d-1] maximu_phym growth rate of phytoplankton
        'Iws'  [uE(m2s)-1] light intensity at the water surface
        'T'      [°C] water temperature
        'mu_Up'     [d-1] maximu_phym nutrient uptake coefficient
        =======  ============================================================        
   
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
        self.f_keys = ('f_N_fert_ing', 'f_P_fert_ing','f_N_fert_org', 'f_P_fert_org', 
                       'f_N_plt_upt', 'f_P_plt_upt','f_N_phy_upt', 'f_P_phy_upt',
                       'f_N_fis_sol', 'f_P_fis_sol', 'f_N_fis_prt', 'f_P_fis_prt')
        for k in self.f_keys:
            self.f[k] = np.full((self.t.size,), np.nan)
        
    # Define system of differential equations of the model
    def diff(self,_t,_x0):
        # State variables
        M_N_net = _x0[0]     #Total N nutrient input to the system
        M_P_net = _x0[1]     #Total P nutrient input to the system
        # M_phy = _x0[2]     #Phytoplankton mass

        # -- Parameters
        
        # - physical parameter
        d_pond = 0.6 #[m] pond depth
        V_phy = 3200 #[m3] volume of water for phytoplankton growth
        Topt = 28 #[°C] optimum temperature for phytoplankton growth
        
        # - phytoplankton
        mu_Up = self.p['mu_Up'] #[d-1] maximum nutrient uptake coefficient
        l_sl = self.p['l_sl'] #[m2 g-1] phytoplankton biomass-specific light attenuation
        l_bg = self.p['l_bg'] #[m-1] light attenuation by non-phytoplankton components
        Kpp = self.p['Kpp'] # [J (m2 s)-1]half-saturation constant of phytoplankton production
        c1 = self.p['c1'] # [-] temperature coefficients 1
        c2 = self.p['c2'] # [-] temperature coefficients 2
        K_N_phy = self.p['K_N_phy'] #[g m-3] half saturation constant for N uptake by phytoplankton
        K_P_phy = self.p['K_P_phy'] #[g m-3] half saturation constant for P uptake by phytoplankton       
        kNdecr = self.p['kNdecr'] #[d-1] decomposition rate (to replace decomposition rate of bacteria)
        kPdecr = self.p['kPdecr'] #[d-1] decomposition rate (to replace decomposition rate of bacteria)
        
        # - from the soil
        f_N_edg = self.p['f_N_edg'] #[kg ha-1 d-1] N content from endogenous soil
        f_P_edg = self.p['f_P_edg'] #[kg ha-1 d-1] P content from endogenous soil
        
        # - for the rice plant uptake
        MuptN = self.p['MuptN'] #[kg ha-1 d-1] maximum N uptake of rice plants
        MuptP = self.p['MuptP'] #[kg ha-1 d-1] maximum P uptake of rice plants
        
        # -- Disturbances at instant _t
        I0 = self.d['I0']
        Rain = self.d['Rain']
        Tw = self.d['Tw']
        
        #Input (disturbance) from phytoplankton
        Mphy = self.d['Mphy']
        f_N_phy_cmp = self.d['f_N_phy_cmp']
        f_P_phy_cmp = self.d['f_P_phy_cmp']
        
        #Input (disturbance) from fish
        f_N_sol = self.d['f_N_sol']
        f_N_prt = self.d['f_N_prt']
        f_P_sol = self.d['f_P_sol']
        f_P_prt = self.d['f_P_prt']
        
        _I0 = np.interp(_t,I0[:,0],I0[:,1])     # [J m-2 d-1] PAR, light intensity at water surface
        _Rain = np.interp(_t, Rain[:,0], Rain[:,1]) #[mm] Daily precipitation
        _Tw = np.interp(_t,Tw[:,0],Tw[:,1])        # [°C] Water temperature
        _f_N_phy_cmp = np.interp(_t, f_N_phy_cmp[:,0], f_N_phy_cmp[:,1]) #[g d-1] Nitrogen mass flow from phytoplankton dead bodies
        _f_P_phy_cmp = np.interp(_t, f_P_phy_cmp[:,0], f_P_phy_cmp[:,1]) #[g d-1] Phosphorus mass flow from phytoplankton dead bodies
        _f_N_sol = np.interp(_t, f_N_sol[:,0], f_N_sol[:,1]) #[g d-1] Nitrate concentration
        _f_N_prt = np.interp(_t, f_N_prt[:,0], f_N_prt[:,1]) #[g d-1] Phosphorus concentration
        _f_P_sol = np.interp(_t, f_P_sol[:,0], f_P_sol[:,1]) #[g d-1] Nitrate concentration
        _f_P_prt = np.interp(_t, f_P_prt[:,0], f_P_prt[:,1]) #[g d-1] Phosphorus concentration
        _Rain = np.interp(_t, Rain[:,0], Rain[:,1]) #[mm] Daily precipitation 
        # dr = 0.8 if 0 <= _Rain < 0.5 else 0.45 if 0.5 <= _Rain < 50 else 0.21 # derived from available dilution rate value and rainfall category from BMKG
        
        # -- Controlled inputs
        I_N = self.u['I_N'] #N concentration in inorganic fertilizer
        I_P = self.u['I_P'] #P concentration in inorganic fertilizer
        Norgf = self.u['Norgf'] #N concentration in organic fertilizer
        Porgf = self.u['Norgf'] #P concentration in organic fertilizer
        
        # -- Supporting equations
        dilution_rate = [0.8, 0.72, 0.45, 0.21] #[d-1]
        rainfall_category = [ 0, 20, 100, 150] #[mm]
        dr = np.interp(_Rain, rainfall_category, dilution_rate)
        
        #light limitation
        k_In = _I0*np.exp(-d_pond*(l_sl*Mphy[:,1]/V_phy+l_bg))
        k_I_lim = (1/(d_pond*(l_sl*Mphy[:,1]+l_bg)))*np.log((Kpp+_I0)/(Kpp+k_In))
        
        #temperature limitation
        k_T_lim = c1*np.exp(-c2*np.abs(_Tw-Topt))
        
        #nutrient limitation
        k_N_lim = M_N_net/(M_N_net+K_N_phy) #[-]
        k_P_lim = M_P_net/(M_P_net+K_P_phy) #[-]
        
        #uptake from phytoplankton
        f_N_phy_upt = mu_Up*k_I_lim*k_T_lim*k_N_lim*Mphy[:,1]
        f_P_phy_upt = mu_Up*k_I_lim*k_T_lim*k_P_lim*Mphy[:,1]
        
        print('k_I_lim shape ', np.shape(k_I_lim))
        print('k_T_lim shape ', np.shape(k_T_lim))
        print('k_N_lim shape ', np.shape(k_N_lim))
        print('k_P_lim shape ', np.shape(k_P_lim))
        print('Mphy[:,1] shape ', np.shape(Mphy[:,1]))
        
        #input from inorganic fertilizer, give conditional 
        f_N_fert_ing = I_N if _t == 12 else 0
        f_P_fert_ing = I_P if _t ==12 else 0
        
        #input from organic fertilizer, give conditional 
        f_N_fert_org = Norgf*kNdecr if _t == 0 else 0
        f_P_fert_org = Porgf*kPdecr if _t == 0 else 0
        
        #uptake from rice plants
        f_N_plt_upt = np.minimum(MuptN, ((f_N_edg + f_N_fert_ing + f_N_fert_org + _f_N_sol + _f_N_prt + _f_N_phy_cmp)*dr - f_N_phy_upt))
        f_P_plt_upt = np.minimum(MuptP, ((f_P_edg + f_P_fert_ing + f_P_fert_org + _f_P_sol + _f_P_prt + _f_P_phy_cmp)*dr - f_P_phy_upt))
        
        # -- Differential equations [kg DM d-1]
        dM_N_net_dt = (f_N_edg + f_N_fert_ing + f_N_fert_org + _f_N_sol + _f_N_prt + _f_N_phy_cmp)*dr - f_N_phy_upt - f_N_plt_upt      #[g]
        dM_P_net_dt = (f_P_edg + f_P_fert_ing + f_P_fert_org + _f_P_sol + _f_P_prt + _f_P_phy_cmp)*dr - f_P_phy_upt - f_P_plt_upt     #[g]
        
        #store flows
        # -- Store flows [g m-3 d-1]
        idx = np.isin(self.t, _t)
        self.f['f_N_fert_ing'][idx] = f_N_fert_ing
        self.f['f_P_fert_ing'][idx] = f_P_fert_ing
        self.f['f_N_fert_org'][idx] = f_N_fert_org
        self.f['f_P_fert_org'][idx] = f_P_fert_org
        self.f['f_N_phy_upt'][idx] = f_N_phy_upt
        self.f['f_P_phy_upt'][idx] = f_P_phy_upt
        self.f['f_N_plt_upt'][idx] = f_N_plt_upt
        self.f['f_P_plt_upt'][idx] = f_P_plt_upt

        return np.array([dM_N_net_dt,
                         dM_P_net_dt
                         ])

    # Define model outputs from numerical integration of differential equations.
    # This function is called by the Module method 'run'.
    def output(self,tspan):
        # Retrieve object properties
        dt = self.dt        # integration time step size
        diff = self.diff    # function with system of differential equations
        M_N_net0 = self.x0['M_N_net']
        M_P_net0 = self.x0['M_P_net']
        
        # Numerical integration
        # (for numerical integration, y0 mu_physt be numpy array)
        y0 = np.array([M_N_net0,
                       M_P_net0
                       ])
        y_int = fcn_euler_forward(diff,tspan,y0,h=dt)
        
        # Retrieve results from numerical integration output
        t = y_int['t']              # time
        M_N_net = y_int['y'][0,:]      # first output (row 0, all columns)
        M_P_net = y_int['y'][1,:]
        
        return {'t':t, 
                'M_N_net':M_N_net,
                'M_P_net':M_P_net
                }

class PNut(Module):
    """Module for calculate net nitrogen in the pond, combined with phytoplankton growth calculation
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
       'N_net'    [kg] Net nitrogen in the system
       'P_net'    [kg] Net phosphorus in the system
       'Mphy'    [kg] Phytoplankton mass
       ======  =============================================================        
          
    p : dictionary of scalars
        Model parameters \n

        =======  ============================================================
        key      meaning
        =======  ============================================================
        'mu_phy'     [d-1] maximum growth rate of phytoplankton
        'mu_Up'     [d-1] maximum nutrient uptake coefficient
        =======  ============================================================        
    
    d : dictionary
    Model disturbances
        =======  ============================================================
        key      meaning
        =======  ============================================================
        'T'      [°C] water temperature
        
        =======  ============================================================
        
        
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
        self.f_keys = ('f_N_fert_ing', 'f_P_fert_ing','f_N_fert_org', 'f_P_fert_org', 
                       'f_N_plt_upt', 'f_P_plt_upt','f_N_phy_upt', 'f_P_phy_upt',
                       'f_phy_grw', 'f_phy_prd', 'f_phy_cmp','f_N_phy_cmp', 'f_P_phy_cmp',
                       'f_N_fis_sol', 'f_P_fis_sol', 'f_N_fis_prt', 'f_P_fis_prt')
        for k in self.f_keys:
            self.f[k] = np.full((self.t.size,), np.nan)
        
    # Define system of differential equations of the model
    def diff(self,_t,_x0):
        # State variables
        M_N_net = _x0[0]     #Total N nutrient input to the system
        M_P_net = _x0[1]     #Total P nutrient input to the system
        Mphy = _x0[2]     #Phytoplankton mass
        Nrice = _x0[3]

        # -- Parameters
        
        # - physical parameter
        d_pond = 0.6 #[m] pond depth
        Topt = 28 #[°C] optimum temperature for phytoplankton growth
        
        # - phytoplankton
        mu_phy = self.p['mu_phy'] #[d-1] maximum growth rate of phytoplankton
        mu_Up = self.p['mu_Up'] #[d-1] maximum nutrient uptake coefficient
        l_sl = self.p['l_sl'] #[m2 g-1] phytoplankton biomass-specific light attenuation
        l_bg = self.p['l_bg'] #[m-1] light attenuation by non-phytoplankton components
        Kpp = self.p['Kpp'] # [J (m2 d)-1]half-saturation constant of phytoplankton production
        c_prd = self.p['c_prd'] #[d-1] phytoplankton natural mortality constant
        c_cmp = self.p['c_cmp'] #[m3 (d g)-1] phytoplankton crowding loss constant
        c1 = self.p['c1'] # [-] temperature coefficients 1
        c2 = self.p['c2'] # [-] temperature coefficients 2
        K_N_phy = self.p['K_N_phy'] #[g m-3] half saturation constant for N uptake by phytoplankton
        K_P_phy = self.p['K_P_phy'] #[g m-3] half saturation constant for P uptake by phytoplankton       
        kNdecr = self.p['kNdecr'] #[d-1] decomposition rate (to replace decomposition rate of bacteria)
        kPdecr = self.p['kPdecr'] #[d-1] decomposition rate (to replace decomposition rate of bacteria)
        kNphy = self.p['kNphy'] #[-] fraction of N in phytoplankton dead bodies
        kPphy = self.p['kPphy'] #[-] fraction of P in phytoplankton dead bodies
        V_phy = self.p['V_phy'] #[m3] volume of water for phytoplankton growth
        
        # - from the soil
        f_N_edg = self.p['f_N_edg'] #[g d-1] N content from endogenous soil
        f_P_edg = self.p['f_P_edg'] #[g d-1] P content from endogenous soil
        
        # - for the rice plant uptake
        MuptN = self.p['MuptN'] #[g d-1] maximum N uptake of rice plants
        MuptP = self.p['MuptP'] #[g d-1] maximum P uptake of rice plants
        
        # -- Disturbances at instant _t
        I0 = self.d['I0']
        Rain = self.d['Rain']
        Tw = self.d['Tw']
        DVS = self.d['DVS']
        
        #Input (disturbance) from fish
        f_N_sol = self.d['f_N_sol']
        f_N_prt = self.d['f_N_prt']
        f_P_sol = self.d['f_P_sol']
        f_P_prt = self.d['f_P_prt']
        
        _I0 = np.interp(_t,I0[:,0],I0[:,1])     # [J m-2 d-1] PAR, light intensity at water surface
        _Rain = np.interp(_t, Rain[:,0], Rain[:,1]) #[mm] Daily precipitation
        _Tw = np.interp(_t,Tw[:,0],Tw[:,1])        # [°C] Water temperature
        _DVS = np.interp(_t,DVS[:,0], DVS[:,1])        # [°C] Water temperature
        _f_N_sol = np.interp(_t, f_N_sol[:,0], f_N_sol[:,1]) #[g d-1] Nitrate concentration
        _f_N_prt = np.interp(_t, f_N_prt[:,0], f_N_prt[:,1]) #[g d-1] Phosphorus concentration
        _f_P_sol = np.interp(_t, f_P_sol[:,0], f_P_sol[:,1]) #[g d-1] Nitrate concentration
        _f_P_prt = np.interp(_t, f_P_prt[:,0], f_P_prt[:,1]) #[g d-1] Phosphorus concentration
        _Rain = np.interp(_t, Rain[:,0], Rain[:,1]) #[mm] Daily precipitation 
        
        f_N_fis_sol = _f_N_sol
        f_P_fis_sol = _f_P_sol
        f_N_fis_prt = _f_N_prt
        f_P_fis_prt = _f_P_prt
        
        # -- Controlled inputs
        N_ingf_1 = self.u['N_ingf_1'] #[kg] N concentration in inorganic fertilizer
        N_ingf_2 = self.u['N_ingf_2'] #[kg] N concentration in inorganic fertilizer
        N_ingf_3 = self.u['N_ingf_3'] #[kg] N concentration in inorganic fertilizer
        
        P_ingf = self.u['P_ingf'] #[kg] P concentration in inorganic fertilizer
        Norgf = self.u['Norgf'] #[kg] N concentration in organic fertilizer
        Porgf = self.u['Norgf'] #[kg] P concentration in organic fertilizer
        
        
        # -- Supporting equations
        dilution_rate = [0.8, 0.72, 0.45, 0.21] #[%]
        rainfall_category = [ 0, 20/24, 100/24, 150/24] #[mm]
        dr = np.interp(_Rain, rainfall_category, dilution_rate)
        
        #light limitation
        k_In = _I0*np.exp(-d_pond*(l_sl*Mphy/V_phy+l_bg))
        k_I_lim = (1/(d_pond*(l_sl*Mphy+l_bg)))*np.log((Kpp+_I0)/(Kpp+k_In))
        
        #temperature limitation
        k_T_lim = c1*np.exp(-c2*np.abs(_Tw-Topt))
        
        #nutrient limitation
        k_N_lim = (M_N_net/V_phy)/((M_N_net/V_phy)+K_N_phy) #[-]
        k_P_lim = (M_P_net/V_phy)/((M_P_net/V_phy)+K_P_phy) #[-]
        
        #input and output from phytoplankton
        f_phy_grw = mu_phy*k_I_lim*k_T_lim*k_N_lim*k_P_lim*Mphy
        f_phy_prd = c_prd*Mphy      #[g d-1]
        f_phy_cmp = c_cmp*(Mphy**2)/V_phy #[g d-1]
        f_N_phy_cmp = kNphy*f_phy_cmp #[g d-1]
        f_P_phy_cmp = kPphy*f_phy_cmp #[g d-1]
        f_N_phy_upt = mu_Up*k_I_lim*k_T_lim*k_N_lim*Mphy #[g d-1]
        f_P_phy_upt = mu_Up*k_I_lim*k_T_lim*k_P_lim*Mphy #[g d-1]
        
        #input from inorganic fertilizer, give conditional 
        # Rec = 0.3
        Rec = 0.3 if 0 <= _DVS < 0.4 else 0.5 if 0.4 <= _DVS < 1 else 0.75 #[-] fraction of the nutrient that can be absorbed by plants, 
        
        #inorganic fertilizer application according to Bedriyetti 2000
        f_N_fert_ing = N_ingf_1 if _t == 31*24 else N_ingf_2 if _t==61*24 else 0
        f_P_fert_ing = P_ingf if _t == 41*24 else 0
        
        #input from organic fertilizer, given at the beginning of the fertilizer and 1 week after the first application (Kang'ombe 2006) 
        f_N_fert_org = Norgf*kNdecr if _t == 0 or _t == 7*24 else 0
        f_P_fert_org = Porgf*kPdecr if _t == 0 or _t == 7*24 else 0
        
        #uptake from rice plants
        f_N_plt_upt = np.maximum(0, np.minimum(MuptN, ((f_N_edg + f_N_fert_ing + f_N_fert_org + _f_N_sol + _f_N_prt + f_N_phy_cmp)*dr*Rec - f_N_phy_upt))) if _t>= 21*24 else 0 #[g d-1]
        f_P_plt_upt = np.maximum(0, np.minimum(MuptP, ((f_P_edg + f_P_fert_ing + f_P_fert_org + _f_P_sol + _f_P_prt + f_P_phy_cmp)*dr*Rec - f_P_phy_upt)) if _t>= 21*24 else 0) #[g d-1]
        
        # -- Differential equations [kg DM d-1]
        dM_N_net_dt = (f_N_edg + f_N_fert_ing + f_N_fert_org + _f_N_sol + _f_N_prt + f_N_phy_cmp)*dr - f_N_phy_upt - f_N_plt_upt      #[g]
        dM_P_net_dt = (f_P_edg + f_P_fert_ing + f_P_fert_org + _f_P_sol + _f_P_prt + f_P_phy_cmp)*dr - f_P_phy_upt - f_P_plt_upt     #[g]
        dMphy_dt = f_phy_grw - f_phy_prd - f_phy_cmp
        d_Nrice_dt = (f_N_edg + f_N_fert_ing + f_N_fert_org + _f_N_sol + _f_N_prt + f_N_phy_cmp)*dr - f_N_phy_upt      #[g]
        
        #store flows
        # -- Store flows [g m-3 d-1]
        idx = np.isin(self.t, _t)
        self.f['f_N_fert_ing'][idx] = f_N_fert_ing
        self.f['f_P_fert_ing'][idx] = f_P_fert_ing
        self.f['f_N_fert_org'][idx] = f_N_fert_org
        self.f['f_P_fert_org'][idx] = f_P_fert_org
        self.f['f_N_fis_sol'][idx] = f_N_fis_sol
        self.f['f_P_fis_sol'][idx] = f_P_fis_sol
        self.f['f_N_fis_prt'][idx] = f_N_fis_prt
        self.f['f_P_fis_prt'][idx] = f_P_fis_prt
        self.f['f_phy_grw'][idx] = f_phy_grw
        self.f['f_phy_prd'][idx] = f_phy_prd
        self.f['f_phy_cmp'][idx] = f_phy_cmp
        self.f['f_N_phy_cmp'][idx] = f_N_phy_cmp
        self.f['f_P_phy_cmp'][idx] = f_P_phy_cmp
        self.f['f_N_phy_upt'][idx] = f_N_phy_upt
        self.f['f_P_phy_upt'][idx] = f_P_phy_upt
        self.f['f_N_plt_upt'][idx] = f_N_plt_upt
        self.f['f_P_plt_upt'][idx] = f_P_plt_upt

        return np.array([dM_N_net_dt,
                         dM_P_net_dt,
                         dMphy_dt,
                         d_Nrice_dt
                         ])

    # Define model outputs from numerical integration of differential equations.
    # This function is called by the Module method 'run'.
    def output(self,tspan):
        # Retrieve object properties
        dt = self.dt        # integration time step size
        diff = self.diff    # function with system of differential equations
        M_N_net0 = self.x0['M_N_net']
        M_P_net0 = self.x0['M_P_net']
        Mphy0 = self.x0['Mphy']
        Nrice0 = self.x0['Nrice']
        
        # Numerical integration
        # (for numerical integration, y0 mu_physt be numpy array)
        y0 = np.array([M_N_net0,
                       M_P_net0,
                       Mphy0,
                       Nrice0
                       ])
        y_int = fcn_euler_forward(diff,tspan,y0,h=dt)
        
        # Retrieve results from numerical integration output
        t = y_int['t']              # time
        M_N_net = y_int['y'][0,:]      # first output (row 0, all columns)
        M_P_net = y_int['y'][1,:]
        Mphy = y_int['y'][2,:]
        Nrice = y_int['y'][3,:]
        
        return {'t':t, 
                'M_N_net':M_N_net,
                'M_P_net':M_P_net,
                'Mphy': Mphy,
                'Nrice': Nrice #input for rice plants
                }