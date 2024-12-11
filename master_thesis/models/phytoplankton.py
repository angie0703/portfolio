# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 14:47:44 2023

"""

# -*- coding: utf-8 -*-
"""
@authors:   Angela

Class for Phytoplankton
"""
import numpy as np
from models.module import Module
from models.integration import fcn_euler_forward,fcn_rk4

# Model definition
class Phygrowth(Module):
    """ Module for  calculate Phytoplankton biomass growth and nutrient 
    availability in the pond

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
       'Mphy'    [g Chla] Initial phytoplankton biomass growth
       ======  =============================================================        
          
        
    p : dictionary of scalars
        Model parameters \n

        =======  ============================================================
        key      meaning
        =======  ============================================================
        'mu_phy'     [d-1] maximu_phym growth rate of phytoplankton

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
        self.f_keys = ('f1', 'f2', 'f3', 'f4', 'f5', 'f6')
        for k in self.f_keys:
            self.f[k] = np.full((self.t.size,), np.nan)
        
    # Define system of differential equations of the model
    def diff(self,_t,_x0):
        # State variables
        Mphy = _x0[0]
        NA = _x0[1]
        
        # Parameters
        mu_phy = self.p['mu_phy'] #[d-1] maximum growth rate of phytoplankton
        mu_Up = self.p['mu_Up'] #[d-1] maximum nutrient uptake coefficientd
        l_sl = self.p['l_sl'] #[m2 g-1] phytoplankton biomass-specific light attenuation
        l_bg = self.p['l_bg'] #[m-1] light attenuation by non-phytoplankton components
        Kpp = self.p['Kpp'] # [J (m2 s)-1]half-saturation constant of phytoplankton production
        cm = self.p['cm'] #[d-1] phytoplankton natural mortality constant
        cl = self.p['cl'] #[m3 (d g)-1] phytoplankton crowding loss constant
        c1 = self.p['c1'] # [-] temperature coefficients 1
        c2 = self.p['c2'] # [-] temperature coefficients 2
        Topt = self.p['Topt'] #optimum temperature for phytoplankton growth
        Mp = self.p['Mp'] #[g m-3] half saturation constant for nutrient uptake
        
        # -- Disturbances at instant _t
        I0 = self.d['I0']
        T = self.d['T']
        d_pond = self.d['d_pond']
        # SNH4 = self.d['SNH4']
        # SNO2= self.d['SNO2']
        SNO3= self.d['SNO3']
        SP = self.d['SP']
        Rain = self.d['Rain']

        _I0 = np.interp(_t,I0[:,0],I0[:,1])     # [J m-2 d-1] PAR, light intensity at water surface
        _T = np.interp(_t,T[:,0],T[:,1])        # [°C] Environment temperature
        _d_pond = np.interp(_t,d_pond[:,0], d_pond[:,1])        # [m] Pond depth
        # _S_NH4 = np.interp(_t, SNH4[:,0], SNH4[:,1]) #[g d-1] Ammonium concentration
        # _S_NO2 = np.interp(_t, SNO2[:,0], SNO2[:,1]) #[g d-1] Nitrite concentration
        _S_NO3 = np.interp(_t, SNO3[:,0], SNO3[:,1]) #[g d-1] Nitrate concentration
        _S_P = np.interp(_t, SP[:,0], SP[:,1]) #[g d-1] Phosphorus concentration
        _Rain = np.interp(_t, Rain[:,0], Rain[:,1]) #[mm] Daily precipitation 


        # dr = 0.8 if 0 <= _Rain < 0.5 else 0.45 if 0.5 <= _Rain < 50 else 0.21 # derived from available dilution rate value and rainfall category from BMKG
        dilution_rate = [0.8, 0.72, 0.45, 0.21] #[d-1]
        rainfall_category = [ 0, 20, 100, 150] #[mm]
        dr = np.interp(_Rain, rainfall_category, dilution_rate)
        print('dilution rate', dr)
        
        # supporting equations
        k_In = _I0*np.exp(-_d_pond*(l_sl*Mphy+l_bg))
        k_lm = (1/(_d_pond*(l_sl*Mphy+l_bg)))*np.log((Kpp+_I0)/(Kpp+k_In))
        
        kTw = c1*np.exp(-c2*np.abs(_T-Topt))
        
            
        # kNA = 1 - (1/NA)
        # NA = _S_NH4 + _S_NO2 + _S_NO3 + _S_P #[g d-1]
        NA = _S_NO3 + _S_P #[g d-1]
        kNA = NA/(NA+Mp)
        
        f1 = mu_phy*k_lm*kTw*kNA*Mphy
        f2 = cm*Mphy
        f3 = cl*(Mphy**2)
        f4 = dr*_S_NO3
        f5 = dr*_S_P
        f6 = mu_Up*k_lm*kTw*kNA*Mphy

        # Differential equations
        dMphy_dt = f1 - f2 - f3
        dNA_dt = f4 + f5 - f6 
        # dMphy_dt = f1 - f2 - f3
        # dNA_dt = f4 - f5 
        
        # Ensure NA does not go below zero
        if dNA_dt < 0 and NA + dNA_dt < 0:
            dNA_dt = 0  # Set dNA_dt to -NA to ensure NA does not go below zero
            
        #store flows
        # -- Store flows [g m-3 d-1]
        idx = np.isin(self.t, _t)
        self.f['f1'][idx] = f1
        self.f['f2'][idx] = f2
        self.f['f3'][idx] = f3
        self.f['f4'][idx] = f4
        self.f['f5'][idx] = f5
        self.f['f6'][idx] = f6
        
        return np.array([dMphy_dt,
                         dNA_dt
                         ])

    # Define model outputs from numerical integration of differential equations.
    # This function is called by the Module method 'run'.
    def output(self,tspan):
        # Retrieve object properties
        dt = self.dt        # integration time step size
        diff = self.diff    # function with system of differential equations
        Mphy0 = self.x0['Mphy'] # initial condition
        NA0 = self.x0['NA'] # initial condiiton
        cl = self.p['cl'] #[m3 (d g)-1] phytoplankton crowding loss constant
        
        # Numerical integration
        # (for numerical integration, y0 mu_physt be numpy array)
        y0 = np.array([Mphy0,
                       NA0
                       ])
        # y_int = fcn_euler_forward(diff,tspan,y0,h=dt)
        y_int = fcn_rk4(diff, tspan, y0, h=dt)
        
        # Retrieve results from numerical integration output
        t = y_int['t']              # time
        Mphy = y_int['y'][0,:]      # first output (row 0, all columns)
        NA = y_int['y'][1,:]      # second output (row 1, all columns)
        f3 = cl*(Mphy**2)

        return {'t':t, 
                'Mphy':Mphy, 
                'NA':NA,
                'f3':f3
                }

class Phyto(Module):
    """ Revised Module for calculate Phytoplankton biomass growth and nutrient 
    availability in the pond

    Parameters
    ----------
    tsim : array
        Sequence of time points for the simu_phylation
    dt : float
        Time step size [d]
    x0 : dictionary
        Initial conditions of the state variables \n

       ======    =============================================================
       key       meaning
       ======    =============================================================
       'Mphy'    [g Chla] Initial phytoplankton biomass growth
       'NA'      [] Total nutrient availability for phytoplankton
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
        self.f_keys = ('f_phy_grw', 'f_phy_prd', 'f_phy_cmp')
        for k in self.f_keys:
            self.f[k] = np.full((self.t.size,), np.nan)
        
    # Define system of differential equations of the model
    def diff(self,_t,_x0):
        # State variables
        Mphy = _x0[0]
        
        # Parameters
        mu_phy = self.p['mu_phy'] #[d-1] maximum growth rate of phytoplankton
        l_sl = self.p['l_sl'] #[m2 g-1] phytoplankton biomass-specific light attenuation
        l_bg = self.p['l_bg'] #[m-1] light attenuation by non-phytoplankton components
        Kpp = self.p['Kpp'] # [J (m2 d)-1]half-saturation constant of phytoplankton production
        c_prd = self.p['c_prd'] #[d-1] phytoplankton natural mortality constant
        c_cmp = self.p['c_cmp'] #[m3 (d g)-1] phytoplankton crowding loss constant
        c1 = self.p['c1'] # [-] temperature coefficients 1
        c2 = self.p['c2'] # [-] temperature coefficients 2
        Topt = self.p['Topt'] #optimum temperature for phytoplankton growth
        K_N_phy = self.p['K_N_phy'] #[g m-3] half saturation constant for N uptake
        K_P_phy = self.p['K_P_phy'] #[g m-3] half saturation constant for P uptake        
        kNphy = self.p['kNphy'] #[-] fraction of N in phytoplankton dead bodies
        kPphy = self.p['kPphy'] #[-] fraction of P in phytoplankton dead bodies
        V_phy = self.p['V_phy']
        
        # -- Disturbances at instant _t
        I0 = self.d['I0']
        T = self.d['T']
        N_net = self.d['N_net']
        P_net = self.d['P_net']

        _I0 = np.interp(_t,I0[:,0],I0[:,1])     # [J m-2 d-1] PAR, light intensity at water surface
        _T = np.interp(_t,T[:,0],T[:,1])        # [°C] Environment temperature
        _N_net = np.interp(_t, N_net[:,0], N_net[:,1]) #[g d-1] Nitrate concentration
        _P_net = np.interp(_t, P_net[:,0], P_net[:,1]) #[g d-1] Phosphorus concentration
       
        
        # supporting equations
        d_pond = 0.6
        
        #light limitation
        k_In = _I0*np.exp(-d_pond*(l_sl*Mphy/V_phy+l_bg))
        k_I_lim = (1/(d_pond*(l_sl*Mphy+l_bg)))*np.log((Kpp+_I0)/(Kpp+k_In))
        
        #temperature limitation
        k_T_lim = c1*np.exp(-c2*np.abs(_T-Topt))
        
        #nutrient limitation
        k_N_lim = _N_net/(_N_net+K_N_phy) #[-]
        k_P_lim = _P_net/(_P_net+K_P_phy) #[-]
        
        
        f_phy_grw = mu_phy*k_I_lim*k_T_lim*k_N_lim*k_P_lim*Mphy
        f_phy_prd = c_prd*Mphy      #[g m-3 d-1]
        f_phy_cmp = c_cmp*(Mphy**2) #[g m-3 d-1]    

        print('k_I_lim shape ', np.shape(k_I_lim))
        print('k_T_lim shape ', np.shape(k_T_lim))
        print('k_N_lim shape ', np.shape(k_N_lim))
        print('k_P_lim shape ', np.shape(k_P_lim))
        print('Mphy shape ', np.shape(Mphy))
        
        # -- Differential equations [gDM day-1]
        dMphy_dt = f_phy_grw - f_phy_prd - f_phy_cmp
         
        #store flows
        # -- Store flows [g m-3 d-1]
        idx = np.isin(self.t, _t)
        self.f['f_phy_grw'][idx] = f_phy_grw
        self.f['f_phy_prd'][idx] = f_phy_prd
        self.f['f_phy_cmp'][idx] = f_phy_cmp

        return np.array([dMphy_dt,
                         ])

    # Define model outputs from numerical integration of differential equations.
    # This function is called by the Module method 'run'.
    def output(self,tspan):
        # Retrieve object properties
        dt = self.dt        # integration time step size
        diff = self.diff    # function with system of differential equations
        Mphy0 = self.x0['Mphy'] # initial condition
        
        c_cmp = self.p['c_cmp'] #[m3 (d g)-1] phytoplankton crowding loss constant
        kNphy = self.p['kNphy'] #[-] fraction of N in phytoplankton dead bodies
        kPphy = self.p['kPphy'] #[-] fraction of P in phytoplankton dead bodies
        
        # Numerical integration
        # (for numerical integration, y0 mu_physt be numpy array)
        y0 = np.array([Mphy0,
                       ])
        y_int = fcn_euler_forward(diff,tspan,y0,h=dt)
        
        # Retrieve results from numerical integration output
        t = y_int['t']              # time
        Mphy = y_int['y'][0,:]      # first output (row 0, all columns)
        f_phy_cmp = c_cmp*(Mphy**2) #[g m-3 d-1]  
        f_N_phy_cmp = kNphy*f_phy_cmp
        f_P_phy_cmp = kPphy*f_phy_cmp

        return {'t':t, 
                'Mphy':Mphy/1000,
                'f_N_phy_cmp':f_N_phy_cmp/1000, #input for the M_N_net
                'f_P_phy_cmp':f_P_phy_cmp/1000 #input for the M_P_net
                }