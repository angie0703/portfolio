# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   Angela

Class for Monod model, adapted from lotka-volterra model
Class for AOB Model, adapted from Metcalf and Eddy 5th
Class for NOB Model, adapted from Metcalf and Eddy 5th

"""
import numpy as np

from mbps.classes.module import Module
from mbps.functions.integration import fcn_euler_forward

# Model definition
class Monod(Module):
    """ Module for Monod Kinetics:
        Basic formula of substrate oxidation
        
        dS/dt = -mu_max*(X/Y)*(S/(Ks+S))
        dX/dt = mu_max*X*(S/(Ks+S)) - b*X
        dP/dt = mu_max_P*(X/Y)*(S/(Kp+S))
    
    dS/dt = substrate utilization rate
    dX/dt = bacterial growth rate
    dP/dt = production rate
    
    Supporting equations to include the Dissolved Oxygen and Temperature
    disturbances factor:
    
        
        
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

    # Define system of differential equations of the model
    def diff(self,_t,_x0):
        # State variables
        S = _x0[0]
        X = _x0[1]
        P = _x0[2]
        
        # physical constants
        # MrNO2 = 46.00
        # MrNH4 = 18.04
        
        # Parameters
        mu_max_S20 = self.p['mu_max_S20']
        Ks = self.p['Ks']
        K_DO = self.p['K_DO']
        b20 = self.p['b20']
        teta_mu = self.p['teta_mu']
        teta_b = self.p['teta_b']
        Y = self.p['Y']
        MrS = self.p['MrS'] #[g/mol] substrate molecular weight
        MrP = self.p['MrP'] #[g/mol] product molecular weight
        a = self.p['a'] #product coefficient [-]
        
        # -- Disturbances at instant _t
        DO = self.d['DO']
        T = self.d['T']
        _DO = np.interp(_t,DO[:,0],DO[:,1])
        _T = np.interp(_t,T[:,0],T[:,1])
        
        #supporting equations
        mu_max_S = mu_max_S20*(teta_mu**(_T-20))
        b = b20*(teta_b**(_T-20))
        mu_max_S_DO = mu_max_S*(_DO/(_DO+K_DO))
        
        # Differential equations
        dS_dt = -mu_max_S_DO*(X/Y)*(S/(Ks+S))
        dX_dt = (mu_max_S_DO*(S/(Ks+S))-b)*X
        dP_dt = (dS_dt/MrS)*a*MrP
        
        return np.array([dS_dt,dX_dt,dP_dt])

    # Define model outputs from numerical integration of differential equations.
    # This function is called by the Module method 'run'.
    def output(self,tspan):
        # Retrieve object properties
        dt = self.dt        # integration time step size
        diff = self.diff    # function with system of differential equations
        S0 = self.x0['S'] # initial condition of Substrate
        X0 = self.x0['X'] # initial condiiton of Bacteria
        P0 = self.x0['P'] # initial condition of Product
        
        # Numerical integration
        # (for numerical integration, y0 must be numpy array)
        y0 = np.array([S0, X0, P0])
        y_int = fcn_euler_forward(diff,tspan,y0,h=dt)
        
        # Retrieve results from numerical integration output
        t = y_int['t']              # time
        S = y_int['y'][0,:]      # first output (row 0, all columns)
        X = y_int['y'][1,:]      # second output (row 1, all columns)
        P = y_int['y'][2,:]
        
        return {'t':t, 'S':S, 'X':X, 'P':P}

class DR(Module):
    """ Module for decomposition of organic particulate matter, consists of 
        organic fertilizers, particulate matter excretion of fish, and death
        phytoplankton. The decomposition rate focus on decomposition of organic
        matter to ammonium ion for nitrification process. 
        
        m_prt_N = kN*m_prt (the input of this model)
        m_prt_P = kP*m_prt (for Phosphorus solubilization PSB class)
        
        Decomposition rate
        f_dr_N = ...
        
        To calculate bacterial growth when accumulating and producing the ammonium ion:
        dX_dt = (mu_max_S_DO*(S/(Ks+S))-b)*X
        
        Supporting equations:
        #NH4 from soluble excretion of fish
        f_NH4_fish = 0.5*(10**(_pH-pKa))*(f_TAN_sol/MrTAN)*MrNH4
        f_TAN_sol is from fish model
        
        #total NH4 = NH4 from fish excretion + NH4 from decomposed particulate matter (fish, dead phytoplankton, organic fertilizer)
        m_NH4 = m_NH4_fish + m_NH4_dr (decomposition rate)
        mu_max_S = mu_max_S20*(teta_mu**(_T-20))
        b = b20*(teta_b**(_T-20))
        mu_max_S_DO = mu_max_S*(_DO/(_DO+K_DO))
        
        The output:
        m_NH4 = f_NH4_fish + f_dr_N  
        
        
    dNO2/dt = Nitrite production rate
    dX/dt = bacterial growth rate
                
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
    
    d : dictionary
        Disturbances
        * DO : Dissolved Oxygen
        * T : Temperature
        * pH : acidity level pH 
        
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

    # Define system of differential equations of the model
    def diff(self,_t,_x0):
        # State variables
        m_NH4 = _x0[0]
        m_NO2 = _x0[1]
        X_AOB = _x0[2]
        
        # physical constants
        MrNO2 = 46.00
        MrNH4 = 18.04
        
        # Parameters
        mu_max_S20 = self.p['mu_max_S20']
        Ks = self.p['Ks']
        K_DO = self.p['K_DO']
        b20 = self.p['b20']
        teta_mu = self.p['teta_mu']
        teta_b = self.p['teta_b']
        
        # -- Disturbances at instant _t
        DO = self.d['DO']
        T = self.d['T']
        _DO = np.interp(_t,DO[:,0],DO[:,1])
        _T = np.interp(_t,T[:,0],T[:,1])
        
        #supporting equations
        mu_max_S = mu_max_S20*(teta_mu**(_T-20))
        b = b20*(teta_b**(_T-20))
        mu_max_S_DO = mu_max_S*(_DO/(_DO+K_DO))
        
        # Differential equations
        #dS_dt = -mu_max_S_DO*(X/Y)*(S/(Ks+S))
        '''
        Oxydation by AOB
        NH4 + 2 O2 -> NO2 + 2 H2O
        
        m_NH4 = (m_NO2/MrNO2)*MrNH4
        
        '''
        dm_NO2_dt = (m_NH4/MrNH4)*MrNO2
        dX_AOB_dt = (mu_max_S_DO*(m_NO2/(Ks+m_NO2))-b)*X_AOB
        
        
        return np.array([dm_NO2_dt,dX_AOB_dt])

    # Define model outputs from numerical integration of differential equations.
    # This function is called by the Module method 'run'.
    def output(self,tspan):
        # Retrieve object properties
        dt = self.dt        # integration time step size
        diff = self.diff    # function with system of differential equations
        S0 = self.x0['S'] # initial condition of Substrate
        X0 = self.x0['X'] # initial condiiton of Bacteria
        
        # Numerical integration
        # (for numerical integration, y0 must be numpy array)
        y0 = np.array([S0, X0])
        y_int = fcn_euler_forward(diff,tspan,y0,h=dt)
        
        # Retrieve results from numerical integration output
        t = y_int['t']              # time
        S = y_int['y'][0,:]      # first output (row 0, all columns)
        X = y_int['y'][1,:]      # second output (row 1, all columns)

        return {'t':t, 'S':S, 'X':X}
    
class NIB(Module):
    """ Module for Nitrification process, adapted from Monod Kinetics:
        
        1. Oxydation by AOB
        
        NH4 + 2 O2 -> NO2 + 2 H2O
        
        Nitrite production rate
        m_NO2 = (dm_NH4_dt)*MrNO2/MrNH4
        
        To calculate the oxydation rate, use substrate utilization rate:
        dS_dt = 
        To calculate bacterial growth when accumulating and producing the nitrite:
        dX_dt = (mu_max_S_DO*(S_NH4/(Ks+S))-b)*X
        
        
    dNO2/dt = Nitrite production rate
    dX/dt = bacterial growth rate
        2. Oxydation by NOB
        
        2 NO2 + O2 -> 2 NO3
        
        Nitrate production rate
        m_NO3 = (dm_NO2_dt) * MrNO3/MrNO2 
        
    Supporting equations to include the Dissolved Oxygen and Temperature
    disturbances factor:
    
    #NH4 from soluble excretion of fish
    m_NH4_fish = 0.5*(10**(_pH-pKa))*(m_TAN/MrTAN)*MrNH4
    
    #total NH4 = NH4 from fish excretion + NH4 from decomposed particulate matter (fish, dead phytoplankton, organic fertilizer)
    m_NH4 = m_NH4_fish + m_NH4_dr (decomposition rate)
    mu_max_S = mu_max_S20*(teta_mu**(_T-20))
    b = b20*(teta_b**(_T-20))
    mu_max_S_DO = mu_max_S*(_DO/(_DO+K_DO))
                
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

    # Define system of differential equations of the model
    def diff(self,_t,_x0):
        # State variables
        m_NH4 = _x0[0]
        m_NO2 = _x0[1]
        X_AOB = _x0[2]
        
        # physical constants
        MrNO2 = 46.00
        MrNH4 = 18.04
        
        # Parameters
        mu_max_S20 = self.p['mu_max_S20']
        Ks = self.p['Ks']
        K_DO = self.p['K_DO']
        b20 = self.p['b20']
        teta_mu = self.p['teta_mu']
        teta_b = self.p['teta_b']
        
        # -- Disturbances at instant _t
        DO = self.d['DO']
        T = self.d['T']
        _DO = np.interp(_t,DO[:,0],DO[:,1])
        _T = np.interp(_t,T[:,0],T[:,1])
        
        #supporting equations
        mu_max_S = mu_max_S20*(teta_mu**(_T-20))
        b = b20*(teta_b**(_T-20))
        mu_max_S_DO = mu_max_S*(_DO/(_DO+K_DO))
        
        # Differential equations
        #dS_dt = -mu_max_S_DO*(X/Y)*(S/(Ks+S))
        '''
        Oxydation by AOB
        NH4 + 2 O2 -> NO2 + 2 H2O
        
        m_NH4 = (m_NO2/MrNO2)*MrNH4
        
        '''
        dm_NO2_dt = (m_NH4/MrNH4)*MrNO2
        dX_AOB_dt = (mu_max_S_DO*(m_NO2/(Ks+m_NO2))-b)*X_AOB
        
        
        return np.array([dm_NO2_dt,dX_AOB_dt])

    # Define model outputs from numerical integration of differential equations.
    # This function is called by the Module method 'run'.
    def output(self,tspan):
        # Retrieve object properties
        dt = self.dt        # integration time step size
        diff = self.diff    # function with system of differential equations
        S0 = self.x0['S'] # initial condition of Substrate
        X0 = self.x0['X'] # initial condiiton of Bacteria
        
        # Numerical integration
        # (for numerical integration, y0 must be numpy array)
        y0 = np.array([S0, X0])
        y_int = fcn_euler_forward(diff,tspan,y0,h=dt)
        
        # Retrieve results from numerical integration output
        t = y_int['t']              # time
        S = y_int['y'][0,:]      # first output (row 0, all columns)
        X = y_int['y'][1,:]      # second output (row 1, all columns)
        P = y_int['y'][2,:]
        # Pi = y_int['y'][3,:]

        return {'t':t, 'S':S, 'X':X, 'P':P}