# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   Angela

Class for Monod model, adapted from lotka-volterra model
"""
import numpy as np
import copy
from models.module import Module
from models.integration import fcn_euler_forward, fcn_rk4

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
        # Initialize dictionary of flows

    # Define system of differential equations of the model
    def diff(self,_t,_x0):
        # State variables
        S = _x0[0] 
        X = _x0[1]

        
        # Parameters
        mu_max0 = self.p['mu_max0']
        Ks = self.p['Ks']
        K_DO = self.p['K_DO']
        b20 = self.p['b20']
        teta_mu = self.p['teta_mu']
        teta_b = self.p['teta_b']
        Y = self.p['Y']
        MrS = self.p['MrS'] #[g/mol] substrate molecular weight
        MrP = self.p['MrP'] #[g/mol] product molecular weight
        a = self.p['a'] #[-] product coefficient
    
        # -- Disturbances at instant _t
        DO = self.d['DO']
        T = self.d['T']
        S_out = self.d['S_out'] #output from other model to become additional for the state variable substrate
        _DO = np.interp(_t,DO[:,0],DO[:,1])
        _T = np.interp(_t,T[:,0],T[:,1])
        _S_out = np.interp(_t, S_out[:,0], S_out[:,1])
        
        #supporting equations
        b = b20*(teta_b**(_T-20))
        mu_max = mu_max0*(_DO/(_DO+K_DO))*(teta_mu**(_T-20))
        
        S += _S_out 
        
        # Differential equations
        dS_dt = -mu_max*(X/Y)*(S/(Ks+S))
        dX_dt = (mu_max*(S/(Ks+S))-b)*X
        
        
        return np.array([dS_dt,dX_dt])

    # Define model outputs from numerical integration of differential equations.
    # This function is called by the Module method 'run'.
    def output(self,tspan):
        # Retrieve object properties
        dt = self.dt        # integration time step size
        diff = self.diff    # function with system of differential equations
        S0 = self.x0['S'] # initial condition of Substrate
        X0 = self.x0['X'] # initial condiiton of Bacteria
        MrS = self.p['MrS']
        a = self.p['a']
        MrP = self.p['MrP']
        
        # Numerical integration
        # (for numerical integration, y0 must be numpy array)
        y0 = np.array([S0, X0])
        # y_int = fcn_euler_forward(diff,tspan,y0,h=dt)
        y_int = fcn_rk4(diff, tspan, y0, h=dt)
        
        # Retrieve results from numerical integration output
        t = y_int['t']              # time
        S = y_int['y'][0,:]      # first output (row 0, all columns)
        X = y_int['y'][1,:]      # second output (row 1, all columns)
        P = (-S/MrS)*a*MrP
        
        return {'t':t, 'S':S, 'X':X, 'P':P}

class AOB(Module):
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
        # Initialize dictionary of flows

    # Define system of differential equations of the model
    def diff(self,_t,_x0):
        # State variables
        S = _x0[0] 
        X = _x0[1]
        
        # Parameters
        mu_max0 = self.p['mu_max0']
        Ks = self.p['Ks']
        K_DO = self.p['K_DO']
        b20 = self.p['b20']
        teta_mu = self.p['teta_mu']
        teta_b = self.p['teta_b']
        Y = self.p['Y']
        MrS = self.p['MrS'] #[g/mol] substrate molecular weight
        MrP = self.p['MrP'] #[g/mol] product molecular weight
        a = self.p['a'] #[-] product coefficient
    
        
        # -- Disturbances at instant _t
        DO = self.d['DO']
        T = self.d['T']
        SNH4 = self.d['SNH4'] #output from other model to become additional for the state variable substrate
        f_TAN = self.d['f_TAN']
        _DO = np.interp(_t,DO[:,0],DO[:,1])
        _T = np.interp(_t,T[:,0],T[:,1])
        _SNH4 = np.interp(_t, SNH4[:,0], SNH4[:,1])
        _f_TAN = np.interp(_t, f_TAN[:,0], f_TAN[:,1])
        
        #supporting equations
        b = b20*(teta_b**(_T-20))
        mu_max = mu_max0*(_DO/(_DO+K_DO))*(teta_mu**(_T-20))
        
        S += _SNH4 + _f_TAN
        
        # Differential equations
        dS_dt = -mu_max*(X/Y)*(S/(Ks+S))
        dX_dt = (mu_max*(S/(Ks+S))-b)*X
        
        
        return np.array([dS_dt,dX_dt])

    # Define model outputs from numerical integration of differential equations.
    # This function is called by the Module method 'run'.
    def output(self,tspan):
        # Retrieve object properties
        dt = self.dt        # integration time step size
        diff = self.diff    # function with system of differential equations
        S0 = self.x0['S'] # initial condition of Substrate
        X0 = self.x0['X'] # initial condiiton of Bacteria
        MrS = self.p['MrS']
        a = self.p['a']
        MrP = self.p['MrP']
        
        # Numerical integration
        # (for numerical integration, y0 must be numpy array)
        y0 = np.array([S0, X0])
        # y_int = fcn_euler_forward(diff,tspan,y0,h=dt)
        y_int = fcn_rk4(diff, tspan, y0, h=dt)
        
        # Retrieve results from numerical integration output
        t = y_int['t']              # time
        S = y_int['y'][0,:]      # first output (row 0, all columns)
        X = y_int['y'][1,:]      # second output (row 1, all columns)
        P = (-S/MrS)*a*MrP
        
        return {'t':t, 'S':S, 'X':X, 'P':P}    
class DB(Module):
    """ Module for Monod Kinetics:
        Basic formula of substrate oxidation
        
        dS/dt = -mu_max*(X/Y)*(S/(Ks+S))
        dX/dt = mu_max*X*(S/(Ks+S)) - b*X
        dP/dt = mu_max_P*(X/Y)*(S/(Kp+S))
    
    dS/dt = substrate utilization rate
    dX/dt = bacterial growth rate
    dP/dt = production rate
    
    Supporting equations to include the Dissolved Oxygen and Temperature
    disturbances factor    
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
        
    # Define system of differential equations of the model
    def diff(self,_t,_x0):
        
        # State variables
        S = _x0[0]
        X = _x0[1]
        
        # Parameters
        mu_max0 = self.p['mu_max0']
        Ks = self.p['Ks']
        K_DO = self.p['K_DO']
        b20 = self.p['b20']
        teta_mu = self.p['teta_mu']
        teta_b = self.p['teta_b']
        Y = self.p['Y']
        MrS = self.p['MrS'] #[g/mol] substrate molecular weight
        MrP = self.p['MrP'] #[g/mol] product molecular weight
        a = self.p['a'] #[-] product coefficient
        kN = self.p['kN'] #fraction of N in phytoplankton dead bodies
          
        # -- Disturbances at instant _t
        DO = self.d['DO']
        T = self.d['T'] 
        f3 = self.d['f3'] #[g d-1] phytoplankton death rate due to intraspecific competition 
        f_N_prt = self.d['f_N_prt'] #[g d-1] N particulate matter from fish
        _DO = np.interp(_t,DO[:,0],DO[:,1])
        _T = np.interp(_t,T[:,0],T[:,1])
        _f3 = np.interp(_t, f3[:,0], f3[:,1])
        _f_N_prt = np.interp(_t, f_N_prt[:,0], f_N_prt[:,1])
        
        # -- Controlled input
        Norgf = self.u['Norgf'] if _t == 0 else 0
    
        #supporting equations
        S = S + Norgf + _f3*kN + _f_N_prt
        
        b = b20*(teta_b**(_T-20))
        mu_max = mu_max0*(_DO/(_DO+K_DO))*(teta_mu**(_T-20))
        
        # Differential equations
        dS_dt = -mu_max*(X/Y)*(S/(Ks+S))
        dX_dt = (mu_max*(S/(Ks+S))-b)*X

        return np.array([dS_dt,dX_dt])

    # Define model outputs from numerical integration of differential equations.
    # This function is called by the Module method 'run'.
    def output(self,tspan):
        # Retrieve object properties
        dt = self.dt        # integration time step size
        diff = self.diff    # function with system of differential equations
        S0 = self.x0['S'] # initial condition of Substrate
        X0 = self.x0['X'] # initial condiiton of Bacteria
        MrS = self.p['MrS']
        a = self.p['a']
        MrP = self.p['MrP']
        
        # Numerical integration
        # (for numerical integration, y0 must be numpy array)
        y0 = np.array([S0, X0])
        # y_int = fcn_euler_forward(diff,tspan,y0,h=dt)
        y_int = fcn_rk4(diff, tspan, y0, h=dt)
        
        # Retrieve results from numerical integration output
        t = y_int['t']              # time
        S = y_int['y'][0,:]      # first output (row 0, all columns)
        X = y_int['y'][1,:]      # second output (row 1, all columns)
        P = (-S/MrS)*a*MrP
        return {'t':t, 'S':S, 'X':X, 'P':P}

class PSB(Module):
    """ Module for Monod Kinetics:
        Basic formula of substrate oxidation
        
        dS/dt = -mu_max*(X/Y)*(S/(Ks+S))
        dX/dt = mu_max*X*(S/(Ks+S)) - b*X
        dP/dt = mu_max_P*(X/Y)*(S/(Kp+S))
    
    dS/dt = substrate utilization rate
    dX/dt = bacterial growth rate
    dP/dt = production rate
    
    Supporting equations to include the Dissolved Oxygen and Temperature
    disturbances factor    
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
        
        
        # Parameters
        mu_max0 = self.p['mu_max0']
        Ks = self.p['Ks']
        K_DO = self.p['K_DO']
        b20 = self.p['b20']
        teta_mu = self.p['teta_mu']
        teta_b = self.p['teta_b']
        Y = self.p['Y']
        kP = self.p['kP'] #fraction of N in phytoplankton dead bodies
    
        # -- Disturbances at instant _t
        DO = self.d['DO']
        T = self.d['T'] 
        f3 = self.d['f3'] #[g d-1] phytoplankton death rate due to intraspecific competition 
        f_P_prt = self.d['f_P_prt']
        f_P_sol = self.d['f_P_sol']
        _DO = np.interp(_t,DO[:,0],DO[:,1])
        _T = np.interp(_t,T[:,0],T[:,1])
        _f3 = np.interp(_t, f3[:,0], f3[:,1])
        _f_P_prt = np.interp(_t, f_P_prt[:,0], f_P_prt[:,1])
        _f_P_sol = np.interp(_t, f_P_sol[:,0], f_P_sol[:,1])
       
        # -- Controlled input
        Porgf = self.u['Porgf'] 
        if _t == 0:
            S += Porgf + _f3*kP +_f_P_prt    
        elif _t>0:
            S += _f3*kP +_f_P_prt
        else:
            S = 0
    
        #supporting equations
        b = b20*(teta_b**(_T-20))
        mu_max = mu_max0*(_DO/(_DO+K_DO))*(teta_mu**(_T-20))
        
        # Differential equations
        dS_dt = -mu_max*(X/Y)*(S/(Ks+S))
        dX_dt = (mu_max*(S/(Ks+S))-b)*X
        
        
       
        return np.array([dS_dt,dX_dt])

    # Define model outputs from numerical integration of differential equations.
    # This function is called by the Module method 'run'.
    def output(self,tspan):
        # Retrieve object properties
        dt = self.dt        # integration time step size
        diff = self.diff    # function with system of differential equations
        S0 = self.x0['S'] # initial condition of Substrate
        X0 = self.x0['X'] # initial condiiton of Bacteria
        #physical constants
        MrPhytate = 647.94           #[g/mol] Molecular Weight of C6H6O24P6
        MrH2PO4 = 98.00         #[g/mol] Molecular Weight of H2PO4
        
        a = self.p['a']
        
        # Numerical integration
        # (for numerical integration, y0 must be numpy array)
        y0 = np.array([S0, X0])
        # y_int = fcn_euler_forward(diff,tspan,y0,h=dt)
        y_int = fcn_rk4(diff, tspan, y0, h=dt)
        
        # Retrieve results from numerical integration output
        t = y_int['t']              # time
        S = y_int['y'][0,:]      # first output (row 0, all columns)
        X = y_int['y'][1,:]      # second output (row 1, all columns)

        P = (-S/MrPhytate)*a*MrH2PO4
        
        return {'t':t, 'S':S, 'X':X, 'P':P}

class NIB(Module):
    """ Module for Decomposition and Nitrification:
        Using Monod kinetics
        
        dS/dt = -mu_max*(X/Y)*(S/(Ks+S))
        dX/dt = mu_max*X*(S/(Ks+S)) - b*X
        dP/dt = mu_max_P*(X/Y)*(S/(Kp+S))
    
    dS/dt = substrate utilization rate
    dX/dt = bacterial growth rate
    dP/dt = production rate
    
    Supporting equations to include the Dissolved Oxygen and Temperature
    disturbances factor    
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
        self.f_keys = ('SNH4', 'SNO2')

    # Define system of differential equations of the model
    def diff(self,_t,_x0):
        if not self.active:
            return None
        
        # State variables
        
        SNprt = _x0[0]
        SNH4 = _x0[1]
        SNO2 = _x0[2]
        XDB = _x0[3]
        XAOB = _x0[4]
        XNOB = _x0[5]
        SNO3 = _x0[6]
        
        
        #-- physical constants
        MrUrea = 60.07 #[g/mol] Molecular weight of Urea
        MrNH4 = 18.05 #[g/mol] Molecular weight of Ammonium ion
        MrNO2 = 46.01 #[g/mol] Molecular weight of Nitrite
        MrNO3 = 62.01 #[g/mol] Molecular weight of Nitrate
        K_DO = 1 #[g m-3] half-velocity constant DO for bacteria
        
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
        f_N_prt = self.d['f_N_prt'] #fish solid excretion from fish
        f_TAN = self.d['f_TAN'] #[g d-1] TAN flow rate from fish 
        f3 = self.d['f3'] #[g d-1] phytoplankton death rate due to intraspecific competition 
        
        _DO = np.interp(_t,DO[:,0],DO[:,1])
        _T = np.interp(_t,T[:,0],T[:,1])
        _f_N_prt = np.interp(_t, f_N_prt[:,0], f_N_prt[:,1])
        _f_TAN = np.interp(_t, f_TAN[:,0], f_TAN[:,1])
        _f3 = np.interp(_t, f3[:,0], f3[:,1])
        
        #supporting equations
        Norgf = self.u['Norgf']
        if _t ==0: 
            SNprt = _f_N_prt + _f3*kN + Norgf
        elif _t>0:
            SNprt = _f_N_prt + _f3*kN
        else:
            SNprt = 0
        
        #T and DO effects on decay coeffients and maximum rate of substrate use
        
        #DB
        bDB = b20DB*(teta_bDB**(_T-20))
        mu_maxDB = mu_max0DB*(_DO/(_DO+K_DO))*(teta_muDB**(_T-20))
        
        #AOB
        bAOB = b20AOB*(teta_bAOB**(_T-20))
        mu_maxAOB = mu_max0AOB*(_DO/(_DO+K_DO))*(teta_muAOB**(_T-20))
        
        #NOB
        bNOB = b20NOB*(teta_bNOB**(_T-20))
        mu_maxNOB = mu_max0NOB*(_DO/(_DO+K_DO))*(teta_muNOB**(_T-20))
        
        # Differential equations
        #Decomposition bacteria
        dSNprt_dt = -mu_maxDB*(XDB/YDB)*(SNprt/(KsDB+SNprt))
        dXDB_dt = (mu_maxDB*(SNprt/(KsDB+SNprt))-bDB)*XDB
        
        SNH4 = (-dSNprt_dt/MrUrea)*2*MrNH4 + _f_TAN  #Ammonium as a product
        
        #AOB
        dSNH4_dt = -mu_maxAOB*(XAOB/YAOB)*(SNH4/(KsAOB+SNH4))
        dXAOB_dt = (mu_maxAOB*(SNH4/(KsAOB+SNH4))-bAOB)*XAOB
        SNO2 = (-dSNH4_dt/MrNH4)*MrNO2
        
        #NOB
        dSNO2_dt = -mu_maxNOB*(XNOB/YNOB)*(SNO2/(KsNOB+SNO2))
        dXNOB_dt = (mu_maxNOB*(SNO2/(KsNOB+SNO2))-bNOB)*XNOB
        SNO3 = (-dSNO2_dt/MrNO2)*MrNO3

        # -- Store flows [g m-3 d-1]
        idx = np.isin(self.t, _t)
        self.f['SNH4'][idx] = SNH4
        self.f['SNO2'][idx] = SNO2
        
        return np.array([dSNprt_dt, dSNH4_dt, dSNO2_dt, 
                         dXDB_dt, dXAOB_dt, dXNOB_dt, SNO3])

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
        SNO30 = self.x0['SNO3']

        # Numerical integration
        # (for numerical integration, y0 must be numpy array)
        y0 = np.array([SNprt0, SNH40, SNO20, XDB0, XAOB0, XNOB0, SNO30])
        y_int = fcn_euler_forward(diff,tspan,y0,h=dt)
        # y_int = fcn_rk4(diff, tspan, y0, h=dt)
        
        # Retrieve results from numerical integration output
        t = y_int['t']              # time
        SNprt = y_int['y'][0,:]      # first output (row 0, all columns)
        SNH4 = y_int['y'][1,:]      # second output (row 1, all columns)
        SNO2 = y_int['y'][2,:]
        XDB = y_int['y'][3,:]
        XAOB = y_int['y'][4,:]
        XNOB = y_int['y'][5,:]
        SNO3 = y_int['y'][6,:]
        
        return {'t':t, 
                'SNprt':SNprt,
                'SNH4':SNH4,
                'SNO2':SNO2,
                'XDB':XDB,
                'XAOB':XAOB,
                'XNOB':XNOB,
                'SNO3':SNO3
                }