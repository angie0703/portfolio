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
     
    
    dS/dt = substrate utilization rate
    dX/dt = bacterial growth rate
    
    Decomposer bacteria:
    dN_prt_dt = -mu_max_DO*(X_DB/Y_DB)*(SOM/(Ks+SOM))
    dXDB_dt = mu_max_DO*X_DB*(SOM/(Ks+SOM))-(b_DB*X_DB)

    Heterotrophic bacteria convert organic substrates into 
    ammonium ion for nitrification process.

    AOB:
    - Ammonia to Nitrite (modified from Monod Kinetics)
    dS_NH4/dt = -mu_max_AOB*(X_AOB/Y_AOB)*(S_NH4/(K_NH4+S_NH4))
    dX_AOB/dt = mu_max_AOB*X_AOB*(S_NH4/(K_NH4+S_NH4))-(b_AOB*X_AOB)
    
    NOB:
    - Nitrite to Nitrate (modified from Monod Kinetics)
    dS_NO2/dt = -mu_max_NOB*(X_NOB/Y_NOB)*(S_NO2/(K_NO2+S_NO2))
    dX_NOB/dt = mu_max_NOB*X_NOB*(S_NO2/(K_NO2+S_NO2))-(b_NOB*X_NOB)
    
    PSB:
    dP_prt/dt = -mu_max_PSB*X_PSB/Y_PSB*(P_prt/(Kprt+P_prt))           
    dX_PSB/dt = mu_max_PSB*X_PSB*(P_prt/(Kprt+P_prt)) - b_PSB*X_PSB  
    
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
        * mu_max : maximum rate of substrate use (g substrate/g microorganisms.d)
        * Ks : half velocity constant (g L-1)
        * Y : bacteria synthesis yield (g bacteria biomass/ g substrate used)
        * b : specific endogenous decay coefficient [g VSS/g VSS.d]
        
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
        mu_max_S20 = self.p['mu_max_S20']
        # mu_max_P20 = self.p['mu_max_P20']
        Ks = self.p['Ks']
        # Kp = self.p['Kp']
        K_DO = self.p['K_DO']
        Y = self.p['Y']
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
        # mu_max_P = mu_max_P20*(teta_mu**(_T-20))
        b = b20*(teta_b**(_T-20))
        mu_max_S_DO = mu_max_S*(_DO/(_DO+K_DO))
        
        # Differential equations
        dS_dt = -mu_max_S_DO*(X/Y)*(S/(Ks+S))
        dX_dt = mu_max_S_DO*X*(S/(Ks+S))-(b*X)
        
        
        return np.array([dS_dt,dX_dt])

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