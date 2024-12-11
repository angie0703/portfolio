# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 09:21:33 2023

@authors:   Angela

Class for QUEFTS model (Quantitative Evaluation of Fertility of Tropical Soils) 
adapted from Janssen et. al. (1990) and Sattari et. al. (2014)
Model for rice yield in relation of N, P, and K nutrient uptake from fish ecosystems and fertilizers from farmers

Only focus on irrigated rice plants in tropical climate. Irrigated rice plants means 
the soils are under flooded conditions, similar to rice cultivation in
integrated rice and fish farming system.
Therefore, flooding factor f_F is multiplied with alpha_P, beta_P, and alpha_K.



FRAMEWORK OF THE QUEFTS MODEL
Step 1: Assessment of the amounts of available nutrients
Available nutrients may be supplied by soil and by inputs.
Estimate the potential soil supplies of available N, P, and K, based on organic C content,
Olsen-P, exchangeable K, and pH(H20) as independent variable. 
Optional: organic N content and total P content.

Step 2: Relation between supply of available nutrients and actual uptake
The actual uptake of each nutrient is calculated as a function of the 
potential supplies of a nutrient and the other nutrients. 
The relationship between supply and actual uptake of a nutrient is 
a theoretical relation consisting of three zones.
In initial zone (Zone I), the actual uptake of nutrient i (Ui) is equal to the supply ofthe nutrient (Si).
In Zone II, large values of Si does not lead to further increase of Ui.
In Zone III, the relation between Ui and Si is assumed to be parabolic.


Step 3: Relation between actual uptake and yield ranges
When the uptake of a nutrient is very small compared to other nutrients, 
maximum dilution of this nutrient occurs within the plant. When other nutrients are strongly limiting,
the nutrient under study starts to accumulate  in the plant up to a maximum mass fraction.

Step 4: Combining yield ranges to ultimate yield estimate

    
References:
Janssen B.H., F.C.T. Guiking, D. van der Eijk, E.M.A. Smaling, J. Wolf and H. van Reuler, 1990. 
A system for the quantitative evaluation of the fertility of tropical soils (QUEFTS). 
Geoderma 46: 299-318
Sattari, S.Z., M.K. van Ittersum, A.F. Bouwman, A.L. Smit, and B.H. Janssen, 2014. 
Crop yield response to soil fertility and N, P, K inputs in different environments: 
Testing and improving the QUEFTS model. Field Crops Research 157: 35-46
"""
import numpy as np

from models.module import Module
from models.integration import fcn_euler_forward

# Model definition
class QUEFTS(Module):
    """ 
    Step 1:Calculate Supplies of N, P, and K
    SN = alpha_N*f_N*C_org + I_N*R_N + 
    SP = q_P*f_P*P_tot + beta_P*P_Olsen+I_P*R_P
    SK = ((alpha_K*f_K*K_exc)/(gamma_K+beta_K*C_org)) + I_K*R_K
    
    alpha, beta, and gamma values are empirical values that need to be calibrated.
    
    Step 2: Calculate Actual Uptake of N, P, and K (flow) OR put all this equations in ouput without diff function
    Ui =
    (I) Si                     IF Si < ri + (Sj - rj)*(aj/di) 
    (II) ri+((Sj-rj)*(dj/ai))  IF Si > ri + (Sj - rj)*(2(dj/ai)-(aj/di))
    (III)Si - (0.25((Si-ri-(Sj-rj)*(aj/di)))**2)/((Sj-rj)*((dj/ai)-(aj/di)))
    
    Step 3: Calculate Yield Ranges
    Y_i_a = ai*(Ui - ri)
    Y_i_d = di*(Ui - ri)
    i = N, P, K
    
    Step 4: Calculate Yield Estimates and Ultimate Yield
    Yij = Y_j_a + ((2*(min(Y_j_d, Y_k_d, Ymax)-Y_j_a))*(Ui-ri-Y_j_a/di))/((min(Y_j_d, Y_k_d, Ymax)/ai)-(Y_j_a/di)))-((((min(Y_j_d, Y_k_d, Ymax)-Y_j_a))*(Ui-ri-Y_j_a/di))**2/((min(Y_j_d, Y_k_d, Ymax)/ai)-(Y_j_a/di))**2)
    Yij < Ymax
    YU = (YNP + YNK + YPN + YPK + YKN + YKP)/6
    
    Parameters
    ----------
    tsim : array
        Sequence of time points for the simulation
    dt : float
        Time step size [d]
    x0 : dictionary
        Initial conditions of the state variables \n
        
        ======  =============================================================
        key     meaning
        ======  =============================================================
        step 1:
        'SN'    [kg N/ha] Supplies of crop-available Nitrogen
        'SP'    [kg P/ha] Supplies of crop-available Phosphorus
        'SK'    [kg K/ha] Supplies of crop-available Potassium
        step 2:
        'UN'    [kg N/ha] Actual uptake of Nitrogen
        'UP'    [kg P/ha] Actual uptake of Phosphorus
        'UK'    [kg K/ha] Actual uptake of Potassium
        step 4:
        'YU'    [kg/ha] Ultimate Yield
        ======  =============================================================
        
    p : dictionary of scalars
        Model parameters \n
        
        Step 1:
         ======  =============================================================
         key     meaning
         ======  =============================================================
         'alpha_N' [-] empirical parameters for Nitrogen
         'alpha_P' [-] empirical parameters for Phosphorus
         'alpha_K' [-] empirical parameters for Potassium
         'beta_P' [-] empirical parameters for Phosphorus
         'beta_K' [-] empirical parameters for Potassium
         'gamma_K' [-] empirical parameters for Potassium
         'K_exc'  [mmol/kg] soil exchangeable Potassium
         'f_N'    [-] pH correction factor for Nitrogen 
         'f_P'    [-] pH correction factor for Phosphorus 
         'f_K'    [-] pH correction factor for Potassium
         'f_F'    [-] flooding factor
         'C_org'  [g/kg] soil organic carbon
         'P_Olsen'[mg/kg] Soil phosphorus (Olsen P)
         'P_tot'  [mg/kg] Total soil phosphorus
         'R_N'    [-] N maximum recovery fraction
         'R_P'    [-] P maximum recovery fraction
         'R_K'    [-] K maximum recovery fraction
         ======  =============================================================

        Step 2: additional parameters needed
        ======  =============================================================
        key     meaning
        ======  =============================================================
        'rN'    [kg/ha] minimum N uptake to produce any grain
        'rP'    [kg/ha] minimum P uptake to produce any grain
        'rK'    [kg/ha] minimum K uptake to produce any grain
        'aN'    [kg/kg] Physiological efficiency Nitrogen at maximum accumulation of N
        'aP'    [kg/kg] Physiological efficiency Phosphorus at maximum accumulation of P
        'aK'    [kg/kg] Physiological efficiency Potassium at maximum accumulation of K
        'dN'    [kg/kg] Physiological efficiency Nitrogen at maximum dilution of N
        'dP'    [kg/kg] Physiological efficiency Phosphorus at maximum dilution of P
        'dK'    [kg/kg] Physiological efficiency Potassium at maximum dilution of K
        ======  =============================================================
        
        Step 4: additional parameters
        ======  =============================================================
        key     meaning
        ======  =============================================================
        'Ymax'  [kg/ha] potential grain yield
        ======  =============================================================

        Step 5: additional parameters for model modification (only do this after finish with the basic QUEFTS)
        ======  =============================================================
        key     meaning
        ======  =============================================================
        'DM'    [kg/ha] dry matter production
        'XgN'   [g/kg] mass fraction of N in grain
        'XgP'   [g/kg] mass fraction of P in grain
        'XgK'   [g/kg] mass fraction of K in grain
        'XsN'   [g/kg] mass fraction of N in straw
        'XsP'   [g/kg] mass fraction of P in straw
        'XsK'   [g/kg] mass fraction of K in straw
        ======  =============================================================            

    d : dictionary of floats or arrays
        Model disturbances (required for method 'run'),
        of shape (len(t_d),2) for time and disturbance.
        
        =======  ============================================================
        key      meaning
        =======  ============================================================
         'pH'     [-] pH (water)
        =======  ============================================================

    u : dictionary of 2D arrays
        Controlled inputs (required for method 'run'),
        of shape (len(t_d),2) for time and controlled input.
        
        =======  ============================================================
        key      meaning
        =======  ============================================================
         'I_N'    [kg/ha] N fertilizer application
         'I_P'    [kg/ha] P fertilizer application
         'I_K'    [kg/ha] K fertilizer application
        =======  ============================================================ 

    Returns
    -------
    y : dictionary
        Model outputs as 1D arrays ('SN', 'SP', 'SK', 'UN', 'UP', 'UK', 'YU')
        and the evaluation time 't'.
    """
    # Initialize object. Inherit methods from object Module
    def __init__(self,tsim,dt,x0,p):
        Module.__init__(self,tsim,dt,x0,p)
        # Initialize dictionary of flows
        self.f = {}
        self.f_keys = ('f_N_upt', 'f_P_upt', 'f_K_upt')
        for k in self.f_keys:
            self.f[k] = np.full((self.t.size,), np.nan)
            
    # Define system of differential equations of the model
    def diff(self,_t,_x0):
        # State variables
        SN = _x0[0]    #[kg N/ha] Supplies of crop-available Nitrogen
        SP = _x0[1]    #[kg P/ha] Supplies of crop-available Phosphorus
        SK = _x0[2]    #[kg K/ha] Supplies of crop-available Potassium
        YU = _x0[3]    #[kg/ha] Ultimate Yield
        
        #parameters
        #for step 1
        alpha_N = self.p['alpha_N'] #[-] empirical parameters for Nitrogen
        alpha_P = self.p['alpha_P'] #[-] empirical parameters for Phosphorus
        alpha_K = self.p['alpha_K'] #[-] empirical parameters for Potassium
        beta_P = self.p['beta_P'] #[-] empirical parameters for Phosphorus
        beta_K = self.p['beta_K'] #[-] empirical parameters for Potassium
        f_F = self.p['f_F'] #[-] flooding factor
        gamma_K = self.p['gamma_K'] #[-] empirical parameters for Potassium
        K_exc = self.p['K_exc']  #[mmol/kg] soil exchangeable Potassium
        C_org = self.p['C_org']  #[g/kg] soil organic carbon
        P_Olsen = self.p['P_Olsen']#[mg/kg] Soil phosphorus (Olsen P)
        P_tot = self.p['P_tot']  #[mg/kg] Total soil phosphorus
        R_N = self.p['R_N']    #[-] N maximum recovery fraction
        R_P = self.p['R_P']    #[-] P maximum recovery fraction
        R_K = self.p['R_K']    #[-] K maximum recovery fraction
               
        #for step 2 onwards
        rN = self.p['rN']    #[kg/ha] minimum N uptake to produce any grain
        rP = self.p['rP']    #[kg/ha] minimum P uptake to produce any grain
        rK = self.p['rK']    #[kg/ha] minimum K uptake to produce any grain
        aN = self.p['aN']    #[kg/kg] Physiological efficiency Nitrogen at maximum accumulation of N
        aP = self.p['aP']    #[kg/kg] Physiological efficiency Phosphorus at maximum accumulation of P
        aK = self.p['aK']    #[kg/kg] Physiological efficiency Potassium at maximum accumulation of K
        dN = self.p['dN']    #[kg/kg] Physiological efficiency Nitrogen at maximum dilution of N
        dP = self.p['dP']    #[kg/kg] Physiological efficiency Phosphorus at maximum dilution of P
        dK = self.p['dK']    #[kg/kg] Physiological efficiency Potassium at maximum dilution of K
        
        #step 4
        Ymax = self.p['Ymax']  #[kg/ha] potential grain yield
        
        # -- Disturbances at instant _t
        #pH
        pH = self.d['pH']     #[-] pH (water)
        # _pH = np.interp(_t,pH[:,0],pH[:,1])     # [-] pH (water)
        
        #equations for f_N, f_P, and f_K
        f_N = 0.25*(pH - 3)
        f_P = 1-0.5*((pH-6)**2)
        f_K = 0.625*(3.4 - 0.4*pH)
        
        # -- Controlled inputs
        I_N = self.u['I_N']    #[kg/ha] N fertilizer application
        I_P = self.u['I_P']    #[kg/ha] P fertilizer application
        I_K = self.u['I_K']    #[kg/ha] K fertilizer application
        
        #Equations for Step 1
        SN = alpha_N*f_N*C_org + I_N*R_N
        SP = alpha_P*f_F*f_P*P_tot + beta_P*f_F*P_Olsen+I_P*R_P
        SK = ((alpha_K*f_F*f_K*K_exc)/(gamma_K+beta_K*C_org)) + I_K*R_K 
        
        #Equations for Step 2
        ##for calculating UN
        # Ui =
        # (I) Si                     IF Si < ri + (Sj - rj)*(aj/di) 
        # (II) ri+((Sj-rj)*(dj/ai))  IF Si > ri + (Sj - rj)*(2(dj/ai)-(aj/di))
        # (III)Si - (0.25((Si-ri-(Sj-rj)*(aj/di)))**2)/((Sj-rj)*((dj/ai)-(aj/di)))
       
        def Uij(Si, ri, Sj, rj, ai, di, aj, dj):
            """
            Calculate Uij based on the specified conditions.
        
            Parameters:
            - Si, Sj: Supplies of crop-available nutrient for element i and j
            - ri, rj: Constants for element i and j
            - ai, di: Constants for element i
            - aj, dj: Constants for element j
        
            Returns:
            - Uij: Calculated Uij value
            """
            condition1 = Si < ri + (Sj - rj) * (aj / di)
            condition2 = Si > ri + (Sj - rj) * (2 * (dj / ai) - (aj / di))
        
            result = np.select(
                [condition1, condition2],
                [Si, ri + ((Sj - rj) * (dj / ai))],
                Si - (0.25 * ((Si - ri - (Sj - rj) * (aj / di)) ** 2) / ((Sj - rj) * ((dj / ai) - (aj / di))))
            )
            return result

        #calculate nutrient     
        UN_P = Uij(SN, rN, SP, rP, aN, dN, aP, dP)
        UN_K = Uij(SN, rN, SK, rK, aN, dN, aK, dK)
        UN = np.minimum(UN_P, UN_K)

        UP_N = Uij(SP, rP, SN, rN, aP, dP, aN, dN)
        UP_K = Uij(SP, rP, SK, rK, aP, dP, aK, dK)
        UP = np.minimum(UP_N, UP_K)

        UK_N = Uij(SK, rK, SN, rN, aK, dK, aN, dN)
        UK_P = Uij(SK, rK, SP, rP, aK, dK, aP, dP)
        UK = np.minimum(UK_N,UK_P)
        
        #Equations for Step 3
        YNA = aN*(UN-rN)
        YPA = aP*(UP-rP)
        YKA = aK*(UK-rK)
        YND = dN*(UN-rN)
        YPD = dP*(UP-rP)
        YKD = dK*(UK-rK)
        
        #Equations for Step 4
        #Yij = Y_j_a + (((2*(min(Y_j_d, Y_k_d, Ymax)-Y_j_a))*(Ui-ri-Y_j_a/di))/((min(Y_j_d, Y_k_d, Ymax)/ai)-(Y_j_a/di)))-((((min(Y_j_d, Y_k_d, Ymax)-Y_j_a))*(Ui-ri-Y_j_a/di))**2/((min(Y_j_d, Y_k_d, Ymax)/ai)-(Y_j_a/di))**2)
        # Yij = Yja + (a1/b1) - (a2/b2)
        # a1= 2(min(Yjd, Ykd, Ymax)-Yja)*(Ui-ri-(Yja/di))
        # b1 = (min(Yjd, Ykd, Ymax)/ai)-(Yja/di)
        # a2 = (min(Yjd, Ykd, Ymax)-Yja)*((Ui-ri-(Yja/di))**2)
        # b2 = b1**2
        def Yij(Yja, Ui, ri, di, ai, Yjd, Ykd, Ymax):
            """
            Calculate Yij based on the specified equations.
        
            Parameters:
            - Yja: Y value for element j
            - Ui: Some value for element i
            - ri: Constant for element i
            - di: Constant for element i
            - ai: Constant for element i
            - Yjd, Ykd: Y values for element j and k
            - Ymax: Maximum Y value
        
            Returns:
            - Yij: Calculated Yij value
            """
            min_Y = np.minimum(np.minimum(Yjd, Ykd), Ymax)
            
            a1 = 2 * (min_Y - Yja) * (Ui - ri - (Yja / di))
            b1 = (min_Y / ai) - (Yja / di)
            a2 = (min_Y - Yja) * ((Ui - ri - (Yja / di)) ** 2)
            b2 = b1 ** 2
        
            Yij = Yja + (a1 / b1) - (a2 / b2)
            Yij_final = np.min(Yij, Ymax)
            
            return Yij_final
        
        YNP = Yij(YPA, UN, rN, dN, aN, YPD, YKD, Ymax)
        YNK = Yij(YKA, UN, rN, dN, aN, YPD, YKD, Ymax)
        YPN = Yij(YNA, UP, rP, dP, aP, YND, YKD, Ymax)
        YPK = Yij(YKA, UP, rP, dP, aP, YND, YKD, Ymax)
        YKN = Yij(YNA, UK, rK, dK, aK, YND, YPD, Ymax)
        YKP = Yij(YPA, UK, rK, dK, aK, YND, YPD, Ymax)
        YU = (YNP + YNK + YPN + YPK + YKN + YKP)/6

        return np.array([SN, SP, SK, YU])

    # Define model outputs from numerical integration of differential equations.
    # This function is called by the Module method 'run'.
    def output(self,tspan):
        # Retrieve object properties
        dt = self.dt        # integration time step size
        diff = self.diff    # function with system of differential equations
        SN0 = self.x0['SN'] # initial condition
        SP0 = self.x0['SP'] # initial condiiton
        SK0 = self.x0['SK'] # initial condiiton
        YU0 = self.x0['YU'] # initial condition
        
        # Numerical integration
        # (for numerical integration, y0 must be numpy array)
        y0 = np.array([SN0,SP0,SK0,YU0])
        y_int = fcn_euler_forward(diff,tspan,y0,h=dt)
         
        # Retrieve results from numerical integration output
        t = y_int['t']              # time
        SN = y_int['y'][0,:]      # first output (row 0, all columns)
        SP = y_int['y'][1,:]      # second output (row 1, all columns)
        SK = y_int['y'][2,:]      
        YU = y_int['y'][3,:]
        
        return {'t':t, 'SN':SN, 'SP':SP, 'SK':SK,'YU':YU}