# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 19:46:50 2024

@author: alegn
"""

import numpy as np

from models.module import Module
from models.integration import fcn_euler_forward, fcn_rk4
from models.bacterialgrowth import NIB, PSB
from models.fish import Fish
from models.phytoplankton import Phygrowth

class fishpond(Module):
    """ Module for all process in the fishpond, consists of
    - Bacteria
    - Phytoplankton
    - Fish
    
    The output would be the N and P available for the rice plants
    The external disturbances involve 
    - light intensity
    - water temperature
    - Dissolved Oxygen level
    
    The controllable inputs:
    - N and P content of organic fertilizers
    - Artificial fish feed
    
    Flow rates inside the system:
           
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
        self.f_keys = (
                       #phytoplankton
                       'f1', 'f2', 'f3', 'f4', 'f5',
                       #fish
                       'f_upt', 'f_prt', 'f_sol', 'f_digout', 'f_N_sol', 
                       'f_P_sol','f_diguri', 'f_fed', 'f_N_upt', 
                       'f_P_upt', 'f_N_prt', 'f_P_prt', 'f_TAN', 'r_phy',
                       )
        for k in self.f_keys:
            self.f[k] = np.full((self.t.size,), np.nan)
            

    # Define system of differential equations of the model
    def diff(self,_t,_x0):
        # State variables
        
        SNprt = _x0[0]
        SPprt = _x0[1]
        XDB = _x0[2]
        XAOB = _x0[3]
        XNOB = _x0[4]
        XPSB = _x0[5]
        SNH4 = _x0[6]
        SNO2 = _x0[7]
        SNO3 = _x0[8]
        SP   = _x0[9]
        Mphy = _x0[10]
        NA   = _x0[11] 
        Mfish = _x0[12]
        Mdig = _x0[13]
        Muri=  _x0[14]
        
        # -- Physical constants
        DOmin = 0.3            # [mg/l] DO minimal for open-pond Nile Tilapia
        DOcrit = 1             # [mg/l] DO critical for open-pond Nile Tilapia
        Tminf = 22             # [] min temperature for fish growth
        Toptf = 28             # [] optimal temperature for fish growth
        Tmaxf = 32             # [] max temperature for fish growth
        Toptp = 28             # [] optimal temperature for phytoplankton growth
        MrPhytate = 647.94     #[g/mol] Molecular Weight of C6H6O24P6
        MrH2PO4 = 98.00        #[g/mol] Molecular Weight of H2PO4
        MrUrea = 60.07         #[g/mol] Molecular weight of Urea
        MrNH4 = 18.05          #[g/mol] Molecular weight of Ammonium ion
        MrNO2 = 46.01          #[g/mol] Molecular weight of Nitrite
        MrNO3 = 62.01          #[g/mol] Molecular weight of Nitrate
        K_DO = 1               #[g m-3] half-velocity constant DO for bacteria
        
        # Parameters DB
        mu_max0DB = self.p['mu_max0DB']
        KsDB = self.p['KsDB']
        b20DB = self.p['b20DB']
        teta_muDB = self.p['teta_muDB']
        teta_bDB = self.p['teta_bDB']
        YDB = self.p['YDB']
        
        # Parameters AOB
        mu_max0AOB = self.p['mu_max0AOB']
        KsAOB = self.p['KsAOB']
        b20AOB = self.p['b20AOB']
        teta_muAOB = self.p['teta_muAOB']
        teta_bAOB = self.p['teta_bAOB']
        YAOB = self.p['YAOB']
        
        #Parameters NOB
        mu_max0NOB = self.p['mu_max0NOB']
        KsNOB = self.p['KsNOB']
        b20NOB = self.p['b20NOB']
        teta_muNOB = self.p['teta_muNOB']
        teta_bNOB = self.p['teta_bNOB']
        YNOB = self.p['YNOB']
        
        # Parameters PSB
        mu_max0PSB = self.p['mu_max0PSB']
        KsPSB = self.p['KsPSB']
        b20PSB = self.p['b20PSB']
        teta_muPSB = self.p['teta_muPSB']
        teta_bPSB = self.p['teta_bPSB']
        YPSB = self.p['YPSB']
        
        kN = self.p['kN'] #[-] fraction of N content in phytoplankton dead bodies
        kP = self.p['kP'] #[-] fraction of N content in phytoplankton dead bodies
        
        # Phytoplankton Parameters
        mu_phy = self.p['mu_phy'] #[d-1] maximum growth rate of phytoplankton
        mu_Up = self.p['mu_Up'] #[d-1] maximum nutrient uptake coefficientd
        l_sl = self.p['l_sl'] #[m2 g-1] phytoplankton biomass-specific light attenuation
        l_bg = self.p['l_bg'] #[m-1] light attenuation by non-phytoplankton components
        Kpp = self.p['Kpp'] # [J (m2 s)-1]half-saturation constant of phytoplankton production
        cm = self.p['cm'] #[d-1] phytoplankton natural mortality constant
        cl = self.p['cl'] #[m3 (d g)-1] phytoplankton crowding loss constant
        c1 = self.p['c1'] # [-] temperature coefficients 1
        c2 = self.p['c2'] # [-] temperature coefficients 2
        Mp = self.p['Mp'] #[g m-3] half saturation constant for nutrient uptake
        
        # - fish growth
        tau_dig = self.p['tau_dig']  # [h] time constants for digestive
        tau_uri = self.p['tau_uri']  # [h] time constants for urinary
        k_N_upt = self.p['k_N_upt']  # [-] fraction of N uptake by fish
        k_P_upt = self.p['k_P_upt']  # [-] fraction of P uptake by fish
        k_upt = self.p['k_upt']  # [-] fraction of nutrient uptake for fish weight
        k_prt = self.p['k_prt']  # [-] fraction of particulate matter excreted
        k_N_prt = self.p['k_N_prt']  # [-] fraction of N particulate matter excreted
        k_P_prt = self.p['k_P_prt']  # [-] fraction of P particulate matter excreted
        x_N_fed = self.p['x_N_fed'] #[-] fraction of N in feed
        x_P_fed = self.p['x_P_fed'] #[-] fraction of P in feed
        k_DMR = self.p['k_DMR']
        k_TAN_sol = self.p['k_TAN_sol']
        Ksp = self.p['Ksp'] #[g m-3] Half-saturation constant for phytoplankton feeding 
        k_N_sol = 1 - k_N_upt - k_N_prt # [-] fraction of N in soluble matter
        k_P_sol = 1 - k_P_upt - k_P_prt # [-] fraction of P in soluble matter
        
        # -- Disturbances at instant _t
        DO = self.d['DO']
        T = self.d['T'] 
        I0 = self.d['I0']
        Rain = self.d['Rain']
        DVS = self.d['DVS']
        
        _DO = np.interp(_t,DO[:,0],DO[:,1])
        _T = np.interp(_t,T[:,0],T[:,1])
        _I0 = np.interp(_t,I0[:,0],I0[:,1])
        _Rain = np.interp(_t, Rain[:,0], Rain[:,1]) #[mm] Daily precipitation
        _DVS = np.interp(_t, DVS[:,0], DVS[:,1])
        
        pd = 0.501 if 0 <= _DVS < 0.4 else 0.511 if 0.4 <= _DVS < 0.65 else 0.601   
        print('pond level: ', pd)
        # dr = 0.8 if 0 <= _Rain < 0.5 else 0.45 if 0.5 <= _Rain < 50 else 0.21 # derived from available dilution rate value and rainfall category from BMKG
        dilution_rate = [0.8, 0.72, 0.45, 0.21] #[d-1]
        rainfall_category = [ 0, 20, 100, 150] #[mm]
        
        def estimate_dilution(_Rain):
            return np.interp(_Rain, rainfall_category, dilution_rate)
        
        dr = estimate_dilution(_Rain)
        
        #--Supporting equations
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
        
        #PSB
        bPSB = b20PSB*(teta_bPSB**(_T-20))
        mu_maxPSB = mu_max0PSB*(_DO/(_DO+K_DO))*(teta_muPSB**(_T-20))
        
        #PHYTOPLANKTON
        Iws = _I0*0.5
        k_In = Iws*np.exp(-pd*(l_sl*Mphy+l_bg))
        k_lm = (1/(pd*(l_sl*Mphy+l_bg)))*np.log((Kpp+Iws)/(Kpp+k_In))
        
        kTw = c1*np.exp(-c2*np.abs(_T-Toptp))
        
            
        NA = SNH4 + SNO2 + SNO3 + SP #[g m-3 d-1]
        kNA = NA/(NA+Mp)
        
        f1 = mu_phy*k_lm*kTw*kNA*Mphy
        f2 = cm*Mphy
        f3 = cl*(Mphy**2)
        f4 = dr*NA
        f5 = mu_Up*k_lm*kTw*kNA*Mphy 
        
       
        #Fish Model
        # [gDM h-1] total mass flow rate leaving the digestive system
        f_digout = (1/tau_dig)*Mdig
        # [gDM h-1] nutrient uptake rate
        f_upt = k_upt*f_digout
        # [gDM h-1] particulate matter excretion rate
        f_prt = k_prt*f_digout
        # [gDM h-1] soluble waste excretion rate
        f_sol = (1/tau_uri)*Muri
        # [gDM h-1] digestive to urinary rate
        f_diguri = f_digout - f_upt - f_prt
        
        #Nutrient Balances
        #N excretion flow rate by fish during growth
        f_N_prt = k_N_prt*x_N_fed*f_digout
        #P excretion flow rate by fish during growth
        f_P_prt = k_P_prt*x_P_fed*f_digout
        #N flow rate in soluble matter
        f_TAN = k_N_sol*x_N_fed*(f_digout/f_diguri)*f_sol*k_TAN_sol
        #P flow rate in soluble matter
        f_P_sol = k_P_sol*x_P_fed*(f_digout/f_diguri)*f_sol    
        
        # -- Supporting equations
        #Mfed = Mfish*0.03 #recommended amount of feed is 3% of the biomass weight
        fT = 0
        if _T >= Toptf:
            fT = np.exp(-4.6*(((_T-Toptf)/(Tmaxf-Toptf))**4))
        else:
            fT = np.exp(-4.6*(((Toptf-_T)/(Toptf-Tminf))**4))
        
        fDO = 0
        if DOmin <= _DO <= DOcrit:
            fDO = (_DO-DOmin)/(DOcrit - DOmin)
        else:
            fDO = 1
        
        Mfed = self.u['Mfed'] #[g] Artifical feed
        #-feed intake rate with disturbances factor
        r_phy = Mphy/(Ksp+Mphy) #[-] ratio of phytoplankton eaten by fish
        fed = Mphy + Mfed
        f_fed = fT*fDO*(r_phy*fed + (1-r_phy)*fed)
        
        #supporting equations
        # Controllable inputs
        Norgf = self.u['Norgf']
        if _t ==0: 
            if f3 == 0:
                SNprt += f_N_prt + Norgf
                if f_N_prt == 0:
                    SNprt += Norgf
            if f_N_prt == 0:
                SNprt += f3*kN + Norgf
                if f3 == 0:
                    SNprt += Norgf
        elif _t>0:
            if f3 ==0:
                SNprt += f_N_prt
            else: 
                SNprt = f_N_prt + f3*kN
        else:
            SNprt = 0
        
        Porgf = self.u['Porgf'] 
        if _t ==0: 
            if f3 == 0:
                SPprt += f_P_prt + Porgf
                if f_P_prt == 0:
                    SPprt += Porgf
            if f_P_prt == 0:
                SPprt += f3*kP + Porgf
                if f3 == 0:
                    SPprt += Porgf
        elif _t>0:
            if f3 ==0:
                SPprt += f_P_prt
            else: 
                SPprt = f_P_prt + f3*kP
        else:
            SPprt = 0

        # Differential equations
        # NIB
        #Decomposition bacteria
        dSNprt_dt = -mu_maxDB*(XDB/YDB)*(SNprt/(KsDB+SNprt))
        dXDB_dt = (mu_maxDB*(SNprt/(KsDB+SNprt))-bDB)*XDB
        
        if f_TAN ==0:
            SNH4 = (-dSNprt_dt/MrUrea)*2*MrNH4
        else:
            SNH4 = (-dSNprt_dt/MrUrea)*2*MrNH4 + f_TAN 
        
        #AOB
        dSNH4_dt = -mu_maxAOB*(XAOB/YAOB)*(SNH4/(KsAOB+SNH4))
        dXAOB_dt = (mu_maxAOB*(SNH4/(KsAOB+SNH4))-bAOB)*XAOB
        SNO2 = (-dSNH4_dt/MrNH4)*MrNO2
        
        #NOB
        dSNO2_dt = -mu_maxNOB*(XNOB/YNOB)*(SNO2/(KsNOB+SNO2))
        dXNOB_dt = (mu_maxNOB*(SNO2/(KsNOB+SNO2))-bNOB)*XNOB
        SNO3 = (-dSNO2_dt/MrNO2)*MrNO3
        
        # PSB
        dSPprt_dt = -mu_maxPSB*(XPSB/YPSB)*(SPprt/(KsPSB+SPprt))
        dXPSB_dt = (mu_maxPSB*(SPprt/(KsPSB+SPprt))-bPSB)*XPSB
        
        if f_P_sol == 0:
            SP = (-dSNprt_dt/MrPhytate)*2*MrH2PO4
        else:
            SP = (-dSNprt_dt/MrPhytate)*2*MrH2PO4 + f_P_sol 
        
        #Phytoplankton
        dMphy_dt = f1 - f2 - f3
        dNA_dt = f4 - f5
        
        # Ensure NA does not go below zero
        if dNA_dt < 0 and NA + dNA_dt < 0:
           dNA_dt = 0  # Set dNA_dt to -NA to ensure NA does not go below zero
            
        #FISH
        dMfish_dt = f_upt
        dMdig_dt = f_fed - f_digout
        dMuri_dt = f_diguri - f_sol

        # -- Store flows [g m-3 d-1]
        idx = np.isin(self.t, _t)
        
        #phytoplankton
        self.f['f1'][idx] = f1
        self.f['f2'][idx] = f2
        self.f['f3'][idx] = f3
        self.f['f4'][idx] = f4
        self.f['f5'][idx] = f5
        #fish
        self.f['f_digout'][idx] = f_digout
        self.f['f_fed'][idx] = f_fed
        self.f['f_upt'][idx] = f_upt
        self.f['f_prt'][idx] = f_prt
        self.f['f_sol'][idx] = f_sol
        self.f['f_diguri'][idx] = f_diguri
        self.f['f_N_prt'][idx] = f_N_prt
        self.f['f_P_prt'][idx] = f_P_prt
        self.f['f_P_sol'][idx] = f_P_sol
        self.f['f_TAN'][idx] = f_TAN
        self.f['r_phy'][idx] = r_phy
        
        
        return np.array([dSNprt_dt, dSPprt_dt, dSNH4_dt, dSNO2_dt, 
                         dXDB_dt, dXAOB_dt, dXNOB_dt, dXPSB_dt, 
                         SNO3, SP,
                         dMphy_dt,
                         dNA_dt, 
                         dMfish_dt,
                         dMdig_dt, 
                         dMuri_dt,
                         ])

    # Define model outputs from numerical integration of differential equations.
    # This function is called by the Module method 'run'.
    def output(self,tspan):
        # Retrieve object properties
        dt = self.dt        # integration time step size
        diff = self.diff    # function with system of differential equations
        SNprt0 = self.x0['SNprt'] # initial condition of Substrate
        SPprt0 = self.x0['SPprt']
        XDB0 = self.x0['XDB']
        XAOB0 = self.x0['XAOB']
        XNOB0 = self.x0['XNOB']
        XPSB0 = self.x0['XPSB']
        SNH40 = self.x0['SNH4'] # initial condition 
        SNO20 = self.x0['SNO2']
        SNO30 = self.x0['SNO3']
        SP0 = self.x0['SP']
        Mphy0 = self.x0['Mphy']
        NA0 = self.x0['NA']
        Mfish0 = self.x0['Mfish'] # initial condition
        Mdig0 = self.x0['Mdig'] # initial condiiton
        Muri0 = self.x0['Muri'] #initial condition
        
        # Numerical integration
        # (for numerical integration, y0 must be numpy array)
        y0 = np.array([SNprt0, SPprt0, XDB0, XAOB0, XNOB0,XPSB0, 
                       SNH40, SNO20,SNO30, SP0,
                       Mphy0,NA0,
                       Mfish0, Mdig0,Muri0
                       ])
        # y_int = fcn_euler_forward(diff,tspan,y0,h=dt)
        y_int = fcn_rk4(diff, tspan, y0, h=dt)
        
        # Retrieve results from numerical integration output
        t = y_int['t']              # time
              # first output (row 0, all columns)
        SNprt = y_int['y'][0,:]
        SPprt = y_int['y'][1,:]
        XDB = y_int['y'][2,:]
        XAOB = y_int['y'][3,:]
        XNOB = y_int['y'][4,:]
        XPSB = y_int['y'][5,:]
        SNH4 = y_int['y'][6,:]      # second output (row 1, all columns)
        SNO2 = y_int['y'][7,:]
        SNO3 = y_int['y'][8,:]
        SP = y_int['y'][9,:]
        Mphy = y_int['y'][10,:]      # first output (row 0, all columns)
        NA = y_int['y'][11,:] 
        Mfish = y_int['y'][12,:]
        Mdig = y_int['y'][13,:]
        Muri = y_int['y'][14,:]
        
        return {'t':t, 
                'SNprt':SNprt,
                'SPprt': SPprt,
                'XDB':XDB,
                'XAOB':XAOB,
                'XNOB':XNOB,
                'XPSB': XPSB,
                'SNH4':SNH4,
                'SNO2':SNO2,
                'SNO3':SNO3,
                'SP': SP,
                'Mphy':Mphy, 
                'NA':NA,
                'Mfish': Mfish,
                'Mdig': Mdig,
                'Muri': Muri,
                }