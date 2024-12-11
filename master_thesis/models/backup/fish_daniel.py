#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Daniel Reyes Lastiri, Jan-David Wacker
Fish model
"""
### TODO: flows as arrays? (iterating over dictionary keys without y shape yet)

### TODO: get data for average growth (delta_m) in one day, and min-max.

import numpy as np

from vorsetmembership.classes.module import Module
from vorsetmembership.functions.integration import fnc_ef

class Fish(Module):
    """Model for fish nutrient balances::

        dm_fsh/dt = f_upt
        dm_dig/dt = f_fed - f_dig_out
        dm_uri/dt = f_dig_uri - f_uri_out
    
    with f_dig_out = f_upt + f_dig_uri + f_fcs.
    
    Parameters
    ----------
    t : 1D array
        A sequence of time points for the simulation
    x0 : dictionary of floats
        Initial conditions of the state variables \n
        * m_fsh: total mass of fish in tank
        * m_dig : total mass in digestive tracts of fish
        * m_uri : total mass in urinary tracts of fish
    p : dictionary of scalars
        Model parameters \n
        * tau_dig : Time constant of the digestive tract [hr]
        * tau_uri : Time constant of the urinary tract [hr]
        * k_fcs : Ratio of mass flowing to faeces
        * k_upt : Ratio of mass becoming fish uptake for growth
    u : dictionary of 2-column arrays
        Time-series for external control inputs with shape (tu,2),
        with column 0 for time, and column 1 for values of control input
        * feed : mass flow of feed [kg hr-1]
    
    Returns
    -------
    y : dictionary
        Model outputs as 1D arrays for evaluation time 't'
    """
    # Initialize object. Inherit methods from object Module
    def __init__(self,tsim,dt,x0,p):
        Module.__init__(self,tsim,dt,x0,p)
        # Tuple of nutrients
        self.j = ('C', 'N', 'P', 'K', 'Ca', 'Mg', 'Na')
        # Initialize dictionary of flows
        self.f = {}
        self.f_keys = ('f_fcs', 'f_uri', 'f_dig_out')
        for k in self.f_keys:
            self.f[k] = np.full((self.t.size,), np.nan)
        # Intialize dictionary of flows per nutrient
        self.f_j = {}
        self.f_j_keys = ('f_j_upt', 'f_j_fcs', 'f_j_uri')
        for k in self.f_j_keys:
            self.f_j[k] = np.full((self.t.size,len(self.j)), np.nan)
    
    # Define system of differential equations of the model
    def diff(self, _t, _y0):
        # State variables
        m_fsh, m_dig, m_uri = _y0
        # Parameters
        tau_dig, tau_uri = self.p['tau_dig'], self.p['tau_uri']
        k_fcs, k_upt = self.p['k_fcs'], self.p['k_upt'] # 0.4, min(k_upt, 0.99-k_fcs)
        x_j_fed = self.p['x_j_fed'] # feed composition
        k_j_upt = self.p['k_j_upt'] # fraction from feed that ends as uptake
        k_j_fcs = self.p['k_j_fcs'] # fraction of feed that ends in faces
        k_j_uri = self.p['k_j_uri'] # fraction of feed that ends in urine
        # - Control input (feed), interpolated at instant _t
        # (if tsim != t_feed, change to zero-order interpolation
        # and adjust mass balance)
        t_fed = self.u['f_fed'][:,0]
        f_fed = self.u['f_fed'][:,1]
        _f_fed = np.interp(_t, t_fed, f_fed)
        # Supporting equations
        f_dig_out = m_dig*(1/tau_dig)
        f_uri = m_uri*(1/tau_uri) #+ 50*m_fsh/1E6 # Assuming fixed 50 mg kg-1 hr-1
        f_upt = k_upt*f_dig_out
        f_fcs = k_fcs*f_dig_out
        f_dig_uri = f_dig_out - f_upt - f_fcs
        # Differential equations
        dm_fsh_dt = f_upt
        dm_dig_dt = _f_fed - f_dig_out
        dm_uri_dt = f_dig_uri - f_uri
        # Store flows
        idx = np.isin(self.t, _t)
        self.f['f_fcs'][idx] = f_fcs
        self.f['f_uri'][idx] = f_uri
        self.f['f_dig_out'][idx] = f_dig_out
        # Store flows per nutrient
        self.f_j['f_j_upt'][idx] = k_j_upt*x_j_fed*f_dig_out 
        self.f_j['f_j_fcs'][idx] = k_j_fcs*x_j_fed*f_dig_out 
        # (fi_uri = Ci*fv_uri = fi_dig_uri/(f_dig_uri/rho)*(f_uri/rho))
        self.f_j['f_j_uri'][idx] = k_j_uri*x_j_fed*f_dig_out/f_dig_uri*f_uri
        return np.array([dm_fsh_dt, dm_dig_dt, dm_uri_dt])


    # Define model outputs from numerical integration of differential equations.
    # This function is called by the Module method 'run'.
    def output(self, tspan):
        # Retrieve object properties
        dt = self.dt        # integration time step size
        diff = self.diff    # function with system of differential equations
        # # feed composition
        # x_j_fed = self.p['x_j_fed']
        # # fraction from feed that ends as uptake
        # k_j_upt = self.p['k_j_upt']
        # # fraction of feed that ends in faces
        # k_j_fcs = self.p['k_j_fcs']
        # # fraction of feed that ends in urine
        # k_j_uri = self.p['k_j_uri']
        
        # Numerical integration
        # (for numerical integration, y0 must be numpy array)
        y0 = np.array([x0['m_fsh'] ,x0['m_dig'], x0['m_uri']])
        y_int = fnc_ef(diff, tspan, y0, h=dt) # dt must be 0.25 [hr]
        
        # - Retrieve integration outputs
        t_int = y_int['t']
        m_fsh, m_dig, m_uri = y_int['y'][0,:], y_int['y'][1,:], y_int['y'][2,:]
        # # Model outputs
        # f_dig_out = m_dig/tau_dig
        # f_upt = k_upt*f_dig_out
        # f_fcs = k_fcs*m_dig/tau_dig
        # f_dig_uri = f_dig_out - f_fcs - f_upt
        # f_uri = m_uri/tau_uri
        # # Nutrients
        # nj = x_j_fed.shape[0]
        # f_j_upt = np.full((self.t.size,nj), np.nan)
        # f_j_fcs = np.full((self.t.size,nj), np.nan)
        # f_j_uri = np.full((self.t.size,nj), np.nan)
        # for j,j_val in enumerate(x_j_fed):
        #     # Flows [g hr-1]
        #     f_j_upt[:,j] = k_j_upt[j]*x_j_fed[j]*f_dig_out # uptake
        #     f_j_fcs[:,j] = k_j_fcs[j]*x_j_fed[j]*f_dig_out #  faces
        #     # (fi_uri = Ci*fv_uri = fi_dig_uri/(f_dig_uri/rho)*(f_uri/rho))
        #     f_j_uri[:,j] = k_j_uri[j]*x_j_fed[j]*f_dig_out/f_dig_uri*f_uri #soluble
        
        return {'t':t_int, 'm_fsh':m_fsh, 'm_dig':m_dig, 'm_uri':m_uri}
                
# def model_fish_calibration(tsim,x0,p,d,u,*p_hat_0):
#     ''' Model for fish calibration.
#     Defined separately to add x0 as parameter, and measured model outputs.
    
#     Based on module model_fish
#         dm_fsh/dt = f_upt
#         dm_dig/dt = f_fed - f_dig_out
#         dm_uri/dt = f_dig_uri - f_uri_out
    
#     with f_dig_out = f_upt + f_dig_uri + f_fcs.
    
#     Parameters
#     ----------
#     t : 1D array
#         A sequence of time points for the simulation
#     x0 : dictionary of floats
#         Initial conditions of the state variables \n
#         * m_fsh: total mass of fish in tank
#         * m_dig : total mass in digestive tracts of fish
#         * m_uri : total mass in urinary tracts of fish
#     p : dictionary of scalars
#         Known model parameters \n
#     u : dictionary of 2-column arrays
#         Time-series for external control inputs with shape (tu,2),
#         with column 0 for time, and column 1 for values of control input
#         * feed : mass flow of feed [kg hr-1]
#     p_hat_0 : 1D array
#         Parameters to estimate (including x0)
    
#     Returns
#     -------
#     y : dictionary
#         Model outputs as 1D arrays for evaluation time 't'
    
#     '''
#     # - Initial conditions
#     x0['m_fsh'] = p_hat_0[3]
#     x0['m_uri'] = p_hat_0[4]
#     # - Known and fixed parameters (following from sensitivity analysis, SA)
#     dfr = p['dfr'] # Daily feed ratio
#     dmr = p['dmr'] # Dry mass ratio
#     k_fcs = p['k_fcs'] # Mass raction of faces leaving digestive system (SA)
#     # - Parameters for module model_fish
#     p_hat_model = np.array([p_hat_0[0],p_hat_0[1],k_fcs,p_hat_0[2]])
#     # - Inputs
#     # Feed 1x per day in first day, at start of simulation time.
#     # subsequently, feed 3x per day to match commercial growth (0 = 8:00).
#     # First feed is at -0.25 hr, to match dynamics of excretions at t=0 hr.
#     # subsequently, feed is given at start of each day (multiples of t=24 hr)
#     # Amount of feed based on quadratic growth model
#     # Feed eaten in 15 min (0.25 hr)
#     # Quadratic model: m = m0 + 0.239*t + 0.0075*t**2, with m in [g] and t in [d]
#     t_quad = np.linspace(tsim[1],tsim[-1]/24,4*24*tsim[-1]/24+2) #np.linspace(0,90,4*24*90+2) # [d]
#     m_fsh_quad = (x0['m_fsh']/dmr + 0.239*t_quad + 0.0075*t_quad**2) # [g] fresh mass
#     # Assign feed mass
#     t_of_day = (tsim%24)
#     f_feed = np.zeros_like(tsim)
    
#     f_feed[t_of_day==23.75] = dfr*m_fsh_quad[t_of_day==23.75]/0.25/3
#     f_feed[t_of_day==3.75] = dfr*m_fsh_quad[t_of_day==3.75]/0.25/3
#     f_feed[t_of_day==7.75] = dfr*m_fsh_quad[t_of_day==7.75]/0.25/3
#     f_feed[0] = dfr*x0['m_fsh']/dmr/0.25
#     f_feed[16] = 0 # remove feed assigned above for t=4 hr
#     f_feed[32] = 0 # remove feed assigned above for t=8 hr
#     u = {'f_fed':np.array([tsim,f_feed]).T}
#     # Run mass balance model
#     y_fsh = model_fish(tsim,x0,p,d,u,*p_hat_model)
#     # Outputs for calibration
#     # - mass of fish (fresh) [g]
#     # - mass in digestive system with respect to feed at t=0 [-]
#     # - TAN excretion with respect to feed at t=0 [mgN hr-1 per kg feed]
#     #   (assuming f_TAN_uri = 0.9*f_N_uri, 10% N is urea, Timmons 2010 p. 91)
#     y = {'m_fsh_fr':y_fsh['m_fsh']/dmr,
#          'm_dig_n':y_fsh['m_dig']/(dfr*x0['m_fsh']/dmr),
#          'f_TAN_n':1E6*0.9*y_fsh['f_j_uri'][:,1]/(dfr*x0['m_fsh']/dmr),
#          'f_j_upt':y_fsh['f_j_upt'],
#          'f_j_fcs':y_fsh['f_j_fcs'],
#          'f_j_uri':y_fsh['f_j_uri'],
#          'f_feed':f_feed
#          }
#     return y

# ---- Test Run ----
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Simulation time, initial conditions and parameters
    t = np.linspace(0, 7*24, 7*24*4+1) # [hr] 7 day, dt=15 min
    dt = 0.25 # [hr] (15 min)
    # - Parameteres
    
    # Fixed, known nutrient fractions parameters
    # C, N, P, K, Ca, Mg, Na
    # Data from Seawright (1998) Schneider (2004), Neto and Ostrensky (2015)
    # Mass fraction content in feed
    x_j_fed = np.array([0.40, 0.05, 0.01, 0.01, 0.02, 0.003, 0.005])
    # Mass fraction that ends in fish uptake
    k_j_upt = np.array([0.33, 0.44, 0.53, 0.24, 0.37, 0.21, 0.50])     
    # Mass fraction that ends in faeces (solids)
    k_j_fcs = np.array([0.15, 0.11, 0.43, 0.04, 0.26, 0.19, 0.13])         
    # Calculated remaining mass fraction that ends in soluble excretion
    k_j_uri = np.array([0.51, 0.44, 0.03, 0.72, 0.37, 0.61, 0.37])
    
    p = {'tau_dig':4.5, 'tau_uri':4.5, 'k_fcs':0.25, 'k_upt':0.2,
         'x_j_fed':x_j_fed, 'k_j_upt':k_j_upt,
         'k_j_fcs':k_j_fcs, 'k_j_uri':k_j_uri
         } 
    
    d = {}
    tau_dig, tau_uri = 4.5, 4.5 # [hr]
    k_fcs, k_upt = 0.25, 0.2    # [-]
    n_fish = 10
    x0 = {'m_fsh':n_fish*50.0,  # [g]
          'm_dig':n_fish*0.1,   # [g]
          'm_uri':n_fish*0.1    # [g]
          }
    # Feed (once per day at 08:00 hrs)
    dfr = 0.02
    t_feed = np.arange(0, 7*24+1, 0.25)
    t_of_day = np.remainder(t_feed, 24)
    f_feed = np.zeros_like(t_feed)
    f_feed[t_of_day==8] = dfr*x0['m_fsh']/0.25 # eaten in 15 min
    u = {'f_fed':np.array([t_feed,f_feed]).T}
    # Initialize object
    fish = Fish(t, dt, x0, p)
    # Run model
    tspan = (t[0], t[-1])
    y = fish.run(tspan, d=d, u=u)
    # Plot results
    t_plt = t/24 # [d]
    # Overall mass balances
    fig1,(ax1a,ax1b,ax1c) = plt.subplots(3,1, num=1, sharex=True)
    ax1a.plot(t,y['m_fsh'],label=r'$m_{fsh}$')
    ax1a.set_ylabel('mass '+ r'$[g]$')
    ax1a.legend()
    ax1b.plot(t,y['m_dig'],label=r'$m_{dig}$')
    ax1b.plot(t,y['m_uri'],label=r'$m_{uri}$')
    ax1b.legend()
    ax1b.set_ylabel('mass '+ r'$[g]$')
    ax1c.plot(t,fish.f['f_fcs'],label=r'$f_{fcs}$')
    ax1c.plot(t,fish.f['f_uri'],label=r'$f_{uri}$')
    ax1c.legend()
    ax1c.set_xlabel('time '+ r'$[hr]$')
    ax1c.set_ylabel('mass flows '+ r'$[g\ hr^{-1}]$')
    # Nutrient mass balances
    fig2,(ax2a,ax2b,ax2c) = plt.subplots(3,1, num=2, sharex=True)
    ax2a.plot(t,fish.f_j['f_j_upt'][:,0],label=r'$f_{C,upt}$')
    ax2a.plot(t,fish.f_j['f_j_fcs'][:,0],label=r'$f_{C,fcs}$')
    ax2a.plot(t,fish.f_j['f_j_uri'][:,0],label=r'$f_{C,uri}$')
    ax2a.legend()
    ax2b.plot(t,fish.f_j['f_j_upt'][:,1],label=r'$f_{N,upt}$')
    ax2b.plot(t,fish.f_j['f_j_fcs'][:,1],label=r'$f_{N,fcs}$')
    ax2b.plot(t,fish.f_j['f_j_uri'][:,1],label=r'$f_{N,uri}$')
    ax2b.set_ylabel('mass flows '+ r'$[g\ hr^{-1}]$')
    ax2b.legend()
    ax2c.plot(t,fish.f_j['f_j_upt'][:,2],label=r'$f_{P,upt}$')
    ax2c.plot(t,fish.f_j['f_j_fcs'][:,2],label=r'$f_{P,fcs}$')
    ax2c.plot(t,fish.f_j['f_j_uri'][:,2],label=r'$f_{P,uri}$')
    ax2c.legend()
    ax2c.set_xlabel('time '+ r'$[hr]$')


