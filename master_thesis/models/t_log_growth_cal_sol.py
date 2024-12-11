# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   Daniel Reyes Lastiri, Stefan Maranus,
            Rachel van Ooteghem, Tim Hoogstad

Tutorial for the calibration of the logistic growth model
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from mbps.models.log_growth import LogisticGrowth
from mbps.functions.calibration import fcn_residuals, fcn_accuracy

plt.style.use('ggplot')

# Simulation time array
tsim = np.linspace(0.0, 365.0, 365+1) # [d]
tspan = (tsim[0], tsim[-1])

# Initialize reference object
dt = 1.0                # [d] time-step size
x0 = {'m':0.01}         # [kgDM m-2] initial conditions
p = {'r':0.01,'K':1.0}  # [d-1], [kgDM m-2] model parameters (initial guess)
lg = LogisticGrowth(tsim, dt, x0, p)

# Grass data, Wageningen 1984 (Bouman et al. 1996)
tdata = np.array([87, 121, 155, 189, 217, 246, 273, 304])
mdata = np.array([0.05, 0.21, 0.54, 0.88, 0.99, 1.02, 1.04, 1.13])

# Define function to simulate model as a function of estimated array 'p0'.
def fcn_y(p0):
    # Reset initial conditions
    lg.x0 = x0.copy()
    # Reassign parameters from array to object
    lg.p['r'] = p0[0]
    lg.p['K'] = p0[1]
    # Simulate the model
    y = lg.run(tspan)
    # Retrieve result (model output of interest)
    m = y['m']
    return m

# Run calibration function
p0 = np.array([p['r'], p['K']]) # Initial guess
y_ls = least_squares(fcn_residuals, p0,
                     args=(fcn_y, lg.t, tdata, mdata),
                     kwargs={'plot_progress':True})

# Calibration accuracy
y_calib_acc = fcn_accuracy(y_ls)

# Simulate model with initial and estimated parameters
p_hat_arr = y_ls['x']
m_hat = fcn_y(p_hat_arr)

# Plot results
plt.figure(1)
plt.plot(lg.t, m_hat, label=r'$\hat{m}$')
plt.plot(tdata, mdata, label=r'$m_{data}$', linestyle='None', marker='o')
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$mass\ [kgDM\ m^{-2}]$')
plt.legend()

# References
# Bouman, B.A.M. et al. (1996).
#   Description of the growth model LINGRA as implemented inn CGMS.
#   European Commission. p.39, https://edepot.wur.nl/336784