# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   Daniel Reyes Lastiri, Stefan Maranus,
            Rachel van Ooteghem, Tim Hoogstad

Tutorial for the use of the least_squares method
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
import numpy.linalg as LA

plt.style.use('ggplot')

# Simulation time array
t = np.linspace(0.0, 365.0, 365+1) # [d]

# Initial condition
m0 = 0.001            # [kgDM m-2] initial mass

# Grass data, Wageningen 1984 (Bouman et al. 1996)
tdata = np.array([87, 121, 155, 189, 217, 246, 273, 304])
mdata = np.array([0.05, 0.21, 0.54, 0.88, 0.99, 1.02, 1.04, 1.13])

# Define function to simulate the model output of interest
# as a function of the estimated parameter array 'p'
def fcn_y(p):
    # Model parameters (improved iteratively)
    r, K = p[0], p[1] # [d-1], [kgDM m-2] model parameters
    # Model output (analytical solution of logistic growth model)
    m = K/(1+((K-m0)/m0)*np.exp(-r*t))  # [kgDM m-2]
    return m

# Define a function to calculate the residuals: e(k|p) = z(k)-y(k|p)
# Notice that m_k must be interpolated for measurement instants tdata
def fcn_residuals(p):
    m = fcn_y(p)
    f_interp = interp1d(t, m)
    m_k = f_interp(tdata)
    err = mdata-m_k
    return err
    
# Model calibration: least_squares
p0 = np.array([0.01, 1.0])                  # Initial parameter guess
y_lsq = least_squares(fcn_residuals, p0)     # Minimize sum [ e(k|p) ]^2

# Retrieve calibration results
p_hat = y_lsq['x']

# Simulate model with initial guess (p0) and estimated parameters (p_hat)
m_hat0 = fcn_y(p0)
m_hat = fcn_y(p_hat)

# Plot results
plt.figure(1)
plt.plot(t, m_hat0, label=r'$\hat{m0}$', linestyle='--')
plt.plot(t, m_hat, label=r'$\hat{m}$')
plt.plot(tdata, mdata, label=r'$m_{data}$', linestyle='None', marker='o')
plt.xlabel(r'$time\ [d]$')
plt.ylabel(r'$mass\ [kgDM\ m^{-2}]$')
plt.legend()

# Calibration accuracy
# Sensitivity matrix (Jacobian, J) (returned as jac)
J = y_lsq['jac']
# Residuals (returned as fun)
err = y_lsq['fun']
# Calculated variance of residuals
N , p = J.shape[0], J.shape[1]
varres = 1/(N-p) * np.dot(err.T,err)
# Covariance matrix of parameter estimates
covp = varres * LA.inv(np.dot(J.T, J))
# Standard deviations of parameter estimates
sdp = np.sqrt(np.diag(covp))
# Correlation coefficients of parameter estimates
ccp = np.empty_like(covp)
for i,sdi in enumerate(sdp):
    for j,sdj in enumerate(sdp):
        ccp[i,j] = covp[i,j]/(sdi*sdj)
# Mean squared error
mse = 2*y_lsq['cost']/J.shape[0]
rmse = np.sqrt(mse)

# References
# Bouman, B.A.M. et al. (1996).
#   Description of the growth model LINGRA as implemented inn CGMS.
#   European Commission. p.39, https://edepot.wur.nl/336784