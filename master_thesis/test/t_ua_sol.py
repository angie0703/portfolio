# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   Daniel Reyes Lastiri, Stefan Maranus,
            Rachel van Ooteghem, Tim Hoogstad

Tutorial for the uncertainty analysis
"""
import numpy as np
import matplotlib.pyplot as plt

from mbps.functions.uncertainty import fcn_plot_uncertainty

plt.style.use('ggplot')

# Random number generator. A seed is specified to allow for reproducibility.
rng = np.random.default_rng(seed=12)

# Simulation time array
#tsim = np.linspace(0, 50, 50+1) # [d] for simple growth model
tsim = np.linspace(0, 100, 100+1) # [d] for logistic growth model

# Monte Carlo simulations
n_sim = 1000 # number of simulations
# Initialize array of outputs, shape (len(tsim), len(n_sim))
m_arr = np.full((tsim.size,n_sim), np.nan)
# Run simulations
for j in range(n_sim):
    # Simple growth model
    #r = 0.05
    #r = rng.normal(0.05, 0.005)
    #K = 2.0
    #K = rng.normal(2.0, 0.2)
    #m = K*(1-np.exp(-r*tsim))
    
    # Logisitc growth model
    m0 = 0.033 # [kgDM m-2]
    r = rng.normal(0.086,0.001) # [d-1]
    K = rng.normal(1.404, 0.070) # [kgDM m-2]
    m = m0*K/(m0+(K-m0)*np.exp(-r*tsim))
    
    m_arr[:,j] = m
    
# Plot results
plt.figure(1)
plt.plot(tsim, m_arr[:,0:12])
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

plt.figure(2)
ax2 = plt.gca()
ax2 = fcn_plot_uncertainty(ax2, tsim, m_arr, ci=[0.50,0.68,0.95])
#ax2.plot(tsim, np.full(tsim.shape,1.391), color='k')
#ax2.plot(tsim, np.full(tsim.shape,1.696), color='k')
plt.xlabel(r'$time\ [d]$')
plt.ylabel('biomass ' + r'$[kgDM\ m^{-2}]$')

#finding mean, mean(68%+), and mean(68%-)
array_mean = m_arr[100,:]
mean_= np.mean(array_mean)
std_ = np.std(array_mean)
mean_68_plus = mean_ + std_
mean_68_min = mean_ - std_
print("mean = ", mean_)
print("std =", std_)
print("mean(68%+) = ", mean_68_plus)
print("mean(68%-) = ", mean_68_min)

