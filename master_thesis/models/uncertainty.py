# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR
@authors:   Daniel Reyes Lastiri, Stefan Maranus,
            Rachel van Ooteghem, Tim Hoogstad

Function to plot the confidence intervals from tge uncertainty propagation
evaluated with Monte Carlo simulations
"""
import numpy as np
import pandas as pd

from matplotlib import colors, cm
from matplotlib.ticker import AutoMinorLocator

def fcn_plot_uncertainty(ax,tsim,ysim, ci=[0.50,0.68,0.95]):
    '''Function to plot the (weighted) residuals w*abs(y-ydata).
    
    Parameters
    ----------
    ax : Plot axis
        Empty Matplotlib axis object 
    tsim : 1-D array
        Time sequence of values used for the simulation
    ysim : 2-D array
        2-D array of shape (len(t),len(y)), i.e. rows for time and
        columns for model outputs.
    ci : 1-D array-like
        Default is [0.50,0.68,0.95]. Confidence intervals to plot.
    
    Returns
    -------
    ax : Plot axis
        Matplotlib object with confidence intervals
    '''
    # Obtain quantiles
    # ysim given as axis 0 for t, and axis 1 for simulations
    df = pd.DataFrame(ysim)
    q_lo, q_hi = [], []
    quantiles = []
    for i,cii in enumerate(ci):
        q_lo.append(0.50-cii/2)
        q_hi.append(0.50+cii/2)
    quantiles.extend(q_lo)
    quantiles.extend(q_hi)
    quantiles.sort()
    qsim = np.zeros((tsim.size,len(quantiles)))
    # Color map
    for i,qi in enumerate(quantiles):
        qsim[:,i] = df.quantile(qi,axis=1)
    cNorm  = colors.Normalize(vmin=-len(ci), vmax=len(ci))
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cm.get_cmap('cubehelix'))
    color_list=scalarMap.to_rgba(range(len(ci)))
    # Plot the median (plotted twice to allow for pattern when indexing legend)
    ax.plot(tsim, np.mean(ysim, axis=1), label='mean', linestyle='--')
    ax.plot(tsim, np.mean(ysim, axis=1), label='mean', linestyle='--')
    # Plot fill areas
    for i,qi in enumerate(ci):
        ax.fill_between(tsim,qsim[:,i], qsim[:,i+1],
                        facecolor=color_list[i], label=str(ci[-1-i]))
        ax.fill_between(tsim,qsim[:,-1-i], qsim[:,-2-i],
                        facecolor=color_list[i], label=str(ci[-1-i]))
    # Minor tick location, dividing major in 2
    ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))    
    # Set the grid with transparency alpha
    ax.yaxis.grid(True, which='major')
    ax.xaxis.grid(True, which='major')

    # Legend
    hndls, lbls = ax.get_legend_handles_labels()
    ax.legend(handles=hndls[-1:0:-2],
              labels=lbls[-1:0:-2],
              bbox_to_anchor=(0,1.0),
              loc='upper left',ncol=1)
    return ax
       