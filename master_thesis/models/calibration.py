# -*- coding: utf-8 -*-
"""
FTE34806 - Modelling of Biobased Production Systems
MSc Biosystems Engineering, WUR

@authors:   Daniel Reyes Lastiri, Stefan Maranus,
            Rachel van Ooteghem, Tim Hoogstad
"""
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy.linalg as LA

def fcn_residuals(p0,fcn,t,tdata,ydata,plot_progress=False):
    '''Function to calculate the residuals y-ydata.
    
    Parameters
    ----------
    p0 : 1-D array of floats
        Value of model parameters being estimated
    fcn : function method
        Function being called to simulate the submodel or system
    t : 1-D array
        Time sequence of values used for the simulation
    tdata : 1-D array
        Array of measurement times.
    ydata : 1-D array
        Array of measurement values of a model output.
    plot_prgress : boolean
        Default is False. If True, display a plot with model vs. data for each
        least_squares iteration.
    
    Returns
    -------
    residuals : array
        Value for abs(y-ydata) for each time instant.
    '''
    # Run function and ensure column vector (t,ny) from fcn
    yhat = fcn(p0)
    # Define interpolation function based on full simulation-time array
    f_interp = interp1d(t, yhat, axis=0)
    # Interpolate y at corresponding tdata instants
    yhat_k = f_interp(tdata)
    # Calculate residuals
    residuals = ydata-yhat_k
    
    # Show animation of the calibration progress
    if plot_progress:
        plt.figure(99)
        ax = plt.subplot2grid((1,1), (0, 0))
        ax.plot(tdata,ydata,linestyle='None',marker='o')
        ax.plot(t,yhat) # for speed, plot only interpolated model output
        plt.pause(0.05)
    return residuals

def fcn_accuracy(y_lsq):
    ''' Function to calculate the accuracy metrics from results of a
    least squares parameter estimation
    
    Parameters
    ----------
    y_lsq : dictionary
        Output of scipy.optimize.least_squares
    
    Returns
    -------
    results : dictionary
        * cov: covariance matrix of the parameter estimates
        * sd: standard deviations of the parameter estimates
        * cc: correlation coefficients of the parameter estimates
        * rmse: root mean square error (based on residuals at end of estimation)
    '''
    print('')
    print('-'*40)
    print('Accuracy of estimates')
    print('-'*40)
    np.set_printoptions(formatter={'float_kind':'{:.3E}'.format})
    print('Parameter estimates \n {} \n'.format(y_lsq['x']))    
    # Sensitivity matrix (Jacobian, J) (returned as jac)
    J = y_lsq['jac']
    # Residuals (returned as fun)
    err = y_lsq['fun']
    # Calculated variance of residuals
    N , p = J.shape[0], J.shape[1]
    varres = 1/(N-p) * np.dot(err.T,err)
    # Covariance matrix of parameter estimates
    covp = varres * LA.inv(np.dot(J.T, J))
    print('Covariance matrix of parameters \n {} \n'.format(covp))    
    # Standard deviations of parameter estimates
    sdp = np.sqrt(np.diag(covp))
    print('Standard deviations of parameters \n {} \n'.format(sdp))
    # Correlation coefficients of parameter estimates
    ccp = np.empty_like(covp)
    for i,sdi in enumerate(sdp):
        for j,sdj in enumerate(sdp):
            ccp[i,j] = covp[i,j]/(sdi*sdj)
    print('Correlation coefficients of parameters \n {} \n'.format(ccp))
    # Mean squared error
    mse = 2*y_lsq['cost']/J.shape[0]
    rmse = np.sqrt(mse)
    print('Root mean square error \nRMSE = {:.4f}'.format(rmse))
    print('-'*40)
    return {'cov':covp, 'sd':sdp, 'cc':ccp, 'rmse':rmse}
