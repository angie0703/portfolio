"""
FTE34806 - Modelling of Biobased Production Systems
Farm Technology Group, WUR
@author: Stefan Maranus, Daniel Reyes Lastiri
"""

import numpy as np


def fcn_rk4(diff,t_span,y0,h=1.0,t_eval=None,interp_kind='linear'):
    """ Function for Runge-Kutta 4th order numerical integration.
        Based on the syntax of scipy.integrate.solve_ivp

        This function numerically integrates a system of ordinary differential
        equations, given an initial value::

            dy/dt = f(t, y)
            y(t0) = y0

        Here, t is a one-dimensional independent variable (time), y(t) is an
        n-dimensional vector-valued function (state), and an n-dimensional
        vector-valued function f(t, y) determines the differential equations.
        The goal is to find y(t) approximately satisfying the differential
        equations, given an initial value y(t0)=y0.

        Parameters
        ----------
        diff : callable
            Function to calculate the value of d_dt
            (the right-hand side of the system).
        t_span : 2-tuple of floats
            Interval of integration (t0, tf). The solver starts with t=t0 and
            integrates until it reaches t=tf.
        h : float
            Step size for the integration.

        Returns
        -------
        Dictionary with the following:\n
        t : ndarray, shape (n_points,)
            Time vector for desired evaluation (based on `t_span` and `h`,
            or equal to `t_eval`).
        y : ndarray, shape (n_states, n_points)
            Values of the solution at `t`.
        """
    # Number of elements for integration time and model outputs
    nt = int((t_span[1] - t_span[0]) / h) + 1
    # Vectors for integration time and model outputs
    tint = np.linspace(t_span[0], t_span[1], nt)
    yint = np.zeros((y0.size, tint.size))
    # Assign initial condition to first element of output vector
    yint[:, 0] = y0
    # Iterator
    # (stop at second-to-last element, and store index in Fortran order)
    it = np.nditer(tint[:-1], flags=['f_index'])
    for ti in it:
        # Index for current time instant
        idx = it.index
        # Model outputs at next time instant (RK4)
        k1 = diff(ti, y0)
        #TODO: define k2, k3, and k4.
        k2 = diff(ti+(h/2),y0+(h/2*k1))
        k3 = diff(ti+(h/2),y0+(h/2*k2))
        k4 = diff(ti+h,y0+(h*k3))
        yint[:, idx + 1] = y0 + (k1+2*k2+2*k3+k4)/6 * h  #TODO: complete this line
        # Update initial condition for next iteration
        y0 = yint[:, idx + 1]
    return {'t':tint, 'y':yint} #TODO: define the function output

def diff_sample(_t, _x0_place):
    dy_dt = 2*_t - 4
    return np.array([dy_dt])
