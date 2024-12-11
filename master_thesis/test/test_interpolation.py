# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 09:13:48 2024

@author: alegn
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline

tsim = np.linspace(0, 120, 120*5+1)
tsim1 = np.linspace(0, 120, 120+1)


#interpolate data
x = [0, 20, 40, 50, 70, 90, 97, 100,120]
y = [0 ,10, 20, 30, 40, 50, 60, 70, 120]

y_linear_interp = np.interp(tsim, x, y)
y_linear_interp1 = np.interp(tsim1, x, y)
cubic_interp = CubicSpline(x, y)  # Note: Only use if you need smoother results
y_cubic_interp = cubic_interp(tsim)
y_cubic_interp1 = cubic_interp(tsim1)

# Plotting results
plt.figure(figsize=(12, 6))
plt.plot(x, y, 'o', label='Original Data')
plt.plot(tsim, y_linear_interp, '-', label='Linear Interpolation 1/5')
plt.plot(tsim1, y_linear_interp1, '-', label='Linear Interpolation')
plt.plot(tsim, y_cubic_interp, '--', label='Cubic Spline Interpolation 1/5')
plt.plot(tsim1, y_cubic_interp1, '--', label='Cubic Spline Interpolation')

plt.legend()
plt.title('Data Interpolation')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()