# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 09:38:24 2018

@author: RobinKraft
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.integrate

#set-up
start_grid = -5
end_grid = 5
grid_points = 50

grid_range = np.linspace(start_grid, end_grid, num = grid_points)

#container initialization
var_base = np.empty(shape = (len(grid_range), ))

bias_ensemble = np.empty(shape = (len(grid_range), ))
var_ensemble = np.empty(shape = (len(grid_range), ))
mse_ensemble = np.empty(shape = (len(grid_range), ))


def convolve_bias(c):
    bias = scipy.integrate.quad(lambda y: norm.cdf(c-y)*norm.pdf(y), -np.inf, np.inf)[0]
    return bias

def convolve_var(c):
    var = scipy.integrate.quad(lambda y: norm.cdf(c-y)**2*norm.pdf(y), -np.inf, np.inf)[0]
    return var

#unbagged moments
bias_base = np.zeros(shape = (len(grid_range), ))
var_base = [norm.cdf(c) * (1-norm.cdf(c)) for c in grid_range]
mse_base = var_base

#bagged moments
bias_ensemble = np.array([(convolve_bias(c) - norm.cdf(c))**2 for c in grid_range])
var_ensemble = np.array([convolve_var(c) - convolve_bias(c)**2 for c in grid_range])
mse_ensemble = bias_ensemble + var_ensemble

#plot of var, bias2, mse
fig = plt.figure(figsize = (15,10))

plt.subplot(1, 3, 1)
plt.plot(grid_range, bias_base)
plt.plot(grid_range, bias_ensemble)

plt.subplot(1, 3, 2)
plt.plot(grid_range, var_base)
plt.plot(grid_range, var_ensemble)

plt.subplot(1, 3, 3)
plt.plot(grid_range, mse_base)
plt.plot(grid_range, var_ensemble)

plt.show()
fig.savefig("bagging_moments.png")
