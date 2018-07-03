# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 12:13:35 2018

@author: RobinKraft
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#set-up
start_grid = 0
end_grid = 5
grid_points = 50

grid_range = np.linspace(start_grid, end_grid, num = grid_points)

gamma = 1/2
subsampling_fractions = [1/5, 1/2, 2/3]

bias_ensemble = np.empty(shape = (len(grid_range), len(subsampling_fractions)))
var_ensemble = np.empty(shape = (len(grid_range), len(subsampling_fractions)))
mse_ensemble = np.empty(shape = (len(grid_range), len(subsampling_fractions)))

for index, a in enumerate(subsampling_fractions):
    bias_ensemble[:,index] = [(norm.cdf(c*a**gamma) - norm.cdf(c))**2 for c in grid_range]
    var_ensemble[:,index] = [a*norm.cdf(c*a**gamma)*(1 - norm.cdf(c*a**gamma)) for c in grid_range]

mse_ensemble = bias_ensemble + var_ensemble

#plot of var, bias2, mse
fig = plt.figure(figsize = (15,10))

plt.subplot(1, 3, 1)
plt.plot(grid_range, bias_ensemble[:,0])
plt.plot(grid_range, bias_ensemble[:,1])
plt.plot(grid_range, bias_ensemble[:,2])

plt.subplot(1, 3, 2)
plt.plot(grid_range, var_ensemble[:,0])
plt.plot(grid_range, var_ensemble[:,1])
plt.plot(grid_range, var_ensemble[:,2])

plt.subplot(1, 3, 3)
plt.plot(grid_range, var_ensemble[:,0])
plt.plot(grid_range, var_ensemble[:,1])
plt.plot(grid_range, var_ensemble[:,2])

plt.show()
fig.savefig('subagging_stumps.png')
