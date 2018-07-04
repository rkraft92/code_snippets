#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 14:47:05 2018
@author: RobinKraft
"""

import numpy as np
import sklearn.model_selection
import sklearn.ensemble
import simulation_class
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error


n_reg = 10
n_obs = 500
n_sim = 50
n_tree = 50
sigma = 1

start_grid = 0.1
end_grid = 1
n_grid = 100

grid_range = np.linspace(start_grid, end_grid, num = n_grid)

#Creation of Simulation-Data
train_setup = simulation_class.simulation(n_reg = n_reg,
                                          n_obs = n_obs,
                                          n_sim = n_sim,
                                          sigma = sigma,
                                          random_seed_design = 0,
                                          random_seed_noise =  1
                                          )


test_setup = simulation_class.simulation(n_reg = n_reg,
                                         n_obs = n_obs,
                                         n_sim = n_sim,
                                         sigma = sigma,
                                         random_seed_design = 2,
                                         random_seed_noise = 3
                                         )

f_train = train_setup.friedman_model()
X_train, y_train = train_setup.error_term(f_train)

f_test = test_setup.friedman_model()
X_test, y_test = test_setup.error_term(f_test)

#Container Set-up
mse_temp_bagging = np.empty(shape = (n_obs, n_sim))
mse_temp_tree = np.empty(shape = (n_obs, n_sim))

y_predict_tree = np.empty(shape = (n_obs, n_sim))
y_predict_bagging = np.empty(shape = (n_obs, n_sim))

mse_decomp = np.empty(shape = (len(grid_range),3))

#Subagging-Simulation
for index, a in enumerate(grid_range):
    for i in range(0, n_sim):
    
    #bagged estimator
        subagging = sklearn.ensemble.BaggingRegressor(max_samples = math.ceil(a*n_obs), bootstrap = False, n_estimators = 50) 
        y_predict_bagging[:,i] = subagging.fit(X_train, y_train[:,i]).predict(X_test)
        mse_temp_bagging[:,i] = mean_squared_error(y_test[:,i], y_predict_bagging[:,i])
       
    mse_decomp[index, 0] = np.mean(mse_temp_bagging)
    mse_decomp[index, 1] = np.mean((f_test - np.mean(y_predict_bagging, axis = 1))**2)
    mse_decomp[index, 2] = np.mean(np.var(y_predict_bagging, axis = 1))

#Visualisation    
plt.plot(np.arange(0,1,1/n_grid), mse_decomp[:,0], label = 'MSE Subagging')
plt.plot(np.arange(0,1,1/n_grid), mse_decomp[:,1], label = 'Bias Subagging')
plt.plot(np.arange(0,1,1/n_grid), mse_decomp[:,2], label = 'Variance Subagging')
plt.ylabel('MSE Decomposition Subagging')
plt.xlabel('Subsample Fraction')
plt.legend()
plt.show()
