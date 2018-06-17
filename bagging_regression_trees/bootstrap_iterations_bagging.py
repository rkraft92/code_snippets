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
from sklearn.metrics import mean_squared_error


n_obs = 500
n_sim = 50
n_tree = 50

X_train, y_train, f_train = simulation_class.simulation(n_reg = 10,
                                   n_obs = n_obs,
                                   n_sim = n_sim,
                                   sigma = 1,
                                   random_seed_design = 0,
                                   random_seed_noise =  1
                                   ).friedman_model()

X_test, y_test, f_test = simulation_class.simulation(n_reg = 10,
                                   n_obs = n_obs,
                                   n_sim = n_sim,
                                   sigma = 1,
                                   random_seed_design = 2,
                                   random_seed_noise = 3
                                   ).friedman_model()

mse_temp_bagging = np.empty(shape = (n_obs, n_sim))
mse_temp_tree = np.empty(shape = (n_obs, n_sim))

y_predict_tree = np.empty(shape = (n_obs, n_sim))
y_predict_bagging = np.empty(shape = (n_obs, n_sim))

mse_decomp = np.empty(shape = (n_tree,3))

for boot in range(1, n_tree):
    for i in range(0, n_sim):
    
    #bagged estimator
        bagging = sklearn.ensemble.BaggingRegressor(n_estimators = boot) 
        y_predict_bagging[:,i] = bagging.fit(X_train, y_train[:,i]).predict(X_test)
        mse_temp_bagging[:,i] = mean_squared_error(y_test[:,i], y_predict_bagging[:,i])
       
    mse_decomp[boot, 0] = np.mean(mse_temp_bagging)
    mse_decomp[boot, 1] = np.mean((f_test - np.mean(y_predict_bagging, axis = 1))**2)
    mse_decomp[boot, 2] = np.mean(np.var(y_predict_bagging, axis = 1))
    
plt.plot(mse_decomp[1:,:])
plt.ylabel('MSE Decomposition Bagging')
plt.xlabel('Bootstrap Iterations')
plt.show()