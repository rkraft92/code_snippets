#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 14:58:24 2018

@author: RobinKraft
"""

import numpy as np
import sklearn.model_selection
import sklearn.ensemble
import simulation_class
import mse_decomp_class
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

n_reg = 10
n_obs = 500
n_sim = 50
n_tree = 50
sigma = 1



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


mse_temp_bagging = np.empty(shape = (n_obs, n_sim))
mse_temp_tree = np.empty(shape = (n_obs, n_sim))

y_predict_tree = np.empty(shape = (n_obs, n_sim))
y_predict_bagging = np.empty(shape = (n_obs, n_sim))


for i in range(0, n_sim):
    
    #bagged estimator
    bagging = sklearn.ensemble.BaggingRegressor(n_estimators = n_tree) 
    y_predict_bagging[:,i] = bagging.fit(X_train, y_train[:,i]).predict(X_test)
    mse_temp_bagging[:,i] = mean_squared_error(y_test[:,i], y_predict_bagging[:,i])
    
    #tree estimator
    y_predict_tree[:,i] = DecisionTreeRegressor().fit(X=X_train, y=y_train[:,i]).predict(X_test)
    mse_temp_tree[:,i] = mean_squared_error(y_test[:,i], y_predict_tree[:,i])
    
    

bias_squared_bagging, bias_squared_tree, var_bagging, var_tree, var_epsilon = mse_decomp_class.mse_decomp(f_test = f_test,
                                                                                                          y_test = y_test,
                                                                                                          y_predict_ensemble = y_predict_bagging,  
                                                                                                          y_predict_base_estimator = y_predict_tree).decomp()

print('MSE Bagging:{}'.format(np.mean(mse_temp_bagging)))                                                                             
print('Var Bagging:{}'.format(np.mean(var_bagging)))  
print('Bias^2 Bagging:{}'.format(np.mean(bias_squared_bagging)))

print('MSE Tree:{}'.format(np.mean(mse_temp_tree)))                                                                             
print('Var Tree:{}'.format(np.mean(var_tree)))  
print('Bias^2 Tree:{}'.format(np.mean(bias_squared_tree)))
print('Var_Epsilon Tree:{}'.format(np.mean(var_epsilon))) 


#print('Decomposition Sum:{}'.format(np.mean(var)+np.mean(bias_squared)+np.mean(var_epsilon)))
