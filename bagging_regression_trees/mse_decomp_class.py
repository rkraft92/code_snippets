#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 14:27:16 2018

@author: RobinKraft
"""


import numpy as np

   
class mse_decomp:

    def __init__(self, 
                 f_test,
                 y_test,
                 y_predict_ensemble,  
                 y_predict_base_estimator):
        self.f_test = f_test
        self.y_test = y_test
        self.y_predict_ensemble = y_predict_ensemble
        self.y_predict_base_estimator = y_predict_base_estimator
    
    def decomp(self):
    #bagged estimator
        bias_squared_ensemble = (self.f_test - np.mean(self.y_predict_ensemble, axis = 1))**2
        var_ensemble = np.var(self.y_predict_ensemble, axis = 1)
        var_epsilon = np.var(self.y_test, axis = 1)

    #tree estimator
        bias_squared_base = (self.f_test - np.mean(self.y_predict_base_estimator, axis = 1))**2
        var_base = np.var(self.y_predict_base_estimator, axis = 1)
        
        return bias_squared_ensemble, bias_squared_base, var_ensemble, var_base, var_epsilon
    
#    print('MSE Bagging:{}'.format(np.mean(mse_temp_bagging)))                                                                             
#    print('Var Bagging:{}'.format(np.mean(var_bagging)))  
#    print('Bias^2 Bagging:{}'.format(np.mean(bias_squared_bagging)))
#
#    print('MSE Tree:{}'.format(np.mean(mse_temp_tree)))                                                                             
#    print('Var Tree:{}'.format(np.mean(var_tree)))  
#    print('Bias^2 Tree:{}'.format(np.mean(bias_squared_tree)))
#    print('Var_Epsilon Tree:{}'.format(np.mean(var_epsilon))) 
