#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 19:45:39 2018

@author: RobinKraft
"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
 
n_obs = 1000
n_it = 1000
 
mean = [1,0,0]
cov = [[1, 0.8, 0.3],[0.8, 1, 0],[0.3, 0, 1]]
 
true_beta = 3
const = 1
 
beta_est = np.empty(shape =(n_it,2))
 
for i in range(n_it):
    x = np.random.multivariate_normal(mean = mean, cov = cov, size = n_obs)
    x_design = x[:,0]
    eps = x[:,1]
    z = x[:,2]
   
    y = const + true_beta * x_design + eps
 
    x_design = sm.add_constant(x_design)
    results_ols = sm.OLS(y, x_design).fit()
    beta_est[i,0] = results_ols.params[1]
   
    first_stage = sm.OLS(x[:,0],sm.add_constant(z)).fit()
    result_first_stage = first_stage.predict()
    second_stage = sm.OLS(y, sm.add_constant(result_first_stage)).fit()
  
    beta_est[i,1] = second_stage.params[1]
   
plt.hist(beta_est[:,0], color = 'r', label = 'Beta OLS')
plt.hist(beta_est[:,1], color = 'b', label = 'Beta 2SLS (manually implemented)')
plt.legend()
plt.title('Histogram of Biased OLS and Unbiased IV Coefficient')
plt.show()