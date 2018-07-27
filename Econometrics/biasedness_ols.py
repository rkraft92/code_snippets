#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 19:15:04 2018

@author: RobinKraft
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
 
n_it = 1000
n_obs = 100 #bias is small sample property
context_type = "OLS"
 
if context_type == "OLS":
    true_beta = [2,4]
 
    beta_sim = np.empty(shape =(n_it,))
 
    for i in range(n_it):
        x = np.random.normal(0,1,size =(n_obs,1))
        epsilon = np.random.normal(0,1,size =(n_obs,1))
       
        y = true_beta*x+epsilon
 
        beta = np.matmul(inv(np.matmul(x.T, x)), np.matmul(x.T,y))
        beta_sim[i] = beta[0][1]
 
    plt.hist(beta_sim)
    plt.axvline(x=np.mean(beta_sim), color='r', label = 'Mean of simulated Coefficients')
    plt.axvline(x=true_beta[1], color='y', label = 'True Coefficient')
    plt.legend()
    plt.show()
 
elif context_type == "Time Series":
    #Time Series
    true_alpha = 0.7
    
    y_time = np.empty(shape = (n_obs,))
    alpha_sim = np.empty(shape=(n_it,))
 
    y_time[0] = 0
 
    for i in range(n_it):
        for t in range(1, n_obs):
            y_time[t] = true_alpha*y_time[t-1]+ np.random.normal(0,1)
            alpha = np.dot(y_time[:n_obs-1].T, y_time[1:n_obs])/np.dot(y_time[:n_obs-1].T,y_time[:n_obs-1])
            alpha_sim[i] = alpha
 
    plt.hist(alpha_sim, bins = 20)
    plt.axvline(x=np.mean(alpha_sim), color = 'r', label = 'Mean of simulated Coefficients')
    plt.axvline(x=true_alpha, color = 'y', label = 'True Coefficient')
    plt.legend()
    plt.show()
 
else:
    print("Specify context_type either as OLS or Time Series")
