#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 11:31:21 2018

@author: RobinKraft
"""

import numpy as np

class simulation:
    
    def __init__(self, n_reg, n_obs, n_sim, sigma, random_seed_design, random_seed_noise):
        self.n_reg = n_reg
        self.n_obs = n_obs
        self.n_sim = n_sim
        self.sigma = sigma
        self.random_seed_design = random_seed_design
        self.random_seed_noise = random_seed_noise
        self.random_state_design =  np.random.RandomState(self.random_seed_design)
        self.random_state =  np.random.RandomState(self.random_seed_noise)
        
        self.X = self.random_state_design.uniform(0,1, size =(self.n_obs, self.n_reg))
        self.y_act = np.empty(shape=(self.n_obs, self.n_sim))        
  
    
    def friedman_model(self):
        self.f_friedman = 10 * np.sin(np.pi*self.X[:,0]*self.X[:,1])+20*np.power((self.X[:,2]-0.5),2)+10*self.X[:,3]+5*self.X[:,4]
        return self.f_friedman
    
    def linear_model(self):
        self.f_linear = sum((j+1) * self.X[:,j] for j in range(0,5))
        return self.f_linear
    
    def error_term(self, model):
        for m in range(0, self.n_sim):
            epsilon = self.random_state.normal(0, self.sigma, size=(self.n_obs, ))
    
            self.y_act[:,m] = model + epsilon
        
        return self.X, self.y_act

