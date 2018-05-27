#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: RobinKraft
"""


import os
import glob
import pandas as pd
import numpy as np
 
os.chdir('your working path')
 
POB_list = []
route_list = []
 
for index, file in enumerate(glob.glob("*.xlsx")):
 
    #read in data
    f = pd.read_excel(file, sheet_name = 'raw_data')
    #select subset
    f = f[f.Month != False]
 
    #create new column
    file_flist = file.split('_')[2]
    f['date'] = file_flist.split('.')[0]
         
    #create pivot tables
    POB_pivot = pd.pivot_table(f,
                          index = ['POB_COUNTRY'],
                          columns = ['date'],
                          values = ['GRP_PAX CY'],
                          aggfunc = [np.sum],
                          )
 
    route_pivot = pd.pivot_table(f,
                          index = ['ROUTING_VV'],
                          columns = ['date'],
                          values = ['GRP_PAX CY'],
                          aggfunc = [np.sum],
                          )
    POB_list.append(POB_pivot)
    route_list.append(route_pivot)
   
    #export to csv of pivot_tables  
    POB_pivot.to_csv(r"your target path\POB" + str(index) + ".csv")
    route_pivot.to_csv(r"your target path\route" + str(index) + ".csv")        

#create concatenated pivot    
POB_pivot_fin = pd.concat(POB_list, axis = 1)
route_pivot_fin = pd.concat(route_list, axis = 1)

#export pivot to csv 
POB_pivot_fin.to_csv(r"target path\POB_final.csv")
route_pivot_fin.to_csv(r"target path\route_final.csv")