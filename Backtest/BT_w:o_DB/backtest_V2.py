#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 12:02:43 2018

@author: RobinKraft
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
#import os
import csv


# =============================================================================
# variable initialization; specify csv names of data sheets
# =============================================================================
index_start = 100
index_name = 'High Dividend-Low Vola'
weights = 'weights.csv'
database = 'prices_data.csv'


# =============================================================================
# target weight information intialization
# =============================================================================
target_weights = pd.read_csv(weights, sep =';')

target_weights.set_index(keys = 'Unnamed: 0', inplace = True)

rebalancing_days = list(target_weights.columns)
start_date = rebalancing_days[0]


# =============================================================================
# dataframe price information initialization
# =============================================================================
price_data = pd.read_csv(database, sep =";")

price_data.rename(columns = {'Unnamed: 0': 'date'}, inplace = True)
price_data.set_index(keys = 'date', inplace = True)


# =============================================================================
# share and index level calculation
# =============================================================================
share_dict = {}
index_level = {}

index_level[start_date] = index_start

share_dict[start_date]= [target_weights[start_date].loc[ticker]*index_start/price_data[ticker].loc[start_date]
    for ticker in price_data.columns]


for index, date in enumerate(price_data.index[1:]):
    reb_day = list(price_data.index)
    
    if reb_day[index] in rebalancing_days[1:]:
        share_dict[date] = [target_weights[reb_day[index]].loc[ticker]*index_level[reb_day[index]]/price_data[ticker].loc[reb_day[index]]
        for ticker in price_data.columns]
    else:
        share_dict[date] = list(share_dict.values())[index]

    
    index_level[date] = sum(share_dict[date][k]*price_data[ticker].loc[date] 
    for k, ticker in enumerate(price_data.columns))
    
    
# =============================================================================
# #level plot
# =============================================================================
timeline = list(index_level.keys())
levels = list(index_level.values())

timeline_list = []
for time in timeline:
    datetimer = datetime.datetime.strptime(time, '%d.%m.%y')
    timeline_list.append(datetimer)

plt.plot(timeline_list, levels)
_=plt.xticks(rotation=45) 
plt.show()


# =============================================================================
# KPI Calculation
# =============================================================================
#index return p.a.
return_index = (float(levels[-1])/float(levels[0]))**(365/(timeline_list[-1]-timeline_list[0]).days)-1

#std: calulate daily returns
daily_return = pd.Series(levels)/pd.Series(levels).shift(1)-1
std_index = np.std(daily_return)*np.sqrt(250)

#sharpe ratio
sharpe_ratio = return_index/std_index

print("Return p.a.: {}".format(return_index))
print("Std p.a.: {}".format(std_index))
print("Sharpe Ratio: {}".format(sharpe_ratio))

# =============================================================================
# export levels, date to csv
# =============================================================================
with open('index_level_' + index_name + '.csv', "w") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['date', 'index_level'])
    
    for date in index_level:
        csv_writer.writerow([datetime.datetime.strptime(date, '%d.%m.%y'), index_level[date]])

# =============================================================================
# export shares to csv
# =============================================================================
with open("shares.csv", "w") as g:
    csv_writer2 = csv.writer(g)
    csv_writer2.writerow(['date', [ticker for ticker in price_data.columns]])
    
    for date in share_dict:
        csv_writer2.writerow([datetime.datetime.strptime(date, '%d.%m.%y'), share_dict[date]])


