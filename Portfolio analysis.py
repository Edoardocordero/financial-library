# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 15:26:44 2019

@author: Edoardo
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas_datareader import data as web

tlist=['AMS', 'MSFT', 'AAPL', 'IDXX', 'F', 'PM', 'GE', 'WAT', 'BDX', 'SYK']
weights=np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])
data=pd.DataFrame()
for t in tlist:
    data[t]=web.DataReader(t, 'yahoo', '2015-1-1')['Adj Close']
sec_ret=np.log(data/data.shift(1))
port_ret=np.dot(sec_ret, weights)
port_ret[0]=0
print(port_ret)
port_avgret_a=port_ret.mean()*250
port_std_a=np.dot(weights.T, np.dot(sec_ret.cov()*250, weights))**0.5
data_norm=data/data.iloc[0] *100
data_norm.plot( figsize=(16, 5));
plt.show()

port_norm=np.dot(data_norm, weights)
port_norm=pd.DataFrame(port_norm)
port_norm.plot(figsize=(16,5));
plt.show()
print('Annual average returm: ', port_avgret_a)
print('Annual dev std: ', port_std_a)   
port_var_a= port_std_a**2
somm=0
for i in range(10):
    somm+=weights[i]**2 * sec_ret.iloc[:, i].var()*250
dr= port_var_a-somm
print('% of dr: ', round(dr/port_var_a *100,3), '%')