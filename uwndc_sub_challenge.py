# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:00:43 2019

@author: deanpospisil
"""

#submission analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
sub_dir = '/Users/deanpospisil/Desktop/modules/v4_comp/submissions/'
subs=os.listdir('/Users/deanpospisil/Desktop/modules/v4_comp/submissions/')

true = pd.read_csv('/Users/deanpospisil/Downloads/derived(1).csv')
true = true.iloc[:,1:]

for a_sub_nm in subs:
    sub = pd.read_csv(sub_dir+a_sub_nm)
    sub = sub.iloc[:,1:]
    
    rs = []
    for t, m in zip(true.values.T, sub.values.T):
        rs.append(np.corrcoef(t[:], m[:])[0,1]**2)
        
    plt.plot(rs)
    
plt.legend(subs)
