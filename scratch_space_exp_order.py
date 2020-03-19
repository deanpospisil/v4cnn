# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 17:23:02 2018

@author: deanpospisil
"""


import numpy as np
stim = list(range(9))
trials = 5
rf_x = 25
rf_y = 50
rf_d = 10
npos = 3
stim_d = rf_d/np.float(npos)

stim_order_1 = stim*trials*npos
stim_order_2 = -snp.ones(len(stim_order_1))

