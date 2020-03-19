# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 13:13:53 2019

@author: deanpospisil
"""

import numpy as np
n_pixels = 110
n_filters = 10
n_sims = 1000

filts = np.random.normal(loc=1, size=(n_sims, n_filters, n_pixels))

#filts = filts-np.mean(filts,-1, keepdims=True)
outline = np.random.normal(size=(n_sims, n_filters, n_pixels))

fill = np.ones(np.shape(filts))


def filt_resp(filts, inputs):
    
    inputs = inputs/(np.sum(inputs**2, (-1,-2), keepdims=True)**0.5)
    resp = np.sum((filts*inputs), -1)
    resp[resp<0] = 0
    
    tot_resp = np.sum(resp, -1);
    return tot_resp

out_resp = filt_resp(filts, outline)
fill_resp = filt_resp(filts, fill)

print([np.mean(fill_resp), np.std(fill_resp)/np.sqrt(n_sims)])
print([np.mean(out_resp), np.std(out_resp)/np.sqrt(n_sims)])

    
    


