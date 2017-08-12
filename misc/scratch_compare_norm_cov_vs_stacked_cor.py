#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 17:10:17 2017

@author: dean
"""

import numpy as np

x = np.random.randn(100,12)
y = np.random.randn(100,1)
broad = np.ones((1,12))
x = y*broad
#if nxm the get cov mXm
def norm_cov(x):
    x = x.astype(np.float64)
    x = x - np.mean(x, 0, keepdims=True)
    
    diag_inds = np.triu_indices(x.shape[1], k=1)
    numerator = np.sum(np.dot(x.T, x)[diag_inds])
    
    vnrm = np.linalg.norm(x, axis=0, keepdims=True)
    denominator = np.sum(np.multiply(vnrm.T, vnrm)[diag_inds]) 
    norm_cov = numerator/denominator
    return norm_cov

def stacked_cov(x):
    x = x.astype(np.float64)
    x = x - np.mean(x, 0, keepdims=True)
    
    diag_inds = np.triu_indices(x.shape[1], k=1)
    numerator = np.sum(np.dot(x.T, x)[diag_inds])
    
    
    denominator = np.sum(x**2.)
    norm_cov = numerator/denominator
    return norm_cov

print(norm_cov(x))
print(stacked_cov(x))