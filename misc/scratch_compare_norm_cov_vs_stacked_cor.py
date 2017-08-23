#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 17:10:17 2017

@author: dean
"""

import numpy as np

x = np.random.randn(1000,10)
y = np.random.randn(1000,1)
broad = np.ones((1,12))
x = y + x*0.5
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
    
    coords = np.meshgrid(range(x.shape[1]), range(x.shape[1]))
    diag_inds = np.triu_indices(x.shape[1], k=1)
    c = coords[0][diag_inds]
    r = coords[1][diag_inds]
    c_stack = np.concatenate([x[:, c_ind] for c_ind in c])
    r_stack = np.concatenate([x[:, r_ind] for r_ind in r]) 
    c_norm = np.sum(c_stack**2)**0.5 
    r_norm = np.sum(r_stack**2)**0.5 
    denominator = c_norm*r_norm
    norm_cov = numerator/denominator
    return norm_cov

print(norm_cov(x))
print(stacked_cov(x))

x = np.array(range(12)).reshape((3,4))

coords = np.meshgrid(range(x.shape[1]), range(x.shape[1]))
diag_inds = np.triu_indices(x.shape[1], k=1)
c = coords[0][diag_inds]
r = coords[1][diag_inds]
c_stack = np.concatenate([x[:, c_ind] for c_ind in c])
r_stack = np.concatenate([x[:, r_ind] for r_ind in r]) 
c_norm = np.sum(c_stack**2)**0.5 
r_norm = np.sum(r_stack**2)**0.5 

    