#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 12:39:16 2018

@author: dean
"""
import pickle as pk
import matplotlib.pyplot as plt
f = '/loc6tb/data/an_results/bvlc_reference_caffenetpix_width[ 2.89082083]_x_(64, 164, 52)_y_(114.0, 114.0, 1)PC370_analysis.p'
a = pk.load(open(f,'r'))
df = a[1]
#%%
layers_to_examine = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
for layer in layers_to_examine:
    apc_small = df.loc[layer]['apc']**0.5
    plt.hist(apc_small, normed=True, cumulative=True, histtype='step', bins=100)

plt.legend(layers_to_examine)
plt.savefig('/home/dean/Desktop/apc_fits_11_pix_diam_shapes.pdf')

    'bvlc_reference_caffenetpix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370_analysis.p',
