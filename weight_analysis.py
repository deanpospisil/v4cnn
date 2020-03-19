# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 10:33:09 2018

@author: deanpospisil
"""


import os, sys
import matplotlib.pyplot as plt
import pickle
from scipy.stats import linregress
import numpy as np
import pandas as pd
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+'xarray/')
#%%
import xarray as xr
top_dir = top_dir + '/v4cnn'

#%%
if sys.platform == 'linux2': 
    data_dir = '/loc6tb/dean/'
else:
    data_dir = top_dir

layer_labels_b = [b'conv2', b'conv3', b'conv4', b'conv5', b'fc6']
layer_labels = ['conv2', 'conv3', 'conv4', 'conv5', 'fc6']
    

        
with open(top_dir + '/nets/netwtsd.p', 'rb') as f:    
    try:
        netwtsd = pickle.load(f, encoding='latin1')
    except:
        netwtsd = pickle.load(f)