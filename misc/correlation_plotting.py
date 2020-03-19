# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 14:58:01 2017

@author: deanpospisil
"""

import matplotlib.pyplot as plt 
import numpy as np
import os, sys
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn/')
sys.path.append( top_dir + 'xarray');top_dir = top_dir + 'v4cnn/';
sys.path.append( top_dir + 'common/')
sys.path.append(top_dir +'/nets')
import pickle as pk
import xarray as xr;
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

cnn_names =['bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(64.0, 164.0, 51)',]
da = xr.open_dataset(top_dir + 'data/responses/' + cnn_names[0] + '.nc')['resp'].squeeze()
layer = da.coords['layer_label'].values==b'conv5'

cor = np.array([np.corrcoef(unit) for unit in da[:,:,layer].transpose('unit','x','shapes')])
    #%%
plt.imshow(np.nanmean(cor,0))