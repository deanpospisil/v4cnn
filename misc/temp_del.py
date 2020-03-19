# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 17:28:09 2016

@author: deanpospisil
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt 
import numpy as np
from itertools import product
import os, sys
import numpy as np
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn/')
sys.path.append( top_dir + 'xarray');top_dir = top_dir + 'v4cnn/';
sys.path.append( top_dir + 'common/')
sys.path.append(top_dir +'/nets')
import xarray as xr
name = 'bvlc_caffenet_reference_shuffle_layer_APC362_pix_width[64.0]_pos_(64.0, 164.0, 51).nc'
name = 'bvlc_caffenet_reference_shuffle_layer_APC362_pix_width[32.0]_pos_(64.0, 164.0, 51).nc'
cnn = xr.open_dataset(top_dir + 'data/responses/' + name)['resp']
cnn = cnn.sel(x=114).squeeze()

a = np.histogram(cnn.values)