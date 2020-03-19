# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 21:16:42 2016

@author: deanpospisil
"""
import os, sys
import numpy as np
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn/')
sys.path.append( top_dir + 'xarray')
top_dir = top_dir + 'v4cnn/'
sys.path.append( top_dir + 'common/')
import matplotlib
import pickle
import xarray as xr
import apc_model_fit as ac
import pandas as pd
import matplotlib.ticker as mtick
try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except:
    print('no plot')
import pickle as pk
an=pk.load(open(top_dir + 'data/an_results/' + 
        'bvlc_reference_caffenetAPC362_pix_width[30.0]_pos_(64.0, 164.0, 101)_analysis.p','rb'),
        encoding='latin1')
fracvar_x = an[0].sel(concat_dim='r2')
rf = an[0].sel(concat_dim='rf')
meas = an[1]


