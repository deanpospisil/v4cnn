#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 14:01:30 2017

@author: dean
"""

import matplotlib.pyplot as plt 
import numpy as np
import os, sys
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn')
sys.path.insert(0, top_dir + 'xarray/');
top_dir = top_dir + 'v4cnn/';
sys.path.append( top_dir + 'common')
sys.path.append(top_dir +'/nets')
import pickle as pk
import xarray as xr
import pandas as pd
cnn_name = 'bvlc_reference_caffenet_sinusoids'

da = xr.open_dataset('/loc6tb/' + 'data/responses/' + cnn_name + '.nc')['resp']