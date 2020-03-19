# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 12:27:44 2017

@author: deanpospisil
"""
import matplotlib.pyplot as plt 
import numpy as np
import os, sys
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir + 'v4cnn')
top_dir = top_dir + 'v4cnn/'

import pickle as pk
fns = [
'bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(64.0, 164.0, 51)_analysis.p',
'blvc_caffenet_iter_1APC362_pix_width[32.0]_pos_(64.0, 164.0, 51)_analysis.p',
'bvlc_caffenet_reference_shuffle_layer_APC362_pix_width[32.0]_pos_(64.0, 164.0, 51)_analysis.p',
'bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(64.0, 164.0, 51)_null_analysis.p'
]
fn = fns[0]
an=pk.load(open(top_dir + 'data/an_results/' + fn,'rb'), encoding='latin1')

