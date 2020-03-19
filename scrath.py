# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 18:30:29 2018

@author: deanpospisil
"""

import pickle as pk
import pandas as pd
the_dir = '/Users/deanpospisil/Desktop/modules/v4cnn/data/an_results/'
a = pd.read_pickle(the_dir + 'ti_vs_wt_cov_exps_.p')

#b = pd.read_pickle(the_dir + 'bvlc_reference_caffenetpix_width[32.0]_x_(64, 164, 51)_y_(114.0, 114.0, 1)_amp_NonePC370_analysis.p')