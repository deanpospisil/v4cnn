# -*- coding: utf-8 -*-
"""
Created on Fri May 27 13:15:50 2016

@author: dean
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick

top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append( top_dir + 'xarray/')
top_dir = top_dir + 'v4cnn/'
sys.path.append(top_dir + 'common')
sys.path.append(top_dir + 'img_gen')


import xarray as xr
import pickle

v4_spar = xr.open_dataset(top_dir + 'data/an_results/spar_v4.nc')['spar']
alex_spar = xr.open_dataset(top_dir + 'data/an_results/spar_alex.nc').dropna('unit')['spar']


#in_v4_spar_alex = (alex_spar<v4_spar.max())*(alex_spar>v4_spar.min())
in_v4_spar_alex = (alex_spar<v4_spar.max())
in_units_spar = in_v4_spar_alex.coords['unit'].values[in_v4_spar_alex.values]

#with open(top_dir + 'data/models/ds_list_no_degen.p', 'rb') as f:
#    ds_list= pickle.load(f)
with open(top_dir + 'data/models/ds_list_with_degen.p', 'rb') as f:
    ds_list= pickle.load(f)
alex_apc = ds_list[0]['real'].dropna('unit')
v4_apc = ds_list[1]['real']


rthresh=0.5
above_v4_apc_alex = alex_apc>rthresh
in_units_apc = above_v4_apc_alex.coords['unit'].values[above_v4_apc_alex.values]

alex_ti = xr.open_dataset(top_dir + 'data/an_results/alex_TI_data.nc')['ti'].dropna('unit')
alex_ti_not_spar = alex_ti.sel(unit=in_units_spar)

alex_ti=alex_ti.reindex_like(alex_apc)
alex_apc=alex_apc.reindex_like(alex_ti)

alex_apc_not_spar = alex_apc.sel(unit=in_units_spar)

v4_ti = xr.open_dataset(top_dir + 'data/an_results/v4_TI_data.nc')['ti']
ti_thresh=np.median(v4_ti)

for_legend={'plt':[], 'label':[]}
[for_legend['plt'].append(plt.scatter([],[], s=a_size, color= color)) for color, a_size in zip(['blue','red'],[5,5])]
[for_legend['label'].append(str(a_cnt)) for a_cnt in ['greater than max. V4','less than max. V4']]

# Put a legend to the right of the current axis



plt.scatter(alex_ti.values, alex_apc.values, color ='blue', s=0.5 )
plt.scatter(alex_ti_not_spar.values, alex_apc_not_spar.values, color ='red', s=0.5, alpha=0.7 )
plt.legend(for_legend['plt'], for_legend['label'], frameon=True,
                 fontsize='medium', loc = 'upper left', handletextpad=1,
                 title='Sparsity AlexNet', scatterpoints = 1,fancybox=True,
                 framealpha=0.5)

plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel('Translation Invariance')
plt.ylabel('APC Correlation')



