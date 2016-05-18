# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 18:03:36 2016

@author: deanpospisil
"""

import os, sys
top_dir = os.getcwd().split('net_code')[0] 
sys.path.append(top_dir + 'net_code/common/')
sys.path.append( top_dir + 'xarray/')


import xarray as xr
import os
import seaborn as sns
import matplotlib.pyplot as plt

top_dir = os.getcwd().split('net_code')[0] + 'net_code/'

imtype = '.pdf'

fname = 'apc_model_fit_over_trans.nc'
fitm = xr.open_dataset(top_dir +'data/an_results/' + fname )
b = fitm.to_dataframe()
plt.close('all')

#sns.boxplot(x="layer_label", y="r", data=b[['r', 'layer_label']])
b_t=b[b['r']>0.5]

sns.jointplot(kind='kde', x="cur_mean", y="cur_sd", data=b_t[['cur_mean', 'cur_sd']])

g=sns.jointplot(kind='kde', x="or_mean", y="or_sd", data=b_t[['or_mean', 'or_sd']])
g.plot_joint(plt.scatter, alpha=0.01)

g = sns.PairGrid(b_t[['cur_mean','cur_sd', 'or_mean', 'or_sd','r']])
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=30)



fillb = b[['r','layer_label']]

per = b[['r','layer_label']][b['r']>0.5].groupby('layer_label', sort=False).count()/ fillb.groupby('layer_label', sort=False).count()
per.plot(kind = 'bar')
plt.ylim((0,1))
plt.title('Percent units > 0.5 Correlation')
fn =top_dir + 'analysis/figures/images/' +fname + '_p>5' + imtype 

plt.savefig(filename=fn)
open(fn, 'a').write("")

