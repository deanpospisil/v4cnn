# -*- coding: utf-8 -*-
"""
Created on Mon May  2 13:51:48 2016

@author: dean

#making plots over stages of training. need to get performance data.
"""
import os, sys
import numpy as np
import warnings
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append( top_dir + '/xarray')
top_dir = top_dir + 'v4cnn/'
sys.path.append(top_dir)
sys.path.append(top_dir +'common')

import d_misc as dm
import xarray as xr
import apc_model_fit as ac
all_iter = dm.list_files(top_dir + 'data/an_results/r_apc_models_u*.nc')

def tick_format_d(x, pos):
    if x==0:
        return('0')
    else:
        if x>=1:
            return(str(x).split('.')[0])
        else:
            return(np.round(x,2))

def nice_axes(axes, xticks=None, yticks=None, nxticks=5, nyticks=2):
    for i, an_axes in enumerate(axes):

        if yticks==None:
            an_axes.yaxis.set_major_locator(mtick.LinearLocator(numticks=nyticks, presets=None))
        else:
            an_axes.set_yticks(yticks)
        if xticks==None:
            an_axes.xaxis.set_major_locator(mtick.LinearLocator(numticks=nxticks, presets=None))
        else:
            an_axes.set_xticks(xticks)
        an_axes.xaxis.set_tick_params(length=0)
        an_axes.yaxis.set_tick_params(length=0)
        an_axes.yaxis.set_major_formatter(mtick.FuncFormatter(tick_format_d))
        an_axes.xaxis.set_major_formatter(mtick.FuncFormatter(tick_format_d))


#sort fits by iteration.
it_num = []
for fn in all_iter:
    it_num.append( int(fn.split('e_')[1].split('.')[0] ))
all_iter = [all_iter[ind] for ind in  np.argsort(it_num)]
r_thresh=0.5
orignet = xr.open_dataset('/Users/dean/Desktop/modules/v4cnn/data/an_results/alex_net_nat_image_dist.nc')
layers = orignet.coords['layer'].values
lay_count = [ sum(layers==layer) for layer in np.arange(max((layers)))]
lay_count.append(1000)

iteration=-1
da = xr.open_dataset(all_iter[iteration])
#get layer names
_, index = np.unique(da.coords['layer_label'], return_index=True)
layer_labels = list(da.coords['layer_label'][np.sort(index)].data)

da_thresh = (da>r_thresh).groupby('layer_label').sum()['r']
inds =[np.where(da_thresh.coords['layer_label'].values==layer_label)[0][0]
        for layer_label in layer_labels]

frac_v4 = [da_thresh.values[inds[i]]/np.double(lay_count[i]) for i, _ in enumerate(layer_labels)]
count_v4 = [da_thresh.values[inds[i]] for i, _ in enumerate(layer_labels)]
fs=18

frac_v4 = frac_v4[:-1]
count_v4 = count_v4[:-1]
plt.close('all')
fig = plt.figure(figsize=(16, 5))
plt.bar(np.arange(len(frac_v4))-0.5, frac_v4)
plt.xticks(range(len(frac_v4)))
plt.gca().set_xticklabels(layer_labels,fontsize=fs)
plt.gca().set_yticklabels([0,0.1,0.2,0.3,0.4,0.5],fontsize=fs)


annot=[ str(ctv4)+'/'+str(ct) for ctv4, ct in zip(count_v4, lay_count)]
for ind in np.arange(0,len(count_v4)):
    if count_v4[ind]>0:
        plt.annotate(annot[ind], xy=(ind-0.5, frac_v4[ind]+0.01), fontsize=fs-5)
plt.ylabel('Fraction Units', fontsize=fs)
plt.xlabel('Layer', fontsize=fs)
plt.xlim([-1,13])
plt.ylim([0,0.5])
plt.gca().xaxis.set_tick_params(width=0)
plt.gca().yaxis.set_tick_params(width=0)
plt.title('Net Final Iteration')
plt.tight_layout()
#plt.title('Fraction Units > 0.5 Correlation Across Translations')
#plt.savefig(top_dir + 'analysis/figures/images/fits_to_apc_over_trans_fin.eps')
font = {'size' : 22}
mpl.rc('font', **font)
#plt.ylim([0,1])
plt.figure(figsize=(16,8))
n_v4 = []
for fn in all_iter:
     d = (xr.open_dataset(fn)['r']>0.5)
     d = d.loc[d.coords['layer']>7]
     n_v4.append(d.groupby('layer_label').sum().values)

count=np.array([256.,4096.,4096.,1000., 256.])
plt.plot(np.sort(it_num), np.array(n_v4)[:,:-1]/count, lw=4)
plt.legend(list(map(bytes.decode ,d.groupby('layer_label').sum().coords['layer_label'].values[:-1])), loc='left')
plt.xlabel('Training Iterations (~80 imgs each)' )
plt.ylabel('Fraction TI V4 like units' )
plt.xlim(-10000,1400000)
plt.ylim(0,1)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.tight_layout()
plt.savefig(top_dir + 'analysis/figures/images/fits_to_apc_over_trans_train_frac.eps')