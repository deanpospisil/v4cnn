# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 15:12:08 2016

@author: dean
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 13:10:00 2016

@author: dean
"""

import numpy as  np
import scipy.io as  l
import os
import matplotlib.pyplot as plt
import sys
import matplotlib.ticker as mtick
#
import matplotlib as mpl
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir + 'xarray')
top_dir = top_dir+ 'v4cnn/'
sys.path.append(top_dir)
sys.path.append(top_dir + 'common')

import xarray as xr
import d_misc as dm
import pickle as pk
import pandas as pd

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
        if i==len(axes)-1:
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
        else:
            an_axes.set_xticks([])
            an_axes.set_yticks([])



#font = {'size' : 25}
#mpl.rc('font', **font)

cnn_names = [
'APC362_scale_1_pos_(-7, 7, 15)_ref_iter_0',
'APC362_scale_0.45_pos_(-7, 7, 15)_ref_iter_0',
'APC362_scale_0.45_pos_(-50, 48, 50)_ref_iter_0',
'APC362_scale_1_pos_(-50, 48, 50)_ref_iter_0',
]

v4_name = 'V4_362PC2001'
save_folder = top_dir + 'data/an_results/reference/'

#coef_var_v4 = pk.load(open(save_folder + 'coef_var' + v4_name, 'rb'))
#coef_var_alex = pk.load(open(save_folder +'coef_var' + cnn_name, 'rb'))
#eye_r2_v4 = pk.load(open(save_folder  + 'eye_r2_' + v4_name, 'rb'))
#eye_r2_alex = pk.load(open(save_folder  + 'eye_r2_' + cnn_name, 'rb'))
#k_alex = pk.load(open(save_folder  + 'k_' + cnn_name, 'rb'))
#k_v4 = pk.load(open(save_folder  + 'k_' + v4_name, 'rb'))
#ti_v4 = pk.load(open(save_folder  + 'ti_'+ v4_name, 'rb'))
#ti_alex = pk.load(open(save_folder  + 'ti_'+ cnn_name, 'rb'))
#tilc_alex = pk.load(open(save_folder  + 'trans_ill_cond_' + cnn_name, 'rb'))

alex_resp = xr.open_dataset(top_dir + 'data/responses/' + cnn_name + '.nc')['resp'].load()


for cnn_name in cnn_names:
    plt.close('all')
    coef_var_v4 = pk.load(open(save_folder + 'coef_var' + v4_name, 'rb'), encoding='latin1')
    coef_var_alex = pk.load(open(save_folder +'coef_var' + cnn_name, 'rb'), encoding='latin1')
    eye_r2_v4 = pk.load(open(save_folder  + 'eye_r2_' + v4_name, 'rb'), encoding='latin1')
    eye_r2_alex = pk.load(open(save_folder  + 'eye_r2_' + cnn_name, 'rb'), encoding='latin1')
    k_alex = pk.load(open(save_folder  + 'k_' + cnn_name, 'rb'), encoding='latin1')

    k_v4 = pk.load(open(save_folder  + 'k_' + v4_name, 'rb'), encoding='latin1')
    ti_v4 = pk.load(open(save_folder  + 'ti_'+ v4_name, 'rb'), encoding='latin1')
    ti_alex = pk.load(open(save_folder  + 'ti_'+ cnn_name, 'rb'), encoding='latin1')
    tilc_alex = pk.load(open(save_folder  + 'trans_ill_cond_' + cnn_name, 'rb'), encoding='latin1')

    apc_alex = pk.load(open(save_folder  + 'apc_'+ cnn_name, 'rb'), encoding='latin1')
    apc_v4 = pk.load(open(save_folder  + 'apc_'+ v4_name, 'rb'), encoding='latin1')

    ti_v4['layer_label'] = 'v4'
    ti_v4['layer_unit'] = range(len(ti_v4))
    k_v4['layer_unit'] = range(len(k_v4))

    k_v4['layer_label'] = 'v4'
    ti_v4.set_index(['layer_label', 'layer_unit'], inplace=True)

    apc_v4 = pd.concat( [apc_v4, coef_var_v4, eye_r2_v4, k_v4], axis=1).set_index(['layer_label', 'layer_unit'])
    cnn = pd.concat([coef_var_alex, eye_r2_alex, k_alex, ti_alex, tilc_alex, apc_alex], axis=1)

    both=pd.concat([cnn, apc_v4, ti_v4])
    both = both[(both.index.get_level_values('layer_label') != 'prob')]


    #both[('alt','cor')]['apc_v4'] =

    #import re
    #[]
    #lay_type_index = [re.findall('[^0-9]+', label)[0]
    #                 for label in cnn.index.get_level_values('layer_label')]
    #rect_or_not = ['relu' in label
    #                 for label in cnn.index.get_level_values('layer_label')]
    #cnn['lay_type'] = lay_type_index
    #cnn.set_index('lay_type', append=True, inplace=True)

    plt.close('all')
    def stacked_hist_layers(cnn, logx=False, logy=False, xlim=None, maxlim=False, bins=100):
        layers = cnn.index.get_level_values('layer_label').unique()
        if logx:
            cnn = np.log(cnn.dropna())
        if maxlim:
            xlim = [np.min(cnn.dropna().values), np.max(cnn.dropna().values)]
        for i, layer in enumerate(layers):
            plt.subplot(len(layers), 1, i+1)
            vals = cnn.loc[layer].dropna().values.flatten()


            plt.hist(vals, log=logy, bins=bins, histtype='step',
                     range=xlim, normed=False)

            plt.plot([np.median(vals),]*2, np.array(plt.gca().get_ylim()), color='red')
            plt.gca().set_ylabel(layer, ha='right', rotation=0, labelpad=25)
            plt.gca().yaxis.set_label_position("right")

        if logx:
            plt.xlabel('log')
        nice_axes(plt.gcf().axes)
        #plt.suptitle(cnn.name)



    #sparsity plot
    k = both['k']
    stacked_hist_layers(k.dropna(), logx=True, logy=False, xlim=[0, 10], maxlim=True)
    plt.suptitle('Sparsity ' + cnn_name)
    plt.savefig(top_dir + 'analysis/figures/images/k_by_layer '+ cnn_name + '.eps')
    plt.savefig(top_dir + 'analysis/figures/images/k_by_layer '+ cnn_name + '.png')


    #translation invariance plot
    plt.figure()
    ti = both['ti']
    #stacked_hist_layers(ti[cnn.k>40], logx=False, logy=False)
    both['k']['v4'] = 0
    stacked_hist_layers(ti[both.k<40], logx=False, logy=False, xlim=[0,1])
    plt.suptitle('Translation Invariance ' + cnn_name)
    plt.savefig(top_dir + 'analysis/figures/images/translation_invariance_by_layer '+ cnn_name + '.png')
    plt.savefig(top_dir + 'analysis/figures/images/translation_invariance_by_layer '+ cnn_name + '.eps')


    #apc correlation plot.
    plt.figure()
    apc = both[('alt','cor')]
    stacked_hist_layers(apc, logx=False, logy=False, xlim=[0,1],bins=100)
    plt.suptitle('APC correlation ' + cnn_name)
    plt.savefig(top_dir + 'analysis/figures/images/apc_correlation_by_layer '+ cnn_name + '.png')
    plt.savefig(top_dir + 'analysis/figures/images/apc_correlation_by_layer '+ cnn_name + '.eps')



    #v4ness: distance from [1,1], perfect V4
    plt.figure()
    v4ness =  ((1-apc)**2 + (1-ti)**2)**0.5
    v4ness = v4ness[both.k<40]
    v4ness = v4ness[(v4ness.index.get_level_values('layer_label') != 'v4') ]
    stacked_hist_layers(v4ness, logx=False, logy=False, xlim=[0,1], bins=100)
    plt.suptitle('distance from [1,1], perfect V4 ' + cnn_name)
    plt.savefig(top_dir + 'analysis/figures/images/v4ness_by_layer '+ cnn_name + '.png')
    plt.savefig(top_dir + 'analysis/figures/images/v4ness_by_layer '+ cnn_name + '.eps')


    #apc = both[('null','cor')]
    #stacked_hist_layers(apc, logx=False, logy=False, xlim=[0,1],bins=100)

    #plt.legend(layers)
    #cnn['relu'] = rect_or_not
    #cnn.reset_index(inplace=True)
    #cnn.set_index('relu', append=True, inplace=True)
    #cnn.reorder_levels(['lay_type', 'layer_label', 'relu', 'layer_unit'])

    #k_out_of_range = -(cnn['k'].dropna()<v4['k'].max())
    #n_k_nans = (-cnn['k'].isnull()).groupby(level='layer_label').sum()

    #k_all_caught_frac = k_out_of_range.groupby(level='layer_label').sum() / n_k_nans
    #
    #print(k_all_caught_frac)
    #need_caught_oneshape = (eye_r2_alex>0.5).groupby(level='layer_label').sum()
    #k_need_caught_oneshape = (k_out_of_range & (cnn['eye_r2']>0.5)).groupby(level='layer_label').sum()
    #print(need_caught_oneshape == need_caught_oneshape)
    #
    #k_need_caught_trans = (k_out_of_range & (cnn['x_frac_var'] > 0.5)).groupby(level='layer_label').sum()
    #need_caught_trans = (cnn['x_frac_var'] > 0.5).groupby(level='layer_label').sum()
    #print(need_caught_trans == need_caught_trans)

    #does k get rid of good units
    #cnn.k.plot(kind='hist', bins =100, normed=True, alpha=0.5, range=[0,400])
    #v4.k.plot(kind='hist', bins =10, normed=True, alpha=0.5, range=[0,30])
    #import plotly
    #from glue import qglue
    #app = qglue(x=cnn)

