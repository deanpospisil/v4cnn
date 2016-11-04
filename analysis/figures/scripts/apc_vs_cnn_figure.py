# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 12:08:05 2016

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
import xarray as xr;import pandas as pd
import apc_model_fit as ac
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import d_plot as dp
import scipy.io as  l
import d_curve as dc
import d_img_process as imp
import d_net_analysis as dn
from sklearn.neighbors import KernelDensity
import caffe_net_response as cf
def beautify(ax=None, spines_to_remove = ['top', 'right']):
    almost_black = '#262626'
    more_grey = '#929292'
#    text_font = 'serif'
#    number_font = 'helvetica'
    all_spines = ['bottom','left','right','top']
    # Get the axes.
    if ax is None:
        #fig = plt.figure(1)
        ax = plt.axes()
    if not type(ax)==type([]):
        ax = [ax,]
    for a_ax in ax:
        # Remove 'spines' (axis lines)
        for spine in spines_to_remove:
            a_ax.spines[spine].set_visible(False)
    
        # Make ticks only where there are spines
        if 'left' in spines_to_remove:
            a_ax.tick_params(left=0)
        if 'right'  in spines_to_remove:
            a_ax.tick_params(right=0)
        if 'bottom'  in spines_to_remove:
            a_ax.tick_params(bottom=0)
        if 'top'  in spines_to_remove:
            a_ax.tick_params(top=0)
    
        # Now make them go 'out' rather than 'in'
        for axis in ['x', 'y']:
            a_ax.tick_params(axis=axis, which='both', direction='out', pad=7)
            a_ax.tick_params(axis=axis, which='major', color=almost_black, length=6)
            a_ax.tick_params(axis=axis, which='minor', color=more_grey, length=4)
    
        # Make thinner and off-black
        spines_to_keep = list(set(all_spines) - set(spines_to_remove))
        
        for spine in spines_to_keep:
            a_ax.spines[spine].set_linewidth(0.5)
            a_ax.spines[spine].set_color(almost_black)

    
        # Change the labels & title to the off-black and change their font
        for label in [a_ax.yaxis.label, a_ax.xaxis.label, a_ax.title]:
            label.set_color(almost_black)
    
        # Change the tick labels' color and font and padding
        for axis in [a_ax.yaxis, a_ax.xaxis]:
            # padding
            axis.labelpad = 20
            # major ticks
            for major_tick in axis.get_major_ticks():
                label = major_tick.label
                label.set_color(almost_black)
            # minor ticks
            for minor_tick in axis.get_minor_ticks():
                label = minor_tick.label
                label.set_color(more_grey)

    #plt.grid(axis='y', color=more_grey)
def scatter_lsq(ax, a, b, lsq=True, mean_subtract=True, **kw):    

    if len(a.shape)<=1:
        a = np.expand_dims(a,1)
    if len(b.shape)<=1:
        b = np.expand_dims(b,1)
    
    if mean_subtract:
        a -= np.mean(a);b -= np.mean(b)
    if a.shape[1] > 1 :
        print('a second dim to big, just taking the first col')
        a = a[:,1]
    if b.shape[1] > 1 :
        print('b second dim to big, just taking the first col')
        b = b[:,1]   
    if lsq:
        x = np.linalg.lstsq(a, b)[0]
        a_scaled = np.dot(a, x)
    else:
        a_scaled = a
    ax.scatter(a_scaled, b, **kw)
    return a_scaled, b
    
    
def boot_strap_se(a, bstraps=1000):
    stats = []
    for ind in range(bstraps):
        resample = np.random.randint(0, high=np.shape(a)[0], size=np.shape(a)[::-1])
        stats.append([np.mean(a[col, i]) for i, col in enumerate(resample)])
    return np.percentile(np.array(stats), [1,99], axis=0)
def cor2(a,b):
    if len(a.shape)<=1:
        a = np.expand_dims(a,1)
    if len(b.shape)<=1:
        b = np.expand_dims(b,1)
    a -= a.mean(0);b-=b.mean(0)
    a /= np.linalg.norm(a, axis=0);b /= np.linalg.norm(b, axis=0);
    corrcoef = np.dot(a.T, b)       
    return corrcoef
    

 #%%   

v4_name = 'V4_362PC2001'
v4_resp_apc = xr.open_dataset(top_dir + 'data/responses/' + v4_name + '.nc')['resp'].load()
v4_resp_apc = v4_resp_apc.transpose('shapes', 'unit')
file = open(top_dir + 'data/responses/v4_apc_109_neural_labels.txt', 'r')
wyeth_labels = [label.split(' ')[-1] for label in 
            file.read().split('\n') if len(label)>0]
v4_resp_apc['w_lab'] = ('unit', wyeth_labels)
fn = top_dir + 'data/models/' + 'apc_models_362.nc'

if 'apc_fit_v4' not in locals():
    dmod = xr.open_dataset(fn, chunks={'models':50, 'shapes':370})['resp'].load()
    apc_fit_v4 = ac.cor_resp_to_model(v4_resp_apc.chunk({'shapes': 370}), 
                                      dmod.chunk({}), 
                                      fit_over_dims=None, 
                                      prov_commit=False)**2

cnn_names =['bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(64.0, 164.0, 51)',
            'blvc_caffenet_iter_1APC362_pix_width[32.0]_pos_(64.0, 164.0, 51)']
colors = ['r','g','b','m','c', 'k', '0.5']
from sklearn.model_selection import ShuffleSplit
X = np.arange(362)
cv_scores = []
model_ind_lists = []
models = []
for cnn_name in cnn_names:
    da = xr.open_dataset(top_dir + 'data/responses/' + cnn_name + '.nc')['resp']
    da = da.sel(unit=slice(0, None, 1)).squeeze()
    middle = np.round(len(da.coords['x'])/2.).astype(int)
    da_0 = da.sel(x=da.coords['x'][middle])
    da_0 = da_0.sel(shapes=v4_resp_apc.coords['shapes'].values)
    models.append(da_0)
models.append(dmod)

n_splits = 5
for model in models:
    ss = ShuffleSplit(n_splits=n_splits, test_size=1/n_splits,
        random_state=0)
    cv_score = []
    model_ind_list = []
    for train_index, test_index in ss.split(X):
        frac_var_v4_cnn = cor2(model.values[train_index], 
                               v4_resp_apc.values[train_index])**2
        frac_var_v4_cnn[np.isnan(frac_var_v4_cnn)] = 0
        model_sel = frac_var_v4_cnn.argmax(0)
        frac_var_v4_cnn_cv = np.array([cor2(v4_resp_apc[test_index, i], 
                            model[test_index, model_ind])**2
                            for i, model_ind in enumerate(model_sel)]).squeeze()
        model_ind_list.append(model_sel)
        frac_var_v4_cnn_cv[np.isnan(frac_var_v4_cnn_cv)] = 0
        cv_score.append(frac_var_v4_cnn_cv)
    cv_scores.append(cv_score)
    model_ind_lists.append(model_ind_list)
model_ind_lists = np.array(model_ind_lists)
cv_scores = np.array(cv_scores)
#%%
mean_scores = cv_scores.mean(1)
bsci_scores= np.array([boot_strap_se(cv_score) for cv_score in cv_scores])
bsci_scores = bsci_scores - np.expand_dims(mean_scores,1)



#%%
ax_list=[]
plt.figure(figsize=(4,4))
ax = plt.subplot(221)
ax_list.append(ax)
ax.locator_params(nbins=5)
ax.set_title('APC vs CN on V4 $R^2$')
x = mean_scores[0]
y = mean_scores[2]
xsd = bsci_scores[0]
ysd = bsci_scores[2]
ax.errorbar(x, y, yerr=np.abs(ysd), xerr=np.abs(xsd), fmt='o', 
            alpha=0.5, markersize=0, color='r', ecolor='0.5')
colors= np.array(['k',]*len(x))
colors[((np.abs(x-y)-np.max(np.abs(ysd),0))>0) & 
       ((np.abs(x-y)-np.max(np.abs(xsd),0))>0)] = 'r'
ax.scatter(x,y, color=colors, s=3)
#ax.scatter(x, y, alpha=0.5, s=2)
ax.plot([0,1],[0,1], color='0.5')
#ax.set_xlabel('Trained Net')
ax.set_ylabel('APC')
ax.set_ylim(0,.7)
ax.set_xlim(0,.7)
plt.grid()

ax = plt.subplot(223, sharex=ax)
ax_list.append(ax)

ax.locator_params(nbins=5)
x = mean_scores[0]
y = mean_scores[1]
xsd = bsci_scores[0]
ysd = bsci_scores[1]

#ax.scatter(x, y, alpha=0.5, s=2)
ax.errorbar(x, y, yerr=np.abs(ysd), xerr=np.abs(xsd), fmt='o', 
            alpha=0.5, markersize=0, color='r', ecolor='0.5')
colors= np.array(['k',]*len(x))
colors[((np.abs(x-y)-np.max(np.abs(ysd),0))>0) & 
       ((np.abs(x-y)-np.max(np.abs(xsd),0))>0)] = 'r'
ax.scatter(x,y, color=colors, s=3)
ax.plot([0,1],[0,1], color='0.5')
ax.set_ylim(0,.7)
ax.set_xlim(0,.7)

ax.set_xlabel('Trained Net')
ax.set_ylabel('Untrained Net')
plt.grid()
labels = ['A.', 'B.']
for ax, label in zip(ax_list, labels):
    ax.text(-0.1, 1., label, transform=ax.transAxes,
      fontsize=14, fontweight='bold', va='top', ha='right')


plt.savefig(top_dir + '/analysis/figures/images/apc_vs_cnn.pdf')
k = {'s':1, 'color':'r'}
ax = plt.subplot(222)
x = mean_scores[0]
y = mean_scores[2]
cnn_better_unit = (x-y).argmax()
cnn_better_model = model_ind_lists[0][:,cnn_better_unit][0]
cnb_resp = models[0].sel(unit=cnn_better_model).values
cnu_resp = v4_resp_apc[:,cnn_better_unit].values
scatter_lsq(ax, cnb_resp, cnu_resp,**k)
np.corrcoef(models[0].sel(unit=cnn_better_model).values, v4_resp_apc[:,cnn_better_unit].values)
ax.plot([0,1],[0,1], color='0.5')
ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.5, 1])



ax = plt.subplot(224)
apc_better_unit = (y-x).argmax()
apc_better_model = model_ind_lists[2][:,apc_better_unit][0]
apb_resp = models[2].sel(models=apc_better_model).values
apu_resp = v4_resp_apc[:,apc_better_unit].values
scatter_lsq(ax, apb_resp, apu_resp,**k)
np.corrcoef(models[2].sel(models=apc_better_model).values, v4_resp_apc[:,apc_better_unit].values)
ax.plot([0,1],[0,1], color='0.5')
beautify(ax)
plt.tight_layout()

'''  
to_compare=cv_scores.mean(1) 
ax.scatter(to_compare[0],to_compare[1])
#ax.legend(['Trained Net', 'Untrained Net', 'Shuffled'])
ax.axis('equal')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.plot([0,1],[0,1])
ax.set_xlabel('CNN Trained')
ax.set_ylabel('CNN Untrained', rotation=0)
ax.set_title('CNN vs APC fits to V4 $R^2$')
plt.grid()
'''
#kw = {'s':3, 'linewidths':0, 'c':'k'}
#x,y= scatter_lsq(ax, frac_var_v4_cnn, apc_fit_v4.values**2, lsq=False,
#                     mean_subtract=False, **kw)
#
#beautify(ax, spines_to_remove=['top','right'])
#
#data_spines(ax, x, y, mark_zero=[False, False], sigfig=2, fontsize=fs-2, 
#                nat_range=[[0,1],[0,1]], minor_ticks=False, 
#                data_spine=['bottom', 'left'], supp_xticks=[0.25, 1,], 
#                supp_yticks = [0.25, 1,])
#cartesian_axes(ax, x_line=False, y_line=False, unity=True)



