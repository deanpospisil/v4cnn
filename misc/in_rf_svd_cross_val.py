# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 15:49:44 2016

@author: dean
"""
import os, sys
import numpy as np
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn/')
sys.path.append( top_dir + 'xarray')
top_dir = top_dir + 'v4cnn/'
sys.path.append( top_dir + 'common/')

import xarray as xr
import apc_model_fit as ac
import pandas as pd
try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except:
    print('no plot')
def kurtosis(da):
    #take xarray and return coefficient of variation
    #expects shapes X unit
    da = da.transpose('shapes','x', 'unit')
    mu = da.mean('shapes')
   # k = da.reduce(kurtosis,dim='shapes')
    sig = da.reduce(np.var, dim='shapes')
    k = (((da - mu)**4).sum('shapes')/da.shapes.shape[0])/(sig**2)
    return k
drop = ['conv4_conv4_0_split_0', 'conv4_conv4_0_split_1']


if 'alex_resp' not in locals():
    cnn_name = 'APC362_scale_1_pos_(-99, 96, 66)bvlc_reference_caffenet'
    alex_resp = xr.open_dataset(top_dir + 'data/responses/' + cnn_name + '.nc')['resp'].load().squeeze()
    alex_resp.coords['layer_label'] = alex_resp.coords['layer_label'].astype('str')
    for drop_name in drop:    
        alex_resp = alex_resp[:,:,(alex_resp.coords['layer_label'] != drop_name)]
    base_line = alex_resp.sel(shapes=0)[0]
    alex_resp = alex_resp.drop(0, dim='shapes')

    alex_bls = alex_resp - base_line#subtract off baseline
    alex_var = ((alex_bls)**2).sum('shapes')
    had_resp = alex_var > 0
    #widest width 24
    w = 24.
    step_width = np.diff(alex_var.coords['x'].values)[1]
    #add this to the right alt, and subtract it from the left alt
    step_in = w/2.
    min_steps = int(np.ceil(w / step_width))
    
    
    in_rf = np.zeros(had_resp.T.values.shape)
    n_steps = len(alex_var.coords['x'].values)
    rf_pos_all = []
    rf_pos = []
    beg_pos = None
    for n_unit, unit in enumerate(had_resp.T.values):
        
        if sum(unit)<n_steps:
            for i, x in enumerate(unit):
                if x and type(beg_pos)==type(None):
                    beg_pos = i
                elif (not x) and (type(beg_pos)!=type(None)):
                    end_pos = i-1
                    if (end_pos-beg_pos)>(min_steps*2):
                        rf_pos = rf_pos + list(range(beg_pos+min_steps, end_pos-min_steps))
                    beg_pos = None
            if x and (type(beg_pos)!=type(None)):
                end_pos = i
                if (end_pos-beg_pos)>(min_steps*2):
                    rf_pos = rf_pos + list(range(beg_pos+min_steps, n_steps-min_steps))
            
        else:
            rf_pos = list(range(min_steps, n_steps-min_steps))
            
        
        in_rf[n_unit, rf_pos] = 1
        rf_pos_all.append(rf_pos)
        beg_pos = None
        rf_pos = []
    

from sklearn.cross_validation import LeaveOneOut
units_todo = 50000
alex_resp= alex_resp.transpose('unit', 'x', 'shapes')
ti_est = []
ti_est_all = []
counter =0
frac_var_explained = []
alex_resp = alex_resp - alex_resp.mean('shapes')
resp = alex_resp.values

#resp = np.random.randn(*np.shape(resp))
for unit_resp, unit_in_rf in zip(resp, in_rf):
    if unit_in_rf.sum()>1:
        counter = counter + 1
        print(counter)
        unit_resp = unit_resp[unit_in_rf.astype(bool), :]
        loo = LeaveOneOut(unit_resp.shape[0])
        for train, test in loo:
            u, s, v = np.linalg.svd(unit_resp[train])
            ti_est = ti_est + [sum((np.dot(v[0], unit_resp[test].T))**2),]          
            tot_var = (unit_resp**2).sum()
        ti_est = np.sum(np.array(ti_est)/tot_var)
        ti_est_all.append(ti_est)
    else:
        ti_est_all.append(np.nan)
    ti_est = []
    if counter>units_todo:
        break

        
type_change = np.where(np.diff(alex_var.coords['layer'].values))[0]
type_label = alex_var.coords['layer_label'].values[type_change].astype(str)
x_pos = list(range(0, n_steps, 6))
x_label = alex_var.coords['x'].values[0:-1:6]
try:
    plt.xticks(type_change, type_label, rotation='vertical', size = 'small')   
    plt.plot(np.array(ti_est_all))
    plt.title('leave one position out cross validation R^2')
except:
    print('no plot')


import pandas as pd

fn = top_dir + 'data/an_results/reference/apc_' + cnn_name
pd.DataFrame(ti_est_all).to_pickle(fn)

'''
#something wrong here but fuck it.
#beg= 0
#fin = 1000
#alex_k = kurtosis(alex_resp)
#alex_k_mask = alex_k.where(alex_k != np.inf) 
#
#plt.xticks(type_change, type_label, rotation='vertical', size = 'small')
#plt.yticks(x_pos, x_label, size = 'small')
#plt.ylabel('shape center pos')
#plt.imshow((alex_k_mask[:,beg:fin].fillna(450)<40).plot(), interpolation ='nearest', aspect='auto',cmap = cm.viridis)
#


plt.figure()
plt.subplot(211)
plt.title('Normalized response power. Widest shape: ' + str(w) + ' shape step: ' + str(step_width))
norm_resp = (alex_var[:, beg:fin]/alex_var[:, beg:fin].max('x')).fillna(0)
norm_resp = norm_resp - norm_resp.min('x')
plt.xticks(type_change, type_label, rotation='vertical', size = 'small')
plt.yticks(x_pos, x_label, size = 'small')
plt.ylabel('shape center pos')
plt.imshow(norm_resp, interpolation ='nearest', aspect='auto',cmap = cm.viridis)

plt.subplot(212)
plt.title('Normalized response power. Widest shape: ' + str(w) + ' shape step: ' + str(step_width))
norm_resp = (alex_var[:, beg+4096:fin+4096]/alex_var[:, beg+4096:fin+4096].max('x')).fillna(0)
#plt.xticks(type_change, type_label, rotation='vertical', size = 'xx-small')
plt.yticks(x_pos, x_label, size = 'xx-small')
plt.ylabel('shape center pos')

plt.imshow(norm_resp, interpolation ='nearest', aspect='auto',cmap = cm.viridis)

       
     

plt.subplot(311)
plt.title('Normalized response power. Widest shape: ' + str(w) + ' shape step: ' + str(step_width))
norm_resp = (alex_var[:, beg:fin]/alex_var[:, beg:fin].max('x')).fillna(0)
plt.xticks(type_change, type_label, rotation='vertical', size = 'xx-small')
plt.yticks(x_pos, x_label, size = 'xx-small')
plt.ylabel('shape center pos')

plt.imshow(norm_resp, interpolation ='nearest', aspect='auto',cmap = cm.viridis)

plt.subplot(312)
plt.title('Had response to shapes')
plt.xticks(type_change, type_label, rotation='vertical', size = 'xx-small')
plt.yticks(x_pos, x_label, size = 'xx-small')
plt.ylabel('shape center pos')
plt.imshow(had_resp[:, beg:fin], interpolation ='nearest',aspect='auto')


plt.subplot(313)
plt.title('Within Receptive Field')
plt.xlabel('unit layer')
plt.xticks(type_change, type_label, rotation='vertical', size = 'xx-small')
plt.yticks(x_pos, x_label, size = 'xx-small')
plt.ylabel('shape center pos')
plt.imshow(in_rf.T[:, beg:fin], interpolation ='nearest', aspect='auto')



plt.tight_layout()
'''