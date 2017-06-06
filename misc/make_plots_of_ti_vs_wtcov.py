# -*- coding: utf-8 -*-
"""
Created on Tue May 30 20:36:19 2017

@author: deanpospisil
"""

import numpy as np
import xarray as xr
import os, sys
import matplotlib.pyplot as plt
import pickle
layer_labels_b = [b'conv2', b'conv3', b'conv4', b'conv5', b'fc6']
layer_labels = ['conv2', 'conv3', 'conv4', 'conv5', 'fc6']


top_dir = os.getcwd().split('v4cnn')[0]
top_dir = top_dir + 'v4cnn'

with open(top_dir + '/data/an_results/ti_vs_wt_cov_exps_all_lays.p', 'rb') as f:    
    try:
        an = pickle.load(f, encoding='latin1')
    except:
        an = pickle.load(f)
        
with open(top_dir + '/nets/netwtsd.p', 'rb') as f:    
    try:
        netwtsd = pickle.load(f, encoding='latin1')
    except:
        netwtsd = pickle.load(f)
#net num descriptor
net_num_desc = [
        'Orig.',
        'Conv2. Adj. Low',
        'Conv2. Adj. Med',
        'Conv2. Adj. High',
        'Conv5. Adj. Low',
        'Conv5. Adj. High',
        'fc6 Adj. 0.1',
        'fc6. Adj. 0.9',
        'fc6. Adj. 0.2',
        'fc6. Adj. 0.3',
        'fc6. Adj. 0.4',
        'fc6. Adj. 0.5',
        'fc6. Adj. 0.6',
        'fc6. Adj. 0.7',
        'fc6. Adj. 0.8',
        ]


#%%
wt_layer_name_dict = {'relu1':'conv1', 'norm1':'conv1', 'pool1':'conv1', 'conv1':'conv1',
                      'relu2':'conv2', 'norm2':'conv2', 'pool2':'conv2', 'conv2':'conv2',
                      'relu3':'conv3','conv3':'conv3',
                      'relu4':'conv4', 'conv4':'conv4',
                      'relu5':'conv5', 'conv5':'conv5', 'pool5':'conv5',
                      'fc6':'fc6', 'relu6':'fc6'}
def ti_wt_cov_scatters(net_nums, layer_names, an, alpha=1, figsize=None):
#net_nums = range(len(an[1][:]))
    i = 0 
    wt_layer_name_dict = {'relu1':'conv1', 'norm1':'conv1', 'pool1':'conv1', 'conv1':'conv1',
                      'relu2':'conv2', 'norm2':'conv2', 'pool2':'conv2', 'conv2':'conv2',
                      'relu3':'conv3','conv3':'conv3',
                      'relu4':'conv4', 'conv4':'conv4',
                      'relu5':'conv5', 'conv5':'conv5', 'pool5':'conv5',
                      'fc6':'fc6', 'relu6':'fc6'} 
    if figsize is None:
        fig = plt.figure(figsize=(len(layer_names)*2, len(net_nums)*2))
    else:
        fig = plt.figure(figsize=(figsize))
    for j, net_num in enumerate(net_nums):
        wts = an[1][net_num]
        resps = an[0][net_num]  
        for layer_name in layer_names:
            i=i+1
            plt.subplot(len(net_nums), len(layer_names), i)
            x = wts[wts.layer_label==wt_layer_name_dict[layer_name]]
            y = resps[resps.layer_label.values.astype(str)==layer_name]
            plt.scatter(x, y, s=2, edgecolors='none', color='k', alpha=alpha)
            plt.axis('square')
            plt.xlim(-0.1,1);plt.ylim(-0.1,1)
            plt.yticks([0, 0.25, 0.5, 0.75, 1]);plt.gca().set_yticklabels(['','','','',''])
            plt.xticks([0, 0.25, 0.5, 0.75, 1]);plt.gca().set_xticklabels(['','','','',''])
            
            plt.title(str(np.round(np.corrcoef(x,y)[0,1], 2)))
            if i == 1:
                plt.yticks([0,0.25,0.5, 0.75, 1]);plt.gca().set_yticklabels(['0','','0.5','','1'])
                plt.xticks([0,0.25,0.5, 0.75, 1]);plt.gca().set_xticklabels(['0','','0.5','','1'])
                plt.xlabel('Weight Cov.')
                plt.ylabel('Response Cov.')
            if j == 0:
                plt.title(layer_name + '\n r='+ str(np.round(np.corrcoef(x,y)[0,1], 2)))
            plt.grid();
        
    return fig

def ti_ti_cov_scatters(net_nums1, net_nums2,  layer_names, an):
#net_nums = range(len(an[1][:]))
    i = 0 
    wt_layer_name_dict = {'relu1':'conv1', 'norm1':'conv1', 'pool1':'conv1', 'conv1':'conv1',
                      'relu2':'conv2', 'norm2':'conv2', 'pool2':'conv2', 'conv2':'conv2',
                      'relu3':'conv3','conv3':'conv3',
                      'relu4':'conv4', 'conv4':'conv4',
                      'relu5':'conv5', 'conv5':'conv5', 'pool5':'conv5',
                      'fc6':'fc6', 'relu6':'fc6'} 
    fig = plt.figure(figsize=(len(layer_names)*2, len(net_nums1)*2))
    for j, net_num1, net_num2 in zip(range(len(net_nums1)), net_nums1, net_nums2):
        resps1 = an[0][net_num1]
        resps2 = an[0][net_num2]  
        for layer_name in layer_names:
            i=i+1
            plt.subplot(len(net_nums1), len(layer_names), i)
            x = resps1[resps1.layer_label.values.astype(str)==layer_name]
            y = resps2[resps2.layer_label.values.astype(str)==layer_name]
            plt.scatter(x, y, s=2, edgecolors='none', color='k')
            plt.plot([0,1],[0,1])
            plt.axis('square')
            plt.xlim(-0.1,1);plt.ylim(-0.1,1)
            plt.yticks([0, 0.25, 0.5, 0.75, 1]);plt.gca().set_yticklabels(['','','','',''])
            plt.xticks([0, 0.25, 0.5, 0.75, 1]);plt.gca().set_xticklabels(['','','','',''])
            
            plt.title(str(np.round(np.corrcoef(x,y)[0,1], 2)))
            if i == 1:
                plt.yticks([0,0.25,0.5, 0.75, 1]);plt.gca().set_yticklabels(['0','','0.5','','1'])
                plt.xticks([0,0.25,0.5, 0.75, 1]);plt.gca().set_xticklabels(['0','','0.5','','1'])
                plt.xlabel('Response Cov. Orig')
                plt.ylabel('Response Cov. Adj')
            if j == 0:
                plt.title(layer_name + '\n r='+ str(np.round(np.corrcoef(x,y)[0,1], 2)))
            plt.grid();
        
    return fig
#%%
net_nums = [0,]
layer_names = ['conv2', 'relu2','pool2', 'norm2', 'conv3','relu3', 
               'conv4','relu4', 'conv5', 'relu5','pool5', 'fc6', 'relu6']
ti_wt_cov_scatters(net_nums, layer_names, an)  
 
plt.savefig(top_dir+'/analysis/figures/images/ti/ti_wt_cov_orig.pdf', bbox_inches='tight')    
#%%
net_nums = [0, 1, 2, 3]
layer_names = ['conv2', 'conv3','conv4', 'conv5', 'fc6']
ti_wt_cov_scatters(net_nums,layer_names, an) 
plt.tight_layout()
plt.savefig(top_dir+'/analysis/figures/images/ti/ti_wt_cov_conv2_adj.png', bbox_inches='tight')    

#%%
net_nums1 = [0, 0,]
net_nums2 = [1, 3,]
layer_names = ['conv2', 'conv3','conv4', 'conv5', 'fc6']

ti_ti_cov_scatters(net_nums1, net_nums2,  layer_names, an)
plt.tight_layout()
plt.savefig(top_dir+'/analysis/figures/images/ti/ti_wt_cov_conv2_adj_ti_comp.png', bbox_inches='tight')

#%%%
net_nums = [6,8,9,10,11,12,13,14,7]
layer_names = [ 'fc6',]
layer_name = 'fc6' 
i=0
x_lst = []
y_lst = []
for j, net_num in enumerate(net_nums):
    wts = an[1][net_num]
    resps = an[0][net_num]  
    x = wts[wts.layer_label==wt_layer_name_dict[layer_name]]
    y = resps[resps.layer_label.values.astype(str)==layer_name]
    plt.scatter(x,y, s=1, alpha=0.01, color='k')
    x_lst.append(np.mean(x))
    y_lst.append([np.mean(y), np.std(y)])

x = np.array(x_lst)
y = np.array(y_lst)[:, 0]
y_err = np.array(y_lst)[:, 1]
plt.errorbar(x,y,y_err, color='r')
plt.axis('square')
plt.xlim(-0.1,1);plt.ylim(-0.1,1)

#%%
net_nums = [0,]
layer_names = ['fc6',]
ti_wt_cov_scatters(net_nums, layer_names, an, alpha=0.2, figsize=(4,4)) 


net_num = 0
wts = an[1][net_num]
resps = an[0][net_num]
layer_name = 'fc6'  
x = wts[wts.layer_label==wt_layer_name_dict[layer_name]].values
y = resps[resps.layer_label.values.astype(str)==layer_name].values
bins1 = np.linspace(0., 0.4, 10)
bins2 = bins1 + bins1[1] - bins1[0]
binx_mean = []
biny_mean = []
biny_sd = []
i = 0
for bin1, bin2 in zip(bins1, bins2):
    i = i+1
    print(i)
    x_ind = (bin1<=x)*(bin2>x)
    
    if np.sum(x_ind)>0:
        x_in_bin = x[x_ind]
        y_in_bin = y[x_ind]
        binx_mean.append((bin1+bin2)/2.)
        biny_mean.append(np.mean(y_in_bin))
        biny_sd.append(np.std(y_in_bin))

plt.errorbar(binx_mean, biny_mean, biny_sd, color='k')
x = np.array(x_lst)

y = np.array(y_lst)[:, 0]
y_err = np.array(y_lst)[:, 1]
plt.errorbar(x, y, y_err, color='c', alpha=0.8)
plt.plot(x, y + y_err, color='c', alpha=0.4)
plt.plot(x, y - y_err, color='c', alpha=0.4)

plt.axis('square')
plt.xlim(-0.1,1);plt.ylim(-0.1,1)        
plt.annotate('Wt. Cov. Adj.', [0.5,0.5], color='c', fontsize=12)    
plt.annotate('Wt. Cov. Orig.', [0.15,0.1], color='k', fontsize=12) 
   
plt.savefig(top_dir+'/analysis/figures/images/ti/fc6_adj_vs_orig.pdf', bbox_inches='tight')
  
    
    
















    
