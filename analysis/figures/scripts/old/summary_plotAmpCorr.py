# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 17:50:45 2015

@author: dean
"""

#plotting summary, translation amplitude, correlation, 
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as  l
import copy
import xray as xr
form ='eps'


mat = l.loadmat('v4likeInd.mat')
v4like = mat['v4likeInd']
v4like = v4like[0,:]


if 'resp' not in locals():
    f = open('/Users/dean/Desktop/AlexNet_APC_Analysis/AlexNet_370PC2001xray_shape370_x21.pickle')
    resp = pickle.load(f)
    resp = resp[0]
    f.close()
    respC = []
    #make the corrcoeffMatrices
    
    for layer in range(len(resp)):
        print layer
        da = resp [layer]
        c = [np.corrcoef( da[:,:, unit].values.T) for unit in range(len(da.coords['unit'].values)) ]
        
        respC.append(xr.DataArray( c, coords = [da.coords['unit'], da.coords['x'], da.coords['x']] ))
        
##mask by v4ness 
#for layer in range(len(resp)):    
#    v4likeInds = np.where(v4like[layer])[0]    
#    respC[layer] = respC[layer][v4likeInds,:,:]
#    resp[layer] = resp[layer][:,:, v4likeInds]

meanCorr=[]   
for layer in range(len(resp)):
    meanCorr.append(respC[layer][:,10,:].mean(dim = 'unit', skipna = True).values[10,:])

meanAmp=[]   
for layer in range(len(resp)):
    
    meanAmp.append(resp[layer].mean(dim = ('shape', 'unit' )).values)


number = 8
cmap = plt.get_cmap('jet')
pos = da.coords['x'].values
colors = [cmap(i) for i in np.linspace(0, 1, number)]

layer =8
for color in colors:
    layer -=1
    mCorSlice = meanCorr[layer]
    plt.plot(pos,  mCorSlice, color=color, label=layer+1, lw=2)
    plt.xlabel('x (pixels)')
    plt.ylabel('Correlation')
    plt.yticks([0, 0.5, 1])
    plt.xticks(pos[::2])
    
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='on',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='on')

plt.legend(loc = 'center right', bbox_to_anchor=(1., 0.5), ncol=1, fancybox=True, title='Layer')
title = 'Mean Correlation by Layer Across Units'
plt.savefig('/Users/dean/Desktop/CRCNS_figs/' + title + '.'+form, format=form)


plt.figure()

layer =8
for color in colors:
    layer -=1
    mSlice = meanAmp[layer]/np.max(meanAmp[layer])
    plt.plot(pos,  mSlice, color=color, label=layer+1, lw=2)
    plt.xlabel('x (pixels)')
    plt.ylabel('Normalized Response')
    plt.yticks([0, 0.5, 1])
    plt.xticks(pos[::2])
    
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='on',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='on')

plt.legend(loc = 'center right', bbox_to_anchor=(1., 0.5), ncol=1, fancybox=True, title='Layer')
title = 'Mean Response By Layer Across Units'

plt.savefig('/Users/dean/Desktop/CRCNS_figs/' + title + '.'+form, format=form)

