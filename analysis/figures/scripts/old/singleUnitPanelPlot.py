# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 09:48:40 2015

@author: dean
"""

#if 'resp' not in locals():
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as  l
import copy
import xray as xr

mat = l.loadmat('v4likeInd.mat')
v4like = mat['v4likeInd']
v4like = v4like[0,:]
transKey='x'
#plotting translation amplitude, correlation, 
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
        transKey = da.dims[1]
        c = [np.corrcoef( da[:,:, unit].values.T) for unit in range(len(da.coords['unit'].values)) ]
        respC.append(xr.DataArray( c, coords = [da.coords['unit'], da.coords[transKey], da.coords[transKey]] ))


unitLayer = [ [1,6], [187, 2689 ]] #most v4 like
unitLayer = [ [0,1,6], [0, 55, 33 ]]
fig = plt.figure()
nUnits = np.size(unitLayer,1)
totInd=0
for ind in range(nUnits):

    form = 'eps'
    layer = unitLayer[0][ind]
    unit = unitLayer[1][ind]
        
    #the trans to scatter
    trans1 = 0
    trans2 = 7
    refCorInd = 10    
    
    da = resp [layer]
    m = da.mean(dim = 'shape', skipna = True)
    
    mp = m.loc[dict( unit = [unit] )]
    mp = (mp / np.max(mp).values) *50
    
    totInd+=1
    plt.subplot(nUnits,3,totInd)
    plt.stem( mp.coords[transKey].values, mp.values)
    
    #put a shape on there 
    plt.axis([-114, 114, -10, 120])   
    mat = l.loadmat('PC3702001ShapeVerts.mat')
    s = np.array(mat['shapes'][0])
    #make it the right size
    shape = s[1]#reference
    radius = np.max(shape)
    pixSize = 51
    radius = np.max(shape)
    shape = s[1]
    shape = (shape/(radius*2)) * pixSize
    shape[:,1] = shape[:,1] + 89
    shapeToPlot = copy.deepcopy(shape[:,:])
    shapeToPlotList= np.empty((1,2))

    shapeToPlotInds = np.arange(9,12, 1, dtype=np.intp)
    shapeToPlotInds = [10,11]
    positions = mp.coords[transKey].values[shapeToPlotInds]
    
    for pos in positions:
            shapeToPlot[:,0] = shape[:,0] + pos
            shapeToPlotList = np.vstack((shapeToPlotList, np.vstack(( [None,None] , shapeToPlot) )))
            
    plt.stem(mp.coords[transKey].values[shapeToPlotInds], np.ones(len(shapeToPlotInds))*np.nanmin(shapeToPlot[:,1]),'k', markerfmt=" ")
    line = plt.Polygon( shapeToPlotList, closed=True, fill='none', edgecolor='w',fc='k')
    plt.gca().add_patch(line)
    fig.canvas.draw()
    
    plt.xlabel(str(transKey) +' (pixels)')
    
    plt.ylabel('Normalized Response')
    plt.yticks([0, np.nanmax(mp.values)])
    
    labels = [item.get_text() for item in plt.gca().get_yticklabels()]
    
    labels[0] = 0
    labels[1] = 1
    
    plt.gca().set_yticklabels(labels)
    
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='on',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='on')
    title ='Layer ' + str(layer +1) +', Unit ' +str(unit) + ' '
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    
    name = 'amp'  + transKey
    
    
    m = respC[layer]

    mp = m.loc[dict( unit = [unit] )]
    mp = (mp / np.max(mp).values) *50
    
    coords = mp.coords[transKey].values
    c = np.reshape(mp.values[ :, refCorInd], (21,))
    totInd+=1
    
    plt.subplot(nUnits,3,totInd)
    plt.stem( coords, c)
    plt.yticks([0, np.nanmax(mp.values)])
    labels = [item.get_text() for item in plt.gca().get_yticklabels()]
    
    labels[0] = 0
    labels[1] = 1
    plt.gca().set_yticklabels(labels)
    
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='on',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='on')
    #put a shape on there 
    plt.axis([-114, 114, -20, 120])   
    mat = l.loadmat('PC3702001ShapeVerts.mat')
    s = np.array(mat['shapes'][0])
    #make it the right size
    shape = s[1]#reference
    radius = np.max(shape)
    pixSize = 51
    radius = np.max(shape)
    shape = s[1]
    shape = (shape/(radius*2)) * pixSize
    shape[:,1] = shape[:,1] + 89
    shapeToPlot = copy.deepcopy(shape[:,:])
    shapeToPlotList= np.empty((1,2))

    shapeToPlotInds = np.arange(9,12, 1, dtype=np.intp)
    shapeToPlotInds = [10,11]
    positions = mp.coords[transKey].values[shapeToPlotInds]

    for pos in positions:
        shapeToPlot[:,0] = shape[:,0] + pos
        shapeToPlotList = np.vstack((shapeToPlotList, np.vstack(( [None,None] , shapeToPlot) )))
        
    plt.stem(mp.coords[transKey].values[shapeToPlotInds], np.ones(len(shapeToPlotInds))*np.nanmin(shapeToPlot[:,1]),'k', markerfmt=" ")
    line = plt.Polygon( shapeToPlotList, closed=True, fill='none', edgecolor='w',fc='k')
    plt.gca().add_patch(line)
    fig.canvas.draw()
    
    plt.xlabel(str(transKey) +' (pixels)')

    plt.ylabel('Correlation')
    
    plt.gca().set_aspect('equal', adjustable='box') 
    
    
    labels = [item.get_text() for item in plt.gca().get_yticklabels()]
    
    labels[0] = 0
    labels[1] = 1
    
    plt.gca().set_yticklabels(labels)
    
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='on',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='on')
    title ='Layer ' + str(layer +1) +', Unit ' +str(unit) + ' '
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.show()
    
    name = 'corr' + transKey
    

    temp1 = {}
    temp2 = {}
    temp1[transKey] = trans1
    temp2[transKey] = trans2
    
    x = da.loc[dict(unit=unit)].loc[temp1].values
    y = da.loc[dict(unit=unit)].loc[temp2].values
    
    totInd+=1
    plt.subplot(nUnits,3,totInd)
    plt.scatter( x, y , facecolors='none')
    plt.xticks([0, round(np.nanmax(x))])
    plt.yticks([0, round(np.nanmax(y))])
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x= ' + str(trans1))
    plt.ylabel('x= ' + str(trans2))
    name = 'scatter' + transKey 
        
plt.tight_layout()   