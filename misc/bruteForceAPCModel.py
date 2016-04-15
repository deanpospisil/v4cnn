# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 13:04:02 2015

@author: dean
"""
import scipy.io as  l
import scipy.stats as st
import numpy as np
from sklearn.utils.extmath import cartesian
pi = np.pi
#import dMisc as m
import itertools
#import matplotlib.pyplot as plt
from random import shuffle
import os
import pickle
import pandas as pd


import pandas as pd
#import seaborn as sns
#sns.set_context("talk", font_scale=1.4)

def apc370models(nMeans=10, nSD=10, perc=5):
    #the parameters of the shapes

    mat = l.loadmat('/Users/deanpospisil/Desktop/net_code/analysis/data/models/PC2001370Params.mat')
    s = mat['orcurv'][0]
    
    #adjustment for repeats [ 14, 15, 16,17, 318, 319, 320, 321] 
    a = np.hstack((range(14), range(18,318)))
    a = np.hstack((a, range(322, 370)))
    s = s[a]
        
    
    nStim = np.size(s,0) 

    angularPosition = []
    curvature = []
    paramLens = []
    
    for shapeInd in range(nStim):
        angularPosition.append(s[shapeInd][:, 0])
        curvature.append(s[shapeInd][:, 1])
        paramLens.append(np.size(s[shapeInd],0))
        
    angularPosition = np.array(list(itertools.chain.from_iterable(angularPosition)))
    angularPosition.shape = (np.size(angularPosition),1)
    
    curvature = np.array(list(itertools.chain.from_iterable(curvature)))
    curvature.shape = (np.size(curvature),1)
    
    #variable section length striding
    inds = np.empty((2,np.size(paramLens)),dtype = np.intp)
    inds[1,:] = np.cumsum(np.array(paramLens), dtype = np.intp) #ending index
    inds[0,:] = np.concatenate(([0,], inds[1,:-1])) #beginning index
    
    
    
    
#    #the Nonlin fit model for Pasupathy V4 Neurons
#    mat = l.loadmat('V4_370PC2001_LSQnonlin.mat')
#    f = np.array(mat['fI'][0])[0]
#    # orientation, curvature, orientation SD , curvature SD , correlation
#    
#    #use these to generate parameters for brute force model
#    maxAngSD = np.percentile(f[:,2], 100 - perc)
#    minAngSD = np.percentile(f[:,2], perc)
#    maxCurSD = np.percentile(f[:,3], 100 - perc)
#    minCurSD = np.percentile(f[:,3], perc)
    
    maxAngSD = np.deg2rad(171)
    minAngSD = np.deg2rad(23)
    maxCurSD = 0.98
    minCurSD = 0.09
    
    #make this into a pyramid based on d-prime
    orMeans = np.linspace(0, 2*pi-2*pi/nMeans, nMeans) 
    orSDs = np.logspace(np.log10(minAngSD),  np.log10(maxAngSD),  nSD)
    curvMeans = np.linspace(-0.5,1,nMeans)
    curvSDs = np.logspace(np.log10(minCurSD),  np.log10(maxCurSD),  nSD)
    modelParams = cartesian([orMeans,curvMeans,orSDs,curvSDs])
    nModels = np.size( modelParams, 0)
    
    a = st.vonmises.pdf(angularPosition, kappa = modelParams[:,2]**-1 , loc =  modelParams[:,0]) # 
    b = st.norm.pdf(curvature, loc = modelParams[:,1],  scale = modelParams[:,3])
    temp = a * b

    models = np.empty(( 362, nModels ))
    
    for shapeInd in range(nStim):
        models[ shapeInd, : ] = np.max( temp[ inds[ 0, shapeInd ] : inds[ 1 , shapeInd ] , : ] ,  axis = 0 )
    
    models = models - np.mean(models,axis = 0)
    magnitude = np.linalg.norm( models, axis = 0)
    magnitude.shape=(1,nModels)
    models = models / magnitude
    del a,b, temp
    return models, modelParams

def modelFits(resp, models):
    resp = resp - np.mean(resp,axis = 0)
    resp = resp / np.linalg.norm( resp, axis = 0)
    
    #shuffle(resp)
    cov = np.dot(resp.T, models)
    
    bestFitInd = np.argmax((cov),1)
    bestr = cov[(range(cov.shape[0]), bestFitInd) ].T
    fits =  modelParams[ bestFitInd, : ]
    return fits, bestr



def nanargmax(a):
    idx = np.argmax(a, axis=None)
    multi_idx = np.unravel_index(idx, a.shape)
    if np.isnan(a[multi_idx]):
        nan_count = np.sum(np.isnan(a))
        # In numpy < 1.8 use idx = np.argsort(a, axis=None)[-nan_count-1]
        idx = np.argpartition(a, -nan_count-1, axis=None)[-nan_count-1]
        multi_idx = np.unravel_index(idx, a.shape)
    return multi_idx    

def save_pandas(fname, data):
    '''Save DataFrame or Series
    Parameters
    ----------
    fname : str
        filename to use
    data: Pandas DataFrame or Series
    '''
    np.save(open(fname, 'w'), data)
    if len(data.shape) == 2:
        meta = data.index,data.columns
    elif len(data.shape) == 1:
        meta = (data.index,)
    else:
        raise ValueError('save_pandas: Cannot save this type')
    s = pickle.dumps(meta)
    s = s.encode('string_escape')
    with open(fname, 'a') as f:
        f.seek(0, 2)
        f.write(s)

def load_pandas(fname, mmap_mode='r'):
    '''Load DataFrame or Series
    Parameters
    ----------
    fname : str
        filename
    mmap_mode : str, optional
        Same as numpy.load option
    '''
    values = np.load(fname, mmap_mode=mmap_mode)
    with open(fname) as f:
        numpy.lib.format.read_magic(f)
        numpy.lib.format.read_array_header_1_0(f)
        f.seek(values.dtype.alignment*values.size, 1)
        meta = pickle.loads(f.readline().decode('string_escape'))
    if len(meta) == 2:
        return pd.DataFrame(values, index=meta[0], columns=meta[1])
    elif len(meta) == 1:
        return pd.Series(values, index=meta[0])

#now find the best model for the data
if 'models' not in locals():
    models, modelParams = apc370models(nMeans=16, nSD=10, perc=5)
    
####firing rates
##maindir = '/Users/deanpospisil/Desktop/bictPresent/'
##os.chdir( maindir)
##mat = l.loadmat('V4_370PC2001.mat')
##resp = mat['resp'][0][0]
##resp = resp.T
####adjustment for repeats [ 14, 15, 16,17, 318, 319, 320, 321] 
##a = np.hstack((range(14), range(18,318)))
##a = np.hstack((a, range(322, 370)))
##resp = resp[a]
###
##fits, bestrV4 = modelFits(resp, models)
###
#bestralexl = []
#bestralexll= []
#layerFits = []
#layerSizeFits = []
#  
#scramlayerSizeFits=[]
#scambestralexll=[]
#scramlayerFits = []
#scrambestralexl = []
#
#rfs = [ 51, 99, 131, 163 ]
##rfs = [51,]
#layers = range(8)
#
#for rf in rfs:
#    
#    print rf
#    mat = l.loadmat('AlexNet_' +str(rf) + 'rfs370PC2001.mat')
#    bestralexl = []
#    
#    for layer in layers:
#        resp = mat['resp'][0][layer].T
#        resp = resp[a]
#        #resp[resp<0] = 0
#        fits, bestralex = modelFits(resp, models)
#        
#        layerFits.append(fits)
#        bestralexl.append(bestralex)
#        
#        np.random.shuffle(resp)
#        fits, bestralex = modelFits( resp , models )
#        
#        scramlayerFits.append(fits)
#        scrambestralexl.append(bestralex)
#
#    layerSizeFits.append(layerFits)
#    bestralexll.append(bestralexl)
#    
#    scramlayerSizeFits.append(scramlayerFits)
#    scambestralexll.append(scrambestralexl)    
#    
#best = nanargmax(bestralex)
#first = True
#result = pd.DataFrame()
##ang mean, curv mean, ang sd, curv,sd, r, layer, rfs, scram
#for scram in range(2):
#    if scram is 0:
#        allr = bestralexll
#        allfit = layerSizeFits
#    else:
#        allr = scambestralexll
#        allfit = scramlayerSizeFits
#            
#    for rf in rfs:
#        for layer in layers:
#            r=allr[rf][layer]
#            fits = allfit[rf][layer]
#            if first:
#                result = pd.DataFrame( [fits, r, np.tile(layer,(np.shape(r)[0],1)), np.tile(rf,(np.shape(r)[0],1)),np.tile(scram,(np.shape(r)[0],1)) ])
#                first = False
#            else:
#                df = pd.DataFrame( [fits, r, np.tile(layer,(np.shape(r)[0],1)), np.tile(rf,(np.shape(r)[0],1)),np.tile(scram,(np.shape(r)[0],1)) ])
#                result = pd.concat(result,df)
#
#
#result.columns =c[ 'am', 'cm', 'asd', 'csd', 'r', 'layer', 'rfs', 'scram']
#
#save_pandas('layerFitsAlexNetRF51_99_131_163', result)
#    
#
#plt.close('all')
#perV4Like = np.zeros(np.shape(bestralexll))
#for rf in range(4):
#    for layer in range(8):
#        perV4Like[rf,layer] = np.sum(bestralexll[rf][layer]>0.5)/np.double(len(bestralexll[rf][layer]))
#
#import seaborn
#N = 8
#
#ind = np.arange(N)  # the x locations for the groups
#width = 0.2
#
#fig, ax = plt.subplots(figsize=(12,6))
#plotInd=0
#colors = ('b', 'g', 'c', 'm', 'r',  'y', 'k')
#for plotInd in range(4):
#    ax.bar(ind-0.4 + width*plotInd, perV4Like[plotInd,:], width, color = colors[plotInd] )
#plt.show()
#
#legend = plt.legend( ['51 pixels','99 pixels','131 pixels','163 pixels'], title = 'Max Stim. Size', loc = 'upper left', fontsize = 12,frameon=True )
#plt.setp(legend.get_title(),fontsize=15)
#legend.get_frame().set_facecolor('white')
#
#step = 0.05
#ticks= np.arange(0,0.75+step, step)
#plt.yticks(np.arange(0,0.75+step, step))
#labels = [str(t) for t in ticks]
#for ind in np.arange(1, len(labels), 2):
#    labels[ind] = ' '
#plt.gca().set_yticklabels(labels, fontsize =20)
#
#plt.xticks([0,1,2,3,4,5,6,7])
#plt.gca().set_xticklabels(['1', '2','3','4','5','6', '7', '8'], fontsize =20)
#
#plt.gca().xaxis.set_ticks_position('none') 
#plt.gca().yaxis.set_ticks_position('none') 
#
#plt.title('V4-like Units by Layer of AlexNet', fontsize = 20)
#plt.xlabel('Layer', fontsize = 20)
#plt.ylabel('Percent Units V4-like', fontsize = 20)
#plt.tight_layout()

#
##firing rates
#maindir = '/Users/dean/Desktop/AlexNet_APC_Analysis/'
#os.chdir( maindir)
#mat = l.loadmat('V4_370PC2001.mat')
#resp = mat['resp'][0][0]
#resp = resp.T
#fits, bestrV4 = modelFits(resp, models)
#np.random.shuffle(resp)
#fits, bestrV4scramble = modelFits(resp, models)
#
#bestralexl = []
#mat = l.loadmat('AlexNet_51rfs370PC2001.mat')
#for layer in range(8):
#    resp = mat['resp'][0][layer].T
#    resp = resp[a]
#    fits, bestralex = modelFits(resp, models)
#    bestralexl.append(bestralex)
#bestralexl=np.array(list(itertools.chain.from_iterable(bestralexl)))
#  
#  
#bestralexlscramble = []  
#for layer in range(8):
#    resp = mat['resp'][0][layer].T
#    resp = resp[a]
#    np.random.shuffle(resp)
#    fits, bestralex = modelFits(resp, models) 
#    bestralexlscramble.append(bestralex)
#bestralexlscramble=np.array(list(itertools.chain.from_iterable(bestralexlscramble)))

#plt.close('all')
#plt.figure(figsize=(12,6))
#plt.subplot(121)
#plt.hist(bestrV4,bins=20, range=(0,1))
#plt.hist(bestrV4scramble,bins=20, range=(0,1), alpha = 0.7)
#plt.gca().xaxis.set_ticks_position('none') 
#plt.gca().yaxis.set_ticks_position('none') 
#plt.title('V4')
#plt.xlabel('Best Fit Correlation')
#plt.ylabel('Unit Count')
#
#plt.xticks([0,0.25,0.5,0.75,1])
#labels = [item.get_text() for item in plt.gca().get_xticklabels()]
#labels[0] = '0'
#labels[-1] = '1'
#plt.gca().set_xticklabels(labels)
#
#plt.gca().xaxis.set_ticks_position('none') 
#plt.gca().yaxis.set_ticks_position('none') 
#plt.yticks(np.arange(10, 60, 20))
#plt.legend(('Original', 'Scrambled'),loc = 'best')
#
#
##
##plt.subplot(122)
#plt.title('AlexNet All Layers')
#bestralexl = bestralexl[-np.isnan(bestralexl)]
#plt.hist(bestralexl,bins=20, range=(0,1))
#bestralexlscramble = bestralexlscramble[-np.isnan(bestralexlscramble)]
#plt.hist(bestralexlscramble,bins=20, range=(0,1), alpha = 0.7)
#
#plt.xticks([int(0),0.25,0.5,0.75,1])
#labels = [item.get_text() for item in plt.gca().get_xticklabels()]
#labels[0] = '0'
#labels[-1] = '1'
#plt.gca().set_xticklabels(labels)
#
#plt.gca().xaxis.set_ticks_position('none') 
#plt.gca().yaxis.set_ticks_position('none') 
#plt.yticks(np.arange(500, 3500+1000, 1000))
#plt.tight_layout()

#plt.scatter( fits[:,0]+np.random.randn(np.shape(fits)[0])*0.01, fits[:,1]+np.random.randn(np.shape(fits)[0])*0.01)

#
#
#
##the Nonlin fit model for Pasupathy V4 Neurons
#mat = l.loadmat('V4_370PC2001_LSQnonlin.mat')
#v4 = np.array(mat['fI'][0])[0]
#v4 = v4[v4[:,-1]>0.5]
#
##the Nonlin fit model for Pasupathy V4 Neurons
#layer = 5
#mat = l.loadmat('AlexNet_370PC2001_LSQnonlin')
#alex = np.array(mat['fI'][0])[layer]
#alex = alex[ -np.isnan(alex[:,-1]) ]
#alex = alex[alex[:,-1]>0.5]
#toPlot = [v4,alex]
##for s in apc[:]:
##    plt.scatter(s[:,0],s[:,1], color='r')
#plt.close('all')
#plt.figure(figsize=(12,6))
#for ind in range(2):
#    plt.subplot(1,2,ind+1)
#    plt.scatter(360*( toPlot[ind][ :, 2 ]/(2.0*np.pi) ), toPlot[ind][ :, 3 ], color='r', facecolors='none')
#    
#    if ind == 1:
#        plt.title('AlexNet Layer 5')
#    else:
#        plt.title('V4')
#        plt.xlabel('SD Angular Position (degrees)')
#        plt.ylabel('SD Normalized Curvature')
#       
#    
#    plt.gca().set_yscale('log')
#    plt.gca().set_xscale('log')
#    plt.xticks([1*10**-4, 1*10**-2, 1*10**-0 ,  1*10**2,   1*10**4] )
#    plt.yticks([1*10**-3,1*10**-2, 1*10**-1, 1*10**-0] )
#    plt.gca().xaxis.set_ticks_position('none') 
#    plt.gca().yaxis.set_ticks_position('none') 
#    #plt.gca().set_aspect('auto')
#
#    plt.xlim((0.01,360*10))
#    plt.ylim((0.0001,1*10))

#    

    
    #plt.tight_layout()

#
#modelParams[:,2]=360*( modelParams[:,2]/(2.0*np.pi) )
#modelParams[:,0]=360*( modelParams[:,0]/(2.0*np.pi) )

#make this into a pyramid based on d-prime

nMeans = 16
nSD = 10

maxAngSD = 171
minAngSD = 23
maxCurSD = 0.98
minCurSD = 0.09
#minAngSD = np.min(modelParams[:,2])
#maxAngSD= np.max(modelParams[:,2])
#minCurSD= np.min(modelParams[:,3])
#maxCurSD= np.max(modelParams[:,3])
orMeans = np.linspace(0, 2*pi-2*pi/nMeans, nMeans) 
orSDs = np.logspace(np.log10(minAngSD),  np.log10(maxAngSD),  nSD)
curvMeans = np.linspace(-0.5,1,nMeans)
curvSDs = np.logspace(np.log10(minCurSD),  np.log10(maxCurSD),  nSD)

cur = cartesian( (curvMeans, curvSDs)  )
ors = cartesian( (orMeans, orSDs)  )

plt.close('all')
plt.figure(figsize=(8,4))
for ind in range(2):
    plt.subplot(1,2,ind+1)
    
    if ind == 1:
        plt.scatter(cur[:,1], cur[:,0], color='b', facecolors='none', linewidth=1)

        plt.title('Curvature')
        plt.ylim((-1,1.1))
        plt.xlim((0,1))
        plt.gca().set_yticklabels([' ','-0.5','0','0.5', '1'])
        plt.xticks([0,0.5,1])
        
    else:
        plt.scatter( ors[:,1], 360*( ors[:,0]/(2.0*np.pi) ), color='b', facecolors='none', linewidth=1)
        plt.title('Angular Position')
        plt.ylim((-8,360))
        plt.xticks([0,90,180, 270, 360])
        plt.yticks([0,90,180, 270, 360])
    
        plt.xlabel('Standard Deviation')
        plt.ylabel('Mean')
        
    
    
    
    plt.gca().xaxis.set_ticks_position('none') 
    plt.gca().yaxis.set_ticks_position('none') 
    plt.gca().set_aspect('auto')

    
    

    plt.rcParams.update({'font.size': 20})
    #plt.tight_layout()
