# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 19:12:52 2015

@author: dean
"""



import numpy as np
import scipy.io as  l
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches
import pickle

pcshape = 100
plt.close('all')

# give it a list of folders and it will get their parameters from insid and generate 
#images from it
baseImageList = [ 'PC370', 'formlet', 'PCunique', 'natShapes']
baseStim = baseImageList[0] 


stimDir = '/Users/dean/Desktop/shapenet/stim/'
baseStimDir = stimDir + 'basestim/' +baseStim + '/'

img = plt.imread(baseStimDir + str(pcshape) + '.png')


maindir= '/Users/dean/Desktop/'
os.chdir( maindir)
#vertices
mat1 = l.loadmat('APC_PC370.mat')
apc = mat1['orcurv']
apc.shape = (370)

#firing rates
mat2 = l.loadmat('V4_370PC2001.mat')
resp = mat2['resp'][0]
fig, ax = plt.subplots()



ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.subplot(121)
plt.imshow(img, cmap= plt.cm.Greys_r, interpolation = 'none')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
#for s in apc[:]:
#    plt.scatter(s[:,0],s[:,1], color='r')
s = apc[pcshape]
plt.scatter(360*( s[:,0]/(2.0*np.pi) ), s[:,1], color='r')
plt.xticks([0,90,180, 270, 360])
plt.gca().xaxis.set_ticks_position('none') 
plt.gca().yaxis.set_ticks_position('none') 
plt.gca().set_aspect('auto')

    
plt.xlim((0,360))
plt.ylim((-1,1.1))

plt.xlabel('Angular Position (degrees)')
plt.ylabel('Normalized Curvature')
plt.rcParams.update({'font.size': 20})
plt.tight_layout()