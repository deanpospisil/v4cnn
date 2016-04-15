# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 20:06:46 2015

@author: dean
"""


import numpy as np
import scipy.io as  l
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import pandas as pd

plt.close('all')
maindir = '/Users/dean/Desktop/'

minSTDor = 0.28120000000000001
minSTDcur = 0.0424

os.chdir( maindir)
#vertices
mat1 = l.loadmat('APC_PC370.mat')
apc = mat1['orcurv']
apc.shape = (370)

#firing rates

mat = l.loadmat('AlexNet_370PC2001_BruteForceLogspaceSDPerc95')
mat = l.loadmat('AlexNet_51rfs370PC2001_BruteForceLogspaceSDPerc0')
resp = mat['fI'][0]

#minSTDcur = 100
#minSTDor = 100
##find min std fits
#for ind in range(8):
#    tminSTDcur = np.min(resp[ind][:, 2])
#    tminSTDor = np.min(resp[ind][:, 3])
#    
#    if tminSTDcur<minSTDcur:
#        minSTDcur = tminSTDcur
#        
#    if tminSTDor<minSTDor:
#        minSTDor = tminSTDor
    

mat = l.loadmat('AlexNet_51rfs370PC2001_BruteForceLogspaceSDPerc0_ScrambledResp')
respS = mat['fI'][0]

#mean or,  mean cur, std or, std cur 

#lets make a distribution of r values

#get all rs


r = []
rs = []
for ind in range(8):
    tempr=resp[ind][resp[ind][:,2]>minSTDor,:]
    tempr=tempr[tempr[:,3]>minSTDcur,:]
    r.append(tempr[:,-1])
    
    
    temprs=respS[ind][respS[ind][:,2]>minSTDor,:]
    temprs=temprs[temprs[:,3]>minSTDcur,:]
    rs.append(temprs[:,-1])
    
    
r = np.concatenate( r, axis=0 )
rs = np.concatenate( rs, axis=0 )
r = r[~np.isnan(r)]
rs = rs[~np.isnan(rs)]



n1, bins, patches = plt.hist(r,bins=50, range = (0,1))
n2, bins, patches = plt.hist(rs,bins=50,range = (0,1))
plt.legend(('Original', 'Scrambled'),loc = 'left')
plt.xlabel('r-value')
plt.ylabel('Number Units')

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().get_xaxis().tick_bottom()
plt.gca().get_yaxis().tick_left()
plt.xticks([0, .25, .5, .75, 1])
plt.gca().set_xticklabels(['0', .25, .5, .75, 1])
#plt.title('Fits to scrambled vs. original responses')


#lets find the max r values by layer
layer = resp[6]
d = {'mor': layer[:,0], 'mcur': layer[:,1], 'sdor':layer[:,2] ,'sdcur': layer[:,3], 'r':layer[:,4] }
df = pd.DataFrame(data = d, index = range(layer.shape[0]))

df = df[-np.isnan(df['r']) ]
df = df[minSTDcur<=df['sdcur']]
df = df[minSTDor<=df['sdor']]
df.idmax()
    


