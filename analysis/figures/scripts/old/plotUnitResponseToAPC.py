# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 12:53:01 2015

@author: dean
"""


import numpy as np
import scipy.io as  l
import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches

plt.close('all')
maindir = '/Users/dean/Desktop/'


os.chdir( maindir)
#vertices
mat1 = l.loadmat('APC_PC370.mat')
apc = mat1['orcurv']
apc.shape = (370)

#firing rates
mat2 = l.loadmat('V4_370PC2001.mat')
resp = mat2['resp'][0][0]

cell=108

##transparency
#plt.plot(x, y, 'r-', alpha=0.7)
## lines with points
#plt.plot(dates, values, '-o')
plt.subplot(212)
[n, bins, patches] = plt.hist(resp[cell, : ],30, color = 'g')

plt.bar(bins[0:-1], n[0:], width= bins[2]-bins[1], color= (0.5,0.5,0.5))

#lets choose some bins
lowBin = 6
highBin = 15

plt.bar(bins[0:lowBin], n[0:lowBin], width= bins[2]-bins[1], color= 'g')
plt.bar(bins[highBin:-1], n[highBin:], width= bins[2]-bins[1], color= 'b')
plt.xlabel('Normalized Response')
plt.ylabel('Shape Count')

plt.scatter(bins[0:30:3], np.ones(np.size(bins[0:30:3]))*60, s = bins[0:30:3]*500,facecolors='none', edgecolors='k')


plt.legend(handles=[mpatches.Patch(color='green', label='Low Response'), mpatches.Patch(color='blue', label='High Response')], loc = 7)



#find the sort shapes 

s = resp[cell, : ]
sortShapesInds = np.argsort(s)[-1::-1]
sSort = s[sortShapesInds]

lowShapesInds = sortShapesInds[np.where(sSort<bins[lowBin])]
highShapesInds = sortShapesInds[np.where(sSort>bins[highBin])]

plt.subplot(211)
#lets try the top values

for shape in sortShapesInds[:]:
    s = resp[cell, shape ]*500 + 1
    
    plt.scatter(apc[shape][:,0], apc[shape][:,1], s = s , alpha=0.2,facecolors='none', edgecolors='k')
    #plt.plot(apc[shape][:,0], apc[shape][:,1],  alpha=0.01, color = 'r')

for shape in highShapesInds:
    s = resp[cell, shape ]*500 + 1
    
    plt.scatter(apc[shape][:,0], apc[shape][:,1], s = s , alpha=0.3,facecolors='b', edgecolors='k')
    plt.plot(apc[shape][:,0], apc[shape][:,1],  alpha=0.1, color = 'r')
    
    
for shape in lowShapesInds:
    s = resp[cell, shape ]*500 + 1
    plt.scatter(apc[shape][:,0], apc[shape][:,1], s = s , alpha=0.3, facecolors='g', edgecolors='k')
    #plt.plot(apc[shape][:,0], apc[shape][:,1],  alpha=0.01, color = 'r')
plt.xlabel('Orientation (radians)')    
plt.ylabel('Curvature') 








plt.show()

