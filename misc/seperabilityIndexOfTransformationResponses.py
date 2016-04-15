# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 16:10:45 2015

@author: dean
"""

import pandas as pd
#receptive field analysis
import matplotlib.cm as cm
import matplotlib.pyplot as p
import xray as xr
import numpy as np
import pickle as pk
from glue.core import Data, DataCollection
import seaborn as sbs


if 'resp' not in locals():
    f = open( '/Users/dean/Desktop/AlexNet_APC_Analysis/AlexNet_370PC2001xray_shape370_x21.pickle' )
    resp = pk.load(f)
    f.close()
    resp=resp[0]
    
sFracList = []
sparsel = []
for layer in range(8):
    
    r = resp[layer]

    r = r.transpose( 'unit', 'shape', 'x',  )
    
    rp = r.values
    
    #reshape
    ntrans= np.product(np.array(r.shape[2:]))
    
    rpt = np.reshape(rp, (rp.shape[0],rp.shape[1], ntrans))
    
    #mark sparse units
    a = np.sum(np.abs(rpt), 1)
    maxfracTrans = np.max(np.abs(a), 1) / np.sum(np.abs(a),1)    
    
    a = np.sum(np.abs(rpt), 2)
    maxfracID = np.max(np.abs(a), 1) / np.sum(np.abs(a),1)
    sparsel.append(np.all([maxfracID<0.5, maxfracTrans<0.5],0))
    
    u, s, v = np.linalg.svd(rpt, full_matrices=False)
    #columns of u will be the unit length feature sensitvity (e.g. spatial receptive field)
    #rows of v will be the unit length receptive fields, across transformation
    #s is how much variance each of these transformation invariant receptive fields accounts for the variance of the response


    
    sFrac=s[:,0]/np.sum(s,1)
    sFracList.append(sFrac)



#means= np.zeros(8)
#for ind in range(8):
#    means[ind]=np.nanmean(sList[ind])
#    
for ind in range(8):
    notSparse = sFracList[ind][sparsel[ind]]    
    p.scatter( ind*np.ones(np.size(notSparse)), notSparse)