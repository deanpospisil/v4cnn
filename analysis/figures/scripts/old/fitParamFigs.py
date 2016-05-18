# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 15:31:47 2015

@author: deanpospisil
"""
import numpy.lib
import numpy as np
import pandas as pd
import cPickle as pickle


import numpy as np

import matplotlib
matplotlib.use('GTK')
import matplotlib.pyplot as plt
import seaborn as sns

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


x=load_pandas('layerFitsAlexNetRF51_99_131_163', mmap_mode='r+')
x['am'] = (x['am']/(2*np.pi))*360
x['asd'] = (x['asd']/(2*np.pi))*360
k = x.keys()

threshFits = x[k[0:4]][(x['scram'].values==0)*(x['r'].values>0.5) ]
threshFits.column = ['Mean (deg)'] 

plt.close('all')
sns.set_context("talk", font_scale=1.4)
g = sns.jointplot(x='csd', y='cm',data= threshFits, stat_func=None, alpha=0.01)
g.set_axis_labels('Standard Deviation','Mean')
#plt.title('Normalized Curvature')
g.ax_joint.set_ylim((-1,1.1))
g.ax_joint.set_xlim((0,1))
#g.ax_joint.set_xticklabels([' ','-0.5','0','0.5', '1'])
g.ax_joint.set_xticks([0,0.5,1])


plt.gca().xaxis.set_ticks_position('none') 
plt.gca().yaxis.set_ticks_position('none') 
plt.gca().set_aspect('auto')

sns.set_context("talk", font_scale=1.4)
g = sns.jointplot(x='asd', y='am',data= threshFits, stat_func=None, alpha=0.01)
g.set_axis_labels('Standard Deviation','Mean')
#plt.title('Normalized Curvature')
g.ax_joint.set_ylim((-8,360))
g.ax_joint.set_xlim((0,190))
#g.ax_joint.set_xticklabels([' ','-0.5','0','0.5', '1'])
g.ax_joint.set_xticks([0,90,180])


#plt.subplot(121)
#plt.scatter( np.rad2deg(fits[ :, 2 ]), np.rad2deg(fits[ :, 0 ]), color='r',  linewidth=1, alpha = 0.01)
#plt.title('Angular Position (deg)')
#plt.ylim((-8,360))
#plt.xticks([0,90,180, 270, 360])
#plt.yticks([0,90,180, 270, 360])
#
#




plt.gca().xaxis.set_ticks_position('none') 
plt.gca().yaxis.set_ticks_position('none') 
plt.gca().set_aspect('auto')
plt.tight_layout()

