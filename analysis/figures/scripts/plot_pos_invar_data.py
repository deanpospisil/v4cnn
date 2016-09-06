# -*- coding: utf-8 -*-
"""
Created on Mon May  9 12:03:39 2016

@author: deanpospisil
"""

import numpy as  np
import scipy.io as  l
import os
import matplotlib.pyplot as plt
import itertools
import sys
import matplotlib.ticker as mtick
import matplotlib as mpl


top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir + 'xarray')
top_dir = top_dir+ 'v4cnn/'
sys.path.append(top_dir)
sys.path.append(top_dir + 'common')

import xarray as xr
import d_misc as dm
plt.close('all')
font = {'size' : 14}
mpl.rc('font', **font)

def tick_format_d(x, pos):
    if x==0:
        return('0')
    else:
        if x>=1:
            return(str(x).split('.')[0])
        else:
            return(np.round(x,2))

def nice_axes(axes, xticks=None, yticks=None, nxticks=5, nyticks=2):
    for i, an_axes in enumerate(axes):

        if yticks is None:
            an_axes.yaxis.set_major_locator(mtick.LinearLocator(numticks=nyticks, presets=None))
        else:
            an_axes.set_yticks(yticks)
        if xticks is None:
            an_axes.xaxis.set_major_locator(mtick.LinearLocator(numticks=nxticks, presets=None))
        else:
            an_axes.set_xticks(xticks)
        an_axes.xaxis.set_tick_params(length=0)
        an_axes.yaxis.set_tick_params(length=0)
        an_axes.yaxis.set_major_formatter(mtick.FuncFormatter(tick_format_d))
        an_axes.xaxis.set_major_formatter(mtick.FuncFormatter(tick_format_d))


fnum = np.array([2, 5, 6, 11, 13, 16, 17, 19, 20, 21, 22, 23, 24, 25, 27, 29, 31,
        33, 34, 37, 39, 43 ,44 ,45, 46, 48, 49, 50, 52, 54, 55, 56, 57, 58, 62,
        66, 67, 68, 69, 70, 71 ,72, 74, 76, 77, 79, 80, 81, 83, 85, 86, 94, 104,
        106, 108, 116, 117, 118, 123, 127,128 ,131, 133, 137, 138, 141, 142, 145,
        152, 153, 154, 155, 156, 166, 170, 175, 190, 191, 193, 194])


ti = xr.open_dataset(top_dir+ 'data/an_results/cnn_TI_data.nc')
cell_ind = int(ti['ti'].argsort()[-33])
thecell = ti.isel(cells=cell_ind)
for key in list(thecell.coords.dims):
    thecell = thecell.dropna(key, how='all')

fig = plt.figure(figsize=(18, 6))
#lets plot an example cell
plt.subplot(131)
plt.stem(thecell['pos'], thecell['resp'].mean('shapes')/thecell['resp'].mean('shapes').max())
plt.title('V4 cell ' + str(fnum[cell_ind]) + ' TI= ' +
            str(np.round(ti['ti'][cell_ind].values, 2)))
plt.ylim([0,1.1])
plt.ylabel('Normalized Mean Response');plt.xlabel('Fraction RF from center');

plt.subplot(132)
cor = thecell['cor'][np.where(thecell['pos']==0)].T
plt.stem(thecell['pos'], thecell['cor'][np.where(thecell['pos']==0)].T)
plt.ylabel('Correlation')
plt.ylim([0,1.1])
nice_axes(plt.gcf().axes, xticks=thecell['pos'].values, yticks=np.linspace(0, 1, 5))

plt.subplot(133, aspect='equal')
pos1 = np.argmax(cor.values)
pos2 = np.argsort(cor.values)[-2]
plt.scatter(thecell['resp'][pos1], thecell['resp'][pos2], facecolors='none', edgecolor='blue' )
plt.xlabel('Stimuli Position: 0'); plt.ylabel('Stimuli Position: '+ str(round(thecell['pos'][pos2].values[0],2)));
plt.xlim([-1, np.max(thecell['resp'].values)+1]);
plt.ylim([-1,np.max(thecell['resp'].values)+1])

nice_axes([plt.gcf().axes[2],], xticks=np.linspace(0, np.max(thecell['resp'].values), 5),
           yticks=np.linspace(0, np.max(thecell['resp'].values), 5))

x = min(plt.axis()[0:3:2])
y = max(plt.axis()[1::2])
plt.plot([x, y], [x, y], color='black')

plt.tight_layout()
