# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 15:02:55 2016

@author: dean
"""
import numpy as  np
import scipy.io as  l
import os, sys
#
import matplotlib as mpl
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir + 'xarray')
top_dir = top_dir+ 'v4cnn/'
sys.path.append(top_dir)
sys.path.append(top_dir + 'common')

import xarray as xr
import d_misc as dm

#plt.close('all')
fnum = np.array([2, 5, 6, 11, 13, 16, 17, 19, 20, 21, 22, 23, 24, 25, 27, 29, 31,
        33, 34, 37, 39, 43 ,44 ,45, 46, 48, 49, 50, 52, 54, 55, 56, 57, 58, 62,
        66, 67, 68, 69, 70, 71 ,72, 74, 76, 77, 79, 80, 81, 83, 85, 86, 94, 104,
        106, 108, 116, 117, 118, 123, 127,128 ,131, 133, 137, 138, 141, 142, 145,
        152, 153, 154, 155, 156, 166, 170, 175, 190, 191, 193, 194])

maindir = top_dir
os.chdir( maindir)
resps = []

######
#getting v4 data from matlab
rxl = [];ryl = []
transPos = [];rfDiameter = []
for f in fnum:
    mat = l.loadmat(top_dir + 'data/responses/PositionData_Yasmine/pos_'+ str(f)  +'.mat')

    rxl.append(np.squeeze(mat['data'][0][0][0]))
    ryl.append(np.squeeze(mat['data'][0][0][1]))

    rx = np.double(np.squeeze(mat['data'][0][0][0]))
    ry = np.double(np.squeeze(mat['data'][0][0][1]))
    #print ry
    rfDiameter.append(np.sqrt( rx**2 + ry**2 )*0.625 + 40)

    transPos.append(np.squeeze(mat['data'][0][0][2]))
    resps.append(np.squeeze(mat['data'][0][0][3]))

#lets get svd measurements over cells
#originally: resps cellXposXrotXshape --> converted to cell X pos X unique_shape
cell_resps = [np.dstack(cell).T.reshape(cell.shape[0], np.prod(cell[0].shape))
             for cell in resps]

# putting yasmin data into data_array
lsxr = [xr.DataArray(aresp, dims=['x','shapes']) for aresp in cell_resps]
resp= xr.concat(xr.align(*lsxr, join='outer'), dim='unit')
resp.to_dataset('resp').to_netcdf(top_dir + 'data/an_results/v4_ti_resp.nc')


#apc 109
m = l.loadmat(top_dir + 'data/responses/V4_370PC2001.mat')
v4=m['resp'][0][0]
v4_da = xr.DataArray(v4.T, dims=['shapes', 'unit']).chunk()
#adjustment for repeats [ 14, 15, 16,17, 318, 319, 320, 321]
#a = np.hstack((range(14), range(18,318)))
#a = np.hstack((a, range(322, 370)))
#v4_da = v4_da[a, :]
v4_da = v4_da.to_dataset('resp')
v4_da.to_netcdf(top_dir + 'data/responses/V4_370PC2001.nc')