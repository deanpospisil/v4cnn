# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 12:03:29 2016

@author: dean
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

font = {'size' : 25}
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

        if yticks==None:
            an_axes.yaxis.set_major_locator(mtick.LinearLocator(numticks=nyticks, presets=None))
        else:
            an_axes.set_yticks(yticks)
        if xticks==None:
            an_axes.xaxis.set_major_locator(mtick.LinearLocator(numticks=nxticks, presets=None))
        else:
            an_axes.set_xticks(xticks)
        an_axes.xaxis.set_tick_params(length=0)
        an_axes.yaxis.set_tick_params(length=0)
        an_axes.yaxis.set_major_formatter(mtick.FuncFormatter(tick_format_d))
        an_axes.xaxis.set_major_formatter(mtick.FuncFormatter(tick_format_d))

plt.close('all')
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
    mat = l.loadmat('data/responses/PositionData_Yasmine/pos_'+ str(f)  +'.mat')

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
'''
#acell = cell_resps[0]
#acell = acell - np.mean(acell, 1, keepdims=True)
#u, s, v = np.linalg.svd(acell, full_matrices=False)
##use first princomp
#recell = np.dot(np.expand_dims(u[:,0],1), np.expand_dims(v[0,:]*s[0],0))
##convince myself these are all the same
#np.corrcoef(acell.ravel(), recell.ravel())
#np.dot(acell.ravel(), recell.ravel()) / (np.linalg.norm(acell.ravel())*np.linalg.norm(recell.ravel()))
#(s[0]**2/sum(s**2))**0.5


####################
#plotting v4 SVD TI
fig = plt.figure()
singvals = [np.linalg.svd(acell - np.mean(acell, 1, keepdims=True), compute_uv=0)
            for acell in cell_resps]
#get fraction of variance explained by 1st singular value, make histogram.
best_r = [ ((asingval[0]**2)/(sum(asingval**2))) for asingval in singvals]
plt.hist(best_r, bins=20)
plt.xlabel(r'$R^2$'); plt.ylabel('Cell Count')
plt.ylim([0, 9]); plt.xlim([0,1]); nice_axes(fig.axes)
plt.tight_layout()
plt.savefig(top_dir + 'analysis/figures/images/v4_SI_translation_distribution.png')
'''
'''
font = {'size' : 10}
mpl.rc('font', **font)

####################
#plotting Alexnet SVD TI
fig = plt.figure(figsize=[  7.1625,  10.2875])
if not 'best_r_alex' in locals():
    da = xr.open_dataset(top_dir + 'data/responses/PC370_shapes_0.0_369.0_370_x_-50.0_50.0_101.nc')['resp']
    #ensure fraction of variance that can be explained by single translation
    #or shape is not more than minfracvar
    minfracvar = 0.5
    _ = (da**2).sum('shapes')
    just_one_trans = (_.max('x')/_.sum('x'))>minfracvar
    _ = (da**2).sum('x')
    just_one_shape = (_.max('shapes')/_.sum('shapes'))>minfracvar
    degen = just_one_shape + just_one_trans
    da = da[:,:,-degen]

    da = da - da.mean(['shapes'])
    s = np.linalg.svd(da.values.T, compute_uv=0)
    best_r_alex = np.array([(asingval[0]**2)/(sum(asingval**2)) for asingval in s])

#get layer names
_, index = np.unique(da.coords['layer_label'], return_index=True)
layer_labels = list(da.coords['layer_label'][np.sort(index)].values)

#plot all hist of this metric
nlayers = len(layer_labels)-1
for a in range(nlayers):
    best_r_t = best_r_alex[da.coords['layer'].values==a]
    ax = fig.add_subplot(nlayers+1, 1, a+1)
    ax.hist(best_r_t[-np.isnan(best_r_t)], bins=40)
    ax.axis([0,1, 0, ax.axis()[3]])
    ax.set_ylabel(layer_labels[a],ha='right')
    ax.yaxis.set_label_position("right")


    if a==0:
        ax.set_title('Translation Seperability Index AlexNet Unit Count')
#    elif a==nlayers-1:
#        ax.set_xlabel(r'$R^2$')

ax = fig.add_subplot(nlayers+1, 1, a+2)
ax.hist(best_r, bins=40)
ax.axis([0,1, 0, ax.axis()[3]])
ax.set_ylabel('v4',ha='right')
ax.yaxis.set_label_position("right")

nice_axes(fig.axes)
for i, ax in enumerate(fig.axes):
    if i!=len(fig.axes)-1:
        ax.set_xticks([])
plt.show()
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5)
plt.savefig(top_dir + 'analysis/figures/images/alex_SI_translation_distribution.png')

#investigate high TI values in low layers, fixed it by dividing sum(trans**2).max / tot_var
plt.figure()
r_da = xr.DataArray(best_r_alex, coords={'unit':da.coords['unit'].values})
for name in da.coords['unit'].coords.keys():
    r_da.coords[name] = da.coords['unit'].coords[name]
name='prob'
ind=int(r_da[r_da.coords['layer_label']==name].argmax())
hi_unit=int(r_da[r_da.coords['layer_label']==name][ind].coords['unit'].values)
hi_resp= da.loc[:,:,hi_unit]
(np.linalg.svd(hi_resp.values, compute_uv=False)[0]**2)/(sum(np.linalg.svd(hi_resp.values, compute_uv=False)**2))
plt.imshow(hi_resp.values)
plt.colorbar()

font = {'size' : 19}
mpl.rc('font', **font)
r_ranked_cellinds = list(reversed(np.argsort(best_r)))[0:80:20]
for cellind in r_ranked_cellinds:
    fig = plt.figure(figsize=(18, 6))
#    cellind = np.argsort(best_r)[3]
    thecell = cell_resps[cellind]
    rfpos = np.round(transPos[cellind]/rfDiameter[cellind], 2)
    #lets plot an example cell
    plt.subplot(131)
    mr = np.mean(thecell,1)
    plt.stem(rfpos, mr/max(mr))
    plt.title('V4 cell ' + str(fnum[cellind]) +' TI='+str(np.round(best_r[cellind], 2)))
    plt.ylim([0,1.1])
    plt.ylabel('Normalized Mean Response');plt.xlabel('Fraction RF from center');

    plt.subplot(132)
    cor=np.squeeze(np.corrcoef(thecell)[np.where(transPos[cellind]==0), :])
    plt.stem(rfpos, cor)
    plt.ylabel('Correlation')
    plt.ylim([0,1.1])
    nice_axes(plt.gcf().axes, xticks=transPos[cellind] / rfDiameter[cellind],
              yticks=np.linspace(0, 1, 5))

    plt.subplot(133, aspect='equal')
    pos1 = np.argmax(cor)
    pos2 = np.argsort(cor)[-2]
    plt.scatter(thecell[pos1], thecell[pos2], facecolors='none', edgecolor='blue' )
    plt.xlabel('Stimuli Position: 0');plt.ylabel('Stimuli Position: '+ str(rfpos[pos2]));
    plt.xlim([-1,np.max(thecell)+1]); plt.ylim([-1,np.max(thecell)+1])
    nice_axes([plt.gcf().axes[2],], xticks=np.linspace(0, np.max(thecell), 5),
               yticks=np.linspace(0, np.max(thecell), 5))

    x = min(plt.axis()[0:3:2])
    y = max(plt.axis()[1::2])
    plt.plot([x, y], [x, y], color='black')

    plt.tight_layout()
    plt.savefig(top_dir + 'analysis/figures/images/v4_posinvar_examplecell'+
                            str(fnum[cellind]) + '_' + str(np.round(best_r[cellind], 2))+'.png')

'''
def cell_to_xarray(resp, pos, rfd):
    rfpos = pos/rfd
    thecell = xr.DataArray(resp, dims=['x', 'shapes'], coords=[range(rfpos.shape[0]), range(resp.shape[1])])
    return thecell

def degen_filter(da, minfracvar=0.5):
    _ = (da**2).sum('shapes')
    just_one_trans = (_.max('x')/_.sum('x'))>minfracvar
    _ = (da**2).sum('x')
    just_one_shape = (_.max('shapes')/_.sum('shapes'))>minfracvar
    degen = just_one_shape + just_one_trans
    return degen

def drop_nans(da):
    da=da.isel(x=-da.isnull().all('shapes'))
    da=da.isel(shapes=-da.isnull().all('x'))
    return da

## putting yasmin data into data_array
lsxr = [xr.DataArray(aresp, dims=['x','shapes']) for aresp in cell_resps]
respsxr= xr.concat(xr.align(*lsxr, join='outer'), dim='cells')


####
#start from here will shapesX x X cells
degen = degen_filter(respsxr, minfracvar=0.5)
respsxr.coords['degen'] = ('cells', degen)

##correlation
lscr = [xr.DataArray(cell.T.to_pandas().corr().values, dims=['x1', 'x2'])
        for cell in lsxr]
corrsxr= xr.concat(xr.align(*lscr, join='outer'), dim='cells')

##frac rf calc
rffrac = [ xr.DataArray(pos/rfd, dims=['x']) for pos, rfd in zip(transPos,rfDiameter)]
pos= xr.concat( xr.align(*rffrac, join='outer'), dim='cells')
singvals = [np.linalg.svd(acell.values - np.mean(acell.values, 1, keepdims=True), compute_uv=0)
            for acell in [drop_nans(nancell) for nancell in respsxr]]

##SVD calc
sepi = [((asingval[0]**2)/(sum(asingval**2))) for asingval in singvals]
sepixr = xr.DataArray(sepi, dims='cells')
ti_dat = xr.Dataset({'resp':respsxr, 'pos':pos, 'cor':corrsxr, 'ti':sepixr})
data = 'v4cnn/data/'
ti_dat.to_netcdf(top_dir + 'data/an_results/v4_TI_data.nc')

#
##centered_cor =
##svd_ti =
##
##def xr_cor()
#
#fig = plt.figure(figsize=(18, 6))
##lets plot an example cell
#plt.subplot(131)
#mr = np.mean(thecell,1)
#plt.stem(thecell.coords['x'], thecell.mean('shapes')/thecell.mean('shapes').max())
#plt.title('V4 cell ' + str(fnum[cellind]) +' TI='+str(np.round(best_r[cellind], 2)))
#plt.ylim([0,1.1])
#plt.ylabel('Normalized Mean Response');plt.xlabel('Fraction RF from center');
#
#plt.subplot(132)
#cor=np.squeeze(np.corrcoef(thecell)[np.where(transPos[cellind]==0), :])
#plt.stem(rfpos, cor)
#plt.ylabel('Correlation')
#plt.ylim([0,1.1])
#nice_axes(plt.gcf().axes, xticks=transPos[cellind] / rfDiameter[cellind],
#          yticks=np.linspace(0, 1, 5))
#
#plt.subplot(133, aspect='equal')
#pos1 = np.argmax(cor)
#pos2 = np.argsort(cor)[-2]
#plt.scatter(thecell[pos1], thecell[pos2], facecolors='none', edgecolor='blue' )
#plt.xlabel('Stimuli Position: 0');plt.ylabel('Stimuli Position: '+ str(rfpos[pos2]));
#plt.xlim([-1,np.max(thecell)+1]); plt.ylim([-1,np.max(thecell)+1])
#nice_axes([plt.gcf().axes[2],], xticks=np.linspace(0, np.max(thecell), 5),
#           yticks=np.linspace(0, np.max(thecell), 5))
#
#x = min(plt.axis()[0:3:2])
#y = max(plt.axis()[1::2])
#plt.plot([x, y], [x, y], color='black')
#
#plt.tight_layout()