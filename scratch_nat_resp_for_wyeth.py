# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 18:23:41 2019

@author: deanpospisil
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

fn = '/Users/deanpospisil/Desktop/modules/v4cnn/data/responses/'
nm = 'bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(114.0, 114.0, 1)_amp_(100, 255, 2).nc'
#nm = 'bvlc_reference_caffenet_nat_image_resp_371.nc'
da = xr.open_dataset(fn + nm)
da = da['resp'].squeeze()
da = da.isel(amp=0)
mom = da.mean('shapes').groupby('layer_label').mean()
mos = da.std('shapes').groupby('layer_label').mean()
som = da.mean('shapes').groupby('layer_label').std()
sos = da.std('shapes').groupby('layer_label').std()

p = [mom,mos,som,sos]
for ap in p:
    ap = ap[:8]
    labels = ap.coords['layer_label']
    plt.plot(ap)
    plt.xticks(range(len(ap)))
    plt.gca().set_xticklabels(labels.values, rotation=90)
    
plt.legend(['mom', 'mos', 'som','sos'])
plt.title('shapes')

plt.ylabel('raw response')
plt.ylim(-60,100)

#%%
plt.figure()
plt.title('natural images')
fn = '/Users/deanpospisil/Desktop/modules/v4cnn/data/responses/'
nm = 'bvlc_reference_caffenetAPC362_pix_width[32.0]_pos_(114.0, 114.0, 1)_amp_(100, 255, 2).nc'
nm = 'bvlc_reference_caffenet_nat_image_resp_371.nc'
da = xr.open_dataset(fn + nm)
da = da['resp'].squeeze()

mom = da.mean('shapes').groupby('layer_label').mean()
mos = da.std('shapes').groupby('layer_label').mean()
som = da.mean('shapes').groupby('layer_label').std()
sos = da.std('shapes').groupby('layer_label').std()

p = [mom,mos,som,sos]
for ap in p:
    ap = ap[:8]
    labels = ap.coords['layer_label']
    plt.plot(ap)
    plt.xticks(range(len(ap)))
    plt.gca().set_xticklabels(labels.values, rotation=90)
    
plt.legend(['mom', 'mos', 'som','sos'])

plt.ylim(-60,100)
