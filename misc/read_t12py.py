# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 15:06:37 2017

@author: deanpospisil
"""
import matplotlib.pyplot as plt 
import numpy as np
import os, sys
top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append(top_dir+ 'v4cnn/')
sys.path.append( top_dir + 'xarray');top_dir = top_dir + 'v4cnn/';
sys.path.append( top_dir + 'common/')
sys.path.append(top_dir +'/nets')
import xarray as xr 
def nan_unrag_col(mat, needed_len=25):
    
    pot_needed_len = max([len(row) for row in mat])
    if needed_len<pot_needed_len:
        needed_len = pot_needed_len
    
    for i, row in enumerate(mat):
        n_nans = needed_len - len(row)
        mat[i] = np.hstack((row, np.array([np.nan]*n_nans)))
    
    return mat

cells = os.listdir(top_dir+ 'data/responses/apc_orig')
cells = [cell for cell in cells if not '.' in cell]
all_cell_data = []
cell_names = []
for cell in cells:
    text_file = open(top_dir + 'data/responses/apc_orig/'+ cell, "r")
    
    keys = open(top_dir + 'data/responses/ShapeList.txt', "r").readlines()
    keys = np.array([list(map(int, key.split())) for key in keys[3:]])
    a = {0:0, 45:1, 90:2, 135:3,180:4, 225:5, 270:6, 315:7}
    keys[:,-1] = np.array([a[key[-1]] for key in keys])
    keys = np.concatenate( ([[0,0,0]], keys), 0) #add blank stim
    #keys[:, :-1] = keys[:,:-1]-1  
    
    clines = text_file.readlines()
    cell_name = clines[0].split()[1]
    start_time = float(clines[1].split()[1])
    duration = float(clines[2].split()[1])
    fs = float(clines[3].split()[1])
    #sampling rate is in seconds but start and duration in milliseconds
    #for times inclusive
    time = np.arange(start_time, start_time+duration+1, (1/fs)*1000)
    trials = clines[6:]
    
    
    trial_info = np.array([list(map(float, clines[trial].split()[1:]))
                    for trial in range(0,np.shape(trials)[0], 2)[3:]])
    spikes = np.array([list(map(float, clines[trial + 1 ].split()[1:]))
                 for trial in range(0,np.shape(trials)[0], 2)[3:]])
    
    #remove shapes unused in study                               
    apc_2001_shapes = (trial_info[:,1]<38)+(trial_info[:,1]>41)
    spikes_01 = spikes[apc_2001_shapes]
    trial_info_01 = trial_info[apc_2001_shapes]
    change_shape_id = (trial_info_01[:,1]>=38)*-4
    trial_info_01[:, 1] = trial_info_01[:, 1] + change_shape_id
    
    shape_index = []
    for trial in trial_info_01:
        shape_id = int(trial[1])
        rot = int(trial[2])
        shape_index.append(keys[(keys[:,1] == shape_id) * (keys[:,2]==rot), 0][0])
    shape_index = np.array(shape_index)
    
    resp_time_course = [[] for key in keys[:,0]]
    t_index_dict = {t:i for i, t in enumerate(time)}
    for a_shape_index, spike_train in zip(shape_index, spikes_01):
        trace = np.zeros(np.shape(time))
        for a_spike_time in spike_train[1:]:
            trace[t_index_dict[a_spike_time]] = 1
            
        resp_time_course[a_shape_index].append(trace)
    resp_time_course = np.array(resp_time_course)

    post_stim_resp = [[] for key in keys[:,0]]
    for i, a_shape in enumerate(resp_time_course):
        post_stim_resp[i] = np.sum(np.array(a_shape)[:, (time>0)*(time<500)], 1)

            
    post_stim_resp_nan = nan_unrag_col(post_stim_resp)
    cell_name = clines[0].split()[1]
    a_cell = xr.DataArray(post_stim_resp_nan, dims=('shapes', 'trials'), 
                 coords=(np.arange(-1,370), range(25)), name=cell_name)        
    start_time = float(clines[1].split()[1])
    duration = float(clines[2].split()[1])
    fs = float(clines[3].split()[1])  
    a_cell.attrs['start'] = start_time
    a_cell.attrs['dur'] = duration
    a_cell.attrs['fs'] = fs
    cell_names.append(cell_name)
    all_cell_data.append(a_cell)

cells_ds = xr.concat(all_cell_data, dim='unit')
cells_ds['w_lab'] = ('unit', cell_names)
cells_ds = cells_ds.to_dataset(name='resp')
cells_ds.to_netcdf(top_dir+ 'data/responses/apc_orig/apc370_with_trials.nc')

