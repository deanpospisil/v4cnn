# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 15:06:33 2019

@author: deanpospisil
"""

#% Output:
#%   n - 1 by 1 matrix, number of timestamps retrieved
#%   t - n by 4 matrix, timestamp info:
#%       t(:, 1) - timestamp types (1 - neuron, 4 - external event)
#%       t(:, 2) - channel numbers ( =257 for strobed ext events )
#%       t(:, 3) - unit numbers ( strobe value for strobed ext events )
#%       t(:, 4) - timestamps in seconds
#%


import numpy as np
import pandas as pd

f_dir = '/Users/deanpospisil/Downloads/output1.csv'
ps1 = 0.05#time after stim which to collect spikes
ps2 = 0.3#time after stim which to stop collecitng spikes.
channel = 1#channel to collect spikes from

def get_csv_neur_resp(f_dir, channel=1, ps1=0.05, ps2=0.3):
    data = np.genfromtxt(f_dir, delimiter=',')
    data = np.concatenate([np.arange(data.shape[0])[:,np.newaxis], data], 1)#add index column to data
    
    ext_events = data[(data[:,1]==4)*(data[:,2]==257)]#4:external event and 257 strobed event
    #choose indices where 49: stim_id then go one past for each of ext_events where stim_id, then get column with original inds
    stim_id_inds = ext_events[np.arange(len(ext_events))[(ext_events[:,3]==49)].astype(int) + 1][:,0].astype(int)
    stim_ids = data[stim_id_inds, 3] - 200
    
    #get time from start
    t = data[:,-1]
    units = np.unique(data[(data[:,1]==1)][:,3])#1 all neural events, 3 unit numbers, unique to get all unique units
    n_units = len(units)#how many are there for array
    n_spikes = np.zeros((len(stim_id_inds),n_units))#pre-allocate spike array

    for j, unit in enumerate(units):
        spikes = data[(data[:,1]==1)*(data[:,2]==channel)*(data[:,2]==unit)]
        t_spikes = spikes[:,-1]
        for i, stim_id_ind in enumerate(stim_id_inds):
            t_d_spikes = t_spikes-t[stim_id_ind] 
            n_spikes[i,j] = sum((t_d_spikes>ps1)*(t_d_spikes<ps2))
            
    d = pd.DataFrame(n_spikes, columns=units, index=stim_ids)
    
    return d
