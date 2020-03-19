# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 11:42:40 2018

@author: deanpospisil
"""
import numpy as np
import matplotlib.pyplot as plt
from neo import io
load_dir = '/Users/deanpospisil/Desktop/modules/v4cnn/data/responses/plex_resp/'
fn = 'dean_test-01.plx'
fn = 'dean_trial_run_aug29-01.plx'
fn = 'dean_trial_run_aug29_2-01.plx'
fn = 'l180910_deans_task_01_01_recut.plx'
#fn = 'dean_trial_run_aug29_3-01.plx'
r = io.PlexonIO(filename = load_dir+fn) 
blks = r.read()
#%%
blks
st = blks[0].segments[0].spiketrains[1].as_array()
ind = 16
ind = -1 
strobe_anno = blks[0].segments[0].events[ind].labels.astype(int)
strobe_time = blks[0].segments[0].events[ind].as_array()
#%%
annos = np.sort(list(set(strobe_anno)))
a = [sum(strobe_anno==num) for num in annos] 



#%%

plt.stem(annos[:20], a[:20])
#%%
rfx = strobe_anno[np.arange(len(strobe_anno))[strobe_anno==43]+1]
rfy = strobe_anno[np.arange(len(strobe_anno))[strobe_anno==44]+1]

## 49 
stimid_inds = np.arange(len(strobe_anno))[strobe_anno==38]-1

stimon_inds = np.arange(len(strobe_anno))[strobe_anno==38]

print('Number stimuli:' + str(len(stimid_inds)))

#%%
isi = np.min(np.diff(strobe_time[stimid_inds]))

stim_id = strobe_anno[stimid_inds]-200
stim_t = strobe_time[stimon_inds]

print('ISI:' + str(isi))
print('tot time minutes :' + str((stim_t[-1]-stim_t[0])/60))

plt.stem(np.diff(stim_t)[:100])
print(np.median(np.diff(stim_t)))

#%%
trials = np.max([sum(stim_id == ind) for ind in range(len(set(stim_id)))])
nstim = len(set(stim_id))

print([sum(stim_id == ind) for ind in range(len(set(stim_id)))])

#%%
fs = 1000
pre_stim = 0
after_stim = .450
nsamps = int((pre_stim+after_stim)*fs)
#trials=5
data = np.zeros((nstim, trials, nsamps))
#data[...] = np.nan
stim_time_coord = np.zeros((nstim, trials))
trial_time = (np.arange(nsamps) - int(pre_stim*fs))/fs
stim_id_unique = np.sort(list(set(stim_id)))


#%%
for i, a_stim in enumerate(stim_id_unique):
    for j, time in enumerate(stim_t[stim_id==a_stim]):
        
        st_chunk = st[(st>(time-pre_stim))*(st<(time+after_stim))]
        st_chunk = st_chunk - (time-pre_stim)
        inds = (st_chunk*fs).astype(int)
        data[i, j, inds] = 1
        stim_time_coord[i,j] = time
#plt.plot(data.mean(1))

#need to get the original numbers of images. Or you could just get responses
        

import xarray as xr
da = xr.DataArray(data, coords={'stim':stim_id_unique, 'trial':list(range(trials)),
                                't':trial_time}, 
                           dims=('stim', 'trial', 't'))

#%%
#plt.figure(figsize=(10,10))
a= da.sum(['t'])


#%%



print(da)
#%%
'''
import seaborn as sns
pt = 0.15
fs = 5000
b = blks[-1].segments[0]
x1 = b.analogsignals[-2].as_array()
x2 = b.analogsignals[-1].as_array()
x2_st= np.array([x2[int(t*fs):int(t*fs)+round(fs*pt)] for t in stim_t])
x1_st= np.array([x1[int(t*fs):int(t*fs)+round(fs*pt)] for t in stim_t])



#plt.scatter(e1, e2);
joint_kws=dict(gridsize=100)
g = sns.jointplot(x2_st.ravel(), x1_st.ravel(), kind="hex", 
                  joint_kws=joint_kws, ylim=(-1.1,-.1), xlim=(-2.3, -2.2))
'''
#%%
attr_list = ['rfx', 'rfy', 'iti', 'stim_time', 'isi', 'numstim','stimdur', 'position',
             'stimWidth', 'mon_ppd']
strtonum = {'start_trial': 10,
                'stop_trial': 11,
                'start_iti': 12,
                'end_iti': 13,
                'eye_start': 14,
                'eye_stop': 15,
                'start_pre_trial': 16,
                'end_pre_trial': 17,
                'start_post_trial': 18,
                'end_post_trial': 19,
                'start_wait_fixation': 20,
                'end_wait_fixation': 21,
                'fixation_occurs': 22,
                'start_wait_bar': 23,
                'end_wait_bar': 24,
                'bar_up': 25,
                'bar_down': 26,
                'test_on': 27,
                'test_off': 28,
                'fix_on': 29,
                'fix_off': 30,
                'fix_acquired': 31,
                'fix_lost': 33,
                'fix_done': 34,
                'start_spont': 35,
                'stop_spont': 36,
                'reward': 37,
                # DMTS constants
                'sample_on': 38,
                'sample_off': 39,
                'targets_on': 40,
                'targets_off': 41,
                'color' : 42,
                'rfx' : 43,
                'rfy' : 44,
                'iti' : 45,
                'stim_time' : 46,
                'isi' : 47,
                'numstim' : 48,
                'stimid'  : 49,
                'rotid'   : 50,
                'stimdur' : 51,
                'occlmode' :52,
                'occl_info' : 53,
                'mask_info' : 54,
                'mask_on' : 55,
                'mask_off' : 56,
                                'position' : 57,
                                'stimWidth' : 58,
                                'stimHeight' : 59,
                                'stimShape' : 60,
                                'perispace' : 61,
                                'occlshape' : 62,
                                'dot_rad' : 63,
                                'line_width' : 64,
                                'gen_mode' : 65,
                                'gen_submode' : 66,
                                'add_extra_isi' : 67,
                                'midground_info' : 68,
                                'foreground_info' : 69,
                                'background_info' : 70,
                                'onset_time' : 71,
                                'second_stimuli' : 72,
                'location_flip_info' : 73,
                'extra' : 74,
                'ambiguous_info' : 75,
                'fix_x' : 76,
                'fix_y' : 77,
                'mon_ppd' : 78,
                'radius' : 80,
                'pause' : 100,
                'unpause' : 101,
                'plexStimIDOffset' : 200,
                                'plexYOffset' : 600,
                                'plexFloatMult' : 1000,
                'plexRotOffset' : 3736}

#%%

da.attrs['t'] = 'seconds'
da.attrs['stim_folder'] = 'ILSVRC2012_wind_for_v4_c345'

inds = np.arange(len(strobe_anno))
for attr in attr_list:
    loc = inds[strobe_anno==strtonum[attr]]+1
    da.attrs[attr] = strobe_anno[loc]