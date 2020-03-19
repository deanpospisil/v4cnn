#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 07:51:01 2019

@author: dean
"""

import os
import shutil
import numpy as np
import time
import matplotlib.pyplot as plt
import xarray as xr
import itertools
import datetime
import pandas as pd
import traceback


def rank_order(b):
    array = np.array(b)
    order = array.argsort()
    ranks = order.argsort()      
    return(ranks)
def r2score(y_true, y_pred):
    r2 = np.corrcoef(y_true,y_pred)[0,1]**2
    if np.isnan(r2):
        r2=0
    return r2 
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def amp_for_width(width):
    rel_amp = np.exp((-(width**2)/2))
    return rel_amp
def width_for_amp(rel_amp):
    width_for_amp = np.sqrt(2*np.log(rel_amp**-1))
    return width_for_amp
def get_csv_neur_resp(the_file, channel=1, ps1=0.05, ps2=0.3, testing=False, 
                      im_inf=None):
    
    #% Output:
    #%   n - 1 by 1 matrix, number of timestamps retrieved
    #%   t - n by 4 matrix, timestamp info:
    #%       t(:, 1) - timestamp types (1 - neuron, 4 - external event)
    #%       t(:, 2) - channel numbers ( =257 for strobed ext events )
    #%       t(:, 3) - unit numbers ( strobe value for strobed ext events )
    #%       t(:, 4) - timestamps in seconds
    #%

    #get the csv file text
    data = the_file
    #add index column to data
    data = np.concatenate([np.arange(data.shape[0])[:,np.newaxis], data], 1)
    #4:external event and 257 strobed event
    ext_events = data[(data[:,1]==4)*(data[:,2]==257)]
    #choose indices where 49: stim_id then go one past for each of ext_events where stim_id, 
    #then get column with original inds
    stim_id_inds = ext_events[np.arange(len(ext_events))[(ext_events[:,3]==49)].astype(int) + 1][:,0].astype(int)
    #subtract off the offset to get real stim_ids
    stim_ids = data[stim_id_inds, 3] - 200
    
    #get time from start
    t = data[:,-1]
    #1 all neural events, 3 unit numbers, unique to get all unique units on a channel
    units = np.unique(data[(data[:,1]==1)*(data[:,2]==channel)][:,3])
    units = list(set(units) or set([1,2]))#make sure there are two units 
    units = np.array(units).astype(int)
    #in case there are no spikes
    
    n_units = len(units)#how many are there for array
    n_spikes = np.zeros((len(stim_id_inds), n_units))#pre-allocate spike array
    for j, unit in enumerate(units):
        #get all timestamps where there was a neural event (1) 
        #and at the correct channels
        #and for the correct unit
        spikes = data[(data[:,1]==1)*(data[:,2]==channel)*(data[:,3]==unit)]
        t_spikes = spikes[:, -1]
        
        #go through each time there was a stimulus onset
        for i, stim_id_ind in enumerate(stim_id_inds):
            t_d_spikes = t_spikes-t[stim_id_ind]#get time relative to that stim 
            #get the correct time chunk
            n_spikes[i,j] = np.sum((t_d_spikes>ps1)*(t_d_spikes<ps2))
            
    d = pd.DataFrame(n_spikes, columns=units, index=stim_ids)

           
    d = d.rename({1:'b', 2:'s'}, axis=1)    
    return d


def get_stim_pos(amp_close_stim=0.65):
    n_circ_samps = 8
    #choose the rel. amplitude of the close shifted stim
    stim_width = width_for_amp(amp_close_stim)#this sets the width of the stim
    #since stim_width=stim_shift for the close stim so they are non overlapping
    amp_far_stim = amp_for_width(2*stim_width)
    sample_r = width_for_amp(np.linspace(1, amp_far_stim, 5))
    sample_r[:-1] = sample_r[1:]
    sample_r[-1] = stim_width*3
    
    phi = np.linspace(0, np.pi*2-np.pi*2/n_circ_samps, n_circ_samps)

    stim_pos = []
    for i, w in zip([0,1]*3, sample_r):
        stim_pos.append(np.array([np.cos(phi[i::2]), np.sin(phi[i::2])])*w)
   
    stim_pos = np.concatenate(stim_pos, 1)
    #stim_pos = np.append(stim_pos, np.array([0, 0]))
    stim_pos = np.concatenate([stim_pos,np.array([0,0])[:,np.newaxis]],1)
    return stim_pos, stim_width

def trans_upos_to_fitpos(stim_pos, rfx, rfy, sig_ecc, sig_rot):
    #SCALE MAT
    #ecc is right rot is up
    scale_mat = np.array([[sig_ecc, 0],
                          [0, sig_rot]])

    #ROT_MAT
    theta = (np.arctan2(rfy, rfx))
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])
    
    stim_pos_trans = np.dot(scale_mat, stim_pos)
    stim_pos_trans = np.dot(rot_mat, stim_pos_trans)
    stim_pos_trans = stim_pos_trans + np.array([rfx,rfy])[...,np.newaxis]
    return stim_pos_trans

def transform_xryr_ecc_rot(xy, mux, muy):
    muxy = np.array([mux, muy])
    ecc_muxy = muxy/np.linalg.norm(muxy)
    rot_muxy = np.array([ecc_muxy[1], -ecc_muxy[0]])#rotate counter clockwise
    rot = np.dot(xy, rot_muxy)
    ecc = np.dot(xy, ecc_muxy)
    return ecc, rot
def clear_folder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

ps1 = 0.05
#ps2 = 0.150 changed march 13 3:30 pm increase spike count
ps2 = 0.3
ppd=39
ndig=1e3

now = datetime.datetime.now()
seed = int(str(now.year) +str(now.month) + str(now.day))
np.random.seed(seed)

stim_dir = '/loc6tb/data/images/'
orig_stim_folder = stim_dir + 'ILSVRC2012_img_val_windowed_softer_cut/' 

imfns = [fn for fn in os.listdir(orig_stim_folder) if '.png' in fn]
imfn_inds = np.argsort([int(im.split('.')[0]) for im in imfns])
imfns = [imfns[ind] for ind in imfn_inds]
blank_id = int(imfns[-1].split('.')[0])

top_dir = '/home/dean/rf_char_exp/'
stim_receive =top_dir + 'sarus_stim_receive_folder/'
stim_record = top_dir + 'sarus_stim_record_folder/'
stim_send = top_dir + 'sarus_stim_send_folder/'
stim_send_exp = top_dir + 'sarus_stim_exp_send_folder/'
init_stim = top_dir + 'init_ims/'
plx_receive = top_dir + 'sarus_plx_receive_folder/'
test_dir = top_dir+ 'test_files/'



plx_fn = input('name of csv file?')
channel = input('what channel?')

plex_csv = np.loadtxt(plx_receive+plx_fn, delimiter=',')
neur_dat = get_csv_neur_resp(plex_csv, channel=channel, ps1=ps1, ps2=ps2)


mux = input('center rf x')
muy = input('center rf y')
sig_rot = input('sig rot')
sig_ecc = input('sig ecc')

stim_record = stim_record + plx_fn.split('.')[0]  

if os.path.isdir(stim_record):
    shutil.rmtree(stim_record)
    
os.mkdir(stim_record)
os.mkdir(stim_record+'/stim_record_exp/')
os.mkdir(stim_record+'/stim_record_ad/')

shutil.copy(plx_receive+plx_fn, stim_record + '/'+plx_fn) 
np.savetxt(stim_record+'/seed.txt', np.array([seed,]), fmt='%u' )
stim_record = stim_record+'/stim_record_ad/'

#%%
#need to get rid of repeats...
m = neur_dat.groupby('id').mean()
orig_id = m.index.values
b = m['b'].values
s = m['s'].values

nstim = 120

sv = s
s_inds = np.argsort(sv)[::-1]
s_sort = sv[s_inds]
num_stim = np.arange(len(s_sort)) + 1
med = np.median(s_sort)
stim_inds = (s_sort>med)*(s_sort>0)+(num_stim<=nstim)
exp_inds = s_inds[stim_inds]
exp_sv = sv[exp_inds]

s = s[exp_inds]
b = b[exp_inds]
orig_id = orig_id[exp_inds]


br = rank_order(b)
sr = rank_order(s)


shibhi = orig_id[np.argsort(br+sr)[::-1]]#background and single high
shiblo = orig_id[np.argsort(sr-br)[::-1]]#single hi background low
slobhi = orig_id[np.argsort(br-sr)[::-1]]#single low (median low) background hi 
sloblo = orig_id[np.argsort(-br-sr)[::-1]]#single low (median low) background lowest ()

top_stim = shibhi[0]

shibhi= list(shibhi)
shiblo = list(shiblo)
slobhi = list(slobhi)
sloblo = list(sloblo)
#if there are not enough stim just wrap around enough times
if len(shibhi)<nstim:
    needed_stim = nstim-len(shibhi)
    times_more = int(np.ceil(needed_stim/len(shibhi))+1)
    shibhi= shibhi*times_more
    shiblo = shiblo*times_more
    slobhi = slobhi*times_more
    sloblo = sloblo*times_more
stim = {'hihi':shibhi,'hilo':shiblo, 'lohi':slobhi, 'lolo':sloblo}
probes = ['hihi', 'hilo'] 
mods = ['hihi', 'hilo', 'lohi', 'lolo']
nclusters=10

#probes go first so they get the strongest responses
probe_inds = np.zeros((nclusters,2))#the inds of each probe from orig images
probe_inds_names = np.zeros((nclusters,2)).astype(str)#keep track of conditions
for i in range(nclusters):#each cluster has two probes
    for j, probe in enumerate(probes): #get the two clusters
        stim, probe_ind = get_stim_ind(probe, stim) 
        probe_inds[i,j] = probe_ind
        probe_inds_names[i,j] = str(i)+'_p_'+probe
        
#now get the modulators
mod_inds = np.zeros((nclusters,len(mods)))
mod_inds_names = np.zeros((nclusters,len(mods))).astype(str)
for i in range(nclusters):#same number of clusters
    for j, mod in enumerate(mods):#4 conditions
        stim, probe_ind = get_stim_ind(mod, stim) 
        mod_inds[i,j] = probe_ind
        mod_inds_names[i,j] = str(i)+'_m_'+mod
        
for j in range(len(mods)):#permute across clusters so no effect of ordering
    mod_inds[:,j] = np.random.permutation(mod_inds[:,j])


#mod_inds is cluster X conds we replicate it it across its final dimensions
#for the replicates across hihi, hilo probes
mod_inds_b = np.broadcast_to(mod_inds[..., np.newaxis], mod_inds.shape + (2,))
mod_inds_names_b = np.broadcast_to(mod_inds_names[..., np.newaxis], 
                                   mod_inds_names.shape +(2,))

#now we add the hihi and hilo probes for a texture condition
mod_inds_b = np.concatenate([mod_inds_b, 
                             probe_inds[:, np.newaxis, :]], 1)
mod_inds_names_b = np.concatenate([mod_inds_names_b, 
                                   probe_inds_names[:, np.newaxis, :]], 1)

#probe_inds is clusters X (hihi. hilo)
#we broadcast it across the mod conditions 
probe_inds_b = np.broadcast_to(probe_inds[:, np.newaxis, :], mod_inds_b.shape)
probe_inds_names_b = np.broadcast_to(probe_inds_names[:, np.newaxis, :], mod_inds_names_b.shape)

# clusters X conds X (probe hihi vs hilo) X (probe,mod)
exp_conds = np.concatenate([probe_inds_b[...,np.newaxis], 
                            mod_inds_b[...,np.newaxis]],-1).T
    
exp_conds_names = np.concatenate([probe_inds_names_b[...,np.newaxis], 
                            mod_inds_names_b[...,np.newaxis]],-1).T
#(probe,mod) 
#X probe_conds (probe hihi vs hilo) 
#X mod conds (same, hihi, hilo, lohi, lolo) 
#X clusters 
    
#we ravel with the first index changing fasters going into 2(img1,img2)Xconds
# so probeXadj pair at a time with cluster changing slowest
exp_conds_rav = exp_conds.reshape(2, np.product(exp_conds.shape[1:]), 
                                  order='F')
exp_conds_names_rav = exp_conds_names.reshape(2, 
                                              np.product(exp_conds.shape[1:]), 
                                              order = 'F')
pool_stim = exp_conds
#so we have the exp_conds stimuli pairs now we want to try them at different position

#get the unqiue stimuli, we need to show these alone so these are the stim conds for ti
ti_stim = np.unique(exp_conds_rav)  

#these are the only requirements mux, muy
rfx, rfy = (mux, muy)
ecc = np.sqrt(rfx**2 + rfy**2)
theta = (np.arctan2(rfy, rfx))


sw_stim = neur_dat['sc'].iloc[-1]#use the last scale
sw_fit = stim_width_fit # use the fit
sw = sw_stim

w_theta_stim = np.arctan2(sw_stim, ecc)#so that the stim in close condition is adjacent
w_theta_fit = np.arctan2(sw_fit, ecc)#so that stim in far condition is as far as rf allows

ecc_theta = np.arctan2(sw_fit*2.5, ecc)

pos_a = pol2cart(ecc, theta-w_theta_stim/2)
pos_b = pol2cart(ecc, theta+w_theta_stim/2)
pos_c = pol2cart(ecc, theta+w_theta_fit)
pos_cd = pol2cart(ecc, theta+w_theta_fit+w_theta_stim)
pos_d = pol2cart(ecc, theta+ecc_theta)

#each of the pool positions
ti_pos = np.array([pos_a, pos_b, pos_c, pos_d,])

#three pairs of positions close, far, 'surround' and each has (x1,y1, x2, y2)
pool_pos  = np.array([[pos_a, pos_b],[pos_a, pos_c], [pos_a, pos_d]])
pool_pos = pool_pos.reshape((pool_pos.shape[0], 4))


ti_inds = np.array(list((itertools.product(*[range(ti_pos.shape[0]),#number of positions
                                      range(ti_stim.shape[0])]))))#number of ti stim

# work from exp_conds 
#nconds X (x1 y1, )
ti_trials = np.concatenate([ti_pos[ti_inds[:,0]],
                            ti_stim[ti_inds[:,1]][:, np.newaxis]], -1)#get the ti trials actual stim and pos

pool_inds = np.array(list((itertools.product(*[range(pool_pos.shape[0]), # number pos pairs
                                      range(pool_stim.shape[1]),# number probe conds
                                      range(pool_stim.shape[2]),# number mod conds
                                      range(pool_stim.shape[3])])))) # number clusters
pool_trials = []
for ind in pool_inds:
    #nconds X ( x1 y1 x2 y2, stim1, stim 2)
    pool_trials.append(np.concatenate([pool_pos[ind[0]], 
                                       pool_stim[:, ind[1], ind[2], ind[3]]]))
pool_trials = np.array(pool_trials)

#pool trials
pool_inds = pd.DataFrame(pool_inds, 
                         columns=['pos_ind', 'probe_ind', 'mod_ind', 'cluster'])
pool_trials = pd.DataFrame(pool_trials,
                           columns = ['x1', 'y1', 'x2', 'y2', 'stim1', 'stim2'])
pool = pd.concat([pool_inds, pool_trials], 1)

#ti single stim trials
ti_inds = pd.DataFrame(ti_inds[:,1], columns=['pos_ind',])
ti_trials = pd.DataFrame(ti_trials, columns=['x1', 'y1', 'stim1'])
ti = pd.concat([ti_inds, ti_trials], 1)

edge_check = pd.DataFrame([[ti['pos_ind'].iloc[-1]+1,
                           pos_cd[0],
                           pos_cd[1],
                           top_stim],], 
                           columns=['pos_ind', 'x1', 'y1', 'stim1'])
ti = pd.concat([ti,edge_check], ignore_index=True)

#blank trials
blank = pd.DataFrame(np.array([[0,0,blank_id],]*5), columns=['x1', 'y1', 'stim1'])



all_exp = pd.concat([ti, pool, blank], 
                    axis=0, 
                    keys=['t', 'p', 'b'], 
                    join='outer', 
                    sort=False)
#this is the base from which we permute for n trials
n_trials = 20
trials = []
nconds = all_exp.shape[0]
for trial in range(n_trials):
    cond_order = np.random.permutation(nconds)
    trials.append(all_exp.iloc[cond_order])
all_exp_trials = pd.concat(trials, axis=0, keys=range(n_trials))

#now this is the exact order and parameters of all trials we just need to
#write it to disk
new_nms = []
trial_inds = []
for i in range(all_exp_trials.shape[0]):
    trial = all_exp_trials.iloc[i]
    if trial.name[1] == 'p':
        new_nm =  ( str(i) 
        + '_' + str(int(trial['x1']*ndig))
        + '_' + str(int(trial['y1']*ndig))
        + '_' + str(int(sw*ndig))
        + '_' + str(int(trial['stim1']))
        + '_.png' )
        new_nms.append(new_nm)
        trial_inds.append(i)
        
        new_nm =  ( str(i) 
        + '_' + str(int(trial['x2']*ndig))
        + '_' + str(int(trial['y2']*ndig))
        + '_' + str(int(sw*ndig))
        + '_' + str(int(trial['stim2']))
        + '_.png' )
        new_nms.append(new_nm)
        trial_inds.append(i)
        
    else:
        new_nm =  ( str(i) 
        + '_' + str(int(trial['x1']*ndig))
        + '_' + str(int(trial['y1']*ndig))
        + '_' + str(int(sw*ndig))
        + '_' + str(int(trial['stim1']))
        + '_.png' )
        new_nms.append(new_nm)
        trial_inds.append(i)


transfer_nms=[]
for new_nm in new_nms:
    #print(new_nm)
    old_nm = new_nm.split('_')[-2] + '.png'
    transfer_nms.append([orig_stim_folder + old_nm, stim_send_exp + new_nm])
    #shutil.copy(orig_stim_folder + old_nm, stim_send_exp + new_nm) 
df = pd.DataFrame(np.array(transfer_nms))

#send file
df.to_csv(stim_send_exp+'transfer_nms.csv', 
          header=False, index=False, index_label=False)    

# make record of trials 
temp = []
stim_record = (top_dir + 'sarus_stim_record_folder/' 
               + plx_fn.split('.')[0] + '/stim_record_exp/')

df.to_csv(stim_record+'transfer_nms.csv', 
          header=False, index=False, index_label=False)      
all_exp_trials.to_csv(stim_record + 'exp_trials.csv', 
          header=False, index=False, index_label=False)      

print('saving record')
#save all images to record
img_exp_dir = stim_record+'exp_imgs/'
os.mkdir(img_exp_dir)
for i in range(df.shape[0]):
    shutil.copy(df.iloc[i][0], 
                img_exp_dir + df.iloc[i][1].split('/')[-1]) 


