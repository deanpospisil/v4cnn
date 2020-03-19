#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 13:51:18 2019

@author: dean
"""
import os
import shutil
import numpy as np
import time
import matplotlib.pyplot as plt
import xarray as xr
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from scipy.stats import multivariate_normal
from scipy import optimize as op
import itertools
import datetime
import pandas as pd
import traceback

now = datetime.datetime.now()
seed = int(str(now.year) +str(now.month) + str(now.day))
np.random.seed(seed)

def get_stim_ind(stim_nm, stim):
    #grabs item from one stim list then removes from all lists
    #so always get unique ind
    ind = stim[stim_nm][0]#get item
    for key in stim.keys():#for each list
        #return same list except that item
        loc = np.where(np.array(stim['hilo'])==ind)[0]
        if len(loc)>0:
            stim[key] = [val for i, val in enumerate(stim[key]) if not i==loc[0]]
    return stim, ind#give updated dict and the ind

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

    if testing:
        stim_id = im_inf['id'].values
        u_stim = np.sort(np.unique(stim_id).astype(int))
        stim_id_resp_ind = np.array([np.where(an_id==u_stim)[0][0] 
                            for an_id in stim_id]).astype(int)
        x=[]
        x.append(np.concatenate([im_inf[['x','y',]].values, 
                            stim_id_resp_ind[:,np.newaxis]], 1))
        x.append(0)
        avals = len(np.unique(x[0][:,-1]))*[10,]

        mux = 0
        muy = 4
        sig = 1
        args = np.array([mux,muy, sig] + avals)
        resp = gaussian2_circ(x, *args)
        fake_dat = np.random.poisson(resp, size=(2, len(im_inf.index))).T
        d = pd.DataFrame(fake_dat, index=im_inf.index, columns=range(2))
        d = d.rename({0:'b', 1:'s'}, axis=1)   
    
    else:
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
        d = d.groupby(level=0).mean()#to account for repeat stimuli.
    return d


#%%
#ps1 = 0.05
#ps2 = 0.150  
#f_dir = '/home/dean/rf_char_exp/sarus_plx_record_folder/output2.csv'
#data = np.genfromtxt(f_dir, delimiter=',')
#data = np.concatenate([np.arange(data.shape[0])[:,np.newaxis], data], 1)#add index column to data
#
#ext_events = data[(data[:,1]==4)*(data[:,2]==257)]#4:external event and 257 strobed event
##choose indices where 49: stim_id then go one past for each of ext_events where stim_id, then get column with original inds
#stim_id_inds = ext_events[np.arange(len(ext_events))[(ext_events[:,3]==49)].astype(int) + 1][:,0].astype(int)
#stim_ids = data[stim_id_inds, 3] - 200
#
##get time from start
#t = data[:,-1]
##1 all neural events, 3 unit numbers, unique to get all unique units
#units = np.unique(data[(data[:,1]==1)][:,3])
#n_units = len(units)#how many are there for array
#n_spikes = np.zeros((len(stim_id_inds), n_units))#pre-allocate spike array
#
#for j, unit in enumerate(units):
#    spikes = data[(data[:,1]==1)*(data[:,2]==channel)*(data[:,3]==unit)]
#    t_spikes = spikes[:,-1]
#    for i, stim_id_ind in enumerate(stim_id_inds):
#        t_d_spikes = t_spikes-t[stim_id_ind] 
#        n_spikes[i,j] = np.sum((t_d_spikes>ps1)*(t_d_spikes<ps2))
#        
#d = pd.DataFrame(n_spikes, columns=units, index=stim_ids)
    

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

def gaussian2_circ(x, *args): 
    bl = x[1]
    pos = x[0][:,:2]
    stim_id_resp_ind = x[0][:,2].astype(int)
    
    mux = args[0]
    muy = args[1]
    sig = args[2]
    avals = np.array(args[3:])

    cov = np.array([[sig**2, 0], 
          [0, sig**2]])
    resp = multivariate_normal([mux, muy], cov).pdf(pos)
    height = multivariate_normal([mux, muy], cov).pdf(np.array([mux, muy]))
    resp= resp/height
    resp = resp*avals[stim_id_resp_ind] + bl

    return resp

def fit_retinal_circ(d, mux, muy,bl=0):
    
    sig = 1+(mux**2+muy**2)**0.5

    mu_amp = d.groupby('id').max()['resp']
    na = len(mu_amp) 
    p0 = [ mux,  muy, sig, ] + [a for a in mu_amp]
    bounds = (([-np.inf,   -np.inf,     0,  ] + [-np.inf for a in range(na)]), 
              ([np.inf,    np.inf,   np.inf,] + [np.inf for a in range(na)]))
    
    bounds = (([mux-sig/10.,   muy-sig/10.,     sig/1.5,  ] + [a/1.5 for a in mu_amp]), 
          ([mux+sig/10,    muy+sig/10.,   sig*1.5,] + [a*1.5 for a in mu_amp]))


    
    stim_id = d['id'].values
    u_stim = np.sort(np.unique(stim_id).astype(int))
    stim_id_resp_ind = np.array([np.where(an_id==u_stim)[0][0] 
                            for an_id in stim_id]).astype(int)
    d['stim_id_resp_ind'] = stim_id_resp_ind
    x = []
    x.append(d[['x', 'y', 'stim_id_resp_ind']].values)
    x.append(bl)
    y = d['resp'].values
    try:
        popt, pcov = op.curve_fit(gaussian2_circ, x, 
                          y, 
                          p0=p0,
                          bounds = bounds)
                          #ftol=len(y)*np.var(y)*1e-6)

    except:
        popt = p0
        pcov = np.zeros((len(p0),len(p0)))

    se = np.diag(pcov)**0.5
    se[3:] = se[3:]
    popt[3:] = np.array(popt[3:])
    
    index = ['mux', 'muy', 'sig' ] +['amp_' + str(int(a)) for a in mu_amp.index]
    fit = pd.DataFrame(np.array([popt, se, p0]).T, columns=['mean', 'se', 'p0'], index=index)
    x = []
    x.append(d[['x', 'y','stim_id_resp_ind']].values)
    x.append(bl)
    d['_fit_circ'] = gaussian2_circ(x, *popt)
    return d,fit



def axis_mu_elipse(x, *args):    
    bl = x[1]
    pos = x[0][:, :2]
    stim_id_resp_ind = x[0][:,2].astype(int)
    
    mu_ecc = args[0]
    sigx = args[1]
    sigy = args[2]
    avals = np.array(args[3:])

    cov = np.array([[sigx**2, 0], 
                   [0,   sigy**2]])
    resp = multivariate_normal([mu_ecc, 0], cov).pdf(pos)
    
    height = multivariate_normal([mu_ecc, 0], cov).pdf(np.array([mu_ecc, 0]))

    resp= resp/height

    resp = resp*avals[stim_id_resp_ind] + bl
    
    return resp

def fit_axis_mu_elipse(d, mux, muy, sig_ecc, sig_rot, bl=0):
    
    xy = d[['x', 'y']].values
    ecc, rot = transform_xryr_ecc_rot(xy, mux, muy)
    #create new coordinates along ecc and to axis.
    d['ecc_proj'] = ecc
    d['rot_proj'] = rot
    
    ecc_mu = (mux**2+muy**2)**0.5
    
    mu_amp = d.groupby('id').max()['resp']

    na = len(mu_amp) 
    p0 = [ecc_mu, sig_ecc*1.5, sig_rot*1.5, ] + [a for a in mu_amp]
    bounds = (([-np.inf,         0,   0,    ] + [-np.inf for a in range(na)]), 
              ([np.inf,     np.inf, np.inf,] + [np.inf for a in range(na)]))
    
    bounds = (([ecc_mu/1.5,         sig_ecc/1.5,   sig_rot/1.5,    ] + [a/2. for a in mu_amp]), 
          ([ecc_mu*1.5,     sig_ecc*1.5, sig_rot*1.5,] + [a for a in mu_amp]))

    stim_id = d['id'].values
    u_stim = np.sort(np.unique(stim_id).astype(int))
    stim_id_resp_ind = np.array([np.where(an_id==u_stim)[0][0] 
                            for an_id in stim_id]).astype(int)

    d['stim_id_resp_ind'] = stim_id_resp_ind
    x = []
    x.append(d[['ecc_proj', 'rot_proj', 'stim_id_resp_ind']].values)
    x.append(bl)
    y = d['resp'].values
    try:
        popt, pcov = op.curve_fit(axis_mu_elipse, x, 
                          y, 
                          p0=p0,
                          bounds = bounds)
                          #ftol=len(y)*np.var(y)*1e-10)
    except:
        print('fit did not converge, using priors')
        popt = p0
        pcov = np.zeros((len(p0),len(p0)))
        
    #get amps back
    popt[3:] = np.array(popt[3:])
    p0[3:] = np.array(p0[3:])

    
    se = np.diag(pcov)**0.5
    se[3:] = se[3:]
    index = ['mu_ecc', 'sig_ecc', 'sig_rot' ] + ['amp_' + str(int(a)) for a in mu_amp.index]
    fit = pd.DataFrame(np.array([popt, se, p0]).T, columns=['mean', 'se', 'p0'], index=index)
    d['_fit_ax_mu'] = axis_mu_elipse(x, *popt)
    return d, fit



def fit_rf(d, unit, mux, muy, sig_ecc, sig_rot, bl=0):
    d = d[['x','y', 'id', unit]]
    d = d.rename({unit:'resp'}, axis=1)
    
    d, fit_ac = fit_retinal_circ(d, mux, muy, bl)
    
    fit_ac['p0_var'] = np.abs(fit_ac['p0'])
    fit_ac['p0_var'][:2] = (fit_ac['p0'][:2]**2).sum()**0.5
    fit_ac['b_mean'] = (fit_ac['mean']*(fit_ac['p0_var']/(fit_ac['se']**2+fit_ac['p0_var'])) 
                            + fit_ac['p0']*(fit_ac['se']**2/(fit_ac['se']**2+fit_ac['p0_var'])))
    fit_ac = fit_ac[['mean', 'b_mean', 'p0', 'p0_var', 'se']]
    
    mux = fit_ac['b_mean'].loc['mux']
    muy = fit_ac['b_mean'].loc['muy']
    
    d, fit_ae = fit_axis_mu_elipse(d, mux, muy, sig_ecc, sig_rot, bl)
    
    fit_ae['p0_var'] = np.abs(fit_ae['p0'])
    
    fit_ae['b_mean'] = (fit_ae['mean']*(fit_ae['p0_var']/(fit_ae['se']**2+fit_ae['p0_var'])) 
                            + fit_ae['p0']*(fit_ae['se']**2/(fit_ae['se']**2+fit_ae['p0_var'])))
    fit_ae = fit_ae[['mean', 'b_mean', 'p0', 'p0_var', 'se']]
    
    fit = pd.concat([fit_ac.iloc[:2], fit_ae])
    return fit, d


def clear_folder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

def move_folder_pngs(from_folder, to_folder):
    for the_file in os.listdir(from_folder):
        file_path = os.path.join(from_folder, the_file)
        try:
            if os.path.isfile(file_path):
                shutil.copy(file_path, to_folder)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

#%%

r2score_met = make_scorer(r2score)
n_imgs = 200
ps1 = 0.05
#ps2 = 0.150 changed march 13 3:30 pm increase spike count
ps2 = 0.3
ppd=39


net_resp_nm = ('/home/dean/rf_char_exp/net_stim_resp/full_vg11_layer_11ILSVRC2012_img_val_windowed_softer_cut.nc')


resp = xr.open_dataarray(net_resp_nm).chunk({'stim':200000})

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

if os.path.exists(plx_receive+'rf_info.txt'):
    os.remove(plx_receive+'rf_info.txt')

clear_folder(plx_receive)
clear_folder(stim_receive)
clear_folder(stim_send)
clear_folder(stim_send_exp)


#send message that adaptive fitting is starting
np.savetxt(top_dir+'plex_msg.txt', np.array([1,]), fmt='%u')  

#these variables are just for testing
testing = False
if testing:
    nrounds = 3
    channel = 1
    n_init_ad_rounds = 0
else:
    nrounds = input('How many rounds?')
    channel = input('What channel?')
    n_init_ad_rounds = input('How many initial rounds to be adaptive?')

init = True
ndig = 1e3


try:
        #0 is run purely adaptive, 1 is run adaptive and pos, and 3 only pos
    for fit_iter in range(nrounds):
        #wait for a record folder
        np.savetxt(top_dir + 'load_try_again.txt', np.array([1,]), fmt='%u')       
        if testing:
    
            #move appended plx into plx receive or just token plx for now
            shutil.copy(test_dir+'output2.csv', plx_receive+'output2.csv')
                    #read it 
            mux = 4
            muy = 4
            sig_ecc = 1
            sig_rot = 1
            rf_info = np.array([mux,muy, sig_ecc, sig_rot])
            np.savetxt(plx_receive + 'rf_info.txt', rf_info)
                
            
        if init:
            #wait for stim txt 
            no_txt = True
            print('waiting for rf txt')
            while no_txt:
                time.sleep(1)
                txt_fns = [f for f in os.listdir(plx_receive) if 'info.txt' in f]
                if len(txt_fns)>0:
                    #rf info will give us estimates of these params
                    (mux, muy, sig_ecc, sig_rot) = np.loadtxt(plx_receive + 'rf_info.txt')
                    no_txt=False
            mu_eccp = np.sqrt((mux**2+muy**2))
            muxp=mux
            muyp=muy
            sig_eccp=sig_ecc
            sig_rotp=sig_rot
            #params for stim
            stim_pos, stim_u_width = get_stim_pos()
            stim_pos = trans_upos_to_fitpos(stim_pos, mux, muy, sig_ecc, sig_rot)
            stim_width = stim_u_width*sig_rot
            stim_widthp = stim_width
            #go to init_stim folder and transfer files but renamed with
            #n_x_y_sc_orig
            init_im_nms = [f for f in os.listdir(init_stim) if '.png' in f]
            init_im_inds = [int(f.split('_')[0]) for f in init_im_nms]
            sort_inds = np.argsort(init_im_inds)
            init_im_nms = [init_im_nms[i] for i in sort_inds]
        
    
            new_stim_names = [f.split('.')[0].split('_')[0] 
                    +'_' + str(int(mux*ndig)) 
                    + '_' +str(int(muy*ndig))
                    + '_' +str(int(stim_width*ndig)) 
                    + '_' + f.split('.')[0].split('_')[1]+'_.png' for f in init_im_nms]
            #move these images into the send dir and then plexon will grab them
            #[shutil.copy(init_stim + fn, stim_send + new_fn) 
            #for fn, new_fn in zip(init_im_nms, new_stim_names)]
            #now instead we just move file names into stim send dir
            transfer_nms = np.array([[init_stim + fn, stim_send + new_fn] 
            for fn, new_fn in zip(init_im_nms, new_stim_names)])
            df = pd.DataFrame(np.array(transfer_nms))
            df.to_csv(stim_send+'transfer_nms.csv', 
                      header=False, index=False, index_label=False)
            
            #we are now down with init.
            init=False
            
        if testing:
            df[1].to_csv(stim_receive+'transfer_nms.txt', 
              header=False, index=False, index_label=False)
            #move_folder_pngs(stim_send, stim_receive) 
            
             #now these images will be grabbed and shown on plex
            #here we will wait for the first responses to come back
        print('waiting for recording')
        no_plex=True
        plex_load_iter = 0
        plx_fns = []
        while no_plex:
            time.sleep(1)
            
            plx_fns = [f for f in os.listdir(plx_receive) if '.csv' in f]
            if len(plx_fns)>0:
                time.sleep(1)
                plx_fn = plx_fns[0]
                print('csv file is here')
                try:#load text will throw error if file corrupted and we will keep trying
                    #we also want to make sure file isn't changing
                    #
                    plex_load_iter+=1
                    file_loading=True
                    while file_loading:
                        
                        prev = np.loadtxt(plx_receive+plx_fn, delimiter=',')
                        time.sleep(1)#wait a second to see if file changes
                        plex_csv = np.loadtxt(plx_receive+plx_fn, delimiter=',')
                        
                        if np.shape(plex_csv)==np.shape(prev):
                            file_loading=False
                        else:
                            print('file has not loaded yet')                     
                        
                    if len(plex_csv)>4:
                        print('file is full')
                        no_plex=False
                        print(plex_csv.shape)
                        np.savetxt(top_dir + 'load_try_again.txt', np.array([0,]), fmt='%u')   
                        time.sleep(1)
                except Exception as e:
                    print(e)
                    'corrupted csv file, waiting'
    
            if len(plx_fns)>1:
                'warning more than one csv file'
        plx_fn = plx_fns[0]    
        clear_folder(stim_send)
    
        print('got csv file ' + plx_fn)
        
        #wait for the stims to arrive
        no_stims =True
        while no_stims:
            time.sleep(1)
            stim_fn = [f for f in os.listdir(stim_receive) if '.txt' in f]
            if len(stim_fn)>0:
                no_stims=False
        if testing:   
            df = pd.read_csv('/home/dean/rf_char_exp/sarus_stim_receive_folder/transfer_nms.txt',
                header=None, index_col=False, ).values
            df = [fn[0] for fn in df]
        else:
            df = pd.read_csv('/home/dean/rf_char_exp/sarus_stim_receive_folder/transfer_nms.txt',
            header=None, index_col=False, ).values[0]
            
        new_nms_receive = [fn.split('/')[-1] for fn in df]
        old_nms_receive = [fn.split('_')[-2] +'.png' for fn in new_nms_receive]
        
        print('got ' + str(len(old_nms_receive)) + ' images')
        #get stim info and responses
        im_inds = np.array([(np.array(fn.split('_')[:-1])).astype(int) 
                    for fn in new_nms_receive])
        sort_inds = np.argsort(im_inds[:,0])
        im_inds = im_inds[sort_inds]
        
    
    
        if fit_iter==0:#if the first iter just start the data frame with stim info
            im_inf = pd.DataFrame(im_inds[:,1:], index=im_inds[:,0], 
                              columns=[ 'x', 'y', 'sc', 'id'])
            im_inf[['x','y', 'sc']] = im_inf[['x','y', 'sc']]/ndig
            if testing:
                d = get_csv_neur_resp(plx_receive+plx_fn, 
                                      channel=channel, ps1=ps1, ps2=ps2, testing=True,
                                      im_inf = im_inf)
            else:
                d = get_csv_neur_resp(plex_csv, 
                              channel=channel, ps1=ps1, ps2=ps2, testing=False)
        else:
            if testing:
                im_inf_new = pd.DataFrame(im_inds[:,1:], 
                                          index=im_inds[:, 0] + 
                                          im_inf.index[-1] + 1, 
                              columns=[ 'x', 'y', 'sc', 'id'])
            else:
                im_inf_new = pd.DataFrame(im_inds[:,1:], index=im_inds[:,0], 
                              columns=[ 'x', 'y', 'sc', 'id'])
            im_inf_new[['x','y', 'sc']] = im_inf_new[['x','y', 'sc']]/ndig
            im_inf = pd.concat([im_inf, im_inf_new])
            
            if testing:
                d_new = get_csv_neur_resp(plx_receive+plx_fn, 
                                          channel=channel, ps1=ps1, ps2=ps2, 
                                          testing=True,
                                          im_inf = im_inf_new)*(fit_iter+1)
                d = pd.concat([d, d_new])
            else:
                d_new = get_csv_neur_resp(plex_csv, 
                              channel=channel, ps1=ps1, ps2=ps2, testing=False)
            
                d = d_new
            print('read csv with ' + str(len(d_new))  + ' responses')
        neur_dat_no_drop =  pd.concat([im_inf, d], axis=1)
        neur_dat = pd.concat([im_inf, d], axis=1).dropna()
        print('last rounds spike number stats')
        print(neur_dat[['s','b']].describe(percentiles=[0.5,]))
        #make record of stim in new folder and set up folders for saving 
        last_stim_id = int(neur_dat.index.max())
        if fit_iter==0:
            stim_record = stim_record + plx_fn.split('.')[0]  
        
            if os.path.isdir(stim_record):
                shutil.rmtree(stim_record)
            os.mkdir(stim_record)
            os.mkdir(stim_record+'/stim_record_exp/')
            os.mkdir(stim_record+'/stim_record_ad/')
            
            shutil.copy(plx_receive+ 'rf_info.txt', stim_record + '/rf_info.txt') 
            np.savetxt(stim_record+ '/rf_info.txt', (mux, muy, sig_ecc, sig_rot))
            np.savetxt(stim_record+'/seed.txt', np.array([seed,]), fmt='%u' )
            stim_record = stim_record+'/stim_record_ad/'
            
        cur_stim_record = stim_record  + str(last_stim_id) + '_' + str(fit_iter)
        os.mkdir(cur_stim_record)
        if testing:
            neur_dat = neur_dat.rename({0:1, 1:2}, axis=1)
        neur_dat.to_csv(cur_stim_record+'/neur_dat.csv')
    
        #fit model to ims
        dist = ((neur_dat[['x', 'y']] - np.array([mux, muy]))**2).sum(1)**0.5
        av_scale =  neur_dat[neur_dat['id']!=blank_id].mean()['sc']
        neur_dat_c = neur_dat[dist<av_scale/2.]
    
        resp_sub = resp.sel(stim=neur_dat_c['id'].values.astype(int)).values
    
        #now we need to fit 
        reg = RidgeCV(fit_intercept=True, normalize=True)
        ms = []
    
        for i in ['b', 's']:   
            if testing:
                neur = neur_dat_c[i].values.squeeze() + np.abs(resp_sub[:,1].squeeze()*10)
            else:
                neur = neur_dat_c[i].values.squeeze()
            scores = cross_validate(reg, resp_sub, neur**0.5, scoring=r2score_met,
                                            cv=5, return_train_score=False, 
                                            n_jobs=1);
    
            m = np.mean(scores['test_score'])
            v = np.var(scores['test_score'])
            print(('unit ' + str(i) + ' explained variance ' + str(np.round(m,2)) + 
                  ' 95% ci= [' + str(np.round(m - 2*(v/5)**0.5, 2)) + ', '+ 
                  str(np.round(m + 2*(v/5)**0.5, 2)) + ' ] '))
            ms.append(m)
        #adaptive stimuli
        if m-v**0.5<0.03:#if the single unit is not fit well do combo 
            #based on weighting between how well background and single fit.
            sbr = (ms[0] + 0.01)/(ms[1] + ms[0] + 0.01)
            print(np.round(sbr,2))
            
            nb_imgs = int(n_imgs*10*sbr)
            ns_imgs = n_imgs*10 - nb_imgs
            reg.fit(resp_sub, neur_dat_c['s'])
            inds_s = reg.predict(resp).squeeze().argsort()[::-1][:ns_imgs]
            reg.fit(resp_sub, neur_dat_c['b'])
            inds_b = reg.predict(resp).squeeze().argsort()[::-1][:nb_imgs]
            
            inds = []
            for i in range(n_imgs*10):
                if i<len(inds_s):
                    inds.append(inds_s[i])
                if i<len(inds_b):
                    inds.append(inds_b[i])
            inds=np.array(inds)
            _, index = np.unique(inds,  return_index=True)
            best_stim_inds = inds[np.sort([index])].squeeze()
        else:
            reg.fit(resp_sub, neur_dat_c['s'])
            best_stim_inds = reg.predict(resp).squeeze().argsort()[::-1][:n_imgs].squeeze()
        shown_inds = neur_dat['id'].unique().astype(int)
        best_stim_inds_full = best_stim_inds
        best_stim_inds = [ind for ind in best_stim_inds if ind in shown_inds][:(n_imgs+2)]
        if len(best_stim_inds)<200:
            best_stim_inds = best_stim_inds_full[:200]
        np.save(cur_stim_record + '/reg_coeffs.npy', reg.coef_)
        
        #fitting RF for stim pos
        #only use stimuli which has been selected for rf mapping
        #more than 15 different positions
        #get stim_id where different positions were presented
        temp = neur_dat.copy(deep=True)
        temp['posid'] = temp['x'].astype(str) + temp['y'].astype(str)
        moved_stim = temp.groupby('id')[['posid']].nunique()
        moved_stim_id = moved_stim[moved_stim>15].dropna().index.values
        rf_pos_stim_inds = neur_dat['id'].isin(moved_stim_id)
        neur_dat_pos = neur_dat.loc[rf_pos_stim_inds].reset_index(drop=True)
        neur_dat_pos = neur_dat_pos[neur_dat_pos['id'] != blank_id]
        #neur_dat_pos = neur_dat_pos.set_index('id', drop=False, append=True).swaplevel(0,1)
        new_stim_names_pos=[]
        if len(neur_dat_pos)>15 and fit_iter>=n_init_ad_rounds:
            cv = neur_dat_pos.groupby('id').var()/neur_dat_pos.groupby('id').mean()
            cv[cv.isna()] = 0
            print('the cv')
            print(cv.loc[:, ['s','b']])
            if cv.loc[:, 's'].max(0)>3:
                rf_fit_cell = 's'
            else:
                rf_fit_cell = 'b'
      
            try:
                bl = neur_dat[neur_dat['id']==blank_id][rf_fit_cell].mean()
            except:
                bl=0
            fit, _ = fit_rf(neur_dat_pos, rf_fit_cell, #maybe fit to background to start?
                            mux=muxp, muy=muyp, 
                            sig_ecc=sig_eccp, sig_rot=sig_rotp, bl=bl)
            
            (mux, muy, mu_ecc, sig_ecc, sig_rot)  = fit['b_mean'][:5]
            #params for stim
            stim_pos, stim_u_width = get_stim_pos()
            stim_pos = trans_upos_to_fitpos(stim_pos, mux, muy, sig_ecc, sig_rot)
            stim_pos = stim_pos[:, np.random.permutation(stim_pos.shape[1])]
            stim_width = stim_u_width*sig_rot
            stim_width_fit = stim_width
            stim_width = stim_widthp#use prior
            
            fit.to_csv(cur_stim_record+'/rf_fit.csv')
            print(fit[:5]*ppd)  
            print(fit[5:]) 
                    #RF mapping stimuli
        if fit_iter>=n_init_ad_rounds:
            temp = neur_dat[neur_dat['id']!=blank_id]
            temp_ind = -temp['id'].isin(neur_dat_pos['id'].unique())
            temp = temp[temp_ind]
            if neur_dat['s'].max()>3:#want at least 3 spikes from the stimuli
                #or else use background
                pos_im = temp.iloc[temp['s'].argsort()[::-1].values]['id'].iloc[0]
            else:
                pos_im = temp.iloc[temp['b'].argsort()[::-1].values]['id'].iloc[0]
        
            print('first trans stim')
            new_stim_names_pos = [
             '_' + str(int(x*ndig))
             + '_' + str(int(y*ndig))
             + '_' + str(int(stim_width*ndig))
             + '_' + str(int(pos_im))
             + '_.png'
             for x, y in stim_pos.T]
            
        #these are the centered which are based on fit stimuli
        new_stim_names = ['_' + str(int(mux*ndig))
                         + '_' + str(int(muy*ndig))
                         + '_' + str(int(stim_width*ndig))
                         + '_' + str(int(ind))
                         + '_.png'
                         for i, ind in enumerate(best_stim_inds)]
        
        #after this stim_pos will be generated by fitting to data. 
        all_new_stim_names = []
        all_new_stim_names.append('_'+ str(int(mux*ndig))
                                 +'_'+ str(int(muy*ndig))
                                 +'_'+ str(int(stim_width*ndig))
                                 +'_'+ imfns[-1].split('.')[0] + '_.png')
        
        for i in range(n_imgs):
            if i<len(new_stim_names_pos) and fit_iter>=n_init_ad_rounds:
                all_new_stim_names.append(new_stim_names_pos[i])
            if i<len(new_stim_names):
                all_new_stim_names.append(new_stim_names[i])
            
        all_new_stim_names = [str(i) + nm for i, nm 
                              in enumerate(all_new_stim_names)]
        new_stim = [nm.split('_')[-2] + '.png' for nm in all_new_stim_names]
        #copy to stim send folder
        #_ = [shutil.copy(orig_stim_folder + fn, stim_send + new_fn) 
        #    for fn, new_fn in zip(new_stim, all_new_stim_names)]
        transfer_nms = [[orig_stim_folder + fn, stim_send + new_fn]
            for fn, new_fn in zip(new_stim, all_new_stim_names)]
        df = pd.DataFrame(np.array(transfer_nms))
        df.to_csv(stim_send+'transfer_nms.csv', header=False, index=False, index_label=False)
        print('new stim names sent')
        #move plx receive to stim record assosciated
        shutil.move(plx_receive+plx_fn, cur_stim_record+'/'+plx_fn) 
        #move recieve stim to record stim change this to read names then copying CHANGE
        [shutil.copy(orig_stim_folder+old, cur_stim_record + '/' + new) 
                                for old, new in zip(old_nms_receive, new_nms_receive)]
        
        #now clean these folders
        clear_folder(plx_receive)
        clear_folder(stim_receive)
    
    
    print('done with adaptive portion of experiment')
    print(fit.iloc[:5]*ppd)  
    print(fit.iloc[5:]) 
    ans= input('Would you like to use the RF fit (0) or the experimenters initial guess (1)?')
    if ans==1:
        print('using rf fit')
        stim_width_fit = stim_widthp
        mu_ecc = mu_eccp
        mux = muxp
        muy = muyp
        
    else:
        print(' rf fit')

except Exception as e:
    print(e)
    #now clean these folders
    #if there is a problem 
    #clear_folder(plx_receive)
    #clear_folder(stim_receive)
    
    print('error moving on to experiment')
    print('using rf fit')
    stim_width_fit = stim_widthp
    mu_ecc = mu_eccp
    mux = muxp
    muy = muyp
#%% send break signal
#send message that adaptive fitting is over
np.savetxt(top_dir+'plex_msg.txt', np.array([0,]), fmt='%u')   
print(neur_dat_c[['b','s']].sum()) 
#%%
#need to get rid of repeats...
m = neur_dat_c.groupby('id').mean()
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

##condsX(img1, img2)X (s,b)
##just testingtransfer_nms
#resps = np.zeros((exp_conds_rav.shape[1], 2, 2))
#for i, cond in enumerate(exp_conds_rav.T):
#    im1_ind = np.where(orig_id==cond[0])
#    im2_ind = np.where(orig_id==cond[1])
#    resps[i,0,:] = [s[im1_ind], b[im1_ind]]
#    resps[i,1,:] = [s[im2_ind], b[im2_ind]]
#ind = -1
#print(resps[ind])
#print(exp_conds_names_rav[:,ind])#gives (img1 img2)X(s,b)
#print(exp_conds_rav[:,ind])


ecc, rfx, rfy = (mu_ecc, mux, muy)
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
#%%
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

