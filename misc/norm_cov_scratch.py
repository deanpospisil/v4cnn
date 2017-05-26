import numpy as np
import xarray as xr
import os, sys
import matplotlib.pyplot as plt
import pickle

top_dir = os.getcwd().split('v4cnn')[0]
top_dir = top_dir + 'v4cnn'

def ti_av_cov(da):
    dims = da.coords.dims
    #get the da in the right shape
    if ('x' in dims) and ('y' in dims):
        da = da.transpose('unit','shapes', 'x', 'y')
    elif ('x' in dims):
        da = da.transpose('unit', 'shapes', 'x')
    elif ('y' in dims):
        da = da.transpose('unit', 'shapes', 'y')
        
    #some data to store
    ti = np.zeros(np.shape(da)[0])
    dens = np.zeros(np.shape(da)[0])
    nums = np.zeros(np.shape(da)[0])
    tot_vars = np.zeros(np.shape(da)[0])
    kurt_shapes = np.zeros(np.shape(da)[0])
    kurt_x =  np.zeros(np.shape(da)[0])

    for i, unit_resp in enumerate(da):
        if len(unit_resp.shape)>2:
            #unwrap spatial
            unit_resp = unit_resp.values.reshape(unit_resp.shape[0], unit_resp.shape[1]*unit_resp.shape[2])   
        else:
            unit_resp = unit_resp.values
        unit_resp = unit_resp.astype(np.float64)
        unit_resp = unit_resp - np.mean(unit_resp, 0, keepdims=True, dtype=np.float64)
 

        cov = np.dot(unit_resp.T, unit_resp)
        cov[np.diag_indices_from(cov)] = 0
        numerator = np.sum(np.triu(cov))

        vlength = np.linalg.norm(unit_resp, axis=0, keepdims=True)
        max_cov = np.outer(vlength.T, vlength)
        max_cov[np.diag_indices_from(max_cov)] = 0
        denominator= np.sum(np.triu(max_cov))

        kurt_shapes[i] = kurtosis(np.sum(unit_resp**2, 1))
        kurt_x[i] = kurtosis(np.sum(unit_resp**2, 0))
        den = np.sum(max_cov)
        num = np.sum(cov)
        dens[i] = den
        nums[i] = num
        tot_vars[i] = np.sum(unit_resp**2)
        if den!=0 and num!=0:
            ti[i] = num/den 
    return ti, kurt_shapes, kurt_x, dens, nums, tot_vars 

#
    
#x = np.zeros((22096, 300, 400))
def norm_avcov(x):
    x = x.astype(np.float64)
    
    cov = np.matmul(np.transpose(x, axes=(0, 2, 1)), x)
    numerator = np.sum(np.triu(cov, k=1), (1, 2))
    
    vlength = np.linalg.norm(x, axis=1, keepdims=True)
    max_cov = np.multiply(np.transpose(vlength, axes=(0, 2, 1)), vlength)
    denominator= np.sum(np.triu(max_cov, k=1), (1, 2))
    norm_cov = numerator/denominator
    
    return norm_cov



net_resp_name = 'bvlc_reference_caffenety_test_APC362_pix_width[32.0]_x_(104.0, 124.0, 11)_x_(104.0, 124.0, 11)_amp_None.nc'
da = xr.open_dataset(top_dir + '/data/responses/' + net_resp_name)['resp']
layers = ['']
goforit=False     
if 'netwtsd' not in locals() or goforit:
    with open(top_dir + '/nets/netwtsd.p', 'rb') as f:    
        try:
            netwtsd = pickle.load(f, encoding='latin1')
        except:
            netwtsd = pickle.load(f)
            
def spatial_weight_normcov(netwtsd):
    unit_coords = xr.concat([netwtsd[key].coords['unit'] 
                            for key in netwtsd.keys()], 'unit').coords
    netwts_list = [netwtsd[key].values for key in netwtsd.keys()]
    
    av_cov_list = []
    for layer_wt in netwts_list:
        o_shape = np.shape(layer_wt)
        ravel_space = o_shape[:2] + (np.product(o_shape[2:]),)
        av_cov_list.append(norm_avcov(layer_wt.reshape(ravel_space)))
    
    av_cov = np.concatenate(av_cov_list)
    av_cov_da = xr.DataArray(av_cov, unit_coords)    
    return av_cov_da

#%%
#def spatial_resp_normcov(resp):
da = da.squeeze()
da = da[..., ::50]
resp = da
dims = resp.coords.dims
if ('x' in resp) and ('y' in dims):
    resp = resp.transpose('unit','shapes', 'x', 'y')
elif ('x' in dims):
    resp = resp.transpose('unit', 'shapes', 'x')
elif ('y' in dims):
    resp = resp.transpose('unit', 'shapes', 'y')
resp_vals = resp.values

unit_coords = resp.coords['unit']
o_shape = np.shape(resp_vals)
ravel_space = o_shape[:2] + (np.product(o_shape[2:]),)
av_cov = norm_avcov(resp_vals.reshape(ravel_space))

av_cov_da = xr.DataArray(av_cov, unit_coords)    

    
    
    
    
    
    