import numpy as np
import xarray as xr
import os, sys
import matplotlib.pyplot as plt
import pickle

top_dir = os.getcwd().split('v4cnn')[0]
top_dir = top_dir + '/v4cnn'
def norm_avcov(x):
    x = x.astype(np.float64)
    
    cov = np.matmul(np.transpose(x, axes=(0, 2, 1)), x)
    numerator = np.sum(np.triu(cov, k=1), (1, 2))
    
    vlength = np.linalg.norm(x, axis=1, keepdims=True)
    max_cov = np.multiply(np.transpose(vlength, axes=(0, 2, 1)), vlength)
    denominator= np.sum(np.triu(max_cov, k=1), (1, 2))
    norm_cov = np.array(numerator)/np.array(denominator)
    
    return norm_cov

    
def norm_avcov_iter(x, subtract_mean=True):
    x = x.astype(np.float64)
    if subtract_mean:
        x = x - np.mean(x, 1, keepdims=True)
    diag_inds = np.triu_indices(x.shape[-1], k=1)
    numerator = [np.sum(np.dot(unit.T, unit)[diag_inds]) for unit in x]
    
    vnrm = np.linalg.norm(x, axis=1, keepdims=True)
    denominator = [np.sum(np.multiply(unit.T, unit)[diag_inds]) for unit in vnrm]    
    norm_cov = np.array(numerator)/np.array(denominator)
    norm_cov[np.isnan(norm_cov)] = 0
    
    return norm_cov



def spatial_weight_normcov(netwtsd):
    unit_coords = xr.concat([netwtsd[key].coords['unit'] 
                            for key in netwtsd.keys()], 'unit').coords
    netwts_list = []
    for key in netwtsd:
        netwt = netwtsd[key].transpose('unit', 'chan', 'y', 'x').values       
        netwts_list.append(netwt)
    
    av_cov_list = []
    for layer_wt in netwts_list:
        o_shape = np.shape(layer_wt)
        ravel_space = o_shape[:2] + (np.product(o_shape[2:]),)
        av_cov_list.append(norm_avcov_iter(layer_wt.reshape(ravel_space), subtract_mean=True))
    
    av_cov = np.concatenate(av_cov_list)
    av_cov_da = xr.DataArray(av_cov, unit_coords)    
    return av_cov_da


def spatial_resp_normcov(resp):
    dims = resp.coords.dims
    if ('x' in resp) and ('y' in dims):
        resp = resp.transpose('unit','shapes', 'x', 'y')
    elif ('x' in dims):
        resp = resp.transpose('unit', 'shapes', 'x')
    elif ('y' in dims):
        resp = resp.transpose('unit', 'shapes', 'y')
    resp_vals = resp.values
    
    unit_coords = resp.coords['unit'].coords
    o_shape = np.shape(resp_vals)
    ravel_space = o_shape[:2] + (np.product(o_shape[2:]),)
    av_cov = norm_avcov_iter(resp_vals.reshape(ravel_space), subtract_mean=True)
    resp_av_cov_da = xr.DataArray(av_cov, unit_coords)   
    return resp_av_cov_da

from scipy.stats import kurtosis

def kurtosis_da(resp):
    dims = resp.coords.dims   
    
    if ('x' in resp) and ('y' in dims):
        resp = resp.transpose('unit', 'shapes', 'x', 'y')
    elif ('x' in dims):
        resp = resp.transpose('unit', 'shapes', 'x')
    elif ('y' in dims):
        resp = resp.transpose('unit', 'shapes', 'y')
        
    stim_resp = np.array([(unit**2).sum((1, 2)) for unit in resp.values])
    pos_resp = np.array([(unit**2).sum(0).ravel() for unit in resp.values])
    k_stim = kurtosis(stim_resp, axis=1, fisher=False)
    k_pos = kurtosis(pos_resp, axis=1, fisher=False)
    return k_pos, k_stim

def tot_var(resp):
    dims = resp.coords.dims   
    if ('x' in resp) and ('y' in dims):
        resp = resp.transpose('unit','shapes', 'x', 'y')
    elif ('x' in dims):
        resp = resp.transpose('unit', 'shapes', 'x')
    elif ('y' in dims):
        resp = resp.transpose('unit', 'shapes', 'y')
        
    pwr = np.array([(unit**2).sum() for unit in resp.values])
    return pwr
#%%
a = np.array([1,2,3,4,5]).reshape(1,5)


x = np.random.randn(10, 1)
x = x*a

x = x - np.mean(x, axis=0, keepdims=True)

ti = np.dot(x.T, x)

vlength = np.linalg.norm(x, axis=0, keepdims=True)
prod_v = vlength*vlength.T
den = np.sum(np.triu(prod_v, k=1))
num = np.sum(np.triu(ti, k=1))



#%%
if sys.platform == 'linux2': 
    data_dir = '/loc6tb/dean/'
else:
    data_dir = top_dir

net_name = 'bvlc_reference_caffenetpix_width[32.0]_x_(34.0, 194.0, 21)_y_(34.0, 194.0, 21)_amp_NonePC370.nc'
da = xr.open_dataset(data_dir + '/data/responses/'+net_name)['resp']
da = da.squeeze()
da = da.transpose('unit','shapes', 'x', 'y')
#%%
da = da[:5472:1]
da = da - da[:, 0, :, :] #subtract off baseline
da = da[:, 1:, ...] #get rid of baseline shape   

goforit=True    
if 'netwtsd' not in locals() or goforit:
    with open(top_dir + '/nets/netwtsd.p', 'rb') as f:    
        try:
            netwtsd = pickle.load(f, encoding='latin1')
        except:
            netwtsd = pickle.load(f)
            
#%%
wt_av_cov = spatial_weight_normcov(netwtsd) 
resp_av_cov = spatial_resp_normcov(da) 
k_pos, k_stim = kurtosis_da(da)
#%%
pwr = tot_var(da)
#non_k_var = (k_pos<42) * (k_pos>2) * (pwr>0) *(k_stim<42) * (k_stim>2)
#resp_av_cov = resp_av_cov[non_k_var]

#%%
wt_av_cov, resp_av_cov = xr.align(wt_av_cov, resp_av_cov, join='inner')
layer_labels_ind = np.array(map(str, wt_av_cov.coords['layer_label'].values))

n_plots = len(np.unique(layer_labels_ind))
plt.figure(figsize=(12,3))
layer_labels = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6']

for i, layer in enumerate(layer_labels[1:]):
    plt.subplot(1, n_plots, i+1)
    x = wt_av_cov[layer_labels_ind==layer].values
    y = resp_av_cov[layer_labels_ind==layer].values
    if i<4:
        s=4
    else:
        s=1
    plt.scatter(x, y, s=s, color='k', edgecolors='none')
    #plt.semilogx()
    plt.xlim(-0.1,1.02);plt.ylim(-0.1,1.01);
    if i==0:
        plt.xlabel('Weight Covariance'); plt.ylabel('T.I.', rotation=0, va='center',ha='right', labelpad=15)
    if layer == 'conv2':
        plt.yticks([0,0.25,0.5, 0.75, 1]);plt.gca().set_yticklabels(['0','','0.5','','1'])
        plt.xticks([0,0.25,0.5, 0.75, 1]);plt.gca().set_xticklabels(['0','','0.5','','1'])
        plt.title(layer + '\nr = ' + str(np.round(np.corrcoef(x,y)[0,1], 2)))

    else:
        plt.yticks([0,0.25,0.5, 0.75, 1]);plt.gca().set_yticklabels(['','','','',''])
        plt.xticks([0,0.25,0.5, 0.75, 1]);plt.gca().set_xticklabels(['','','','',''])
        plt.title(layer + '\n' + str(np.round(np.corrcoef(x,y)[0,1], 2)))
    plt.tight_layout()
    plt.grid()

#%%
adj_resps=[
'bvlc_reference_caffenetpix_width[32.0]_x_(34.0, 194.0, 21)_y_(34.0, 194.0, 21)_amp_NonePC370.nc',
'bvlc_caffenet_reference_increase_wt_cov_0.5pix_width[32.0]_x_(34.0, 194.0, 21)_y_(34.0, 194.0, 21)_amp_NonePC370.nc',
'bvlc_caffenet_reference_increase_wt_cov_0.75pix_width[32.0]_x_(34.0, 194.0, 21)_y_(34.0, 194.0, 21)_amp_NonePC370.nc',
'bvlc_caffenet_reference_increase_wt_cov_0.95pix_width[32.0]_x_(34.0, 194.0, 21)_y_(34.0, 194.0, 21)_amp_NonePC370.nc',
'bvlc_caffenet_reference_increase_wt_cov_conv5_0.1pix_width[32.0]_x_(34.0, 194.0, 21)_y_(34.0, 194.0, 21)_amp_NonePC370.nc',
'bvlc_caffenet_reference_increase_wt_cov_conv5_0.95pix_width[32.0]_x_(34.0, 194.0, 21)_y_(34.0, 194.0, 21)_amp_NonePC370.nc',
'bvlc_caffenet_reference_increase_wt_cov_fc6_0.1pix_width[32.0]_x_(34.0, 194.0, 21)_y_(34.0, 194.0, 21)_amp_NonePC370.nc',
'bvlc_caffenet_reference_increase_wt_cov_fc6_0.95pix_width[32.0]_x_(34.0, 194.0, 21)_y_(34.0, 194.0, 21)_amp_NonePC370.nc',
'bvlc_caffenet_reference_increase_wt_cov_fc6_0.2pix_width[32.0]_x_(34.0, 194.0, 21)_y_(34.0, 194.0, 21)_amp_NonePC370.nc',
'bvlc_caffenet_reference_increase_wt_cov_fc6_0.3pix_width[32.0]_x_(34.0, 194.0, 21)_y_(34.0, 194.0, 21)_amp_NonePC370.nc',
'bvlc_caffenet_reference_increase_wt_cov_fc6_0.4pix_width[32.0]_x_(34.0, 194.0, 21)_y_(34.0, 194.0, 21)_amp_NonePC370.nc',
'bvlc_caffenet_reference_increase_wt_cov_fc6_0.5pix_width[32.0]_x_(34.0, 194.0, 21)_y_(34.0, 194.0, 21)_amp_NonePC370.nc',
'bvlc_caffenet_reference_increase_wt_cov_fc6_0.6pix_width[32.0]_x_(34.0, 194.0, 21)_y_(34.0, 194.0, 21)_amp_NonePC370.nc',
'bvlc_caffenet_reference_increase_wt_cov_fc6_0.7pix_width[32.0]_x_(34.0, 194.0, 21)_y_(34.0, 194.0, 21)_amp_NonePC370.nc',
'bvlc_caffenet_reference_increase_wt_cov_fc6_0.8pix_width[32.0]_x_(34.0, 194.0, 21)_y_(34.0, 194.0, 21)_amp_NonePC370.nc',
]


adj_netwts = [
'netwtsd_orig.p',
'netwtsd_0.5.p',                   
'netwtsd_0.75.p',    
'netwtsd_0.95.p',
'netwtsd_conv5_0.1.p',
'netwtsd_conv5_0.95.p',
'netwtsd_fc6_0.1.p',
'netwtsd_fc6_0.95.p',
'netwtsd_fc6_0.2.p',
'netwtsd_fc6_0.3.p',
'netwtsd_fc6_0.4.p',
'netwtsd_fc6_0.5.p',
'netwtsd_fc6_0.6.p',
'netwtsd_fc6_0.7.p',
'netwtsd_fc6_0.8.p',] 
#%%
wt_av_covs = []
resp_av_covs = []
#%%
for netwts, net_name in zip(adj_netwts[:], adj_resps[:]):
    with open(top_dir + '/nets/' + netwts, 'rb') as f:    
        try:
            netwtsd = pickle.load(f, encoding='latin1')
        except:
            netwtsd = pickle.load(f)
            
    da = xr.open_dataset(top_dir + '/data/responses/'+net_name)['resp']
    da = da.squeeze()
    da = da.transpose('unit','shapes', 'x', 'y')
    da = da[:11904]
    da = da - da[:, 0, :, :] #subtract off baseline
    da = da[:, 1:, ...] #get rid of baseline shape 
    
    wt_av_cov = spatial_weight_normcov(netwtsd) 
    resp_av_cov = spatial_resp_normcov(da)
    #wt_av_cov, resp_av_cov = xr.align(wt_av_cov, resp_av_cov, join='inner')
    wt_av_covs.append(wt_av_cov)
    resp_av_covs.append(resp_av_cov)

pickle.dump([resp_av_covs, wt_av_covs], open(top_dir + '/data/an_results/ti_vs_wt_cov_exps_all_lays.p', "wb" ) )

