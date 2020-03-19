

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

fn = top_dir + 'data/models/' + 'apc_models_362.nc'
dmod = xr.open_dataset(fn, chunks={'models':50, 'shapes':370})['resp'].load()

coord_array = np.array([dmod.coords['models'].or_mean, dmod.coords['models'].or_sd, 
               dmod.coords['models'].cur_mean, dmod.coords['models'].cur_sd])
models = dmod.values    


    



# In[19]:
'''
fn = 'bvlc_reference_caffenetAPC362_pix_width[30.0]_pos_(64.0, 164.0, 101)_analysis.p'
an=pk.load(open(top_dir + 'data/an_results/' + fn,'rb'),
        encoding='latin1')
fvx = an[0].sel(concat_dim='r2')
rf = an[0].sel(concat_dim='rf')
cnn = an[1]
v4_name = 'V4_362PC2001'
v4_resp_apc = xr.open_dataset(top_dir + 'data/responses/' + v4_name + '.nc')['resp'].load()
v4_resp_apc = v4_resp_apc.transpose('shapes', 'unit')


# In[11]:

import d_curve as dc
import caffe_net_response as cf
import d_img_process as imp
box_lengths = [11,51,99,131,163]#taking from wyeths CaffeNet figure
img_n_pix = 227
max_pix_width = [30.,]

s = l.loadmat(top_dir + 'img_gen/PC3702001ShapeVerts.mat')['shapes'][0]
base_stack = dc.center_boundary(s)
boundaries = imp.center_boundary(s)
scale = max_pix_width/dc.biggest_x_y_diff(boundaries)
shape_ids = range(-1, 370); center_image = round(img_n_pix/2)
x = (center_image, center_image, 1);y = (center_image, center_image, 1)
stim_trans_cart_dict, stim_trans_dict = cf.stim_trans_generator(shapes=shape_ids, scale=scale, x=x, y=y)
plt.figure(figsize=(12,24));
center = 113

trans_img_stack = np.array(imp.boundary_stack_transform(stim_trans_cart_dict, base_stack, npixels=227))
#plot smallest and largest shape
no_blank_image = trans_img_stack[1:]
extents = (no_blank_image.sum(1)>0).sum(1)
plt.subplot(121)
plt.imshow(no_blank_image[np.argmax(extents)],
                          interpolation = 'nearest', cmap=plt.cm.Greys_r)
for box_length in box_lengths:
    rectangle = plt.Rectangle((center-np.ceil(box_length/2.), center-np.ceil(box_length/2)),
                               box_length, box_length, fill=False, edgecolor='r')
    plt.gca().add_patch(rectangle)
plt.subplot(122)
plt.imshow(no_blank_image[np.argmin(extents)],
                          interpolation = 'nearest', cmap=plt.cm.Greys_r)
for box_length in box_lengths:
    rectangle = plt.Rectangle((center-np.ceil(box_length/2.), center-np.ceil(box_length/2)),
                               box_length, box_length, fill=False, edgecolor='r')
    plt.gca().add_patch(rectangle)

a = np.hstack((range(14), range(18,318)));a = np.hstack((a, range(322, 370)))
no_blank_image = no_blank_image[a]
aperture = 20
plt.figure(figsize=(12,12))
no_rotation = [0, 1, 2, 10,14, 18, 26,30, 38, 46, 54, 62, 70, 78, 86, 94,102, 110, 118, 126, 134,
 142, 150, 158,166, 174, 182, 190,198, 206, 214, 222, 224, 232, 236, 244, 252, 254, 
 262, 270, 278, 286, 294, 302, 310, 314, 322, 330, 338, 346, 354, ]
for i, a_shape in enumerate(no_blank_image[no_rotation]):
    plt.subplot(8,7,i+1)
    plt.imshow(a_shape, interpolation = 'nearest', cmap=plt.cm.Greys_r)
    plt.xlim([center-aperture, center+aperture]);plt.ylim([center-aperture, center+aperture])
    plt.xticks([]);plt.yticks([])
plt.tight_layout()


plt.figure(figsize=(12,12))
for i, a_shape in enumerate(no_blank_image):
    plt.subplot(20,19,i+1)
    plt.imshow(a_shape, interpolation = 'nearest', cmap=plt.cm.Greys_r)
    plt.xlim([center-aperture, center+aperture]);plt.ylim([center-aperture, center+aperture])
    plt.xticks([]);plt.yticks([])
plt.tight_layout()

print('max: '+str(np.max(no_blank_image))
            + ' min:' + str(np.min(no_blank_image)))  


# <h3>Sparsity.</h3>
# We observed responses of CaffeNet both to our stimuli could be quite limited. Many units in the rectified layers of CaffeNet did not respond to any stimuli at any position or only several. Typical fits of the APC model predict rich tuning across shapes. In addition very sparse responses pose  a problem for our measure of translation invariance where if the unit only responds to one shape, then it trivially achieves perfect translation invariance. For these reasons we sought a measure of sparsity along which to filter for units with the range of sparsity seen for real V4 cells.
# We considered a few different metrics of sparsity
# 
# A first choice was the coefficient of variation:
# $$C = \frac{\sigma}{\mu}$$
# 
# and the variant made by Treves and Rolls that scales it between 0 and 1:
# 
# $$T = \frac{1}{C^2 + 1}$$
# 
# These were well suited to rectified layers of CaffeNet but for other layers such as FC8 where there were negative values did not work, as the measure was designed for positive distributions.
# 
# We also considered the Gini Coefficient but it suffered from the same problem.
# 
# We settled on kurtosis:
# 
# $$K = \frac{(x-\mu)^4}{n \sigma^4}$$
# 
# For a gaussian distribution $K=3$.
# 
# E T Rolls, M J Tovee, 1995, Sparseness of the neuronal representation of stimuli in the primate temporal visual cortex, Journal of Neurophysiology, 73(2):713-726.
# D J Field, 1994, What is the goal of sensory coding?, Neural Computation, 6:559-601.

# In[15]:

def kurtosis_unit(unit):
    mu = np.mean(unit)
    sig = np.std(unit)
    k = (np.sum(((unit - mu)**4)))/((sig**4)*len(unit))
    return k
def kurtosis(da):
    da = da.transpose('shapes', 'unit')
    mu = da.mean('shapes')
    sig = da.reduce(np.nanvar, dim='shapes')
    k = (((da - mu)**4).sum('shapes', skipna=True) / da.shapes.shape[0])/(sig**2)
    return k


# In[18]:

k_apc = kurtosis(v4_resp_apc).values

plt.figure(figsize=(8,3))
plt.subplot(131)
plt.hist(v4_resp_apc.values.ravel(),bins=30,histtype='step')
plt.xlabel('Normalized firing rate');plt.ylabel('Count');plt.xticks([0,1])

plt.subplot(132)
plt.hist(k_apc, bins=30, histtype='step')
plt.xlabel('K');plt.ylabel('Count');plt.title('K histogram')

plt.subplot(133)
plt.hist(v4_resp_apc[:, np.argmax(k_apc)], bins=30, log=False, normed=False, histtype='step',  range=[0,1])
plt.hist(v4_resp_apc[:, np.argmin(k_apc)], bins=30,log=False, normed=False, histtype='step', range=[0,1])
plt.legend(np.round([np.max(k_apc), np.min(k_apc)],1), 
           title='Kurtosis', loc=1, fontsize='small')
plt.xlabel('Normalized firing rate');plt.ylabel('Count');plt.xticks([0,1])

plt.title('max & min K V4 units.')
plt.tight_layout();plt.show();


# From left to right. 
# 1. Here we see the data is normalized to one with a tendency towards lower values. 
# 2. We find our k-distribution is sparse with most units having a low k, but a few particularly sparse ones.
# 3. Plotting the response distribution of the units with highest and lowest k gives us a sense for what k measures, and the range of distributions.

# In[151]:

k_cnn = cnn['k'].dropna()

hist, bins = np.histogram((k_cnn), bins='auto', normed=False)
hist =  hist/float(len(k_cnn))
hist = [0,] + list(hist)
plt.step(bins, hist)

hist, bins = np.histogram((k_apc), bins=bins, normed=False)
hist =  hist/float(len(k_apc))
hist = [0,] + list(hist)
plt.step(bins, hist)

plt.xscale('log', nonposy='clip');
plt.yscale('log', nonposy='clip');

yticks = np.round(np.array([np.min(hist), np.max(hist) , 1]),1)
#plt.yticks(yticks, yticks)
xticks= np.round(np.array([np.min(k_cnn), 10, 100, np.max(k_cnn)]),1)
plt.xticks(xticks, xticks);
plt.xlabel('Kurtosis');
plt.ylabel('%\nunits', rotation='horizontal', labelpad=20)
plt.tight_layout();
plt.legend(['CaffeNet','V4'], loc=9);



sparsest_unit = np.zeros((371,1))
sparsest_unit[0] = 1
k = kurtosis_unit(sparsest_unit)
print('sparsest possible = '+ str(k))
print('max sparsity = ' + str(max(k_cnn)))


# CaffeNet has a clear peak near the highest possible kurtosis value. These are units that practically only responded to one shape. Clearly there are a fair amount of units with far higher kurtosis than those found in V4 for this shape set.

# In[157]:

plt.figure(figsize=(2, 24))
title = 'Sparsity of Layers.'
dp.stacked_hist_layers(k_cnn, logx=True, logy=False,
                    xlim=[min(k_cnn), max(k_cnn)+10],
                    maxlim=True, bins=100, title=title)

#plt.xscale('log', nonposy='clip');
plt.plot([max(k_apc),]*2,[0,12], lw=2)
xticks= np.round(np.array([np.min(k_cnn), np.max(k_cnn)]),1)
plt.xticks(xticks, xticks);
plt.xlabel('Kurtosis')
print(max(k_apc))
'''
