# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 00:15:16 2016

@author: deanpospisil
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import os


top_dir = os.getcwd().split('v4cnn')[0]
sys.path.append( top_dir + 'xarray/')
top_dir = top_dir + 'v4cnn/'
sys.path.append(top_dir + 'common')
sys.path.append(top_dir + 'img_gen')

import d_img_process as imp
import xarray as xr

#cross validation comparison of APC and AlexNet
v4 = xr.open_dataset(top_dir + 'data/responses/V4_362PC2001.nc', chunks = {'shapes':370})['resp']
daa = xr.open_dataset(top_dir + 'data/responses/PC370_shapes_0.0_369.0_370_x_-100.0_100.0_201.nc')['resp'].loc[:, 0, :]
daa = daa.isel(shapes=v4.coords['shapes'])
alex = daa[:, daa.layer_label==b'fc8']
apc = xr.open_dataset(top_dir + 'data/models/apc_models_362.nc', chunks = {'shapes':370})['resp']

apc = apc[:,:]
alex = alex[:, 1:100]
import numpy as np
import matplotlib.pyplot as plt

from sklearn import cross_validation, datasets, linear_model


lasso = linear_model.Lasso()
alphas = np.logspace(-4, -.5, 30)

scores = list()
scores_std = list()

for alpha in alphas:
    lasso.alpha = alpha
    this_scores = cross_validation.cross_val_score(lasso, alex.values, v4.values.T[:,0], n_jobs=1)
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))

plt.figure(figsize=(4, 3))
plt.semilogx(alphas, scores)
# plot error lines showing +/- std. errors of the scores
plt.semilogx(alphas, np.array(scores) + np.array(scores_std) / np.sqrt(len(X)),
             'b--')
plt.semilogx(alphas, np.array(scores) - np.array(scores_std) / np.sqrt(len(X)),
             'b--')
plt.ylabel('CV score')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')


lasso_cv = linear_model.LassoCV(alphas=alphas)
k_fold = cross_validation.KFold(362, 3)

print("Answer to the bonus question:",
      "how much can you trust the selection of alpha?")
print()
print("Alpha parameters maximising the generalization score on different")
print("subsets of the data:")
for k, (train, test) in enumerate(k_fold):
    lasso_cv.fit(alex.values[train,:], v4.values.T[train,0])
    print("[fold {0}] alpha: {1:.5f}, score: {2:.5f}".
          format(k, lasso_cv.alpha_, lasso_cv.score(alex.values[test], v4.values.T[test,0])))


plt.show()


