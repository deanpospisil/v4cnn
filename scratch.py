# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 15:57:51 2019

@author: deanpospisil
"""

import inspect
def gaussian2(pos, **args): 
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    print(values)
    cov = np.array([[varx, r*(varx*vary)**0.5], 
          [r*(varx*vary)**0.5, vary]])
    print(r)
    resp = multivariate_normal([mux, muy], cov).pdf(pos)
    resp= resp/np.max(resp)
    resp = resp*amp 
    return resp

inv = {'amp':1, 'mux':1, 'muy':1, 'varx':1, 'vary':1, 'r':.1}
gaussian2([0,0], **inv)