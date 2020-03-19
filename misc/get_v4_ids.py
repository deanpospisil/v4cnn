# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 20:44:29 2017

@author: deanpospisil
"""

ap_cred =  [1,12,24,33,38,43,58,87,91,109];  
brown = [68, 70, 71, 72, 79];
area_blue =  [17, 40, 59, 64, 77, 78, 82, 85, 102]
green =  [27, 53, 56, 65];
purple = [36, 83, 92];
gray  = [37,51,61,63];

clusters = {}
clusters['apc_red'] =  [1,12,24,33,38,43,58,87,91,109];  
clusters['brown'] = [68, 70, 71, 72, 79];
clusters['area_blue'] =  [17, 40, 59, 64, 77, 78, 82, 85, 102]
clusters['green'] =  [27, 53, 56, 65];
clusters['purple'] = [36, 83, 92];
clusters['grey']  = [37,51,61,63];
cluster_index = [[] for x in range(109)]
for a_cluster in clusters:
    for ind in clusters[a_cluster]:
        cluster_index[ind-1] = a_cluster
for i, a_entry in enumerate(cluster_index):
    if len(a_entry) == 0:
        cluster_index[i] = 'none'
    
    