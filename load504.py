# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 21:52:09 2018

@author: deanpospisil
"""

import pandas as pd
from datetime import *
from dateutil.relativedelta import *
import numpy as np
from datetime import timedelta
NOW = datetime.now()
stocks = ['^IXIC.csv', '^GSPC.csv','BTC-USD.csv', 'ETH-USD.csv', 
          'GooglesearchBTC.csv', 'vixcurrent.csv', 'Googlesearch-Ethereum.csv',
          'Googlesearch-Cryptocurrency.csv', 'Googlesearch-Darknet.csv']

fdir = '/Users/deanpospisil/Desktop/modules/v4cnn/504 Cryptocurrency Project/'
f = [pd.read_csv(str(fdir+stocks[i])) for i in range(len(stocks))]
date_lists = [f[i].iloc[:,0] for i in range(len(stocks))]
data_lists = [f[i].iloc[:,1:] for i in range(len(stocks))]

import dateutil
from scipy.interpolate import interp1d


parsed_dates = []
relative_dates = []
for a_date_list in date_lists:
    temp_parsed = []
    temp_delt = []
    for a_date in a_date_list:
        datetime = dateutil.parser.parse(a_date)
        temp_parsed.append(datetime)
        temp_delt.append((datetime-NOW).days)
    parsed_dates.append(temp_parsed)
    relative_dates.append(temp_delt)
 
    

lower_cutdate = np.max([np.min(relative_date) for relative_date in relative_dates])
upper_cutdate = np.min([np.max(relative_date) for relative_date in relative_dates])

#do whole range
lower_cutdate = np.min([np.min(relative_date) for relative_date in relative_dates])
upper_cutdate = np.max([np.max(relative_date) for relative_date in relative_dates])


new_x = np.arange(lower_cutdate, upper_cutdate, 1)

new_dates =np.array([(NOW+timedelta(days=np.double(x))).strftime('%m/%d/%Y') for x in new_x])

def resample(orig_y, orig_x, new_x):
    f = interp1d(orig_x, orig_y, kind='linear', bounds_error=False, fill_value=np.nan)
    new_y = f(new_x)
    return new_y

new_dat_list = []
for a_relative_date_list, a_data_frame in zip(relative_dates, data_lists):
    data_vals = a_data_frame.values
    new_dat = np.zeros((len(new_x), data_vals.shape[1]))
    for i, col in enumerate(data_vals.T):
        new_dat[:,i] = resample(col, a_relative_date_list, new_x)
        
    new_dat_list.append(new_dat)


new_data_frames = []
for new_dat, old_data_frame, file_name in zip(new_dat_list, f, stocks):
    plt.plot(new_dat);plt.ylim(0,100);
    old_index = list(old_data_frame.columns[1:])
    new_dat = np.hstack((new_dates[:,np.newaxis], new_dat))
    new_data_frame = pd.DataFrame(new_dat, columns=['date',] + old_index)
    new_data_frame.to_csv(fdir + 'rs_'+file_name)
    new_data_frames.append(new_data_frame)
    
    



