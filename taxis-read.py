# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 19:02:14 2015

@author: Rory
"""

import pandas as pd
import numpy as np
import os 


def get_files(direc):
    for root, dirs, files in os.walk(direc):
        files = files
        
    full_files = []    
    for fi in files:
        full_files.append(os.path.join(root, fi))
        
    return full_files
    
    
full_files = get_files('01')

# Read in the data assign
frames = []
for index, file_path in enumerate(full_files):
    data = pd.read_csv(file_path, infer_datetime_format=True,\
            header=None, parse_dates = [1],\
            names = ['taxi_id', 'date_time', 'longitude', 'latitude'])
    frames.append(data)
    
data = pd.concat(frames)
del frames, index

time_diffs = []
for g in grouped:
    time_diffs.append(g[1].diff()/np.timedelta64(1,'s'))
    
    
#x = data.groupby(by='taxi_id').date_time.diff()
#data = pd.read_csv('01/9754.txt', infer_datetime_format=True,\
#            header=None, parse_dates = [1],\
#            names = ['taxi_id', 'date_time', 'longitude', 'latitude'])
#
#x = data.date_time.diff()/np.timedelta64(1,'s')
#
#x.hist()