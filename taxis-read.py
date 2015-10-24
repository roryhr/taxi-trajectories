# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 19:02:14 2015

@author: Rory
"""

import pandas as pd
import numpy as np
import os 


from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def get_files(direc):
    for root, dirs, files in os.walk(direc):
        files = files
        
    full_files = []    
    for fi in files:
        full_files.append(os.path.join(root, fi))
        
    return full_files
    
    
full_files = get_files('data')

#%% Read in the data 
frames = []
for index, file_path in enumerate(full_files):
    data = pd.read_csv(file_path, infer_datetime_format=True,\
            header=None, parse_dates = [1],\
            names = ['taxi_id', 'date_time', 'longitude', 'latitude'])
    frames.append(data)
    print file_path
    print data.describe()

    
    
data = pd.concat(frames)
del frames, index

grouped = data.groupby('taxi_id')['date_time']


#%% Compute Time Intervals

times = []
for g in grouped:
    times.append(g[1].diff())
#    print pd.Series(g[1].diff())
#    time_diffs.append(pd.Series(g[1].diff()))
    
time_diffs = pd.Series()
time_diffs = pd.concat(times)
time_diffs /= np.timedelta64(1,'s') # Divide by 1 second, for float64 data

time_diffs.dropna()

time_diffs /= 60    # Convert to minutes
time_diffs[(time_diffs > 0) & (time_diffs < 12)].hist(bins = 20)#x = data.groupby(by='taxi_id').date_time.diff()
#data = pd.read_csv('01/9754.txt', infer_datetime_format=True,\
#            header=None, parse_dates = [1],\
#            names = ['taxi_id', 'date_time', 'longitude', 'latitude'])
#
#x = data.date_time.diff()/np.timedelta64(1,'s')
#
#x.hist()