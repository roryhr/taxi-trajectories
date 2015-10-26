# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 19:02:14 2015

@author: Rory H.R. 
"""
from pykalman import KalmanFilter

import pandas as pd
import numpy as np
import os 
#from math import radians, cos, sin, asin, sqrt, pi
import matplotlib.pyplot as plt

# Patch to get values rather than log(10) on hexbin plot
from matplotlib.ticker import LogFormatter 
class LogFormatterHB(LogFormatter):
     def __call__(self, v, pos=None):
         vv = self._base ** v
         return LogFormatter.__call__(self, vv, pos) 

#%% Haversine formula
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    
    Most lat/lon points are closely spaced. Can implement small angle approx 
    to improve speed.
    """
    # convert decimal degrees to radians 
    
#    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    lon1 *= np.pi/180  # Convert from degrees to radians
    lon2 *= np.pi/180
    lat1 *= np.pi/180
    lat2 *= np.pi/180    
    
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r * 1000     # convert to meters
    
    
def my_low_pass_filter(array):
    freqs = np.fft.fft(array)
    half_way = int(freqs.size/2)
    ten_percent = int(freqs.size/10)
    
    freqs[(half_way-2*ten_percent):(half_way+2*ten_percent)] = 0
    return np.fft.ifft(freqs)


#%% Read in data
def get_files(direc):
    for root, dirs, files in os.walk(direc):
        files = files
        
    full_files = []    
    for fi in files:
        full_files.append(os.path.join(root, fi))
        
    return full_files
    
    
full_files = get_files('data/02')

#%% Read in the data 
print "Reading in the .txt files..."
frames = []
for index, file_path in enumerate(full_files):
    data = pd.read_csv(file_path, infer_datetime_format=True,\
            header=None, parse_dates = [1],\
            names = ['taxi_id', 'date_time', 'longitude', 'latitude'])
    frames.append(data)
#    print file_path
#    print data.describe()

data = pd.concat(frames)
del frames, index

grouped = data.groupby('taxi_id')['date_time']


#%% Compute Time Intervals
print "Computing time intervals..."
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
#time_diffs[(time_diffs > 0) & (time_diffs < 12)].hist(bins = 20)#x = data.groupby(by='taxi_id').date_time.diff()


#%% Compute Distance Intervals
print "Computing distance intervals..."
lon1 = np.array(data.longitude[0:-2])
lon2 = np.array(data.longitude[1:-1])
lat1 = np.array(data.latitude[0:-2])
lat2 = np.array(data.latitude[1:-1])

distances = haversine(lon1, lat1, lon2, lat2)


#%% Plotting: Time -- plots a histogram time intervals with 
#                     proportions summing to 1
print "Plotting time intervals..."
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
#fig = plt.figure(figsize = (8,4))
#fig.set_figure_width(6)
axes[0].set_xlabel('Interval (minutes)')
axes[0].set_ylabel('Frequency (proportion)')
axes[0].set_title('Time Intervals')

hist, bins = np.histogram(time_diffs[(time_diffs > 0) & \
            (time_diffs < 12)].astype(np.ndarray), bins=20)
axes[0].bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]))

#% Plotting: Distance -- plots a normed histogram of distance intervals
distances = pd.Series(distances)
distances.dropna(inplace=True)

hist, bins = np.histogram(distances[(distances > 0) & \
            (distances < 8000)].astype(np.ndarray), bins=20)
axes[1].bar(bins[:-1], hist.astype(np.float32) / hist.sum(), width=(bins[1]-bins[0]))
axes[1].set_xlabel('Distance (meters)')
axes[1].set_ylabel('Frequency (proportion)')

fig.tight_layout()

#%% Plot position density 
print "Plotting position density..."
xmin = 116.1
xmax = 116.8
ymin = 39.5
ymax = 40.3

window = data[(xmin < data.longitude) & (data.longitude < xmax) & \
            (ymin < data.latitude) & ( data.latitude < ymax)]

x = np.array(window.longitude)
y = np.array(window.latitude)

plt.figure(figsize = (10,7), dpi=200)
plt.hexbin(x,y,bins='log', gridsize=800, cmap=plt.cm.hot)   # black -> red > white
plt.axis([xmin, xmax, ymin, ymax])
plt.title("Traffic data over Beijing")
plt.xlabel('Longitude (degrees)')
plt.ylabel('Latitude (degrees)')

cb = plt.colorbar(format=LogFormatterHB())
cb.set_label('Number of points')

plt.show()

#%% Make the 5th Ring Road Beijing

# Location details
xmin, xmax = 116.25, 116.5
ymin, ymax = 39.75, 40.1

window = data[(xmin < data.longitude) & (data.longitude < xmax) & \
            (ymin < data.latitude) & ( data.latitude < ymax)]

x = np.array(window.longitude)
y = np.array(window.latitude)

plt.figure(figsize = (10,6), dpi=150)
plt.hexbin(x,y, bins='log', gridsize=800, cmap=plt.cm.hot)   # black -> red > white
plt.axis([xmin, xmax, ymin, ymax])
plt.title("Traffic data for Beijing's 5th Ring Road")
plt.xlabel('Longitude (degrees)')
plt.ylabel('Latitude (degrees)')

cb = plt.colorbar(format=LogFormatterHB())
cb.set_label('Number of points')

plt.tight_layout()
plt.show()


#%% Select data for one taxi
one_taxi = data[data.taxi_id == 2172]





#plt.figure(figsize = (12,8), dpi=100)

#%% Plot
plt.plot(my_low_pass_filter(np.array(one_taxi.longitude)),\
        my_low_pass_filter(np.array(one_taxi.latitude)))
plt.axis([xmin, xmax, ymin, ymax])
plt.title("Traffic data for Beijing -- 5th Ring Road")
plt.show()

#%% Kalman Filter and plot results

#kf = KalmanFilter(transition_matrices = [[1, 1], [0, 1]], observation_matrices = [[0.1, 0.5], [-0.3, 0.0]])
#kf = KalmanFilter(transition_matrices = [[1, 0], [0, 1]], observation_matrices = [[0.1, 0.5], [-0.3, 0.0]], n_dim_obs=2, n_dim_state=2)
measurements = np.asarray([one_taxi.longitude, one_taxi.latitude])
measurements = measurements.T[0:400]
kf = KalmanFilter(initial_state_mean = measurements[0], \
           n_dim_obs=2, n_dim_state=2)


kf = kf.em(measurements)
#(filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
(smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)

#plt.plot(filtered_state_means.T[0],filtered_state_means.T[1])
plt.plot(smoothed_state_means.T[0],smoothed_state_means.T[1])
plt.title("Smoothed Data With Karman Filter")
plt.xlabel('Longitude (degrees)')
plt.ylabel('Latitude (degrees)')
print "Taxi data for Taxi ID: ", 2172
print "Start date and time: ", one_taxi.date_time[0]
print "Duration :", one_taxi.date_time[400] - one_taxi.date_time[0]
plt.figure()
#plt.plot(measurements.T[0],measurements.T[1])
