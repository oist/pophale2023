#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 09:12:55 2022

@author: sam
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
from natsort import natsorted
from scipy import stats
import os
import matplotlib.pyplot
import matplotlib.dates
import datetime
import matplotlib.dates as mdates
from scipy import ndimage

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[int(n/2) :] / n

base_folder='/home/sam/bucket/octopus/apartment/homeostasis/'
folder_list=natsorted(glob.glob(base_folder + '/*/'))


for folder in folder_list:

    file_list=natsorted(glob.glob(folder + '*.detectronResults'))
    allmo=[]
    for filename in file_list:
        file=h5py.File(filename,'r')
        mo=file['com'][:]
        mo[-720:]=mo[-720]
        allmo.extend(mo)
        
    allmo=np.array(allmo)
    
    allV=[]
    for y in range(4):
        
        vel=np.square(np.diff(allmo[:,y,0]))+np.square(np.diff(allmo[:,y,1]))
        yhat = moving_average(vel,720)
        
        allV.append(yhat)
    allV=np.array(allV)
    
    fs=file.attrs['fps']
    basename = folder.split('/')[-2]
    outputFile=base_folder + '/' + basename + '_coms'
    
    writer = h5py.File(outputFile, 'w')
    writer.attrs.create('folder', \
              folder, \
              dtype=h5py.special_dtype(vlen=str))
    writer.create_dataset('com', data=allmo)
    writer.create_dataset('vel', data=allV)
    writer.create_dataset('fps', data=fs)
    writer.close()
