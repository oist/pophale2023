#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 13:59:29 2022

@author: sam
"""
import h5py
import glob
import matplotlib.pyplot as plt
import os

dirname='/home/sam/bucket/octopus/8k/oct_456'
file = dirname + '/htFlip.csv'
af = sorted(glob.glob(dirname + '/' + '*.reg'))

#df = pd.DataFrame(data=af)
#df.to_csv(file)
   
for f in af:
    currDsetFile=h5py.File(f,'r')
    img=currDsetFile['patterns1'][0]
    img2=currDsetFile['patterns2'][0]
    
    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(img2)
    plt.title(os.path.basename(f))
    plt.savefig(f + '_htFig.png')
    plt.close()