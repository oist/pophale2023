#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 09:42:34 2022
@author: sam
"""

import h5py
import glob
from natsort import natsorted
import os
import cv2

folder='/home/sam/bucket/octopus/apartment/high_res'
file_list=natsorted(glob.glob(folder + '/*asTimes'))

for filename in file_list:
      file=h5py.File(filename,'r')
      asTimes=file['as_frames'][:]
      currVideo=folder + '/' + os.path.basename(filename)[0:-8] + '.avi'
      
      for currTime in asTimes:
          outputName=folder + '/' + os.path.basename(filename)[0:-8] + '_' + str(currTime) + '.avi'
          start_secs=currTime/23-50
          end_secs=200
    
          ffmpeg_cmd = 'ffmpeg \
          -ss {} \
          -i {} \
          -t {} \
          -codec copy \
           {}'.format(start_secs, currVideo, end_secs,outputName)
          os.system(ffmpeg_cmd)
                            
          frame = folder + '/' + os.path.basename(filename)[0:-8] + '_' + str(currTime) + '.png'
    
          cap = cv2.VideoCapture(outputName)
          succ, img = cap.read()
          cv2.imwrite(frame, img)
      
      

   

    