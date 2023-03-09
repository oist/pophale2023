#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 12:59:41 2022

@author: sam
"""

import cv2
import pandas as pd
import numpy as np
import os


folder='/home/sam/bucket/octopus/8k/awake'
outputFolder='/home/sam/bucket/octopus/8k/awake/extractedFullFrames'
frameFrame = pd.read_csv('/home/sam/bucket/octopus/8k/awake/awake_pattern_frames.csv')

    
for ind, file in enumerate(frameFrame.columns):
    nameChange=file.split('_')[0]
    file1=nameChange + '.MP4'
    cap = cv2.VideoCapture(folder + '/' + file1)
    currFrames=frameFrame[file]
    currOct=currFrames[0]
    currFrames=currFrames.drop(currFrames.index[0])
    currFrames.reset_index(drop=True, inplace=True)
    
    for f in currFrames:
        if not pd.isnull(f):
            try:
                intF=np.int32(f)
                cap.set(cv2.CAP_PROP_POS_FRAMES, intF)
                succ, img = cap.read() 
                cv2.imwrite(outputFolder + '/' + currOct + '_' + file + '_' + str(intF) + '.png',img)
            except:
                print('issue detected')