#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 17:36:06 2021

@author: sam
"""


import os
import cv2
import numpy as np
import h5py
import argparse
# import glob
# from tqdm import tqdm
# from natsort import natsorted

# put all the avi videos you want in a folder, that is video path. 
#You can make output path the same as videopath if you have write access

if __name__ == '__main__':
    
    p = argparse.ArgumentParser(\
        'meanInt',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--video')
    p.add_argument('--outputPath')
    p.add_argument('--mask')
    args = p.parse_args()
    
    if args.mask:
        mask=cv2.imread(args.mask,0)
  #if you want to just take the mean in a certain region
   # crop=[1300,1300,1800,1800]
    
    # videoList=[]
    # videoList.extend(natsorted(glob.glob(args.videoPath + '/*.avi')))
    # videoList.extend(natsorted(glob.glob(args.videoPath + '/*.MP4')))

    #add in check for output file so things dont get overwritten!
    outputFile=args.outputPath  +  os.path.basename(args.video) + '.meanInt'
    if not os.path.exists(outputFile):
        try:
            print('video is ' + args.video)  
           
            cap = cv2.VideoCapture(args.video)   
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            
          
            mi=np.zeros([length])
            tally=0
            while(cap.isOpened()):
                succ, img = cap.read() 
                if succ == True:
                   # imgG = cv2.cvtColor(img[crop[0]:crop[2],crop[1]:crop[3],:], cv2.COLOR_BGR2GRAY)
                    
                    imgG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    mi[tally]=np.mean(imgG*(mask>0))
    
                    tally+=1
                    if tally % 10000 == 0:
                        print(tally)
                else:
                    break
            cap.release()
                

            writer = h5py.File(outputFile, 'w')
            writer.attrs.create('video', \
                  outputFile, \
                  dtype=h5py.special_dtype(vlen=str))
            writer.attrs.create('fps', fps, dtype='float32')
            writer.create_dataset('int',data=mi)
            writer.close()
        except:
            print('something wrong on  ' + args.video)  
            writer = h5py.File(outputFile, 'w')
            writer.attrs.create('video', \
                  outputFile, \
                  dtype=h5py.special_dtype(vlen=str))
            writer.attrs.create('fps', fps, dtype='float32')
      #      writer.create_dataset('crop', data=crop)
            writer.create_dataset('int',data=mi)
            writer.close()
    
        
