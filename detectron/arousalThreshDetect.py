#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 12:22:35 2021

@author: sam
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 11:45:25 2020

@author: sam
"""


import os
import cv2
import numpy as np
import h5py
import argparse
import glob
from tqdm import tqdm
from scipy import signal, stats


#several steps:
#1. Run through and find hit time as peak in the red channel towards the middle of the clip
#2. Calculate a mask backlag seconds before that peak
#3  run optic flow using that mask

if __name__ == '__main__':
    
    p = argparse.ArgumentParser(\
        'run the detectron. Dont forget about the class number', \
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--videoPath')
    p.add_argument('--labels')
    p.add_argument('--config')
    p.add_argument('--outputPath')
    p.add_argument('--runOF',action='store_true')
    p.add_argument('--runMask',action='store_true')
    args = p.parse_args()
    
    
    codeVersion=1.7
    fastForward=65 #number of sec to fast forward for looking for red light
    backlag=30 #frames before the hit to take the mask
    forelag=30
    numFrames=1000
    
    
    
    videoList=sorted(glob.glob(args.videoPath + '/*.avi'))

    #record red channel to find hit time
    for video in videoList:
        
         outputFile=args.outputPath + os.path.basename(video) + '.findHit'
         print('video is ' + video)  
         if not os.path.exists(outputFile):
            
            
            cap = cv2.VideoCapture(video)   
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            print('fps is ' + str(fps))
            ff=int(fastForward*fps)

            for x in range(ff):
                cap.grab()
            red=np.zeros([numFrames])
            for frame in tqdm(range(numFrames), total=numFrames,desc='finding hit time'):
                succ, img = cap.read() 
                red[frame]=np.mean(img[:,:,2])
            
         
            with h5py.File(outputFile, 'w') as writer:
                writer.attrs.create('video', \
                      outputFile, \
                      dtype=h5py.special_dtype(vlen=str))
                writer.attrs.create('fps', fps, dtype='float32')
                writer.create_dataset('red',data=red)
                writer.attrs.create('version', codeVersion)
                writer.attrs.create('fastforward', ff)

         else:
             
                 reader = h5py.File(outputFile, 'r')
                 ver=reader.attrs['version']
                 reader.close()
                
                 maskExists=os.path.exists(args.outputPath + '/' + os.path.basename(video) + '_mask.png')

                 if args.runMask and ver<codeVersion: #then run mask
                     if not maskExists: #then run mask   
                         reader = h5py.File(outputFile, 'r')
                         red = np.squeeze(reader['red'][:])
                         cap = cv2.VideoCapture(video)   
                         length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                         fps = float(cap.get(cv2.CAP_PROP_FPS))
                         ff=int(reader.attrs['fastforward'])
                         reader.close()
                            
                         import detectron2
                         from detectron2.config import get_cfg
                         from detectron2.engine import DefaultPredictor
            
                         cfg = get_cfg()
                         cfg.merge_from_file(args.config)
                         cfg.OUTPUT_DIR = args.labels + '/' + 'output'
                         cfg.DATALOADER.NUM_WORKERS = 8
                         cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
                         cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
                         cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
                         predictor = DefaultPredictor(cfg)
                                    
             
                         #find peak from red vector, take a certain lag before and get a mask from detectron
                         
                         ledTime,ledMag=signal.find_peaks(stats.zscore(np.abs(np.diff(red))),height=0.5)
                         
                         if len(ledTime)>1:
                             print(outputFile + ' has multiple led times!')
                             ledTime=ledTime[np.argmax(ledMag['peak_heights'])]
    
                         startTime=int(ff+ledTime-backlag)
                         for x in range(startTime):
                              cap.grab()
                                          
                        #the mask
                         succ, img = cap.read() 
                         outputs = predictor(img)    
                         instances = outputs["instances"].to("cpu")
                         bboxs=np.array(instances.pred_boxes.tensor)       
                         cv2.imwrite(args.outputPath + '/' + os.path.basename(video) + '_frame.png',img)
                         if len(bboxs>0):
                             octoMask=np.array(instances.pred_masks[0])
                             cv2.imwrite(args.outputPath + '/' + os.path.basename(video) + '_mask.png',np.uint8(octoMask)*255)
                         writer = h5py.File(outputFile, 'w')
                         writer.attrs.create('video', \
                               outputFile, \
                               dtype=h5py.special_dtype(vlen=str))
                         writer.attrs.create('fps', fps, dtype='float32')
                         writer.create_dataset('red',data=red)
                         writer.attrs.create('version', codeVersion)
                         writer.attrs.create('startTime', startTime)
                         writer.attrs.create('forelag', forelag)
                         writer.attrs.create('fastforward', ff)
                         writer.close()
                    
   
