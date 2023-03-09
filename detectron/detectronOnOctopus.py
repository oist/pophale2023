#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 11:45:25 2020

@author: sam
"""


import os
import cv2
import numpy as np
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
import h5py
import argparse
from tqdm import tqdm
from uuid import uuid1
import matplotlib.pyplot as plt
#video_filename = '/home/sam/bucket/octopus/clips/shortTestVid.avi'
#annotationFolder='/home/sam/bucket/annotations/octopus_sleep'
#config='/home/s/samuel-reiter/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'



if __name__ == '__main__':
    
    p = argparse.ArgumentParser(\
        'run the detectron. Dont forget about the class number', \
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--video')
    p.add_argument('--labels')
    p.add_argument('--config')
    p.add_argument('--outputPath')
    p.add_argument('--nnInterval', default=100)
    p.add_argument('--tank-partition',action='store_true')
    p.add_argument('--debug-imshow', action='store_true', \
            help='Show videos while processing')
    args = p.parse_args()
    
    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.OUTPUT_DIR = args.labels + '/' + 'output'
    cfg.DATALOADER.NUM_WORKERS = 8
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
    predictor = DefaultPredictor(cfg)
    currDataset_name= str(uuid1()) #dummy
    segmentation_titles=['octopus']
    meta=MetadataCatalog.get(currDataset_name + '_train').set(thing_classes=segmentation_titles)

    
    if not os.path.exists(args.outputPath):
           os.makedirs(args.outputPath)

    #add in check for output file so things dont get overwritten!

    cap = cv2.VideoCapture(args.video)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))  #remember fps is rounding down, actual fps is 23.99
    
    succ, img = cap.read() 
    cap = cv2.VideoCapture(args.video)

    cap = cv2.VideoCapture(args.video)
    
    if args.tank_partition:
        part_mask=np.zeros( img.shape[:-1])
        part_mask[0:1518,0:2012]=0
        part_mask[0:1518,2012::]=1
        part_mask[1518::,2012::]=2
        part_mask[1518::,0:2012]=3
    else:
        part_mask=np.zeros( img.shape[:-1])
    
    numParts=int(np.max(part_mask)+1)
    avgInt=np.zeros([length,numParts])
    com=np.zeros([length,numParts,2])
    mask_size=np.zeros([length,numParts])
    
  
    print('length is ' + str(length) + 'frames')
    
    
    outputFile=args.outputPath + '_' + os.path.basename(args.video) + '.detectronResults'
    writer = h5py.File(outputFile, 'w')
    writer.attrs.create('video', \
          outputFile, \
          dtype=h5py.special_dtype(vlen=str))
    writer.attrs.create('fps', fps, dtype='float32')
    
    for frame in tqdm(range(length),total=length): #length
        succ, img = cap.read() 
        if succ:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        

            if frame % args.nnInterval == 0:
                outputs = predictor(img)
                instances = outputs["instances"].to("cpu")
                bboxs=np.array(instances.pred_boxes.tensor)
            
            goodPart=[]
            for part in range(bboxs.shape[0]):
                curr_part=int(part_mask[int(bboxs[part,1]),int(bboxs[part,0])])
                octoMask=np.array(instances.pred_masks[part])
                
                if len(np.unique(part_mask[octoMask]))==1: #prevent double counts
                    goodPart.append(part)  
                    avgInt[frame,curr_part]=np.mean(gray[octoMask])
                    mask_size[frame,curr_part]=np.sum(octoMask.astype('float32'))
                    com[frame,curr_part,:]=[bboxs[part,2]-bboxs[part,0],bboxs[part,3]-bboxs[part,1] ]
            if frame % args.nnInterval == 0:
                if args.debug_imshow:
    
                    v = Visualizer(
                        img[:, :, ::-1],
                        metadata=meta, 
                        scale=1., 
                        instance_mode=ColorMode.IMAGE
                        )
                    v = v.draw_instance_predictions(instances[goodPart])
                    result = v.get_image()[:, :, ::-1]
                    plt.imshow(result)
                    plt.show()
        else:
            print('video couldnt read the frame')
            writer.create_dataset('avg_int', data=avgInt)
            writer.close()
            break

        
    if not os.path.exists(args.outputPath):      os.makedir(args.outputPath)
    print('ran through ok')
   
    writer.create_dataset('avg_int', data=avgInt)
    writer.create_dataset('com', data=com)
    writer.create_dataset('mask_size', data=mask_size)
    writer.close()
    print('wrote file')
    
    
