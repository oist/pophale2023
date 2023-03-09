#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:20:09 2020

@author: sam
"""


import argparse
import os
import cv2
import numpy as np
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


def centroidFromBox(box):
    centroid=np.zeros(2)
    centroid[0]=box[0]+box[1]
    centroid[1]=box[2]+box[3]
    return centroid

if __name__ == '__main__':
    
    p = argparse.ArgumentParser('inference time')
    p.add_argument('--video')
    p.add_argument('--annotation-folder')
    p.add_argument('--output')
    args = p.parse_args()
    

video_filename = args.video
annotationFolder=args.annotation_folder


cfg = get_cfg()
cfg.merge_from_file("/home/sam/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.OUTPUT_DIR = annotationFolder + 'output'
cfg.DATALOADER.NUM_WORKERS = 8
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
predictor = DefaultPredictor(cfg)


cap = cv2.VideoCapture(video_filename)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = float(cap.get(cv2.CAP_PROP_FPS))


avgInt=np.zeros([length,2])
positions=np.zeros([length,2,2])
for frame in range(length):
    succ, img = cap.read()  
    outputs = predictor(img)
    instances = outputs["instances"].to("cpu")
    
    bboxs=np.array(instances.pred_boxes.tensor)
    
    centroids=np.zeros([2,2])
    for ind,box in enumerate(bboxs):
        centroids[ind,:]=centroidFromBox(box)
    positions[frame]=centroids
    masks=np.array(instances.pred_masks)
    
    intInImg=np.zeros([2])
    for ind,mask in enumerate(masks):
       intInImg[ind]=np.mean(img[mask,0]) 
    avgInt[frame]=intInImg
        
    print(frame)