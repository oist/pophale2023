#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 08:32:28 2020

@author: sam
"""

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.logger import setup_logger
setup_logger()

import os
import numpy as np
import cv2
import random
import pandas as pd
from tqdm import tqdm
import urllib
import PIL.Image as Image
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.structures import BoxMode
import codecs, json 
from os.path import exists

currDataset_name='quadCam'
annotationFolder='/home/sam/bucket/annotations/'
segmentation_titles=['octopus']
boundingBox_titles=['fromSeg']
outputFolder = annotationFolder + currDataset_name + '/'
annotation_df = pd.read_json(outputFolder + 'annotation.json') 

meta=MetadataCatalog.get(currDataset_name).set(thing_classes=segmentation_titles)
#go from labelbox to coco format
dataset = []

for index, row in tqdm(annotation_df.iterrows(), total=annotation_df.shape[0]):
    currLabel=annotation_df['External ID'][index]
    currID=annotation_df['ID'][index]
    currImageFolder = annotation_df['Project Name'][index]
    image_name=  currLabel
    nonemptyDict=annotation_df['Label'][index]
    if not nonemptyDict:
        print(currLabel +' dictionary is empty')
    else:
        path_to_image=annotation_df['Labeled Data'][index]
        img=np.array(Image.open(urllib.request.urlopen(path_to_image)))
        if not exists(outputFolder + image_name):
            cv2.imwrite(outputFolder + image_name,img[:,:,::-1])
        
        annotations_LB = annotation_df['Label'][index]['objects']
        
        data = {}
        data['file_name'] =  outputFolder + image_name
        data['width'] = img.shape[1]
        data['height'] = img.shape[0]
        data['image_id'] = index
        
        annotations_coco=[]
              

        for bbox_seg_pair in range(len(segmentation_titles)):
        #get segmentations
            allSeg=np.zeros((img.shape[0],img.shape[1]))
            
            for an in annotations_LB:
                title=an['title']
                #if title==segmentation_titles[bbox_seg_pair]:
          
                path_to_seg=an['instanceURI']
                seg=Image.open(urllib.request.urlopen(path_to_seg))
                seg = np.array(seg.convert('L'))
                cv2.imwrite(outputFolder + currLabel[:-4] + '_' + title + '.png',seg)
                allSeg=allSeg+seg
            
            #get bounding boxes and associate segmentations
            
            #first take care of the possibility of no bounding box, make one from the semantic segmentation. 
            #For now only 1 object per class can not have a bbox
                if boundingBox_titles[bbox_seg_pair]=='fromSeg':
                    contours, hierarchy = cv2.findContours(seg.astype('uint8'), \
                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[-2:]
                
                    if contours:
                        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x)) #take the largest
                        c=cntsSorted[-1]
                        contours_poly = cv2.approxPolyDP(c, 3, True)
                        boundRect = cv2.boundingRect(contours_poly)
                        obj={}
                        width = boundRect[2]
                        height = boundRect[3]
                        xPos=boundRect[0]
                        yPos=boundRect[1]
                        obj['bbox']=[float(xPos), float(yPos), float(width), float(height)]
                        obj["bbox_mode"] = BoxMode.XYWH_ABS
                        obj['category_id'] = bbox_seg_pair
                        obj['iscrowd'] = 0
                        obj['segmentation']=[np.squeeze(c).ravel().tolist()]
                        annotations_coco.append(obj)
                    
            #if not then find the right bounding box
                else:
                    for an in annotations_LB:
                        if an['title']==boundingBox_titles[bbox_seg_pair]:
                          obj={}
                          width = an['bbox']['width']
                          height = an['bbox']['height']
                          xPos=an['bbox']['left']
                          yPos=an['bbox']['top']
                          obj['bbox']=[np.float(xPos), np.float(yPos), np.float(width), np.float(height)]
                          obj["bbox_mode"] = BoxMode.XYWH_ABS
                          obj['category_id'] = bbox_seg_pair
                          obj['iscrowd'] = 0
                          instance_seg=np.zeros((img.shape[0],img.shape[1]))
                          instance_seg[an['bbox']['top']:an['bbox']['top']+height, an['bbox']['left']:an['bbox']['left']+width]= \
                              allSeg[an['bbox']['top']:an['bbox']['top']+height, an['bbox']['left']:an['bbox']['left']+width]
                          contours, hierarchy = cv2.findContours(instance_seg.astype('uint8'), \
                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[-2:]
                          if contours:
                              cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
                              obj['segmentation']=[np.squeeze(cntsSorted[-1]).ravel().tolist()]
                              annotations_coco.append(obj)
        data['annotations']=annotations_coco
        dataset.append(data)
json.dump(dataset, codecs.open(outputFolder + 'annotations_coco.json', 'w', \
            encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format
