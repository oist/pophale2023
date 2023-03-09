#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 13:51:33 2022

@author: sam
"""
import cv2
import pandas as pd
import numpy as np
import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from uuid import uuid1
from scipy import ndimage
import h5py
import glob
from natsort import natsorted


def estimateAffine(src_mask,trg_mask,mode='similarity'):
    cnts, _ = cv2.findContours(src_mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    src_ellipse = cv2.fitEllipse(cnts[0])
    cnts, _ = cv2.findContours(trg_mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    trg_ellipse = cv2.fitEllipse(cnts[0])
    rotation = (src_ellipse[2] - trg_ellipse[2]) / 180. * np.pi
    if mode == 'rotation':
        scale_x = scale_y = 1
    elif mode == 'similarity':
        scale_x = scale_y = (trg_ellipse[1][0] / src_ellipse[1][0] \
                + trg_ellipse[1][1] / src_ellipse[1][1]) / 2
    elif mode == 'full':
        scale_x = trg_ellipse[1][0] / src_ellipse[1][0]
        scale_y = trg_ellipse[1][1] / src_ellipse[1][1]
    else:
        raise RuntimeError('mode %s not in ' \
                '[\'rotation\', \'similarity\', \'full\']' % mode)
    shift_src = src_ellipse[0]
    shift_trg = trg_ellipse[0]
    
    # Compute transformation matrices
    alpha = scale_x * np.cos(rotation)
    beta = scale_y * np.sin(rotation)
    t0 = np.array([[+alpha, +beta,   (1. - alpha) * shift_src[0] \
                                           - beta * shift_src[1] \
                                   + shift_trg[0] - shift_src[0]], \
                   [-beta, +alpha,           beta * shift_src[0] \
                                   + (1. - alpha) * shift_src[1] \
                                   + shift_trg[1] - shift_src[1]]], 'float32')

    alpha = scale_x * np.cos(np.pi + rotation)
    beta = scale_y * np.sin(np.pi + rotation)
    t1 = np.array([[+alpha, +beta,   (1. - alpha) * shift_src[0] \
                                           - beta * shift_src[1] \
                                   + shift_trg[0] - shift_src[0]], \
                   [-beta, +alpha,           beta * shift_src[0] \
                                   + (1. - alpha) * shift_src[1] \
                                   + shift_trg[1] - shift_src[1]]], 'float32')

    return t0, t1



cfg = get_cfg()
cfg.merge_from_file('/home/sam/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
#cfg.OUTPUT_DIR = '/home/sam/bucket/annotations/octo_8k/output'
cfg.OUTPUT_DIR ='/home/sam/bucket/annotations/octo_8k/output'
cfg.DATALOADER.NUM_WORKERS = 8
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
predictor = DefaultPredictor(cfg)
currDataset_name= str(uuid1()) #dummy
segmentation_titles=['octopus']
meta=MetadataCatalog.get(currDataset_name + '_train').set(thing_classes=segmentation_titles)
    
#%%

awakeFolder='/home/sam/bucket/octopus/8k/awake/'
outputFolder='/home/sam/bucket/octopus/8k/awake/extracted_frames/'
frameFrame = pd.read_csv('/home/sam/bucket/octopus/8k/awake/awake_pattern_frames.csv')

#awakeFolder='/home/sam/bucket/octopus/8k/oct_07/tmp'
#outputFolder='/home/sam/bucket/octopus/8k/oct_07/tmp'
#frameFrame = pd.read_csv(awakeFolder + '/awake_frames1.csv')


src_mask='/home/sam/bucket/octopus/8k/oct_05/OCT14862_mask.png'
src_mask = cv2.imread(src_mask,0).astype('bool')
scale_percent=30
#calculating mask rotation stuff
width = int(src_mask.shape[1] * scale_percent / 100)
height = int(src_mask.shape[0] * scale_percent / 100)
dim = (width, height)
resizedMask = cv2.resize(src_mask.astype('uint8'), dim, interpolation = cv2.INTER_AREA)
cnts, _ = cv2.findContours(resizedMask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
genElipse = cv2.fitEllipse(cnts[0])
rotation=genElipse[2]
maskRotated=ndimage.rotate(resizedMask*255, rotation)>128
maskRotated=np.dstack((maskRotated,maskRotated,maskRotated))
invMaskRotated=~maskRotated*127+1
mask_supportX=np.where(np.sum(maskRotated,axis=0)>0)[0]
mask_supportY=np.where(np.sum(maskRotated,axis=1)>0)[0]
maskCrop=[mask_supportY[0], mask_supportY[-1],mask_supportX[0],mask_supportX[-1]]
h,w=[mask_supportY[-1]-mask_supportY[0],mask_supportX[-1]-mask_supportX[0]]

for ind, file in enumerate(frameFrame.columns):

    cap = cv2.VideoCapture(awakeFolder +  file + '.MP4')
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
                outputs = predictor(img)
                if len(outputs['instances'])>0:
                    ff_mask=np.array(outputs["instances"].pred_masks[0].to("cpu"))
                    t0, t1 = estimateAffine(src_mask, ff_mask)

                    t_inv0 = cv2.invertAffineTransform(t0)
                    img1 = cv2.warpAffine(img, t_inv0, src_mask.shape[1::-1])
                    resized = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
                    resized=ndimage.rotate(resized, rotation)*maskRotated+1
                    resized=resized*invMaskRotated-1
                    embeddingData=np.array(resized[maskCrop[0]:maskCrop[1],maskCrop[2]:maskCrop[3]])
                    cv2.imwrite(outputFolder + currOct + '_' + file + '_' + str(f) + '_0.png',embeddingData)
            
                    t_inv1 = cv2.invertAffineTransform(t1)
                    img2 = cv2.warpAffine(img, t_inv1, src_mask.shape[1::-1])
                    resized = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
                    resized=ndimage.rotate(resized, rotation)*maskRotated+1
                    resized=resized*invMaskRotated-1
                    embeddingData=np.array(resized[maskCrop[0]:maskCrop[1],maskCrop[2]:maskCrop[3]])
                    cv2.imwrite(outputFolder  + currOct + '_' + file + '_' + str(f) + '_1.png',embeddingData)
            except:
                print(ind)
                
#%% for images
awakeImgs=glob.glob(awakeFolder + '/*png')

for ind, file in enumerate(awakeImgs):
    nameChange=file.split('.')[0]
    basename=os.path.basename(nameChange)

    img=cv2.imread(file)
    outputs = predictor(img)
    if len(outputs['instances'])>0:
        ff_mask=np.array(outputs["instances"].pred_masks[0].to("cpu"))
        t0, t1 = estimateAffine(src_mask, ff_mask)

        t_inv0 = cv2.invertAffineTransform(t0)
        img1 = cv2.warpAffine(img, t_inv0, src_mask.shape[1::-1])
        resized = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
        resized=ndimage.rotate(resized, rotation)*maskRotated+1
        resized=resized*invMaskRotated-1
        embeddingData=np.array(resized[maskCrop[0]:maskCrop[1],maskCrop[2]:maskCrop[3]])
        cv2.imwrite(outputFolder + '/' + basename  + '_0.png',embeddingData)

        t_inv1 = cv2.invertAffineTransform(t1)
        img2 = cv2.warpAffine(img, t_inv1, src_mask.shape[1::-1])
        resized = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
        resized=ndimage.rotate(resized, rotation)*maskRotated+1
        resized=resized*invMaskRotated-1
        embeddingData=np.array(resized[maskCrop[0]:maskCrop[1],maskCrop[2]:maskCrop[3]])
        cv2.imwrite(outputFolder + '/' + basename  + '_1.png',embeddingData)

                
                
                

#%% After you go through and remove bad rotations

outputFolder='/home/sam/bucket/octopus/8k/awake/good_extracted_frames'
imgList=natsorted(glob.glob(outputFolder + '/*.png'))


smallMask=maskRotated[maskCrop[0]:maskCrop[1],maskCrop[2]:maskCrop[3],0]
imgArray=[]
allOctNum=[]
for i,img in enumerate(imgList):
    basename = os.path.split(img)[1]
    octNum=int(basename.split('_')[1])
    allOctNum.append(octNum)
    currImg=cv2.imread(img)
    imgArray.append(currImg[:,:,::-1])
imgArray=np.array(imgArray)
allOctNum=np.array(allOctNum)
    
pattern_file = h5py.File(outputFolder + '/' + 'collectedFrames.reg', 'w')
pattern_dset = pattern_file.create_dataset('patterns1',data=imgArray)
pattern_file.create_dataset('octNum',data=allOctNum)
pattern_file.close()



   
