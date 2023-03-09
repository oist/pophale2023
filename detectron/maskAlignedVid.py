#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 09:00:20 2022

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
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
import h5py
import argparse
from tqdm import tqdm
from uuid import uuid1
from scipy import ndimage
import random
import string
#video_filename = '/home/sam/bucket/octopus/clips/shortTestVid.avi'
#annotationFolder='/home/sam/bucket/annotations/octopus_sleep'
#config='/home/s/samuel-reiter/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'


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

if __name__ == '__main__':
    
    p = argparse.ArgumentParser(\
        'run the detectron and align frames', \
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--video')
    p.add_argument('--labels')
    p.add_argument('--config')
    p.add_argument('--src-mask')
    p.add_argument('--startFrame',default=0, type=int)
    p.add_argument('--numFrames', default=0, type=int)
    p.add_argument('--scale-percent',type=float, default=10)
    p.add_argument('--debug-imshow', action='store_true', \
            help='Show videos while processing') 
    p.add_argument('output')
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

        
    # fast forward to start of the action
    cap = cv2.VideoCapture(args.video)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))  #remember fps is rounding down, actual fps is 23.99
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.startFrame)
    num_frames=num_frames-args.startFrame-10 #sometimes not estimated right
    if args.numFrames != 0:
        num_frames=np.minimum(num_frames,args.numFrames)
        
    print('length is ' + str(num_frames) + 'frames')
    
    #get first frame mask for alignment
    succ, img = cap.read() 
    outputs = predictor(img)
    if len(outputs["instances"].pred_masks) != 1:
        print('too many masks detected on frame ' + str(0))
    ff_mask=np.array(outputs["instances"].pred_masks[0].to("cpu"))
    
    #get external mask for additional alignment
    if args.src_mask!=None:
        src_mask=cv2.imread(args.src_mask)
        src_mask=np.squeeze(src_mask[:,:,0])>0
    else:
        src_mask=ff_mask
        

    #resize mask and get rotation parameters
    width = int(src_mask.shape[1] * args.scale_percent / 100)
    height = int(src_mask.shape[0] * args.scale_percent / 100)
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
    #restart the vid
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.startFrame)
    
    #setup output file
    output_dirname = os.path.dirname(args.output)
    output_basename = os.path.basename(args.output)
    suffix = ''.join(random.choice(string.ascii_uppercase + string.digits) \
         for _ in range(4))
    tmp_output_basename = '.' + output_basename + '.' + suffix
    tmp_output = os.path.join(output_dirname, tmp_output_basename)
        
    pattern_file = h5py.File(tmp_output, 'w')
    pattern_dset1 = pattern_file.create_dataset('patterns1', \
                shape=[num_frames,h,w,3], \
                dtype='uint8')
    pattern_dset2 = pattern_file.create_dataset('patterns2', \
                shape=[num_frames,h,w,3], \
                dtype='uint8')
    pattern_file.create_dataset('mask', \
                data=resizedMask, \
                dtype='bool')
    pattern_file.attrs.create('scale_percent',args.scale_percent, \
                        dtype='uint32')
    pattern_file.attrs.create('startFrame',args.startFrame, \
                        dtype='uint32')
    pattern_file.attrs.create('video', \
            args.video, \
            dtype=h5py.special_dtype(vlen=str))
    #run through the video
    goodRun=1
    for ind,frame in tqdm(enumerate(range(num_frames)),total=num_frames): #length
        succ, img = cap.read() 
        if succ:
            outputs = predictor(img)

            if len(outputs["instances"].pred_masks) == 0:   #it will use the previous mask in this case
                print('no masks detected on frame ' + str(ind) + '!')
            elif len(outputs["instances"].pred_masks) > 1:  #it will use the previous mask in this case
                print('too many masks detected on frame ' + str(ind) + '!')
            else:
                mask=np.array(outputs["instances"].pred_masks[0].to("cpu"))
              
                
                t0, t1 = estimateAffine(src_mask, mask)

                t_inv0 = cv2.invertAffineTransform(t0)
                img1 = cv2.warpAffine(img, t_inv0, src_mask.shape[1::-1])
                resized = cv2.resize(img1, dim, interpolation = cv2.INTER_AREA)
                resized=ndimage.rotate(resized, rotation)*maskRotated+1
                resized=resized*invMaskRotated-1
                embeddingData=np.array(resized[maskCrop[0]:maskCrop[1],maskCrop[2]:maskCrop[3]])
                pattern_dset1[ind]=embeddingData[:,:,::-1] #back to rgb
        
                t_inv1 = cv2.invertAffineTransform(t1)
                img2 = cv2.warpAffine(img, t_inv1, src_mask.shape[1::-1])
                resized = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
                resized=ndimage.rotate(resized, rotation)*maskRotated+1
                resized=resized*invMaskRotated-1
                embeddingData=np.array(resized[maskCrop[0]:maskCrop[1],maskCrop[2]:maskCrop[3]])
                pattern_dset2[ind]=embeddingData[:,:,::-1] #back to rgb
        
        else:
            print('something wrong on frame ' + str(ind))
            goodRun=0
            
    pattern_file.close()
    if goodRun==1:
         os.rename(tmp_output, args.output)
    
