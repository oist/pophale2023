#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 15:39:25 2021

@author: sam
"""

import argparse
import numpy as np
from tqdm import tqdm
import os.path
import h5py
import buffer_handle 
import cv2
import moving_least_squares
from scipy import ndimage


def warp_image(img, p, q, alpha, grid_size):
    identity_maps = np.dstack(np.meshgrid( \
            *tuple(np.arange(0, s + grid_size, grid_size) \
                  for s in img.shape[1::-1]))).astype('float32')
    coords = identity_maps.reshape((-1, 2))
    mapped_coords = moving_least_squares.similarity( \
            p, q, coords, alpha=alpha)
    maps = mapped_coords.reshape(identity_maps.shape)
    t = np.array([[grid_size, 0, 0], [0, grid_size, 0]], 'float32')
    maps = cv2.warpAffine(maps, t, img.shape[1::-1])
    return cv2.remap(img, maps, None, interpolation=cv2.INTER_LINEAR)


def estimateAffine(src_mask,trg_mask,mode='similarity'):
    cnts, _ = cv2.findContours(src_mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    src_ellipse = cv2.fitEllipse(cnts[0])
    cnts, _ = cv2.findContours(trg_mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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
    p = argparse.ArgumentParser('register pattern')
    p.add_argument('--output')
    p.add_argument('registration', help='Registration hd5 file')
    p.add_argument('--src_mask')  
    p.add_argument('--scale-percent',type=float, default=10)
    p.add_argument('--mls-alpha',type=float, default=11)
    p.add_argument('--grid-size',type=float, default=4)
    p.add_argument('--ht-flip', type=int,default=0)
    args = p.parse_args()
    

    # Read the registration file
    registration = h5py.File(args.registration, 'r')
    maps_dset = registration['inv_maps']
    reg_num_frames = maps_dset.shape[0]
    fps = registration.attrs['fps']
    startFrame = registration.attrs['startFrame']
    rel_video_filename = registration.attrs['video']
    basepath = os.path.split(args.registration)[0]
    
    # Transformation to scale maps up
    scale_maps_t = np.array([[maps_dset.attrs['grid_size'], 0., 0.], \
                             [0., maps_dset.attrs['grid_size'], 0.]], 'float32')
    # Open video
   # video_filename = basepath + '/' + rel_video_filename 
    video_filename =  args.registration[:-4] + '.MP4'
    mask_filename = args.registration[:-4] + '_mask.png'
 
    
    #fast forward to first frame of video that gets registered
    video = cv2.VideoCapture(video_filename)
    video.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
    
    fps = float(video.get(cv2.CAP_PROP_FPS))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
   
    trg_mask=cv2.imread(mask_filename)
    trg_mask=np.squeeze(trg_mask[:,:,0])>0
    
    if args.src_mask!=None:
        src_mask=cv2.imread(args.src_mask)
        src_mask=np.squeeze(src_mask[:,:,0])>0
    else:
        src_mask=trg_mask
        
    mask_supportX=np.where(np.sum(src_mask,axis=0)>0)[0]
    mask_supportY=np.where(np.sum(src_mask,axis=1)>0)[0]
    maskCrop=[mask_supportY[0], mask_supportY[-1],mask_supportX[0],mask_supportX[-1]]

    croppedMask=src_mask[maskCrop[0]:maskCrop[1],maskCrop[2]:maskCrop[3]]
    h,w=croppedMask.shape

    width = int(croppedMask.shape[1] * args.scale_percent / 100)
    height = int(croppedMask.shape[0] * args.scale_percent / 100)
    dim = (width, height)
    resizedMask = cv2.resize(croppedMask.astype('uint8'), dim, interpolation = cv2.INTER_AREA)
    
    #get affine transform
    t0, t1 = estimateAffine(src_mask, trg_mask)
    if args.ht_flip==0:
        t_inv = cv2.invertAffineTransform(t0)
    else:
        t_inv  = cv2.invertAffineTransform(t1)
        

    pattern_file = h5py.File(args.output, 'w')
    pattern_dset = pattern_file.create_dataset('patterns', \
                shape=[reg_num_frames,height,width,3], \
                dtype='uint8')
    pattern_file.create_dataset('mask', \
                data=resizedMask, \
                dtype='bool')
    pattern_file.create_dataset('mask_crop', \
                data=maskCrop, \
                dtype='uint32')
    pattern_file.attrs.create('scale_percent',args.scale_percent, \
                        dtype='uint32')
    pattern_file.attrs.create('startFrame',startFrame, \
                        dtype='uint32')

    def create_video_iter():
        for _ in range(reg_num_frames):
            succ, img = video.read()
            if not succ:
                raise RuntimeError('Video finished too early')
            yield img
    video_iter = create_video_iter()
    def create_iter_maps():
        with buffer_handle.Reader(maps_dset, maps_dset.chunks[0]) \
                as maps_reader:
            while not maps_reader.done():
                yield maps_reader.read()
    maps_iter = create_iter_maps()
    
    for ind, [frame, maps] in tqdm(enumerate(zip(video_iter, maps_iter)), \
            total=reg_num_frames, \
            desc='calculating intensity'):

        maps = cv2.warpAffine(maps, scale_maps_t, frame.shape[1::-1])
        
        
       # import pdb;pdb.set_trace()
        warped_mask = cv2.remap(trg_mask.astype('uint8'), maps, None, \
                interpolation=cv2.INTER_LINEAR) 
            
        t0, t1 = estimateAffine(trg_mask,warped_mask>0,mode='rotation')
        t0_inv = cv2.invertAffineTransform(t0)
        
            #replace with a rigid body transform estimated 
            #from the mask inv_mapped back into frame N from frame 1
        frame = cv2.warpAffine(frame, t0_inv, frame.shape[1::-1])
        frame = cv2.warpAffine(frame, t_inv, frame.shape[1::-1])

        cropped_frame=frame[maskCrop[0]:maskCrop[1],maskCrop[2]:maskCrop[3]]
        resized = cv2.resize(cropped_frame, dim, interpolation = cv2.INTER_AREA)
        pattern_dset[ind]=resized[:,:,::-1] #back to rgb
          
    video.release()
    registration.close()
    
    
    #next rotate, mask, save
    dataMat=pattern_dset[:]
    mask=resizedMask
 
    cnts, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    genElipse = cv2.fitEllipse(cnts[0])
    newData=[]
    maskRotated=ndimage.rotate(mask*255, genElipse[2])>128
    weight=np.sum(maskRotated,axis=0)
    mantleLength=int(np.floor(len(weight)/2))
    rotation=genElipse[2]
    maskRotated=ndimage.rotate(mask*255, rotation)>128
    maskRotated=np.dstack((maskRotated,maskRotated,maskRotated))
    invMaskRotated=~maskRotated*127+1
    for data in dataMat:
        newData.append(ndimage.rotate(data, rotation))
    newData=np.array(newData)
    newData=newData*maskRotated+1
    newData=newData*invMaskRotated-1
    embeddingData=np.array(newData)
      
    pattern_file.create_dataset('embeddingData', data=embeddingData,dtype='uint8')
    print('wrote embeddingData')
    pattern_file.close()
