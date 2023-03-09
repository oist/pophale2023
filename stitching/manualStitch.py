
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 15:30:40 2019

@author: reiters
"""

import cv2
import numpy as np
import moving_least_squares
import os.path
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import glob
import time

def norm(data):
        with np.errstate(invalid='ignore'):
            return (data - np.mean(data)) / np.std(data)

def get_patch_at(img, pos, patch_size):
        x, y = pos
        t = np.array([[1., 0., patch_size - x], \
                      [0., 1., patch_size - y]], 'float32')
        patch = cv2.warpAffine(img, t, \
                (int(2 * patch_size + .5), int(2 * patch_size + .5)))
        return norm(patch)

def warp_image(img, p, q, alpha, grid_size):
    #img is trg
    #p is src
    #q is trg
    
    identity_maps = np.dstack(np.meshgrid( \
            *tuple(np.arange(0, s + grid_size, grid_size) \
                  for s in img.shape[1::-1]))).astype('float32')
    coords = identity_maps.reshape((-1, 2))
    mapped_coords = moving_least_squares.similarity( \
            p, q, coords, alpha=alpha)
    maps = mapped_coords.reshape(identity_maps.shape)
    
    mapped_coords_inv = moving_least_squares.similarity( \
            q, p, coords, alpha=alpha)
    maps_inv = mapped_coords_inv.reshape(identity_maps.shape)
    
    t = np.array([[grid_size, 0, 0], [0, grid_size, 0]], 'float32')
    maps = cv2.warpAffine(maps, t, img.shape[1::-1])
    maps_inv = cv2.warpAffine(maps_inv, t, img.shape[1::-1])
    
   
    return cv2.remap(img, maps, None, interpolation=cv2.INTER_LINEAR), maps


#%% Get the list of images to stitch and their initial matchfeatures files

#Put the 2 images in a folder, this is the folder address
dirname='/home/sam/bucket/octopus/8k/pattern_matching/'

#these are the 2 image names. Maybe have file 1 always be wake, file 2 sleep.
file1='oct34_awake_cam0_2021-06-18-15-06-18_2.11.28.png' #awake
file2='oct34_asleep_cam0_2021-06-18-15-06-18_1.39.38.png' #asleep


src_mf=cv2.imread(dirname + '/' + file1,0)
trg_mf=cv2.imread(dirname + '/' + file2,0)

    #%% initial point correspondences, pick at least 3
 
imgMontage=np.concatenate((src_mf,trg_mf),axis=1)

tempPtsSrc=np.array([[0,0]]).astype('float32')
tempPtsTrg=np.array([[0,0]]).astype('float32')

fig, ax = plt.subplots()
plt.imshow(imgMontage)
xl=ax.get_xlim()
yl=ax.get_ylim()
ax.set_xlim(xmin=xl[0],xmax=xl[1])
ax.set_ylim(ymin=yl[0],ymax=yl[1])


def onclick(event, ax):
    ax.time_onclick = time.time()
    
    
def onrelease(event, ax):
    
    MAX_CLICK_LENGTH = 0.2 # in seconds; anything longer is a drag motion
    
    global tempPtsSrc
    global tempPtsTrg
    global maps

    currPt=np.array([[event.xdata,event.ydata]])
    
# Only clicks inside this axis are valid.
    if event.inaxes == ax:
        if event.button == 1 and ((time.time() - ax.time_onclick) < MAX_CLICK_LENGTH):
            
            print(currPt[0][0])
            if currPt[0][0]<src_mf.shape[1]:
                tempPtsSrc=np.concatenate((tempPtsSrc,currPt),axis=0)
            else:
                tempPtsTrg=np.concatenate((tempPtsTrg, \
                    np.array([[currPt[0][0]-src_mf.shape[1],currPt[0][1]]])),axis=0)
         
        elif event.button == 3: #remove a pt pair
            minPtsNum=np.min([tempPtsTrg.shape[0],tempPtsSrc.shape[0]])
            maxPtsNum=np.max([tempPtsTrg.shape[0],tempPtsSrc.shape[0]])
            dist = (currPt - tempPtsSrc)**2
            dist = np.sum(dist, axis=1)
            dist = np.sqrt(dist)
            minDist_ind=np.argmin(dist)
            goodInd=np.zeros(maxPtsNum,dtype='bool')
            goodInd[0:minPtsNum]=1
            goodInd[minDist_ind]=0
            tempPtsSrc=tempPtsSrc[goodInd[0:tempPtsSrc.shape[0]],:]
            tempPtsTrg=tempPtsTrg[goodInd[0:tempPtsTrg.shape[0]],:]

        minPtsNum=np.min([tempPtsTrg.shape[0],tempPtsSrc.shape[0]])
        xl=ax.get_xlim()
        yl=ax.get_ylim()
        ax.clear()
        plt.imshow(imgMontage)
        ax.scatter(tempPtsSrc[:,0],tempPtsSrc[:,1], 50, picker=True,color='r')
        ax.scatter(tempPtsTrg[:,0]+src_mf.shape[1],tempPtsTrg[:,1], 50, picker=True,color='r')
        ax.set_xlim(xmin=xl[0],xmax=xl[1])
        ax.set_ylim(ymin=yl[0],ymax=yl[1])
        plt.draw()
    

fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, ax))
fig.canvas.mpl_connect('button_release_event', lambda event: onrelease(event, ax))

tS=tempPtsSrc
tT=tempPtsTrg

mls_alpha=11
grid_size=4
srcMult=1
trgMult=1
contrastMult=1

    #%% fine correspondance. Left click purple, right click green, middle click undo


srcMult=1  #turn down if the purple is too bright
trgMult=1     #turn down if the green is too bright
contrastMult=1.1 #turn down if the image is too bright



tempPtsSrc=tempPtsSrc.astype('float32')
tempPtsTrg=tempPtsTrg.astype('float32')

trg_mf_warped,maps = warp_image(trg_mf, \
          tempPtsSrc, tempPtsTrg, mls_alpha, grid_size)
    
fig, ax = plt.subplots()
D = np.uint8(np.dstack((src_mf*srcMult,trg_mf_warped*trgMult,src_mf*srcMult)))
C = cv2.addWeighted( D, contrastMult, D, 0, 0)

ax.imshow(C)
ax.scatter(tempPtsSrc[:,0],tempPtsSrc[:,1], 50, picker=True,color='r')
xl=ax.get_xlim()
yl=ax.get_ylim()
ax.set_xlim(xmin=xl[0],xmax=xl[1])
ax.set_ylim(ymin=yl[0],ymax=yl[1])


def onclick(event, ax):
    ax.time_onclick = time.time()
    
    
def onrelease(event, ax):
    
    MAX_CLICK_LENGTH = 0.2 # in seconds; anything longer is a drag motion
    
    global tempPtsSrc
    global tempPtsTrg
    global maps

    currPt=np.array([[event.xdata,event.ydata]])
    
# Only clicks inside this axis are valid.
    if event.inaxes == ax:
        if event.button == 1 and ((time.time() - ax.time_onclick) < MAX_CLICK_LENGTH):
            tempPtsSrc=np.concatenate((tempPtsSrc,currPt),axis=0)
         
    
        elif event.button == 3: #add a mapped trg point (magenta)
            p_mapped=maps[np.round(event.ydata).astype('int'),np.round(event.xdata).astype('int'),:]
            tempPtsTrg=np.concatenate((tempPtsTrg,np.array([p_mapped])),axis=0)
         
        elif event.button == 2: #remove a pt pair
            minPtsNum=np.min([tempPtsTrg.shape[0],tempPtsSrc.shape[0]])
            maxPtsNum=np.max([tempPtsTrg.shape[0],tempPtsSrc.shape[0]])
            dist = (currPt - tempPtsSrc)**2
            dist = np.sum(dist, axis=1)
            dist = np.sqrt(dist)
            minDist_ind=np.argmin(dist)
            goodInd=np.zeros(maxPtsNum,dtype='bool')
            goodInd[0:minPtsNum]=1
            goodInd[minDist_ind]=0
            tempPtsSrc=tempPtsSrc[goodInd[0:tempPtsSrc.shape[0]],:]
            tempPtsTrg=tempPtsTrg[goodInd[0:tempPtsTrg.shape[0]],:]

        minPtsNum=np.min([tempPtsTrg.shape[0],tempPtsSrc.shape[0]])
        xl=ax.get_xlim()
        yl=ax.get_ylim()
        ax.clear()
        trg_mf_warped, maps = warp_image(trg_mf, \
                tempPtsSrc[0:minPtsNum,:].astype('float32'), tempPtsTrg[0:minPtsNum,:].astype('float32'), mls_alpha, grid_size)
        D = np.uint8(np.dstack((src_mf*srcMult,trg_mf_warped*trgMult,src_mf*srcMult)))
        C = cv2.addWeighted( D, contrastMult, D, 0, 0)
        ax.imshow(C)
        ax.scatter(tempPtsSrc[:,0],tempPtsSrc[:,1], 50, picker=True,color='r')
        ax.set_xlim(xmin=xl[0],xmax=xl[1])
        ax.set_ylim(ymin=yl[0],ymax=yl[1])
        plt.draw()
    
  #  print(event)
fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, ax))
fig.canvas.mpl_connect('button_release_event', lambda event: onrelease(event, ax))
    
    
    #%% run after manual refinement to save data
mls_alpha=11
grid_size=4

ptNum=np.min([len(tempPtsSrc),len(tempPtsTrg)])
coarse_src_coords=tempPtsSrc[0:ptNum].astype('float32')
coarse_trg_coords=tempPtsTrg[0:ptNum].astype('float32')

# Warp src_mf with the coarse grid transformation
trg_mfFull=cv2.imread(dirname + '/' + file2)
#src_mf=cv2.imread(dirname + '/' + file1)

trg_mf_warped,maps = warp_image(trg_mfFull, \
         coarse_src_coords, coarse_trg_coords, mls_alpha, 32)
    

cv2.imwrite(dirname + '/' + file2 + '_trgImgWarped.png',trg_mf_warped)

feat_file=h5py.File(dirname + '/' + file2 + '_matchFeatures','w')
feat_file.create_dataset('tempPtsSrc', data=coarse_src_coords)
feat_file.create_dataset('tempPtsTrg', data=coarse_trg_coords)
feat_file.attrs.create('srcName',  dirname + '/' + file1, dtype=h5py.special_dtype(vlen=str))
feat_file.attrs.create('trgName',  dirname + '/' + file2, dtype=h5py.special_dtype(vlen=str))

feat_file.close()

#%% load saved progress
feat_file=h5py.File(dirname + '/' + file2 + '_matchFeatures','r')
tempPtsSrc=feat_file['tempPtsSrc'][:].astype('float32')
tempPtsTrg=feat_file['tempPtsTrg'][:].astype('float32')


src_mf=cv2.imread(dirname + '/' + file1,0)
trg_mf=cv2.imread(dirname + '/' + file2,0)
feat_file.close()


#%% For loading things up
# mls_alpha=3.0
# feat_file=h5py.File(dirname + '/' + 'matchFeatures_Test1','r')
# tempPtsSrc=feat_file['feat2'][:].astype('float32')
# tempPtsTrg=feat_file['feat1'][:].astype('float32')
# feat_file.close()

# trg_mf_warped,maps = warp_image(trg_mf, \
#         tempPtsSrc, tempPtsTrg, mls_alpha, 32)

    

# D = np.uint8(np.dstack((src_mf*srcMult,trg_mf_warped*trgMult,src_mf*srcMult)))
# C = cv2.addWeighted( D, contrastMult, D, 0, 0)
# plt.figure()
# plt.imshow(C)



