
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 15:30:40 2019

@author: reiters
"""

import cv2
import scipy.ndimage
import numpy as np
import moving_least_squares
import libreg.affine_registration
import os.path
import sklearn.linear_model
import scipy.fftpack
from PIL import Image
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
    
  #  q_mapped=maps_inv[np.round(q[:,1]).astype('int'),np.round(q[:,0]).astype('int'),:]
  #  p_mapped=maps[np.round(p[:,1]).astype('int'),np.round(p[:,0]).astype('int'),:]
    
    return cv2.remap(img, maps, None, interpolation=cv2.INTER_LINEAR), maps





#%% Get the list of images to stitch and their initial matchfeatures files
dirname='/home/sam/bucket/octopus/high_res_top_view/oct_32'
src_ind = 6

trg_img = '/home/sam/bucket/octopus/high_res_top_view/oct_32/awake/vlcsnap-2021-09-14-16h04m46s299.png'

#%%

basepath = os.path.split(dirname)[0] 
basename = os.path.split(dirname)[1]

chunktimes = pd.read_csv(dirname + '/' + basename + '.chunking')
as_videos=chunktimes['date'][:].to_numpy()
as_times=chunktimes['time'][:].to_numpy()
masks=[]
images=[]
for ind,t in enumerate(as_times):
    images.append(dirname + '/' + as_videos[ind] + '_' + t + '.avi.png')
    masks.append(dirname + '/' + as_videos[ind] + '_' + t + '_mask.png')
    
src_mf = cv2.imread(images[src_ind],0)
src_mask = cv2.imread(masks[src_ind],0).astype('bool')
trg_mf = cv2.imread(trg_img,0)


    #%% initial point correspondences
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
    
  #  print(event)
fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, ax))
fig.canvas.mpl_connect('button_release_event', lambda event: onrelease(event, ax))

tS=tempPtsSrc
tT=tempPtsTrg

# tempPtsTrg=tT
# tempPtsSrc=tS

    #%% removing bad points round 1

tempPtsSrc=tempPtsSrc.astype('float32')
tempPtsTrg=tempPtsTrg.astype('float32')
mls_alpha=11
grid_size=4

trg_mf_warped,maps = warp_image(trg_mf, \
         tempPtsSrc, tempPtsTrg, mls_alpha, grid_size)
    
fig, ax = plt.subplots()
C = np.dstack((src_mf,trg_mf_warped,src_mf))
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
        C = np.dstack((src_mf.astype('uint8'),trg_mf_warped.astype('uint8'),src_mf.astype('uint8')))
        ax.imshow(C)
        ax.scatter(tempPtsSrc[:,0],tempPtsSrc[:,1], 50, picker=True,color='r')
        ax.set_xlim(xmin=xl[0],xmax=xl[1])
        ax.set_ylim(ymin=yl[0],ymax=yl[1])
        plt.draw()
    
  #  print(event)
fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, ax))
fig.canvas.mpl_connect('button_release_event', lambda event: onrelease(event, ax))
    
    
    #%% after manual refinement

coarse_src_coords=tempPtsSrc.astype('float32')
coarse_trg_coords=tempPtsTrg.astype('float32')

# Warp src_mf with the coarse grid transformation
trg_mf_warped,maps = warp_image(trg_mf, \
        coarse_src_coords, coarse_trg_coords, mls_alpha, 32)



C = np.dstack((src_mf.astype('uint8'),trg_mf_warped.astype('uint8'),src_mf.astype('uint8')))
plt.figure()
plt.imshow(C)



feat_file=h5py.File(dirname + '/' + 'matchFeatures_' + str(src_ind) + '_' + trg_img.split('/')[-1][:-4],'w')
feat_file.create_dataset('feat2', data=coarse_src_coords)
feat_file.create_dataset('feat1', data=coarse_trg_coords)
feat_file.attrs.create('src_mask',masks[src_ind], \
                        dtype=h5py.special_dtype(vlen=str))
feat_file.close()


#%%

mls_alpha=3.0
feat_file=h5py.File(dirname + '/' + 'matchFeatures_' + str(src_ind) + '_' + trg_img.split('/')[-1][:-4],'r')
tempPtsSrc=feat_file['feat2'][:].astype('float32')
tempPtsTrg=feat_file['feat1'][:].astype('float32')
feat_file.close()

trg_mf_warped,maps = warp_image(trg_mf, \
        tempPtsSrc, tempPtsTrg, mls_alpha, 32)

    

C = np.dstack((src_mf.astype('uint8'),trg_mf_warped.astype('uint8'),src_mf.astype('uint8')))
plt.figure()
plt.imshow(C)



