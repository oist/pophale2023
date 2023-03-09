#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 12:14:14 2021

@author: sam

"""
import cv2
import numpy as np
import h5py
import argparse
import glob
from tqdm import tqdm
from scipy import signal, stats, spatial

# put all the avi videos you want in a folder, that is video path. 
#You can make output path the same as videopath if you have write access


def detect_points(gray, mask, det_max_points):
 
    gray=gray*mask;
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=sift_thresh)
    kp = sift.detect(gray,None)
    points = np.array([keypoint.pt for keypoint in kp], dtype='float32')
  #  print('num points is ' + str(points.shape[0]))
  
    # find ~evenly spaced centers
    
    minXY = np.amin(points,axis=0).astype(int)
    maxXY = np.amax(points,axis=0).astype(int)
    gridPts=np.sqrt(min(det_max_points, points.shape[0])).astype(int)
    xgrid = np.linspace(minXY[0], maxXY[0], gridPts)
    ygrid = np.linspace(minXY[1], maxXY[1], gridPts)
    xv, yv = np.meshgrid(xgrid, ygrid)
    xv=xv.astype(int)
    yv=yv.astype(int)
    gridList=np.array(list(zip(xv.ravel(), yv.ravel())))
    goodPt=[]
    
    for pt in gridList:
        goodPt.append(gray[pt[1],pt[0]])
    goodPt=np.array(goodPt)
    gridList=list(gridList[goodPt.astype(bool)])
    tree = spatial.KDTree(points)
    distance, index = tree.query(gridList,k=1)
    points = np.unique(points[index],axis=0)
    return points





if __name__ == '__main__':
    
    p = argparse.ArgumentParser(\
        'movementCalc',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--vidPath')
    p.add_argument('--outputPath')
    p.add_argument('--at', action='store_true')
    
    
    max_points=500
    sift_thresh=0.005
    numFrames=60
    backlag=30
    winsize=512
    
    args = p.parse_args()
    
    vidList=sorted(glob.glob(args.vidPath + '/*.avi'))
    
    #add in check for output file so things dont get overwritten!
    for filename in vidList:

       #maskName=filename + '_mask.png'
       # maskName=filename + '_eye.png'
        maskName=filename + '_breath.png'
        output=filename + '_movement'
        hitTimeFile= filename + '.findHit'
        cap = cv2.VideoCapture(filename)   
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))

        if args.at: #THe arousal threshold calculation, where I just consider a clip around the hit time
            f=h5py.File(hitTimeFile,'r')
            ff=int(f.attrs['fastforward'])
            red=f['red'][:]
            ledTime,ledMag=signal.find_peaks(stats.zscore(np.abs(np.diff(red))),height=0.5)
                             
            if len(ledTime)>1:
                print(output + ' has multiple led times!')
                ledTime=ledTime[np.argmax(ledMag['peak_heights'])]
        
            startTime=int(ff+ledTime-backlag)
        else:
            startTime=0
            numFrames=length
            
        mask=cv2.imread(maskName)
        m=np.squeeze(mask[:,:,0]>0)

        for x in range(startTime):
            cap.grab()
        
        succ, img = cap.read()
        prev_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        points = detect_points(prev_gray,m, max_points)
        original_points = points
        
        mag=[]
        magRel=[]
        for x in tqdm(range(0,numFrames-1)):
            succ, img = cap.read()
            if succ:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                scattered_status = np.empty((len(points), ), 'uint8')
                next_points = np.empty_like(points)
                cv2.calcOpticalFlowPyrLK(prev_gray,gray, \
                        points, \
                        next_points, \
                        scattered_status,
                        winSize=(winsize, winsize))
              
                scattered_status = scattered_status.astype('bool')
             
                points = \
                        points[scattered_status]
                next_points = \
                        next_points[scattered_status]
            
                com=np.mean(points,axis=0)
                nextCOM=np.mean(next_points,axis=0)
                avgMovement=nextCOM-com
                xDiff=next_points[:,0]-points[:,0]
                yDiff=next_points[:,1]-points[:,1]
                fm= np.mean(np.sqrt(np.square(xDiff)+np.square(yDiff)))
                
                xDiffRel=next_points[:,0]-points[:,0]-avgMovement[0]
                yDiffRel=next_points[:,1]-points[:,1]-avgMovement[1]
                
                fmRel= np.mean(np.sqrt(np.square(xDiffRel)+np.square(yDiffRel)))
                mag.append(fm)
                magRel.append(fmRel)
                prev_gray = gray
                points=next_points
             
        mag=np.array(mag)
        magRel=np.array(magRel)
        
        move_file = h5py.File(output, 'w')
        move_file.create_dataset('movement', \
                        data=mag, \
                        dtype='float32')
        move_file.create_dataset('movRel', \
                        data=magRel, \
                        dtype='float32')
        move_file.close()

        
     
