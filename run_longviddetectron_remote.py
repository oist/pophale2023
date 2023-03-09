#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 17:21:09 2021

@author: sam
"""

import os
import argparse
from pipeline_utils import submit_job
import glob
from natsort import natsorted

if __name__ == '__main__':

    p = argparse.ArgumentParser('Running pipeline on remote', add_help=False)
    p.add_argument('--basePath', default="/bucket/.deigo/ReiterU")  #remember /bucket/.deigo for saion!
    p.add_argument('--videoPath', default="")
    
    args = p.parse_args()
 
    config= '~/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
    labels='/bucket/.deigo/ReiterU/annotations/octo_apartment'

    
    videoList=sorted(glob.glob(args.basePath + '/' + args.videoPath + '/*.avi'))
    jobID=[]
    
    for video in videoList:
        videoName= os.path.split(video)[1]
        clipFolder=args.basePath + '/' +args.videoPath + '/' + videoName[:-4]
 
        clips=natsorted(glob.glob( clipFolder + '/*.avi'))
        outputFolder='/hpacquire/users/samuel-reiter/' + videoName[:-4]
                
        jobID=[]
        for ind, clip in enumerate(clips):
             
             if not os.path.exists(video[:-4] + '/' + videoName[:-4]  + str(ind) + '.avi.meanInt'):
                 slurm_args = ['--output', outputFolder + '_' + str(ind) + '.log', \
                          '--job-name', 'chunked_detectron',
                          '-p', 'powernv',
                          '-t', '0-24',
                          '-c', '32',
                          '--mem', '32G',
                          '--gres', 'gpu:1',
                          '-x','saion-power01,saion-power02']
                     
                  job_args = ['~/octoSleep/detectron/detectronOnOctopus.py', \
                                    '--video', clip, \
                                    '--labels', labels, \
                                    '--config', config, \
                                    '--outputPath', outputFolder, \
                                    '--tank-partition']
                     
                                     
                 jobID.append(submit_job(job_args, slurm_args, 'srun'))
                 print('>>> Submitted {}: {}'.format(jobID[ind], clip))
             else:
                 jobID.append([])

  
