#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri May 29 15:46:05 2020

@author: sam
"""


import os
import argparse
from pipeline_utils import submit_job
from pipeline_utils import check_filedeps
import configparser as ConfigParser
import glob
import h5py
import numpy as np
if __name__ == '__main__':

    p = argparse.ArgumentParser('Running pipeline on remote', add_help=False)
    p.add_argument('--registration-slurm', default="")
    p.add_argument('--registration-args', default="")
    p.add_argument('--registration-partition', default="")   
    p.add_argument('--mpicmd', default='mpiexec')
    p.add_argument('--rerun-registration', default="False",type=str)
    p.add_argument('--skip-registration', default="False",type=str)
    p.add_argument('--data-dir', help='Input video')
    args = p.parse_args()
       
    #adjust arguments
    if args.skip_registration == 'True':
        skip_registration  = True
    else:
        skip_registration = False
        
    if args.rerun_registration == 'True':
        rerun_registration  = True
    else:
        rerun_registration = False

    config = ConfigParser.ConfigParser(allow_no_value=True)
    config.read([args.data_dir + '/configuration'])
    
    code_dir = config.get('host', 'code_dir')
    working_dir = config.get('host', 'working_dir')
    data_dir = config.get('host', 'data_dir_fromCluster')
    detectron_config=config.get('registration', 'detectron_config')
    labels=config.get('registration', 'labels')
    scale_percent=config.get('registration', 'scale-percent')
    src_mask=config.get('registration','src_mask')


    inputVideos=sorted(glob.glob(data_dir + '/' + '*.avi'))
    inputVideos2=sorted(glob.glob(data_dir + '/' + '*.MP4'))
    inputVideos.extend(inputVideos2)
    
    if os.path.exists(data_dir + '/' + 'startFrames'):
        startFile=h5py.File(data_dir + '/' + 'startFrames','r')
        startFrames=startFile['startFrames'][:]
        numFrames=startFile['numFrames'][:]
        startFile.close()
    else:
        print('no start frame file detected, starting from frame 0!')
        startFrames=np.zeros(len(inputVideos),'int32')
        numFrames=np.zeros(len(inputVideos),'int32')
        
    registration=[]; registration_log=[]; 
    for ind,video in enumerate(inputVideos):
        videoName= os.path.split(video)[1]
        registration.append(working_dir +'/' + videoName[:-4] + '.reg')
        registration_log.append(working_dir +'/' + videoName[:-4] + '.reg.log')
          
    registration_jobid=[]


    if skip_registration != True:
        tally=0
        for chunkidx, output in enumerate(registration):
          goodToRun = check_filedeps([inputVideos[chunkidx]],output,rerun_registration)
      		    
          if goodToRun==1:     
              slurm_args = ['--output', registration_log[chunkidx], \
                         '--job-name', 'registration',
                         '--partition', args.registration_partition]
              if args.registration_slurm != '':
                  slurm_args += args.registration_slurm.split(' ')
                  
                  job_args = [os.path.join(code_dir, 'detectron','maskAlignedVid.py'), \
                                  '--video', inputVideos[chunkidx], \
                                  '--labels', labels, \
                                  '--config', detectron_config, \
                                  '--src-mask', src_mask, \
                                  '--startFrame',str(startFrames[chunkidx]), \
                                  '--numFrames', str(numFrames[chunkidx]), \
                                  '--scale-percent', scale_percent, \
                                   output]
    
              if args.registration_args != '':
                  job_args += args.registration_args.split(' ')
              registration_jobid.append(
                         submit_job(job_args, slurm_args, args.mpicmd))
              print('>>> Submitted {}: {}'.format(registration_jobid[tally], output))
              tally+=1
