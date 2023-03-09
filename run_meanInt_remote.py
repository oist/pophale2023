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
import glob


if __name__ == '__main__':

    p = argparse.ArgumentParser('Running pipeline on remote', add_help=False)
    p.add_argument('--videoPath', default="")
    p.add_argument('--mpicmd', default='mpiexec')
    args = p.parse_args()
 
        
   
    inputVideos=sorted(glob.glob(args.videoPath + '/*.avi'))
    outputFolder='/flash/ReiterU/mantleInt/'
     
    meanInt=[]; meanInt_log=[]
    for ind,video in enumerate(inputVideos):
        videoName= os.path.split(video)[1]
        meanInt.append(outputFolder + videoName[:-4] + '.movei')
        meanInt_log.append(outputFolder + videoName[:-4] + '.movei_log')
       
    
    
    #%% Submit all jobs 
    meanInt_jobid=[]
    tally=0
    for chunkidx, output in enumerate(meanInt):
      goodToRun = check_filedeps([inputVideos[chunkidx]],output,0)
  		    
      if goodToRun==1:     
          slurm_args = ['--output', meanInt_log[chunkidx], \
                     '--job-name', 'resMotion',
                     '-p', 'compute',
                     '-t', '0-12',
                     '-c', '64',
                     '--mem', '64G']

           
          job_args = [os.path.join('/home/s/samuel-reiter/octoSleep/', 'analysis','meanInt_simple.py'), \
                              '--video', inputVideos[chunkidx], \
                              '--mask', inputVideos[chunkidx][:-4] + '_mask.png', \
                              '--outputPath',outputFolder]

          meanInt_jobid.append(
                     submit_job(job_args, slurm_args, args.mpicmd))
          print('>>> Submitted {}: {}'.format(meanInt_jobid[tally], output))
          tally+=1

   

