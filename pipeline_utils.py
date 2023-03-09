#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:50:17 2020

@author: sam
"""

import os
import subprocess
import time
import numpy as np
import json
import scipy.io
import pandas as pd

def submit_job(job_args, slurm_args=None, mpicmd='srun', \
    max_queued=None, wait_time=None, debug=False):
    """Submit job to SLURM using srun and return jobid.
    job_args: list
    slurm_args: list
    """

    # Default to parsed command-line arguments...
    if max_queued is None:
        max_queued = 300000000
 #   if wait_time is None:
   #     wait_time = 100



    # Check current queue l
    import subprocess
    user_queue = subprocess.check_output(['squeue', '--user', os.path.expandvars('$USER')])
    queue_str=user_queue.decode()
    if len(queue_str.split('\n')) > max_queued:
        print('Maximum number of queued jobs of {} reached, waiting for {} s.'.format(
            max_queued, wait_time))
    #    time.sleep(wait_time)

    # Submit job using sbatch via heredoc batch script
    sbatch_options = ['sbatch']
    if slurm_args is not None:
        sbatch_options += slurm_args

    p = subprocess.Popen(sbatch_options, 
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    batch_script_heredoc = ('#!/bin/bash\n source ~/.bashrc; ' 
                            + '{} {}'.format(mpicmd, ' '.join(job_args)))

    stdout, _ = p.communicate(batch_script_heredoc.encode())

    if debug:
        print(' '.join(sbatch_options))
        print(batch_script_heredoc)
    
    # Retrieve jobid
    jobid = stdout.split()[3]

    return jobid




def check_filedeps(inputFileList, outputFile, rerun):
    
    nonExistantFiles=[]
    nonExistantFiles.append(0)
    for file in inputFileList:
        if os.path.isfile(file)==False:
            nonExistantFiles.append(1)
    
    outputExists=0
    if os.path.isfile(outputFile):
        outputExists=1
    if rerun:
        outputExists=0
    
    if outputExists == 0 and np.amax(np.array(nonExistantFiles)) == 0:
        return 1
    else:
        return 0
        


def check_QueueLength(queueSize, wait_time=None):

    keepWaiting=1
    while keepWaiting==1:
        user_queue = subprocess.check_output(['squeue', '--user', os.path.expandvars('$USER')])
        queue_str=user_queue.decode()
        if (len(queue_str.split('\n'))-2) > queueSize:
            print('waiting for the queue to reduce  before continuing...')
            time.sleep(5)
    #import pdb; pdb.set_trace()
        else:
            keepWaiting=0
            
            
            
def checkRelCamExistance(directory, chunk, array_params, pano_params, relCamsFile):
    df =  pd.read_pickle(array_params)
    with open(relCamsFile, 'r') as fr:
        relCams=fr.read(); 
    relCams = relCams.strip('[]\n').split()
    for ind in range(len(relCams)):
       relCams[ind] = relCams[ind].strip(',') 

    registration_files=[]
    goodRanks=[]
    goodRankInds=[]
    for img in range(len(relCams)):
        registration_fileName = (directory + '/' + df['names'][int(relCams[img])] + '-' + chunk + '.reg')
        if os.path.isfile(registration_fileName):
            registration_files.append(registration_fileName)  
            goodRanks.append(int(relCams[img]))
            goodRankInds.append(img)
            
    f = open(relCamsFile, 'w')
    json.dump(goodRanks, f)
    f.close()
    print('there are ' + str(len(goodRankInds)) + ' good files, and ' + str(len(relCams)) + ' relcams')
    #adjust panorama if a registration file drops out
    if len(goodRankInds) != len(relCams):
        print('about to modify the mat file')
        mFile=scipy.io.loadmat(pano_params) 
        uStruct=mFile['uStruct'] 
        vStruct=mFile['vStruct'] 
        m_v0_=mFile['m_v0_'].astype('int32')
        m_v1_=mFile['m_v1_'].astype('int32')
        m_u0_=mFile['m_u0_'].astype('int32')
        m_u1_=mFile['m_u1_'].astype('int32')
        mosaich=np.asscalar(mFile['mosaich'])
        mosaicw=np.asscalar(mFile['mosaicw'])
        
        uStruct = uStruct[0,goodRankInds]
        vStruct = vStruct[0,goodRankInds]
        m_v0_ =  m_v0_[goodRankInds]
        m_v1_ =  m_v1_[goodRankInds]
        m_u0_ =  m_u0_[goodRankInds]
        m_u1_ =  m_u1_[goodRankInds]
                
        scipy.io.savemat(pano_params, mdict={'uStruct': uStruct, 'vStruct': vStruct, \
                                             'm_v0_': m_v0_, \
                                             'm_v1_': m_v1_, \
                                             'm_u0_': m_u0_, \
                                             'm_u1_': m_u1_, \
                                             'mosaich':mosaich, \
                                             'mosaicw':mosaicw})
    