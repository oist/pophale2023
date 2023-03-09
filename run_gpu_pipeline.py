#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 08:48:12 2021

@author: sam
"""

import os
import argparse
import configparser as ConfigParser
import glob
import shutil
from ssh_tools import retry_ssh

#%%
if __name__ == '__main__':

    # Arguments used in run_pipeline
    p = argparse.ArgumentParser('Running pipeline', add_help=False)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--keepfiles', action='store_true')
    p.add_argument('--sleep-between-sync', default=10, type=int)
    p.add_argument('--inputFolder', help='directory containing videos')
    p.add_argument('--PullRemoteFirst', action='store_true')
    p.add_argument('--skip-initialSync', action='store_true')
    p.add_argument('--skip-registration', action='store_true')
    p.add_argument('--rerun-registration', action='store_true')
    p.add_argument('--rerun-regPattern', action='store_true')
    args, remaining_argv = p.parse_known_args()

    # Set up basic blocks
    dirname = args.inputFolder 
    basepath = os.path.split(dirname)[0] 
    basename = os.path.split(dirname)[1]
 
    configName = dirname + '/' + basename + '.cfg'

    # Parse dataset config, more options under 'submitting jobs'
    config = ConfigParser.ConfigParser(allow_no_value=True)
    config.read([configName])
    hostname = config.get('slurm', 'hostname')
    data_dir_fromCluster = config.get('host', 'data_dir_fromCluster')
    code_dir = config.get('host', 'code_dir')
    data_dir = config.get('host', 'data_dir')
    sshcmd = 'ssh ' + hostname  
        
    inputVideos=sorted(glob.glob(dirname + '/' + '*.MP4'))
    shutil.copy(configName, dirname + '/configuration')
 
#%% Submit registration jobs

    print('Submitting registration jobs')
    cmd_args = [os.path.join(code_dir, 'run_mask_alignment.py'), \
        '--registration-slurm "{}"'.format(config.get('registration', 'slurm')), \
        '--registration-partition "{}"'.format(config.get('registration', 'partition')), \
        '--mpicmd "{}"'.format(config.get('host', 'mpicmd')), \
        '--rerun-registration "{}"'.format(args.rerun_registration), \
        '--skip-registration "{}"'.format(args.skip_registration), \
        '--data-dir "{}"'.format(data_dir_fromCluster)] 
    cmd_args = ' '.join(cmd_args)
    print(cmd_args)
    stdout = retry_ssh(sshcmd, cmd_args, flush=False)
     
   
    
