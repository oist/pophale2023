#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import time


CODE_DIR = os.path.dirname(os.path.abspath(__file__))


def retry_ssh(sshcmd, cmd, flush=False, connection_retries=30, connection_retrysleep=5):

    
    num_ssh_layers = sshcmd.count('ssh ')
    cmd = cmd.replace('"', '\"')
    for i in range(2, num_ssh_layers+1):
        cmd = cmd.replace('"', '\\'*(2**(i-2)) + '"')
    cmd_args = sshcmd.split(' ') + [cmd]
    n = 0
    while n < connection_retries:
        try:
         #   import pdb; pdb.set_trace()
            if flush:
                p = subprocess.Popen(cmd_args, stdout=subprocess.PIPE)
                for line in iter(p.stdout.readline, b''):
                    print('>> {}: {}'.format(sshcmd, line.rstrip()))
            else:
                stdout = subprocess.check_output(cmd_args).rstrip()
                return stdout
        except subprocess.CalledProcessError as err:
            if err.returncode == 255:
                print('Retrying ssh ...')
                time.sleep(connection_retrysleep)
                n += 1
            else:
                print('Cannot connect to server')
                return err

def retry_rsync(sshcmd, cmd, flush=False, connection_retries=30, connection_retrysleep=10):
    cmd_args = ['rsync', '-e'] + [sshcmd] + cmd.replace('"', '').split(' ')

    n = 0
    while n < connection_retries:
        try:
            if flush:
                p = subprocess.Popen(cmd_args, stdout=subprocess.PIPE)
                for line in iter(p.stdout.readline, b''):
                    print('>> rsync: {}'.format(line.rstrip()))
                return True
            else:
                stdout = subprocess.check_output(cmd_args).rstrip()
                return stdout
        except subprocess.CalledProcessError as err:
            if err.returncode == 255:
                print('Retrying rsync ...')
                time.sleep(connection_retrysleep)
                n += 1
            else:
                print('Cannot connect to server')
                return err

def expand_path(x, sshcmd=None):
    """
    Relative path makes no sense in an ssh environment.
    """
    if sshcmd is None:
        return os.path.abspath(os.path.expandvars(os.path.expanduser(x)))
    else:
        cmd = 'import os; ' \
              + 'print os.path.abspath(os.path.expandvars(os.path.expanduser(\\"{}\\")))'.format(x)
        cmd = 'python -c "{}"'.format(cmd)
        return retry_ssh(sshcmd, cmd)
