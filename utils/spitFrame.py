#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 14:59:25 2022

@author: sam
"""


import argparse
from tqdm import tqdm
import cv2


if __name__ == '__main__':
    p = argparse.ArgumentParser('write frame')

    p.add_argument('--frames', nargs="+", type=int)
    p.add_argument('--video')
    args = p.parse_args()
    
    video = cv2.VideoCapture(args.video)
    for f in args.frames:
        video.set(cv2.CAP_PROP_POS_FRAMES, f)
        succ, img = video.read()
        cv2.imwrite('frame_' + str(f) + '.png', img)
    video.release()
