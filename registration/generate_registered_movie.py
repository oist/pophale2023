#!/usr/bin/env python3

#
# (C) Copyright 2015 Frankfurt Institute for Advanced Studies
# (C) Copyright 2016 Max-Planck Institute for Brain Research
#
# Author: Philipp Huelsdunk  <huelsdunk@fias.uni-frankfurt.de>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the author nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#     * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#     * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#     * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import argparse
import numpy as np
from tqdm import tqdm
import os.path
import h5py
from buffertools3 import buffer_handle
import cv2
import skvideo.io

if __name__ == '__main__':
p = argparse.ArgumentParser('Play registration')
p.add_argument('--crop')
p.add_argument('--scale', type=float)
p.add_argument('--output')
p.add_argument('--codec', default="MP4V")
p.add_argument('registration', help='Registration hd5 file')
args = p.parse_args()

# Read the registration file
registration = h5py.File(args.registration, 'r')
maps_dset = registration['maps']
reg_num_frames = maps_dset.shape[0]
fps = registration.attrs['fps']
rel_video_filename = registration.attrs['video']
startFrame = registration.attrs['startFrame']

# Transformation to scale maps up
scale_maps_t = np.array([[maps_dset.attrs['grid_size'], 0., 0.], \
                         [0., maps_dset.attrs['grid_size'], 0.]], 'float32')

# Open video
video_filename = args.registration[:-3] + 'MP4'
video = cv2.VideoCapture(video_filename)
if (cv2.__version__.split('.')[0] == '3') or (cv2.__version__.split('.')[0] == '4'):
    fps = float(video.get(cv2.CAP_PROP_FPS))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
else:
    fps = float(video.get(cv2.cv.CV_CAP_PROP_FPS))
    height = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
 
# Goto chunk start
video.set(cv2.CAP_PROP_POS_FRAMES, startFrame)

def create_video_iter():
    for _ in range(reg_num_frames):
        succ, img = video.read()
        if not succ:
            raise RuntimeError('Video finished too early')
        yield img
video_iter = create_video_iter()
def create_iter_maps():
    with buffer_handle.Reader(maps_dset, maps_dset.chunks[0]) \
            as maps_reader:
        while not maps_reader.done():
            yield maps_reader.read()
maps_iter = create_iter_maps()

# Calculate output video frame size
output_framesize = (width, height)
if args.crop is not None:
    x, y, width, height = (int(s) for s in args.crop.split(','))
    output_framesize = (width, height)
if args.scale is not None:
    output_framesize = tuple(int(s * args.scale + .5) \
            for s in output_framesize)

 # Open output video
if args.output is not None:
    output_video_file = skvideo.io.FFmpegWriter(args.output)
else:
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image',2800,1900)


# Play registered video
for frame, maps in tqdm(zip(video_iter, maps_iter), \
        total=reg_num_frames, \
        desc='Playing registration'):
        # Warp frame

        maps = cv2.warpAffine(maps, scale_maps_t, frame.shape[1::-1])
        
        frame = cv2.remap(frame, maps, None, \
                interpolation=cv2.INTER_LINEAR)
        
        # Crop and scale
        if args.crop is not None:
            x, y, width, height = (int(s) for s in args.crop.split(','))
            frame = frame[y : y + height, x : x + width]
        if args.scale is not None:
            frame = cv2.resize(frame, output_framesize)

        if args.output is None:
            cv2.imshow('image', frame)
            cv2.waitKey(1)
        else:
            output_video_file.writeFrame(frame[...,::-1]) #bgr to rgb
    
    if args.output is not None:
          output_video_file.close()
          
    video.release()
    registration.close()
