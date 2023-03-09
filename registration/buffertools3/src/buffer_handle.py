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

'''
This file contains classes for reading and writing using buffers.
'''

class Reader:

    def __init__(self, dataset, buffersize, start=0, stop=None):
        self._dataset = dataset
        self._buffersize = int(buffersize)
        self._start = int(start)
        self._stop = len(dataset) if stop is None else int(stop)

    def __enter__(self):
        self._buffernum = self._start // self._buffersize
        self._pos_start = max(self._buffernum * self._buffersize, \
                self._start)
        self._pos_end = min((self._buffernum + 1) * self._buffersize, \
                self._stop)
        if self._pos_start >= self._pos_end:
            self._buffer = None
        else:
            self._buffer = self._dataset[self._pos_start : self._pos_end]
        return self

    def __exit__(self, type, value, tb):
        pass

    def read(self):
        if self._buffer is None:
            raise RuntimeError('Already finished reading')

        item = self._buffer[0]
        self._buffer = self._buffer[1 : ]
        if len(self._buffer) == 0:
            self._buffernum += 1
            self._pos_start = max(self._buffernum * self._buffersize, \
                    self._start)
            self._pos_end = min((self._buffernum + 1) * self._buffersize, \
                    self._stop)
            if self._pos_start >= self._pos_end:
                self._buffer = None
            else:
                self._buffer = self._dataset[self._pos_start : self._pos_end]
        return item
    
    def done(self):
        return self._buffer is None

class Writer:

    def __init__(self, dataset, buffersize, start=0, stop=None):
        self._dataset = dataset
        self._buffersize = int(buffersize)
        self._start = int(start)
        self._stop = len(dataset) if stop is None else int(stop)
        
    def __enter__(self):
        self._buffernum = self._start // self._buffersize
        self._pos_start = max(self._buffernum * self._buffersize, \
                self._start)
        self._pos_end = min((self._buffernum + 1) * self._buffersize, \
                self._stop)
        if self._pos_start >= self._pos_end:
            self._buffer = None
        else:
            self._buffer = []
        return self

    def __exit__(self, type, value, tb):
        if type is not None:
            return
        if self._buffer is not None:
            self._pos_end = self._pos_start + len(self._buffer)
            self._dataset[self._pos_start : self._pos_end] = self._buffer

    def write(self, item):
        if self._buffer is None:
            raise RuntimeError('Already finished writing')

        self._buffer.append(item)
        if len(self._buffer) >= self._pos_end - self._pos_start:
            self._dataset[self._pos_start : self._pos_end] = self._buffer
            self._buffernum += 1
            self._pos_start = max(self._buffernum * self._buffersize, \
                    self._start)
            self._pos_end = min((self._buffernum + 1) * self._buffersize, \
                    self._stop)
            if self._pos_start >= self._pos_end:
                self._buffer = None
            else:
                self._buffer = []
    
    def done(self):
        return self._buffer is None

