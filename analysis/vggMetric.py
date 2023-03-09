#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 08:47:02 2022

@author: sam
"""



import numpy as np
from tensorflow.keras.applications import vgg19
import tensorflow as tf
import h5py
import glob
import os
import pandas as pd

class Batcher: #from James D. McCaffrey, https://jamesmccaffrey.wordpress.com/2018/08/28/a-custom-iterable-batcher-using-python/ 
  def __init__(self, num_items, batch_size, seed=0):
    self.indices = np.arange(num_items)
    self.num_items = num_items
    self.batch_size = batch_size
    self.rnd = np.random.RandomState(seed)
   # self.rnd.shuffle(self.indices)
    self.ptr = 0

  def __iter__(self):
    return self

  def __next__(self):
      
    if self.ptr  > self.num_items:
      self.ptr = 0
      raise StopIteration  # ugly Python
    elif self.ptr + self.batch_size > self.num_items:
      #self.rnd.shuffle(self.indices)
      result=self.indices[self.ptr:self.num_items]
      self.ptr += self.batch_size
      return result
    else:
      result = \
        self.indices[self.ptr:self.ptr+self.batch_size]
      self.ptr += self.batch_size
      return result



def gram_matrix(x,dims):
    
    gram=[]
    for img in range(0,x.shape[0]):
        currImg=np.squeeze(x[img,:,:])
        gram.append(np.dot(np.transpose(currImg),currImg).ravel())
    return np.array(gram)


def style_rep(style):
    style=np.squeeze(style)
    dims = style.shape
    S = gram_matrix(style,dims)

    return S 


            

model = vgg19.VGG19(weights='imagenet',
                    include_top=False)
b1c1 = tf.keras.Model(model.input, model.get_layer('block1_conv1').output)
b2c1 = tf.keras.Model(model.input, model.get_layer('block2_conv1').output)
b3c1 = tf.keras.Model(model.input, model.get_layer('block3_conv1').output)
b4c1 = tf.keras.Model(model.input, model.get_layer('block4_conv1').output)
b5c1 = tf.keras.Model(model.input, model.get_layer('block5_conv1').output)
b5pool = tf.keras.Model(model.input, model.get_layer('block5_pool').output)

#dirname='/home/sam/bucket/octopus/8k/oct_07'
#dirname='/home/sam/bucket/octopus/8k/oct_05'

#for waking image data
dirname='/home/sam/bucket/octopus/8k/awake/good_extracted_frames'
flowFiles = sorted(glob.glob(dirname + '/' + '*.reg'))
htFlip =np.ones(len(flowFiles))
flowFile=h5py.File(flowFiles[0],'r')
octInd =flowFile['octNum'][:]
# #for sleeping data
# dirname='/home/sam/bucket/octopus/8k/oct_456'
# flowFileList=pd.read_csv(dirname + '/allDays.csv')
# flowFiles=flowFileList['file'].values.tolist()
# htFlip=flowFileList['htFlip'].values.tolist()
# octInd=flowFileList['oct'].values.tolist()

outDir=dirname
for flow_ind, file in enumerate(flowFiles):
    
    try:
        basename = os.path.split(file)[1]
       # file=dirname + '/' + basename  #hack name so I don't have to redo csv file
      #  file=file[:-9]
        if not os.path.exists(outDir + '/' + basename + '_allFeats'):
             
                flowFile=h5py.File(file,'r')
                if htFlip[flow_ind]==1:
                    embeddingData=flowFile['patterns1']
                else:
                    embeddingData=flowFile['patterns2']
                
             
                print(file)
                allb1=[]
                allb2=[]
                allb3=[]
                allb4=[]
                allb5=[]
                allb5p=[]
                

                ni = embeddingData.shape[0] # number items
                bs = 40   # batch size -- poor, not evenly divisible
                my_batcher = Batcher(ni, bs, seed=1)
             
                for i,b in enumerate(my_batcher):
                    currBatch = vgg19.preprocess_input(embeddingData[b,400:800,187:487])
                    
                    b1 = np.array(b1c1(currBatch))
                    b2 = np.array(b2c1(currBatch))
                    b3 = np.array(b3c1(currBatch))
                    b4 = np.array(b4c1(currBatch))
                    b5 = np.array(b5c1(currBatch))
                    b5p = np.array(b5pool(currBatch))
                    
                    b1=b1.reshape(b1.shape[0],b1.shape[1]*b1.shape[2],b1.shape[3])
                    b2=b2.reshape(b2.shape[0],b2.shape[1]*b2.shape[2],b2.shape[3])
                    b3=b3.reshape(b3.shape[0],b3.shape[1]*b3.shape[2],b3.shape[3])
                    b4=b4.reshape(b4.shape[0],b4.shape[1]*b4.shape[2],b4.shape[3])
                    b5=b5.reshape(b5.shape[0],b5.shape[1]*b5.shape[2],b5.shape[3])
                    b5p=b5p.reshape(b5p.shape[0],b5p.shape[1]*b5p.shape[2],b5p.shape[3])
                    
                    b1m=np.max(b1,axis=1)
                    b2m=np.max(b2,axis=1)
                    b3m=np.max(b3,axis=1)
                    b4m=np.max(b4,axis=1)
                    b5m=np.max(b5,axis=1)
                    b5pm=np.max(b5p,axis=1)
                    
                    allb1.extend(b1m)
                    allb2.extend(b2m)
                    allb3.extend(b3m)
                    allb4.extend(b4m)
                    allb5.extend(b5m)
                    allb5p.extend(b5pm)

                    print(i)
 
                allb1=np.array(allb1)
                allb2=np.array(allb2)
                allb3=np.array(allb3)
                allb4=np.array(allb4)
                allb5=np.array(allb5)
                allb5p=np.array(allb5p)
                
                
        
              # with h5py.File(outDir + '/' + basename + '_features', 'r') as reader:
             #    allFeats=reader['feats'][:]
      
                with h5py.File(outDir + '/' + basename + '_allFeats', 'w') as writer:
                 writer.create_dataset('feats1', data=allb1)
                 writer.create_dataset('feats2', data=allb2)
                 writer.create_dataset('feats3', data=allb3)
                 writer.create_dataset('feats4', data=allb4)
                 writer.create_dataset('feats5', data=allb5)
                 writer.create_dataset('feats5p', data=allb5p)
                 writer.attrs.create('htFlip',htFlip[flow_ind])
                 writer.attrs.create('octInd',octInd)

      
    except:
        print("An exception occurred " + file)
    


