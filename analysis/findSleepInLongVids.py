#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 21:34:56 2022

@author: sam
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
from natsort import natsorted
from scipy import stats
import os
import matplotlib.pyplot
import matplotlib.dates
import datetime
import matplotlib.dates as mdates



folder='/home/sam/bucket/octopus/electrophysiology/2022_03_07/cam0_2022-03-07-22-31-25.avi.meanInt'
file_list1=natsorted(glob.glob(folder + '/*.detectronResults'))
file_list2=natsorted(glob.glob(folder + '/*.meanInt'))
ephysData=0

if os.path.exists(folder + '_asTimes'):  #after annotating apartment data
    file=h5py.File(folder + '_asTimes','r')
    allmi=file['combinedInt'][:]
    fs=30#23.0
    if folder[-7]=='m':
        ephysData=1
        wakeVec=np.zeros([allmi.shape[0]])
        meanSig=np.mean(allmi)
        
elif folder[-7]=='m': #for ephys data
  #  file_list=natsorted(glob.glob(folder + '*.meanInt'))
    file=h5py.File(folder,'r')
    allmi=np.expand_dims(np.array(file['int'][:]),1)
    fs=file.attrs['fps']
    ephysData=1
    wakeVec=np.zeros([allmi.shape[0]])
    meanSig=np.mean(allmi)
    fs=24
    
elif len(file_list1)>0:  #first annotation of apartment data
    
    allmi=[]
    for filename in file_list1:
        file=h5py.File(filename,'r')
        mi=file['avg_int'][:]
        allmi.extend(mi)
    allmi=np.array(allmi)
    fs=file.attrs['fps']
    
else:  #first annotation of apartment data
    allmi=[]
    for filename in file_list2:
        file=h5py.File(filename,'r')
        mi=file['int'][:]
        allmi.extend(mi)
    allmi=np.array(allmi)
    fs=file.attrs['fps']
    allmi=np.expand_dims(allmi,1)

numSecs=[x /fs for x in range(0, allmi.shape[0])]
x = [str(datetime.timedelta(seconds=i)) for i in numSecs]

allmi_plot=allmi.copy()
allmi_plot2=allmi.copy()



for trace in range(allmi.shape[1]):
    y=allmi[:,trace]
    y2=y.copy()
    y[np.isnan(y)]=np.nanmean(y)
    allmi_plot[:,trace]=y+200*trace
    allmi_plot2[:,trace]=y2+200*trace
   # allmi_plot_checkY[:,trace]=


if os.path.exists(folder + '_asTimes'):
    try:
        allX=file['as_frames'][:]
        allRows=file['as_part'][:]
        allY=allmi_plot[allX,allRows]
       # meanVal=np.nanmean(allmi_plot,axis=0)
       # allY[np.isnan(allY)]=meanVal[allRows[np.isnan(allY)]]
        allX=allX[~np.isnan(allY)].tolist()
        allRows=allRows[~np.isnan(allY)].tolist()
        allY=allmi_plot[allX,allRows].tolist()
    except:
        allX=[]
        allRows=[]
        allY=[]
    # try:
    #     allX2=file['wake_frames'][:]
    #     allRows2=file['wake_part'][:]
    #     allY2=allmi_plot[allX2,allRows2]
    #     # allY2[np.isnan(allY2)]=meanVal[allRows[np.isnan(allY2)]]
    #     allX2=allX2[~np.isnan(allY2)].tolist()
    #     allRows2=allRows2[~np.isnan(allY2)].tolist()
    #     allY2=allmi_plot[allX2,allRows2].tolist()
  #  except:
    allX2=[]
    allRows2=[]
    allY2=[]
    
    file.close()
else:
    allX=[]
    allY=[]
    allX2=[]
    allY2=[]
    allRows=[]
    allRows2=[]
    
lag=4500
thresh=15
wakeDur=240
asDur=60

def onclick(event, ax):
    if event.dblclick:
        print("You double-clicked", event.button, event.xdata, event.ydata)
        X,Y=[event.xdata.astype('int'),event.ydata.astype('int')]
      
        currX=allmi_plot[int(X),:]
        currX[np.isnan(currX)]=0
        
        distX=np.min(np.abs(X-allX))
        distX2=np.min(np.abs(X-allX2))
        closestX=np.argmin(np.abs(X-allX))
        closestX2=np.argmin(np.abs(X-allX2))
                
        if distX<distX2:
            allX.pop(closestX)
            allY.pop(closestX)
            allRows.pop(closestX)
        else:
            allX2.pop(closestX2)
            allY2.pop(closestX2)
            allRows2.pop(closestX2)
            
        xl=ax.get_xlim()
        yl=ax.get_ylim()
        ax.clear()
        ax.plot(allmi_plot2,zorder=1)
        ax.scatter(allX,allY,200,'r',zorder=2)
        ax.scatter(allX2,allY2,200,'k',zorder=2)
        if ephysData==1:
            wakeVec=np.zeros([allmi_plot.shape[0]])
            for x1 in allX:
                wakeVec[x1:x1+int(asDur*fs)]=2
            for x1 in allX2:
                wakeVec[x1:x1+int(wakeDur*fs)]=1
            plt.plot(wakeVec*10+meanSig)
        ax.set_xlim(xmin=xl[0],xmax=xl[1])
        ax.set_ylim(ymin=yl[0],ymax=yl[1])
        plt.draw()

    else:
        if event.inaxes == ax:
           print(x[event.xdata.astype('int')])
      
           if event.button == 3:
               X,Y=[event.xdata.astype('int'),event.ydata.astype('int')]
               currX=allmi_plot[int(X),:]
               currX[np.isnan(currX)]=0
               rowID=np.argmin(np.abs(currX-Y))
               print(rowID)
               allRows.append(rowID)
               allX.append(X)
               allY.append(Y)
               xl=ax.get_xlim()
               yl=ax.get_ylim()
               ax.clear()
          
               ax.scatter(allX,allY,200,'r',zorder=2)
               ax.scatter(allX2,allY2,200,'k',zorder=2)  
               ax.plot(allmi_plot2,zorder=1)
               if ephysData==1:
                    wakeVec=np.zeros([allmi_plot.shape[0]])
                    for x1 in allX:
                        wakeVec[x1:x1+int(asDur*fs)]=2
                    for x1 in allX2:
                        wakeVec[x1:x1+int(wakeDur*fs)]=1
                    plt.plot(wakeVec*10+meanSig)
               ax.set_xlim(xmin=xl[0],xmax=xl[1])
               ax.set_ylim(ymin=yl[0],ymax=yl[1])
               plt.draw()
         
           if event.button == 2:
                X,Y=[event.xdata.astype('int'),event.ydata.astype('int')]
                currX=allmi_plot[int(X),:]
                currX[np.isnan(currX)]=0
                rowID=np.argmin(np.abs(currX-Y))
                allRows2.append(rowID)
                allX2.append(X)
                allY2.append(Y)
                xl=ax.get_xlim()
                yl=ax.get_ylim()
                ax.clear()
                ax.plot(allmi_plot2,zorder=1)
                ax.scatter(allX,allY,200,'r',zorder=2)
                ax.scatter(allX2,allY2,200,'k',zorder=2)
                ax.set_xlim(xmin=xl[0],xmax=xl[1])
                ax.set_ylim(ymin=yl[0],ymax=yl[1])
                if ephysData==1:
                    wakeVec=np.zeros([allmi_plot.shape[0]])
                    for x1 in allX:
                        wakeVec[x1:x1+int(asDur*fs)]=2
                    for x1 in allX2:
                        wakeVec[x1:x1+int(wakeDur*fs)]=1
                    plt.plot(wakeVec*10+meanSig)
                plt.draw()
           
fig, ax = plt.subplots()
ax.plot(allmi_plot2)
fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, ax))

#%%
allX=np.array(allX)
allX2=np.array(allX2)
allRows=np.array(allRows)
allRows2=np.array(allRows2)

if ephysData==1:
    wakeVec=np.zeros([allmi_plot.shape[0]])
    for x1 in allX:
        wakeVec[x1:x1+int(asDur*fs)]=2
    for x1 in allX2:
        wakeVec[x1:x1+int(asDur*fs)]=1
                        
# asBouts=[]
# for trace in range(allmi.shape[1]):
#     asBouts.append(natsorted(allX[np.where(allRows==trace)]))

basepath = os.path.split(folder)[0] 
basename = os.path.split(folder)[1]
 
outputFile=basepath + '/' + basename + '_asTimesForBCs'

writer = h5py.File(outputFile, 'w')
writer.attrs.create('folder', \
          folder, \
          dtype=h5py.special_dtype(vlen=str))
writer.create_dataset('as_frames', data=allX)
writer.create_dataset('wake_frames', data=allX2)
writer.create_dataset('as_part', data=allRows)
writer.create_dataset('wake_part', data=allRows2)
writer.create_dataset('combinedInt', data=allmi)
if ephysData==1:
    writer.create_dataset('wakeVec',data=wakeVec)
writer.close()




