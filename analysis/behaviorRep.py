#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 09:30:55 2019

@author: reiters
"""
#careful to include only full AS bouts with good posture to not bias any measurement of differences
from geosketch import gs
import numpy as np
import h5py
from sklearn.decomposition import PCA
import cv2
import glob
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
from scipy import stats
import pandas as pd

dirnames=['/home/sam/bucket/octopus/8k/oct_456']

#for generating a day file
# file = dirnames[2] + '/day.csv'
# af = sorted(glob.glob(dirnames[2] + '/' + '*.reg_features'))
# df = pd.DataFrame(data=af)
# df.to_csv(file)


awakeDirs=['/home/sam/bucket/octopus/8k/awake/good_extracted_frames']

allFeats=[]
allDatasets=[]
allDatasetNames=[]
allED=[]
allOct=[]
allFiles1=[]
allHT1=[]
X_train=[]

i=0
for di, d in enumerate(dirnames):
    af = sorted(glob.glob(d + '/' + '*_allFeats'))
    allFiles1.extend(af)

    for file in af:
        f = h5py.File(file,'r')
        feats=f['feats5p'][:]
        ht=int(f.attrs['htFlip'])
        octNum=int(f.attrs['octInd'])
        f.close()
        dset=np.zeros(feats.shape[0],)+i
        X_train.extend(feats)
        allDatasetNames.append(file)
        
        allDatasets.extend(dset)
        allOct.append(octNum)
        allHT1.append(ht)
        i+=1
  
    allFeats.extend(X_train)

feats1=np.array(allFeats)
dset1=np.array(allDatasets).astype('int32')
allOct1=np.array(allOct)


allFeats=[]
allOct=[]
allDatasets=[]
allFiles2=[];
allDatasetNames2=[]
allHT2=[]
X_train=[]

sleepDs=di+1
for di, d in enumerate(awakeDirs):
    af = sorted(glob.glob(d + '/' + '*allFeats'))
    allFiles2.extend(af)
    X_train=[]
    for file in af:
        f = h5py.File(file,'r')
        feats=f['feats5p'][:]
        ht=1
        octNum=f.attrs['octInd'][:]
        f.close()
        dset=np.zeros(feats.shape[0],)+di
        X_train.extend(feats)
        
        allDatasetNames.append(file)
        allDatasets.extend(dset)
        allOct.extend(octNum)
        allHT2.append(ht)
        i+=1


    allFeats.extend(X_train)

feats2=np.array(allFeats)
dset2=np.array(allDatasets).astype('int32')
allOct2=np.array(allOct)

    

# reader = h5py.File('behaviorWork', 'r')
# embedding=reader['embedding'][:]
# mappedRep=reader['mappedRep'][:]
# clustering=reader['clustering'][:]
# sketch_index=reader['sketch_index'][:]
# dset=reader['dset'][:]
# si=reader['si'][:]
# reader.close()

# writer=h5py.File('/home/sam/bucket/octopus/8k/oct_456/behaviorWork','w')
# writer.create_dataset('feats',data=feats1)
# writer.create_dataset('featsB',data=feats2)
# writer.create_dataset('dset',data=dset1)
# writer.create_dataset('dsetB',data=dset2)
# writer.create_dataset('oct',data=allOct1)
# writer.create_dataset('octB',data=allOct2)
# writer.create_dataset('ht',data=allHT1)
# writer.close()

#%%

pca = PCA(n_components=50)#check this
PCAmodel=pca.fit(feats1) 
mappedRep=PCAmodel.transform(feats1)
mappedRep2=PCAmodel.transform(feats2)

plt.figure()
plt.scatter(mappedRep[:, 0], mappedRep[:, 1],.5,c='k')
plt.scatter(mappedRep2[:, 0], mappedRep2[:, 4],30,c='r')
            
#%%
# C = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
# plt.figure()
# plt.scatter(embedding[np.argwhere(allOct[sketch_index]==0), 0], embedding[np.argwhere(allOct[sketch_index]==0), 1],.1,[1,0,0])
# plt.scatter(embedding[np.argwhere(allOct[sketch_index]==1), 0], embedding[np.argwhere(allOct[sketch_index]==1), 1],.1,[0,1,0])
# plt.scatter(embedding[np.argwhere(allOct[sketch_index]==2), 0], embedding[np.argwhere(allOct[sketch_index]==2), 1],.1,[0,0,1])


plt.figure()
plt.scatter(embedding[:, 0], embedding[:, 1],.1)



#pick closest samples to cluster means, restrict to different datasets and different octopuses
# si = gs(embedding, 20, replace=False)
# siO=si.copy()
# si=np.array(si)
# si=si[[0,2,9,27,31,36]]

plt.scatter(mappedRep[si, 0], mappedRep[si, 1],50,c='r')

# si=range(147673,147693)
# plt.figure()
# plt.scatter(mappedRep[:, 0], mappedRep[:, 1],.1,c=allOct)
# plt.scatter(mappedRep[si, 0], mappedRep[si, 1],10,c='r')


import scipy

def find_indices(points,lon,lat,tree=None):
    if tree is None:
      #  lon,lat = lon.T,lat.T
        lonlat = np.column_stack((lon.ravel(),lat.ravel()))
        tree = scipy.spatial.cKDTree(points)
    dist,idx = tree.query(lonlat,k=1)
    return idx

# def find_indices1d(points,lon,tree=None):
#     if tree is None:

#         tree = scipy.spatial.cKDTree(points)
#     dist,idx = tree.query(np.expand_dims(lon,1),k=1)
#     return idx

def find_indicesnd(points,lon,tree=None):
    tree = scipy.spatial.cKDTree(points)
    dist,idx = tree.query(lon,k=1)
    return idx

currSpace=mappedRep[allOct1==0,:]
currSpace=embedding
res=int(2)
minX=np.percentile(currSpace[:,0],1).astype('int64')
maxX=np.percentile(currSpace[:,0],99).astype('int64')
minY=np.percentile(currSpace[:,1],1).astype('int64')
maxY=np.percentile(currSpace[:,1],99).astype('int64')
xv, yv = np.meshgrid(range(minX,maxX,res), range(minY,maxY,res), indexing='xy')

si=np.unique(find_indices(currSpace[0:np.where(allOct1==0)[0][-1],0:2],xv,yv))



#funny, I don't understand the structure of the PCs. Maybe need to synthesize images from the pcs to know
si=[]
for dim in range(0,3):
    minX=np.min(mappedRep[:,dim]).astype('int64')
    maxX=np.max(mappedRep[:,dim]).astype('int64')
    xv=np.arange(minX, maxX, 30)
    print(xv)
    querries=np.zeros((len(xv),mappedRep.shape[1]))
    querries[:,dim]=xv
    si.extend(np.unique(find_indicesnd(mappedRep,querries)))
    

currInds=np.where(dset1==8)[0]
plt.plot(mappedRep[currInds,0])

si=np.unique(find_indices(currSpace[0:np.where(allOct1==0)[0][-1],0:2],xv,yv))


# plt.figure()
# plt.scatter(embedding[:, 0], embedding[:, 1],5,allOct[sketch_index])
# plt.scatter(embedding[si, 0], embedding[si, 1],20,c='r')


# sketch=np.array(sketch_index)
# si=sketch[si]
# siToUse=si[[2,0,3,18,9]]

# si=np.where(allOct==1)[0]
# siToUse=si
#%% Examples


# #sleep
#siToUse=si
#sleepRun=1


# # #wake
for awakeSample in np.unique(dset2):
    siToUse=np.where(dset2==awakeSample)[0]
siToUse=siToUse[[3,8,19,44, 46,49]] #for now
sleepRun=0


exampleNum=3
fig,axs=plt.subplots(exampleNum*3+1,len(siToUse),gridspec_kw = {'wspace':0, 'hspace':0})

for clust in range(0,len(siToUse)):
    print(clust)
    currInd=siToUse[clust]
  
    column=0
    if sleepRun==1:
        currSamp=feats1[currInd]
        currDset=dset1[currInd]
        dSetIndices=np.argwhere(dset1==currDset)
        imgIndex=currInd-dSetIndices[0]
        currDsetFile=allFiles1[currDset]
        currHT=allHT1[currDset]
    else:
        currSamp=feats2[currInd]
        currDset=dset2[currInd]
        dSetIndices=np.argwhere(dset2==currDset)
        currDsetFile=allFiles2[currDset]
        currHT=allHT2[currDset]
        
    sampleDist=np.sum(np.square(feats1-currSamp),axis=1)
    imgIndex=currInd-dSetIndices[0]
    currEmbedFile1=currDsetFile.split('.')[0] + '.reg' 
    currDsetFile=h5py.File(currEmbedFile1,'r')
    
    if currHT==1:
        embeddingData=np.squeeze(currDsetFile['patterns1'][imgIndex])
    else:
       embeddingData=np.squeeze(currDsetFile['patterns2'][imgIndex])
    
    m=np.mean(embeddingData,axis=2)
    axs[column,clust].imshow(embeddingData,vmin=0, vmax=155)
    axs[column,clust].set_aspect('auto')
    column=column+1

    #loop over octopus
    for currO in np.unique(allOct1):
    
            currDsetInds=dset1[np.where(allOct1==currO)]

            dsetSamp=[]
            dsetD=[]
            for d in np.unique(currDsetInds):
                currInds=np.where(dset1==d)[0]
                oneDsetDist=np.sort(sampleDist[currInds])
                oneDsetSamp=np.argsort(sampleDist[currInds])
                dsetSamp.append(currInds[oneDsetSamp[0]])
                dsetD.append(oneDsetDist[0])
            dsetD=np.array(dsetD)
            dsetSamp=np.array(dsetSamp)
    
            # #take the closest N of these samples, belonging to the right octopus
            sortedDsetInd=dsetSamp[np.argsort(dsetD)][0:exampleNum]
            #print(mappedRep[sortedDsetInd]) 
            print(sortedDsetInd)
          #  plt.scatter(mappedRep[oneDsetInds[0:exampleNum],0],mappedRep[oneDsetInds[0:exampleNum],1])
            for currInd in sortedDsetInd:
                
                currDset=dset1[currInd]
                dSetIndices=np.argwhere(dset1==currDset)
                imgIndex=currInd-dSetIndices[0]
                currDsetFile=allFiles1[currDset]
                currEmbedFile=currDsetFile.split('.')[0] + '.reg'
                currDsetFile=h5py.File(currEmbedFile,'r')
              
                if allHT1[currDset]==1:
                    embeddingData=np.squeeze(currDsetFile['patterns1'][imgIndex])
                else:
                    embeddingData=np.squeeze(currDsetFile['patterns2'][imgIndex])
                
                m=np.mean(embeddingData,axis=2)
                embeddingData[m==127]=0
                embeddingData[m==127,int(currO)]=255
             #   axs[column,clust].imshow(np.squeeze(embeddingData[400:800,187:487]),vmin=0, vmax=155)
                axs[column,clust].imshow(np.squeeze(embeddingData),vmin=0, vmax=155)
                axs[column,clust].set_aspect('auto')
                column=column+1
    plt.setp(axs, xticks=[], yticks=[])
    plt.subplots_adjust(wspace=0, hspace=0)



#%%


pca = PCA(n_components=50)#check this
PCAmodel=pca.fit(feats1) 
mappedRep=PCAmodel.transform(feats1)
clustering=dset
N = 20000 # Number of samples to obtain from the data set.
sketch_index = gs(mappedRep, N, replace=False)
X_sketch = mappedRep[sketch_index]

import umap
reducer = umap.UMAP(min_dist=1, n_neighbors=100)
embedding=reducer.fit(X_sketch)
embedding = reducer.transform(mappedRep)

exp_var_pca = pca.explained_variance_ratio_

cum_sum_eigenvalues = np.cumsum(exp_var_pca)

plt.figure()
plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10,init='k-means++', random_state=0).fit(mappedRep)
clustering=kmeans.labels_

#reorder clusters along 1st pc for display
pcOneMean=np.zeros(10)
for c in np.unique(clustering):
    pcOneMean[c]=np.mean(mappedRep[np.where(clustering==c),0])
clusterSort=np.argsort(pcOneMean)

sortCluster=np.zeros(len(clustering))
for c in np.unique(clustering):
    sortCluster[np.argwhere(clustering==clusterSort[c])]=c
clustering=sortCluster.astype('int32')


writer = h5py.File('behaviorWork1', 'w')
writer.create_dataset('embedding', data=embedding)
writer.create_dataset('mappedRep', data=mappedRep)
writer.create_dataset('feats', data=feats1)
writer.create_dataset('oct', data=allOct)

#writer.create_dataset('clustering', data=clustering)
writer.create_dataset('sketch_index', data=sketch_index)
writer.create_dataset('dset', data=dset1)
#writer.create_dataset('si', data=si)

writer.close()

#%% different order of patterns








from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10,init='k-means++', random_state=0).fit(mappedRep)
clustering=kmeans.labels_

import similaritymeasures



numDsets=np.max(dset).astype('int32')
allC=np.zeros([numDsets,numDsets])
for x in range(0,numDsets):
    dset1=mappedRep[np.where(dset==x)[0],:]
      
    for y in range(0,numDsets):
        dset2=mappedRep[np.where(dset==y)[0],:]
        dist = tdist.sspd(dset1,dset2)
        allC[x,y]=dist



from scipy import signal

numDsets=np.max(dset).astype('int32')
allC=np.zeros([numDsets,numDsets])
for x in range(0,numDsets):
    dset1=clustering[np.where(dset==x)[0]]
    corr = signal.correlate(dset1,dset1,'same')
    normValue=np.max(corr)
    
    for y in range(0,numDsets):
        dset2=clustering[np.where(dset==y)[0]]
        corr = signal.correlate(dset1,dset2,'same')
        corr /= normValue
        allC[x,y]=np.max(corr)




from scipy import signal

numDsets=np.max(dset).astype('int32')

allC=np.zeros([numDsets,numDsets])
for x in range(0,numDsets):
    dset1=mappedRep[np.where(dset==x)[0],:]
    corr = signal.correlate(dset1,dset1,'same')
    normValue=np.max(corr)
    
    for y in range(0,numDsets):
        dset2=mappedRep[np.where(dset==y)[0],:]
        corr = signal.correlate(dset1,dset2,'same')
        corr /= normValue
        allC[x,y]=np.max(corr)
        
        
        

#normalization

# embedding2 = reducer.transform(mappedRep2)
# # import hdbscan


heatmap, xedges, yedges = np.histogram2d(embedding[:, 0], embedding[:, 1], bins=50)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
octInd=[0,1,2]
for currOct in octInd:
    currInds=np.where(allOct[sketch_index]==currOct)[0]
    heatmap, xedges, yedges = np.histogram2d(embedding[currInds, 0], embedding[currInds, 1], bins=50)
    plt.figure()
    plt.imshow(heatmap.T, extent=extent, origin='lower',vmin=0, vmax=10)


# def smooth(y, box_pts):
#     box = np.ones(box_pts)/box_pts
#     y_smooth = np.convolve(y, box, mode='same')
#     return y_smooth

# from scipy import stats
# mappedRep=PCAmodel.transform(feats)

# a=smooth(mappedRep[:,0],21)
# diffA=np.abs(np.diff(a))
# plt.figure()
# #plt.plot(a)
# plt.plot(diffA)
# b=stats.zscore(diffA)
# slowRegions=np.squeeze(np.argwhere(b<0))

# mappedRep=mappedRep[slowRegions,:]
# dset=dset[slowRegions]




plt.figure()
for d in range(0,int(np.max(refinedDsets))):
    plt.plot(mp[np.argwhere(refinedDsets==d)]+d)



# #oct 4 return day shows increasing length over the day
# mp=np.mean(mappedRep,axis=1)
# mp=mp/max(mp)
plt.figure()
for d in range(0,int(np.max(refinedDsets))):
    plt.plot(mp[np.argwhere(refinedDsets==d)]+d)



fig=plt.figure()
plt.scatter(mappedRep[:, 0], mappedRep[:, 1],c=clustering)






from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10,init='k-means++', random_state=0).fit(X_sketch)
clustering=kmeans.labels_


import umap
reducer = umap.UMAP(min_dist=.1, n_neighbors=5)
embedding=reducer.fit(feats[range(0,mappedRep.shape[0],100)])
embedding = reducer.transform(mappedRep)
embedding2 = reducer.transform(mappedRep2)
# import hdbscan
# clusterer = hdbscan.HDBSCAN(min_cluster_size=200)
# clustering = clusterer.fit_predict(mappedRep)

plt.figure()
plt.scatter(embedding[:, 0], embedding[:, 1],s=1)
plt.scatter(embedding2[:, 0], embedding2[:, 1],s=1)


plt.figure()
plt.scatter(embedding[:, 0], embedding[:, 1],s=.1,c=allOct)
plt.scatter(embedding2[:, 0], embedding2[:, 1],s=1)



,c=clustering)


fig=plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(mappedRep[:, 0], mappedRep[:, 1],mappedRep[:, 2],c=clustering)



#%% once I agree on a clustering
plt.figure()
allArr=np.zeros((len(af),np.max(clustering)+1))
dSetNum=np.unique(dset).astype('int32')
for i,d in enumerate(dSetNum):
    currInd=np.squeeze(np.argwhere(dset==d))
    currClust=clustering[currInd]
    currBins=np.bincount(currClust)
    allArr[i,0:currBins.shape[0]]=currBins/np.sum(currBins)

scaler = StandardScaler()
allArr = scaler.fit_transform(allArr)
# plt.imshow(allArr)

allArr = np.delete(allArr, [46,64,70],axis=0)
allOcttmp = np.delete(allOct, [46,64,70],axis=0)
allArr=allArr[:-4]
allOcttmp=allOcttmp[:-4]

# from sklearn.model_selection import train_test_split
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# X_train, X_test, y_train, y_test = train_test_split(allArr, allOcttmp, test_size=0.5)


# from sklearn.linear_model import SGDClassifier
# clf = SGDClassifier()
# # fit (train) the classifier
# clf.fit(X_train, y_train)

# from sklearn import metrics
# y_test_pred = clf.predict(X_test)
# print (metrics.accuracy_score(y_test, y_test_pred))



#Create a new classifier: a pipeline of the standarizer and the linear model. 
#Measure the cross-validation accuracy.

from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
# create a composite estimator made by a pipeline of the standarization and the linear model
#clf = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])

clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=1000, tol=1e-3))
# create a k-fold croos validation iterator of k=5 folds
cv = KFold(n_splits= 30, shuffle=True, random_state=33)
# by default the score used is the one returned by score method of the estimator (accuracy)
scores = cross_val_score(clf, allArr, allOcttmp, cv=cv)
scores2 = cross_val_score(clf, allArr, np.random.permutation(allOcttmp), cv=cv)

plt.figure()
plt.bar([0,1], [np.mean(scores),np.mean(scores2)], yerr=[np.std(scores),np.std(scores2)], align='center', alpha=0.5, ecolor='black', capsize=10)


plt.figure()
plt.scatter(scores2,scores)
plt.xlim([0,1])
plt.ylim([0,1])

lda = LinearDiscriminantAnalysis(n_components=2, store_covariance=True)


X_r2 = lda.fit(X_train,y_train).transform(X_test)

plt.figure()
plt.scatter(X_r2[:,0],X_r2[:,1],c=y_test)
    
r=r=np.corrcoef(allArr)
plt.figure()
plt.imshow(r)
    
    




#%%
plt.scatter(mappedRep[:,0],mappedRep[:,1],c)




r=np.corrcoef(mappedRep)

#maybe this will work with more data?
import umap
# pca = PCA(n_components=50)
# pcaRep = pca.fit(matforPCA)

reducer = umap.UMAP(random_state=42)
embedding=reducer.fit_transform(mappedRep)
plt.figure()
plt.scatter(embedding[:, 0], embedding[:, 1])


















    print(tally)
    tally+=1    
    
animalTex=np.array(vggRep)    
    
    


totalRep=np.concatenate((fabricTex,animalTex),axis=0)


tally=0
for img in video_iter:
    
    if tally % 25 == 0:
        

    
vggRep=np.array(vggRep)
pixelRep=np.array( imgCrop)
vggRep5=vggRep[:,348161:]
vggRep1=vggRep[:,:4097]

c=[]
for i in range(0,30):
    c=np.concatenate((c, i*np.ones(numCrops)))
    
    
c=np.concatenate((c, (i+1)*np.ones(3)))

c=range(totalRep.shape[0])

from sklearn import preprocessing
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import distance_matrix
import numpy.matlib
import scipy.io
from mpl_toolkits.mplot3d import Axes3D

#X_train=preprocessing.scale(totalRep)
X_animal=preprocessing.scale(animalTex)
X_fabric=preprocessing.scale(fabricTex)


pca = PCA(n_components=20)  #check this
fabricPCA=pca.fit(X_fabric) 

mappedFabric=fabricPCA.transform(X_fabric)
mappedAnimal=fabricPCA.transform(X_animal)

c=[]
for i in range(0,2):
    c=np.concatenate((c, i*np.ones(numCrops)))
fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mappedFabric[:, 0], mappedFabric[:, 1],mappedFabric[:, 2],c=c)

c=range(mappedAnimal.shape[0])
ax.scatter(mappedAnimal[:, 0], mappedAnimal[:, 1],mappedAnimal[:, 2],c=c)
ax.plot3D(mappedAnimal[:, 0], mappedAnimal[:, 1],mappedAnimal[:, 2])


#X_train=preprocessing.scale(pixelRep)

pca = PCA(n_components=20)  #check this
fabricPCA=pca.fit(X_train) 
plt.figure()
plt.imshow(mappedX,aspect='auto')
plt.figure()
plt.scatter(mappedX[:, 0], mappedX[:, 1],c=c)
plt.scatter(mappedX[-2:, 0], mappedX[-2:, 1],c=c)



pca = PCA(n_components=20)  #check this
animalPCA=pca.fit(X_animal) 
mappedFabric=animalPCA.transform(X_fabric)
mappedAnimal=animalPCA.transform(X_animal)

fabricUmap = umap.UMAP(n_components=3, n_neighbors=200,min_dist=0,random_state=42).fit(mappedAnimal)

fUM=fabricUmap.transform(mappedFabric)
aUM=fabricUmap.transform(mappedAnimal)
                        
                        
c=[]
for i in range(0,4):
    c=np.concatenate((c, i*np.ones(numCrops)))
fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(fUM[:, 0], fUM[:, 1],fUM[:, 2],c=c)




from mpl_toolkits import mplot3d
fig=plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(aUM[:, 0], aUM[:, 1],aUM[:, 2],c=c)

c=range(aUM.shape[0])
ax.plot3D(aUM[:, 0], aUM[:, 1],aUM[:, 2])








d=distance_matrix(mappedX, mappedX)
plt.figure()
plt.imshow(d)

d=distance_matrix(trans, trans)
plt.figure()
plt.imshow(d)

plt.figure()
plt.imshow(d[1::16,1::16])

matStr = '/home/reiters/Documents/analysisImages/visualizations.mat'
scipy.io.savemat(matStr, mdict={'umap': trans, 'pca': mappedX, 'dist': d})   
