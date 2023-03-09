%% Pattern analysis

%compute pattern space

%~60 dimensional, consistent over animals and in total

%highly overlapping over animals and days: siloette score


%an active sleep bout traces out a trajectory in this space

%and overlapping
%distributions in nearest points (all-all vs within octopus). Should look
%within a trajectory vs between. Also do trajectory to nearest pt in other
%trajectories from same octopus vs different


%looking over time, trajectories differ and do not seperate over days and
%octopuses

%even warping time to control for the same set of patterns held for
%different sets of time, we see trajectories differ. Relative to something? Maybe relative to
%starting pattern?

%this means similar points in the space are revisited in different orders
%over different active sleep bouts across octopuses. Pattern comparison

%Wake data also falls in this space, with a unified set of patterns in wake
%and sleep, and waking patterns reactivated during different sleep bouts.



%edf with distribution, wake data for each octopus
%trajectory examples

%edf with  calculation of dimensionality, distances, silloette score, same example, using
%different layer of the neural net


dataFolder='/home/sam/bucket/octopus/8k/oct_456';
suffix='reg_features_allFeats';


behaveFile=[dataFolder '/behaviorWork'];
feats=double(h5read(behaveFile,'/feats'))';
oct=double(h5read(behaveFile,'/oct'))+1;
dset=double(h5read(behaveFile,'/dset'))+1;
ht=double(h5read(behaveFile,'/ht'));

%check ht on
%'/home/sam/bucket/octopus/8k/oct_456/OCT14832.reg_features_conv5pool' and
%'/home/sam/bucket/octopus/8k/oct_456/OCT14947.reg_features_conv5pool'
%'/home/sam/bucket/octopus/8k/oct_456/OCT14781.reg_features_conv5pool'

wakeFeats=double(h5read(behaveFile,'/featsB'))';
wakeOct=double(h5read(behaveFile,'/octB'))';
wakeOct(wakeOct==4)=1;
wakeOct(wakeOct==5)=2;
wakeOct(wakeOct==6)=3;

wakeDataFolder='/home/sam/bucket/octopus/8k/awake/good_extracted_frames'
wakeSuffix='reg_allFeats';
wakeHT=ones(1,size(wakeFeats,1));

allFiles = dir([dataFolder '/*' suffix]);
af=[];
embedFiles=[];
for x=1:numel(allFiles)
    af{x}=[allFiles(x).folder '/' allFiles(x).name];
    basename=split(allFiles(x).name,'_');
    embedFiles{x}=[allFiles(x).folder '/' basename{1}];
end
allFiles=af;


T=readtable('/home/sam/bucket/octopus/8k/oct_456/allDays.csv','Delimiter',',');
%days=T{:,3};
%octNum=T{:,4}+1;
%% Truncating data to AS times   %can take the whole thing for pca, but should trim for alignments in time

moveThresh=.1 %for now keep trunc dset aas dset, later adjust

%need to remove time before as
numDsets=max(dset);
mX=mean(feats);
[W,mappedRep,latent] = pca(feats);
behaveRep= (wakeFeats-mX)*W;


[bb,aa]=butter(3,[.1/15],'low');

currFeats=feats(dset==1,:);
currDset=dset(dset==1);

filtRep=filtfilt(bb,aa,mappedRep(dset==1,1));
moveTime=diff(filtRep);
asStart=find(abs(moveTime)>moveThresh,1);
asEnd=length(currDset);
% asEnd=min(find(abs(moveTime)>moveThresh,1,'last'),asEnd);

dsetInds=asStart:asEnd;
truncFeats=currFeats(asStart:asEnd,:);
truncDset=currDset(asStart:asEnd);
truncOct=ones(length(asStart:asEnd),1);

for x=2:numDsets
    
    currFeats=feats(dset==x,:);
    currDset=dset(dset==x);
    
    filtRep=filtfilt(bb,aa,mappedRep(dset==x,1));
    moveTime=diff(filtRep);
    asStart=1;
    if length(find(abs(moveTime)>moveThresh,1))==1
        asStart=find(abs(moveTime)>moveThresh,1);
    end
    asEnd=length(currDset);
    % asEnd=min(find(abs(moveTime)>moveThresh,1,'last'),asEnd);
    
    dsetInds=[dsetInds asStart:asEnd];
    truncFeats=[truncFeats; currFeats(asStart:asEnd,:)];
    truncDset=[truncDset; currDset(asStart:asEnd)];
    truncOct=[truncOct; oct(x)*ones(length(asStart:asEnd),1)];
    x;
end

mX=mean(truncFeats);
[pcs,truncRep,latent] = pca(truncFeats); %need to remove time before as

behaveRep= (wakeFeats-mX)*pcs;


% figure
% for x=1:max(truncDset)
%     currFeats=mappedRep(dset==x,:);
%     plot(currFeats(:,1))
%     title([num2str(x)])
% pause
% clf
% end


h5create([dataFolder  '/reducedFeats'],'/feats',size(truncFeats))
h5write([dataFolder  '/reducedFeats'],'/feats',truncFeats)
h5create([dataFolder  '/reducedFeats'],'/oct',size(truncOct))
h5write([dataFolder  '/reducedFeats'],'/oct',truncOct)

%% Dimensionality estimate

N_sub=10000;
alpha = 0.05;
rng(42)

latentAll=[];latentLowAll=[];
for x=1:10
    randSamps = truncFeats(randsample(size(truncFeats,1),N_sub),:);
    [latentAll(x,:), latentLowAll(x,:), latentHighAll] = pa_test(zscore(randSamps));
    numDims(x)=find(latentHighAll>latentAll(x,:),1)
end

figure
hold on
plot(latentAll'./sum(latentAll'),'k','linewidth',3);
plot(latentHighAll'./sum(latentAll'),'r');
plot(latentLowAll'./sum(latentAll'),'r');

xlabel('dimensionality')
ylabel('variance explained')
axis tight
xlim([0 100])
set(gca, 'YScale', 'log')
%% Calculating overlap in feature space occupancy over octopus
%silhoette score
rng(42)
numDims=6;%60
sampSizes=10000;
faceColor=[.1,.5,.8];

s=[];
for repeat=1:10
    s(repeat) = mean(mySilhouette(truncRep(:,1:numDims),truncOct,sampSizes));
end
%6 dim, s=0.076 +/-3.148

o1=[.1,.3,.7];
o1=[.5,.5,.8];
o2=[1,.7,0];
o3=[.7,.1,.7];
dataset_colors = {o1,o2,o3};


for o=1:3
    figure
    hold on
    N= hist3(truncRep(truncOct==o,1:2),'Edges',{-80:2:130 -90:2:100 });
    imagesc(imgaussfilt(N'./max(N(:)),2));
    % myColorMap = jet(256);
    % myColorMap(1,:) = 1;
    %colormap(myColorMap);
    colormap jet
    caxis([0 .7])
    % xlim([10 120])
    % ylim([10 100])
    set(gca,'YDir','normal')
    title(num2str(o))
    % colorbar
    figure
    hold on
    scatter(truncRep(truncOct==o,1),truncRep(truncOct==o,2),.5,'k','filled')
    scatter(behaveRep(wakeOct==o,1),behaveRep(wakeOct==o,2),20,'r','filled')
    xlim([-80 130])
    ylim([-90 100])
    title(num2str(o))
end





%% display trajectories and data points
o1=[.6,.6,.6];
o2=[1,.7,0];
o3=[.7,.1,.7];
dataset_colors = {o1,o2,o3};

%dsetsToShow=[1, 15,81,84,24,28,38,40,44,46,50,61]

dsetsToShow=[4,25,44,5,38,52,78,34,54]

%dsetsToShow=[36:46]
figure

for x=1:20
    subplot(3,3,x)
    hold on
    % for o=1:3
    % scatter(truncRep(:,1),truncRep(:,2),.3,dataset_colors{1},'filled')
    % end
    currDset=dsetsToShow(x)
    currInds=find(truncDset==currDset);
    a1=smooth(truncRep(truncDset==currDset,1),5);
    a2=smooth(truncRep(truncDset==currDset,2),5);
    a3=smooth(truncRep(truncDset==currDset,3),5);
    
    plot(a1,a2,'k')
    scatter(truncRep(currInds,1),truncRep(currInds,2),3,1:numel(currInds),'filled')
    colormap jet
    xlim([-80 130])
    ylim([-90 100])
    title(['oct ' num2str(truncOct(currInds(1))) '-' num2str(dsetsToShow(x))])
    caxis([1 2700])
end

%% 11,29,73

for xi=11
    figure
    hold on
    currDset=xi%cds(xi)
    a1=smooth(truncRep(truncDset==currDset,1),5);
    a2=smooth(truncRep(truncDset==currDset,2),5);
    a3=smooth(truncRep(truncDset==currDset,3),5);
    color=1:numel(find(truncDset==currDset));
    plot(a1,a2,'k','linewidth',1)
    scatter(a1,a2,5,color,'filled')
    
    
    axis square
    xlim([-70 140])
    xlabel('PC 1')
    ylabel('PC 2')
    %colorbar
    title(num2str(xi))
    
    
    % nearest points along the trajectory
    currInds=find(truncDset==currDset);
    %indsToPull=1:250:length(currInds);
    indsToPull=1:300:2300;
    h=1004;w=675;
    exampleNum=1;
    numDims=6%512;
    
    colormap jet
    title(currDset)
    ylim([-75 75])
    xlim([-70 140])
    caxis([1 2500])
    set(gcf,'position',[1,10,400,400])
    %end
    %
    
    
    
    %for xi=38
    
    currImg=zeros(h*3*exampleNum+1,w*numel(indsToPull),3,'uint8');
    allClosestDset=zeros(3*exampleNum+1,numel(indsToPull));
    allClosestDset(1,:)=currDset;
    %plot original trajectory
    for x=1:numel(indsToPull)
        currImg(1:h,((x-1)*w+1):(w*x),:) = pullPattern(dataFolder, suffix,ht,currDset,dsetInds(currInds(indsToPull(x)))); %remember this will go back to original dset, if I truncate I need to give the rignt indices
    end
    
    
    %find closest patterns on other trajectories
    for x=1:numel(indsToPull)
        currentIndex=currInds(indsToPull(x));
        Idx=[];D=[];
        for d=1:numel(allFiles)
            [Idx(d),D(d)] = knnsearch(truncFeats(truncDset==d,1:numDims),truncFeats(currentIndex,1:numDims));
            truncAddition=dsetInds(truncDset==d);
            Idx(d)=Idx(d)+truncAddition(1);
        end
        
        %exclude current index
        D(D==0)=1000;
        
        tally=1;
        for o=1:3
            currDsets=find(oct==o);
            currI=Idx(oct==o);
            currD=D(oct==o);
            
            [~,sortDist]=sort(currD);
            closestDsets=currDsets(sortDist);
            dsetRelIndices=currI(sortDist)';
            
            for i=1:exampleNum
                currImg((tally*h+1):((tally+1)*h),((x-1)*w+1):(w*x),:) = pullPattern(dataFolder, suffix,ht,closestDsets(i),dsetRelIndices(i));
                allClosestDset(tally+1,x)=closestDsets(i);
                tally=tally+1;
                
            end
        end
        
        x
    end
    
    figure
    imshow(currImg)
end


%% Waking pattern examples
% nearest points along the trajectory


indsToPull=[61:90 ];

currDset=1
h=1004;w=675;
exampleNum=1;

currImg=zeros(h*3*exampleNum+1,w*numel(indsToPull),3,'uint8');
allClosestDset=zeros(3*exampleNum+1,numel(indsToPull));
allClosestDset(1,:)=currDset';
%plot original trajectory
for x=1:numel(indsToPull)
    currImg(1:h,((x-1)*w+1):(w*x),:) = pullPattern(wakeDataFolder, wakeSuffix,wakeHT,1,indsToPull(x)); %remember this will go back to original dset, if I truncate I need to give the rignt indices
end

for x=1:numel(indsToPull)
    Idx=[];D=[];
    for d=1:numel(allFiles)
        [Idx(d),D(d)] = knnsearch(truncFeats(truncDset==d,:),wakeFeats(indsToPull(x),:));
        truncAddition=dsetInds(truncDset==d);
        Idx(d)=Idx(d)+truncAddition(1);
    end
    
    tally=1;
    for o=1:3
        currDsets=find(oct==o);
        currI=Idx(oct==o);
        currD=D(oct==o);
        
        [~,sortDist]=sort(currD);
        closestDsets=currDsets(sortDist);
        dsetRelIndices=currI(sortDist)';
        
        for i=1:exampleNum
            currImg((tally*h+1):((tally+1)*h),((x-1)*w+1):(w*x),:) = pullPattern(dataFolder, suffix,ht,closestDsets(i),dsetRelIndices(i));
            allClosestDset(tally+1,x)=closestDsets(i);
            tally=tally+1;
            
        end
    end
    
    x
end
figure
imshow(currImg)


%%  Distance calculations

%how close are points on a trajectory?
numD=6; %60

nextPtDist=zeros(numDsets,1);
for x=1:numDsets
    currDset=truncFeats(truncDset==x,1:numD);
    nextPtDist(x)=mean(sqrt(sum(diff(currDset).^2,2)));
    x
end

%shortest dist: how close is one trajectory to any other one, on average
shortestDist=zeros(numDsets,numDsets);
avgDist=zeros(numDsets,numDsets);
for x=1:numDsets
    for y=1:numDsets
        if x>y %its symmetric
            D=pdist2(truncRep(truncDset==x,1:numD),truncRep(truncDset==y,1:numD),'euclidean');
            avgDist(x,y)=mean(diag(D));
            shortestDist(x,y) = min(D(:));
        end
    end
    x
end

%dynamic time warping?
dtwDist=zeros(numDsets,numDsets);
for x=1:numDsets
    for y=1:numDsets
        if x>y
            [ totDist,ix,iy]=dtw(truncRep(truncDset==x,1:numD)',truncRep(truncDset==y,1:numD)');
            dtwDist(x,y)=totDist./numel(ix); %average distance
        end
    end
    x
end

%sleep/wake distances
currS=truncRep(:,1:6);
currA=behaveRep(:,1:6);

D=pdist2(truncRep(:,1:numD),behaveRep(:,1:numD),'euclidean');
swD=min(D);



figure
hold on
histogram(nextPtDist,0:4:140,'Normalization','probability')
sd=tril(shortestDist);
sd(sd==0)=[];
histogram(sd,0:4:140,'Normalization','probability')
ad=tril(avgDist);
ad(ad==0)=[];
histogram(ad,0:4:140,'Normalization','probability')

xlabel('distance')
ylabel('probability')


figure
hold on
dtwd=tril(dtwDist);
dtwd(dtwd==0)=[];
histogram(dtwd,'Normalization','probability')
histogram(swD,0:4:140,'Normalization','probability')

xlabel('distance')
ylabel('probability')


%% additional display trajectories for edf

o1=[.6,.6,.6];
o2=[1,.7,0];
o3=[.7,.1,.7];
dataset_colors = {o1,o2,o3};


dsetsToShow=[1, 4, 11,21,72,39,49,24,5]

figure

for x=1:20
    subplot(3,4,x)
    hold on
    for o=1:3
        scatter(truncRep(truncOct==o,1),truncRep(truncOct==o,2),.3,dataset_colors{1},'filled')
    end
    currDset=dsetsToShow(x)
    currInds=find(truncDset==currDset);
    a1=smooth(truncRep(truncDset==currDset,1),5);
    a2=smooth(truncRep(truncDset==currDset,2),5);
    a3=smooth(truncRep(truncDset==currDset,3),5);
    
    plot(a1,a2,'k')
    scatter(truncRep(currInds,1),truncRep(currInds,2),3,1:numel(currInds),'filled')
    colormap jet
    axis tight
    title(['oct ' num2str(truncOct(currInds(1))) '-' num2str(dsetsToShow(x))])
end


%% Wake sleep matching stuff

patternMatchingDir='/home/sam/bucket/octopus/8k/pattern_matching'
cd(patternMatchingDir)
T=readtable('pattern_matching_log1.csv','Delimiter',',');

o=5

srcName=T{2,o}
trgName=T{3,o}
filename=[trgName{1} '.png_matchFeatures']
srcPts=double(h5read(filename,'/tempPtsSrc')');
trgPts=double(h5read(filename,'/tempPtsTrg')');
srcName=[srcName{1} '.png'];
trgName=[trgName{1} '.png'];
img1=imread(srcName);
img2=imread(trgName);
tform = fitgeotrans(trgPts,srcPts,'similarity')
img2AW = imwarp(img2,tform,'OutputView',imref2d(size(img2)));
img2Warped=imread([trgName '_trgImgWarped.png']);
img1C=img1;
img2WC=img2Warped;

%dataset specific rotate
img1=imrotate(img1,rotateAng(f));
img1C=img1(3100:6800,3300:5800,:);

img2AW=imrotate(img2AW,rotateAng(f));
img2Warped=imrotate(img2Warped,rotateAng(f));
img2WC=img2Warped(3100:6800,3300:5800,:);

figure
imshow(img2AW(3100:6800,2800:5300,:)*1.5);

figure
imshow(img1C*1.2);
rectangle('pos',[500 1700 1100 1695],'edgecolor','m','linewidth',3)

img1g=rgb2gray(img1C);
img2g=rgb2gray(img2WC);
thresh1=180;
thresh2=210;
img1T=imcomplement(img1g);
img1T(img1T<thresh1)=0;
img2T=imcomplement(img2g);
img2T(img2T<thresh2)=0;
C=imfuse(img1T,img2T);
figure
imshow(C)
ylim([1700 3295])
xlim([500 1600])



o=3

srcName=T{2,o}
trgName=T{3,o}
filename=[trgName{1} '.png_matchFeatures']
srcPts=double(h5read(filename,'/tempPtsSrc')');
trgPts=double(h5read(filename,'/tempPtsTrg')');
srcName=[srcName{1} '.png'];
trgName=[trgName{1} '.png'];
img1=imread(srcName);
img2=imread(trgName);
tform = fitgeotrans(trgPts,srcPts,'similarity')
img2AW = imwarp(img2,tform,'OutputView',imref2d(size(img2)));
img2Warped=imread([trgName '_trgImgWarped.png']);
img1C=img1;
img2WC=img2Warped;

%dataset specific rotate
img1=imrotate(img1,-50);
img1C=img1(1100:2700,1900:2800,:);

img2AW=imrotate(img2AW,-50);
img2Warped=imrotate(img2Warped,-50);
img2WC=img2Warped(1100:2700,1900:2800,:);

figure
imshow(img2AW(1100:2700,1900:2800,:)*1.4);

figure
imshow(img1C*1.4);
rectangle('pos',[213 613 400 580],'edgecolor','m','linewidth',3)

img1g=rgb2gray(img1C);
img2g=rgb2gray(img2WC);
thresh1=215;
thresh2=215;
img1T=imcomplement(img1g);
img1T(img1T<thresh1)=0;
img2T=imcomplement(img2g);
img2T(img2T<thresh2)=0;
C=imfuse(img1T,img2T);
figure
imshow(C)
ylim([613 1193])
xlim([213 613])



o=4

srcName=T{2,o}
trgName=T{3,o}
filename=[trgName{1} '.png_matchFeatures']
srcPts=double(h5read(filename,'/tempPtsSrc')');
trgPts=double(h5read(filename,'/tempPtsTrg')');
srcName=[srcName{1} '.png'];
trgName=[trgName{1} '.png'];
img1=imread(srcName);
img2=imread(trgName);
tform = fitgeotrans(trgPts,srcPts,'similarity')
img2AW = imwarp(img2,tform,'OutputView',imref2d(size(img2)));
img2Warped=imread([trgName '_trgImgWarped.png']);
img1C=img1;
img2WC=img2Warped;

img1=imrotate(img1,90);
img1C=img1(1800:3200,700:1650,:);

img2AW=imrotate(img2AW,90);
img2Warped=imrotate(img2Warped,90);
img2WC=img2Warped(1800:3200,700:1650,:);

figure
imshow(img2AW(1700:3100,700:1650,:)*1.5);

figure
imshow(img1C*1.5);
rectangle('pos',[300 520 400 580],'edgecolor','m','linewidth',3)


img1g=rgb2gray(img1C);
img2g=rgb2gray(img2WC);
thresh1=180;
thresh2=180;
img1T=imcomplement(img1g);
img1T(img1T<thresh1)=0;
img2T=imcomplement(img2g);
img2T(img2T<thresh2)=0;
C=imfuse(img1T,img2T);
figure
imshow(C)
ylim([520 1100])
xlim([300 700])




o=7

srcName=T{2,o}
trgName=T{3,o}
filename=[trgName{1} '.png_matchFeatures']
srcPts=double(h5read(filename,'/tempPtsSrc')');
trgPts=double(h5read(filename,'/tempPtsTrg')');
srcName=[srcName{1} '.png'];
trgName=[trgName{1} '.png'];
img1=imread(srcName);
img2=imread(trgName);
tform = fitgeotrans(trgPts,srcPts,'similarity')
img2AW = imwarp(img2,tform,'OutputView',imref2d(size(img2)));
img2Warped=imread([trgName '_trgImgWarped.png']);
img1C=img1;
img2WC=img2Warped;

rotate=-95;
img1=imrotate(img1,rotate);
img1C=img1(750:2250,1500:2450,:);

img2AW=imrotate(img2AW,rotate);
img2Warped=imrotate(img2Warped,rotate);
img2WC=img2Warped(750:2250,1500:2450,:);

figure
imshow(img2AW(750:2250,1500:2450,:)*1.7);

figure
imshow(img1C*1.7);
rectangle('pos',[300 520 400 580],'edgecolor','m','linewidth',3)


img1g=rgb2gray(img1C);
img2g=rgb2gray(img2WC);
thresh1=230;
thresh2=230;
img1T=imcomplement(img1g);
img1T(img1T<thresh1)=0;
img2T=imcomplement(img2g);
img2T(img2T<thresh2)=0;
C=imfuse(img1T,img2T);
figure
imshow(C)
ylim([520 1100])
xlim([300 700])



o=9

srcName=T{2,o}
trgName=T{3,o}
filename=[trgName{1} '.png_matchFeatures']
srcPts=double(h5read(filename,'/tempPtsSrc')');
trgPts=double(h5read(filename,'/tempPtsTrg')');
srcName=[srcName{1} '.png'];
trgName=[trgName{1} '.png'];
img1=imread(srcName);
img2=imread(trgName);
tform = fitgeotrans(trgPts,srcPts,'similarity')
img2AW = imwarp(img2,tform,'OutputView',imref2d(size(img2)));
img2Warped=imread([trgName '_trgImgWarped.png']);
img1C=img1;
img2WC=img2Warped;

rotate=87;
img1=imrotate(img1,rotate);
img1C=img1(750:2250,1500:2450,:);

img2AW=imrotate(img2AW,rotate);
img2Warped=imrotate(img2Warped,rotate);
img2WC=img2Warped(750:2250,1500:2450,:);

figure
imshow(img2AW(750:2250,1500:2450,:)*1.5);

figure
imshow(img1C*1.5);
rectangle('pos',[300 640 400 580],'edgecolor','m','linewidth',3)


img1g=rgb2gray(img1C);
img2g=rgb2gray(img2WC);
thresh1=190;
thresh2=190;
img1T=imcomplement(img1g);
img1T(img1T<thresh1)=0;
img2T=imcomplement(img2g);
img2T(img2T<thresh2)=0;
C=imfuse(img1T,img2T);
figure
imshow(C)
ylim([640 1220])
xlim([300 700])



o=11

srcName=T{2,o}
trgName=T{3,o}
filename=[trgName{1} '.png_matchFeatures']
srcPts=double(h5read(filename,'/tempPtsSrc')');
trgPts=double(h5read(filename,'/tempPtsTrg')');
srcName=[srcName{1} '.png'];
trgName=[trgName{1} '.png'];
img1=imread(srcName);
img2=imread(trgName);
tform = fitgeotrans(trgPts,srcPts,'similarity')
img2AW = imwarp(img2,tform,'OutputView',imref2d(size(img2)));
img2Warped=imread([trgName '_trgImgWarped.png']);
img1C=img1;
img2WC=img2Warped;

rotate=-36;
img1=imrotate(img1,rotate);
img1C=img1(1800:3200,1700:2750,:);

img2AW=imrotate(img2AW,rotate);
img2Warped=imrotate(img2Warped,rotate);
img2WC=img2Warped(1800:3200,1700:2750,:);

figure
imshow(img2AW(1800:3200,1700:2750,:)*1.5);

figure
imshow(img1C*1.5);
rectangle('pos',[500 620 400 580],'edgecolor','m','linewidth',3)


img1g=rgb2gray(img1C);
img2g=rgb2gray(img2WC);
thresh1=190;
thresh2=190;
img1T=imcomplement(img1g);
img1T(img1T<thresh1)=0;
img2T=imcomplement(img2g);
img2T(img2T<thresh2)=0;
C=imfuse(img1T,img2T);
figure
imshow(C)
ylim([620 1200])
xlim([500 900])




