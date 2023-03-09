%initialize ephys recordings
initEphys

%% spindle analysis


% pull example traces for figure from oct 3
currOct=3
dsets= find(oct==currOct);
bl=brainLoc(dsets(1)); %1 for FL, 2 for VL

dset=2
currDset=dsets(dset);
[dataFolder, rec]=fileparts(dataFolderList{currDset});
cd(dataFolder)
analysisFile=[rec '_analysis'];
load([rec '_behaviorAnalysis.mat']);
load([rec '_selectChannel.mat']);
indToUse=lfpInds(dsets(1));
allLFP=allLFP(3,:);
if length(allLFP)>length(mi)
    allLFP=allLFP(1:length(mi));
else
    mi=mi(1:length(allLFP));
    wakeVec=wakeVec(1:length(allLFP));
end

%load('/home/sam/bucket/octopus/electrophysiology/oct3_spindleAnalysis.mat')

[peaks,locs]=detectSpindles(allLFP, wakeVec,[],bbSpindle,aaSpindle,spindleMinHeight,spindleMinProm, spindleMinDist);
[ev_avg,lags,allSpindleMat] = getEventTrigAvg(allLFP,locs,spindleBackLag,spindleForLag);


figure
hold on
plot((1:length(allLFP))/60000,allLFP,'k')
scatter(locs/60000,320*ones(1,length(locs)),'kv')
xlim([294.2 294.4])

[s wfreqs] = wt(allLFP(294.2*60000:294.4*60000),1000,0);
sNorm=s./max(s(:)) ;
figure
pcolor((1:size(s,2))/1000,wfreqs,sNorm)
shg
shading flat
ylabel('frequency (Hz)')
xlabel('time (s)')
caxis([0 .7])
ylim([1 60])
% xlim(xlimits/1000)
colormap(jet)


figure
hold on
plot((1:length(allLFP))/60000,allLFP,'k')
scatter(locs/60000,320*ones(1,length(locs)),'kv')
xlim([294.332 294.349])

[s wfreqs] = wt(allLFP(294.332*60000:294.349*60000),1000,0);
sNorm=s./max(s(:)) ;
figure
pcolor((1:size(s,2))/1000,wfreqs,sNorm)
shg
shading flat
ylabel('frequency (Hz)')
xlabel('time (s)')
caxis([0 .65])
ylim([1 60])
% xlim(xlimits/1000)
colormap(jet)

%Spindle spectrogram
% peakFreq=zeros(1,size(allSpindleMat,2));
sAll=zeros(100,size(allSpindleMat,2));
for x=1:size(allSpindleMat,1)
    [s wfreqs] = wt(allSpindleMat(x,:),1000,0);
    %     maxPower=max(s');
    %     peakFreq(x)=wfreqs(find(maxPower==max(maxPower(1:60))));
    sAll=sAll+s;
    x
end
sAllN=sAll./x;
sNorm=sAllN./max(sAllN(:)) ;
%spectrogram
figure
pcolor((1:size(sAllN,2))/1000,wfreqs,sNorm)
shg
shading flat
ylabel('frequency (Hz)')
xlabel('time (s)')
caxis([0 .65])
ylim([1 60])
xlim([.3 1.3])
colormap(jet)


cutoffPercent= length(find((diff(locs)/1000)>90)) %18
totalSpindles=length(locs) %1896

%relation with behavior
[ev_avg1,lags,behaveMat] = getEventTrigAvg(mi,locs,3000,100000);
[ev_avg2,lags,asBehaveMat] = getEventTrigAvg(mi,asTimes,3000,100000);


figure
plot(ev_avg1)


%% EDFs for QS, IEIs, VL data, relation to movement

%bce example
currOct=3
dsets= find(oct==currOct);
bl=brainLoc(dsets(1)); %1 for FL, 2 for VL

dset=2
currDset=dsets(dset);
[dataFolder, rec]=fileparts(dataFolderList{currDset});
cd(dataFolder)
analysisFile=[rec '_analysis'];
load([rec '_behaviorAnalysis.mat']);
load([rec '_selectChannel.mat']);
indToUse=lfpInds(dsets(1));
allLFP=allLFP(3,:);
if length(allLFP)>length(mi)
    allLFP=allLFP(1:length(mi));
else
    mi=mi(1:length(allLFP));
    wakeVec=wakeVec(1:length(allLFP));
end

figure
hold on
plot((1:length(allLFP))/60000,allLFP,'k')
plot((1:length(allLFP))/60000,zscore(mi)*100-200,'r')
scatter(locs/60000,320*ones(1,length(locs)),'kv')
xlim([314  324])

figure
hold on
plot((1:length(allLFP))/60000,allLFP,'k')
plot((1:length(allLFP))/60000,zscore(mi)*100-200,'r')
xlim([321  321.2])



load('/home/sam/bucket/octopus/electrophysiology/oct1_spindleAnalysis.mat','allBCEspindles','spindleISI')
bce1=allBCEspindles
isi1=spindleISI

load('/home/sam/bucket/octopus/electrophysiology/oct2_spindleAnalysis.mat','allBCEspindles','spindleISI')
bce2=allBCEspindles
isi2=spindleISI

load('/home/sam/bucket/octopus/electrophysiology/oct3_spindleAnalysis.mat','allBCEspindles','spindleISI')
bce3=allBCEspindles
isi3=spindleISI

sum([bce1 bce2 bce3])/numel([bce1 bce2 bce3])

figure
hold on
histogram(allSpindleMatBehave(:,3001)-allSpindleMatBehave(:,1),[-10:0.5:10],'Normalization','probability')
histogram(allBCEMatBehave(:,3001)-allBCEMatBehave(:,1),[-10:0.5:10],'Normalization','probability')
histogram(min(allASMatBehave,[],2)-allASMatBehave(:,1),[-10:0.5:10],'Normalization','probability')



% ISI distribution
figure
histogram([isi1 isi2 isi3]/1000,0:5:500,'Normalization','probability')
xlabel('inter event interval (sec)')
ylabel('probability')
xlim([0 500]) %
set(gca, 'YScale', 'log')




% pull example traces for figure from oct 6
currOct=6
dsets= find(oct==currOct);
bl=brainLoc(dsets(1)); %1 for FL, 2 for VL

dset=2
currDset=dsets(dset);
[dataFolder, rec]=fileparts(dataFolderList{currDset});
cd(dataFolder)
analysisFile=[rec '_analysis'];
load([rec '_behaviorAnalysis.mat']);
load([rec '_selectChannel.mat']);
indToUse=lfpInds(dsets(1));
allLFP=allLFP(2,:);
if length(allLFP)>length(mi)
    allLFP=allLFP(1:length(mi));
else
    mi=mi(1:length(allLFP));
    wakeVec=wakeVec(1:length(allLFP));
end



%
[bb,aa]=butter(3,[.5/500 150/500 ],'bandpass');
%
filtLFP=filtfilt(bb,aa,allLFP);
%detect color flashes
[peaks,bceLocs]=detectBCEs(mi(1:40:end), wakeVec(1:40:end),bceLow, bceHigh, 25,bceMin, bceProm,bceDist);
bceLocs=bceLocs*40;

%detect spindles
[peaks,spindleLocs]=detectSpindles(allLFP, wakeVec,bceLocs, bbSpindle,aaSpindle,spindleMinHeight,spindleMinProm,spindleMinDist);


[ev_avg,lags,allSpindleMat] = getEventTrigAvg(allLFP,spindleLocs,spindleBackLag,spindleForLag);


figure
hold on
plot((1:length(allLFP))/60000,filtLFP,'k')
scatter(locs/60000,100*ones(1,length(locs)),'kv')
xlim([286.2  286.7])

[s wfreqs] = wt(allLFP(286.2*60000:286.7*60000),1000,0);
sNorm=s./max(s(:)) ;
figure
pcolor((1:size(s,2))/1000,wfreqs,sNorm)
shg
shading flat
ylabel('frequency (Hz)')
xlabel('time (s)')
caxis([0 0.6])
ylim([1 40])
% xlim(xlimits/1000)
colormap(jet)

figure
hold on
plot((1:length(allLFP))/60000,allLFP,'k')
scatter(locs/60000,100*ones(1,length(locs)),'kv')
xlim([286.357 286.3737])

[s wfreqs] = wt(allLFP(286.357*60000:286.3737*60000),1000,0);
sNorm=s./max(s(:)) ;
figure
pcolor((1:size(s,2))/1000,wfreqs,sNorm)
shg
shading flat
ylabel('frequency (Hz)')
xlabel('time (s)')
caxis([0 0.6])
ylim([1 40])
% xlim(xlimits/1000)
colormap(jet)


%Spindle spectrogram
sAll=zeros(100,size(allSpindleMat,2));
for x=1:size(allSpindleMat,1)
    [s wfreqs] = wt(allSpindleMat(x,:),1000,0);
    sAll=sAll+s;
    x
end
sAllN=sAll./x;
sNorm=sAllN./max(sAllN(:)) ;
%spectrogram
figure
pcolor((1:size(sAllN,2))/1000,wfreqs,sNorm)
shg
shading flat
ylabel('frequency (Hz)')
xlabel('time (s)')
caxis([0 .6])
ylim([1 60])
xlim([.3 1.3])
colormap(jet)


% ISI distribution
figure
histogram(diff(locs)/1000,0:5:100,'Normalization','probability')
xlabel('inter event interval (sec)')
ylabel('probability')
xlim([0 90]) %

cutoffPercent= length(find((diff(locs)/1000)>90)) %69
totalSpindles=length(locs) %1597



%spindles in VL vs SubV
[peaks,spindleLocs]=detectSpindles(allLFP, wakeVec,bceTimes, bbSpindle,aaSpindle,spindleMinHeight,spindleMinProm,spindleMinDist);
[peaks,spindleLocs1]=detectSpindles(allLFP1, wakeVec,bceTimes, bbSpindle,aaSpindle,spindleMinHeight,spindleMinProm,spindleMinDist);

bpp=zeros(1,round(length(allLFP)/100));
bpp(round(spindleLocs1/100))=1;

bpp2=zeros(1,round(length(allLFP1)/100));
bpp2(round(spindleLocs/100))=1;

[xcf,lags] = crosscorr(bpp,bpp2,1000);
[xcf1,lags] = crosscorr(allLFP(1:100:end),allLFP1(1:100:end),1000);

figure
hold on
plot(lags,xcf)
plot(lags,xcf1)

