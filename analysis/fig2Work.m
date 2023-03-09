initEphys;

%% plots for fig 2 spectrograms and spectra

currOct=9
dsets= find(oct==currOct);
bl=brainLoc(dsets(1)); %1 for FL, 2 for VL
indToUse=lfpInds(dsets(1));
dset=2
currDset=dsets(dset);
[dataFolder, rec]=fileparts(dataFolderList{currDset});
cd(dataFolder)
analysisFile=[rec '_analysis'];
load([rec '_behaviorAnalysis.mat']);
load([rec '_selectChannel.mat']);
allLFP=allLFP(indToUse,:);
if length(allLFP)>length(mi)
    allLFP=allLFP(1:length(mi));
else
    mi=mi(1:length(allLFP));
    wakeVec=wakeVec(1:length(allLFP));
end
figure
hold on
plot((1:length(allLFP))/60000,allLFP)
plot((1:length(allLFP))/60000,zscore(mi)*60,'r')
ylim([-700 600])
xlim([asTimes(3)/60000-0.5 asTimes(3)/60000+1])

[bb1,aa1]=butter(3,[.1/(fs/2) 10/(fs/2)],'bandpass');
[bb2,aa2]=butter(3,[20/(fs/2) 150/(fs/2)],'bandpass');
filtLow=filtfilt(bb1,aa1,allLFP);
filtHigh=filtfilt(bb2,aa2,allLFP);

figure
hold on
plot((1:length(allLFP))/60000,filtLow+500)
plot((1:length(allLFP))/60000,filtHigh+1000)
plot((1:length(allLFP))/60000,zscore(mi)*20,'r')
xlim([asTimes(3)/60000-0.5 asTimes(3)/60000+1])
ylim([-700 1200])

[s wfreqs] = wt(allLFP(1764372:1854372),1000,0);

sNorm=s./max(s(:)) ;
figure
pcolor((1:size(s,2))/1000,wfreqs,sNorm)
shg
shading flat
ylabel('frequency (Hz)')
xlabel('time (s)')
caxis([0 .5])
ylim([0 40])
colormap(jet)
set(gca, 'YScale', 'log')


currOct=3
dsets= find(oct==currOct);
bl=brainLoc(dsets(1)); %1 for FL, 2 for VL
indToUse=lfpInds(dsets(1));
dset=2
currDset=dsets(dset);
[dataFolder, rec]=fileparts(dataFolderList{currDset});
cd(dataFolder)
analysisFile=[rec '_analysis'];
load([rec '_behaviorAnalysis.mat']);
load([rec '_selectChannel.mat']);
allLFP=allLFP(indToUse,:);
if length(allLFP)>length(mi)
    allLFP=allLFP(1:length(mi));
else
    mi=mi(1:length(allLFP));
    wakeVec=wakeVec(1:length(allLFP));
end

figure
hold on
plot((1:length(allLFP))/60000,allLFP)
plot((1:length(allLFP))/60000,zscore(mi)*20,'r')
xlim([asTimes(3)/60000-0.5 asTimes(3)/60000+1])
ylim([-700 600])

[s wfreqs] = wt(allLFP(16907286:16997286),1000,0);
caxis([0 1500])

sNorm=s./max(s(:)) ;
figure
pcolor((1:size(s,2))/1000,wfreqs,sNorm)
shg
shading flat
ylabel('frequency (Hz)')
xlabel('time (s)')
caxis([0 .5])
colormap(jet)
set(gca, 'YScale', 'log')



[bb1,aa1]=butter(3,[.1/(fs/2) 10/(fs/2)],'bandpass');
[bb2,aa2]=butter(3,[20/(fs/2) 150/(fs/2)],'bandpass');
filtLow=filtfilt(bb1,aa1,allLFP);
filtHigh=filtfilt(bb2,aa2,allLFP);

figure
hold on
plot((1:length(allLFP))/60000,filtLow+500)
plot((1:length(allLFP))/60000,filtHigh+1000)
plot((1:length(allLFP))/60000,zscore(mi)*20,'r')
xlim([asTimes(3)/60000-0.5 asTimes(3)/60000+1])
ylim([-700 1200])


figure
hold on
specFile=[experimentDirectory 'oct' num2str(3) '_spectra']
S1=h5read(specFile,'/S1');
f=h5read(specFile,'/f');

for sleepCat=1:3
    plot(f,S1(sleepCat,:))
end
set(gca,'yscale','log');
ylabel('lfp power')
xlabel('frequency')
xlim([0 100])
ylim([10^-2 10^3])


specFile=[experimentDirectory 'oct' num2str(9) '_spectra']
S1=h5read(specFile,'/S1');
f=h5read(specFile,'/f');

for sleepCat=1:3
    plot(f,S1(sleepCat,:),'-')
end
























