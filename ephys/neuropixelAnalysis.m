%initialize ephys recordings
initEphys


%% amplitude over channels during active sleep %taking mean/max starting at wake transition
allLFP=zeros(384,120000);
allOct=[];
allDset=[];
allBout=[];

for currOct=1:9
    dsets= find(oct==currOct);
    
    for dset=1:numel(dsets)
        currDset=dsets(dset);
        
        [dataFolder, rec]=fileparts(dataFolderList{currDset});
        cd(dataFolder)
        analysisFile=[rec '_analysis'];
        load([rec '_behaviorAnalysis.mat']);
        
        asTimes(asTimes==0)=[];
        for bout=1:numel(asTimes)
            asLFP=h5read(analysisFile,'/lfpMS',[1,asTimes(bout)-60000],[chans,120000]);
            asLFP=asLFP*1000000;
            allLFP=cat(3,allLFP,asLFP);
            allOct=[allOct currOct];
            allDset=[allDset dset];
            allBout=[allBout bout];
        end
        
    end
    currOct
end

allLFP(:,:,1)=[];
%
h5create([experimentDirectory 'asBoutOverChans150.h5'],'/lfp',[size(allLFP)])
h5write([experimentDirectory 'asBoutOverChans150.h5'],'/lfp',allLFP);
h5create([experimentDirectory 'asBoutOverChans150.h5'],'/oct',[size(allOct)])
h5write([experimentDirectory 'asBoutOverChans150.h5'],'/oct',allOct);
h5create([experimentDirectory 'asBoutOverChans150.h5'],'/dset',[size(allDset)])
h5write([experimentDirectory 'asBoutOverChans150.h5'],'/dset',allDset);
h5create([experimentDirectory 'asBoutOverChans150.h5'],'/bout',[size(allBout)])
h5write([experimentDirectory 'asBoutOverChans150.h5'],'/bout',allBout);

%
%allLFP=h5read([experimentDirectory 'asBoutOverChans.h5'],'/lfp');
% allOct=h5read([experimentDirectory 'asBoutOverChans.h5'],'/oct');


[bb1,aa1]=butter(3,[.1/(fs/2) 10/(fs/2)],'bandpass');
[bb2,aa2]=butter(3,[20/(fs/2) 150/(fs/2)],'bandpass');


maxValL=zeros(9,chans);
maxValH=zeros(9,chans);
meanValL=zeros(9,chans);
meanValH=zeros(9,chans);

for o=1:9
   [c,time,trials]= size(allLFP(:,:,allOct==o));
currLFP=allLFP(:,:,allOct==o);

 currLFP=reshape(currLFP,size(currLFP,1),size(currLFP,2)*size(currLFP,3));
 filt1=currLFP;
 filt2=currLFP;

for c=1:size(currLFP,1)
    flfp1=filtfilt(bb1,aa1,currLFP(c,:));
    [YUPPER,YLOWER] = envelope(flfp1,150);
    filt1(c,:)=YUPPER;
    
    flfp2=filtfilt(bb2,aa2,currLFP(c,:));
    [YUPPER,YLOWER] = envelope(flfp2,150);
    filt2(c,:)=YUPPER;
    c
end

lowFreqMag=reshape(filt1,chans,time,trials);
highFreqMag=reshape(filt2,chans,time,trials);

   
    meanValL(o,:)=mean(medfilt1(squeeze(mean(lowFreqMag(:,60001:end,:),2)),5),2)';
    meanValH(o,:)=mean(medfilt1(squeeze(mean(highFreqMag(:,60001:end,:),2)),5),2)';
 
    meanValLQS(o,:)=mean(medfilt1(squeeze(mean(lowFreqMag(:,1:60000,:),2)),5),2)';
    meanValHQS(o,:)=mean(medfilt1(squeeze(mean(highFreqMag(:,1:60000,:),2)),5),2)';
    
    
end


h5create([experimentDirectory 'asBoutOverChans150.h5'],'/meanL',[size(meanValL)])
h5write([experimentDirectory 'asBoutOverChans150.h5'],'/meanL',meanValL);
h5create([experimentDirectory 'asBoutOverChans150.h5'],'/meanH',[size(meanValH)])
h5write([experimentDirectory 'asBoutOverChans150.h5'],'/meanH',meanValH);

h5create([experimentDirectory 'asBoutOverChans150.h5'],'/meanLQS',[size(meanValLQS)])
h5write([experimentDirectory 'asBoutOverChans150.h5'],'/meanLQS',meanValLQS);
h5create([experimentDirectory 'asBoutOverChans150.h5'],'/meanHQS',[size(meanValHQS)])
h5write([experimentDirectory 'asBoutOverChans150.h5'],'/meanHQS',meanValHQS);

%% amplitude over channels while awake %taking mean/max starting at wake transition
allLFP=zeros(384,60000);
allOct=[];
allDset=[];
allBout=[];

for currOct=1:9
    dsets= find(oct==currOct);
    
    for dset=1:numel(dsets)
        currDset=dsets(dset);
        
        [dataFolder, rec]=fileparts(dataFolderList{currDset});
        cd(dataFolder)
        analysisFile=[rec '_analysis'];
        load([rec '_behaviorAnalysis.mat']);
        
        wakeTimes=find(diff(wakeVec)==1);
        
        
        for bout=1:numel(wakeTimes)
            wakeLFP=h5read(analysisFile,'/lfpMS',[1,wakeTimes(bout)],[chans,60000]);
            wakeLFP=wakeLFP*1000000;
            allLFP=cat(3,allLFP,wakeLFP);
            allOct=[allOct currOct];
            allDset=[allDset dset];
            allBout=[allBout bout];
        end
        
    end
    currOct
end

allLFP(:,:,1)=[];
%
h5create([experimentDirectory 'wakeBoutOverChans.h5'],'/lfp',[size(allLFP)])
h5write([experimentDirectory 'wakeBoutOverChans.h5'],'/lfp',allLFP);
h5create([experimentDirectory 'wakeBoutOverChans.h5'],'/oct',[size(allOct)])
h5write([experimentDirectory 'wakeBoutOverChans.h5'],'/oct',allOct);
h5create([experimentDirectory 'wakeBoutOverChans.h5'],'/dset',[size(allDset)])
h5write([experimentDirectory 'wakeBoutOverChans.h5'],'/dset',allDset);
h5create([experimentDirectory 'wakeBoutOverChans.h5'],'/bout',[size(allBout)])
h5write([experimentDirectory 'wakeBoutOverChans.h5'],'/bout',allBout);

 %allLFP=h5read([experimentDirectory 'wakeBoutOverChans.h5'],'/lfp');
 allOct=h5read([experimentDirectory 'wakeBoutOverChans.h5'],'/oct');

% allLFPInfo=h5info([experimentDirectory 'wakeBoutOverChans.h5'],'/lfp');
% allLFPSize=allLFPInfo.Dataspace.Size;

[bb1,aa1]=butter(3,[.1/(fs/2) 10/(fs/2)],'bandpass');
[bb2,aa2]=butter(3,[20/(fs/2) 150/(fs/2)],'bandpass');

maxValL=zeros(9,chans);
maxValH=zeros(9,chans);
meanValL=zeros(9,chans);
meanValH=zeros(9,chans);

for o=1:9

 currLFP=h5read([experimentDirectory 'wakeBoutOverChans.h5'],'/lfp',[1,1,find(allOct==o,1)],[384 60000 numel(find(allOct==o))]);
 [c,time,trials]= size( currLFP);
 currLFP=reshape(currLFP,size(currLFP,1),size(currLFP,2)*size(currLFP,3));
 filt1=currLFP;
 filt2=currLFP;

 
for c=1:size(currLFP,1)
    flfp1=filtfilt(bb1,aa1,currLFP(c,:));
    [YUPPER,YLOWER] = envelope(flfp1,150);
    filt1(c,:)=YUPPER;
    
    flfp2=filtfilt(bb2,aa2,currLFP(c,:));
    [YUPPER,YLOWER] = envelope(flfp2,150);
    filt2(c,:)=YUPPER;
    c
end

filt1=reshape(filt1,chans,time,trials);
filt2=reshape(filt2,chans,time,trials);

    meanValL(o,:)=mean(medfilt1(squeeze(mean(filt1,2)),5),2)';
    meanValH(o,:)=mean(medfilt1(squeeze(mean(filt2,2)),5),2)';
  

end


h5create([experimentDirectory 'wakeBoutOverChans150.h5'],'/meanL',[size(meanValL)])
h5write([experimentDirectory 'wakeBoutOverChans150.h5'],'/meanL',meanValL);
h5create([experimentDirectory 'wakeBoutOverChans150.h5'],'/meanH',[size(meanValH)])
h5write([experimentDirectory 'wakeBoutOverChans150.h5'],'/meanH',meanValH);

%% wake vs active sleep

lW=h5read([experimentDirectory 'wakeBoutOverChans150.h5'],'/meanL');
hW=h5read([experimentDirectory 'wakeBoutOverChans150.h5'],'/meanH');
%lS=h5read([experimentDirectory 'asBoutOverChans150.h5'],'/meanL');
%hS=h5read([experimentDirectory 'asBoutOverChans150.h5'],'/meanH');
lS=h5read([experimentDirectory 'bceBoutOverChans150.h5'],'/meanL');
hS=h5read([experimentDirectory 'bceBoutOverChans150.h5'],'/meanH');

allRegionLabels=zeros(size(lW));
for currOct=1:9
    
    dsets= find(oct==currOct);
    lutFile=anatomyLUT{dsets(1)};
    T = readtable(lutFile);
    regionLabel=[T{:,8} ;zeros(chans,1)];
    chan=T{:,1};
    firstChan=find(chan==probeOffset(currOct));
    regionLabel=regionLabel(firstChan:(firstChan+chans-1));
    allRegionLabels(currOct,:)=regionLabel;
    
    lS(currOct,allRegionLabels(currOct,:)==0)=min(lS(currOct,:));
    lW(currOct,allRegionLabels(currOct,:)==0)=min(lW(currOct,:));
    hS(currOct,allRegionLabels(currOct,:)==0)=min(hS(currOct,:));
    hW(currOct,allRegionLabels(currOct,:)==0)=min(hW(currOct,:));
    
end


allRegionLabels=allRegionLabels(:);
goodLabels=find(allRegionLabels~=0&allRegionLabels~=3&allRegionLabels~=8);
lW=lW(:);
lS=lS(:);
hW=hW(:);
hS=hS(:);
lWG=lW(goodLabels);
lSG=lS(goodLabels);
hWG=hW(goodLabels);
hSG=hS(goodLabels);
arG=allRegionLabels(goodLabels);

colors=zeros(numel(arG),3);
for x=1:numel(arG)
    colors(x,:)=regionColor{arG(x)}/255;
end

close all
allT=[]
for region=1:max(arG)
    
    lhVal=[lSG(arG==region) hSG(arG==region) lWG(arG==region) hWG(arG==region)];
    meanVal=mean(lhVal);
    SEM = std(lhVal)/sqrt(size(lhVal,1));               % Standard Error
    ts = tinv([0.025  0.975],size(lhVal,1)-1);      % T-Score
    CI =  ts(2).*SEM';                      % Confidence Intervals
    figure(1)
    hold on
    H=errorbarxy(meanVal(3),meanVal(1),CI(3),CI(1),{'.k', regionColor{region}/255, regionColor{region}/255});
    figure(2)
    hold on
    H=errorbarxy(meanVal(4),meanVal(2),CI(4),CI(2),{'.k', regionColor{region}/255, regionColor{region}/255});
    [h,p] = ttest2(lSG(arG==region),lWG(arG==region)) ;
    allT=[allT p];
%     pause
end

figure(1)
 line([15,20],[15,20],'color','k')
xlabel('waking signal strength')
ylabel('as signal strength')
axis tight
title('0-10 hz')

figure(2)
 line([5,6],[5,6],'color','k')
xlabel('waking signal strength')
ylabel('as signal strength')
axis tight
title('20-150 hz')



mdl = fitlm(hWG,hSG);  %correlation good for high freq, mixed for low
mdl = fitlm(lWG,lSG);


[R,P] = corrcoef(lWG,lSG)
[R,P] = corrcoef(hWG,hSG)

%partial regression leverage plot https://uk.mathworks.com/help/stats/linearmodel.plot.html


%% qs activity and spindle detection across the brain

%take a certain amount of quiet sleep for all recordings to keep things
%fair
[bb1,aa1]=butter(3,[.1/(fs/2) 10/(fs/2)],'bandpass');
[bb2,aa2]=butter(3,[20/(fs/2) 40/(fs/2)],'bandpass');

spindleRate=zeros(9,384);
allOct=[];
allDset=[];
allBout=[];
maxValL=[];
maxValH=[];
meanValL=[];
meanValH=[];

for currOct=1:9
    dsets= find(oct==currOct);
    
    boutTally=1;
    for dset=1:numel(dsets)
        
        currDset=dsets(dset);
        
        [dataFolder, rec]=fileparts(dataFolderList{currDset});
        cd(dataFolder)
        analysisFile=[rec '_analysis'];
        load([rec '_behaviorAnalysis.mat']);
        
        asTimes(asTimes==0)=[];
        asTimes(asTimes<1200000)=[]; %dummy value for early bouts
        for bout=1:numel(asTimes)
            qsLFP=h5read(analysisFile,'/lfpMS',[1,asTimes(bout)-1200000],[chans,1200000]);
            qsLFP=qsLFP*1000000;
            currWakeVec=wakeVec((asTimes(bout)-1200000):(asTimes(bout)-1));
            
            filt1=qsLFP;
            filt2=qsLFP;
            for c=1:chans
                
                flfp1=filtfilt(bb1,aa1,qsLFP(c,:));
                [YUPPER,YLOWER] = envelope(flfp1,150);
                filt1(c,:)=YUPPER.*(currWakeVec==0);
                
                flfp2=filtfilt(bb2,aa2,qsLFP(c,:));
                [YUPPER,YLOWER] = envelope(flfp2,150);
                filt2(c,:)=YUPPER.*(currWakeVec==0);
                
                
                [peaks,locs]=detectSpindles(qsLFP(c,:), currWakeVec,[],bbSpindle,aaSpindle,spindleMinHeight,spindleMinProm, spindleMinDist);
                spindleRate(currOct,c)=spindleRate(currOct,c)+length(locs)/(length(find(currWakeVec==0))/1000);
                c
            end
            
            maxValL(currOct,:,boutTally)=medfilt1(squeeze(max(filt1,[],2)),5);
            maxValH(currOct,:,boutTally)=medfilt1(squeeze(max(filt2,[],2)),5);
               meanValL(currOct,:,boutTally)=medfilt1(squeeze(mean(filt1,2)),5);
            meanValH(currOct,:,boutTally)=medfilt1(squeeze(mean(filt2,2)),5);
            
            allOct=[allOct currOct];
            allDset=[allDset dset];
            allBout=[allBout bout];
            boutTally=boutTally+1;
        end
    end
    currOct
end


%allMaxL=mean(maxValL,3)';
%allMaxH=mean(maxValH,3)';

smRate=zeros(size(spindleRate));
spindleRate1=spindleRate;
for o=1:9
    currVL=squeeze(maxValL(o,:,:));
    currVL(:,sum(currVL)==0)=[];
    allMaxL(o,:)=mean(currVL,2);
     currVH=squeeze(maxValH(o,:,:));
    currVH(:,sum(currVH)==0)=[];
    allMaxH(o,:)=mean(currVH,2);
    
      currVL=squeeze(meanValL(o,:,:));
    currVL(:,sum(currVL)==0)=[];
    allMeanL(o,:)=mean(currVL,2);
     currVH=squeeze(meanValH(o,:,:));
    currVH(:,sum(currVH)==0)=[];
    allMeanH(o,:)=mean(currVH,2);
    
    spindleRate1(o,:)=spindleRate(o,:)./numel(find(allOct==o));
    smRate(o,:)=medfilt1(spindleRate1(o,:),5);
end

h5create([experimentDirectory 'spindleRate.h5'],'/spindleRate',[size(spindleRate)])
h5write([experimentDirectory 'spindleRate.h5'],'/spindleRate',spindleRate1);
h5create([experimentDirectory 'spindleRate.h5'],'/smoothedRate',[size(smRate)])
h5write([experimentDirectory 'spindleRate.h5'],'/smoothedRate',smRate);
h5create([experimentDirectory 'spindleRate.h5'],'/maxL',[size(allMaxL)])
h5write([experimentDirectory 'spindleRate.h5'],'/maxL',allMaxL);
h5create([experimentDirectory 'spindleRate.h5'],'/maxH',[size(allMaxH)])
h5write([experimentDirectory 'spindleRate.h5'],'/maxH',allMaxH);
h5create([experimentDirectory 'spindleRate.h5'],'/meanL',[size(allMeanL)])
h5write([experimentDirectory 'spindleRate.h5'],'/meanL',allMeanL);
h5create([experimentDirectory 'spindleRate.h5'],'/meanH',[size(allMeanH)])
h5write([experimentDirectory 'spindleRate.h5'],'/meanH',allMeanH);

% spindleRate=h5read([experimentDirectory 'spindleRate.h5'],'/spindleRate');



%% neural activity during bces
[bb1,aa1]=butter(3,[.1/(fs/2) 10/(fs/2)],'bandpass');
[bb2,aa2]=butter(3,[20/(fs/2) 150/(fs/2)],'bandpass');

allOct=[];
allDset=[];
allBout=[];
maxValL=[];
maxValH=[];
meanValL=[];
meanValH=[];

forlag=350;
backlag=350;

[bbEnv,aaEnv]=butter(2,[.5/(fs/2)  ],'low');

for currOct=1:9
    boutTally=1;
    
    dsets= find(oct==currOct);
    bl=brainLoc(dsets(1)); %1 for FL, 2 for VL
    indToUse=lfpInds(dsets(1));
    
    for dset=1:numel(dsets)
        currDset=dsets(dset);
        
        [dataFolder, rec]=fileparts(dataFolderList{currDset});
        cd(dataFolder)
        analysisFile=[rec '_analysis'];
        load([rec '_behaviorAnalysis.mat']);
     
        %detect color flashes
        [peaks,bceLocs]=detectBCEs(mi(1:40:end), wakeVec(1:40:end),bceLow, bceHigh, 25,bceMin, bceProm,bceDist);
        bceLocs=bceLocs*40;
        
        for bout=1:numel(bceLocs)
            currLFP=h5read(analysisFile,'/lfpMS',[1,bceLocs(bout)-backlag],[chans,forlag+backlag+1]);
            currLFP=currLFP*1000000;
            currWakeVec=wakeVec((bceLocs(bout)-(backlag)):(bceLocs(bout)+forlag));
            filt1=currLFP;
            filt2=currLFP;
            for c=1:chans
                
                flfp1=filtfilt(bb1,aa1,currLFP(c,:));
                [YUPPER,YLOWER] = envelope(flfp1,150);
                filt1(c,:)=YUPPER.*(currWakeVec==0);
                
                flfp2=filtfilt(bb2,aa2,currLFP(c,:));
                [YUPPER,YLOWER] = envelope(flfp2,150);
               filt2(c,:)=YUPPER.*(currWakeVec==0);
                
          
                c
            end
            
            maxValL(currOct,:,boutTally)=medfilt1(squeeze(max(filt1,[],2)),5);
            maxValH(currOct,:,boutTally)=medfilt1(squeeze(max(filt2,[],2)),5);
               meanValL(currOct,:,boutTally)=medfilt1(squeeze(mean(filt1,2)),5);
            meanValH(currOct,:,boutTally)=medfilt1(squeeze(mean(filt2,2)),5);
            
            allOct=[allOct currOct];
            allDset=[allDset dset];
            allBout=[allBout bout];
            boutTally=boutTally+1;
  
        end
        
    end

    currOct
end




for o=1:9
    currVL=squeeze(maxValL(o,:,:));
    currVL(:,sum(currVL)==0)=[];
    allMaxL(o,:)=mean(currVL,2);
     currVH=squeeze(maxValH(o,:,:));
    currVH(:,sum(currVH)==0)=[];
    allMaxH(o,:)=mean(currVH,2);
    
      currVL=squeeze(meanValL(o,:,:));
    currVL(:,sum(currVL)==0)=[];
    allMeanL(o,:)=mean(currVL,2);
     currVH=squeeze(meanValH(o,:,:));
    currVH(:,sum(currVH)==0)=[];
    allMeanH(o,:)=mean(currVH,2);
    
end


% h5create([experimentDirectory 'bceBoutOverChans150.h5'],'/lfp',[size(allLFP)])
% h5write([experimentDirectory 'bceBoutOverChans150.h5'],'/lfp',allLFP);
h5create([experimentDirectory 'bceBoutOverChans150.h5'],'/oct',[size(allOct)])
h5write([experimentDirectory 'bceBoutOverChans150.h5'],'/oct',allOct);
h5create([experimentDirectory 'bceBoutOverChans150.h5'],'/dset',[size(allDset)])
h5write([experimentDirectory 'bceBoutOverChans150.h5'],'/dset',allDset);
h5create([experimentDirectory 'bceBoutOverChans150.h5'],'/bout',[size(allBout)])
h5write([experimentDirectory 'bceBoutOverChans150.h5'],'/bout',allBout);
h5create([experimentDirectory 'bceBoutOverChans150.h5'],'/meanL',[size( allMeanL)])
h5write([experimentDirectory 'bceBoutOverChans150.h5'],'/meanL', allMeanL);
h5create([experimentDirectory 'bceBoutOverChans150.h5'],'/meanH',[size( allMeanH)])
h5write([experimentDirectory 'bceBoutOverChans150.h5'],'/meanH', allMeanH);
h5create([experimentDirectory 'bceBoutOverChans150.h5'],'/maxL',[size(allMaxL)])
h5write([experimentDirectory 'bceBoutOverChans150.h5'],'/maxL',allMaxL);
h5create([experimentDirectory 'bceBoutOverChans150.h5'],'/maxH',[size(allMaxH)])
h5write([experimentDirectory 'bceBoutOverChans150.h5'],'/maxH',allMaxH);


%% Extract event info for all recordings in a dataset

for currOct=[1:3]
    
    dsets= find(oct==currOct);
    bl=brainLoc(dsets(1)); %1 for FL, 2 for VL
    indToUse=lfpInds(dsets(1));
    lutFile=anatomyLUT{dsets(1)};
    T = readtable(lutFile);
    regionLabel=T{:,8};
    chan=T{:,1};
    firstChan=find(chan==0);
    regionLabel=regionLabel(firstChan:end);
    allBCEspindles=[];
    spindleISI=[]
    dsetSpindleMat=[];
    dsetSpindleBehave=[];
    dsetASMatBehave=[];
    dsetSpindleISI=[];
    bceMatBehave=[];
    
     for dset=1:numel(dsets)
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
        mi=zscore(mi);
        asTimes(asTimes==0)=[];
        
        %detect color flashes
        [peaks,bceLocs]=detectBCEs(mi(1:40:end), wakeVec(1:40:end),bceLow, bceHigh, 25,bceMin, bceProm,bceDist);
        bceLocs=bceLocs*40;
        
        [ev_avg,lags,ev_mat] = getEventTrigAvg(mi,bceLocs,3000,3000);
        bceMatBehave{dset}=ev_mat;
        
        %detect spindles
        [peaks,spindleLocs]=detectSpindles(allLFP, wakeVec,bceTimes, bbSpindle,aaSpindle,spindleMinHeight,spindleMinProm,spindleMinDist);
        spindleLocs(spindleLocs>length(allLFP-spindleForLag))=[];
        spindleLocs(spindleLocs<spindleBackLag)=[];
        
        bceSpindle=zeros(1,numel(spindleLocs));
        for x=1:numel(spindleLocs)
            if min(abs(spindleLocs(x)-bceLocs))<10000; %10 sec of a bce peak
                bceSpindle(x)=1;
            end
        end
        allBCEspindles=[allBCEspindles bceSpindle];
        
        [ev_avg,lags,allSpindleMat] = getEventTrigAvg(allLFP,spindleLocs,spindleBackLag,spindleForLag);
        dsetSpindleMat{dset}=allSpindleMat;
        
        [ev_avg,lags,spindleBehaveMat] = getEventTrigAvg(mi,spindleLocs,3000,3000);
        dsetSpindleBehave{dset}=spindleBehaveMat;      
        
        %as bouts
        [ev_avg,lags,allASMatBehave] = getEventTrigAvg(mi,asTimes,3000,10000);
        if numel(asTimes)>0
        dsetASMatBehave{dset}=allASMatBehave;
        end
        
        isi=diff(spindleLocs);
        dsetSpindleISI{dset}=isi;
    
        
    end
    
    allSpindleMat=cell2mat(dsetSpindleMat');
    allASMatBehave=cell2mat(dsetASMatBehave');
      allSpindleMatBehave=cell2mat(dsetSpindleBehave');
    allBCEMatBehave=cell2mat(bceMatBehave');
    spindleISI=cell2mat(dsetSpindleISI);

    save([experimentDirectory 'oct' num2str(currOct) '_spindleAnalysis.mat'],'allSpindleMat', ...
        'allASMatBehave','allSpindleMatBehave','allBCEMatBehave','spindleISI','regionLabel','allBCEspindles','-v7.3');   
    
end


%% spectra

addpath(genpath('/home/sam/chronux_2_12/'))
params=[];
params.tapers=[5 9];
params.Fs=1000;
params.fpass=[.1 100];
params.trialave=0

for currOct=1:9
    dsets= find(oct==currOct);
    bl=brainLoc(dsets(1)); %1 for FL, 2 for VL
    indToUse=lfpInds(dsets(1));
    
    S=cell(numel(dsets),3);
    Serr=cell(numel(dsets),3);
    
    for dset=1:numel(dsets)
        currDset=dsets(dset);
        
        [dataFolder, rec]=fileparts(dataFolderList{currDset});
        cd(dataFolder)
        analysisFile=[rec '_analysis'];
        load([rec '_behaviorAnalysis.mat'])
        load([rec '_selectChannel.mat'])
        allLFP=allLFP(indToUse,:);
        if length(allLFP)>length(mi)
            allLFP=allLFP(1:length(mi));
        else
            mi=mi(1:length(allLFP));
            wakeVec=wakeVec(1:length(allLFP));
        end
        
        
        [bb,aa]=butter(3,[.1/500 150/500],'bandpass');
        allLFP=filtfilt(bb,aa,allLFP);
        for sleepCat=0:2
            currT=find(wakeVec==sleepCat);
            currV=allLFP(currT);
            chunkLength=1000;
            numChunks=floor(length(currV)/chunkLength);
            specMat=reshape(currV(1:numChunks*chunkLength),chunkLength,numChunks);
            %             [S{dset,sleepCat+1},f,Serr{dset,sleepCat+1}]=mtspectrumc(specMat,params);
            [S{dset,sleepCat+1},f]=mtspectrumc(specMat,params);
        end
    end
    
    
    S1=zeros(3,102);
    
    for sleepCat=1:3
        Stmp=cell2mat(S(:,sleepCat)')';
        S1(sleepCat,:)=mean(Stmp);
    end
    
    specFile=[experimentDirectory 'oct' num2str(currOct) '_spectra']
    delete(specFile)
    h5create(specFile,'/S1',size(S1))
    h5write(specFile,'/S1',S1)
    h5create(specFile,'/f',size(f))
    h5write(specFile,'/f',f)
    
end

rmpath(genpath('/home/sam/chronux_2_12/'))


%% example AS plots for EDF

forLag=90000;
backLag=90000;
allBehave=zeros(1,forLag+backLag+1);
allTrig=allBehave;

expInd=[]
dsetInd=[]
for currOct=1:3
    currOct
    dsets= find(oct==currOct);
    bl=brainLoc(dsets(1)); %1 for FL, 2 for VL
    indToUse=lfpInds(dsets(1));
    
    
    S=cell(numel(dsets),4);
    
    for dset=1:numel(dsets)
        currDset=dsets(dset);
        
        [dataFolder, rec]=fileparts(dataFolderList{currDset});
        cd(dataFolder)
        analysisFile=[rec '_analysis'];
        load([rec '_behaviorAnalysis.mat'])
        load([rec '_selectChannel.mat'])
        allLFP=allLFP(indToUse,:);
        if length(allLFP)>length(mi)
            allLFP=allLFP(1:length(mi));
        else
            mi=mi(1:length(allLFP));
            wakeVec=wakeVec(1:length(allLFP));
        end
        
        %         asTimes=h5read(analysisFile,'/asTimes');
        asTimes(asTimes==0)=[];
        asTimes((asTimes+forLag)>length(mi))=[];
        if length(asTimes)>0
            [ev_avg,lags,asTrigLFP] = getEventTrigAvg(allLFP,asTimes,forLag,backLag);
            [ev_avg,lags,asTrigBehave] = getEventTrigAvg(mi,asTimes,forLag,backLag);
            allBehave=[allBehave; asTrigBehave];
            allTrig=[allTrig; asTrigLFP];
            expInd=[expInd ,ones(1,numel(asTimes))*currOct];
            dsetInd=[dsetInd ,ones(1,numel(asTimes))*dset];
        end
        
    end
end
allBehave(1,:)=[];
allTrig(1,:)=[];

figure

%subplot(1,20,1:19)
hold on
[bb,aa]=butter(3,[.1/500 150/500],'bandpass');
[bb2,aa2]=butter(3,[.5/500],'low');

sumBehave=zeros(1,size(allBehave,2));
for x=1:size(allTrig,1)
    currTr=zscore(allTrig(x,:));
    currTr=currTr-mean(currTr(1:1000));
    currBehave=zscore(filtfilt(bb2,aa2,allBehave(x,:)));
    currBehave=currBehave-mean(currBehave(1:1000));
    filtTr=filtfilt(bb,aa,currTr);
    plot(filtTr+8*x+10*expInd(x),'k')
    plot(currBehave*20-50,'r')
    sumBehave=sumBehave+currBehave;
end
sumBehave=sumBehave/size(allBehave,1);
plot(sumBehave*20-50,'k','linewidth',3)
set(gca,'xtick',[])
set(gca,'ytick',[])

% subplot(1,20,20)
% imagesc(flipud(expInd'))




%% behavior during head fixation compared to normal behavior


currOct=3
dsets= find(oct==currOct);
indToUse=lfpInds(dsets(1));

dset=2
currDset=dsets(dset);
asTimes=[];
[dataFolder, rec]=fileparts(dataFolderList{currDset});
cd(dataFolder)
mantleVid_filename=mantleIntList{currDset};
mantle_int=h5read( mantleVid_filename,'/combinedInt');

figure
hold on
plot(mantle_int)
xlim([40800 42500])
scatter(41025:125:42025,mantle_int(41025:125:42025))


cd('/home/sam/bucket/octopus/electrophysiology/2022_03_07/frames')
imgs=dir('*png')

w=2101;
h=2777;
motage=zeros(h,w*numel(imgs),3,'uint8');
for i=1:numel(imgs)
    currImg=imread(imgs(i).name);
    currImg=currImg(1:h,1200:3300,:);
    montage(:,((i-1)*w+1):(w*i),:)=currImg;
end


%3/07 rec 1, cam0_2022-03-07-12-28-35.avi at 7:27:00 shows movement
%this corresponds to frame number 670500, occuring 26800730 ms into the
%recording. I ran optic flow on a clip starting at this time, in the
%movementTest folder
cd('/home/sam/bucket/octopus/electrophysiology/2022_03_07')
load('rec1_g0_selectChannel.mat')
load('rec1_g0_behaviorAnalysis.mat')
movement=h5read('/home/sam/bucket/octopus/electrophysiology/2022_03_07/movementTest/7.27.00_vidClip.avi_movement','/movement');
movMS=interp1(1:40:(numel(movement)*40),movement,1:(numel(movement)*40)); %to ms from 25 hz
movMS(isnan(movMS))=[]; %end bit


figure
subplot(211)
hold on
xrange=[26800730 26800730+300000]
plot(allLFP(1,xrange(1):10:xrange(2))-200)
plot(allLFP(3,xrange(1):10:xrange(2)))
axis tight
subplot(212)
hold on
plot(zscore(mi(xrange(1):10:xrange(2)))+30)
plot(zscore(movMS(1:10:end))+20)
axis tight


%time of as bouts, rhythm

forLag=100000;
backLag=10000;
allBehave=zeros(1,forLag+backLag+1);
thresh=0.2

expInd=[]
dsetInd=[]
allIEWake=[]
iei=[]
for currOct=1:9
    currOct
    dsets= find(oct==currOct);
    indToUse=lfpInds(dsets(1));
    
    for dset=1:numel(dsets)
        currDset=dsets(dset);
        asTimes=[];
        [dataFolder, rec]=fileparts(dataFolderList{currDset});
        cd(dataFolder)
        analysisFile=[rec '_analysis'];
        load([rec '_behaviorAnalysis.mat'])
        
        if numel(asTimes)>1
            asTimes=sort(asTimes);
            for as=2:numel(asTimes)
                currT=wakeVec(asTimes(as-1):asTimes(as));
                allIEWake=[allIEWake sum(currT==1)/(60000)];
                iei=[iei (asTimes(as)-asTimes(as-1))/60000];
            end
            
        end
        asTimes(asTimes==0)=[];
        asTimes(find((asTimes+forLag)>length(mi)))=[];
        if length(asTimes)>0
            [ev_avg,lags,asTrigBehave] = getEventTrigAvg(mi,asTimes,backLag,forLag);
            allBehave=[allBehave; asTrigBehave];
            expInd=[expInd ,ones(1,numel(asTimes))*currOct];
            dsetInd=[dsetInd ,ones(1,numel(asTimes))*dset];
        end
        
    end
end
allBehave(1,:)=[];


figure
hold on
ksdensity(iei(allIEWake==0)) %inter event interval when there isn't too much waking
load('/home/sam/bucket/octopus/apartment/temperature_round2/tempData.mat')
ksdensity(allIfi(allTemps>23.5&allTemps<24.5)) %from temp experiment
xlabel('inter event interval')

 
%duration calculation

figure
hold on

%get refined asStart and end times
mi=reshape(allBehave',1,size(allBehave,1)*size(allBehave,2));
asTimes=backLag:(backLag+forLag):length(mi)

%downsample to 24 hz so filtering works same as other data
mi=mi(1:42:end);
asTimes=round(asTimes/42);
forLag=2400;
backLag=240;

threshVals=[-2 :.1:2];
durEphys=[];
md=[];
for x=1:numel(threshVals)
[startT, endT] = asInterval(mi,asTimes, forLag,backLag,threshVals(x));
durEphys=(endT-startT)/24/60;
md(x,:)=mean(durEphys);
sd(x,:)=std(durEphys);
end

plot(threshVals,md,'k')
plot(threshVals,md+sd,'k')
plot(threshVals,md-sd,'k')


%compare to lights on natural data
apartmentDir ='/home/sam/bucket/octopus/apartment/lightsOn/';
cd(apartmentDir)
fn=dir('*asTimes');
filenames=[];
for x=1:numel(fn)
    filenames{x}=fn(x).name;
end
startTime='2022-03-14-06-00-00';

threshVals=[-2 :.1:2];
durEphysN=[];
mdN=[];
sdN=[];
for x=1:numel(threshVals)
[durMin,allTimes,numOcts] = getASLengths(apartmentDir,filenames,startTime, forlag,backlag,threshVals(x));
mdN(x,:)=mean(durMin);
sdN(x,:)=std(durMin);
end

plot(threshVals,mdN,'r')
plot(threshVals,mdN+sdN,'r')
plot(threshVals,mdN-sdN,'r')

line([0.2 0.2], [0 2])

xlabel('threshold (z)')
ylabel('AS duration (min)')


figure
boxplot([durEphys{1} durEphys{2} durEphys{3} normDur],[ones(1,length(durEphys{1})),2*ones(1,length(durEphys{1})),3*ones(1,length(durEphys{1})),4*ones(1,length(normDur))]) 
[h,p] = ranksum(durEphys,normDur)


%% get times from mi trace
currOct=1
thresh=0.2
dsets= find(oct==currOct);
indToUse=lfpInds(dsets(1));

dset=1
currDset=dsets(dset);
asTimes=[];
[dataFolder, rec]=fileparts(dataFolderList{currDset});
cd(dataFolder)
analysisFile=[rec '_analysis'];
load([rec '_behaviorAnalysis.mat'])

mi=mi(1:42:end);
mi=(mi - nanmean(mi))/nanstd(mi);

asTimes=round(asTimes/42);
forLag=2400;
backLag=240;

[startT, endT] = asInterval(mi,asTimes, forLag,backLag,thresh);

[bb,aa]=butter(2,[ .05/(24/2)],'low');   
filtMI=filtfilt(bb,aa,mi);  
figure
hold on
  plot((1:length(mi))/(24*60),mi,'k')  
  plot(endT/(24*60),ones(length(endT)),'v','color','r')
plot(startT/(24*60),ones(length(startT)),'v','color','g')
xlim([200 364])

figure
hold on
  plot((1:length(mi))/(24*60),mi,'k')
    plot((1:length(mi))/(24*60),filtMI,'r')
  plot(endT/(24*60),ones(length(endT)),'v','color','r')
plot(startT/(24*60),ones(length(startT)),'v','color','g')
xlim([210 215])

% comparing to natural behavior
cd('/home/sam/bucket/octopus/apartment/temperature')
filename='cam2_2022-02-11-08-57-55_asTimes';

    mi=h5read(filename,'/combinedInt');
    asFrames=h5read(filename,'/as_frames');
     asPart=h5read(filename,'/as_part');
    mi=mi(4,1:702310);

    mi=(mi - nanmean(mi))/nanstd(mi); 
    mi(isnan(mi))=0;
currF=asFrames(asPart==3);
[startT, endT] = asInterval(mi,currF, 2400,240,thresh);

[bb,aa]=butter(2,[ .01/(24/2)],'low');   
filtMI=filtfilt(bb,aa,mi);  
figure
hold on
  plot((1:length(mi))/(24*60),mi,'k')
    plot((1:length(mi))/(24*60),filtMI,'r')
  plot(endT/(24*60),ones(length(endT)),'v','color','r')
plot(startT/(24*60),ones(length(startT)),'v','color','g')
xlim([285 449])


figure
hold on
  plot((1:length(mi))/(24*60),mi,'k')
    plot((1:length(mi))/(24*60),filtMI,'r')
  plot(endT/(24*60),ones(length(endT)),'v','color','r')
plot(startT/(24*60),ones(length(startT)),'v','color','g')
xlim([398 403])



%% Summary plots across octopus

%can't just combine the spectra, as some have lower SNR than others
cd(experimentDirectory)

figure


for currOct=1:9
    subplot(3,3,currOct)
    hold on
    specFile=[experimentDirectory 'oct' num2str(currOct) '_spectra']
    S1=h5read(specFile,'/S1');
    f=h5read(specFile,'/f');
    
    for sleepCat=1:3
        plot(f,S1(sleepCat,:))
    end
    
    %
    set(gca,'yscale','log');
    ylabel('lfp power')
    xlabel('frequency')
    xlim([0 50])
    ylim([10^-2 10^3])
    title(['oct ' num2str(currOct)])
    
    
    
end
legend('inactive sleep','wake','active sleep')

%VL peak work

vlPeak=zeros(9,3);
for currOct=1:9
    load([experimentDirectory 'oct' num2str(currOct) '_spindleAnalysis'])
    for x=1:3
        vlPeak(currOct,x)=mean(spindlePeaks(find(spindleBehaveState==(x-1))));
    end
end
vlPeak(1:3,:)=[];

boxplot(vlPeak)

[P,ANOVATAB,STATS]=anova1(vlPeak)
comp = multcompare(STATS)

currOct=9
load([experimentDirectory 'oct' num2str(currOct) '_spindleAnalysis'])
figure
violinplot(spindlePeaks,spindleBehaveState)

[P,ANOVATAB,STATS]=anova1(spindlePeaks,spindleBehaveState)
comp = multcompare(STATS)


%% How long into the recording did an AS bout happen?

lengthToAS=zeros(1,9);

for currOct=1:9
    foundBout=0
    dsets= find(oct==currOct);
    
    for dset=1:numel(dsets)
        if foundBout==0
            currDset=dsets(dset);
            
            
            startVidTime=hours(str2num(firstVidTime{currDset}(1:2)))+minutes(str2num(firstVidTime{currDset}(4:5)))
            currVid=mantleIntList{currDset};
            [dataFolder, rec]=fileparts(dataFolderList{currDset});
            cd(dataFolder)
            analysisFile=[rec '_analysis'];
            load([rec '_behaviorAnalysis.mat']);
            
            if asTimes(1)~=0
                firstAS=asTimes(1);
                foundBout=1;
            end
        end
    end
    
    fp=split(currVid,'/');
    vidName=fp{end};
    vidTime=hours(str2num(vidName(17:18)))+minutes(str2num(vidName(20:21)))
    
       
    if vidTime<startVidTime
        vidTime=vidTime+hours(24);
    end
    
    lengthToAS(currOct)=hours(hours(firstAS/3600000)+vidTime-startVidTime)
    currOct
end

