%% Example traces

cd('/home/sam/bucket/octopus/8k/oct_06')
[bb,aa]=butter(3,[.5/(24/2)],'low');

currMI=h5read('OCT14936.MP4.meanInt','/int');
currMI=filtfilt(bb,aa,currMI);

frames=[405, 783, 1181,1522, 1941, 2318,2668]

figure
plot((1:length(currMI))/30,currMI)
hold on
scatter(frames/30,currMI(frames),'k')
xlabel('time (s)')


cd('/home/sam/bucket/octopus/apartment/temperature')

filenames={'cam0_2022-02-11-08-57-55_asTimes',
    'cam1_2022-02-11-08-57-55_asTimes',
    'cam2_2022-02-11-08-57-55_asTimes'}

figure
hold on
for f=1:numel(filenames)
    currMI=h5read(filenames{f},'/combinedInt');
    asFrames=h5read(filenames{f},'/as_frames');
    asPart=h5read(filenames{f},'/as_part');
    
    currMI=currMI(:,1:702310);
    for x=1:size(currMI,1)
        plot((1:length(currMI))/(24*60),currMI(x,:)+80*x*f*4,'k')
        currF=asFrames(asPart==(x-1));
        plot(currF/(24*60),255*ones(length(currF))+80*x*f*4,'v','color','k')
    end
    
end


figure
hold on
plot((1:length(currMI))/(24*60),currMI(x,:),'k')
currF=asFrames(asPart==(x-1));
plot(currF/(24*60),175*ones(length(currF)),'v','color','k')
xlim([170 420])


%% Arousal calculation

pxPerMM=24.2;
fps=25;

cd('/home/sam/bucket/octopus/arousal_threshold/clips')

fileNames=dir('*movement');

allNames={};
allFM=zeros(numel(fileNames),59);
hs=zeros(numel(fileNames),1);
hitCond=zeros(numel(fileNames),1);


for f = 1:numel(fileNames)
    currName=fileNames(f).name;
    metaData=split(currName,'_');
    allNames{f}=currName;
    fm=h5read(currName,'/movement');
    allFM(f,:)=fm/pxPerMM*fps; %pix/frame to mm/s
    
    %hit strength
    hitStr=metaData{7};
    hitStr=split(hitStr,'.');
    hitStr=hitStr{1};
    if strcmp(hitStr,'215p')
        hs(f)=1; %6.386;
    elseif strcmp(hitStr,'180')
        hs(f)=2; %43.310;
    elseif strcmp(hitStr,'230')
        hs(f)=3; %86.301;
    end
    
    %wake condition
    if strcmp(metaData{5},'active')
        hitCond(f)=1;
    elseif strcmp(metaData{5},'inactive')
        hitCond(f)=2;
    elseif strcmp(metaData{5},'awake')
        hitCond(f)=3;
    end
end

close all
phaseNames={'active','quiet','awake'};




Y=[];
g=[];
T=[];


for trialType=1:3
    
    currTrials=find(hitCond==trialType);
    currFM=allFM(currTrials,:);
    currStr=hs(currTrials);
    hitMag=max(currFM(:,29:54)');
    baseMag=max(currFM(:,3:28)');
    
    for str=1:3
        Y{str,trialType}=hitMag(currStr==str);
        T{str,trialType}=baseMag(currStr==str);
    end
end


pMat=[];
YList=[]
gList=[];
group=1


for str=1:3
    for trialType=1:3
        
        [p,h] = signrank( Y{str,trialType},T{str,trialType});
        pMat(str,trialType)=p;
        YList=[YList Y{str,trialType}-T{str,trialType}];
        gList=[gList group*ones(1,numel(Y{str,trialType}))];
        group=group+1;
        
    end
end

figure
boxplot(YList,gList,'Labels',{'as', 'qs', 'wake', 'as', 'qs', 'wake','as', 'qs', 'wake'})
ylabel('extra movement (mm/s)')


YList=[]
gList=[];
group=1

for str=1:3
    for trialType=1:3
        
        
        YList=[YList T{str,trialType}];
        gList=[gList group*ones(1,numel(Y{str,trialType}))];
        group=group+1;
        
    end
end


figure
boxplot(YList,gList,'Labels',{'as', 'qs', 'wake', 'as', 'qs', 'wake','as', 'qs', 'wake'})
ylabel('movement (mm/s) ')


%% homeostatic control

apartmentDir ='/home/sam/bucket/octopus/apartment/homeostasis/';
cd(apartmentDir)
filenames={'cam0_2022-03-26-16-32-06_asTimes',
    'cam1_2022-03-26-16-32-06_asTimes',
    'cam3_2022-03-26-16-32-06_asTimes',
    'cam0_2022-03-29-17-00-53_asTimes',
    'cam1_2022-03-29-17-00-53_asTimes',
    'cam3_2022-03-29-17-00-53_asTimes',
    'cam0_2022-03-30-10-54-43_asTimes',
    'cam0_2022-03-30-17-36-11_asTimes',
    'cam1_2022-03-30-10-54-43_asTimes',
    'cam1_2022-03-30-17-36-11_asTimes',
    'cam3_2022-03-30-10-54-42_asTimes',
    'cam3_2022-03-30-17-36-11_asTimes'};
startTimes={'2022-03-26-18-00-00',
    '2022-03-29-18-00-00'};    %start at night time

cam=[1,2,3,1,2,3,1,1,2,2,3,3];
sessionType=[1,1,1,2,2,2,2,2,2,2,2,2];


%only got 2 well recorded octopus in cam 3, 1 on cam 0

for x=1:numel(startTimes)
    startDateNum(x)=datenum(datetime(startTimes{x},'InputFormat','yyyy-MM-dd-HH-mm-ss'));
end

allTimes=cell(2,3,4); %2 sessions, 3 cams, 4 partitions

behavLowPass=0.0250; %Hz
behavHighPass=1.25; %Hz
[bbBehav,aaBehav]=butter(2,[behavLowPass/12 behavHighPass/12],'bandpass');

for f=1:length(filenames)
    currCam=cam(f);
    currSess=sessionType(f);
    currTime=filenames{f}(6:24);
    currDateNum=datenum(datetime(currTime,'InputFormat','yyyy-MM-dd-HH-mm-ss'));
    sessionTimeSeconds=(currDateNum-startDateNum(sessionType(f)))*24*60*60;
    asFrames=double(h5read([apartmentDir filenames{f}],'/as_frames'));
    asParts=double(h5read([apartmentDir filenames{f}],'/as_part'))+1;
    mi=double(h5read([apartmentDir filenames{f}],'/combinedInt'));
    
    currTimesSeconds=asFrames/24+sessionTimeSeconds;
    tH=currTimesSeconds/3600; %hours
    relTimes=tH(tH>0&tH<36);
    
    relParts=asParts(tH>0&tH<36);
    
    for x=1:4
        allTimes{currSess,currCam,x}=[allTimes{currSess,currCam,x}; relTimes(relParts==x)];
    end
    
end


sessTimes=cell(2,1);
for a=1:3
    for b=1:4
        for c=1:2
            sessTimes{c}=[sessTimes{c}; allTimes{c,a,b}];
        end
    end
end


partComp=zeros(2,3,3,4); %2 session types, 3 time periods
timeRanges=[0 12; 12 24; 24 36];
for a=1:2 %pre/post
    for b=1:3 %night1/day/night2
        for c=1:3 %cam
            for d=1:4 %part
                currTimes=allTimes{a,c,d};
                partComp(a,b,c,d)=numel(currTimes(currTimes>timeRanges(b,1)&currTimes<timeRanges(b,2)));
            end
        end
    end
end

sessByTime=reshape(partComp,2,3,12);
emptySess=squeeze(sum(sum(sessByTime)));
sessByTime(:,:,emptySess==0)=[];

allP=zeros(1,3);
allN=zeros(2,3);
for time=1:3
    [p,h] = ranksum(squeeze(sessByTime(2,time,:)),squeeze(sessByTime(1,time,:)));
    allN(1,time)=sum( squeeze(sessByTime(1,time,:)));
    allN(2,time)=sum( squeeze(sessByTime(2,time,:)));
    allP(time)=p;
end

edges=0:36;
smoothSD=40;
fineDT=0.1;
edgesFine=0:fineDT:max(edges);

rate1 = rateCircBound(sessTimes{1}, edgesFine,smoothSD);
rate2 = rateCircBound(sessTimes{2}, edgesFine,smoothSD);

figure
hold on
histogram( sessTimes{1},edges,'FaceColor','b','FaceAlpha',0.2); %need to divide by number of animalsssssss!
histogram( sessTimes{2},edges,'FaceColor','r','FaceAlpha',0.2);
plot(edgesFine,rate1/fineDT,'b','linewidth',3)
plot(edgesFine,rate2/fineDT,'r','linewidth',3)
xlim([0 36])
line([12 12],[0 7])
line([24 24],[0 7])
xlabel('hours')
ylabel('active sleep bouts/hour')
axis tight


%% AS selective homeostasis

expRecord=readtable('/home/sam/bucket/octopus2/AS_deprivation/AS_dep_log.csv')

disrupted=expRecord{:,7};
interval=minutes(expRecord{:,6});

interval(1)=[]; %associate next interval with prev disruption
disrupted(end)=[];
man=interval(disrupted==1);
control=interval(disrupted==0);


edges=0:5:100;

figure
hold on
histogram( control,edges,'FaceColor','b','FaceAlpha',0.2,'Normalization','pdf'); %need to divide by number of animalsssssss!
histogram( man,edges,'FaceColor','r','FaceAlpha',0.2,'Normalization','pdf');
[a,b,bw]=ksdensity(control)
plot(b,a,'b','linewidth',3)
[a,b,bw]=ksdensity(man)
plot(b,a,'r','linewidth',3)

legend('normal','disrupted')
xlim([0 100])
p=ranksum(control, man)


[p,h] = ranksum(man,control);
nums=[numel(man), numel(control)]



%% lights on continuously

apartmentDir ='/home/sam/bucket/octopus/apartment/lightsOn/';
cd(apartmentDir)

cd(apartmentDir)
fn=dir('*asTimes');
filenames=[];
for x=1:numel(fn)
    filenames{x}=fn(x).name;
end

startTime='2022-03-14-06-00-00';
startDateNum=datenum(datetime(startTime,'InputFormat','yyyy-MM-dd-HH-mm-ss'));

octCount=zeros(1,60*60*24*8);
allTimes=[];
allIFI=[];
for f=1:length(filenames)
    
    currTime=filenames{f}(6:24);
    currDateNum=datenum(datetime(currTime,'InputFormat','yyyy-MM-dd-HH-mm-ss'));
    sessionTimeSeconds=(currDateNum-startDateNum)*24*60*60;
    asFrames=double(h5read([apartmentDir filenames{f}],'/as_frames'));
    asPart=double(h5read([apartmentDir filenames{f}],'/as_part'))+1;
    wakePart=double(h5read([apartmentDir filenames{f}],'/wake_part'))+1;
    currTimesSeconds=sort(asFrames/24)+sessionTimeSeconds;
    allTimes=[allTimes;currTimesSeconds];
    
    miInfo=h5info([apartmentDir filenames{f}],'/combinedInt');
    lengthSec=round(miInfo.Dataspace.Size(2)/24);
    octNum=numel(unique([asPart; wakePart]));
    
    tmpTime=sessionTimeSeconds+(60*60*24); %accomidate sessions that predate start time
    octCount(tmpTime:(tmpTime+lengthSec))=octCount(tmpTime:(tmpTime+lengthSec))+octNum;
    
end
allTimes(allTimes<0)=[];
octCount(1:(60*60*24))=[]; %start at the right time
rateDivider=octCount(round(allTimes))


dt=2;
dtFine=0.1;
smoothSD=40;

figure
hold on
tH=allTimes/3600;
maxEdge=ceil(max(tH)/24)*24-1;
timeBins=0:dt:maxEdge;

rateNorm=interp1((1:length(octCount))/(3600),octCount,timeBins(1:end-1),'nearest','extrap')
[Values, Edges] = histcounts(tH,timeBins);
Values = Values ./ rateNorm ./dt;
bar_centres = 0.5*(Edges(1:end-1) + Edges(2:end));
bar(bar_centres,Values)

tbFine=0:dtFine:maxEdge;
bpp=histcounts(tH, tbFine);
rateNorm=interp1((1:length(octCount))/(3600),octCount,tbFine(1:end-1),'nearest','extrap')
rateNorm(rateNorm==0)=1;
bpp=bpp./rateNorm

w = gausswin(smoothSD);
w=w/sum(w);
rate = filtfilt(w,1,bpp);
plot(tbFine(1:end-1),rate/dtFine,'r','linewidth',3)
axis tight
xlabel('time (hr)')

for x=1:14
    time=12*(x-1);
    line([time time],[0 1])
end
xlim([0 144])



%stats
currTH=tH(tH>24&tH<108);
tHMod=mod(currTH,24);
tHRad=tHMod/24*2*pi;
[pval z] = circ_rtest(tHRad);


%% Lights off

apartmentDir ='/home/sam/bucket/octopus/apartment/lightsOff/';
cd(apartmentDir)

cd(apartmentDir)
fn=dir('*asTimes');
filenames=[];
for x=1:numel(fn)
    filenames{x}=fn(x).name;
end

startTime='2022-04-27-06-00-00';
startDateNum=datenum(datetime(startTime,'InputFormat','yyyy-MM-dd-HH-mm-ss'));


octCount=zeros(1,60*60*24*8);
allTimes=[];
allIFI=[];
for f=1:length(filenames)
    
    currTime=filenames{f}(6:24);
    currDateNum=datenum(datetime(currTime,'InputFormat','yyyy-MM-dd-HH-mm-ss'));
    sessionTimeSeconds=(currDateNum-startDateNum)*24*60*60;
    asFrames=double(h5read([apartmentDir filenames{f}],'/as_frames'));
    asPart=double(h5read([apartmentDir filenames{f}],'/as_part'))+1;
    wakePart=double(h5read([apartmentDir filenames{f}],'/wake_part'))+1;
    wakeFrames=double(h5read([apartmentDir filenames{f}],'/wake_frames'));
    currTimesSeconds=sort(asFrames/24)+sessionTimeSeconds;
    allTimes=[allTimes;currTimesSeconds];
    
    miInfo=h5info([apartmentDir filenames{f}],'/combinedInt');
    lengthSec=round(miInfo.Dataspace.Size(2)/24);
    octNum=numel(unique([asPart; wakePart]));
    
    tmpTime=sessionTimeSeconds+(60*60*24); %accomidate sessions that predate start time
    octCount(tmpTime:(tmpTime+lengthSec))=octCount(tmpTime:(tmpTime+lengthSec))+octNum;
    
    times=[];
    for part=1:4
        currTimesSeconds=sort(asFrames(asPart==(part))/24);
        currWakeTimesSeconds=sort(wakeFrames(wakePart==(part))/24);
        
        if length(currWakeTimesSeconds)>0&&length(currTimesSeconds)>0
            for a=1:numel(currWakeTimesSeconds)
                closestTime=sort( currWakeTimesSeconds(a)- currTimesSeconds)
                disp([num2str(f) '_' filenames{f} ':' num2str(closestTime(1))])
            end
        end
        
        times=[times;currTimesSeconds+sessionTimeSeconds];
    end
    allTimes=[allTimes;times];
end
allTimes(allTimes<0)=[];
octCount(1:(60*60*24))=[]; %start at the right time
rateDivider=octCount(round(allTimes));


dt=2;
dtFine=0.1;
smoothSD=40;

figure
hold on
tH=allTimes/3600;
maxEdge=ceil(max(tH)/24)*24-1;
timeBins=0:dt:maxEdge;

rateNorm=interp1((1:length(octCount))/(3600),octCount,timeBins(1:end-1),'nearest','extrap');

[Values, Edges] = histcounts(tH,timeBins);
Values = Values ./ rateNorm./dt;
bar_centres = 0.5*(Edges(1:end-1) + Edges(2:end));
bar(bar_centres,Values)

tbFine=0:dtFine:maxEdge;
bpp=histcounts(tH, tbFine);
rateNorm=interp1((1:length(octCount))/(3600),octCount,tbFine(1:end-1),'nearest','extrap');
rateNorm(rateNorm==0)=1;
bpp=bpp./rateNorm;

w = gausswin(smoothSD);
w=w/sum(w);
rate = filtfilt(w,1,bpp);
plot(tbFine(1:end-1),rate/dtFine,'r','linewidth',3)
axis tight
xlabel('time (hr)')


for x=1:14
    time=12*(x-1);
    line([time time],[0 1]);
end

xlim([0 144])

currTH=tH(tH>12&tH<96);
tHMod=mod(currTH,24);
tHRad=tHMod/24*2*pi;
[pval z] = circ_rtest(tHRad);

%% Temp experiments
apartmentDir='/home/sam/bucket/octopus/apartment/temperature_round2/';
tempRecord='/home/sam/bucket/octopus/apartment/tempRecord/Lab4a_May2-June17_2022/data.csv';
tempInterval=900; %every 15 minutes logging

cd(apartmentDir)
fn=dir('*asTimes');
filenames=[];
for x=1:numel(fn)
    filenames{x}=fn(x).name;
end

T = readtable(tempRecord);
T=T(89:end,:);   %start of 2/3
startTime='2022-05-03-00-00-00';
startDateNum=datenum(datetime(startTime,'InputFormat','yyyy-MM-dd-HH-mm-ss'));
r=table2array(T(:,3));

temp=[];
for x=1:size(r,1)
    temp(x)=str2num(r{x}(1:6));
end
%interp up to seconds
tempSeconds=interp1(1:tempInterval:(numel(temp)*tempInterval),temp, 1:(numel(temp)*tempInterval));

allIfi=[];
allTimes=[];
tally=1;
octoID=[];
forlag=3000;
backlag=3000;
allSWMat=zeros(1,forlag+backlag+1);
for f=1:length(filenames)
    
    currTime=filenames{f}(6:24);
    currCam=str2num(filenames{f}(4));
    currDateNum=datenum(datetime(currTime,'InputFormat','yyyy-MM-dd-HH-mm-ss'));
    sessionTimeSeconds=(currDateNum-startDateNum)*24*60*60;
    
    asFrames=double(h5read([apartmentDir filenames{f}],'/as_frames'));
    asPart=double(h5read([apartmentDir filenames{f}],'/as_part'))+1;
    mi=double(h5read([apartmentDir filenames{f}],'/combinedInt'));
    
    
    
    ifi=[];
    times=[];
    for part=1:4
        currTimesSeconds=sort(asFrames(asPart==(part))/24);
        times=[times;currTimesSeconds(1:(end-1))+sessionTimeSeconds]; %because we compare with an ifi, 2:end
        
        currWaveforms=mi(part,asFrames(asPart==(part)));
        [ev_avg,lags,ev_mat] = getEventTrigAvg(mi(part,:),asFrames(asPart==(part)),backlag,forlag);
        if ~isnan(ev_mat)
            allSWMat=[allSWMat; ev_mat(2:end,:)];
        end
        currIfi=diff(currTimesSeconds/60);
        octoID=[octoID; ones(numel(currIfi),1)*part+currCam*10];
        tally=tally+1;
        ifi=[ifi; currIfi];
    end
    
    
    allIfi=[allIfi; ifi];
    allTimes=[allTimes;times];
end
allSWMat(1,:)=[];
allSWMat=zscore(allSWMat')';

allTemps=tempSeconds(round(allTimes));
allTemps=allTemps';

badIFI=allIfi>120;% 2 series of bouts annotated
badIFI(allIfi<1)=1; %double count error
allIfi(badIFI)=[];
allTemps(badIFI)=[];
octoID(badIFI)=[];
allSWMat(badIFI,:)=[];
mdl = fitlm(allTemps, allIfi);
figure
plot(mdl) %partial regression leverage plot https://uk.mathworks.com/help/stats/linearmodel.plot.html
xlabel('temp (c)')
ylabel('inter event interval (sec)')
title('temp dependence of as bout interval')
axis tight
hold on
scatter(allTemps,allIfi,5,'k','filled')

save('tempData.mat','allTemps','allIfi')

%get refined asStart and end times
startT=[];endT=[];
thresh=.04
for x=1:size(allSWMat,1)
    currTr=medfilt1(envelope(diff(allSWMat(x,:))),24)>thresh;
    
    if max(currTr)==1
        startT(x)=find(currTr==1,1);
        endT(x)=find(diff(currTr)==-1,1,'last');
    else
        startT(x)=0;
        endT(x)=0;
    end
end

durMin=(endT-startT)/24/60;
badBout=durMin<0.1667;
tmpTemp=allTemps;
tmpTemp(badBout)=[];
durMin(badBout)=[];

figure
scatter(tmpTemp,durMin)
xlabel('temperature')
ylabel('as duration (minutes)')




%% Describing brief coloration events

apartmentDir ='/home/sam/bucket/octopus/apartment/lightsOn/';
cd(apartmentDir)

cd(apartmentDir)
fn=dir('*asTimesForBCs');
filenames=[];
for x=1:numel(fn)
    filenames{x}=fn(x).name;
end


forlag=120000
allFI=zeros(1, forlag+1);
allI=allFI;
allBCE=allFI;

[bb,aa]=butter(3,[.005/12 ],'high');
[bbBCE,aaBCE]=butter(3,[bceLow/12 bceHigh/12 ],'bandpass');


iv=[];
for f=1:length(filenames)
    asFrames=double(h5read([apartmentDir filenames{f}],'/as_frames'));
    asPart=double(h5read([apartmentDir filenames{f}],'/as_part'))+1;
    sFrames=double(h5read([apartmentDir filenames{f}],'/wake_frames'));
    sPart=double(h5read([apartmentDir filenames{f}],'/wake_part'))+1;
    
    mi=double(h5read([apartmentDir filenames{f}],'/combinedInt'));
    
    goodParts=unique(asPart);
    
    for cp=1:numel(goodParts)
        currPart=goodParts(cp);
        
        currMI=mi(currPart,:);
        currAS=asFrames(asPart==currPart);
        currS=sort(sFrames(sPart==currPart));
        
        for interval=2:numel(currS)
            
            if diff(currS((interval-1):interval))<120000
                if length(currMI)>currS(interval-1)+forlag
                    currInt=currMI(currS(interval-1):currS(interval-1)+forlag);
                    currInt(currInt==0)=mean(currInt);
                    
                    filtInt=filtfilt(bb,aa,currInt);
                    filtBCE=filtfilt(bbBCE,aaBCE,currInt);
                    
                    nextAS=currS(interval)-currS(interval-1);
                    filtBCE(1:(120*24))=0;
                    filtBCE(nextAS:(nextAS+120*24))=0; %remove as bouts from bce detection
                    
                    allFI=[allFI; filtInt];
                    allBCE=[allBCE; filtBCE];
                    allI=[allI; currInt];
                    iv=[iv diff(currS((interval-1):interval))];
                    
                    
                end
            end
            f
        end
    end
end
allFI(1,:)=[];
allI(1,:)=[];
allBCE(1,:)=[];

[~,order]=sort(iv);
figure
imagesc(allFI(order,:))
colormap gray
caxis([-3 0])
xlim([0 80*60*24])

colorbar


allIEI=[];
allFrac=[];
allFracP=[];
allDur=[];
forlag=1000;
backlag=500;
for bout=1:numel(iv)
    currT=zscore(allI(bout,(1:iv(bout))));
    filtTr=filtfilt(bbBCE,aaBCE,currT);
    asPeak=min( filtTr);
    currT(1:(90*24))=0;
    currT(nextAS:(nextAS+90*24))=0; %pad to remove as bouts from bce detection
    
    %detect color flashes
    [peaks,locs]=detectBCEs(currT, zeros(1,length(currT)),bceLow, bceHigh, 24,bceMin, bceProm,bceDist);
    
    
    [ev_avg,lags,ev_mat] = getEventTrigAvg(-filtTr, locs,backlag,forlag);
    
    currDur=zeros(1,size(ev_mat,1));
    
    currDur=zeros(1,size(ev_mat,1));
    for bce=1:size(ev_mat,1)
        currLocs=find(zscore(ev_mat(bce,:))>2)
        bceStart=currLocs(find(currLocs<backlag,1,'last'));
        bceEnd=currLocs(find(currLocs>backlag,1,'last'));
        if numel(bceEnd-bceStart)>0
            currDur(bce)=bceEnd-bceStart;
        end
    end
    currDur(currDur==0)=[];
    
    allDur=[allDur currDur];
    allIEI=[allIEI diff(locs)];
    allFrac=[allFrac locs./iv(bout)];
    allFracP=[allFracP -peaks./asPeak];
end
allFracP(allFracP==1)=[]; %remove the AS peak
allDur=allDur/24; %seconds


for x=23
    figure;hold on
    currT=zscore(allI(order(x),:));
    filtTr=allBCE(order(x),:);
    
    
    [peaks,locs]=findpeaks(zscore(-filtTr),'MinPeakHeight',bceMin,'MinPeakProminence',bceProm);
    plot((1:size(currT,2))/(24*60),currT)
    plot(locs/(24*60),currT(locs),'o','color','k')
    xlabel('time (min)')
    ylabel('amp (z)')
    xlim([0 80])
end

figure
histogram(allIEI/24/60,1:0.5:10,'Normalization','probability')
xlabel('inter event interval (min)')
ylabel('probability')

figure
fracHist=histogram(allFrac,20,'Normalization','probability')
xlabel('interval fraction')
ylabel('probability')




[b,bint,~,~,stats] = regress(fracHist.Values',[ones(numel(fracHist.Values),1), (1:numel(fracHist.Values))']);


allFracP(allFracP==1)=[];
figure
fracHist=histogram(allFracP,20,'Normalization','probability')
xlabel('peak size (fraction of AS peak)')
ylabel('probability')

%
% %bce example
% info=h5info('/home/sam/bucket/octopus/8k/oct_456/not as/OCT14801.reg')
%
% g=zeros(1,2598);
% for x=1:2598
%     img=rgb2gray(permute(squeeze(h5read('/home/sam/bucket/octopus/8k/oct_456/not as/OCT14801.reg','/patterns1',[1 1 1 x],[3,675,1004,1])),[2,3,1]));
%      g(x)=mean(img(:));
%      x
% end
%
% figure
% hold on
% plot((1:length(g))/30,zscore(g))
% scatter([1520,1575,1630,1685,1740]/30,ones(1,5))


%% movements to active/inactive sleep

%fix sec scale bar

asStartT=[1500 1500 1500 1100 1500  1900 1500 1500 1700 1750];
pxPerMM=54
fps=24
%eye movements
resultsDir='/home/sam/bucket/octopus/high_res/newAnalysis/eye';
cd(resultsDir)
close all
fileNames=dir('*.avi_movement');
halves1=zeros(numel(fileNames),2);

figure
hold on
for x=1:numel(fileNames)
    movRel=h5read(fileNames(x).name,'/movRel');
    halves1(x,1)=mean(movRel(1:720));
    halves1(x,2)=mean(movRel(asStartT(x):(asStartT(x)+720)));
    %     line([1 2],[halves(x,1) halves(x,2)])
    
end
halves1=halves1./pxPerMM*fps; % pix/frame to mm/sec
[p,h] = ranksum( halves1(:,1),halves1(:,2));
boxplot([halves1])
ylabel('eye movement (mm/s)')

%body movements
resultsDir='/home/sam/bucket/octopus/high_res/newAnalysis/body';
cd(resultsDir)
fileNames=dir('*.avi_movement');
halves2=zeros(numel(fileNames),2);

figure
hold on
for x=1:numel(fileNames)
    mov=h5read(fileNames(x).name,'/movement');
    halves2(x,1)=mean(mov(1:720));
    halves2(x,2)=mean(mov(asStartT(x):(asStartT(x)+720)));
    % plot(mov)
    %     line([1 2],[halves(x,1) halves(x,2)])
end
halves2=halves2./pxPerMM*fps; % pix to to mm/sec
[p,h] = ranksum( halves2(:,1),halves2(:,2));
boxplot([halves2 ])
ylabel('body movement (mm/s)')

%breathing
resultsDir='/home/sam/bucket/octopus/high_res/newAnalysis/breath';
cd(resultsDir)
fileNames=dir('*.avi_movement');

halves3=zeros(numel(fileNames),2);
cv=halves3;

for x=1:numel(fileNames)
    mov=h5read(fileNames(x).name,'/movRel');
    
    [peaks,locs]=findpeaks(zscore(smooth(mov,10)),'MinPeakProminence',0.3);
    breathing=interp1(locs(2:end),1./(diff(locs)/30)*60,1:numel(mov),'liner','extrap'); %bpm
    
    halves3(x,1)=mean(breathing(1:720));
    halves3(x,2)=mean(breathing(asStartT(x):(asStartT(x)+720)));
    
    cv(x,1)=std(breathing(1:720))/mean(breathing(1:720));
    cv(x,2)=std(breathing(asStartT(x):(asStartT(x)+720)))/mean(breathing(asStartT(x):(asStartT(x)+720)));

end

figure
hold on
boxplot([halves3])
ylabel('breathing rate (bpm)')

figure
hold on
boxplot(cv)
ylabel('breathing rate cv')

[p,h] = ranksum( halves3(:,1),halves3(:,2));
[p,h] = ranksum( cv(:,1),cv(:,2));


%breathing during wake
resultsDir='/home/sam/bucket/octopus/high_res/newAnalysis/awake';
cd(resultsDir)
fileNames=dir('*.avi_movement');

brateW=zeros(numel(fileNames),1);
cvW=brate;
clipT=[1 720; 721 1440; 1441 2160;2161 2881];
tally=1;
for x=1:numel(fileNames)
    mov=h5read(fileNames(x).name,'/movRel');
    
    numClips=floor(length(mov)/720);
    
    for clip=1:3
        currMov=mov(clipT(clip,1):clipT(clip,2));
        
        [peaks,locs]=findpeaks(zscore(smooth(currMov,10)),'MinPeakProminence',0.3);
        breathing=interp1(locs(2:end),1./(diff(locs)/30)*60,1:numel(currMov),'liner','extrap'); %bpm
        
        brateW(tally)=mean(breathing);
        cvW(tally)=std(breathing)/mean(breathing);
        tally=tally+1;
    end
end

figure
hold on
boxplot([cv(:); cvW(:)],[ones(1,size(cv,1)) 2*ones(1,size(cv,1)) 3*ones(1,size(cvW,2))])

[p,h] = ranksum( cv(:,1),cvW);


asImg=imread('/home/sam/bucket/octopus/high_res/newAnalysis/asImg.png');
qsImg=imread('/home/sam/bucket/octopus/high_res/newAnalysis/qsImg.png');

a=asImg(1000:2100, 2700:3800,:)*1.3;
%print('-dpng','asImg','-r600')
q=qsImg(1000:2100, 2700:3800,:)*1.3;
%print('-dpng','qsImg','-r600')




figure
hold on
%single example for si
resultsDir='/home/sam/bucket/octopus/high_res/newAnalysis/eye';
cd(resultsDir)
fileNames=dir('*.avi_movement');
mov=h5read(fileNames(4).name,'/movRel');
t=0:0.0417:0.0417*numel(mov);
t(end)=[];

plot(t,mov./pxPerMM*fps)
resultsDir='/home/sam/bucket/octopus/high_res/newAnalysis/body';
cd(resultsDir)
fileNames=dir('*.avi_movement');
mov=h5read(fileNames(4).name,'/movement');
plot(t,mov./pxPerMM*fps+1)
mi=h5read('/home/sam/bucket/octopus/high_res/newAnalysis/3.avi.meanInt','/int')
mi(end)=[];
plot(t,zscore(mi)+3,'r')

%example image for si
%breathing during wake
resultsDir='/home/sam/bucket/octopus/high_res/newAnalysis/awake';
cd(resultsDir)
fileNames=dir('*.avi_movement');

halves3=zeros(numel(fileNames),1);
cv=halves3;
clipT=[1 720; 721 1440; 1441 2160;2161 2881];
tally=1;
for x=1:numel(fileNames)
    mov=h5read(fileNames(x).name,'/movRel');
    
    numClips=floor(length(mov)/720)
    
    for clip=1:numClips
        currMov=mov(clipT(clip,1):clipT(clip,2));
        
        [peaks,locs]=findpeaks(zscore(smooth(currMov,10)),'MinPeakProminence',0.3);
        breathing=interp1(locs(2:end),1./(diff(locs)/30)*60,1:numel(currMov),'liner','extrap'); %bpm
        
        brate(tally)=mean(breathing);
        cv(tally)=std(breathing)/mean(breathing);
        tally=tally+1;
    end
    
end



%% Check length of AS bouts

forlag=2400;
backlag=240;
thresh=0.2; %.045;
fps=24;

allDurations=[];
allGroups=[];
recordingLength=0;

%homeostasis -pre deprivation
apartmentDir ='/home/sam/bucket/octopus/apartment/homeostasis/';
filenames={'cam0_2022-03-26-16-32-06_asTimes',
    'cam1_2022-03-26-16-32-06_asTimes',
    'cam3_2022-03-26-16-32-06_asTimes'};
startTime='2022-03-26-18-00-00';

[durMin,allTimes] = getASLengths(apartmentDir,filenames,startTime, forlag,backlag,thresh);
allDurations=[allDurations durMin];
allGroups=[allGroups ones(1,length(durMin))];
rl = getTotalRecordingLength(apartmentDir,filenames,fps);
recordingLength=recordingLength+rl;

%homeostasis -post deprivation
apartmentDir ='/home/sam/bucket/octopus/apartment/homeostasis/';
filenames={ 'cam0_2022-03-29-17-00-53_asTimes',
    'cam1_2022-03-29-17-00-53_asTimes',
    'cam3_2022-03-29-17-00-53_asTimes',
    'cam0_2022-03-30-10-54-43_asTimes',
    'cam0_2022-03-30-17-36-11_asTimes',
    'cam1_2022-03-30-10-54-43_asTimes',
    'cam1_2022-03-30-17-36-11_asTimes',
    'cam3_2022-03-30-10-54-42_asTimes',
    'cam3_2022-03-30-17-36-11_asTimes'};
startTime='2022-03-29-18-00-00';

[durMin,allTimes,numOcts] = getASLengths(apartmentDir,filenames,startTime, forlag,backlag,thresh);

allDurations=[allDurations durMin];
allGroups=[allGroups 2*ones(1,length(durMin))];
rl = getTotalRecordingLength(apartmentDir,filenames,fps);
recordingLength=recordingLength+rl;

%lights on continuously
apartmentDir ='/home/sam/bucket/octopus/apartment/lightsOn/';
cd(apartmentDir)
fn=dir('*asTimes');
filenames=[];
for x=1:numel(fn)
    filenames{x}=fn(x).name;
end
startTime='2022-03-14-06-00-00';

[durMin,allTimes,numOcts] = getASLengths(apartmentDir,filenames,startTime, forlag,backlag,thresh);
allDurations=[allDurations durMin];


normal1=find(allTimes<24);
normal2=find(allTimes>24&allTimes<108);
normal3=find(allTimes>108);
allGroups=[allGroups ones(1,length(normal1)) 3*ones(1,length(normal2)) ones(1,length(normal3))];
rl = getTotalRecordingLength(apartmentDir,filenames,fps);
recordingLength=recordingLength+rl;


%lights off continuously
apartmentDir ='/home/sam/bucket/octopus/apartment/lightsOff/';
cd(apartmentDir)
fn=dir('*asTimes');
filenames=[];
for x=1:numel(fn)
    filenames{x}=fn(x).name;
end
startTime='2022-04-27-06-00-00';

[durMin,allTimes] = getASLengths(apartmentDir,filenames,startTime, forlag,backlag,thresh);
allDurations=[allDurations durMin];

normal1=find(allTimes<12);
normal2=find(allTimes>12&allTimes<96);
normal3=find(allTimes>96);
allGroups=[allGroups ones(1,length(normal1)) 4*ones(1,length(normal2)) ones(1,length(normal3))];
rl = getTotalRecordingLength(apartmentDir,filenames,fps);
recordingLength=recordingLength+rl;

%temperature
apartmentDir='/home/sam/bucket/octopus/apartment/temperature_round2/';
tempRecord='/home/sam/bucket/octopus/apartment/tempRecord/Lab4a_May2-June17_2022/data.csv';
tempInterval=900; %every 15 minutes logging
cd(apartmentDir)
fn=dir('*asTimes');
filenames=[];
for x=1:numel(fn)
    filenames{x}=fn(x).name;
end
T = readtable(tempRecord);
T=T(89:end,:);   %start of 2/3
startTime='2022-05-03-00-00-00';
startDateNum=datenum(datetime(startTime,'InputFormat','yyyy-MM-dd-HH-mm-ss'));
r=table2array(T(:,3));
temp=[];
for x=1:size(r,1)
    temp(x)=str2num(r{x}(1:6));
end
%interp up to seconds
tempSeconds=interp1(1:tempInterval:(numel(temp)*tempInterval),temp, 1:(numel(temp)*tempInterval));

[durMin,allTimes,numOcts] = getASLengths(apartmentDir,filenames,startTime, forlag,backlag,thresh);
allTemps=tempSeconds(round(allTimes*3600));

temp1=find(allTemps<23);
temp2=find(allTemps>23);


allDurations=[allDurations durMin];
allGroups=[allGroups 5*ones(1,length(temp1)) 6*ones(1,length(temp2))];
rl = getTotalRecordingLength(apartmentDir,filenames,fps);
recordingLength=recordingLength+rl;


figure
boxplot(allDurations,allGroups)
xlabel('normal, post, lights on, lights off, temp<23, temp>=23')
ylabel('as duration')

%[p,tbl,stats] = anova1(allDurations,allGroups)
[p,tbl,stats] = kruskalwallis(allDurations,allGroups)
[c,~,~,gnames] = multcompare(stats);

%normal duration
m=mean(allDurations(allGroups==1))*60
se=std(allDurations(allGroups==1))*60/sqrt(numel(find(allGroups==1)))
numel(find(allGroups==1))

normDur=allDurations(allGroups==1);


%% duration of video recordings

apartmentDir ='/home/sam/bucket/octopus/apartment/homeostasis/';
filenames={'cam0_2022-03-26-16-32-06_asTimes',
    'cam1_2022-03-26-16-32-06_asTimes',
    'cam3_2022-03-26-16-32-06_asTimes'};

recordingLength=getTotalRecordingLength(apartmentDir,filenames);

