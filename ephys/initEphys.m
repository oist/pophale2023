%% Run ephys experiments

chans=384; %neuropixel
fs=1000;
chunkSize=1000000; %100s


%spindle detection
spindleLowPass=5; %Hz
spindleHighPass=40; %Hz
spindleMinHeight=5; 
spindleMinProm=2;
spindleMinDist=1*fs; %sec
spindleBackLag=.8*fs;
spindleForLag=.8*fs;
[bbSpindle,aaSpindle]=butter(3,[spindleLowPass/(fs/2) spindleHighPass/(fs/2) ],'bandpass');



%hpi index for lfp
hfLow=20; % Hz
hfHigh=40; % Hz
[bbHF,aaHF]=butter(3,[hfLow/(fs/2) hfHigh/(fs/2)],'bandpass');

%envelope filter for hpi
[bbEnv,aaEnv]=butter(2,[.5/(fs/2)  ],'low');


%bce filter
bceLow=0.005; %Hz
bceHigh= 2; 
bceMin=1;
bceProm=0.5;
bceDist=10; %seconds

experimentDirectory='/home/sam/bucket/octopus/electrophysiology/';
%experimentDirectory='/bucket/ReiterU/octopus/electrophysiology/'

cd(experimentDirectory)
T = readtable('ephysDatabase1.csv','Delimiter',',');

dataFolderList=T{:,1};
mantleIntList=T{:,2};
anatomyLUT=T{:,3};
goodList=T{:,4};
chansToExamine=T{:,5};
oct=T{:,6};
brainLoc=T{:,7};
lfpInds=T{:,8};
refChans=T{:,9};
firstVidTime=T{:,11};
goodList=find(goodList==1);
dataFolderList=dataFolderList(goodList);
mantleIntList=mantleIntList(goodList);
anatomyLUT=anatomyLUT(goodList);
chansToExamine=chansToExamine(goodList);
oct=oct(goodList);
brainLoc=brainLoc(goodList);
lfpInds=lfpInds(goodList);
refChans=refChans(goodList);
firstVidTime=firstVidTime(goodList);

for x=1:numel(dataFolderList)
    dataFolderList{x}=[experimentDirectory dataFolderList{x}];
    mantleIntList{x}=[experimentDirectory mantleIntList{x} '.meanInt_asTimes'];
    anatomyLUT{x}=['/home/sam/bucket' anatomyLUT{x}(16:end)];
end

%from Tomo
regionColor={[128,174,128],[0,0,0],[216,100,79],[216,100,79],[255,147,114],[244,128,26],[251,202,0],[89,114,255],[82,145,163],[0,0,0],[24,151,54],[255,170,255]};
probeOffset=[0 0 0 -10 0 0 -10 0 0];