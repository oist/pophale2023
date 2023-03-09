function syncingToHd5(dataFolder, rec, analysisFile, mantleVid_filename)

camChan=2;

cd(dataFolder)

    if exist(analysisFile)~=2 %no existing analysis folder
    
    [niVec,tVecSync, meta]=readSGLXData([dataFolder '/' rec '/' rec '_t0.nidq.bin'],[],0,1);
    dataLengthS=str2num(meta.fileTimeSecs);
    fs=str2num(meta.niSampRate);
    [niVec,tVecSync, meta]=readSGLXData([dataFolder '/' rec '/' rec '_t0.nidq.bin'],[1:2],0,dataLengthS);
    [peaks,locs]=findpeaks(diff(niVec(camChan,:)),'MinPeakHeight',1);
    camSyncTimes=round(tVecSync(locs+1)*1000);% ms
    camRate=1000/mean(diff(camSyncTimes));
    
    try
    h5create(analysisFile,'/camSyncTimes',size(camSyncTimes))
    end
    h5write(analysisFile,'/camSyncTimes',camSyncTimes);
    h5writeatt(analysisFile,'/','camRate', camRate);
    end
 
 camSyncTimes=h5read(analysisFile,'/camSyncTimes');
 camRate=1000/mean(diff(camSyncTimes));
mantle_int=h5read( mantleVid_filename,'/combinedInt');
asFrames=double(h5read( mantleVid_filename,'/as_frames'));
wakeFrames=double(h5read( mantleVid_filename,'/wake_frames'));
wakeVec=double(h5read( mantleVid_filename,'/wakeVec'));

if length(camSyncTimes)>length(mantle_int)
   firstFrame= length(camSyncTimes)-length(mantle_int)+1;
   camSyncTimes=camSyncTimes(firstFrame:end);
    mi=interp1(camSyncTimes,mantle_int,1:camSyncTimes(end),'linear','extrap');
    wakeVec=interp1(camSyncTimes,wakeVec,1:camSyncTimes(end),'nearest','extrap');
else
%     dt=(camSyncTimes(end)-camSyncTimes(1))/length(mantle_int)
%     mi=interp1(camSyncTimes(1):dt:(camSyncTimes(end)-1),mantle_int,1:camSyncTimes(end));
    mi=interp1(camSyncTimes,mantle_int(1:length(camSyncTimes)),1:camSyncTimes(end),'linear','extrap');
    wakeVec=interp1(camSyncTimes,wakeVec(1:length(camSyncTimes)),1:camSyncTimes(end),'nearest','extrap');
end

[bb,aa]=butter(3,[.5/(1000/2)],'low'); %remove high frequency noise from behavior trace
 mi=filtfilt(bb,aa,mi);
 

if length(asFrames)==0
    asTimes=0;
else
    asTimes=sort(camSyncTimes(asFrames));
end


if length(wakeFrames)==0
    wakeTimes=0;
else
    wakeTimes=camSyncTimes(wakeFrames(wakeFrames>0));
end


save([rec '_behaviorAnalysis.mat'],'wakeVec','asTimes','wakeTimes','mi')


delete(analysisFile)
h5create(analysisFile,'/camSyncTimes',size(camSyncTimes))
h5write(analysisFile,'/camSyncTimes',camSyncTimes);
h5writeatt(analysisFile,'/','camRate', camRate);
h5create(analysisFile,'/asTimes',numel(asTimes));
h5write(analysisFile,'/asTimes',asTimes);
h5create(analysisFile,'/wakeTimes',numel(wakeTimes));
h5write(analysisFile,'/wakeTimes',wakeTimes);
h5create(analysisFile,'/wokenUpTimes',numel(wokenUpTimes));
h5write(analysisFile,'/wokenUpTimes',wokenUpTimes);
h5create(analysisFile,'/mantle',size(mi));
% %   h5write(analysisFile,'/mantle',mi);
   h5writeatt(analysisFile,'/','folder', dataFolder);
