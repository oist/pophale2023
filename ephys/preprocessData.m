%initialize ephys recordings
initEphys


%%  First go through and save synced video and lfp to hd5 file
for x=length(dataFolderList)
    dataFolderList{x}
    lutFile=anatomyLUT{x};
    mantleVid_filename=mantleIntList{x};
    lutFile=['/bucket/ReiterU' lutFile(17:end)]
    
    T = readtable(lutFile);
    regionLabel=T{:,8};
    chan=T{:,1};
    firstChan=find(chan==0);
    regionLabel=regionLabel(firstChan:end);
    [dataFolder, rec]=fileparts(dataFolderList{x});
    cd(dataFolder)
    analysisFile=[ rec '_analysis'];
    currRefs=(numel(regionLabel)+1):(numel(regionLabel)+11); %take 10 channels out of brain for re referencing

    syncingToHd5(dataFolder, rec, analysisFile, mantleVid_filename)
    lfpDataToHd5(dataFolder, rec,analysisFile,currRefs, 1000); 
    
end

%% save a few channels for quick access and later calculation of events
for dataSetNum=1:length(dataFolderList)
    
    [dataFolder, rec]=fileparts(dataFolderList{dataSetNum});
    cd(dataFolder)
    analysisFile=[rec '_analysis']
    info=h5info(analysisFile,'/lfpMS');
    dataSize=info.Dataspace.Size;
    dataLength=dataSize(2); %there will be a bit more lfp data without camera, we don't need that
    
    chunks=1:chunkSize:dataLength;
    lastChunkLength=floor(dataLength)-chunks(end);
    chunkSizeList=repmat(chunkSize,1,numel(chunks));
    chunkSizeList(end)=lastChunkLength;
    
    swChans=str2num(chansToExamine{dataSetNum});
    allLFP=zeros(numel(swChans),dataLength);
    
    for c=1:numel(chunks)
        asLFP=h5read(analysisFile,'/lfpMS',[1,chunks(c)],[chans,chunkSizeList(c)]);
        asLFP=asLFP*1000000;
        allLFP(:,chunks(c):(chunks(c)+chunkSizeList(c)-1))=asLFP(swChans,:);
        c
    end
    
    save([ rec '_selectChannel.mat'],'swChans','allLFP','-v7.3')
    
end
