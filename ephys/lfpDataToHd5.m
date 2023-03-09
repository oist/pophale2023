function lfpDataToHd5(dataFolder, rec, analysisFile, currRefs, chunkSize)
cd(dataFolder)

[lfp,tVec, meta]=readSGLXData([dataFolder '/' rec '/' rec '_imec0/' rec '_t0.imec0.lf.bin'],[],0,1);

chunks=1:chunkSize:str2num(meta.fileTimeSecs);
lastChunkLength=floor(str2num(meta.fileTimeSecs)-chunks(end));
chunkSizeList=repmat(chunkSize,1,numel(chunks));
chunkSizeList(end)=lastChunkLength;

lfpFs=str2num(meta.imSampRate);
[bbLFP,aaLFP]=butter(3,[.1/500 150/500 ],'bandpass');

h5create(analysisFile,'/lfpMS',[385,sum(chunkSizeList)*1000])
msTally=1;
for c=1:length(chunks)
    
    try
    [lfp,tVec, meta]=readSGLXData([dataFolder '/' rec '/' rec '_imec0/' rec '_t0.imec0.lf.bin'],[1:385],chunks(c),chunkSizeList(c));
  
       [lfpMS_, ty] = resample(lfp(1,:), (1:size(lfp,2))/lfpFs, 1000);
     lfpMS=zeros(size(lfp,1),length(lfpMS_));
     for chan=1:size(lfp,1)
         resampleLFP=resample(lfp(chan,:), (1:size(lfp,2))/lfpFs, 1000);
         filtLFP=filtfilt(bbLFP,aaLFP,resampleLFP);
         lfpMS(chan,:)=filtLFP;
         disp(chan)
     end
     lfpMS=lfpMS-median(lfpMS(currRefs,:));
    
           
        h5write(analysisFile,'/lfpMS',lfpMS,[1 msTally],size(lfpMS));
        msTally=msTally+length(lfpMS);
    catch
        disp('something went wrong in loading lfp data!')
    end
    
end