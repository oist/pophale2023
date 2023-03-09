function [peaks,locs]=detectSpindles(lfp, wakeVec,bceTimes,bbSpindle,aaSpindle,spindleMinHeight,spindleMinProm, spindleMinDist)

bceTimes(bceTimes<500)=[];
bceTimes(bceTimes>(length(lfp)-500))=[];

fpindleFilt=filtfilt(bbSpindle,aaSpindle,lfp);
[peaks,locs]=findpeaks(zscore(fpindleFilt),'MinPeakHeight',spindleMinHeight,'MinPeakProminence',spindleMinProm,'MinPeakDistance',spindleMinDist);


badVec=zeros(1,length(lfp));
badVec(wakeVec(locs)~=0)=1;
badInds=find(badVec(locs));

locs(badInds)=[];
peaks(badInds)=[];
