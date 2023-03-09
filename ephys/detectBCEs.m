function [peaks,locs]=detectBCEs(lfp, wakeVec,bceLow, bceHigh, fps, bceMin, bceProm,bceDist)


[bb,aa]=butter(2,[bceLow/fps/2, bceHigh/fps/2],'bandpass');
filtTr=filtfilt(bb,aa,lfp);
filtTr(isnan(filtTr))=nanmean(filtTr);
[peaks,locs]=findpeaks(zscore(-filtTr),'MinPeakHeight',bceMin,'MinPeakProminence',bceProm,'MinPeakDistance',bceDist*fps);


badInds=find(wakeVec(locs)~=0);

locs(badInds)=[];
peaks(badInds)=[];