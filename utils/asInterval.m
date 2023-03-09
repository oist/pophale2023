function [startT, endT] = asInterval(mi,asTimes, forLag,backLag,thresh)

[bb,aa]=butter(2,[ .01/(24/2)],'low');  

asTimes=sort(asTimes);
asTimes(asTimes==0)=[];
asTimes(find((asTimes+forLag)>length(mi)))=[];

if length(asTimes)>0
    [ev_avg,lags,asTrigBehave] = getEventTrigAvg(mi,asTimes,backLag,forLag);
end


startT=[];endT=[];badBout=zeros(1,size(asTrigBehave,1));
for x=1:size(asTrigBehave,1)
    
     currT=zscore(asTrigBehave(x,:));
    if length(find(isnan(currT)))>0
        badBout(x)=1;
    else
        ftr=filtfilt(bb,aa,currT);
        currTr=ftr<thresh;
        
        if max(currTr)==1
            
            i=reshape(find(diff([0;currTr';0])~=0),2,[]);
            [lgtmax,jmax]=max(diff(i));
            startT(x)=i(1,jmax) +asTimes(x)-backLag;
            endT(x)=startT(x)+lgtmax; % length of the longest sequence of 1s
        end
    end
end