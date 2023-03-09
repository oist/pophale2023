function rate = rateCircBound(times,edges,smoothSD)

bpp=histcounts(times,edges);
origLength=length(bpp);
bpp=[bpp bpp bpp(end:-1:1)];
w = gausswin(smoothSD);
w=w/sum(w);
rate = filtfilt(w,1,bpp);
rate=rate((origLength+1):(2*origLength+1));