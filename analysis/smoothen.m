function smoothedTrace = smoothen(inputTrace, sigma, fs)

%sigma is in ms

%create guassian distribution
sd=round(sigma/1000*fs);
pts=-6*sd:6*sd;
f = normpdf(pts, 0, sd);

%convolve with data
smoothedTrace=zeros(size(inputTrace));
for x=1:size(inputTrace,1)
smTrace=conv(inputTrace(x,:), f);
smoothedTrace(x,:)=smTrace(ceil(numel(pts)/2):(end-floor(numel(pts)/2)));
end


    w = gausswin(10);
    y = filter(w,1,x);