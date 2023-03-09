function [specgram wfreqs] = wt(v,Fs,plot)

%v is data vector
%fs is sampling frequency

nfreqs=100;
min_freq = 1;
max_freq = 100;
min_scale = 1/max_freq*Fs;
max_scale = 1/min_freq*Fs;
wavetype = 'cmor1.5-1';
scales = logspace(log10(min_scale),log10(max_scale),nfreqs);
wfreqs = scal2frq(scales,wavetype,1/Fs);
V_wave = cwt(v,scales,wavetype);
specgram = squeeze(abs(V_wave));

if plot==1
    figure
    pcolor((1:size(specgram,2))/Fs,wfreqs,specgram)
    shg
    shading flat
    set(gca,'yscale','log');
      colormap jet
      ylabel('frequency (Hz)')
      xlabel('time (s)')
end