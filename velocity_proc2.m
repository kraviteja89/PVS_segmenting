%% velocity_proc2.m
% Bingxing Huo, July 2013
% Modified based on velocity_proc.m
% NOTE: NO assumption on velocity acquisition frequency.
%%
% Input:
%   velocity: raw velocity data
%   Fs: acquisition frequency
%   Ft: target frequency
%   acc_cutoff: acceleration threshold
function [imp_bin,velfilt]=velocity_proc2(velocity,Fs,Ft)
global acc_cutoff
% 1. low-pass filter velocity
Flp=10;
% [zeroa,poleb,gain]=butter(5,2/Fs*[Fhp,Flp],'bandpass');
[zeroa,poleb,gain]=butter(5,2/Fs*Flp,'low');
[sos,g]=zp2sos(zeroa,poleb,gain);
% h2=dfilt.df2sos(sos,g);
velfilt=filtfilt(sos,g,velocity);
% 2. take derivative
acc=diff(velfilt); % acceleration
acc=[0;acc]; 
N=length(acc);
Flight=floor(Fs/Ft);
L=floor(N/Flight);
% 3. binarize
accbin=(abs(acc)>acc_cutoff);
% 4. average within each window
imp_bin=zeros(L,1);
for i=1:L
    imp_bin(i)=(mean(accbin((i-1)*Flight+1:i*Flight))>.1);
end
imp_ave=zeros(L,1);
for i=floor(Ft/2)+1:L-floor(Ft/2)-1
    imp_ave(i)=sum(imp_bin(i-floor(Ft/2):(i+floor(Ft/2)+1)));
end

% figure
% bar([1:L]/Ft,imp_ave)
% axis tight
% set(gca,'fontsize',14)
% xlabel('time (second) ','fontsize',18)
% ylabel('thresholded acceleration (bin=1sec) ','fontsize',14)
