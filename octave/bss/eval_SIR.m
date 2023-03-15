warning('off','all');
folder = argv(){1}; % the folder to store the tested results
nFiles = str2num(argv(){2}); % the number of wave files to evaluation

sirs = zeros(nFiles,2);
sirsUp = zeros(nFiles,1);

for index = 0:1:nFiles-1

    xs = audioread(sprintf('%s%08u.wav',folder,index));
    
    
    [nsamplt,nchant]= size(xs);
    xTrue = zeros(2,nsamplt,1);
    xRef = xs(:,1)';
    xTrue(1,:,1) = xs(:,1)';
    xTrue(2,:,1) = xs(:,4)';
    xMix = xs(:,2)';
    xGev = xs(:,3)';
    
    t_interference = xMix-xRef;
    e_interference = xGev-xRef;
    
    SDR=zeros(1,2);
    SIR=zeros(1,2);
    SAR=zeros(1,2);
    for jest=1:1,
        for jtrue=1:1,
            [s_true,e_spat,e_interf,e_artif]=bss_decomp_mtifilt(xMix(jest,:),xTrue,jtrue,512);
            [SDR(jest,jtrue),SIR(jest,jtrue),SAR(jest,jtrue)]=bss_source_crit(s_true,e_spat,e_interf,e_artif);
        end
    end
   
    sirs(index+1,1) = SIR(1,1);
    
    for jest=1:1,
        for jtrue=1:1,
            [s_true,e_spat,e_interf,e_artif]=bss_decomp_mtifilt(xGev(jest,:),xTrue,jtrue,512);
            [SDR(jest,jtrue),SIR(jest,jtrue),SAR(jest,jtrue)]=bss_source_crit(s_true,e_spat,e_interf,e_artif);
        end
    end
    
    sirs(index+1,2) = SIR(1,1);    
    disp(index);
    
end
mean_sirs = mean(sirs,1);
mean_sirs_improvement = mean_sirs(2)-mean_sirs(1); # mean value of SIR improvement 

function [s_true,e_spat,e_interf,e_artif]=bss_decomp_mtifilt(se,s,j,flen)
% BSS_DECOMP_MTIFILT Decomposition of an estimated source image into four
% components representing respectively the true source image, spatial (or
% filtering) distortion, interference and artifacts, derived from the true
% source images using multichannel time-invariant filters.
%
% [s_true,e_spat,e_interf,e_artif]=bss_decomp_mtifilt(se,s,j,flen)
%
% Inputs:
% se: nchan x nsampl matrix containing the estimated source image (one row per channel)
% s: nsrc x nsampl x nchan matrix containing the true source images
% j: source index corresponding to the estimated source image in s
% flen: length of the multichannel time-invariant filters in samples
%
% Outputs:
% s_true: nchan x nsampl matrix containing the true source image (one row per channel)
% e_spat: nchan x nsampl matrix containing the spatial (or filtering) distortion component
% e_interf: nchan x nsampl matrix containing the interference component
% e_artif: nchan x nsampl matrix containing the artifacts component

%%% Errors %%%
if nargin<4, error('Not enough input arguments.'); end
[nchan2,nsampl2]=size(se);
[nsrc,nsampl,nchan]=size(s);
if nchan2~=nchan, error('The number of channels of the true source images and the estimated source image must be equal.'); end
if nsampl2~=nsampl, error('The duration of the true source images and the estimated source image must be equal.'); end

%%% Decomposition %%%
% True source image
s_true=[reshape(s(j,:,:),nsampl,nchan).',zeros(nchan,flen-1)];
% Spatial (or filtering) distortion
e_spat=project(se,s(j,:,:),flen)-s_true;
% Interference
tmp = project(se,s,flen);
e_interf=project(se,s,flen)-s_true-e_spat;
% Artifacts
e_artif=[se,zeros(nchan,flen-1)]-s_true-e_spat-e_interf;

return;

end



function sproj=project(se,s,flen)

% SPROJ Least-squares projection of each channel of se on the subspace
% spanned by delayed versions of the channels of s, with delays between 0
% and flen-1

[nsrc,nsampl,nchan]=size(s);
s=reshape(permute(s,[3 1 2]),nchan*nsrc,nsampl);

%%% Computing coefficients of least squares problem via FFT %%%
% Zero padding and FFT of input data
s=[s,zeros(nchan*nsrc,flen-1)];
se=[se,zeros(nchan,flen-1)];
fftlen=2^nextpow2(nsampl+flen-1);
sf=fft(s,fftlen,2);
sef=fft(se,fftlen,2);
% Inner products between delayed versions of s
G=zeros(nchan*nsrc*flen);
for k1=0:nchan*nsrc-1,
    for k2=0:k1,
        ssf=sf(k1+1,:).*conj(sf(k2+1,:));
        ssf=real(ifft(ssf));
        ss=toeplitz(ssf([1 fftlen:-1:fftlen-flen+2]),ssf(1:flen));
        G(k1*flen+1:k1*flen+flen,k2*flen+1:k2*flen+flen)=ss;
        G(k2*flen+1:k2*flen+flen,k1*flen+1:k1*flen+flen)=ss.';
    end
end
% Inner products between se and delayed versions of s
D=zeros(nchan*nsrc*flen,nchan);
for k=0:nchan*nsrc-1,
    for i=1:nchan,
        ssef=sf(k+1,:).*conj(sef(i,:));
        ssef=real(ifft(ssef,[],2));
        D(k*flen+1:k*flen+flen,i)=ssef(:,[1 fftlen:-1:fftlen-flen+2]).';
    end
end

%%% Computing projection %%%
% Distortion filters
C=G\D;
C=reshape(C,flen,nchan*nsrc,nchan);
% Filtering
sproj=zeros(nchan,nsampl+flen-1);
for k=1:nchan*nsrc,
    for i=1:nchan,
        sproj(i,:)=sproj(i,:)+fftfilt(C(:,k,i).',s(k,:));
    end
end

return;
end



function [SDR,SIR,SAR]=bss_source_crit(s_true,e_spat,e_interf,e_artif)

% BSS_SOURCE_CRIT Measurement of the separation quality for a given source
% in terms of filtered true source, interference and artifacts.
%
% [SDR,SIR,SAR]=bss_source_crit(s_true,e_spat,e_interf,e_artif)
%
% Inputs:
% s_true: nchan x nsampl matrix containing the true source image (one row per channel)
% e_spat: nchan x nsampl matrix containing the spatial (or filtering) distortion component
% e_interf: nchan x nsampl matrix containing the interference component
% e_artif: nchan x nsampl matrix containing the artifacts component
%
% Outputs:
% SDR: Signal to Distortion Ratio
% SIR: Source to Interference Ratio
% SAR: Sources to Artifacts Ratio

%%% Errors %%%
if nargin<4, error('Not enough input arguments.'); end
[nchant,nsamplt]=size(s_true);
[nchans,nsampls]=size(e_spat);
[nchani,nsampli]=size(e_interf);
[nchana,nsampla]=size(e_artif);
if ~((nchant==nchans)&&(nchant==nchani)&&(nchant==nchana)), error('All the components must have the same number of channels.'); end
if ~((nsamplt==nsampls)&&(nsamplt==nsampli)&&(nsamplt==nsampla)), error('All the components must have the same duration.'); end

%%% Energy ratios %%%
s_filt=s_true+e_spat;
% SDR
SDR=10*log10(sum(sum(s_filt.^2))/sum(sum((e_interf+e_artif).^2)));
% SIR
SIR=10*log10(sum(sum(s_filt.^2))/sum(sum(e_interf.^2)));
% SAR
SAR=10*log10(sum(sum((s_filt+e_interf).^2))/sum(sum(e_artif.^2)));

return;
end


