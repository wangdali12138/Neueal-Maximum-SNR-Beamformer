
warning('off','all');
folder = argv(){1}; % the folder to store the tested results
nFiles = str2num(argv(){2});% the number of wave files to evaluation

sdrs = zeros(nFiles,2);
sdrsUp = zeros(nFiles,1);

for index = 0:1:nFiles-1

    xs = audioread(sprintf('%s%08u.wav',folder,index));


    xRef = xs(:,1)'; % the 
    xMix = xs(:,2)';
    xGev = xs(:,3)';

    SDRmix= bss_eval_sources(xMix,xRef);
    SDRgev= bss_eval_sources(xGev,xRef);

    sdrs(index+1,1) = SDRmix;
    sdrs(index+1,2) = SDRgev;
    sdrsUp(index+1,1)= SDRgev-SDRmix;
    
    disp(index);
    
end
mean_sdrs = mean(sdrs,1);
mean_sdrs_imp = mean_sdrs(2)-mean_sdrs(1); # mean value of SDR improvement 



