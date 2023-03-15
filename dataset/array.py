
from torch.utils.data import Dataset
from .audio import Audio

import json
import librosa as lr
import numpy as np
import scipy
import math

import os.path

class Array(Dataset):

    """
    
    the array dataset
	
    The dataset class is used to generate the arrays of the speech of each sources for each microphones, the array of additive noise 
    for each microphones of microphone array, the received speech at each mirophones, the arrays of cross spectrum between microphones 
    pairs derived from microphone array, the oracle mask for each microphone array, for estimate the desired speech from the received 
    speeches of microphone arrays .

    Parameters:
    
    file_meta (str):
    
    the meta file which store the list of speeches and generated RIRs
    
    file_json (str):
    
    the json file which store the information of specified paramters
     
    Attributes:
    
        framesize (int): the size of frame in STFT setting 
	
        hopsize (int): the size of each hop in STFT setting
	       
        alpha (list): the paramter of generating time-frequency mask (steepness)
	
        beta (float): the paramter of generating time-frequency mask (offset)
	    
        epsilon (str): the paramter avoid large negative values as the energy goes to zero, 
		as specifing the log absolute value of cross spetrum 
	
	Return: 
		
		Xs(array_like): 4D array shaped `(nSrcs, nMics, T, F)`, representing the STFTs of the speech of each sources for each microphones, 
		where nSrcs denotes the number of speech sources, nMics denotes the number of micrpohones, 'T' denotes the number of frames 
        and 'F' denotes the number of frequency bins  

        Ns(array_like): 3D array shaped `(nMics, T, F)`, representing the STFTs of additive noise for each microphones of microphone array, 
		where nMics denotes the number of micrpohones, 'T' denotes the number of frames and 'F' denotes the number of frequency bins 

        Ys(array_like): 3D array shaped `(nMics, T, F)`, representing the STFTs of the received speech signal at each mirophones, 
		where nMics denotes the number of micrpohones, 'T' denotes the number of frames and 'F' denotes the number of frequency bins 

        YYs(array_like): 4D array shaped `( nComb, T, F, 2)`, representing the log absolute value and phase of cross spectrum between microphones 
        pairs,  where 'nComb' denotes the number of combinations of microphone pairs derived from microphone array    

		M(array_like): 2D array shaped `(nComb, T, F)` ,  representing the oracle time-frequncy mask for each microphone pairs, 
		where 'nComb' denotes the number of combinations of microphone pairs derived from microphone array, 'T' denotes the number of frames and 
        'F' denotes the number of frequency bins 

    """
        
    def __init__(self, file_meta, file_json):

        self.audio = Audio(file_meta=file_meta)

        with open(file_json, 'r') as f:
            features = json.load(f)

        self.frameSize = features['frame_size']
        self.hopSize = features['hop_size']
        self.alpha = features['alpha']
        self.beta = features['beta']
        self.epsilon = features['epsilon']

    def __len__(self):

        return len(self.audio)

    def __getitem__(self, idx):

        xs, ns, tdoas = self.audio[idx]

        nSrcs = xs.shape[0]
        nMics = xs.shape[1]
        nSamples = xs.shape[2]
        
        ## get addictive noises at microphones in STFT domain
        Ns = []

        for iMic in range(0, nMics):

            Ns.append(np.expand_dims(np.transpose(lr.core.stft(ns[iMic, :], n_fft=self.frameSize, hop_length=self.hopSize)), axis=0))

        ys = ns
        
        ## get received signals at microphones in STFT domain
        for iSrc in range(0, nSrcs):

            ys += xs[iSrc, :, :]

        Ys = []

        for iMic in range(0, nMics):

            Ys.append(np.expand_dims(np.transpose(lr.core.stft(ys[iMic, :], n_fft=self.frameSize, hop_length=self.hopSize)), axis=0))

        
        YYs = []

        k = np.transpose(np.expand_dims(np.arange(0, self.frameSize/2+1), axis=1))
        f = np.transpose(np.ones((1, Ys[0].shape[0]), dtype=np.float32))

        Mask = []

        ## iterate over all possible microphone piars from the microphone array to compute the cross-spectrums and masks
        for iMic1 in range(0, nMics):

            for iMic2 in range(iMic1+1, nMics):
                
                ## compute the cross-spectrum between two microphones
                tau12ref = tdoas[0, iMic1] - tdoas[0, iMic2]
                tau12_inter = tdoas[1, iMic1] - tdoas[1, iMic2]

                A = np.exp(-1j*2*np.pi*tau12ref*k*f/self.frameSize)

                YY = A * Ys[iMic1] * np.conj(Ys[iMic2])

                YY2 = np.zeros((Ys[0].shape[0], Ys[0].shape[1], Ys[0].shape[2], 2), dtype=np.float32)

                YY2[:, :, :, 0] = np.log(np.abs(YY)**2 + self.epsilon) - np.log(self.epsilon)
                YY2[:, :, :, 1] = np.angle(YY)

                YYs.append(YY2)
                
                ## compute the oracle pairwise ratio mask for each microphone pairs
                dtau = np.abs(tau12_inter - tau12ref)
                gain = 1.0 - 1.0 / (1.0 + np.exp(-1.0 * self.alpha * (dtau - self.beta)))

                X11 = lr.core.stft(xs[0, iMic1, :], n_fft=self.frameSize, hop_length=self.hopSize)
                X12 = lr.core.stft(xs[0, iMic2, :], n_fft=self.frameSize, hop_length=self.hopSize)
                X21 = lr.core.stft(xs[1, iMic1, :], n_fft=self.frameSize, hop_length=self.hopSize)
                X22 = lr.core.stft(xs[1, iMic2, :], n_fft=self.frameSize, hop_length=self.hopSize)

                Mask_up1 = np.abs(X11) ** 2 + np.abs(gain * X21) ** 2
                Mask_up2 = np.abs(X12) ** 2 + np.abs(gain * X22) ** 2
                Mask_blow1 = np.abs(X11) ** 2 + np.abs(X21) ** 2 + self.epsilon
                Mask_blow2 = np.abs(X12) ** 2 + np.abs(X22) ** 2 + self.epsilon

                M1 = Mask_up1/Mask_blow1
                M2 = Mask_up2/Mask_blow2
                M12 = np.transpose(M1 * M2)
                Mask.append(M12)
        
        Ns = np.concatenate(Ns, axis=0)
        Ys = np.concatenate(Ys, axis=0)
        YYs = np.concatenate(YYs, axis=0)

        Xs = np.zeros((nSrcs, nMics, Ys.shape[1], Ys.shape[2]), dtype=np.complex64)
        
        ## clean speech signals of target and interference at microphones
        for iSrc in range(0, nSrcs):

            for iMic in range(0, nMics):
                Xs[iSrc, iMic, :, :] = np.transpose(lr.core.stft(xs[iSrc, iMic, :], n_fft=self.frameSize, hop_length=self.hopSize))
        
        return Xs, Ns, Ys, YYs, Mask


