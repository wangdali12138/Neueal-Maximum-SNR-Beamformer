from torch.utils.data import Dataset
from dataset.audio import Audio

import json
import librosa as lr
import numpy as np
import scipy
import math

import os.path


class Pair(Dataset):
    
	"""
	
	the pair dataset
	
    The dataset class is used to generate the arrays of cross spectrum between microphones pair and 
	corresponding oracle time-frequency mask for training the network.

    Parameters:
    
    file_meta (str):
    
    the meta file which store the list of speeches and generated RIRs
    
    file_json (str):
    
    the json file which store the information of specified paramters
    
    dir_scratch (str): 

    the directory to store temporary files
     
    Attributes:
    
        framesize (int): the size of frame in STFT setting 
	
        hopsize (int): the size of each hop in STFT setting
	
       	c (float): the speed of sound
	       
        alpha (list): the paramter of generating time-frequency mask (steepness)
	
        beta (float): the paramter of generating time-frequency mask (offset)
	    
        epsilon (str): the paramter avoid large negative values as the energy goes to zero, 
		as specifing the log absolute value of cross spetrum 
	    
        dir_scratch (float): the directory to store temporary files
	
	Return: 
		
		X(array_like): 3D array shaped `(T, F, 2)`, representing the log absolute value and phase of cross spetrum between microphone pair, 
		where 'T' denotes the number of frames and 'F' denotes the number of frequency bins   

		M(array_like): 2D array shaped `(T, F)` ,  representing the time-frequncy mask for each microphone pair, 
		where 'T' denotes the number of frames and 'F' denotes the number of frequency bins 

	
    """
    
	def __init__(self, file_meta, file_json, dir_scratch):

		self.audio = Audio(file_meta=file_meta)

		with open(file_json, 'r') as f:
			features = json.load(f)

		self.frameSize = features['frame_size']
		self.hopSize = features['hop_size']
		self.c = features['c'] 
		self.alpha = features['alpha']
		self.beta = features['beta']
		self.epsilon = features['epsilon']
		self.dir_scratch = dir_scratch

	def __len__(self):

		return len(self.audio)

	def __getitem__(self, idx):

		if self.dir_scratch is not None:

			file_scratch = '%s%08u.npz' % (self.dir_scratch, idx)

		else:

			file_scratch = ""

		if not os.path.exists(file_scratch):
           
			
			xs, ns, tdoas = self.audio[idx]

			nSrcs = xs.shape[0]
			nMics = 2 

			y1 = ns[0, :]
			y2 = ns[1, :]
			T1 = 0.0
			T2 = 0.0
			I1 = np.abs(lr.core.stft(y1, n_fft=self.frameSize, hop_length=self.hopSize, win_length=self.frameSize)) ** 2

			I2 = np.abs(lr.core.stft(y2, n_fft=self.frameSize, hop_length=self.hopSize, win_length=self.frameSize)) ** 2

			tau12ref = tdoas[0, 0] - tdoas[0, 1]
            
			## caculate the oracle pairwise ratio mask according to defference of TDOAs
			for iSrc in range(0, nSrcs):
				tau12 = tdoas[iSrc, 0] - tdoas[iSrc, 1]

				dtau = np.abs(tau12 - tau12ref)
                
				## the gain which goes to a value of 1 when both TDOAs are similar, and goes to zero when they are different.
				gain = 1.0 - 1.0 / (1.0 + np.exp(-1.0 * self.alpha * (dtau - self.beta))) 

				x1 = xs[iSrc, 0, :]
				x2 = xs[iSrc, 1, :]

				y1 += x1
				y2 += x2

				X1 = lr.core.stft(x1, n_fft=self.frameSize, hop_length=self.hopSize)
				X2 = lr.core.stft(x2, n_fft=self.frameSize, hop_length=self.hopSize)

				T1 += np.abs(gain * X1) ** 2
				T2 += np.abs(gain * X2) ** 2
				I1 += np.abs(X1) ** 2
				I2 += np.abs(X2) ** 2
                       
			M1 = T1 / (I1 + self.epsilon)
			M2 = T2 / (I2 + self.epsilon)
			M12 = np.transpose(M1 * M2)
			M = M12
            
            ## compute the cross-spectrum between microphones 
			Y1 = lr.core.stft(y1, n_fft=self.frameSize, hop_length=self.hopSize) 
			Y2 = lr.core.stft(y2, n_fft=self.frameSize, hop_length=self.hopSize)
            
			Y12 = Y1 * np.conj(Y2)
            
			## use steering vector aims to cancel the phase difference of the target source in the cross-spectrum 
			k = np.expand_dims(np.arange(0, self.frameSize / 2 + 1), axis=1)
			f = np.ones((1, Y12.shape[1]), dtype=np.float32)
			A = np.exp(-1j * 2 * np.pi * tau12ref * k * f / self.frameSize) # steering vetor

			X12 = np.transpose(A * Y12)
            
            ## extract the log absolute value and phase of cross-spectrum as input feature of network
			X = np.zeros((X12.shape[0], X12.shape[1], 2), dtype=np.float32)
			X[:, :, 0] = np.log(np.abs(X12) ** 2 + self.epsilon) - np.log(self.epsilon)
			X[:, :, 1] = np.angle(X12)


			if file_scratch != "":
				np.savez(file_scratch, X=X, M=M)

		else:

			data = np.load(file_scratch)

			X = data['X']
			M = data['M']

		X = X.astype(np.float32)
		M = M.astype(np.float32)

		return X, M
