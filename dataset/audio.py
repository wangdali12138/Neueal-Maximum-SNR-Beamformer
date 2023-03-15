from torch.utils.data import Dataset
import json
import librosa as lr
import numpy as np
import matplotlib.pyplot as plt

class Audio(Dataset):


	"""the audio dataset
	
    The dataset class is used to generate the arrays of audio audio time series of speeches and additive noise, 
	and the arrays of times of arriaval of speech:
    
    Parameters:
	
		file_meta(str):
		the meta file which store the list of speeches and generated RIRs
	
	Return: 
		
		xs(array_like): 3D array shaped `(nSrcs, nMics, nSamples)` , 
		representing the audio audio time series of each speech for each mirophones , where 'nSample' is the number of samples 

		ns(array_like): 2D array shaped `(nMics, nSamples)` , 
		representing the audio time series of additive noise for each microphone

		tdoas(array_like): 2D array shaped `(nSrcs, nMics)`,  
		representing the time of arrivals of speech signals of different sources for microhones 
	
    """

	def __init__(self, file_meta):

		with open(file_meta) as f:
			self.elements = f.read().splitlines()

	def __len__(self):

		return len(self.elements)

	def __getitem__(self, idx):

		audio = json.loads(self.elements[idx])

		fs = 16000
        
		mics = audio['farfield']['mics'] # locations of microphones(locations)
		srcs = audio['farfield']['srcs'] # locations of sources(locations)
		speed = audio['farfield']['speed'] # the speed of sound
		snrs = audio['farfield']['snr']    # the SNRs for each speech source
		gains = audio['farfield']['gain']  # the random gains for each microphones
		volume = audio['farfield']['volume'] # volume gain 
		path = audio['farfield']['path']   # path for RIRs
		noise = audio['farfield']['noise'] # noise variance 

		## read the RIR from the directory  
		hs, _ = lr.core.load(path=path, sr=16000, mono=False)

		nSrcs = len(srcs)
		nMics = len(mics)

		duration = audio['speech'][0]['duration']
		N = round(duration * fs)

		xs = np.zeros((nSrcs, nMics, N), dtype=np.float32) # clean signals of sources for each microphone 

		for iSrc in range(0, nSrcs):
            
			## read the audio from the directory
			path = audio['speech'][iSrc]['path']
			offset = audio['speech'][iSrc]['offset']
			s, _ = lr.core.load(path=path, sr=fs, mono=True, offset=offset, duration=duration)
            
			snr = snrs[iSrc]

			for iMic in range(0, nMics):

				gain = gains[iMic]

				index = iSrc * nMics + iMic
				h = hs[index, :]

				x = np.convolve(h, s, mode='same')

				x /= np.sqrt(np.mean(x**2))
				x *= 10.0 ** (snr/20.0)
				x *= gain

				xs[iSrc, iMic, range(0, x.shape[0])] = x

		xs /= np.max(xs)
		xs *= volume


        
		## when type of noise is white noise 
		ns = noise * np.random.randn(nMics, N)
        
		## when applied noise is from noise data

		# noise_signal, _ = lr.core.load(path='./noise/factory2_16k.wav', sr=fs)
		# noise_signal = np.repeat(np.expand_dims(noise_signal, 1), nMics, 1)
		# ns = np.transpose(noise_signal)
        

		## compute the time of arrivals of speech signals of different sources for microhones 
		tdoas = np.zeros((nSrcs, nMics), dtype=np.float32)
		for iSrc in range(0, nSrcs):

			src = np.array(srcs[iSrc])
			src -= np.mean(np.array(mics), axis=0)
			src /= np.sqrt(np.sum(src**2))

			for iMic in range(0, nMics):

				mic = mics[iMic] - np.mean(np.array(mics), axis=0)
				tdoas[iSrc, iMic] = (fs/speed) * np.dot(mic, src)
        
        ## save the clean signals of sources, defuss noise, TDOAs 
		return xs, ns, tdoas