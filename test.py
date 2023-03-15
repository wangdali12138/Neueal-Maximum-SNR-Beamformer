import argparse
import librosa as lr
import numpy as np
import torch
import json
import beam_maxSNR
import beam_GEV
from scipy.io.wavfile import write


from dataset.array import Array
from model.blstm import Blstm

parser = argparse.ArgumentParser()
parser.add_argument('--audio', default='', type=str, help='Meta for audio')
parser.add_argument('--json', default='n', type=str, help='JSON of parameters')
parser.add_argument('--model_src', default='', type=str, help='Model to evaluate from')
parser.add_argument('--wave_dst', default='', type=str, help='Output wave file to save result')
args = parser.parse_args()

# Dataset

dataset = Array(file_meta=args.audio, file_json=args.json)


# Model

net = Blstm(file_json=args.json)
net.load_state_dict(torch.load(args.model_src))

# Evaluate

with open(args.json, 'r') as f:
    features = json.load(f)
frameSize = features['frame_size']
hopSize = features['hop_size']

# the number of consicutive time-frames considered in maximum SNR beamformer
lengh = 1

for index in range(0, 50):

    print(index)
    
    ## load the 
    Xs, Ns, Ys, YYs, ref_Masks = dataset[index]
    
    ## estimate the desired speech signal with maximum SNR beamformer
    M = beam_maxSNR.mask(YYs, net)
    XXs, NNs, fullY = beam_maxSNR.max_cov(Ys, M, lengh)
    Zs = beam_maxSNR.max(fullY,  XXs, NNs)
    
    ## estimate the desired speech signal with GEV beamformer
    M = beam_GEV.mask(YYs, net)
    TTs, IIs = beam_GEV.cov(Ys, M)
    Zs = beam_GEV.gev(Ys, TTs, IIs)

    ## save the target speech, noisy speech, estimated speech, interference speech into a wave file  
    Cs = Xs[0, 0, :, :]
    X2 = Xs[1, 0, :, :]
 
    XsTarget = np.transpose(Cs)
    XsMixed = np.transpose(Ys[0, :, :])
    XsOut = np.transpose(Zs)
    XsInterference = np.transpose(X2)
    xsOut = np.expand_dims(lr.core.istft(XsOut), 1)
    xsTarget = np.expand_dims(lr.core.istft(XsTarget), 1)
    xsMixed = np.expand_dims(lr.core.istft(XsMixed), 1)
    xsInterference = np.expand_dims(lr.core.istft(XsInterference), 1)
    xs = np.concatenate((xsTarget, xsMixed, xsOut, xsInterference), axis=1)
    
    wave_pre = '{0:>08d}'.format(index)
    write(args.wave_dst+wave_pre+'.wav', 16000, xs)

