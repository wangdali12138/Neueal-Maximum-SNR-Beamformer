'''
GEV-BAN beamformer:

1.step: generate the estimation masks with the cross-spectrum through the network;
2.step: use masks to estimate target and noise covariance matrices;
3.step: use estimation of covariance matrices to compute coeffcients of beamformer and then estimate the desired signal

'''

import librosa as lr
import numpy as np
import scipy.linalg as la
import torch


def mask(YYs, net):
    """
    function: generate the overall mask with the trained neural network

    :param  YYs:
         cross spectrums between microphone piars which are derived from the mircophone arrays

    :param  net:
        trained network

    :return M:
        the overall time-frequency mask
    """
    net.eval()

    nPairs = YYs.shape[0]

    M = 0.0

    for iPair in range(0, nPairs):
        YY = torch.from_numpy(YYs[iPair, :, :, :]).unsqueeze(0)
        MM = net(YY)
        MM = MM.squeeze(0).detach().cpu().numpy()

        M += MM

    M /= nPairs

    return M

def cov(Ys, M):
    """
    function: estiamate the noise and target speech covariance matirx

    :param  Ys:
        spectrums of received signals at the microphones

    :param  M:
        the overall time-frequency mask

    :return TTs, IIs:
        the estimated noise and terget speech covariance maxtirx
    """
    Ms = np.expand_dims(M, 0).repeat(Ys.shape[0], 0)

    Ts = Ys * Ms
    Is = Ys * (1.0 - Ms)
    F = Ms.shape[2]
    m = Ys.shape[0]

    TTs = np.zeros((F, m, m), dtype=np.complex64)
    IIs = np.zeros((F, m, m), dtype=np.complex64)


    for f in range(0, F):
        T = np.squeeze(Ts[:, :, f])
        I = np.squeeze(Is[:, :, f])

        TT = np.matmul(T, np.conj(np.transpose(T)))
        II = np.matmul(I, np.conj(np.transpose(I)))

        TTs[f, :, :] = TT
        IIs[f, :, :] = II


    return TTs, IIs

def gev(Ys, TTs, IIs):
    """
    function: to do the GEV-BAN beamformer with the estimated covariance matrices

    :param Ys:
        the spectrums of the received noisy speeches at the microphones

    :param TTs:
        the estimatied target speech covariance matrix

    :param IIs:
        the estimatied target speech covariance matrix

    :return Z:
        the estimated target speech
    """
    F = TTs.shape[0]
    M = TTs.shape[1]
    T = Ys.shape[1]

    Ws = np.zeros((M, F), dtype=np.complex64)
    Gs = np.zeros((M, F), dtype=np.complex64)

    for f in range(0, F):
        TT = np.squeeze(TTs[f, :, :])
        II = np.squeeze(IIs[f, :, :])

        D, V = la.eigh(TT, II)

        fGEV = V[:, M - 1]

        Ws[:, f] = fGEV

        fGEV = np.expand_dims(fGEV, 1)

        expr1 = np.matmul(np.transpose(np.conj(fGEV)), II)
        expr2 = np.matmul(II, fGEV)
        expr3 = np.matmul(expr1, fGEV)

        gBAN = np.sqrt(np.matmul(expr1, expr2) / M) / expr3

        Gs[:, f] = gBAN

    Ws = np.repeat(np.expand_dims(Ws, 1), T, 1)
    Gs = np.repeat(np.expand_dims(Gs, 1), T, 1)

    Z = np.sum(Gs * np.conj(Ws) * Ys, 0)

    return Z


