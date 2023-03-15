'''
Maximum SNR beamformer:

1.step: use generate the estimation masks with the cross-spectrum through the network;
2.step: use masks to estimate target speech and noise corrlation matrices;
3.step: estimate the desired signal through contructing Maximum SNR beamformer with the estimated target corrlation matrices

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

def max_cov(Ys, M, N):
    """
     function: estiamate the noise and desired speech correlation matirx

     :param  Ys:
         spectrums of received signals at the microphones in STFT domain

     :param  M:
         the overall time-frequency mask

     :param  N:
         the number of considered consecutive time frames in maximum SNR beamformer

     :return

         TTs: the estimated desired speech correlation matirx

         IIs: the estimated noise correlation matirx

         fullYs: the STFTs of observation signals as considering N consecutive time-frames for each mic

     """

    Ms = np.expand_dims(M, 0).repeat(Ys.shape[0], 0)

    Ts = Ys * Ms
    Is = Ys * (1.0 - Ms)
    F = Ms.shape[2]
    M1 = Ys.shape[0]
    T = Ys.shape[1]

    TTs = np.zeros((F, M1, M1), dtype=np.complex64)
    IIs = np.zeros((F, M1, M1), dtype=np.complex64)

    ##create and initilize new buffs

    inWinBuf = np.zeros((M1 * N, F), dtype=np.complex64)
    inWin_xbuf = np.zeros((M1 * N, F), dtype=np.complex64)
    inWin_nbuf = np.zeros((M1 * N, F), dtype=np.complex64)
    fullYs = np.zeros((T, M1 * N, F), dtype=np.complex64)


    ## correlation matrix

    Rfx = np.zeros((M1 * N, M1 * N, F), dtype=np.complex64)
    Rfn = np.zeros((M1 * N, M1 * N, F), dtype=np.complex64)
    Rfy = np.zeros((M1 * N, M1 * N, F), dtype=np.complex64)

    ## caculate the correlation matrix

    for t in range(0, T):

        for m in range(0, M1):

            inY = np.squeeze(Ys[m, t, :])
            inX = np.squeeze(Ts[m, t, :])
            inN = np.squeeze(Is[m, t, :])

            inWinBuf[m * N + 1:(m + 1)*N, :] = inWinBuf[m * N:(m + 1) * N - 1, :]
            inWinBuf[m * N, :] = inY

            fullYs[t, m * N + 1:(m + 1) * N, :] = fullYs[t, m * N:(m + 1) * N - 1, :]
            fullYs[t, m*N, :] = inY

            inWin_xbuf[m * N + 1:(m + 1)*N, :] = inWin_xbuf[m * N:(m + 1) * N - 1, :]
            inWin_xbuf[m * N, :] = inX

            inWin_nbuf[m * N + 1:(m + 1)*N, :] = inWin_nbuf[m * N:(m + 1) * N - 1, :]
            inWin_nbuf[m * N, :] = inN

        fullYs[t, :, :] = inWinBuf

        for f in range(0, F):

            Fyvector = np.expand_dims(inWinBuf[:, f], 1)
            Fnvector = np.expand_dims(inWin_nbuf[:, f], 1)

            tmp_RfyPlus = np.matmul(Fyvector, np.conj(np.transpose(Fyvector)))
            Rfy[:, :, f] = Rfy[:, :, f] + tmp_RfyPlus

            tmp_RfnPlus = np.matmul(Fnvector, np.conj(np.transpose(Fnvector)))
            Rfn[:, :, f] = Rfn[:, :, f] + tmp_RfnPlus


    Rfx = Rfy - Rfn
    IIs = np.transpose(Rfn, (2, 0, 1))
    TTs = np.transpose(Rfx, (2, 0, 1))

    return TTs, IIs, fullYs,

def max_snr(fullYs, TTs, IIs):
    """

    function: estimate the desired speech from the noisy by maximum SNR beamformer

    :param fullYs:
        the STFTs of observation signals as considering consecutive time-frames for each mic

    :param TTs:
        the estimated target speech correlation matrix

    :param IIs:
        the estimated noise correlation matrix

    :return: Z:
        the estimated desired speech
    """
    ##create and initilize new buffs
    F = IIs.shape[0]
    MN = IIs.shape[1]
    T = fullYs.shape[0]

    b1_s = np.zeros((MN, F), dtype=np.complex64)
    belta_s = np.zeros((MN, F), dtype=np.complex64)
    iN1 = np.zeros((MN, 1))
    iN1[0, 0] = 1

    for f in range(0, F):
        TT = np.squeeze(TTs[f, :, :])/T
        II = np.squeeze(IIs[f, :, :])/T

        ##function thmat:avoid the  diagonal element < 0
        rRfx_diag = np.diag(TT)
        row, col = np.diag_indices_from(TT)
        rRfx_diag = np.where(rRfx_diag >= 0, rRfx_diag, rRfx_diag.max() * 1e-4)
        TT[row, col] = rRfx_diag

        ##inverse the matrix Rfn, where non-positive eigenvalues are set to zero
        [Dn, Vn] = la.eig(II)
        An = np.where(Dn <= 0, 0, 1.0 / Dn)
        AA = np.diag(An)
        iII = np.matmul(np.matmul(Vn, AA), np.conj(np.transpose(Vn)))

        ## compute the parameter belta and eigenvector b1 and then get coeffcients of maximum SNR beamformer
        [D, V] = la.eig(np.matmul(iII, TT))
        index = np.argsort(-D)
        b1 = np.expand_dims(V[:, index[0]], axis=1)
        b1_s[:, f] = np.squeeze(b1)

        myeps = 1e-6
        expr_up1 = np.matmul(np.conj(np.transpose(b1)), TT)
        expr_up2 = np.squeeze(np.matmul(expr_up1, iN1))
        expr_up = expr_up2

        expr_blow1 = np.matmul(np.conj(np.transpose(b1)), TT)
        expr_blow2 = np.matmul(expr_blow1, b1)
        expr_blow = np.squeeze(expr_blow2.real + myeps)

        belta = expr_up / expr_blow


        belta_s[:, f] = np.squeeze(belta)

    hmax = belta_s*b1_s

    ## estimate the desired speech with the constructed maximum SNR beamformer
    Z = np.zeros((T, F), dtype=np.complex64)
    for t in range(0, T):
        for f in range(0, F):
            fy = np.squeeze(fullYs[t, :, f])
            Z[t, f] = np.squeeze(np.matmul(np.conj(np.transpose(hmax[:, f])), fy))

    return Z




