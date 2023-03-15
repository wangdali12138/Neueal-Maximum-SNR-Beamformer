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

    ## initilize the correlation matrix

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

def max(fullYs, TTs, IIs):
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

    F = IIs.shape[0]
    MN = IIs.shape[1]
    T = fullYs.shape[0]

    Ws = np.zeros((MN, F), dtype=np.complex64)
    Gs = np.zeros((MN, F), dtype=np.complex64)
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

        [D, V] = la.eig(np.matmul(iII, TT))
        index = np.argsort(-D)
        b1 = np.expand_dims(V[:, index[0]], axis=1)
        Ws[:, f] = np.squeeze(b1)
        myeps = 1e-6

        ## construct the coeffcients of maximum SNR beamformer
        expr_up1 = np.matmul(np.conj(np.transpose(b1)), TT)
        expr_up2 = np.squeeze(np.matmul(expr_up1, iN1))
        expr_up = expr_up2

        expr_blow1 = np.matmul(np.conj(np.transpose(b1)), TT)
        expr_blow2 = np.matmul(expr_blow1, b1)
        expr_blow = np.squeeze(expr_blow2.real + myeps)

        hmax = expr_up / expr_blow


        Gs[:, f] = np.squeeze(hmax)

    belta = Gs*Ws

    ## estimate the desired speech with the constructed maximum SNR beamformer
    Z = np.zeros((T, F), dtype=np.complex64)
    for t in range(0, T):
        for f in range(0, F):
            fy = np.squeeze(fullYs[t, :, f])
            Z[t, f] = np.squeeze(np.matmul(np.conj(np.transpose(belta[:, f])), fy))

    return Z
