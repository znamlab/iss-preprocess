import scipy
import numpy as np

def gfocus(im, wsize):
    """
    Calculate focus measure as the local variance of the images

    """
    f = np.ones((wsize, wsize)) / wsize**2
    u = scipy.ndimage.convolve(im, f, mode='nearest')
    fm = (im-u)**2
    return scipy.ndimage.convolve(fm, f, mode='nearest')


def gauss3P(x, Y):
    """
    Fast 3-point gaussian interpolation

    """
    STEP = 2
    M,N,P = Y.shape
    I = np.argmax(Y, axis=2)
    Ic = I.flatten()
    Ymax = np.amax(Y, axis=2)
    [IN,IM] = np.meshgrid(np.arange(N), np.arange(M))
    Ic[Ic<STEP] = STEP
    Ic[Ic>P-STEP-1] = P-STEP-1
    Index1 = np.ravel_multi_index([IM.flatten(), IN.flatten(), Ic-STEP], [M,N,P])
    Index2 = np.ravel_multi_index([IM.flatten(), IN.flatten(), Ic], [M,N,P]);
    Index3 = np.ravel_multi_index([IM.flatten(), IN.flatten(), Ic+STEP], [M,N,P])
    Index1[I.flatten()<=STEP-1] = Index3[I.flatten()<=STEP-1]
    Index3[I.flatten()>=STEP-1] = Index1[I.flatten()>=STEP-1]
    x1 = np.reshape(x[Ic-STEP],(M,N))
    x2 = np.reshape(x[Ic],(M,N))
    x3 = np.reshape(x[Ic+STEP],(M,N))
    Y = Y.flatten()
    y1 = np.reshape(np.log(Y[Index1]),(M,N))
    y2 = np.reshape(np.log(Y[Index2]),(M,N))
    y3 = np.reshape(np.log(Y[Index3]),(M,N))
    c = ( (y1-y2)*(x2-x3) - (y2-y3)*(x1-x2) ) / ( (x1**2-x2**2)*(x2-x3) - (x2**2-x3**2)*(x1-x2) )
    b = ( (y2-y3)-c*(x2-x3)*(x2+x3) ) / (x2-x3)
    s = np.sqrt(-1/(2*c))
    u = b * s**2
    a = y1 - b * x1 - c * x1**2
    A = np.exp(a + u**2 / (2*s**2))
    return u, s, A, Ymax

def fstack(im, wsize=9, alpha=0.2, sth=13, focus=None):
    M, N, P = im.shape

    if focus is None:
        focus = np.arange(P)

    fm = np.zeros(im.shape)
    for plane in range(P):
        fm[:,:,plane] = gfocus(im[:,:,plane], wsize)

    u, s, A, fmax = gauss3P(focus, fm)

    err = np.zeros((M, N))
    for plane in range(P):
        err += np.abs( fm[:,:,plane] - A * np.exp(-(focus[plane]-u)**2 / (2*s**2)))
        fm[:,:,plane] = fm[:,:,plane] / fmax

    h = np.ones((wsize, wsize)) / wsize**2
    inv_psnr = scipy.ndimage.convolve(err / (P*fmax), h, mode='nearest')

    S = 20 * np.log10(1 / inv_psnr)
    S[np.isnan(S)] = np.min(S)

    phi = 0.5 * (1+np.tanh(alpha*(S - sth))) * alpha
    phi = scipy.signal.medfilt2d(phi, kernel_size=3)

    for plane in range(P):
        fm[:,:,plane] = 0.5 + 0.5 * np.tanh(phi * (fm[:,:,plane]-1))

    fmn = np.sum(fm, axis=2)

    return np.sum(im * fm / fmn[:,:,np.newaxis], axis=2)
