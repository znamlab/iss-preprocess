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
    [IN,IM] = meshgrid(1:N,1:M);
    Ic[Ic<=STEP-1] = STEP
    Ic[Ic>=P-STEP-1] = P-STEP-1
    Index1 = sub2ind([M,N,P], IM(:), IN(:), Ic-STEP);
    Index2 = sub2ind([M,N,P], IM(:), IN(:), Ic);
    Index3 = sub2ind([M,N,P], IM(:), IN(:), Ic+STEP);
    Index1[I.flatten()<=STEP-1] = Index3[I.flatten()<=STEP-1]
    Index3[I.flatten()>=STEP-1] = Index1[I.flatten()>=STEP-1]
    x1 = np.reshape(x[Ic-STEP],(M,N))
    x2 = np.reshape(x[Ic],(M,N))
    x3 = np.reshape(x[Ic+STEP],(M,N))
    y1 = np.reshape(np.log(Y[Index1]),(M,N))
    y2 = np.reshape(np.log(Y[Index2]),(M,N))
    y3 = np.reshape(np.log(Y[Index3]),M,N);
    c = ( (y1-y2)*(x2-x3) - (y2-y3)*(x1-x2) ) / ( (x1**2-x2**2)*(x2-x3) - (x2**2-x3**2)*(x1-x2) )
    b = ( (y2-y3)-c*(x2-x3)*(x2+x3) ) / (x2-x3)
    s = np.sqrt(-1/(2*c))
    u = b * s**2
    a = y1 - b * x1 - c * x1**2
    A = exp(a + u**2 / (2*s**2))
    return u, s, A, Ymax

def fstack(im, wsize=9, alpha=0.2, sth=13, focus=None):
    if focus is None:


    fm = np.zeros(im.shape)
    for plane in range(im.shape[2]):
        fn[:,:,plane] = gfocus(im[:,:,plane], wsize)
