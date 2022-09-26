import scipy
import numpy as np

# Adapted from MATLAB implementation by S. Petruz
#
# Copyright (c) 2016, Said Pertuz
# Copyright (c) 2017, Said Pertuz
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution
# * Neither the name of Universidad Industrial de Santander nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Ported to Python and modified by P. Znamenskiy, 2021

def gfocus(im, wsize):
    """
    Calculate focus measure as the local variance of the images

    """
    f = np.ones((wsize, wsize)) / (wsize**2)
    u = scipy.ndimage.correlate(im, f, mode='nearest')
    fm = (im-u)**2
    return scipy.ndimage.correlate(fm, f, mode='nearest')


def gauss3P(x, Y):
    """
    Fast 3-point gaussian interpolation

    Rewritten based on equations in "Shape from Focus", Shree K. Nayar, 1989

    """
    STEP = 2
    M,N,P = Y.shape
    I = np.argmax(Y, axis=2)
    Ic = I.flatten()
    Ic[Ic <= STEP - 1] = STEP
    Ic[Ic >= P - 1 - STEP] = P - 1 - STEP
    Ymax = np.max(Y, axis=2)
    IN, IM = np.meshgrid(np.arange(N), np.arange(M))
    Index1 = np.ravel_multi_index([IM.flatten(), IN.flatten(), Ic - STEP], [M,N,P])
    Index2 = np.ravel_multi_index([IM.flatten(), IN.flatten(), Ic], [M,N,P])
    Index3 = np.ravel_multi_index([IM.flatten(), IN.flatten(), Ic + STEP], [M,N,P])
    Index1[I.flatten() <= STEP - 1] = Index3[I.flatten() <= STEP - 1]
    Index3[I.flatten() >= STEP - 1] = Index1[I.flatten() >= STEP - 1]
    x1 = np.reshape(x[Ic-STEP], (M,N))
    x2 = np.reshape(x[Ic], (M,N))
    x3 = np.reshape(x[Ic+STEP], (M,N))
    y1 = np.reshape(np.log(Y[np.unravel_index(Index1, np.shape(Y))]),(M,N))
    y2 = np.reshape(np.log(Y[np.unravel_index(Index2, np.shape(Y))]),(M,N))
    y3 = np.reshape(np.log(Y[np.unravel_index(Index3, np.shape(Y))]),(M,N))
    d = ((x1**2 - x2**2) * (x2 - x3) - (x2**2 - x3**2) * (x1 - x2))
    c = ((y1 - y2) * (x2 - x3) - (y2 - y3) * (x1 - x2)) / d

    b = ((y2-y3) - c * (x2-x3) * (x2 + x3 + 2)) / (x2-x3)
    s = -1 / ( 2 * c + 1e-6 )
    u = b * s
    a = y1 - b * (x1 + 1) - c * (x1 + 1)**2
    A = np.exp(a + u**2 / (2*s))
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
        err += np.abs(fm[:,:,plane] - A * np.exp(-(focus[plane] + 1 - u)**2 / (2*s)))

    fm = fm / fmax[:,:,np.newaxis]

    h = np.ones((wsize, wsize)) / wsize**2
    inv_psnr = scipy.ndimage.correlate(err / (P*fmax), h, mode='nearest')

    S = 20 * np.log10(1 / inv_psnr)
    S[np.isnan(S)] = np.min(S)

    phi = 0.5 * (1 + np.tanh(alpha*(S - sth))) / alpha
    phi = scipy.signal.medfilt2d(phi, kernel_size=3)

    fm = 0.5 + 0.5 * np.tanh(phi[:,:,np.newaxis] * (fm - 1))

    fmn = np.sum(fm, axis=2)

    return np.sum(im * fm / fmn[:,:,np.newaxis], axis=2)


def fstack_channels(im, wsize=9, alpha=0.2, sth=13, focus=None):
    nchannels = im.shape[2]
    im_out = np.empty(im.shape[:-1])
    for channel in range(nchannels):
        im_out[:,:,channel] = fstack(
            im[:,:,channel,:].squeeze(),
            wsize,
            alpha,
            sth,
            focus
        )

    return im_out
