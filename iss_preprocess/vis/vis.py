import numpy as np
import matplotlib.pyplot as plt


def to_rgb(stack, colors, vmax=None, vmin=None):
    """
    Convert multichannel stack to RGB image.

    Args:
        stack: X x Y x C stack.
        colors: maximum RGB values for each channel.
        vmax: maximum level for each channel
        vmin: minimum level for each channel

    Returns:
        X x Y x 3 RGB image

    """
    nchannels = stack.shape[2]

    if vmax is None:
        vmax = np.max(stack, axis=(0,1))
    if vmin is None:
        vmin = np.min(stack, axis=(0,1))

    scale = vmax - vmin

    stack_norm = ((stack - vmin[np.newaxis, np.newaxis, :]) / scale[np.newaxis, np.newaxis, :] * 255).astype(int)
    stack_norm = np.clip(stack_norm, 0, 255)
    rgb_stack = np.zeros([stack.shape[0], stack.shape[1], 3])
    for ich in range(nchannels):
        lut = make_lut(colors[ich])
        rgb_stack += lut[stack_norm[:,:,ich], :]

    return np.clip(rgb_stack, 0., 1.)


def make_lut(color, nlevels=256):
    """
    Create a look-up table by interpolating between (0,0,0) and provided color.

    Args:
        color: the maximum RGB value for look-up table.
        nlevels: nummber of LUT levels

    Returns:
        nlevels x 3 matrix.

    """
    r = np.linspace(0., 1., nlevels)
    return r[:,np.newaxis] * color


def plot_spots(stack, spots,
               colors=[[1.,0.,0.], [0.,1.,0.], [1.,0.,1.], [0.,1.,1.]], vmax=10000., vmin=500.):
    """
    Visualize detected plots.

    Args:
        stack: X x Y x C stack
        spots: pandas.DataFrame of spot locations
        colors: maximum RGB values for each channel.
        vmax: maximum level for each channel
        vmin: minimum level for each channel

    """
    im = to_rgb(stack, colors,
                vmax=np.array([vmax]),
                vmin=np.array([vmin]))
    plt.figure(figsize=(20,20))
    ax = plt.subplot(1,1,1)
    ax.imshow(im)
    for _, spot in spots.iterrows():
        c = plt.Circle((spot.x, spot.y), spot.size, color='w', linewidth=.5, fill=False)
        ax.add_patch(c)