import matplotlib.pyplot as plt
from skimage.util import img_as_float
import numpy as np

try:
    from . import utils
except ImportError:
    import utils

import holoviews as hv


def _numpy_fft2(im):
    F = np.fft.fft2(im)  # Perform 2-dimensional discrete Fourier transform
    C = np.fft.fftshift(F)  # Center spectrum on minimum frequency
    Mag = np.abs(C).real  # Get magnitude and phase
    Ph = np.angle(C).real
    return Mag, Ph


def _raised_cosine(im):
    m, n = np.shape(im)
    w1 = np.cos(np.linspace(-np.pi / 2, np.pi / 2, m))
    w1 = w1[:, None]
    w2 = np.cos(np.linspace(-np.pi / 2, np.pi / 2, n))
    w = w1 * w2
    return utils.normalise(im * w)


def view_spectrum(time_slice):
    A, _ = _numpy_fft2(_raised_cosine(time_slice))
    return hv.Image(np.log(A + 1)).opts(cmap="cubehelix", title="FFT Spectrum")


#     fig = plt.figure(figsize=(12, 12))
#     ax = fig.add_subplot(1, 1, 1)
#     ax.set_title('FFT Spectrum', fontsize=24)
#     plt.imshow(np.log(A +1),  cmap='cubehelix', origin = 'lower', interpolation = 'none'), plt.axis('off');


def view_difference(ori, new):
    # display results
    fig, axes = plt.subplots(1, 3, figsize=(10, 20), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(ori, cmap="gray", aspect=0.5, origin="lower", interpolation="none")
    ax[0].axis("off")
    ax[0].set_title("2D", fontsize=24)

    ax[1].imshow(new, cmap="gray", aspect=0.5, origin="lower", interpolation="none")
    ax[1].axis("off")
    ax[1].set_title("3D", fontsize=24)

    diff = img_as_float(ori - new)

    ax[2].imshow(diff, cmap="gray", aspect=0.5, origin="lower", interpolation="none")
    ax[2].axis("off")
    ax[2].set_title("Difference", fontsize=24)

    fig.tight_layout()
    plt.show()
