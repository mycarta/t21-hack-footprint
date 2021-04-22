import numpy as np
import xarray as xr
from scipy import signal
from numpy.fft import fft2, ifft2, ifftshift
import dask.array as da
import dask.array.fft as dff
import torch

###################################

# Normalisation

###################################


def normalise(arr):
    """
    Normalises an array of 2 or 3 dimensions. If a 3D array is passed the normalisation occurs on the first two axes.
    """
    if len(arr.shape) == 2:
        return _normalise_2D(arr)
    elif len(arr.shape) == 3:
        return _normalise_3D(arr)
    else:
        raise AttributeError(
            f"Unknown number of dimensions of array to normalise. Array deimensions: {len(arr.shape)}"
        )


def _normalise_2D(arr):
    return (arr - np.amin(arr)) / (np.amax(arr) - np.amin(arr))


def _normalise_3D(arr):
    """
    Normalisation along first two axes
    """
    result = (arr - np.amin(arr, axis=(0, 1))) / (
        np.amax(arr, axis=(0, 1)) - np.amin(arr, axis=(0, 1))
    )
    result[np.isnan(result)] = 0.0
    # return (arr - np.amin(arr, axis=(0, 1))) / (np.amax(arr, axis=(0, 1)) - np.amin(arr, axis=(0, 1)))
    return result


###################################

# Padding

###################################


def pad_next_square_size(arr):
    """
    Normalises an array of 2 or 3 dimensions. If a 3D array is passed the normalisation occurs on the first two axes.
    """
    if len(arr.shape) == 2:
        return _pad_next_square_size_2D(arr)
    elif len(arr.shape) == 3:
        return _pad_next_square_size_3D(arr)
    else:
        raise AttributeError(
            f"Unknown number of dimensions of array to normalise. Array deimensions: {len(arr.shape)}"
        )


def reverse_padding(arr, filtered_arr, slc):
    """
    Normalises an array of 2 or 3 dimensions. If a 3D array is passed the normalisation occurs on the first two axes.
    """
    if len(arr.shape) == 2:
        return _reverse_padding_2D(arr, filtered_arr, slc)
    elif len(arr.shape) == 3:
        return _reverse_padding_3D(arr, filtered_arr, slc)
    else:
        raise AttributeError(
            f"Unknown number of dimensions of array to normalise. Array deimensions: {len(arr.shape)}"
        )


def _pad_next_square_size_3D(im):
    """Function to pad a rectangualr image to a square image.
    Parameters:
    im (2D array): input grayscale image

    Returns:
    out (2D array): padded input image
    padding (slice object): a slice object that can be later passed to reverse_padding

    Example:
    out, padding = pad_next_square_size(im)
    """
    m, n, _ = np.shape(im)  # get input shape
    deficit = max([m, n]) - min([m, n])  # get deficit between size lengths

    # difference in dimensions is even, pad both sides of short dimension by deficit//2
    if deficit % 2 == 0:
        deficit1 = deficit // 2
        deficit2 = deficit // 2

    # difference in dimensions is odd, pad one side by deficit//2 +1
    else:
        deficit1 = deficit // 2
        deficit2 = deficit1 + 1

    if m > n:
        #         print("Padded image columns")
        return (
            np.pad(im, ((0, 0), (deficit1, deficit2), (0, 0)), "reflect"),
            slice(deficit1, -deficit2),
        )

    else:
        #         print("Padded image rows")
        return (
            np.pad(im, ((deficit1, deficit2), (0, 0), (0, 0)), "reflect"),
            slice(deficit1, -deficit2),
        )


def _pad_next_square_size_2D(im):
    """Function to pad a rectangualr image to a square image.
    Parameters:
    im (2D array): input grayscale image

    Returns:
    out (2D array): padded input image
    padding (slice object): a slice object that can be later passed to reverse_padding

    Example:
    out, padding = pad_next_square_size(im)
    """
    m, n = np.shape(im)  # get input shape
    deficit = max([m, n]) - min([m, n])  # get deficit between size lengths

    # difference in dimensions is even, pad both sides of short dimension by deficit//2
    if deficit % 2 == 0:
        deficit1 = deficit // 2
        deficit2 = deficit // 2

    # difference in dimensions is odd, pad one side by deficit//2 +1
    else:
        deficit1 = deficit // 2
        deficit2 = deficit1 + 1

    if m > n:
        #         print("Padded image columns")
        return (
            np.pad(im, ((0, 0), (deficit1, deficit2)), "reflect"),
            slice(deficit1, -deficit2),
        )

    else:
        #         print("Padded image rows")
        return (
            np.pad(im, ((deficit1, deficit2), (0, 0)), "reflect"),
            slice(deficit1, -deficit2),
        )


def _reverse_padding_2D(im, filtered_im, slc):
    m, n = np.shape(im)  # get input shape

    if m > n:
        #         print("Unpadding image columns")
        return filtered_im[:, slc]
    else:
        #         print("Unpadding image rows")
        return filtered_im[slc, :]


def _reverse_padding_3D(im, filtered_im, slc):
    m, n, _ = np.shape(im)  # get input shape

    if m > n:
        #         print("Unpadding image columns")
        return filtered_im[:, slc, :]
    else:
        #         print("Unpadding image rows")
        return filtered_im[slc, :, :]


###################################

# FFTS

###################################
def apply_filter(arr, _filter, axes=(0, 1)):
    return ifft2(
        np.multiply(ifftshift(1 - _filter), fft2(arr, axes=axes)),
        axes=axes,
    ).real


def apply_filter_iterative(_filter, arr):
    # %%timeit
    out = arr.copy()
    # loop through time slices
    for i in range(arr.shape[-1]):

        # pad slice
        ts, slc = pad_next_square_size(out[:, :, i])

        # do all the FFT magic to apply the filter to the slice
        temp = apply_filter(ts, _filter)

        # reverse the padding
        out[:, :, i] = reverse_padding(arr[:, :, i], temp, slc)

    return out


def apply_filter_vector(_filter, arr):
    out = arr.copy()
    tc, slc = pad_next_square_size(out)
    temp = apply_filter(tc, _filter[:, :, None])
    return reverse_padding(arr, temp, slc)


def apply_filter_vector_dask(_filter, arr):
    out = arr.copy()
    tc, slc = pad_next_square_size(out)
    tc = da.from_array(tc, (-1, -1, 5))
    _filter = da.from_array(_filter)
    temp = apply_filter(tc, _filter[:, :, None])
    return reverse_padding(arr, temp, slc)


def apply_filter_vector_dask_true(_filter, arr, chunk=5):
    out = arr.copy()
    tc, slc = pad_next_square_size(out)
    tc = da.from_array(tc, (-1, -1, chunk))
    _filter = da.from_array(_filter)
    temp = dff.ifft2(
        da.multiply(dff.ifftshift(1 - _filter[:, :, None]), dff.fft2(tc, axes=(0, 1))),
        axes=(0, 1),
    ).real
    return reverse_padding(arr, temp, slc)


###################################

# FFTS Pytorch

###################################


def apply_filter_pytorch(arr, _filter, axes=(0, 1), cuda=False):
    """
    Applying Filter using pytorch for iterative method.
    """
    _filter = torch.from_numpy(_filter)
    arr = torch.from_numpy(arr)
    if cuda is True:
        _filter = _filter.cuda()
        arr = arr.cuda()
    result = torch.fft.ifft2(
        torch.multiply(
            torch.fft.ifftshift(1.0 - _filter), torch.fft.fft2(arr, dim=axes)
        )
    )
    if cuda is True:
        result = result.cpu()
    result = result.numpy().real
    return result


def apply_filter_pytorch_vec(arr, _filter, axes=(0, 1), cuda=False):
    """
    Applying Filter using pytorch for vectorized method.
    """
    _filter = torch.from_numpy(_filter)
    arr = torch.from_numpy(arr)
    if cuda is True:
        _filter = _filter.cuda()
        arr = arr.cuda()
    result = torch.fft.ifft2(
        torch.multiply(
            torch.fft.ifftshift(1.0 - _filter), torch.fft.fft2(arr, dim=axes)
        ),
        dim=axes,
    )
    if cuda is True:
        result = result.cpu()
    result = result.numpy().real
    return result


def apply_filter_iterative_pytorch(_filter, arr, cuda=False):
    """
    Filter workflow (pad-filter-unpad) using iterative method.
    """
    out = arr.copy()
    # loop through time slices
    for i in range(arr.shape[-1]):

        # pad slice
        ts, slc = pad_next_square_size(out[:, :, i])

        # do all the FFT magic to apply the filter to the slice
        if cuda is True:
            temp = apply_filter_pytorch(ts, _filter, cuda=True)
        else:
            temp = apply_filter_pytorch(ts, _filter)
        # reverse the padding
        out[:, :, i] = reverse_padding(arr[:, :, i], temp, slc)

    return out


def apply_filter_vector_pytorch(_filter, arr, cuda=False):
    """
    Filter workflow (pad-filter-unpad) using vectorized method.
    """
    out = arr.copy()
    tc, slc = pad_next_square_size(out)

    if cuda is True:
        temp = apply_filter_pytorch_vec(tc, _filter[:, :, None], cuda=True)
    else:
        temp = apply_filter_pytorch_vec(tc, _filter[:, :, None], cuda=False)
    return reverse_padding(arr, temp, slc)


###################################

#  Generic

###################################
def time_cube_to_xarray(time_cube):
    """
    Converts a 3D numpy array of x,y,time to an xarray object.
    Xarray can be very useful.
        - You can deal with coordiantes by name and not axes
        - Great built in interaction with holiviews / panel
        - Stores underlying data as `.values`
    """
    ds_xr = xr.Dataset(
        coords={
            "x": np.arange(time_cube.shape[1]),
            "y": np.arange(time_cube.shape[0]),
            # Here we assume time slices are equally but we could actually set the seconds
            # That correspond
            "time_slice": np.arange(time_cube.shape[2]),
        },
        data_vars={"amplitude": (("y", "x", "time_slice"), time_cube)},
    )
    return ds_xr


def scipy_gaussian_2D(std):
    """
    2D Gaussian filter kernel similar to astropy\'s Gaussian2DKernel
    (https://docs.astropy.org/en/stable/api/astropy.convolution.Gaussian2DKernel.html#astropy.convolution.Gaussian2DKernel)
    using scipy.signal.gaussian
    (and inspired by https://gist.github.com/thomasaarholt/267ec4fff40ca9dff1106490ea3b7567)

    Parameters:
    std (int) : standard deviation of the Gaussian in pixels

    Returns:
    out (2D array): 2D Gaussian filter kernel
    """
    ksp1D = signal.gaussian(std * 8 + 1, std)
    ksp2D = np.outer(ksp1D, ksp1D)
    ksp2D /= 2 * np.pi * (std ** 2)
    return ksp2D
