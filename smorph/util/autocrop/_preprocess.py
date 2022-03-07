import czifile
import psf
import skimage
import numpy as np

from skimage.exposure import (
    equalize_adapthist,
    match_histograms,
)
from skimage.restoration import (
    calibrate_denoiser,
    denoise_nl_means,
    estimate_sigma,
    richardson_lucy,
    rolling_ball,
)

from .util import (
    imnorm,
)


def calibrate_nlm_denoiser(img):
    """Calibrate Non-local means denoiser parameters Using J-Invariance.

    Parameters
    ----------
    img : ndarray
        Image data.

    Returns
    -------
    dict
        Calculated parameters for Non-local means denoiser.

    """
    img_max_projection = np.max(img, 0)
    sigma_est = estimate_sigma(img_max_projection)

    # higher patch_size, lesser particles
    # lesser patch_distance, more branching maintained
    parameter_ranges = {'h': np.arange(.8, 1.2, .2) * sigma_est,
                        'patch_size': np.arange(2, 6),
                        'patch_distance': np.arange(2, 6)}

    denoiser = calibrate_denoiser(img_max_projection,
                                  denoise_nl_means, parameter_ranges)
    denoise_parameters = denoiser.keywords['denoiser_kwargs']
    return denoise_parameters


def denoise(img, denoise_parameters):
    """Removes noise from image using calibrated parameters.

    Parameters
    ----------
    img : ndarray
        Image data.
    denoise_parameters : dict
        Parameters returned by `calibrate_nlm_denoiser`.

    Returns
    -------
    ndarray
        Denoised image data.

    """
    denoised = np.empty_like(img, dtype=np.float64)

    if img.ndim > 2:
        for i in range(denoised.shape[0]):
            denoised[i] = denoise_nl_means(img[i], **denoise_parameters)
    else:
        denoised = denoise_nl_means(img, **denoise_parameters)

    # denoised = imnorm(denoised)
    return denoised


def deconvolve(pipe, iters=8, **kwargs):
    """Do in-place deconvolution.

    Parameters
    ----------
    # img : ndarray
    #     Image data.
    # impath : str
    #     Path to the original image file.
    iters : int, optional
        Number of iterations for deconvolution, by default 8
    # pinhole_shape : str, optional
    #     Shape of the pinhole of the confocal microscope. Either 'round' or
    #     'square', by default 'round'

    Returns
    -------
    ndarray
        Deconvolved image data.

    """
    img = pipe.impreprocessed
    impath = pipe.im_path.lower()
    dim_z = pipe.SCALE[0]
    dim_r = pipe.SCALE[-1]

    if (
        kwargs.get('ex_wavelen') and kwargs.get('ex_wavelen')
        and kwargs.get('num_aperture') and kwargs.get('refr_index')
        and kwargs.get('pinhole_radius') and kwargs.get('pinhole_shape')
    ):
        args = dict(
            shape=(3, 3),  # # of samples in z & r direction
            dims=(dim_z, dim_r),  # size in z & r direction in microns (very important)
            ex_wavelen=kwargs['ex_wavelen'],  # nm
            em_wavelen=kwargs['em_wavelen'],  # nm
            num_aperture=kwargs['num_aperture'],
            refr_index=kwargs['refr_index'],
            pinhole_radius=kwargs['pinhole_radius'],  # microns
            pinhole_shape=kwargs['pinhole_shape']
        )
        obsvol = psf.PSF(psf.ISOTROPIC | psf.CONFOCAL, **args)
        impsf = obsvol.volume()

        img = richardson_lucy(img, impsf, num_iter=iters)

    img = (img - img.min()) / (img.max() - img.min())

    return img


def subtract_background(im, radius=50):
    bg = np.empty_like(im)

    for i in range(im.shape[0]):
        bg[i] = rolling_ball(im[i], radius=radius)

    return im - bg


def equalize(im, clip_limit=.01):
    equalized = equalize_adapthist(im, clip_limit=clip_limit)
    return equalized
