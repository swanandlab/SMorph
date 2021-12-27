import czifile
import psf
import skimage
import numpy as np

from skimage.restoration import (calibrate_denoiser, denoise_nl_means,
                                 estimate_sigma)

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
    denoised = np.zeros(img.shape, dtype=np.float64)

    for i in range(denoised.shape[0]):
        denoised[i] = denoise_nl_means(img[i], **denoise_parameters)

    denoised = imnorm(denoised)
    return denoised


def deconvolve(img, impath, iters=8, pinhole_shape='round'):
    """Do in-place deconvolution.

    Parameters
    ----------
    img : ndarray
        Image data.
    impath : str
        Path to the original image file.
    iters : int, optional
        Number of iterations for deconvolution, by default 8
    pinhole_shape : str, optional
        Shape of the pinhole of the confocal microscope. Either 'round' or
        'square', by default 'round'

    Returns
    -------
    ndarray
        Deconvolved image data.

    """
    impath = impath.lower()
    if impath.split('.')[-1] == 'czi':
        czimeta = czifile.CziFile(impath).metadata(False)
        metadata = czimeta['ImageDocument']['Metadata']
        im_meta = metadata['Information']['Image']
        refr_index = im_meta['ObjectiveSettings']['RefractiveIndex']

        selected_channel = None
        for i in im_meta['Dimensions']['Channels']['Channel']:
            if i['ContrastMethod'] == 'Fluorescence':
                selected_channel = i
        ex_wavelen = selected_channel['ExcitationWavelength']
        em_wavelen = selected_channel['EmissionWavelength']

        selected_detector = None
        for i in metadata['Experiment']['ExperimentBlocks']['AcquisitionBlock'
            ]['MultiTrackSetup']['TrackSetup']['Detectors']['Detector']:  # [channel]['Detectors']['Detector']:
            if i['PinholeDiameter'] > 0:
                selected_detector = i
        pinhole_radius = selected_detector['PinholeDiameter'] / 2 * 1e6

        num_aperture = metadata['Information']['Instrument']['Objectives'][
            'Objective']['LensNA']
        dim_r = metadata['Scaling']['Items']['Distance'][0]['Value'] * 1e6
        dim_z = metadata['Scaling']['Items']['Distance'][-1]['Value'] * 1e6

        args = dict(
            shape=(3, 3),  # # of samples in z & r direction
            dims=(dim_z, dim_r),  # size in z & r direction in microns
            ex_wavelen=ex_wavelen,  # nm
            em_wavelen=em_wavelen,  # nm
            num_aperture=num_aperture,
            refr_index=refr_index,
            pinhole_radius=pinhole_radius,  # microns
            pinhole_shape=pinhole_shape
        )
        obsvol = psf.PSF(psf.ISOTROPIC | psf.CONFOCAL, **args)
        impsf = obsvol.volume()

        img = skimage.restoration.richardson_lucy(img, impsf, num_iter=iters)
        img = (img - img.min()) / (img.max() - img.min())

    return img
