import numpy as np
import skimage.filters as filters

from matplotlib.pyplot import show
from scipy.ndimage import binary_fill_holes
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from skimage.filters import (
    threshold_otsu,
    threshold_isodata,
    threshold_li,
    threshold_mean,
    threshold_minimum,
    threshold_triangle,
    threshold_yen
)
from skimage.measure import label
from skimage.morphology import closing, square
from skimage.exposure import match_histograms  # skimage >= 0.18
from skimage.util import invert

THRESHOLD_METHODS = ('isodata', 'li', 'mean', 'minimum',
                     'otsu', 'triangle', 'yen')


def _validate_img_args(crop_tech, contrast_ptiles, threshold_method):
    if (
        (threshold_method is None)
        and (crop_tech != 'auto')
        and (contrast_ptiles == (0, 100))
    ):
        raise ValueError('`threshold_method` can be None only if images are '
                         'autocropped & image isn\'t contrast stretched.')

    threshold_method = (threshold_method.lower()
                        if threshold_method is not None else threshold_method)

    if (
        (threshold_method not in THRESHOLD_METHODS)
        and threshold_method is not None
    ):
        raise ValueError('`threshold_method` must be either of otsu, '
                         'isodata, li, mean, minimum, triangle, or yen')

    if (
        (type(contrast_ptiles) is not tuple)
        and (0 <= contrast_ptiles[0] < contrast_ptiles[1] <= 100)
    ):
        raise ValueError('`contrast_ptiles` must be a tuple with low & '
                         'high contrast percentiles')

    return contrast_ptiles, threshold_method


def _contrast_stretching(img, contrast_ptiles):
    """Stretches contrast of an image to a band of percentiles of intensities,

    Parameters
    ----------
    img : ndarray
        Image data.
    contrast_ptiles : tuple of size 2
        `(low_percentile, hi_percentile)` Contains ends of band of percentile
        values for pixel intensities to which the contrast of image would be
        stretched

    Returns
    -------
    ndarray
        Contrast rescaled image data.

    """
    p_low, p_hi = np.percentile(img, contrast_ptiles)
    img_rescale = rescale_intensity(img, in_range=(p_low, p_hi))
    return img_rescale


def try_all_threshold(img, contrast_ptiles=(0, 100), figsize=(10, 6)):
    """Applies available automatic single intensity thresholding methods.

    Parameters
    ----------
    img : ndarray
        Image data.
    contrast_ptiles : tuple of size 2, optional
        `(low_percentile, hi_percentile)` Contains ends of band of percentile
        values for pixel intensities to which the contrast of image would be
        stretched, by default (0, 100)
    figsize : tuple, optional
        Figure size (in inches), by default (10, 6)

    """
    contrast_ptiles, _ = _validate_img_args('auto', contrast_ptiles, 'li')
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = rgb2gray(img)
    img_rescale = _contrast_stretching(img, contrast_ptiles)
    filters.try_all_threshold(img_rescale, figsize, False)
    show()


def preprocess_image(
    image,
    image_type,
    reference_image,
    crop_tech='manual',
    contrast_ptiles=(0, 100),
    threshold_method='otsu'
):
    """Extract the individual cell by thresholding & removing background noise.

    Parameters
    ----------
    image : ndarray
        Image data of cell of nervous system.
    image_type : str
        Neuroimaging technique used to get image data of neuronal cell,
        either 'confocal' or 'DAB'.
    reference_image : ndarray
        `image` would be standardized to the exposure level of this example.
    crop_tech : str
        Technique used to crop cell from tissue image,
        either 'manual' or 'auto', by default 'manual'.
    contrast_ptiles : tuple of size 2, optional
        `(low_percentile, hi_percentile)` Contains ends of band of percentile
        values for pixel intensities to which the contrast of cell image
        would be stretched, by default (0, 100)
    threshold_method : str or None, optional
        Automatic single intensity thresholding method to be used for
        obtaining ROI from cell image either of 'otsu', 'isodata', 'li',
        'mean', 'minimum', 'triangle', 'yen'. If None & crop_tech is 'auto' &
        contrast stretch is (0, 100), a single intensity threshold of zero is
        applied, by default 'otsu'

    Returns
    -------
    ndarray
        Thresholded, denoised, boolean transformation of `image` with solid
        soma.

    """
    contrast_ptiles, threshold_method = _validate_img_args(crop_tech,
                                                           contrast_ptiles,
                                                           threshold_method)
    thresholded_image = _threshold_image(image, image_type, reference_image,
                                         crop_tech, contrast_ptiles,
                                         threshold_method)
    cleaned_image = _remove_small_object_noise(thresholded_image)
    cleaned_image_filled_holes = _fill_holes(cleaned_image)

    # Auto-contrast stretching aiding soma detection
    masked_image = cleaned_image_filled_holes * image
    min_intensity, max_intensity = masked_image.min(), masked_image.max()
    image[image < min_intensity] = min_intensity
    image[image > max_intensity] = max_intensity
    image = (image - min_intensity) / (max_intensity - min_intensity)

    return image, cleaned_image_filled_holes


def _threshold_image(
    image,
    image_type,
    reference_image,
    crop_tech,
    contrast_ptiles,
    method
):
    """Single intensity threshold via Otsu's method."""
    if reference_image is not None:
        gray_reference_image = rgb2gray(reference_image)
        image = match_histograms(image, gray_reference_image)

    img_rescale = _contrast_stretching(image, contrast_ptiles)

    THRESHOLD_METHODS = {'otsu': threshold_otsu, 'isodata': threshold_isodata,
                         'li': threshold_li, 'yen': threshold_yen,
                         'mean': threshold_mean, 'minimum': threshold_minimum,
                         'triangle': threshold_triangle}

    if crop_tech == 'auto' and contrast_ptiles == (0, 100) and method is None:
        thresholded_cell = image > 0
    else:
        thresholded_cell = img_rescale > THRESHOLD_METHODS[method](img_rescale)

    if image_type == "DAB":
        return thresholded_cell
    elif image_type == "confocal":
        return invert(thresholded_cell)


def _label_objects(thresholded_image):
    """Label connected regions of a `thresholded_image`."""
    inverted_thresholded_image = invert(thresholded_image)
    bw = closing(inverted_thresholded_image, square(1))
    # label image regions
    labelled_image = label(bw, return_num=True)[0]

    return labelled_image


def _remove_small_object_noise(thresholded_image):
    """Denoise a binary image."""
    labelled_image = _label_objects(thresholded_image)
    labelled_image_1D = labelled_image.reshape(labelled_image.size)
    object_areas = np.bincount(labelled_image_1D[labelled_image_1D != 0])

    largest_object_label = np.argmax(object_areas)
    denoised_image = np.where(labelled_image == [largest_object_label], 1, 0)

    return denoised_image


def _fill_holes(cleaned_image):
    """Fill holes in a binary image."""
    return binary_fill_holes(cleaned_image).astype(int)
