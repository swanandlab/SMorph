import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.morphology import closing, square
from skimage.transform import match_histograms  # TODO:0.18 moved to exposure
from skimage.util import invert


def preprocess_image(image, image_type, reference_image, crop_tech='manual'):
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

    Returns
    -------
    ndarray
        Thresholded, denoised, boolean transformation of `image` with solid
        soma.

    """
    thresholded_image = _threshold_image(image, image_type,
                                         reference_image, crop_tech)
    cleaned_image = _remove_small_object_noise(thresholded_image)
    cleaned_image_filled_holes = _fill_holes(cleaned_image)
    return cleaned_image_filled_holes


def _threshold_image(image, image_type, reference_image, crop_tech):
    """Single intensity threshold via Otsu's method."""
    if reference_image is not None:
        gray_reference_image = rgb2gray(reference_image)
        image = match_histograms(image, gray_reference_image)

    # Contrast stretching
    p2, p98 = np.percentile(image, (2, 98))
    img_rescale = rescale_intensity(image, in_range=(p2, p98))

    if crop_tech == 'manual':
        thresholded_cell = img_rescale > threshold_otsu(img_rescale)
    else:
        thresholded_cell = image > 0

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
