import czifile
import matplotlib.pyplot as plt
import numpy as np
import psf
import skimage

from matplotlib import gridspec
from psutil import virtual_memory
from scipy.spatial import ConvexHull
from scipy.ndimage import find_objects
from skimage.draw import polygon2mask
from skimage.filters import apply_hysteresis_threshold, sobel
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion
from skimage.restoration import (calibrate_denoiser, denoise_nl_means,
                                 estimate_sigma)
from skimage.segmentation import clear_border
from skimage.util import unique_rows


def _testThresholds(  #TODO: redundant
    im,
    low_thresh=.06,
    high_thresh=.45,
    low_delta=.01,
    high_delta=.05,
    n=1
):
    n_labels = 2 * n + 1
    low_thresh -= low_delta * n
    high_thresh -= high_delta * n
    out = []
    for i in range(n_labels):
        thresholded = threshold(im, low_thresh + i * low_delta,
                                high_thresh + i * high_delta)
        labels = label_thresholded(thresholded)
        filtered_regions = arrange_regions(labels)
        labels = np.zeros_like(thresholded, dtype=int)
        reg_itr = 1
        for region in filtered_regions:
            minz, miny, minx, maxz, maxy, maxx = region['bbox']
            labels[minz:maxz, miny:maxy, minx:maxx] += region['image'] * reg_itr
            reg_itr += 1
        out.append({'data': labels,
                    'name': (f'L:{low_thresh + i * low_delta:.3f}, '
                             f'H:{high_thresh + i * high_delta:.3f}')})
    return out

def testThresholds(
    edge_filtered,
    low_thresh=.06,
    high_thresh=.45,
    low_delta=.01,
    high_delta=.05,
    n=1,
    cmap='gray'
):
    N_COLS = 2 * n + 1
    fig, axes = plt.subplots(ncols=N_COLS, nrows=1, figsize=((8, 8)
                             if N_COLS == 1 else (16, 16)))
    low_thresh -= low_delta * n
    high_thresh -= high_delta * n
    out = []
    for i in range(N_COLS):
        thresholded = threshold(edge_filtered, low_thresh + i * low_delta,
                                high_thresh + i * high_delta)
        labels = label_thresholded(thresholded)
        filtered_regions = arrange_regions(labels)
        labels = np.zeros_like(thresholded, dtype=int)
        reg_itr = 1
        for region in filtered_regions:
            minz, miny, minx, maxz, maxy, maxx = region['bbox']
            labels[minz:maxz, miny:maxy, minx:maxx] += region['image'] * reg_itr
            reg_itr += 1

        curr_ax = axes if N_COLS == 1 else axes[i]
        curr_ax.imshow(labels.max(axis=0), cmap=cmap)
        curr_ax.set_title(f'L:{low_thresh + i * low_delta:.4f}, '
                        f'H:{high_thresh + i * high_delta:.4f}')
        out.append({'data': labels, 'colormap': 'gist_earth', 'gamma': .8,
                    'name': (f'L:{low_thresh + i * low_delta:.3f}, '
                             f'H:{high_thresh + i * high_delta:.3f}')})

    fig.tight_layout()
    plt.show()
    return out


def projectXYZ(img, voxel_sz_x, voxel_sz_y, voxel_sz_z, cmap='gray'):
    """Projects a 3D image in all planes.

    Parameters
    ----------
    img : ndarray
        Image data.
    voxel_sz_x : int
        Spacing of voxel in X axis.
    voxel_sz_y : int
        Spacing of voxel in Y axis.
    voxel_sz_z : int
        Spacing of voxel in Z axis.
    cmap : str, optional
        Color map for displaying, by default 'gray'

    """
    aspect_xz = voxel_sz_z / voxel_sz_x
    aspect_yz = voxel_sz_z / voxel_sz_y
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=[img.shape[2]*voxel_sz_x,
                                               img.shape[0]*voxel_sz_z],
                           height_ratios=[img.shape[1]*voxel_sz_y,
                                          img.shape[0]*voxel_sz_z],
                           wspace=.05, hspace=.05)
    ax = plt.subplot(gs[0, 0])
    axr = plt.subplot(gs[0, 1], sharey=ax)
    axb = plt.subplot(gs[1, 0], sharex=ax)

    ax.imshow(np.max(img, axis=0), cmap=cmap)
    axb.imshow(np.max(img, axis=1), cmap, aspect=aspect_xz)
    axr.imshow(np.max(img, axis=2).T, cmap, aspect=1/aspect_yz)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(axr.get_yticklabels(), visible=False)
    plt.show()


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
    return denoiser


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
    denoised = np.zeros(img.shape)

    for i in range(denoised.shape[0]):
        denoised[i] = denoise_nl_means(img[i], **denoise_parameters)

    return denoised


def deconvolve(img, impath, iters=8):
    """Do in-place deconvolution.

    Parameters
    ----------
    img : ndarray
        Image data.
    impath : str
        Path to the original image file.
    iters : int, optional
        Number of iterations for deconvolution, by default 8

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
            pinhole_shape='square'
        )
        obsvol = psf.PSF(psf.ISOTROPIC | psf.CONFOCAL, **args)
        impsf = obsvol.volume()

        img = skimage.restoration.richardson_lucy(img, impsf, iterations=iters)
        img = (img - img.min()) / (img.max() - img.min())

    return img


def filter_edges(img):
    sobel_edges = np.zeros(img.shape)

    for j in range(img.shape[0]):
        sobel_edges[j] = sobel(img[j])

    return sobel_edges


def threshold(edge_filtered, lw_thresh, hi_thresh):
    thresholded = apply_hysteresis_threshold(edge_filtered,
                                             lw_thresh, hi_thresh)
    return thresholded


def label_thresholded(thresholded):
    labels = label(thresholded)
    return labels


def _check_coords_in_hull(gridcoords, hull_equations, tolerance):
    ndim, n_gridcoords = gridcoords.shape
    coords_in_hull = np.zeros(n_gridcoords, dtype=np.bool_)
    n_hull_equations = hull_equations.shape[0]

    available_mem = .75 * virtual_memory().available
    required_mem = 96 + n_hull_equations * n_gridcoords * 8
    chunk_size = (int(required_mem // available_mem)
                  if available_mem < required_mem else n_gridcoords)

    # Pre-allocate arrays to cache intermediate results for reducing overheads
    dot_array = np.zeros((n_hull_equations, chunk_size))
    test_ineq_temp = np.zeros((n_hull_equations, chunk_size))
    coords_chunk_ineq = np.zeros((n_hull_equations, chunk_size),
                                 dtype=np.bool_)
    coords_all_ineq = np.zeros(chunk_size, dtype=np.bool_)

    # Apply the test in chunks
    for idx in range(0, n_gridcoords, chunk_size):
        coords = gridcoords[:, idx: idx + chunk_size]
        n_coords = coords.shape[1]

        if n_coords == chunk_size:
            np.dot(hull_equations[:, :ndim], coords,
                   out=dot_array if n_coords == chunk_size else None)
            np.add(dot_array, hull_equations[:, ndim:],
                   out=test_ineq_temp if n_coords == chunk_size else None)
            np.less(test_ineq_temp, tolerance, out=coords_chunk_ineq)
            coords_in_hull[idx : idx + chunk_size] = np.all(
                coords_chunk_ineq,
                axis=0,
                out=coords_all_ineq
            )
        else:
            coords_in_hull[idx: idx + chunk_size] = np.all(
                np.add(
                    np.dot(hull_equations[:, :ndim], coords),
                    hull_equations[:, ndim:]
                ) < tolerance,
                axis=0
            )

    return coords_in_hull


def _compute_convex_hull(thresholded, tolerance=1e-10):
    surface = thresholded ^ binary_erosion(thresholded)
    ndim = surface.ndim
    if np.count_nonzero(surface) == 0:
        return np.zeros(surface.shape, dtype=np.bool_)

    coords = np.transpose(np.nonzero(surface))  # 3d only

    # repeated coordinates can *sometimes* cause problems in
    # scipy.spatial.ConvexHull, so we remove them.
    coords = unique_rows(coords)

    # Find the convex hull
    hull = ConvexHull(coords)
    vertices = hull.points[hull.vertices]

    gridcoords = np.reshape(np.mgrid[tuple(map(slice, surface.shape))],
                            (ndim, -1))
    # A point is in the hull if it satisfies all of the hull's inequalities
    coords_in_hull = _check_coords_in_hull(gridcoords,
                                           hull.equations, tolerance)
    mask = np.reshape(coords_in_hull, surface.shape)

    return mask


def _unwrap_polygon(polygon):
    if type(polygon) != np.ndarray:
        X, Y = polygon.xs, polygon.ys
    else:
        polygon = list(zip(*polygon))
        X, Y = polygon[0], polygon[1]
    return X, Y


def filter_labels(labels, thresholded, polygon=None, conservative=True):
    filtered_labels = clear_border(labels, mask=None if conservative
                                   else _compute_convex_hull(thresholded))

    if polygon is not None:
        X, Y = _unwrap_polygon(polygon)
        min_x, max_x = int(min(X)), int(max(X) + 1)
        min_y, max_y = int(min(Y)), int(max(Y) + 1)
        shape = labels.shape
        roi_mask = np.empty(shape)
        roi_mask[0] = polygon2mask((max_x, max_y), list(zip(X, Y))
                                  )[min_x:max_x, min_y:max_y].T

        for i in range(1, shape[0]):
            roi_mask[i] = roi_mask[0]
        roi_mask = binary_erosion(roi_mask)
        filtered_labels = clear_border(filtered_labels, mask=roi_mask)

    return filtered_labels


def _filter_small_objects(regions, cutoff_volume=64):
    idx = 0
    for region in regions:
        if region['vol'] > cutoff_volume:
            break
        idx += 1
    filtered = regions[idx:]
    return filtered


def arrange_regions(filtered_labels):
    objects = find_objects(filtered_labels)
    ndim = filtered_labels.ndim

    regions = []
    for itr, slice in enumerate(objects):
        if slice is None:
            continue
        label = itr + 1
        template = dict(image=None, bbox=None, centroid=None, vol=None)
        template['image'] = (filtered_labels[slice] == label)
        template['bbox'] = tuple([slice[i].start for i in range(ndim)] +
                                 [slice[i].stop for i in range(ndim)])
        indices = np.nonzero(template['image'])
        coords = np.vstack([indices[i] + slice[i].start
                            for i in range(ndim)]).T
        template['centroid'] = tuple(coords.mean(axis=0))
        template['vol'] = np.sum(template['image'])
        regions.append(template)

    regions = sorted(regions,
                     key=lambda region: region['vol'])
    regions = _filter_small_objects(regions)
    return regions


def paginate_objs(regions, pg_size=50):
    N_OBJECTS = len(regions)
    print(f'{N_OBJECTS} objects detected.')
    N_BATCHES = N_OBJECTS // pg_size + (N_OBJECTS % pg_size > 0)
    print(f'There will be {N_BATCHES} batches, set `BATCH_NO` '
          f'from 0 to {N_BATCHES-1} inclusive')
    return N_BATCHES


def extract_obj(region, tissue_img):
    minz, miny, minx, maxz, maxy, maxx = region['bbox']

    extracted_obj = tissue_img[minz:maxz, miny:maxy, minx:maxx].copy()
    extracted_obj[~region['image']] = 0.0

    print(f'Volume of this object is: {region["vol"]}')
    return extracted_obj


def project_batch(BATCH_NO, N_BATCHES, regions, tissue_img):
    N_OBJECTS = len(regions)

    if BATCH_NO >= N_BATCHES:
        raise ValueError(f'BATCH_NO should only be from 0 to {N_BATCHES-1}!')

    fig = plt.figure(figsize=(11, 20))
    rows, columns = 10, 5
    ax = []

    idx = 0
    l_obj = BATCH_NO * 50
    for obj in range(l_obj, min(50 + l_obj, N_OBJECTS)):
        minz, miny, minx, maxz, maxy, maxx = regions[obj]['bbox']
        ax.append(fig.add_subplot(rows, columns, idx+1))
        idx += 1
        ax[-1].set_title(f'Obj {obj}; Vol: {regions[obj]["vol"]}')
        ax[-1].axis('off')

        extracted_cell = tissue_img[minz:maxz, miny:maxy, minx:maxx].copy()
        extracted_cell[~regions[obj]['image']] = 0.0

        plt.imshow(np.max(extracted_cell, 0), cmap='gray')
