import json
from os import getcwd, mkdir, path
from shutil import rmtree

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import tifffile
from aicsimageio.readers import CziReader
from psutil import virtual_memory
from scipy.spatial import ConvexHull
from skimage import img_as_float, img_as_ubyte
from skimage.filters import apply_hysteresis_threshold, sobel
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion
from skimage.restoration import (calibrate_denoiser, denoise_nl_means,
                                 estimate_sigma)
from skimage.segmentation import clear_border
from skimage.util import unique_rows


def projectXYZ(img, voxel_sz_x, voxel_sz_y, voxel_sz_z, cmap='gray'):
    """Projects image in all planes.

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
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(15, 8))

    axes[0].imshow(np.max(img, axis=0), cmap=cmap)
    axes[1].imshow(np.max(img, axis=1), cmap, aspect=aspect_xz)
    axes[2].imshow(np.max(img, axis=2), cmap, aspect=aspect_yz)
    plt.show()


def import_confocal_image(img_path):
    """Loads the 3D confocal image.

    Tested on: CZI, TIFF

    Parameters
    ----------
    img_path : str
        Path to the confocal tissue image.

    """
    # image has to be converted to float for processing
    if img_path.split('.')[-1] == 'czi':
        img = CziReader(img_path)
        img = img.data
        img = img_as_float(np.squeeze(img)[0])
    else:
        img = img_as_float(io.imread(img_path))

    return img


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

    denoiser = calibrate_denoiser(img_max_projection, denoise_nl_means, parameter_ranges)
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

    for i in range(denoised.shape[1]):
        denoised[:, i] = denoise_nl_means(denoised[:, i], **denoise_parameters)

    return denoised


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
    chunk_size = int(required_mem // available_mem) if available_mem < required_mem else n_gridcoords

    # Pre-allocate arrays to cache intermediate results for reducing overheads
    dot_array = np.zeros((n_hull_equations, chunk_size))
    test_ineq_temp = np.zeros((n_hull_equations, chunk_size))
    coords_chunk_ineq = np.zeros((n_hull_equations, chunk_size), dtype=np.bool_)
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


def compute_convex_hull(thresholded, tolerance=1e-10):
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

    gridcoords = np.reshape(np.mgrid[tuple(map(slice, surface.shape))], (ndim, -1))
    # A point is in the hull if it satisfies all of the hull's inequalities
    coords_in_hull = _check_coords_in_hull(gridcoords, hull.equations, tolerance)
    mask = np.reshape(coords_in_hull, surface.shape)

    return mask


def filter_labels(labels, convex_hull):
    filtered_labels = clear_border(clear_border(labels), mask=binary_erosion(convex_hull))
    return filtered_labels


def arrange_regions(filtered_labels):
    regions = sorted(regionprops(filtered_labels),
                     key=lambda region: region.area)
    return regions


def project_batch(BATCH_NO, N_BATCHES, N_OBJECTS, regions, denoised):
    if BATCH_NO >= N_BATCHES:
        raise ValueError(f'BATCH_NO should only be from 0 to {N_BATCHES-1}!')

    w = h = 10
    fig = plt.figure(figsize=(11, 20))
    rows, columns = 10, 5
    ax = []

    idx = 0
    l_obj = BATCH_NO * 50
    for obj in range(l_obj, min(50 + l_obj, N_OBJECTS)):
        minz, miny, minx, maxz, maxy, maxx = regions[obj].bbox
        ax.append(fig.add_subplot(rows, columns, idx+1))
        idx += 1
        ax[-1].set_title(f'Obj {obj}; Vol: {regions[obj].area}')
        ax[-1].axis('off')

        extracted_cell = denoised[minz:maxz, miny:maxy, minx:maxx].copy()
        extracted_cell[~regions[obj].filled_image] = 0.0

        plt.imshow(np.max(extracted_cell, 0), cmap='gray')


def export_cells(img_path, low_vol_cutoff, hi_vol_cutoff, output_option, denoised, regions):
    OUTPUT_TYPE = ('3D', 'MIP')[output_option]

    DIR = getcwd() + '/autocropped/'
    if not (path.exists(DIR) and path.isdir(DIR)):
        mkdir(DIR)

    IMAGE_NAME = path.basename(img_path).split('.')[0]
    OUTPUT_DIR = DIR + IMAGE_NAME + f'_{OUTPUT_TYPE}/'
    if path.exists(OUTPUT_DIR) and path.isdir(OUTPUT_DIR):
        rmtree(OUTPUT_DIR)
    mkdir(OUTPUT_DIR)

    if img_path.split('.')[-1] == 'tif':
        with tifffile.TiffFile(img_path) as file:
            metadata = file.imagej_metadata

        cell_metadata = {}
        cell_metadata['unit'] = metadata['unit']
        cell_metadata['spacing'] = metadata['spacing']
        cell_metadata = json.dumps(cell_metadata)
    else:
        cell_metadata = None

    for (obj, region) in enumerate(regions):
        if low_vol_cutoff < region.area < hi_vol_cutoff:
            minz, miny, minx, maxz, maxy, maxx = region.bbox

            segmented = img_as_ubyte(denoised[minz:maxz, miny:maxy, minx:maxx].copy())
            segmented[~region.filled_image] = 0

            out_data = segmented if OUTPUT_TYPE == '3D' else np.max(segmented, 0)

            name = f'{OUTPUT_DIR}cell{obj}-{minz},{maxz},{miny},{maxy},{minx},{maxx}.tif'
            tifffile.imsave(name, out_data, description=cell_metadata)
