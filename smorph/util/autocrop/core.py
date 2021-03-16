import json
import uuid
from os import getcwd, mkdir, path
from shutil import rmtree
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import tifffile
import czifile
from psutil import virtual_memory
from scipy.spatial import ConvexHull
from skimage import img_as_float, img_as_ubyte
from skimage.draw import polygon2mask
from skimage.filters import apply_hysteresis_threshold, sobel
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion
from skimage.restoration import (calibrate_denoiser, denoise_nl_means,
                                 estimate_sigma)
from skimage.segmentation import clear_border
from skimage.util import unique_rows


def testThresholds(
    edge_filtered,
    voxel_sz_x,
    voxel_sz_y,
    voxel_sz_z,
    cmap='gray',
    low_thresh=.06,
    high_thresh=.45,
    low_delta=.01,
    high_delta=.05,
    n=1
):
    aspect_xz = voxel_sz_z / voxel_sz_x
    aspect_yz = voxel_sz_z / voxel_sz_y
    N_ROWS = 2 * n + 1
    fig, axes = plt.subplots(ncols=3, nrows=N_ROWS, figsize=(15, 8))
    low_thresh -= low_delta * n
    high_thresh -= high_delta * n
    out = []

    for i in range(N_ROWS):
        thresholded = threshold(edge_filtered, low_thresh + i * low_delta,
                                high_thresh + i * high_delta)
        labels = label_thresholded(thresholded)
        axes[i, 0].set_ylabel(f'L:{low_thresh + i * low_delta:.2f},\n'
                              f'H:{high_thresh + i * high_delta:.2f}',
                              rotation=75)
        axes[i, 0].imshow(np.max(labels, axis=0), cmap=cmap)
        # axes[i, 0].yaxis.tick_right()
        axes[i, 1].imshow(np.max(labels, axis=1), cmap, aspect=aspect_xz)
        axes[i, 2].imshow(np.max(labels, axis=2), cmap, aspect=aspect_yz)
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
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(15, 8))

    axes[0].imshow(np.max(img, axis=0), cmap=cmap)
    axes[1].imshow(np.max(img, axis=1), cmap, aspect=aspect_xz)
    axes[2].imshow(np.max(img, axis=2), cmap, aspect=aspect_yz)
    plt.show()


def import_confocal_image(img_path, channel_interest=0):
    """Loads the 3D confocal image.

    - Tested on: CZI, LSM, TIFF

    Parameters
    ----------
    img_path : str
        Path to the confocal tissue image.
    channel_interest : int
        Channel of interest containing image data to be processed,
        by default 0

    """
    # image has to be converted to float for processing
    if img_path.split('.')[-1] == 'czi':
        img = czifile.imread(img_path)
        img = img.data
        img = img_as_float(np.squeeze(img))
        if img.ndim > 3:
            img = img[channel_interest]
    elif img_path.split('.')[-1] == 'lsm':
        img = tifffile.imread(img_path)
        img = img_as_float(np.squeeze(img))
        if img.ndim > 3:
            img = img[:, channel_interest]
    else:
        img = np.squeeze(io.imread(img_path))
        img = img_as_float(img)
        if img.ndim > 3:
            img = img[channel_interest]

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

def filter_labels(labels, thresholded, polygon, prune_3D_borders=True):
    filtered_labels = clear_border(labels)

    if prune_3D_borders:
        # Find convex hull that approximates tissue structure
        convex_hull = _compute_convex_hull(thresholded)
        filtered_labels = clear_border(filtered_labels,
                                       mask=binary_erosion(convex_hull))

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
        filtered_labels = clear_border(filtered_labels,
                                       mask=binary_erosion(roi_mask))

    return filtered_labels


def _filter_small_objects(regions):
    CUTOFF_VOLUME_3D = 27
    idx = 0
    for region in regions:
        if region.area > CUTOFF_VOLUME_3D:
            break
        idx += 1
    filtered = regions[idx:]
    return filtered


def arrange_regions(filtered_labels):
    regions = sorted(regionprops(filtered_labels),
                     key=lambda region: region.area)
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
    minz, miny, minx, maxz, maxy, maxx = region.bbox

    extracted_obj = tissue_img[minz:maxz, miny:maxy, minx:maxx].copy()
    extracted_obj[~region.filled_image] = 0.0

    print(f'Volume of this object is: {region.area}')
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
        minz, miny, minx, maxz, maxy, maxx = regions[obj].bbox
        ax.append(fig.add_subplot(rows, columns, idx+1))
        idx += 1
        ax[-1].set_title(f'Obj {obj}; Vol: {regions[obj].area}')
        ax[-1].axis('off')

        extracted_cell = tissue_img[minz:maxz, miny:maxy, minx:maxx].copy()
        extracted_cell[~regions[obj].filled_image] = 0.0

        plt.imshow(np.max(extracted_cell, 0), cmap='gray')


def export_cells(
    img_path,
    low_vol_cutoff,
    hi_vol_cutoff,
    output_option,
    tissue_img,
    regions,
    roi_name,
    roi_polygon,
    seg_type='segmented'
):
    OUT_TYPE = ('3D', 'MIP')[output_option]
    SEG_TYPES = ('segmented', 'unsegmented', 'both')

    DIR = getcwd() + '/Autocropped/'
    if not (path.exists(DIR) and path.isdir(DIR)):
        mkdir(DIR)

    IMAGE_NAME = '.'.join(path.basename(img_path).split('.')[:-1])
    OUT_DIR = DIR + IMAGE_NAME + \
              f'{"" if roi_name == "" else "-" + str(roi_name)}_{OUT_TYPE}/'
    if path.exists(OUT_DIR) and path.isdir(OUT_DIR):
        rmtree(OUT_DIR)
    mkdir(OUT_DIR)

    if seg_type == SEG_TYPES[0] or seg_type == SEG_TYPES[2]:
        mkdir(OUT_DIR + SEG_TYPES[0])

    if seg_type == SEG_TYPES[1] or seg_type == SEG_TYPES[2]:
        mkdir(OUT_DIR + SEG_TYPES[1])

    cell_metadata = {}

    if img_path.split('.')[-1] == 'tif':
        with tifffile.TiffFile(img_path) as file:
            metadata = file.imagej_metadata
            cell_metadata['unit'] = metadata['unit']
            cell_metadata['spacing'] = metadata['spacing']
    elif img_path.split('.')[-1] == 'czi':
        with czifile.CziFile(img_path) as file:
            metadata = file.metadata(False)['ImageDocument']['Metadata']
            cell_metadata['scaling'] = metadata['Scaling']

    if roi_polygon is not None:
        X, Y = _unwrap_polygon(roi_polygon)
        roi = (int(min(Y)), int(min(X)), int(max(Y) + 1), int(max(X) + 1))
        cell_metadata['roi_name'] = roi_name
        cell_metadata['roi'] = roi

    for (obj, region) in enumerate(regions):
        if low_vol_cutoff < region.area < hi_vol_cutoff:
            minz, miny, minx, maxz, maxy, maxx = region.bbox
            name = (str(uuid.uuid4()) + '.tif')

            # Cell-specific metadata
            cell_metadata['parent_image'] = path.abspath(img_path)
            cell_metadata['bounds'] = region.bbox
            cell_metadata['cell_volume'] = int(region.filled_area)
            cell_metadata['centroid'] = region.centroid
            cell_metadata['territorial_volume'] = int(region.convex_area)
            out_metadata = json.dumps(cell_metadata)

            if seg_type == SEG_TYPES[0] or seg_type == SEG_TYPES[2]:
                segmented = tissue_img[minz:maxz, miny:maxy, minx:maxx].copy()
                segmented = img_as_ubyte(segmented)
                segmented[~region.filled_image] = 0

                out = segmented if OUT_TYPE == '3D' else np.pad(
                    np.max(segmented, 0),
                    pad_width=max(segmented.shape[1:]) // 5, mode='constant')
                out_name = f'{OUT_DIR}{SEG_TYPES[0]}/' + name
                tifffile.imsave(out_name, out, description=out_metadata,
                                software='Autocrop')

            if seg_type == SEG_TYPES[1] or seg_type == SEG_TYPES[2]:
                scale_z = (maxz - minz) // 5
                scale_y = (maxy - miny) // 5
                scale_x = (maxx - minx) // 5
                minz = max(0, minz - scale_z)
                miny = max(0, miny - scale_y)
                minx = max(0, minx - scale_x)
                maxz += scale_z
                maxy += scale_y
                maxx += scale_x

                segmented = tissue_img[minz:maxz, miny:maxy, minx:maxx].copy()
                segmented = img_as_ubyte(segmented)
                out = segmented if OUT_TYPE == '3D' else np.max(segmented, 0)
                out_name = f'{OUT_DIR}{SEG_TYPES[1]}/' + name
                tifffile.imsave(out_name, out, description=out_metadata,
                                software='Autocrop')
