import json
from os import (
    getcwd,
    listdir,
    path,
)

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import napari
import PyQt5
import superqt

from matplotlib import gridspec
from magicgui import magicgui
from psutil import virtual_memory
from scipy.spatial import ConvexHull
from scipy.ndimage import (
    find_objects,
    distance_transform_edt,
)
from skimage.draw import polygon2mask
from skimage.exposure import (
    equalize_adapthist,
    match_histograms,
)
from skimage.feature import (
    peak_local_max,
)
from skimage.filters import (
    apply_hysteresis_threshold,
    gaussian,
    sobel,
)
from skimage.measure import label
from skimage.morphology import (
    binary_erosion,
    opening,
)
from skimage.restoration import (
    rolling_ball,
)
from skimage.segmentation import clear_border
from skimage.util import unique_rows
from scipy import ndimage as ndi
from vispy.geometry.rect import Rect

from ._io import (
    imread,
    export_cells,
)
from ._max_rect_in_poly import (
    get_maximal_rectangle,
)
from ._preprocess import (
    calibrate_nlm_denoiser,
    deconvolve,
    denoise,
)
from ._roi_extract import (
    select_ROI,
    mask_ROI
)
from .util import (
    imnorm,
    only_name,
    _unwrap_polygon,
)

from ._postprocessing import _segment_clump
from ...analysis._skeletal import _get_blobs


def _testThresholds(  # TODO: redundant
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

    zproj = imnorm(np.max(img, axis=0))
    ax.imshow(zproj, cmap=cmap)

    yproj = imnorm(np.max(img, axis=1))
    axb.imshow(yproj, cmap, aspect=aspect_xz)

    xproj = imnorm(np.max(img, axis=2))
    axr.imshow(xproj.T, cmap, aspect=1/aspect_yz)

    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(axr.get_yticklabels(), visible=False)
    plt.show()


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
            coords_in_hull[idx: idx + chunk_size] = np.all(
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


def approximate_somas(im, regions, src=None):
    somas_estimates = []
    if src is not None:
        somas_estimates = np.load(src, allow_pickle=True)[0]
        return somas_estimates
    for region in regions:
        minz, miny, minx, maxz, maxy, maxx = region['bbox']
        ll = np.array([minz, miny, minx])
        segmented_cell = im[minz:maxz, miny:maxy, minx:maxx] * region['image']
        try:
            # problematic cause multiple forks in prim
            distance = distance_transform_edt(region['image'])
            # distance = segmented_cell
            normed_distance = (distance - distance.min()) / \
                (distance.max() - distance.min())
            normed_distance = gaussian(normed_distance, sigma=1)
            blurred_opening = opening(normed_distance)
            blobs = _get_blobs(blurred_opening, 'confocal')
            coords = np.round_(
                blobs[blobs[:, 3].argsort()][:, :-1]).astype(int)

            areg = []
            opened_labels = label(opening(distance))
            for coord in coords:
                areg.append(opened_labels[tuple(coord)])
            visited = []
            filtered_coords = []
            for i in range(len(areg)):
                if areg[i] not in visited:
                    filtered_coords.append(coords[i])
                    visited.append(areg[i])
            filtered_coords = np.array(filtered_coords)
            mask = np.zeros_like(distance, dtype=bool)
            mask[tuple(filtered_coords.T)] = True
            coords = peak_local_max(distance, footprint=np.ones((3, 3, 3)))
            mask[tuple(coords.T)] *= True

            # sanity check if all empty pxls
            final_coords = np.array([c for c in np.transpose(
                np.nonzero(mask)) if segmented_cell[tuple(c.T)]])

            if len(final_coords) == 0:
                raise Exception('Failed automatic soma detection!')

            somas_estimates.extend(ll + final_coords)
        except:
            centroid = np.round_(region['centroid']).astype(np.int64)
            if region['image'][tuple(centroid - ll)]:
                somas_estimates.append(centroid)
            else:
                # Desparate measure: set the max value index
                somas_estimates.append(
                    ll + np.unravel_index(np.argmax(segmented_cell), region['image'].shape))
    return somas_estimates


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


class TissueImage:
    """
    Container object for single tissue image analysis.

    Parameters
    ----------
    cell_image : ndarray
        Grayscale image data of cell of nervous system.
    image_type : str
        Neuroimaging technique used to get image data of neuronal cell,
        either 'confocal' or 'DAB'.
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
    reference_image : ndarray
        `image` would be standardized to the exposure level of this example.
    sholl_step_size : int, optional
        Difference (in pixels) between concentric Sholl circles, by default 3
    polynomial_degree : int, optional
        Degree of polynomial for fitting regression model on sholl values, by
        default 3

    Attributes
    ----------
    image : ndarray
        Grayscale image data of cell of nervous system.
    image_type : str
        Neuroimaging technique used to get image data of neuronal cell,
        either 'confocal' or 'DAB'.
    cleaned_image : ndarray
        Thresholded, denoised, boolean transformation of `image` with solid
        soma.
    features : dict
        23 Morphometric features derived of the cell.
    skeleton : ndarray
        2D skeletonized, boolean transformation of `image`.
    convex_hull : ndarray
        2D transformation of `skeleton` representing a convex envelope
        that contains it.

    """
    __slots__ = ('im_path', 'imoriginal', 'imdeconvolved', 'impreprocessed',
                 'imdenoised', 'refdenoised', 'imsegmented', 'labels',
                 'regions', 'in_box', 'roi_polygon', 'low_volume_cutoff', 'region_inclusivity', 'REGION_INCLUSIVITY_LABELS',
                 'somas_estimates', 'DECONV_ITR', 'CLIP_LIMIT', 'SCALE', 'ROI_PATH', 'ROI_NAME', 'cache_dir',
                 'ALL_PT_ESTIMATES', 'FINAL_PT_ESTIMATES', 'residue', 'n_region', 'OBJ_INDEX', 'reconstructed_labels',
                 'LOW_VOLUME_CUTOFF', 'HIGH_VOLUME_CUTOFF', 'OUTPUT_OPTION', 'SEGMENT_TYPE')

    def __init__(
        self,
        im_path,
        roi_path=None,
        roi_name=None,
        deconv_itr=30,
        clip_limit=.02,
        ref_impath=None,
        ref_roipath=None,
        cache_dir='Cache',
    ):
        if not (
            (roi_path is None and roi_name is None)
            or (type(roi_path) == str and type(roi_name) == str)
        ):
            raise ValueError('Load ROI properly')
        self.im_path = im_path
        self.imoriginal = imoriginal = imread(im_path)
        self.ROI_PATH = roi_path
        self.ROI_NAME = roi_name
        self.cache_dir = cache_dir
        self.DECONV_ITR = deconv_itr
        self.CLIP_LIMIT = clip_limit
        self.labels = None
        self.regions = None
        self.residue = None
        self.in_box = None
        self.ALL_PT_ESTIMATES = None
        self.FINAL_PT_ESTIMATES = None

        cached_filename = only_name(im_path) + '.npy'
        cached = False

        try:
            if cached_filename in listdir(cache_dir):
                imdeconvolved = np.load(path.join(cache_dir, cached_filename))
                cached = True
            else:
                raise ValueError('Preprocessed image not cached')
        except:
            cached = False
            imdeconvolved = deconvolve(imoriginal, im_path, iters=deconv_itr)

        self.imdeconvolved = imdeconvolved

        SCALE = None
        try:
            if im_path.split('.')[-1] == 'czi':
                import czifile
                metadata = czifile.CziFile(im_path).metadata(False)[
                    'ImageDocument']['Metadata']
                dim_r = metadata['Scaling']['Items']['Distance'][0]['Value'] * 1e6
                dim_z = metadata['Scaling']['Items']['Distance'][-1]['Value'] * 1e6
                SCALE = (dim_z, dim_r, dim_r)
        except:
            pass

        self.SCALE = SCALE

        if cached:
            impreprocessed = imdeconvolved
        else:
            #TODO: hardcoded 50
            rb_radius = (min(imdeconvolved.shape) -
                         1)//2 if imoriginal.ndim == 3 else 50
            background = rolling_ball(imdeconvolved, radius=rb_radius)
            impreprocessed = imdeconvolved-background
            impreprocessed = equalize_adapthist(impreprocessed,
                                                clip_limit=clip_limit)

        self.impreprocessed = impreprocessed

        select_roi = roi_path is not None and roi_name is not None

        roi_polygon = (np.array([[0, 0], [impreprocessed.shape[0]-1, 0],
                                 [impreprocessed.shape[0]-1, impreprocessed.shape[1]-1],
                                 [0, impreprocessed.shape[1]-1]])
                       if not select_roi else
                       select_ROI(impreprocessed, f'{only_name(im_path)}-{roi_name}', roi_path))
        self.roi_polygon = roi_polygon

        if select_roi:
            imoriginal = mask_ROI(imoriginal, roi_polygon)
            imdeconvolved = mask_ROI(imdeconvolved, roi_polygon)
            impreprocessed = mask_ROI(impreprocessed, roi_polygon)

        self.imoriginal = imoriginal
        self.imdeconvolved = imdeconvolved
        self.impreprocessed = impreprocessed

        # Load reference
        if ref_impath is not None:
            imname = only_name(ref_impath)
            REF_SUFFIX = f'-{roi_name}-ref.npy'
            try:
                cached_filename = imname + REF_SUFFIX
                refdenoised = np.load(path.join(cache_dir, cached_filename))
            except:
                cached_filename = imname + '.npy'
                cached = False

                try:
                    if cached_filename in listdir(cache_dir):
                        refdeconvolved = np.load(path.join(cache_dir, cached_filename))
                        cached = True
                    else:
                        raise ValueError('Preprocessed image not cached')
                except:
                    cached = False
                    refdeconvolved = deconvolve(imread(ref_impath), ref_impath, iters=deconv_itr)

                if cached:
                    refpreprocessed = refdeconvolved
                else:
                    background = rolling_ball(refdeconvolved, radius=(min(refdeconvolved.shape)-1)//2)
                    refpreprocessed = equalize_adapthist(refdeconvolved-background, clip_limit=clip_limit)


                SELECT_ROI = ref_roipath is not None

                roi_polygon = (np.array([[0, 0], [refpreprocessed.shape[0]-1, 0],
                                         [refpreprocessed.shape[0]-1, refpreprocessed.shape[1]-1],
                                         [0, refpreprocessed.shape[1]-1]])
                    if not SELECT_ROI else select_ROI(refpreprocessed, f'{imname}-{roi_name}', ref_roipath))

                refpreprocessed = mask_ROI(refpreprocessed, roi_polygon)
                ll, ur = get_maximal_rectangle([roi_polygon])
                if SELECT_ROI:
                    ll, ur = np.ceil(ll).astype(int), np.floor(ur).astype(int)
                    llx, lly = ll; urx, ury = ur
                    llx -= roi_polygon[:, 0].min(); urx -= roi_polygon[:, 0].min()
                    lly -= roi_polygon[:, 1].min(); ury -= roi_polygon[:, 1].min()
                else:
                    lly = 0; llx = 0
                    ury, urx = refpreprocessed.shape[1:]
                    ury -= 1; urx -= 1

                denoise_parameters = calibrate_nlm_denoiser(refpreprocessed[:, lly:ury, llx:urx])
                refdenoised = denoise(refpreprocessed, denoise_parameters)
                np.save(path.join(cache_dir, imname + REF_SUFFIX), refdenoised)

            self.refdenoised = refdenoised

        # get the maximally inscribed rectangle
        ll, ur = get_maximal_rectangle([roi_polygon])

        if select_roi:
            ll, ur = np.ceil(ll).astype(int), np.floor(ur).astype(int)
            llx, lly = ll; urx, ury = ur
            llx -= roi_polygon[:, 0].min(); urx -= roi_polygon[:, 0].min()
            lly -= roi_polygon[:, 1].min(); ury -= roi_polygon[:, 1].min()
        else:
            lly = 0; llx = 0
            ury, urx = impreprocessed.shape[1:]
            ury -= 1; urx -= 1
        
        self.in_box = (lly, llx, ury, urx)

        denoise_parameters = calibrate_nlm_denoiser(impreprocessed[:, lly:ury, llx:urx])
        imdenoised = denoise(impreprocessed, denoise_parameters)

        imdenoised = match_histograms(imdenoised, refdenoised)
        self.imdenoised = imdenoised

    def segment(
        self,
        low_thresh,
        high_thresh,
        particle_filter_size=64
    ):
        """."""
        thresholded = threshold(self.imdenoised, low_thresh, high_thresh)
        labels = label_thresholded(thresholded)

        prefiltering_volume = thresholded.sum()
        print(f'Prefiltering Volume: {prefiltering_volume}')
        regions = arrange_regions(labels)

        # reconstruct despeckled filtered_labels
        filtered_labels = np.zeros_like(labels, dtype=int)

        reg_itr = 1

        for region in regions:
            minz, miny, minx, maxz, maxy, maxx = region['bbox']
            filtered_labels[minz:maxz, miny:maxy, minx:maxx] += region['image'] * reg_itr
            reg_itr += 1

        self.imsegmented = self.imdenoised * (filtered_labels > 0)
        self.labels = filtered_labels
        self.regions = regions

    def volume_cutoff(
        self,
        low_volume_cutoff=64
    ):
        self.low_volume_cutoff = low_volume_cutoff
        SCALE = self.SCALE
        viewer = napari.Viewer(ndisplay=3)
        viewer.add_image(self.imdenoised, scale=SCALE)
        viewer.add_labels(self.labels, rendering='translucent', opacity=.5, scale=SCALE)

        region_props = PyQt5.QtWidgets.QLabel()
        self.region_inclusivity = np.ones(len(self.regions), dtype=bool)
        self.REGION_INCLUSIVITY_LABELS = ['Include Region', 'Exclude Region']

        n_region = 0
        PROPS = ['vol',
        # 'convex_area',
        # 'equivalent_diameter',
        # 'euler_number',
        # 'extent',
        # 'feret_diameter_max',
        # 'major_axis_length',
        # 'minor_axis_length',
        # 'solidity'
        ]

        @magicgui(
            call_button='Exclude regions by volume',
            cutoff={'widget_type': 'Slider', 'max': self.regions[-1]['vol']}
        )
        def vol_cutoff_update(
            cutoff=low_volume_cutoff
        ):
            layer_names = [layer.name for layer in viewer.layers]
            self.low_volume_cutoff = cutoff
            self.labels.fill(0)
            itr = 1
            filtered_regions = []
            for region in self.regions:
                if cutoff <= region['vol']:
                    minz, miny, minx, maxz, maxy, maxx = region['bbox']
                    self.labels[minz:maxz, miny:maxy, minx:maxx] += region['image'] * itr
                    itr += 1
                    filtered_regions.append(region)
            viewer.layers[layer_names.index('Labels')].data = self.labels
            self.regions = filtered_regions
            self.imsegmented = self.imdenoised * (self.labels > 0)
            select_region()

        @magicgui(
            auto_call=True,
            selected_region={'maximum': len(self.regions)-1},
            select_region=dict(widget_type='PushButton', text='Select Region')
        )
        def select_region(
            selected_region=n_region,
            select_region=True  # just for activating the method
        ):
            global n_region
            n_region = selected_region
            minz, miny, minx, maxz, maxy, maxx = self.regions[n_region]['bbox']
            centroid = self.regions[n_region]['centroid']
            if SCALE is not None:
                centroid = centroid * np.array(SCALE)
                minz *= SCALE[0]; maxz *= SCALE[0]
                miny *= SCALE[1]; maxy *= SCALE[1]
                minx *= SCALE[2]; maxx *= SCALE[2]

            if viewer.dims.ndisplay == 3:
                viewer.camera.center = centroid
            elif viewer.dims.ndisplay == 2:
                viewer.dims.set_current_step(0, round(centroid[0]))
                viewer.window.qt_viewer.view.camera.set_state({'rect': Rect(minx, miny, maxx-minx, maxy-miny)})

            data = '<table cellspacing="8">'
            for prop in PROPS:
                name = prop
                data += '<tr><td><b>' + name + '</b></td><td>' + str(eval(f'self.regions[{n_region}]["{prop}"]')) + '</td></tr>'
            data += '</table>'
            region_props.setText(data)

        viewer.window.add_dock_widget(vol_cutoff_update, name='Volume Cutoff', area='bottom')
        viewer.window._dock_widgets['Volume Cutoff'].setFixedHeight(90)
        viewer.window.add_dock_widget(select_region, name='Select Region')
        viewer.window._dock_widgets['Select Region'].setFixedHeight(100)
        viewer.window.add_dock_widget(region_props, name='Region Properties')
        viewer.window._dock_widgets['Select Region'].setFixedHeight(270)
        select_region()

    def approximate_somas(
        self,
        src=None
    ):
        """."""
        somas_estimates = approximate_somas(self.imsegmented, self.regions, src=src)
        self.somas_estimates = somas_estimates

    def separate_clumps(
        self,
        seed_src=None
    ):
        SCALE = self.SCALE
        region_inclusivity = self.region_inclusivity
        somas_estimates = self.somas_estimates

        self.regions = sorted(self.regions, key=lambda region: region['vol'])
        regions = self.regions

        reconstructed_labels = np.zeros(self.imdenoised.shape, dtype=int)
        for itr in range(len(regions)):
            minz, miny, minx, maxz, maxy, maxx = regions[itr]['bbox']
            reconstructed_labels[minz:maxz, miny:maxy, minx:maxx] += regions[itr]['image'] * (itr + 1)
        self.reconstructed_labels = reconstructed_labels


        region_props = PyQt5.QtWidgets.QLabel()
        region_inclusivity = np.ones(len(regions), dtype=bool)
        REGION_INCLUSIVITY_LABELS = self.REGION_INCLUSIVITY_LABELS
        filtered_regions = None
        self.n_region = 0

        PROPS = ['vol',
        # 'convex_area',
        # 'equivalent_diameter',
        # 'euler_number',
        # 'extent',
        # 'feret_diameter_max',
        # 'major_axis_length',
        # 'minor_axis_length',
        # 'solidity'
        ]

        @magicgui(
            auto_call=True,
            selected_region={'maximum': len(regions)-1},
            select_region=dict(widget_type='PushButton', text='Select Region')
        )
        def select_region(
            selected_region=self.n_region,
            select_region=True  # pseudo: just for activating the method
        ):
            self.n_region = selected_region
            minz, miny, minx, maxz, maxy, maxx = regions[self.n_region]['bbox']
            centroid = regions[self.n_region]['centroid']
            if SCALE is not None:
                centroid = centroid * np.array(SCALE)
                minz *= SCALE[0]; maxz *= SCALE[0]
                miny *= SCALE[1]; maxy *= SCALE[1]
                minx *= SCALE[2]; maxx *= SCALE[2]
            if viewer.dims.ndisplay == 3:
                viewer.camera.center = centroid
            elif viewer.dims.ndisplay == 2:
                viewer.dims.set_current_step(0, round(centroid[0]))
                viewer.window.qt_viewer.view.camera.set_state({'rect': Rect(minx, miny, maxx-minx, maxy-miny)})

            data = '<table cellspacing="8">'
            data += '<tr><td><b>included</b></td><td>' + str(region_inclusivity[self.n_region]) + '</td></tr>'
            for prop in PROPS:
                name = prop
                data += '<tr><td><b>' + name + '</b></td><td>' + str(eval(f'regions[{self.n_region}]["{prop}"]')) + '</td></tr>'
            data += '</table>'
            region_props.setText(data)
            viewer.window._dock_widgets['Region Inclusivity'].children(
            )[4].children()[1].setText(REGION_INCLUSIVITY_LABELS[region_inclusivity[self.n_region]])


        @magicgui(
            call_button=REGION_INCLUSIVITY_LABELS[region_inclusivity[self.n_region]]
        )
        def exclude_region():
            region_inclusivity[self.n_region] = ~region_inclusivity[self.n_region]
            data = '<table cellspacing="8">'
            data += '<tr><td><b>included</b></td><td>' + str(region_inclusivity[self.n_region]) + '</td></tr>'
            for prop in PROPS:
                name = prop
                data += '<tr><td><b>' + name + '</b></td><td>' + str(eval(f'regions[{self.n_region}]["{prop}"]')) + '</td></tr>'
            data += '</table>'
            region_props.setText(data)
            viewer.window._dock_widgets['Region Inclusivity'].children(
            )[4].children()[1].setText(REGION_INCLUSIVITY_LABELS[region_inclusivity[self.n_region]])


        @magicgui(
            call_button="Use soma_coord approximation"
        )
        def interactive_segment1():
            somas_estimates = self.somas_estimates
            layer_names = [layer.name for layer in viewer.layers]
            minz, miny, minx, maxz, maxy, maxx = regions[self.n_region]['bbox']
            ll = np.array([minz, miny, minx])
            ur = np.array([maxz, maxy, maxx]) - 1  # upper-right
            inidx = np.all(np.logical_and(ll <= somas_estimates, somas_estimates <= ur), axis=1)
            somas_coords = np.array(somas_estimates)[inidx]
            somas_coords -= ll
            somas_coords = np.array([x for x in somas_coords if regions[self.n_region]['image'][tuple(x.astype(np.int64))] > 0])
            labels = label(regions[self.n_region]['image'])
            areg = []
            for coord in somas_coords:
                areg.append(labels[tuple(coord.astype(np.int64))])
            visited = []
            filtered_coords = []
            for i in range(len(areg)):
                if areg[i] not in visited:
                    filtered_coords.append(somas_coords[i])
                    visited.append(areg[i])
            filtered_coords = np.array(filtered_coords)

            if 'filtered_coords' in layer_names:
                data = viewer.layers[layer_names.index('filtered_coords')].data
                for coord in filtered_coords:
                    if ~(data == coord).all(axis=1).any():
                        viewer.layers[layer_names.index('filtered_coords')].add(np.round_(coord + ll).astype(np.int64))
            else:
                viewer.add_points(filtered_coords + ll, face_color='red', edge_width=0,
                                opacity=.6, size=5, name='filtered_coords', scale=SCALE)


        @magicgui(
            call_button="Watershed region coords",
            hyst_thresh={'widget_type': 'FloatSlider', 'max': 1}
        )
        def interactive_segment2(
            hyst_thresh=0  # add button for use as is
        ):
            somas_estimates = self.somas_estimates
            layer_names = [layer.name for layer in viewer.layers]
            minz, miny, minx, maxz, maxy, maxx = regions[self.n_region]['bbox']
            ll = np.array([minz, miny, minx])
            ur = np.array([maxz, maxy, maxx]) - 1  # upper-right
            inidx = np.all(np.logical_and(ll <= somas_estimates, somas_estimates <= ur), axis=1)
            somas_coords = np.array(somas_estimates)[inidx]
            somas_coords -= ll
            somas_coords = np.array([x for x in somas_coords if regions[self.n_region]['image'][tuple(x.astype(np.int64))] > 0])
            clump = self.imsegmented[minz:maxz, miny:maxy, minx:maxx] * regions[self.n_region]['image']
            # hyst_thresh_ptile = np.percentile(clump[clump > 0].ravel(), hyst_thresh * 100)
            threshed = clump > hyst_thresh
            labels = label(threshed)
            areg = []
            for coord in somas_coords:
                areg.append(labels[tuple(coord.astype(np.int64))])
            visited = []
            filtered_coords = []
            for i in range(len(areg)):
                if areg[i] not in visited:
                    filtered_coords.append(somas_coords[i])
                    visited.append(areg[i])
            filtered_coords = np.array(filtered_coords)

            if 'filtered_coords' in layer_names:
                data = viewer.layers[layer_names.index('filtered_coords')].data
                for coord in filtered_coords:
                    coord_to_add = np.round_(coord + ll).astype(np.int64)
                    #TODO: remove extra pts: Clump > 0 ko invert krke mask mai multiply kro aur fir wapis coord mai convert kro
                    if ~(data == coord_to_add).all(axis=1).any():
                        viewer.layers[layer_names.index('filtered_coords')].add(coord_to_add)
            else:
                viewer.add_points(filtered_coords + ll, face_color='red', edge_width=0,
                                opacity=.6, size=5, name='filtered_coords', scale=SCALE)


        @magicgui(
            call_button='Watershed regions'
        )
        def clump_separation_napari():
            SEARCH_LAYER = 'filtered_coords'
            layer_names = [layer.name for layer in viewer.layers]
            somas_estimates = np.unique(viewer.layers[layer_names.index(SEARCH_LAYER)].data, axis=0)
            filtered_regions, residue = [], []
            separated_clumps = []

            print('Please recheck if you REALLY want these changes.')
            itr = 0
            for region in self.regions:
                minz, miny, minx, maxz, maxy, maxx = region['bbox']
                ll = np.array([minz, miny, minx])  # lower-left
                ur = np.array([maxz, maxy, maxx]) - 1  # upper-right
                inidx = np.all(np.logical_and(ll <= somas_estimates, somas_estimates <= ur), axis=1)
                somas_coords = somas_estimates[inidx].astype(np.int64)

                if len(somas_coords) == 0:
                    print('Delete region:', itr)
                    residue.append(region)
                elif len(np.unique(somas_coords.astype(int), axis=0)) > 1:  # clumpSep
                    somas_coords = somas_coords.astype(int)
                    somas_coords -= ll
                    im = self.imdenoised[minz:maxz, miny:maxy, minx:maxx].copy()
                    im[~region['image']] = 0
                    markers = np.zeros(region['image'].shape)

                    somas_coords = np.array([x for x in somas_coords if region['image'][tuple(x)] > 0])

                    if len(somas_coords) == 0:  # no marked point ROI
                        print('Delete region:', itr)

                    if somas_coords.shape[0] == 1:
                        filtered_regions.append(region)
                        continue

                    for i in range(somas_coords.shape[0]):
                        markers[tuple(somas_coords[i])] = i + 1
                        separated_clumps.append(somas_coords[i])

                    labels = _segment_clump(im, markers)
                    separated_regions = arrange_regions(labels)
                    for r in separated_regions:
                        r['centroid'] = (minz + r['centroid'][0], miny + r['centroid'][1], minx + r['centroid'][2])
                        r['bbox'] = (minz + r['bbox'][0], miny + r['bbox'][1], minx + r['bbox'][2], minz + r['bbox'][3], miny + r['bbox'][4], minx + r['bbox'][5])
                        # r.slice = (slice(minz + r.bbox[0], minz + r.bbox[3]),
                        #            slice(miny + r.bbox[1], miny + r.bbox[4]),
                        #         slice(minx + r.bbox[2], minx + r.bbox[5]))
                    print('Split clump region:', itr)
                    filtered_regions.extend(separated_regions)
                else:
                    filtered_regions.append(region)
                itr += 1

            self.regions = filtered_regions
            self.residue = residue
            watershed_results = np.zeros(self.imdenoised.shape, dtype=int)
            for itr in range(len(filtered_regions)):
                minz, miny, minx, maxz, maxy, maxx = filtered_regions[itr]['bbox']
                watershed_results[minz:maxz, miny:maxy, minx:maxx] += filtered_regions[itr]['image'] * (itr + 1)
            viewer.add_labels(watershed_results, rendering='translucent', opacity=.5, scale=SCALE)


        @magicgui(
            call_button="Confirm and apply changes"
        )
        def save_watershed():
            layer_names = [layer.name for layer in viewer.layers]
            viewer.layers.remove('watershed_results')
            self.regions = sorted(self.regions, key=lambda region: region['vol'])
            regions = self.regions
            watershed_results = np.zeros(self.imdenoised.shape, dtype=int)
            for itr in range(len(regions)):
                minz, miny, minx, maxz, maxy, maxx = regions[itr]['bbox']
                watershed_results[minz:maxz, miny:maxy, minx:maxx] += regions[itr]['image'] * (itr + 1)

            if 'reconstructed_labels' in layer_names:
                viewer.layers[layer_names.index('reconstructed_labels')].data = watershed_results

            # After all changes (for reproducibility)
            final_soma = np.unique(viewer.layers[layer_names.index('filtered_coords')].data, axis=0)

            # discarded clump ROI: to be subtracted: np.setdiff1d(somas_estimates, final_soma)
            ALL_PT_ESTIMATES, FINAL_PT_ESTIMATES = self.somas_estimates.copy(), final_soma
            self.ALL_PT_ESTIMATES = ALL_PT_ESTIMATES
            self.FINAL_PT_ESTIMATES = FINAL_PT_ESTIMATES
            self.reconstructed_labels = watershed_results
            self.imsegmented = self.imdenoised * (watershed_results > 0)


        viewer = napari.Viewer(ndisplay=3)
        viewer.add_image(self.imdenoised, scale=SCALE, name='denoised')
        viewer.add_image(self.imsegmented, scale=SCALE, name='segmented')
        viewer.add_labels(self.reconstructed_labels, rendering='translucent', opacity=.5,
                          scale=SCALE, name='reconstructed_labels')
        viewer.add_points(somas_estimates, face_color='orange', edge_width=0,
                        opacity=.6, size=5, name='somas_coords', scale=SCALE)
        # viewer.add_points(FINAL_PT_ESTIMATES, face_color='red',
        #                   edge_width=0, opacity=.6, size=5, name='filtered_coords', scale=SCALE)
        if seed_src is not None:
            final_pts = np.load(seed_src, allow_pickle=True)[1].astype(int)
            viewer.add_points(final_pts, face_color='red', edge_width=0,
                              opacity=.6, size=5, name='filtered_coords',
                              scale=SCALE)
        elif self.FINAL_PT_ESTIMATES is not None:
            viewer.add_points(self.FINAL_PT_ESTIMATES, face_color='red', edge_width=0,
                              opacity=.6, size=5, name='filtered_coords',
                              scale=SCALE)

        viewer.window.add_dock_widget(select_region, name='Select Region')
        viewer.window._dock_widgets['Select Region'].setFixedHeight(100)
        viewer.window.add_dock_widget(region_props, name='Region Properties')
        viewer.window._dock_widgets['Region Properties'].setFixedHeight(270)
        viewer.window.add_dock_widget(exclude_region, name='Region Inclusivity')
        viewer.window._dock_widgets['Region Inclusivity'].setFixedHeight(70)
        viewer.window.add_dock_widget(interactive_segment1, name='Interactive Segmentation 1')
        viewer.window._dock_widgets['Interactive Segmentation 1'].setFixedHeight(70)
        viewer.window.add_dock_widget(interactive_segment2, name='Interactive Segmentation 2')
        viewer.window._dock_widgets['Interactive Segmentation 2'].setFixedWidth(350)
        viewer.window._dock_widgets['Interactive Segmentation 2'].setFixedHeight(100)
        viewer.window.add_dock_widget(clump_separation_napari, name='Clump Separation')
        viewer.window._dock_widgets['Clump Separation'].setFixedHeight(70)
        viewer.window.add_dock_widget(save_watershed, name='Confirm Changes')
        viewer.window._dock_widgets['Confirm Changes'].setFixedHeight(70)

        select_region()

    def show_segmented(self, view='single', grid_size=None):
        """View segmentation results.

        Parameters
        ----------
        view : str
            How to display segmented cells, either 'grid' or 'single'.
        grid_size : int or None
            Size of the grid view.
        """
        denoised = self.imdenoised
        regions = self.regions
        segmented = self.imsegmented

        def plot_batch(BATCH_NO):
            project_batch(BATCH_NO, N_BATCHES, regions, denoised)
            plt.show()

        self.OBJ_INDEX = 0
        extracted_cell = None
        # minz, miny, minx, maxz, maxy, maxx = 0, 0, 0, 0, 0, 0

        def plot_single(obj_index):
            self.OBJ_INDEX = obj_index
            extracted_cell = extract_obj(regions[obj_index], segmented)
            # minz, miny, minx, maxz, maxy, maxx = regions[obj_index]['bbox']
            projectXYZ(extracted_cell, .5, .5, 1)

        if view == 'grid':
            # Set `BATCH_NO` to view detected objects in paginated 2D MIP views.
            N_BATCHES = paginate_objs(regions, grid_size)
            widgets.interact(plot_batch, BATCH_NO=widgets.IntSlider(min=0,
                max=N_BATCHES-1, layout=widgets.Layout(width='100%')))
        else:
            widgets.interact(plot_single,
                obj_index=widgets.IntSlider(min=0, max=len(regions)-1,
                layout=widgets.Layout(width='100%')))
    
    def refine_soma_approx(self):
        self.n_region = 0
        denoised = self.imdenoised
        regions = self.regions
        SCALE = self.SCALE

        viewer = napari.Viewer(ndisplay=3)

        @magicgui(
            auto_call=True,
            selected_region={'maximum': len(regions)-1}
        )
        def view_single_region(selected_region=0):
            self.n_region = selected_region
            somas_estimates = self.FINAL_PT_ESTIMATES
            minz, miny, minx, maxz, maxy, maxx = regions[selected_region]['bbox']
            extracted_cell = denoised[minz:maxz, miny:maxy, minx:maxx]*regions[selected_region]['image']
            ll = np.array([minz, miny, minx])
            ur = np.array([maxz, maxy, maxx]) - 1  # upper-right
            inidx = np.all(np.logical_and(ll <= somas_estimates, somas_estimates <= ur), axis=1)
            somas_coords = np.array(somas_estimates)[inidx]
            somas_coords -= ll
            somas_coords = np.array([x for x in somas_coords if regions[self.n_region]['image'][tuple(x.astype(np.int64))] > 0])
            labels = label(regions[self.n_region]['image'])
            areg = []
            for coord in somas_coords:
                areg.append(labels[tuple(coord.astype(np.int64))])
            visited = []
            filtered_coords = []
            for i in range(len(areg)):
                if areg[i] not in visited:
                    filtered_coords.append(somas_coords[i])
                    visited.append(areg[i])
            filtered_coords = np.array(filtered_coords)

            if len(viewer.layers) == 0:
                viewer.add_image(denoised[minz:maxz, miny:maxy, minx:maxx], name='denoised', colormap='red', scale=SCALE)
                viewer.add_image(denoised[minz:maxz, miny:maxy, minx:maxx]*regions[selected_region]['image'], name='segmented', colormap='red', scale=SCALE)
                viewer.add_points(filtered_coords, face_color='lime', edge_width=0,
                                  opacity=.6, size=5, name='filtered_coords', scale=SCALE)
            else:
                viewer.layers[0].data = denoised[minz:maxz, miny:maxy, minx:maxx]
                viewer.layers[1].data = extracted_cell
                viewer.layers[2].data = filtered_coords
            viewer.camera.center = ((maxz-minz)//2, (maxy-miny)//2, (maxx-minx)//2)

        @magicgui(
            call_button="Update soma estimates"
        )
        def update_soma_estimates():
            FINAL_PT_ESTIMATES = self.FINAL_PT_ESTIMATES
            layer_names = [layer.name for layer in viewer.layers]
            region = regions[self.n_region]
            minz, miny, minx, maxz, maxy, maxx = region['bbox']
            ll = np.array([minz, miny, minx])
            somas_coords = viewer.layers[layer_names.index('filtered_coords')].data.astype(int)
            im = denoised[minz:maxz, miny:maxy, minx:maxx].copy()
            im[~region['image']] = 0
            markers = np.zeros(region['image'].shape)

            for i in range(somas_coords.shape[0]):
                markers[tuple(somas_coords[i])] = i + 1

            labels = _segment_clump(im, markers)
            viewer.add_labels(labels, rendering='translucent', opacity=.5, scale=SCALE)
            FINAL_PT_ESTIMATES = np.vstack((FINAL_PT_ESTIMATES, somas_coords+ll))
            FINAL_PT_ESTIMATES = np.unique(FINAL_PT_ESTIMATES, axis=0)
            self.FINAL_PT_ESTIMATES = FINAL_PT_ESTIMATES

        viewer.window.add_dock_widget(view_single_region, name='View individual region')
        viewer.window._dock_widgets['View individual region'].setFixedHeight(70)
        viewer.window.add_dock_widget(update_soma_estimates, name='Update soma estimates')
        view_single_region()

    def export_cropped(
        self
    ):
        self.LOW_VOLUME_CUTOFF = self.regions[0]['vol']  # filter noise/artifacts
        self.HIGH_VOLUME_CUTOFF = self.regions[-1]['vol']  # filter cell clusters
        self.OUTPUT_OPTION = 'both'  # '3d' for 3D cells, 'mip' for Max Intensity Projections
        self.SEGMENT_TYPE = 'both'
        SCALE = self.SCALE
        reconstructed_cells = None

        viewer = napari.Viewer(ndisplay=3)
        reconstructed_cells = np.zeros_like(self.imdenoised)
        viewer.add_image(reconstructed_cells, colormap='inferno', scale=SCALE)
        minz, miny, minx, maxz, maxy, maxx = self.regions[0]['bbox']

        self.n_region = 0

        vol_cutoff_slider = superqt.QLabeledRangeSlider()
        vol_cutoff_slider.setRange(0, self.regions[-1]['vol'])
        vol_cutoff_slider.setOrientation(1)
        vol_cutoff_slider.setValue([self.LOW_VOLUME_CUTOFF, self.HIGH_VOLUME_CUTOFF])
        vol_cutoff_slider.setEdgeLabelMode(superqt.sliders._labeled.EdgeLabelMode.NoLabel)
        vol_cutoff_slider.setContentsMargins(25, 5, 25, 5)
        for i in (0, 1):
            # vol_cutoff_slider.children()[i].setAlignment(PyQt5.QtCore.Qt.AlignCenter)
            vol_cutoff_slider.children()[i].setFixedWidth(len(str(int(self.regions[-1]['vol']))) * 20)

        def vol_cutoff_update():
            reconstructed_cells.fill(0)
            self.LOW_VOLUME_CUTOFF, self.HIGH_VOLUME_CUTOFF = vol_cutoff_slider.value()
            for region in self.regions:
                if self.LOW_VOLUME_CUTOFF <= region['vol'] <= self.HIGH_VOLUME_CUTOFF:
                    minz, miny, minx, maxz, maxy, maxx = region['bbox']
                    segmented_cell = region['image'] * self.imdenoised[minz:maxz, miny:maxy, minx:maxx]
                    segmented_cell = segmented_cell / (segmented_cell.max() - segmented_cell.min())
                    reconstructed_cells[minz:maxz, miny:maxy, minx:maxx] += segmented_cell
            minz, miny, minx, maxz, maxy, maxx = self.regions[n_region]['bbox']
            viewer.layers[0].data = reconstructed_cells

        vol_cutoff_slider.valueChanged.connect(vol_cutoff_update)


        @magicgui(
            call_button='Export Cells',
            output_option=dict(choices=['both', '3d', 'mip']),
            segment_type=dict(choices=['both', 'segmented', 'unsegmented'])
        )
        def export(
            output_option,
            segment_type
        ):
            self.OUTPUT_OPTION, self.SEGMENT_TYPE = output_option, segment_type
            self.imsegmented = reconstructed_cells
            export_cells(self.im_path, self.LOW_VOLUME_CUTOFF,
                         self.HIGH_VOLUME_CUTOFF, output_option, self.imdenoised,
                         self.regions, None, segment_type, self.ROI_NAME, self.roi_polygon)

            params = {
                    'ref_impath': self.ref_impath,
                    'CLIP_LIMIT': self.CLIP_LIMIT,
                    'LOW_THRESH': self.LOW_THRESH,
                    'HIGH_THRESH': self.HIGH_THRESH,
                    'NAME_ROI': self.ROI_NAME,
                    'PRE_LOW_VOLUME_CUTOFF': self.PRE_LOW_VOLUME_CUTOFF,
                    'LOW_VOLUME_CUTOFF': self.LOW_VOLUME_CUTOFF,  # filter noise/artifacts
                    'HIGH_VOLUME_CUTOFF': self.HIGH_VOLUME_CUTOFF,  # filter cell clusters
                    'OUTPUT_TYPE': self.SEGMENT_TYPE
            }

            DIR = getcwd() + '/Autocropped/'
            OUT_DIR = DIR + only_name(self.im_path) + \
                    f'{"" if self.ROI_NAME == "" else "-" + str(self.ROI_NAME)}/'

            with open(OUT_DIR + '.params.json', 'w') as out:
                json.dump(params, out)

            out = np.array([self.ALL_PT_ESTIMATES, self.FINAL_PT_ESTIMATES])
            np.save(OUT_DIR + '.somas_estimates.npy', out)

        viewer.window.add_dock_widget(vol_cutoff_slider, name='Volume Cutoff', area='bottom')

        viewer.window._dock_widgets['Volume Cutoff'].setFixedHeight(80)
        viewer.window.add_dock_widget(export, name='Export Cells')
        vol_cutoff_update()
