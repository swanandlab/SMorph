import json
from os import (
    getcwd,
    listdir,
    path,
)
import json
import pathlib2
import uuid
from shutil import rmtree

import dask.array as da
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import napari
import PyQt5
import superqt
import zarr
import roifile
import tifffile

from matplotlib import gridspec
from matplotlib.path import Path
import matplotlib.patches as patches

from magicgui import magicgui
from psutil import virtual_memory
from scipy.spatial import ConvexHull
from scipy import (
    ndimage as ndi,
    sparse,
)
from skan.csr import (
    Skeleton,
    summarize,
)
from skimage import img_as_float, img_as_ubyte, exposure
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
    threshold_otsu,
    threshold_isodata,
    threshold_li,
    threshold_mean,
    threshold_minimum,
    threshold_triangle,
    threshold_yen,
)
from skimage.graph import central_pixel
from skimage.measure import label
from skimage.morphology import (
    binary_erosion,
    opening,
    skeletonize,
)
from skimage.restoration import (
    rolling_ball,
)
from skimage.segmentation import (
    clear_border,
)
from skimage.util._map_array import ArrayMap
from skimage.util import unique_rows
from vispy.geometry.rect import Rect

from ._io import (
    _build_multipoint_roi,
    _mkdir_if_not,
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


def projectXYZ(img, voxel_sz_z, voxel_sz_y, voxel_sz_x, cmap='gray'):
    """Projects a 3D image in all planes.

    Parameters
    ----------
    img : ndarray
        Image data.
    voxel_sz_z : int
        Spacing of voxel in Z axis.
    voxel_sz_y : int
        Spacing of voxel in Y axis.
    voxel_sz_x : int
        Spacing of voxel in X axis.
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


def _simplify_graph(skel):
    """Iterative removal of all nodes of degree 2 while reconnecting their
    edges.

    Parameters
    ----------
    skel : skan.csr.Skeleton
        A Skeleton object containing graph to be simplified.

    Returns
    -------
    simp_csgraph : scipy.sparse.csr_matrix
        A sparse adjacency matrix of the simplified graph.
    reduced_nodes : tuple of int
        The index nodes of original graph in simplified graph.
    """
    if np.sum(skel.degrees > 2) == 0:  # no junctions
        # don't reduce
        return skel.graph, np.arange(skel.graph.shape[0])

    summary = summarize(skel, separator='-')
    src = np.asarray(summary['node-id-src'])
    dst = np.asarray(summary['node-id-dst'])
    distance = np.asarray(summary['branch-distance'])

    # to reduce the size of simplified graph
    nodes = np.unique(np.append(src, dst))
    n_nodes = len(nodes)
    nodes_sequential = np.arange(n_nodes)

    fw_map = ArrayMap(nodes, nodes_sequential)
    inv_map = ArrayMap(nodes_sequential, nodes)

    src_relab, dst_relab = fw_map[src], fw_map[dst]

    edges = sparse.coo_matrix(
            (distance, (src_relab, dst_relab)),
            shape=(n_nodes, n_nodes)
            )
    dir_csgraph = edges.tocsr()
    simp_csgraph = dir_csgraph + dir_csgraph.T  # make undirected

    reduced_nodes = inv_map[np.arange(simp_csgraph.shape[0])]

    return simp_csgraph, reduced_nodes


def _fast_graph_center_idx(skel):
    """Accelerated graph center finding using simplified graph.

    Parameters
    ----------
    skel : skan.csr.Skeleton
        A Skeleton object containing graph whose center is to be found.

    Returns
    -------
    original_center_idx : int
        The index of central node of graph.
    """
    simp_csgraph, reduced_nodes = _simplify_graph(skel)
    simp_center_idx, _ = central_pixel(simp_csgraph)
    original_center_idx = reduced_nodes[simp_center_idx]

    return original_center_idx


def approximate_somas(im, regions, src=None):
    somas_estimates = []
    if src is not None:
        somas_estimates = np.load(src, allow_pickle=True)[0]
        return somas_estimates

    for region in regions:
        minz, miny, minx, maxz, maxy, maxx = region['bbox']
        ll = np.array([minz, miny, minx])
        seg_cell = im[minz:maxz, miny:maxy, minx:maxx] * region['image']

        # try:
        # problematic cause multiple forks in prim
        # distance = ndi.distance_transform_edt(region['image'])
        # distance = seg_cell
        # distance = imnorm(distance)
        # distance = gaussian(distance, sigma=1)
        # blurred_opening = opening(distance)
        # blobs = _get_blobs(blurred_opening, 'confocal')
        # coords = np.round(
        #     blobs[blobs[:, 3].argsort()][:, :-1]).astype(int)

        # # assure well separated blobs
        # opened_labels = label(opening(distance))
        # areg = [opened_labels[tuple(coord)] for coord in coords]
        # visited = []
        # filtered_coords = []

        # for i in range(len(areg)):
        #     if areg[i] not in visited:
        #         filtered_coords.append(coords[i])
        #         visited.append(areg[i])

        # filtered_coords = np.array(filtered_coords)

        # mask = np.zeros_like(seg_cell, dtype=bool)
        # mask[tuple(filtered_coords.T)] = True

        # # filter blobs with local max peaks
        # coords = peak_local_max(distance, footprint=np.ones((3, 3, 3)))
        # mask[tuple(coords.T)] *= True

        # filter blobs with forks
        imskel = skeletonize(region['image'])
        skel = Skeleton(imskel)

        # soma finding
        original_center_idx = _fast_graph_center_idx(skel)

        # neighbors = [original_center_idx]
        # original_center_idx = neighbors.pop(0)

        itr = 0
        ITR_CUTOFF = 100
        # while (
        #     region['image'][tuple(skel.coordinates[original_center_idx].astype(int))] == 0
        #     and itr < ITR_CUTOFF
        # ):
        #     if len(neighbors) == 0:
        #         neighbors = list(skel.nbgraph.neighbors(original_center_idx))
        #     original_center_idx = neighbors.pop(0)
        #     itr += 1

        if itr < ITR_CUTOFF:
            final_coords = np.asarray([
                skel.coordinates[original_center_idx].astype(int)
                ])
        else:
            # fail-safe select coord w/ highest degree
            final_coords = np.asarray([
                skel.coordinates[np.argmax(skel.degrees)].astype(int)
                ])
        # min spanning tree
        # _, px_coords = skeleton_to_csgraph(skel)
        # px_coords = px_coords.astype(int)
        # mst_skel = np.zeros_like(seg_cell, dtype=bool)
        # for c in px_coords:
        #     skel[tuple(c)] = True

        # imdegree = make_degree_image(skel)
        # fork_degree = 2
        # mask = (imdegree > fork_degree)

        # sanity check if all empty pxls
        # final_coords = np.asarray([
        #     c for c in np.transpose(np.nonzero(mask)) if seg_cell[tuple(c.T)]
        #     ])
        # final_coords = skel.coordinates[skel.degrees > fork_degree].astype(int)

        # if len(final_coords) == 0:
        #     # raise Exception('Failed automatic soma detection!')
        #     final_coords = np.asarray([
        #         np.unravel_index(np.argmax(imdegree), imskel.shape)
        #         ])

        somas_estimates.extend(ll + final_coords)
        # except:
        #     centroid = np.round_(region['centroid']).astype(np.int64)
        #     if region['image'][tuple(centroid - ll)]:
        #         somas_estimates.append(centroid)
        #     else:
        #         # Desparate measure: set the max value index
        #         somas_estimates.append(
        #             ll + np.unravel_index(np.argmax(seg_cell), region['image'].shape))

    return somas_estimates


def arrange_regions(filtered_labels):
    objects = ndi.find_objects(filtered_labels)
    ndim = filtered_labels.ndim

    regions = []
    for itr, slice in enumerate(objects):
        if slice is None:
            continue
        label = itr + 1
        template = dict(image=None, bbox=None, centroid=None, vol=None)
        template['image'] = (filtered_labels[slice] == label)
        template['image'] = ndi.binary_fill_holes(template['image'])
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
    __slots__ = ('im_path', 'DECONV_ITR', 'CLIP_LIMIT', 'ROI_PATH', 'ROI_NAME',
            'SCALE', 'imoriginal', 'impreprocessed', 'imsegmented', 'labels', 'imbinary',
            'REF_IM_PATH', 'REF_ROI_PATH', 'refdenoised', "skip_preprocess",
            'roi_polygon', 'in_box', 'regions', 'residue', 'OUT_DIR',
            'LOW_THRESH', 'HIGH_THRESH', 'LOW_AUTO_THRESH', 'HIGH_AUTO_THRESH',
            'n_region', 'PROPS', 'metadata',
            'filtered_regions',  # watershed cache
            'LOW_VOLUME_CUTOFF', 'HIGH_VOLUME_CUTOFF', 'OUT_DIMS', 'SEGMENT_TYPE',  # export cells
            'label_diff',  # reproducibility of manual label refinement
            'somas_estimates', 'ALL_PT_ESTIMATES', 'FINAL_PT_ESTIMATES'  # reproducibility of watershed label refinement
            )

    def __init__(
        self,
        im_path,
        channel=0,
        roi_path=None,
        roi_name=None,
        ref_im_path=None,
        ref_roi_path=None,
        out_dir='Autocropped'
    ):
        if not (
            (roi_path in (None, '') and roi_name in (None, ''))
            or (type(roi_path) == str and type(roi_name) == str)
        ):
            raise ValueError('Load ROI properly')
        self.im_path = im_path
        self.REF_IM_PATH = ref_im_path
        imoriginal, SCALE, metadata = imread(im_path, ref_im_path, channel)
        self.imoriginal, self.SCALE = imoriginal, SCALE
        self.impreprocessed = imoriginal
        self.metadata = metadata
        self.ROI_PATH = roi_path
        self.REF_ROI_PATH = ref_roi_path
        self.ROI_NAME = roi_name
        self.labels = None
        self.regions = None
        self.residue = None
        self.in_box = None
        self.ALL_PT_ESTIMATES = None
        self.FINAL_PT_ESTIMATES = None
        self.n_region = None

        DIR = path.join(getcwd(), out_dir)
        OUT_DIR = path.join(DIR, only_name(self.im_path) + \
                f'{"" if self.ROI_NAME == "" else "-" + str(self.ROI_NAME)}/')
        self.OUT_DIR = OUT_DIR
        _mkdir_if_not(OUT_DIR)
        _mkdir_if_not(path.join(OUT_DIR, '.cache'))

    def _suppl(self):
        cached_filename = only_name(im_path) + '.zarr'
        cached = False

        if skip_preprocess:
            imdeconvolved = imoriginal
        else:
            try:
                if cached_filename in listdir(cache_dir):
                    imdeconvolved = np.asarray(da.from_zarr(path.join(cache_dir, cached_filename)))
                    cached = True
                else:
                    raise ValueError('Preprocessed image not cached')
            except:
                cached = False
                imdeconvolved = deconvolve(imoriginal, im_path, iters=deconv_itr)

        imshape = imdeconvolved.shape

        if cached or skip_preprocess:
            impreprocessed = imdeconvolved
        else:
            #TODO: hardcoded 50
            rb_radius = (min(imdeconvolved.shape) -
                         1)//2 if imoriginal.ndim == 3 else 50
            if rb_radius < 25:
                background = rolling_ball(imdeconvolved, radius=rb_radius)
            else:
                background = np.zeros_like(imdeconvolved)
                for i in range(imdeconvolved.shape[0]):
                    background[i] = rolling_ball(imdeconvolved[i], radius=rb_radius)

            impreprocessed = equalize_adapthist(imdeconvolved-background,
                clip_limit=clip_limit
                )

            zarr.save(f"{cache_dir}/{cached_filename}", impreprocessed)

        select_roi = not(roi_path in (None, '')) and not(roi_name in (None, ''))

        roi_polygon = (np.array([[0, 0], [impreprocessed.shape[0]-1, 0],
                                 [impreprocessed.shape[0]-1, impreprocessed.shape[1]-1],
                                 [0, impreprocessed.shape[1]-1]])
                       if not select_roi else
                       select_ROI(impreprocessed, f'{only_name(im_path)}-{roi_name}', roi_path))
        self.roi_polygon = roi_polygon

        if select_roi:
            mask, bounds = mask_ROI(imoriginal, roi_polygon)
            imoriginal *= np.broadcast_to(mask, imoriginal.shape)
            imoriginal = imoriginal[bounds]  # reduce non-empty

            impreprocessed *= np.broadcast_to(mask, impreprocessed.shape)
            impreprocessed = impreprocessed[bounds]
        self.imoriginal = imoriginal

        # Load reference
        if not(ref_im_path in (None, '')):
            imname = only_name(ref_im_path)
            REF_SUFFIX = f'-{roi_name}-ref.zarr'
            try:
                cached_filename = imname + REF_SUFFIX
                refdenoised = np.asarray(da.from_zarr(path.join(cache_dir, cached_filename)))
            except:
                cached_filename = imname + '.zarr'
                cached = False

                try:
                    if cached_filename in listdir(cache_dir):
                        refdeconvolved = np.asarray(da.from_zarr(path.join(cache_dir, cached_filename)))
                        cached = True
                    else:
                        raise ValueError('Preprocessed image not cached')
                except:
                    cached = False
                    refdeconvolved = deconvolve(imread(ref_im_path)[0], ref_im_path, iters=deconv_itr)

                if cached or skip_preprocess:
                    refpreprocessed = refdeconvolved
                else:
                    rb_radius = (min(refdeconvolved.shape)-1)//2 if refdeconvolved.ndim == 3 else 128
                    if rb_radius < 25:
                        background = rolling_ball(refdeconvolved, radius=rb_radius)
                    else:
                        background = np.zeros_like(refdeconvolved)
                        for i in range(refdeconvolved.shape[0]):
                            background[i] = rolling_ball(refdeconvolved[i], radius=rb_radius)
                    if clip_limit == 0:
                        refpreprocessed = refdeconvolved-background
                    else:
                        refpreprocessed = equalize_adapthist(refdeconvolved-background, clip_limit=clip_limit)

                SELECT_ROI = not(ref_roi_path in (None, ''))

                roi_polygon = (np.array([[0, 0], [refpreprocessed.shape[0]-1, 0],
                                         [refpreprocessed.shape[0]-1, refpreprocessed.shape[1]-1],
                                         [0, refpreprocessed.shape[1]-1]])
                    if not SELECT_ROI else select_ROI(refpreprocessed, f'{imname}-{roi_name}', ref_roi_path))

                ref_denoise_parameters = calibrate_nlm_denoiser(refpreprocessed)
                mask, bounds = mask_ROI(refpreprocessed, roi_polygon)
                refpreprocessed *= np.broadcast_to(mask, refpreprocessed.shape)
                refpreprocessed = refpreprocessed[bounds]  # reduce non-empty

                # ll, ur = get_maximal_rectangle([roi_polygon])
                # if SELECT_ROI:
                #     ll, ur = np.ceil(ll).astype(int), np.floor(ur).astype(int)
                #     llx, lly = ll; urx, ury = ur
                #     llx -= roi_polygon[:, 0].min(); urx -= roi_polygon[:, 0].min()
                #     lly -= roi_polygon[:, 1].min(); ury -= roi_polygon[:, 1].min()
                # else:
                #     lly = 0; llx = 0
                #     ury, urx = refpreprocessed.shape[1:]
                #     ury -= 1; urx -= 1

                refdenoised = denoise(refpreprocessed, ref_denoise_parameters)
                zarr.save(path.join(cache_dir, imname + REF_SUFFIX), refdenoised)

            self.refdenoised = refdenoised

        # change y axis from rows to cartesian
        factor = np.array([0, imshape[1]-1])
        coords = (factor - roi_polygon) * np.array([-1, 1])

        # get the maximally inscribed rectangle
        # ll, ur = None, None
        # try:
        #     ll, ur = get_maximal_rectangle([coords])
        # except:
        #     i = 0
        #     while (
        #         (ll is not None or ur is not None)
        #         and (ll[0] < 0 or ll[1] < 0 or ur[0] < 0 or ur[1] < 0)
        #         and i < coords.shape[0]
        #     ):
        #         ll, ur = get_maximal_rectangle([coords], i)
        #         i += 1
        
        # if (
        #     ll is None or ur is None
        #     or (ll[0] < 0 or ll[1] < 0 or ur[0] < 0 or ur[1] < 0)
        # ):
        #     ll, ur = np.array([roi_polygon[:, 0].min(), roi_polygon[:, 1].min()]), np.array(self.imrect.shape[1:][::-1])-1 + np.array((roi_polygon[:, 0].min(), roi_polygon[:, 1].min()))
        # else:
        #     ll = (ll - factor) * np.array([1, -1])
        #     ur = (ur - factor) * np.array([1, -1])

        # if select_roi:
        #     ll, ur = np.ceil(ll).astype(int), np.floor(ur).astype(int)
        #     minx, miny = ll; maxx, maxy = ur
        #     minx -= roi_polygon[:, 0].min(); maxx -= roi_polygon[:, 0].min()
        #     miny -= roi_polygon[:, 1].min(); maxy -= roi_polygon[:, 1].min()
        # else:
        #     miny = 0; minx = 0
        #     maxy, maxx = impreprocessed.shape[1:]
        #     maxy -= 1; maxx -= 1

        mask = mask[bounds[-2:]]
        miny, maxy, minx, maxx = get_maximal_rectangle(mask)

        self.in_box = (miny, minx, maxy, maxx)

        # # maxrectOverlay
        # fig, ax = plt.subplots()

        # # Display the image
        # ax.imshow(mask)

        # # Create a Rectangle patch
        # rect = patches.Rectangle((minx, miny), maxx-minx, maxy-miny,
        #                          linewidth=1, edgecolor='r',
        #                          facecolor='none')
        # # Add the patch to the Axes
        # ax.add_patch(rect)
        # plt.show()

        denoise_parameters = calibrate_nlm_denoiser(impreprocessed[:, miny:maxy, minx:maxx])
        imdenoised = denoise(impreprocessed, denoise_parameters)

        if not(ref_im_path in (None, '')):
            imdenoised = match_histograms(imdenoised, refdenoised)

        imdenoised = ndi.median_filter(imdenoised, size=2)

        self.imdenoised = imdenoised

    def segment(
        self,
        low_thresh,
        high_thresh,
        low_auto_thresh=None,
        high_auto_thresh=None,
        particle_filter_size=64
    ):
        """."""
        miny, minx, maxy, maxx = self.in_box
        if low_auto_thresh is not None:
            low_thresh = eval(f'threshold_{low_auto_thresh}(self.impreprocessed[:, miny:maxy, minx:maxx])')
        if high_auto_thresh is not None:
            high_thresh = eval(f'threshold_{high_auto_thresh}(self.impreprocessed[:, miny:maxy, minx:maxx])')
        self.LOW_THRESH, self.HIGH_THRESH = low_thresh, high_thresh
        self.LOW_AUTO_THRESH = low_auto_thresh
        self.HIGH_AUTO_THRESH = high_auto_thresh

        thresholded = threshold(self.impreprocessed, low_thresh, high_thresh)
        labels = label_thresholded(thresholded)

        prefiltering_volume = thresholded.sum()
        print(f'Prefiltering Volume: {prefiltering_volume}')
        regions = arrange_regions(labels)

        # reconstruct despeckled filtered_labels
        filtered_labels = np.zeros_like(labels, dtype=int)

        reg_itr = 1

        # build volume sorted region labels
        for region in regions:
            minz, miny, minx, maxz, maxy, maxx = region['bbox']
            filtered_labels[minz:maxz, miny:maxy, minx:maxx] += region['image'] * reg_itr
            reg_itr += 1

        self.imsegmented = self.impreprocessed * (filtered_labels > 0)
        self.labels = filtered_labels
        self.regions = regions

    def volume_cutoff(
        self,
        low_volume_cutoff=64,
        gui=True
    ):
        self.LOW_VOLUME_CUTOFF = low_volume_cutoff
        self.region_inclusivity = np.ones(len(self.regions), dtype=bool)
        self.REGION_INCLUSIVITY_LABELS = ['Include Region', 'Exclude Region']
        if gui:
            SCALE = self.SCALE
            viewer = napari.Viewer(ndisplay=3)
            viewer.add_image(self.impreprocessed, scale=SCALE)
            viewer.add_labels(self.labels, rendering='translucent', opacity=.5, scale=SCALE)

            region_props = PyQt5.QtWidgets.QLabel()

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
                self.LOW_VOLUME_CUTOFF = cutoff
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
                self.imsegmented = self.impreprocessed * (self.labels > 0)
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
        else:
            self.labels.fill(0)
            itr = 1
            filtered_regions = []
            for region in self.regions:
                if low_volume_cutoff <= region['vol']:
                    minz, miny, minx, maxz, maxy, maxx = region['bbox']
                    self.labels[minz:maxz, miny:maxy, minx:maxx] += region['image'] * itr
                    itr += 1
                    filtered_regions.append(region)
            self.regions = filtered_regions
            self.imsegmented = self.impreprocessed * (self.labels > 0)

    def approximate_somas(
        self,
        src=None
    ):
        """."""
        somas_estimates = approximate_somas(self.imsegmented, self.regions, src=src)
        self.somas_estimates = somas_estimates

    def separate_clumps(
        self,
        seed_src=None,
        gui=True,
        sync=False
    ):
        SCALE = self.SCALE
        somas_estimates = self.somas_estimates

        self.regions = sorted(self.regions, key=lambda region: region['vol'])
        regions = self.regions

        reconstructed_labels = np.zeros(self.impreprocessed.shape, dtype=int)
        for itr in range(len(regions)):
            minz, miny, minx, maxz, maxy, maxx = regions[itr]['bbox']
            reconstructed_labels[minz:maxz, miny:maxy, minx:maxx] += regions[itr]['image'] * (itr + 1)
        self.reconstructed_labels = reconstructed_labels

        region_props = PyQt5.QtWidgets.QLabel()
        region_inclusivity = self.region_inclusivity #  np.ones(len(regions), dtype=bool)
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
                data = viewer.layers['filtered_coords'].data
                for coord in filtered_coords:
                    if ~(data == coord).all(axis=1).any():
                        viewer.layers['filtered_coords'].add(np.round_(coord + ll).astype(np.int64))
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
                data = viewer.layers['filtered_coords'].data
                for coord in filtered_coords:
                    coord_to_add = np.round_(coord + ll).astype(np.int64)
                    #TODO: remove extra pts: Clump > 0 ko invert krke mask mai multiply kro aur fir wapis coord mai convert kro
                    if ~(data == coord_to_add).all(axis=1).any():
                        viewer.layers['filtered_coords'].add(coord_to_add)
            else:
                viewer.add_points(filtered_coords + ll, face_color='red', edge_width=0,
                                opacity=.6, size=5, name='filtered_coords', scale=SCALE)


        @magicgui(
            call_button='Watershed regions'
        )
        def clump_separation_napari():
            SEARCH_LAYER = 'filtered_coords'
            layer_names = [layer.name for layer in viewer.layers]
            somas_estimates = np.unique(viewer.layers[SEARCH_LAYER].data, axis=0)
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
                    im = self.impreprocessed[minz:maxz, miny:maxy, minx:maxx].copy()
                    im[~region['image']] = 0
                    markers = np.zeros(region['image'].shape, dtype=int)

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
            watershed_results = np.zeros(self.impreprocessed.shape, dtype=int)
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
            watershed_results = np.zeros(self.impreprocessed.shape, dtype=int)
            for itr in range(len(regions)):
                minz, miny, minx, maxz, maxy, maxx = regions[itr]['bbox']
                watershed_results[minz:maxz, miny:maxy, minx:maxx] += regions[itr]['image'] * (itr + 1)

            if 'reconstructed_labels' in layer_names:
                viewer.layers['reconstructed_labels'].data = watershed_results

            # After all changes (for reproducibility)
            final_soma = np.unique(viewer.layers['filtered_coords'].data, axis=0)

            # discarded clump ROI: to be subtracted: np.setdiff1d(somas_estimates, final_soma)
            ALL_PT_ESTIMATES, FINAL_PT_ESTIMATES = self.somas_estimates.copy(), final_soma
            self.ALL_PT_ESTIMATES = ALL_PT_ESTIMATES
            self.FINAL_PT_ESTIMATES = FINAL_PT_ESTIMATES
            self.reconstructed_labels = watershed_results
            self.imsegmented = self.impreprocessed * (watershed_results > 0)

            if sync:
                self.refine_soma_approx(sync=True)

        # if sync and gui is False:
        # else:
        viewer = napari.Viewer(ndisplay=3)
        viewer.add_image(self.impreprocessed, scale=SCALE, name='denoised')
        viewer.add_image(self.imsegmented, scale=SCALE, name='segmented')
        viewer.add_labels(self.reconstructed_labels, rendering='translucent', opacity=.5,
                        scale=SCALE, name='reconstructed_labels')
        viewer.add_points(somas_estimates, face_color='orange', edge_width=0,
                        opacity=.6, size=5, name='somas_coords', scale=SCALE)
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

        if seed_src is not None:
            final_pts = np.load(seed_src, allow_pickle=True)[1].astype(int)
            viewer.add_points(final_pts, face_color='red', edge_width=0,
                              opacity=.6, size=5, name='filtered_coords',
                              scale=SCALE)
        elif self.FINAL_PT_ESTIMATES is not None:
            viewer.add_points(self.FINAL_PT_ESTIMATES, face_color='red', edge_width=0,
                              opacity=.6, size=5, name='filtered_coords',
                              scale=SCALE)
        else:
            for i in range(len(self.regions)):
                select_region(i, True)
                interactive_segment1()

    def show_segmented(self, view='single', grid_size=None):
        """View segmentation results.

        Parameters
        ----------
        view : str
            How to display segmented cells, either 'grid' or 'single'.
        grid_size : int or None
            Size of the grid view.
        """
        denoised = self.impreprocessed
        regions = self.regions
        segmented = self.imsegmented

        def plot_batch(BATCH_NO):
            project_batch(BATCH_NO, N_BATCHES, regions, denoised)
            plt.show()

        self.n_region = 0
        extracted_cell = None
        # minz, miny, minx, maxz, maxy, maxx = 0, 0, 0, 0, 0, 0

        def plot_single(obj_index):
            self.n_region = obj_index
            extracted_cell = extract_obj(regions[obj_index], segmented)
            # minz, miny, minx, maxz, maxy, maxx = regions[obj_index]['bbox']
            projectXYZ(extracted_cell, *self.SCALE)

        if view == 'grid':
            # Set `BATCH_NO` to view detected objects in paginated 2D MIP views.
            N_BATCHES = paginate_objs(regions, grid_size)
            widgets.interact(plot_batch, BATCH_NO=widgets.IntSlider(min=0,
                max=N_BATCHES-1, layout=widgets.Layout(width='100%')))
        else:
            widgets.interact(plot_single,
                obj_index=widgets.IntSlider(min=0, max=len(regions)-1,
                layout=widgets.Layout(width='100%')))
    
    def refine_soma_approx(self, sync=False):
        self.n_region = 0
        denoised = self.impreprocessed
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
                viewer.add_image(denoised[minz:maxz, miny:maxy, minx:maxx],
                                 name='denoised', colormap='red', scale=SCALE,
                                 contrast_limits=(0,1))
                viewer.add_image(denoised[minz:maxz, miny:maxy, minx:maxx]*regions[selected_region]['image'],
                                 name='segmented', colormap='red', scale=SCALE,
                                 contrast_limits=(0,1))
                viewer.add_points(filtered_coords, face_color='lime', edge_width=0,
                                  opacity=.6, size=5, name='filtered_coords',
                                  scale=SCALE)
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
            markers = np.zeros(region['image'].shape, dtype=int)

            for i in range(somas_coords.shape[0]):
                markers[tuple(somas_coords[i])] = i + 1

            labels = _segment_clump(im, markers)
            viewer.add_labels(labels, rendering='translucent', opacity=.5, scale=SCALE)
            FINAL_PT_ESTIMATES = np.vstack((FINAL_PT_ESTIMATES, somas_coords+ll))
            FINAL_PT_ESTIMATES = np.unique(FINAL_PT_ESTIMATES, axis=0)
            self.FINAL_PT_ESTIMATES = FINAL_PT_ESTIMATES

        @magicgui(
            call_button="Confirm and apply all changes"
        )
        def save_approximates():
            somas_estimates = np.unique(self.FINAL_PT_ESTIMATES, axis=0)
            filtered_regions, residue = [], []
            separated_clumps = []

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
                    im = self.impreprocessed[minz:maxz, miny:maxy, minx:maxx].copy()
                    im[~region['image']] = 0
                    markers = np.zeros(region['image'].shape, dtype=int)

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

            self.regions = sorted(filtered_regions, key=lambda region: region['vol'])
            self.residue = residue
            watershed_results = np.zeros(self.impreprocessed.shape, dtype=int)
            for itr in range(len(filtered_regions)):
                minz, miny, minx, maxz, maxy, maxx = filtered_regions[itr]['bbox']
                watershed_results[minz:maxz, miny:maxy, minx:maxx] += filtered_regions[itr]['image'] * (itr + 1)

            # After all changes (for reproducibility)
            # discarded clump ROI: to be subtracted: np.setdiff1d(somas_estimates, final_soma)
            ALL_PT_ESTIMATES, FINAL_PT_ESTIMATES = self.somas_estimates.copy(), somas_estimates
            self.ALL_PT_ESTIMATES = ALL_PT_ESTIMATES
            self.FINAL_PT_ESTIMATES = FINAL_PT_ESTIMATES
            self.reconstructed_labels = watershed_results
            self.imsegmented = self.impreprocessed * (watershed_results > 0)

            if sync:
                self.export_cropped(self.LOW_VOLUME_CUTOFF, self.HIGH_VOLUME_CUTOFF,
                    self.OUT_DIMS, self.SEGMENT_TYPE)

        viewer.window.add_dock_widget(view_single_region, name='View individual region')
        viewer.window._dock_widgets['View individual region'].setFixedHeight(70)
        viewer.window.add_dock_widget(update_soma_estimates, name='Update soma estimates')
        viewer.window._dock_widgets['Update soma estimates'].setFixedHeight(70)
        viewer.window.add_dock_widget(save_approximates, name='Confirm all changes')
        view_single_region()

    def _export(self):
        img_path = self.im_path
        low_vol_cutoff = self.LOW_VOLUME_CUTOFF
        hi_vol_cutoff = self.HIGH_VOLUME_CUTOFF
        out_type = self.OUT_DIMS
        tissue_img = self.impreprocessed
        regions = self.regions
        residue_regions = None
        seg_type = self.SEGMENT_TYPE
        roi_name = self.ROI_NAME
        roi_polygon = self.roi_polygon
        roi_path = self.ROI_PATH
        OUT_TYPES = ('3d', 'mip', 'both')
        SEG_TYPES = ('segmented', 'unsegmented', 'both')

        if out_type not in OUT_TYPES:
            raise ValueError('`out_type` must be either of `3d`, '
                            '`mip`, `both`')
        if seg_type not in SEG_TYPES:
            raise ValueError('`seg_type` must be either of `segmented`, '
                            '`unsegmented`, `both`')

        DIR = getcwd() + '/Autocropped/'
        _mkdir_if_not(DIR)

        IMAGE_NAME = '.'.join(path.basename(img_path).split('.')[:-1])
        OUT_DIR = self.OUT_DIR

        _mkdir_if_not(OUT_DIR)

        if out_type == OUT_TYPES[2]:
            if seg_type == SEG_TYPES[2]:
                _mkdir_if_not(path.join(OUT_DIR, SEG_TYPES[0] + '_' + OUT_TYPES[0]))
                _mkdir_if_not(path.join(OUT_DIR, SEG_TYPES[0] + '_' + OUT_TYPES[1]))
                _mkdir_if_not(path.join(OUT_DIR, SEG_TYPES[1] + '_' + OUT_TYPES[0]))
                _mkdir_if_not(path.join(OUT_DIR, SEG_TYPES[1] + '_' + OUT_TYPES[1]))
            else:
                _mkdir_if_not(path.join(OUT_DIR, seg_type + '_' + OUT_TYPES[0]))
                _mkdir_if_not(path.join(OUT_DIR, seg_type + '_' + OUT_TYPES[1]))
        else:
            if seg_type == SEG_TYPES[2]:
                _mkdir_if_not(path.join(OUT_DIR, SEG_TYPES[0] + '_' + out_type))
                _mkdir_if_not(path.join(OUT_DIR, SEG_TYPES[1] + '_' + out_type))
            else:
                _mkdir_if_not(path.join(OUT_DIR, seg_type + '_' + out_type))

        cell_metadata = {}

        if img_path.split('.')[-1] == 'tif':
            with tifffile.TiffFile(img_path) as file:
                metadata = file.imagej_metadata
                cell_metadata['unit'] = metadata['unit']
                cell_metadata['spacing'] = metadata['spacing']
        # elif img_path.split('.')[-1] == 'czi':
        #     with czifile.CziFile(img_path) as file:
        #         metadata = file.metadata(False)['ImageDocument']['Metadata']
        #         cell_metadata['scaling'] = metadata['Scaling']
        cell_metadata['parent_image'] = path.abspath(img_path)
        cell_metadata['scale'] = self.SCALE

        if roi_polygon is not None:
            X, Y = _unwrap_polygon(roi_polygon)
            roi = (int(min(Y)), int(min(X)), int(max(Y) + 1), int(max(X) + 1))
            cell_metadata['roi_name'] = roi_name
            cell_metadata['roi'] = roi
            cell_metadata['roi_path'] = path.abspath(roi_path)

        for (obj, region) in enumerate(regions):
            if region['vol'] > hi_vol_cutoff:  # for postprocessing
                minz, miny, minx, maxz, maxy, maxx = region['bbox']
                segmented = tissue_img[minz:maxz, miny:maxy, minx:maxx].copy()
                segmented = img_as_ubyte(segmented)
                segmented[~region['image']] = 0

                try:
                    markers = _get_blobs(segmented, 'confocal').astype(int)[:, :-1]
                except:
                    markers = np.array([np.array(segmented.shape)]) // 2
                roi = _build_multipoint_roi(markers)

                name = str(uuid.uuid4().hex)
                out_name = path.join(OUT_DIR, name)
                self.regions[obj]['name'] = name

                cell_metadata['bounds'] = region['bbox']
                out_metadata = json.dumps(cell_metadata)

                tifffile.imsave(
                    out_name + '.tif',
                    segmented,
                    description=out_metadata,
                    software='Autocrop'
                )
                tifffile.imsave(
                    out_name + '_mip.tif',
                    np.max(segmented, 0),
                    description=out_metadata,
                    software='Autocrop'
                )
                roi.tofile(out_name + '.roi')
            if low_vol_cutoff <= region['vol'] <= hi_vol_cutoff:
                minz, miny, minx, maxz, maxy, maxx = region['bbox']
                name = str(uuid.uuid4().hex) + '.tif'
                self.regions[obj]['name'] = name

                # Cell-specific metadata
                cell_metadata['bounds'] = region['bbox']
                cell_metadata['cell_volume'] = int(region['vol'])
                cell_metadata['centroid'] = region['centroid']
                # cell_metadata['territorial_volume'] = int(region.convex_area)
                out_metadata = json.dumps(cell_metadata)

                if seg_type == SEG_TYPES[0] or seg_type == SEG_TYPES[2]:
                    segmented = tissue_img[minz:maxz, miny:maxy, minx:maxx].copy()
                    segmented[~region['image']] = 0
                    segmented = segmented / segmented.max()  # contrast stretch
                    segmented = img_as_ubyte(segmented)

                    if out_type == OUT_TYPES[2]:
                        out = segmented
                        out_name = path.join(OUT_DIR, f'{SEG_TYPES[0]}_{OUT_TYPES[0]}', name)
                        tifffile.imsave(out_name, out, description=out_metadata,
                                        software='Autocrop')

                        out = np.pad(np.max(segmented, 0),
                                    pad_width=max(segmented.shape[1:]) // 5,
                                    mode='constant')
                        out_name = path.join(OUT_DIR, f'{SEG_TYPES[0]}_{OUT_TYPES[1]}', name)
                        tifffile.imsave(out_name.replace('.tif', '_mip.tif'), out,
                                        description=out_metadata,
                                        software='Autocrop')
                    else:
                        out = segmented if out_type == OUT_TYPES[0] else np.pad(
                            np.max(segmented, 0),
                            pad_width=max(segmented.shape[1:]) // 5,
                            mode='constant')
                        out_name = path.join(OUT_DIR, f'{SEG_TYPES[0]}_{out_type}', name)
                        tifffile.imsave(out_name.replace('.tif', '_mip.tif'), out,
                                        description=out_metadata,
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
                    # contrast stretch
                    minv, maxv = segmented.min(), segmented.max()
                    segmented = (segmented - minv) / (maxv - minv)

                    segmented = img_as_ubyte(segmented)

                    if out_type == OUT_TYPES[2]:
                        out = segmented
                        out_name = path.join(OUT_DIR, f'{SEG_TYPES[1]}_{OUT_TYPES[0]}', name)
                        tifffile.imsave(out_name, out, description=out_metadata,
                                        software='Autocrop')

                        out = np.max(segmented, 0)
                        out_name = path.join(OUT_DIR, f'{SEG_TYPES[1]}_{OUT_TYPES[1]}', name)
                        tifffile.imsave(out_name, out, description=out_metadata,
                                        software='Autocrop')
                    else:
                        out = segmented if out_type == OUT_TYPES[0] else np.max(
                            segmented, 0)
                        out_name = path.join(OUT_DIR, f'{SEG_TYPES[1]}_{out_type}', name)
                        tifffile.imsave(out_name, out, description=out_metadata,
                                        software='Autocrop')

        if residue_regions is not None:
            RES_DIR = path.join(OUT_DIR, 'residue')
            _mkdir_if_not(RES_DIR)
            for (obj, region) in enumerate(residue_regions):
                if low_vol_cutoff <= region['vol']:  # for postprocessing
                    minz, miny, minx, maxz, maxy, maxx = region['bbox']
                    segmented = tissue_img[minz:maxz, miny:maxy, minx:maxx].copy()
                    segmented[~region['image']] = 0
                    segmented = segmented / segmented.max()  # contrast stretch
                    segmented = img_as_ubyte(segmented)

                    try:
                        markers = _get_blobs(segmented, 'confocal')
                        markers = markers.astype(int)[:, :-1]
                    except:
                        markers = np.array([np.array(segmented.shape)]) // 2
                        markers = markers[:, [0, 2, 1]]
                    roi = _build_multipoint_roi(markers)

                    name = str(uuid.uuid4().hex)
                    out_name = path.join(RES_DIR, name)

                    cell_metadata['bounds'] = region['bbox']
                    out_metadata = json.dumps(cell_metadata)

                    tifffile.imsave(
                        out_name + '.tif',
                        segmented,
                        description=out_metadata,
                        software='Autocrop'
                    )
                    tifffile.imsave(
                        out_name + '_mip.tif',
                        np.max(segmented, 0),
                        description=out_metadata,
                        software='Autocrop'
                    )
                    roi.tofile(out_name + '.roi')

        params = {
            'NAME_ROI': self.ROI_NAME,
            'REF_IM_PATH': pathlib2.Path(self.REF_IM_PATH).as_posix(),
            'LOW_THRESH': self.LOW_THRESH,
            'HIGH_THRESH': self.HIGH_THRESH,
            'LOW_AUTO_THRESH': self.LOW_AUTO_THRESH,
            'HIGH_AUTO_THRESH': self.HIGH_AUTO_THRESH,
            'LOW_VOLUME_CUTOFF': int(self.LOW_VOLUME_CUTOFF),  # filter noise/artifacts
            'HIGH_VOLUME_CUTOFF': int(self.HIGH_VOLUME_CUTOFF),  # filter cell clusters
            'SEGMENT_TYPE': self.SEGMENT_TYPE,
            'OUTPUT_DIMS': self.OUT_DIMS,
        }

        with open(path.join(self.OUT_DIR, '.params.json'), 'w') as out:
            json.dump(params, out)

        zarr.save(path.join(self.OUT_DIR, '.label_diff.zarr'), self.label_diff)
        zarr.save(path.join(self.OUT_DIR, '.somas_coords.zarr'), self.FINAL_PT_ESTIMATES)

    def export_cropped(
        self,
        low_volume_cutoff=None,
        high_volume_cutoff=None,
        out_dims='both',
        segment_type='both',
        gui=True,
    ):
        if low_volume_cutoff is None:
            low_volume_cutoff = self.LOW_VOLUME_CUTOFF# self.regions[0]['vol']
        if high_volume_cutoff is None:
            high_volume_cutoff = self.regions[-1]['vol']

        if low_volume_cutoff < self.LOW_VOLUME_CUTOFF:
            raise ValueError('low_volume_cutoff should not be lesser than previously set.')

        self.LOW_VOLUME_CUTOFF = low_volume_cutoff  # filter noise/artifacts
        self.HIGH_VOLUME_CUTOFF = high_volume_cutoff  # filter cell clusters
        self.OUT_DIMS = out_dims  # '3d' for 3D cells, 'mip' for Max Intensity Projections
        self.SEGMENT_TYPE = segment_type
        SCALE = self.SCALE

        reconstructed_cells = np.zeros_like(self.impreprocessed)

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
                    segmented_cell = region['image'] * self.impreprocessed[minz:maxz, miny:maxy, minx:maxx]
                    segmented_cell = segmented_cell / (segmented_cell.max() - segmented_cell.min())
                    reconstructed_cells[minz:maxz, miny:maxy, minx:maxx] += segmented_cell
            minz, miny, minx, maxz, maxy, maxx = self.regions[self.n_region]['bbox']
            if gui:
                viewer.layers[0].data = reconstructed_cells

        vol_cutoff_slider.valueChanged.connect(vol_cutoff_update)


        @magicgui(
            call_button='Export Cells',
            output_option=dict(choices=['both', '3d', 'mip']),
            segment_type=dict(choices=['both', 'segmented', 'unsegmented'])
        )
        def export(
            output_option=out_dims,
            segment_type=segment_type
        ):
            self.OUT_DIMS, self.SEGMENT_TYPE = output_option, segment_type
            self.imsegmented = reconstructed_cells
            export_cells(self.im_path, self.LOW_VOLUME_CUTOFF,
                         self.HIGH_VOLUME_CUTOFF, output_option, self.impreprocessed,
                         self.regions, None, segment_type, self.ROI_NAME, self.roi_polygon)

            params = {
                'NAME_ROI': self.ROI_NAME,
                'REF_IM_PATH': self.REF_IM_PATH,
                'skip_preprocess': self.skip_preprocess,
                'DECONV_ITR': self.DECONV_ITR,
                'CLIP_LIMIT': self.CLIP_LIMIT,
                'LOW_THRESH': self.LOW_THRESH,
                'HIGH_THRESH': self.HIGH_THRESH,
                'LOW_AUTO_THRESH': self.LOW_AUTO_THRESH,
                'HIGH_AUTO_THRESH': self.HIGH_AUTO_THRESH,
                'LOW_VOLUME_CUTOFF': self.LOW_VOLUME_CUTOFF,  # filter noise/artifacts
                'HIGH_VOLUME_CUTOFF': self.HIGH_VOLUME_CUTOFF,  # filter cell clusters
                'SEGMENT_TYPE': self.SEGMENT_TYPE,
                'OUTPUT_DIMS': self.OUT_DIMS,
            }

            with open(self.OUT_DIR + '.params.json', 'w') as out:
                json.dump(params, out)

            out = np.array([self.ALL_PT_ESTIMATES, self.FINAL_PT_ESTIMATES])
            np.save(self.OUT_DIR + '.somas_estimates.npy', out)

        if gui:
            viewer = napari.Viewer(ndisplay=3)
            viewer.add_image(reconstructed_cells, colormap='inferno', scale=SCALE)
            viewer.window.add_dock_widget(vol_cutoff_slider, name='Volume Cutoff', area='bottom')

            viewer.window._dock_widgets['Volume Cutoff'].setFixedHeight(80)
            viewer.window.add_dock_widget(export, name='Export Cells')
            vol_cutoff_update()
        else:
            vol_cutoff_update()
            export(out_dims, segment_type)
