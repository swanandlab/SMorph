import json
import uuid
from os import getcwd, mkdir, path
from shutil import rmtree

import czifile
import numpy as np
import roifile
import skimage.io as io
import tifffile
from skimage import img_as_float, img_as_ubyte

from .core import _unwrap_polygon
from ...analysis._skeletal import _get_blobs


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
        img = np.squeeze(img)
        if img.ndim > 3:
            img = img[channel_interest]
    elif img_path.split('.')[-1] == 'lsm':
        img = tifffile.imread(img_path)
        img = np.squeeze(img)
        if img.ndim > 3:
            img = img[:, channel_interest]
    else:
        img = np.squeeze(io.imread(img_path))
        if img.ndim > 3:
            img = img[channel_interest]

    img = img_as_float(img)
    return img


def _mkdir_if_not(name):
    """Collision-free mkdir"""
    if not (path.exists(name) and path.isdir(name)):
        mkdir(name)


def _build_multipoint_roi(markers):
    xy, z = markers[:, [2, 1]], markers[:, 0] + 1
    left, top = xy.min(axis=0)
    right, bottom = xy.max(axis=0)

    roi = roifile.ImagejRoi()
    roi.version = 227
    roi.roitype = roifile.ROI_TYPE(10)
    roi.options = roifile.ROI_OPTIONS(1024)
    roi.n_coordinates = xy.shape[0]
    roi.name = 'somas'
    roi.left, roi.top = int(left), int(top)
    roi.right, roi.bottom = int(right), int(bottom)
    roi.integer_coordinates = xy - [roi.left, roi.top]
    roi.counters = np.array([0] * roi.n_coordinates)
    roi.counter_positions = markers[:, 0]
    roi.arrow_style_or_aspect_ratio = 0
    roi.stroke_width = 4
    roi.stroke_color = b'\xFF\xFF\x00\x00'  # RED
    return roi


def export_cells(
    img_path,
    low_vol_cutoff,
    hi_vol_cutoff,
    out_type,
    tissue_img,
    regions,
    residue_regions=None,
    seg_type='segmented',
    roi_name='',
    roi_polygon=None,
    roi_path=''
):
    """Exports cropped cells.

    Parameters
    ----------
    img_path : str
        Path to the tissue image.
    low_vol_cutoff : int
        Least volume (number of voxels) of segmented object
        representing an individual cell.
    hi_vol_cutoff : int
        Highest volume (number of voxels) of segmented object
        representing an individual cell.
    out_type : str
        Required type of output cells, either of '3d', 'mip', 'both'
    tissue_img : ndarray
        Original image.
    regions : list
        Segmentation resuts from the tissue image.
    seg_type : str, optional
        Type of segmentation to be performed on the output cells, either of
        'segmented', 'unsegmented', 'both', by default 'segmented'
    roi_name : str, optional
        Name of the selected ROI, by default ''
    roi_polygon : list or LineBuilder or None, optional
        Coordinates of vertices of polygon representing the Region of
        Interest, by default None

    """
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
    OUT_DIR = DIR + IMAGE_NAME + \
              f'{"" if roi_name == "" else "-" + str(roi_name)}/'
    # if path.exists(OUT_DIR) and path.isdir(OUT_DIR):  # mandatory new dir
    #     rmtree(OUT_DIR)
    # mkdir(OUT_DIR)
    _mkdir_if_not(OUT_DIR)

    if out_type == OUT_TYPES[2]:
        if seg_type == SEG_TYPES[2]:
            _mkdir_if_not(OUT_DIR + SEG_TYPES[0] + '_' + OUT_TYPES[0])
            _mkdir_if_not(OUT_DIR + SEG_TYPES[0] + '_' + OUT_TYPES[1])
            _mkdir_if_not(OUT_DIR + SEG_TYPES[1] + '_' + OUT_TYPES[0])
            _mkdir_if_not(OUT_DIR + SEG_TYPES[1] + '_' + OUT_TYPES[1])
        else:
            _mkdir_if_not(OUT_DIR + seg_type + '_' + OUT_TYPES[0])
            _mkdir_if_not(OUT_DIR + seg_type + '_' + OUT_TYPES[1])
    else:
        if seg_type == SEG_TYPES[2]:
            _mkdir_if_not(OUT_DIR + SEG_TYPES[0] + '_' + out_type)
            _mkdir_if_not(OUT_DIR + SEG_TYPES[1] + '_' + out_type)
        else:
            _mkdir_if_not(OUT_DIR + seg_type + '_' + out_type)

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
    cell_metadata['parent_image'] = path.abspath(img_path)

    if roi_polygon is not None:
        X, Y = _unwrap_polygon(roi_polygon)
        roi = (int(min(Y)), int(min(X)), int(max(Y) + 1), int(max(X) + 1))
        cell_metadata['roi_name'] = roi_name
        cell_metadata['roi'] = roi
        cell_metadata['roi_path'] = path.abspath(roi_path)

    for (obj, region) in enumerate(regions):
        if region.area > hi_vol_cutoff:  # for postprocessing
            minz, miny, minx, maxz, maxy, maxx = region.bbox
            segmented = tissue_img[minz:maxz, miny:maxy, minx:maxx].copy()
            segmented = img_as_ubyte(segmented)
            segmented[~region.filled_image] = 0

            try:
                markers = _get_blobs(segmented, 'confocal').astype(int)[:, :-1]
            except:
                markers = np.array([np.array(segmented.shape)]) // 2
            roi = _build_multipoint_roi(markers)

            name = str(uuid.uuid4().hex)
            out_name = f'{OUT_DIR}/'+name

            cell_metadata['bounds'] = region.bbox
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
        if low_vol_cutoff <= region.area <= hi_vol_cutoff:
            minz, miny, minx, maxz, maxy, maxx = region.bbox
            name = str(uuid.uuid4().hex) + '.tif'

            # Cell-specific metadata
            cell_metadata['bounds'] = region.bbox
            cell_metadata['cell_volume'] = int(region.filled_area)
            cell_metadata['centroid'] = region.centroid
            cell_metadata['territorial_volume'] = int(region.convex_area)
            out_metadata = json.dumps(cell_metadata)

            if seg_type == SEG_TYPES[0] or seg_type == SEG_TYPES[2]:
                segmented = tissue_img[minz:maxz, miny:maxy, minx:maxx].copy()
                segmented[~region.filled_image] = 0
                segmented = segmented / segmented.max()  # contrast stretch
                segmented = img_as_ubyte(segmented)

                if out_type == OUT_TYPES[2]:
                    out = segmented
                    out_name = f'{OUT_DIR}{SEG_TYPES[0]}_{OUT_TYPES[0]}/'+name
                    tifffile.imsave(out_name, out, description=out_metadata,
                                    software='Autocrop')

                    out = np.pad(np.max(segmented, 0),
                                 pad_width=max(segmented.shape[1:]) // 5,
                                 mode='constant')
                    out_name = f'{OUT_DIR}{SEG_TYPES[0]}_{OUT_TYPES[1]}/'+name
                    tifffile.imsave(out_name.replace('.tif', '_mip.tif'), out, description=out_metadata,
                                    software='Autocrop')
                else:
                    out = segmented if out_type == OUT_TYPES[0] else np.pad(
                        np.max(segmented, 0),
                        pad_width=max(segmented.shape[1:]) // 5,
                        mode='constant')
                    out_name = f'{OUT_DIR}{SEG_TYPES[0]}_{out_type}/'+name
                    tifffile.imsave(out_name.replace('.tif', '_mip.tif'), out, description=out_metadata,
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
                    out_name = f'{OUT_DIR}{SEG_TYPES[1]}_{OUT_TYPES[0]}/'+name
                    tifffile.imsave(out_name, out, description=out_metadata,
                                    software='Autocrop')

                    out = np.max(segmented, 0)
                    out_name = f'{OUT_DIR}{SEG_TYPES[1]}_{OUT_TYPES[1]}/'+name
                    tifffile.imsave(out_name, out, description=out_metadata,
                                    software='Autocrop')
                else:
                    out = segmented if out_type == OUT_TYPES[0] else np.max(
                        segmented, 0)
                    out_name = f'{OUT_DIR}{SEG_TYPES[1]}_{out_type}/'+name
                    tifffile.imsave(out_name, out, description=out_metadata,
                                    software='Autocrop')

    if residue_regions is not None:
        RES_DIR = OUT_DIR + 'residue'
        _mkdir_if_not(RES_DIR)
        for (obj, region) in enumerate(residue_regions):
            if low_vol_cutoff <= region.area:  # for postprocessing
                minz, miny, minx, maxz, maxy, maxx = region.bbox
                segmented = tissue_img[minz:maxz, miny:maxy, minx:maxx].copy()
                segmented[~region.filled_image] = 0
                segmented = segmented / segmented.max()  # contrast stretch
                segmented = img_as_ubyte(segmented)

                try:
                    markers = _get_blobs(segmented, 'confocal').astype(int)[:, :-1]
                except:
                    markers = np.array([np.array(segmented.shape)]) // 2
                    markers = markers[:, [0, 2, 1]]
                roi = _build_multipoint_roi(markers)

                name = str(uuid.uuid4().hex)
                out_name = f'{RES_DIR}/'+name

                cell_metadata['bounds'] = region.bbox
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
