import json
import uuid
from os import getcwd, mkdir, path
from shutil import rmtree

import czifile
import numpy as np
import skimage.io as io
import tifffile
from skimage import img_as_float, img_as_ubyte

from .core import _unwrap_polygon


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


def export_cells(
    img_path,
    low_vol_cutoff,
    hi_vol_cutoff,
    out_type,
    tissue_img,
    regions,
    seg_type='segmented',
    roi_name='',
    roi_polygon=None
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
    if not (path.exists(DIR) and path.isdir(DIR)):
        mkdir(DIR)

    IMAGE_NAME = '.'.join(path.basename(img_path).split('.')[:-1])
    OUT_DIR = DIR + IMAGE_NAME + \
              f'{"" if roi_name == "" else "-" + str(roi_name)}/'
    if path.exists(OUT_DIR) and path.isdir(OUT_DIR):
        rmtree(OUT_DIR)
    mkdir(OUT_DIR)

    if out_type == OUT_TYPES[2]:
        if seg_type == SEG_TYPES[2]:
            mkdir(OUT_DIR + SEG_TYPES[0] + '_' + OUT_TYPES[0])
            mkdir(OUT_DIR + SEG_TYPES[0] + '_' + OUT_TYPES[1])
            mkdir(OUT_DIR + SEG_TYPES[1] + '_' + OUT_TYPES[0])
            mkdir(OUT_DIR + SEG_TYPES[1] + '_' + OUT_TYPES[1])
        else:
            mkdir(OUT_DIR + seg_type + '_' + OUT_TYPES[0])
            mkdir(OUT_DIR + seg_type + '_' + OUT_TYPES[1])
    else:
        if seg_type == SEG_TYPES[2]:
            mkdir(OUT_DIR + SEG_TYPES[0] + '_' + out_type)
            mkdir(OUT_DIR + SEG_TYPES[1] + '_' + out_type)
        else:
            mkdir(OUT_DIR + seg_type + '_' + out_type)

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
        if low_vol_cutoff <= region.area <= hi_vol_cutoff:
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

                if out_type == OUT_TYPES[2]:
                    out = segmented
                    out_name = f'{OUT_DIR}{SEG_TYPES[0]}_{OUT_TYPES[0]}/'+name
                    tifffile.imsave(out_name, out, description=out_metadata,
                                    software='Autocrop')

                    out = np.pad(np.max(segmented, 0),
                                 pad_width=max(segmented.shape[1:]) // 5,
                                 mode='constant')
                    out_name = f'{OUT_DIR}{SEG_TYPES[0]}_{OUT_TYPES[1]}/'+name
                    tifffile.imsave(out_name, out, description=out_metadata,
                                    software='Autocrop')
                else:
                    out = segmented if out_type == OUT_TYPES[0] else np.pad(
                        np.max(segmented, 0),
                        pad_width=max(segmented.shape[1:]) // 5,
                        mode='constant')
                    out_name = f'{OUT_DIR}{SEG_TYPES[0]}_{out_type}/'+name
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

                if out_type == OUT_TYPES[2]:
                    out = segmented
                    out_name = f'{OUT_DIR}{SEG_TYPES[1]}_{OUT_TYPES[0]}/'+name
                    tifffile.imsave(out_name, out, description=out_metadata,
                                    software='Autocrop')

                    out = np.pad(np.max(segmented, 0),
                                 pad_width=max(segmented.shape[1:]) // 5,
                                 mode='constant')
                    out_name = f'{OUT_DIR}{SEG_TYPES[1]}_{OUT_TYPES[1]}/'+name
                    tifffile.imsave(out_name, out, description=out_metadata,
                                    software='Autocrop')
                else:
                    out = segmented if out_type == OUT_TYPES[0] else np.max(
                        segmented, 0)
                    out_name = f'{OUT_DIR}{SEG_TYPES[1]}_{out_type}/'+name
                    tifffile.imsave(out_name, out, description=out_metadata,
                                    software='Autocrop')
