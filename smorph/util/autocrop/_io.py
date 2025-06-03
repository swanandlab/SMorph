import json
import uuid
from collections import defaultdict
from os import getcwd, makedirs, path
from pathlib2 import Path
from shutil import rmtree
from xml.etree import ElementTree

import czifile
import numpy as np
import roifile
import skimage.io as io
import tifffile
from aicsimageio import AICSImage
from ome_types.model import ome
from skimage import img_as_float, img_as_ubyte, exposure

from .util import _unwrap_polygon
from ...analysis._skeletal import _get_blobs


def etree_to_dict(t):
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v
                     for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(('@' + k, v)
                        for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[t.tag]['#text'] = text
        else:
            d[t.tag] = text
    return d


def _import_image(im_path, channel_interest):
    # image has to be converted to float for processing
    if im_path.split('.')[-1] in ('czi', 'lif', 'ims', 'lsm', 'tiff', 'nd2'):
        imfile = AICSImage(im_path)
        imfile.set_scene(0)
        im = imfile.get_image_data("ZYX", T=0, C=channel_interest)

        # if im_path.split('.')[-1] == 'czi': # assumes resolution unit is in meters
        #     scale = tuple(map(lambda a: a * 1e6, tuple(imfile.physical_pixel_sizes)))
        # else:
        # assumes resolution unit is in microns
        scale = tuple(imfile.physical_pixel_sizes)
        if isinstance(imfile.metadata, ElementTree.Element):
            metadata = etree_to_dict(imfile.metadata)
        elif isinstance(imfile.metadata, ome.OME):
            metadata = imfile.metadata.dict()
        else:
            metadata = imfile.metadata
    else:
        im = np.squeeze(io.imread(im_path))
        if im.ndim > 3:
            im = im[:, :, :, channel_interest]
        scale = (1,1,1)
        metadata = {}

    im = img_as_float(im)
    im = (im - im.min()) / (im.max() - im.min())
    return im, scale, metadata


def imread(im_path, ref_path=None, channel_interest=0):
    """Loads the image & scale.

    - Tested on: CZI, IMS, LIF, LSM, TIFF, ND2.

    Parameters
    ----------
    im_path : str
        Path to the confocal tissue image.
    ref_path : str
        Path to the reference image to whole exposure level
        `im_path` would be standardized.
    channel_interest : int
        Channel of interest containing image data to be processed,
        by default 0

    """
    # image has to be converted to float for processing
    im, scale, metadata = _import_image(im_path, channel_interest)

    im = im if im.ndim == 3 else np.expand_dims(im, 0)

    if not(ref_path in (None, '')):
        ref_im = _import_image(ref_path, channel_interest)[0]
        im = exposure.match_histograms(im, ref_im)

    return im, scale, metadata


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

    DIR = path.join(getcwd(), '/Autocropped/')
    makedirs(DIR, exist_ok=True)

    IMAGE_NAME = '.'.join(path.basename(img_path).split('.')[:-1])
    OUT_DIR = path.join(DIR, IMAGE_NAME + \
                f'{"" if roi_name == "" else "-" + str(roi_name)}/')
    # if path.exists(OUT_DIR) and path.isdir(OUT_DIR):  # mandatory new dir
    #     rmtree(OUT_DIR)
    # mkdir(OUT_DIR)
    makedirs(OUT_DIR, exist_ok=True)

    if out_type == OUT_TYPES[2]:
        if seg_type == SEG_TYPES[2]:
            makedirs(path.join(OUT_DIR, SEG_TYPES[0] + '_' + OUT_TYPES[0]), exist_ok=True)
            makedirs(path.join(OUT_DIR, SEG_TYPES[0] + '_' + OUT_TYPES[1]), exist_ok=True)
            makedirs(path.join(OUT_DIR, SEG_TYPES[1] + '_' + OUT_TYPES[0]), exist_ok=True)
            makedirs(path.join(OUT_DIR, SEG_TYPES[1] + '_' + OUT_TYPES[1]), exist_ok=True)
        else:
            makedirs(path.join(OUT_DIR, seg_type + '_' + OUT_TYPES[0]), exist_ok=True)
            makedirs(path.join(OUT_DIR, seg_type + '_' + OUT_TYPES[1]), exist_ok=True)
    else:
        if seg_type == SEG_TYPES[2]:
            makedirs(path.join(OUT_DIR, SEG_TYPES[0] + '_' + out_type), exist_ok=True)
            makedirs(path.join(OUT_DIR, SEG_TYPES[1] + '_' + out_type), exist_ok=True)
        else:
            makedirs(path.join(OUT_DIR, seg_type + '_' + out_type), exist_ok=True)

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
            out_name = f'{OUT_DIR}/'+name

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
        makedirs(RES_DIR, exist_ok=True)
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

def get_image_metadata_summary(im_path):
    """
    Get a comprehensive summary of image metadata without loading the full image.
    
    Parameters
    ----------
    im_path : str
        Path to the microscopy image file
        
    Returns
    -------
    dict
        Dictionary containing metadata summary including:
        - file_format: Detected file format
        - num_channels: Number of channels
        - image_shape: Image dimensions
        - physical_pixel_sizes: Physical pixel sizes
        - channels_info: List of channel information
    """
    try:
        from aicsimageio import AICSImage
        from xml.etree import ElementTree
        from ome_types.model import ome
        
        # Load image metadata only
        imfile = AICSImage(im_path)
        
        # Basic image information
        summary = {
            'file_path': im_path,
            'file_format': im_path.split('.')[-1].lower(),
            'num_channels': imfile.dims.C,
            'image_shape': {
                'X': imfile.dims.X,
                'Y': imfile.dims.Y, 
                'Z': imfile.dims.Z,
                'T': imfile.dims.T,
                'C': imfile.dims.C
            },
            'physical_pixel_sizes': {
                'X': imfile.physical_pixel_sizes.X,
                'Y': imfile.physical_pixel_sizes.Y,
                'Z': imfile.physical_pixel_sizes.Z
            }
        }
        
        # Process metadata
        if isinstance(imfile.metadata, ElementTree.Element):
            metadata = etree_to_dict(imfile.metadata)
        elif isinstance(imfile.metadata, ome.OME):
            metadata = imfile.metadata.dict()
        else:
            metadata = imfile.metadata
            
        # Extract channel information using enhanced metadata utilities
        try:
            from .metadata_utils import get_channel_summary
            channels_info = get_channel_summary(metadata, summary['file_format'])
            summary['channels_info'] = channels_info
        except ImportError:
            summary['channels_info'] = []
            
        return summary
        
    except Exception as e:
        return {
            'file_path': im_path,
            'error': str(e),
            'file_format': im_path.split('.')[-1].lower() if '.' in im_path else 'unknown'
        }


def compare_image_metadata(im_paths):
    """
    Compare metadata across multiple images for batch processing insights.
    
    Parameters
    ----------
    im_paths : list of str
        List of image file paths
        
    Returns
    -------
    dict
        Dictionary containing comparison results
    """
    if not isinstance(im_paths, (list, tuple)):
        im_paths = [im_paths]
        
    summaries = []
    for path in im_paths:
        summary = get_image_metadata_summary(path)
        summaries.append(summary)
    
    # Analyze commonalities and differences
    comparison = {
        'num_images': len(summaries),
        'file_formats': list(set(s.get('file_format', 'unknown') for s in summaries)),
        'common_channels': None,
        'pixel_size_ranges': {},
        'shape_ranges': {},
        'summaries': summaries
    }
    
    # Find common channel configurations
    if summaries and 'channels_info' in summaries[0]:
        # Check if all images have same number of channels
        channel_counts = [len(s.get('channels_info', [])) for s in summaries]
        if len(set(channel_counts)) == 1 and channel_counts[0] > 0:
            # Extract common channel properties
            common_channels = []
            for ch_idx in range(channel_counts[0]):
                ch_names = []
                ex_wavelengths = []
                em_wavelengths = []
                
                for summary in summaries:
                    channels = summary.get('channels_info', [])
                    if ch_idx < len(channels):
                        ch = channels[ch_idx]
                        ch_names.append(ch.get('name', 'Unknown'))
                        if 'excitation_wavelength' in ch:
                            ex_wavelengths.append(ch['excitation_wavelength'])
                        if 'emission_wavelength' in ch:
                            em_wavelengths.append(ch['emission_wavelength'])
                
                common_ch = {
                    'index': ch_idx,
                    'names': list(set(ch_names)),
                    'excitation_range': (min(ex_wavelengths), max(ex_wavelengths)) if ex_wavelengths else None,
                    'emission_range': (min(em_wavelengths), max(em_wavelengths)) if em_wavelengths else None
                }
                common_channels.append(common_ch)
                
            comparison['common_channels'] = common_channels
    
    # Analyze pixel size ranges
    x_sizes = [s.get('physical_pixel_sizes', {}).get('X') for s in summaries if s.get('physical_pixel_sizes', {}).get('X')]
    y_sizes = [s.get('physical_pixel_sizes', {}).get('Y') for s in summaries if s.get('physical_pixel_sizes', {}).get('Y')]
    z_sizes = [s.get('physical_pixel_sizes', {}).get('Z') for s in summaries if s.get('physical_pixel_sizes', {}).get('Z')]
    
    if x_sizes:
        comparison['pixel_size_ranges']['X'] = (min(x_sizes), max(x_sizes))
    if y_sizes:
        comparison['pixel_size_ranges']['Y'] = (min(y_sizes), max(y_sizes))
    if z_sizes:
        comparison['pixel_size_ranges']['Z'] = (min(z_sizes), max(z_sizes))
    
    return comparison


def print_metadata_comparison(comparison):
    """
    Print a formatted comparison of image metadata.
    
    Parameters
    ----------
    comparison : dict
        Result from compare_image_metadata
    """
    print(f"=== Metadata Comparison for {comparison['num_images']} Images ===")
    print(f"File formats: {', '.join(comparison['file_formats'])}")
    
    if comparison['pixel_size_ranges']:
        print("\nPixel size ranges:")
        for dim, (min_val, max_val) in comparison['pixel_size_ranges'].items():
            print(f"  {dim}: {min_val:.4f} - {max_val:.4f} µm")
    
    if comparison['common_channels']:
        print(f"\nCommon channel structure ({len(comparison['common_channels'])} channels):")
        for ch in comparison['common_channels']:
            names = ', '.join(ch['names']) if len(ch['names']) > 1 else ch['names'][0]
            print(f"  Channel {ch['index']}: {names}")
            if ch['excitation_range']:
                ex_min, ex_max = ch['excitation_range']
                print(f"    Excitation: {ex_min}-{ex_max} nm")
            if ch['emission_range']:
                em_min, em_max = ch['emission_range']
                print(f"    Emission: {em_min}-{em_max} nm")
    
    print(f"\nIndividual summaries:")
    for i, summary in enumerate(comparison['summaries']):
        if 'error' in summary:
            print(f"  {i+1}. {summary['file_path']}: ERROR - {summary['error']}")
        else:
            shape = summary['image_shape']
            print(f"  {i+1}. {summary['file_path']}")
            print(f"     Shape: {shape['X']}x{shape['Y']}x{shape['Z']} ({shape['C']} channels)")
