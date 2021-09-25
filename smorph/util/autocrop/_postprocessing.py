import json
from os import listdir, path

import napari
import numpy as np
import roifile
import tifffile
from skimage.measure import regionprops, label
from skimage.segmentation import watershed

from ._io import import_confocal_image, export_cells
from ._roi_extract import _load_ROI
from .core import _unwrap_polygon


def _segment_clump(image, markers):
    mask = image > 0
    labels = watershed(-image, markers, mask=mask, watershed_line=True)
    return labels


def postprocess_segment(SOMA_SELECTED_DIR, reconstructed_labels=None):
    folder = SOMA_SELECTED_DIR
    parent_path = None
    roi_path = None

    for file in listdir(folder):
        if not file.startswith('.') and file.endswith(('.tif', '.tiff')) and '_mip' not in file:
            name = folder + '/' + file
            image = tifffile.TiffFile(name)
            metadata = image.pages[0].tags['ImageDescription'].value
            # print(file)
            metadata = json.loads(metadata)

            try:
                # look for ROI in same directory
                roi = roifile.roiread(folder + '/' + \
                                        '.'.join(file.split('.')[:-1]) + '.roi')
                yx = roi.coordinates()[:, [1, 0]]
                z = roi.counter_positions - 1
                somas_coords = np.insert(yx, 0, z, axis=1).astype(int)
                im = image.asarray()

                markers = np.zeros(im.shape)
                for i in range(1, somas_coords.shape[0] + 1):
                    markers[tuple(somas_coords[i-1])] = i

                labels = _segment_clump(im, markers)
                parent_path = metadata['parent_image']
                roi_path = metadata['roi_path']

                if reconstructed_labels is None:
                    parent = import_confocal_image(parent_path)
                    reconstructed_labels = np.zeros(parent.shape)

                minz, miny, minx, maxz, maxy, maxx = metadata['bounds']
                linebuilder = _load_ROI(roi_path)
                X, Y = _unwrap_polygon(linebuilder)
                min_x, max_x = int(min(X)), int(max(X) + 1)
                min_y, max_y = int(min(Y)), int(max(Y) + 1)
                miny += min_y
                maxy += min_y
                minx += min_x
                maxx += min_x
                reconstructed_labels[minz:maxz, miny:maxy, minx:maxx] += labels
            except Exception as e:
                print(e)
    reconstructed_labels = label(reconstructed_labels)
    return reconstructed_labels, parent_path, roi_path


def manual_postprocess(SOMA_SELECTED_DIR, reconstructed_seg=None):
    folder = SOMA_SELECTED_DIR
    parent_path = None
    roi_path = None
    somas_est = []

    for file in listdir(folder):
        if not file.startswith('.') and file.endswith(('.tif', '.tiff')) and '_mip' not in file:
            name = folder + '/' + file
            image = tifffile.TiffFile(name)
            metadata = image.pages[0].tags['ImageDescription'].value
            # print(file)
            metadata = json.loads(metadata)

            try:
                # look for ROI in same directory
                roi = roifile.roiread(folder + '/' + \
                                        '.'.join(file.split('.')[:-1]) + '.roi')
                yx = roi.coordinates()[:, [1, 0]]
                z = roi.counter_positions - 1
                somas_coords = np.insert(yx, 0, z, axis=1).astype(int)
                im = image.asarray()

                parent_path = metadata['parent_image']
                roi_path = metadata['roi_path']

                if reconstructed_seg is None:
                    parent = import_confocal_image(parent_path)
                    reconstructed_seg = np.zeros(parent.shape)

                minz, miny, minx, maxz, maxy, maxx = metadata['bounds']
                linebuilder = _load_ROI(roi_path)
                X, Y = _unwrap_polygon(linebuilder)
                min_x, max_x = int(min(X)), int(max(X) + 1)
                min_y, max_y = int(min(Y)), int(max(Y) + 1)
                miny += min_y
                maxy += min_y
                minx += min_x
                maxx += min_x
                reconstructed_seg[minz:maxz, miny:maxy, minx:maxx] += im

                somas_coords = np.array(somas_coords) + np.array([[minz, miny, minx]])
                somas_est.extend(somas_coords)
            except Exception as e:
                print(e)

    return reconstructed_seg, parent_path, roi_path, np.array(somas_est)
