import json
from os import listdir, path

import napari
import numpy as np
import roifile
import tifffile
from skimage.measure import regionprops, label
from skimage.segmentation import watershed

from ._io import export_cells


def _segment_clump(image, markers):
    mask = image > 0
    labels = watershed(-image, markers, mask=mask, watershed_line=True)
    return labels


def postprocess_segment(SOMA_SELECTED_DIR, reconstructed_labels):
    folder = SOMA_SELECTED_DIR
    parent_path = None
    roi_path = None

    for file in listdir(folder):
        if not file.startswith('.') and file.endswith(('.tif', '.tiff')):
            name = folder + '/' + file
            image = tifffile.TiffFile(name)
            metadata = image.pages[0].tags['ImageDescription'].value
            print(file)
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

                minz, miny, minx, maxz, maxy, maxx = metadata['bounds']
                reconstructed_labels[minz:maxz, miny:maxy, minx:maxx] += labels
            except Exception as e:
                print(e)
    reconstructed_labels = label(reconstructed_labels)
    return reconstructed_labels, parent_path, roi_path
