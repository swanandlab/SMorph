import json
from os import listdir, path

import napari
import numpy as np
import roifile
import tifffile
import smorph.util.autocrop as ac


def _read_images(file_names, prefix=''):
    dataset = {}
    """
    {
        parent: {
            roi: {
                feats
            }
        }
    }
    """
    tree = {}
    for i, fname in enumerate(file_names):
        if fname.split('.')[-1] == 'tif':
            with tifffile.TiffFile(fname) as f:
                metadata = f.pages[0].tags['ImageDescription'].value
                metadata = json.loads(metadata)

                if metadata['parent_image'] in dataset:
                    if all(x in metadata.keys() for x in ('roi_name', 'roi')):
                        if not isinstance(dataset[metadata['parent_image']], dict):
                            dataset[metadata['parent_image']] = {}
                            tree[metadata['parent_image']] = {}
                        if metadata['roi_name'] not in dataset[metadata['parent_image']].keys():
                            dataset[metadata['parent_image']][metadata['roi_name']] = {}
                            tree[metadata['parent_image']][metadata['roi_name']] = {}

                        dataset[metadata['parent_image']][metadata['roi_name']][fname] = metadata
                        tree[metadata['parent_image']][metadata['roi_name']][fname] = {}
                else:
                    dataset[metadata['parent_image']] = {
                        metadata['roi_name']: {
                            fname: metadata
                        }
                    }
                    tree[metadata['parent_image']] = {
                        metadata['roi_name']: {
                            fname: {}
                        }
                    }
    return tree, dataset


def _read_groups_folders(groups_folders):
    """Synchronously read list of folders for images' metadata.

    Parameters
    ----------
    groups_folders : list
        A list of strings containing path of each folder with autocropped
        image dataset.

    Returns
    -------
    dataset : dict
        A Python dict of confocal image data paths and associated cell
        metadata.

    """
    dataset = {}

    for group in groups_folders:
        dataset.update(_read_images(listdir(group), group + '/'))

    return dataset


def _get_roi_scaled_points(data):
    bounds_pts, centroid_pts, cluster_labels = [], [], []
    for cell_data in data:
        centroid = cell_data['centroid'].copy()
        bounds = cell_data['bounds'].copy()
        if 'roi' in cell_data.keys():
            centroid[1] += cell_data['roi'][0]
            centroid[2] += cell_data['roi'][1]
            bounds[1] += cell_data['roi'][0]
            bounds[2] += cell_data['roi'][1]
            bounds[4] += cell_data['roi'][0]
            bounds[5] += cell_data['roi'][1]
        centroid_pts.append(centroid)
        bounds_pts.append(bounds)
        if 'cluster_label' in cell_data.keys():
            cluster_labels.append(cell_data['cluster_label'])
    return bounds_pts, centroid_pts, cluster_labels


def _model_image_with_labels(viewer, img, name, data):
    name = path.basename(r'{}'.format(name))

    _, centroid_pts, cluster_labels = _get_roi_scaled_points(data)
    props = {'cluster_labels': np.array(cluster_labels)}

    viewer.add_points(centroid_pts, edge_color='transparent',
                      face_color='transparent', properties=props,
                      text='cluster_labels', name='cluster_labels_'+name)


def label_clusters_spatially(groups_folders):
    dataset = _read_groups_folders(groups_folders)

    with napari.gui_qt():
        viewer = napari.Viewer(ndisplay=3)
        for tissue_img in dataset.keys():
            IMG_NAME = path.basename(r'{}'.format(tissue_img))
            img = ac.imread(tissue_img)

            if type(dataset[tissue_img][0]) == dict:
                PARENT_NAME = dataset[tissue_img][0]['parent_image']
                PARENT_NAME = path.basename(r'{}'.format(PARENT_NAME))
                viewer.add_image(img, colormap='inferno',
                                 name='tissue_'+PARENT_NAME)
                _model_image_with_labels(viewer, img, IMG_NAME,
                                         dataset[tissue_img])
            else:
                for cell_data in dataset[tissue_img]:
                    PARENT_NAME = cell_data[0]['parent_image']
                    PARENT_NAME = path.basename(r'{}'.format(PARENT_NAME))
                    viewer.add_image(
                        img, colormap='inferno',
                        name='tissue_' + (
                            PARENT_NAME if 'roi' not in cell_data[0].keys()
                            else IMG_NAME+'_'+cell_data[0]['roi_name'])
                    )
                    _model_image_with_labels(
                        viewer, img,
                        IMG_NAME,
                        cell_data
                    )


def _make_bounding_cuboid(bounds):
    minz, miny, minx, maxz, maxy, maxx = bounds

    cuboid_faces = [[  # back
        [minz, miny, minx],
        [minz, miny, maxx],
        [minz, maxy, maxx],
        [minz, maxy, minx]
    ], [  # left
        [minz, miny, minx],
        [minz, maxy, minx],
        [maxz, maxy, minx],
        [maxz, miny, minx]
    ], [  # right
        [minz, miny, maxx],
        [minz, maxy, maxx],
        [maxz, maxy, maxx],
        [maxz, miny, maxx]
    ], [  # front
        [maxz, maxy, maxx],
        [maxz, miny, maxx],
        [maxz, miny, minx],
        [maxz, maxy, minx]
    ]]

    return cuboid_faces


def _identify_cell_in_tissue(img_path):
    with napari.gui_qt():
        cell_data = tifffile.TiffFile(img_path).pages[
                        0].tags['ImageDescription'].value

        try:  # error handling
            cell_data = json.loads(cell_data)
        except json.decoder.JSONDecodeError:
            print('Ensure image is autocropped')
            return
        if 'parent_image' not in cell_data:
            print('Failed to load parent image!')
            return

        img = ac.imread(cell_data['parent_image'])

        (
            bounds, centroid_pts, cluster_labels
        ) = _get_roi_scaled_points([cell_data])

        viewer = napari.Viewer(ndisplay=3)
        viewer.add_image(img, colormap='inferno',
                         name='tissue_'+path.basename(r'{}'.format(
                                cell_data['parent_image'])))

        # segmented image
        minz, miny, minx, maxz, maxy, maxx = bounds[0]
        mask = np.empty(img.shape, dtype=bool)
        mask[minz:maxz+1, miny:maxy+1, minx:maxx+1] = 1
        viewer.add_image(img*mask, colormap='inferno',
                         blending='additive', name='bbox_segmented')

        if len(cluster_labels) == 0:
            cluster_labels = [1]
        props = {'cluster_labels': np.array(cluster_labels)}
        viewer.add_points(centroid_pts, edge_color='transparent',
                          face_color='transparent',
                          properties=props, text='cluster_labels',
                          name='cluster_label')

        viewer.add_shapes(
            _make_bounding_cuboid(cell_data['bounds']),
            face_color='transparent',
            edge_color='green',
            blending='additive',
            name='bbox'
        )
        # to view roi polygon
        # viewer.add_shapes([roifile.roiread(cell_data['roi_path']).coordinates()[:, [1, 0]]],
        #                   shape_type='polygon')
        viewer.camera.center = cell_data['centroid']


def _label_cells_in_tissue(
    viewer,
    img,
    bounds,
    centroid_pts,
    cluster_labels,
    name
):
    name = path.basename(r'{}'.format(name))

    for i in range(len(bounds)):
        # segmented image
        minz, miny, minx, maxz, maxy, maxx = bounds[i]
        mask = np.empty(img.shape, dtype=bool)
        mask[minz:maxz+1, miny:maxy+1, minx:maxx+1] = 1
        viewer.add_image(
            data=img*mask,
            blending='additive',
            colormap='inferno',
            name='bbox_segmented_' + name
        )

        viewer.add_shapes(
            _make_bounding_cuboid(bounds[i]),
            face_color='transparent',
            edge_color='green',
            blending='additive',
            name='bbox_' + name
        )

    if len(cluster_labels) == 0:
        cluster_labels = [1]*len(bounds)
    props = {'cluster_labels': np.array(cluster_labels)}
    viewer.add_points(centroid_pts, edge_color='transparent',
                      face_color='transparent',
                      properties=props, text='cluster_labels',
                      name='cluster_labels_'+name)


def identify_cells_in_tissue(img_paths):
    if type(img_paths) is str:
        return _identify_cell_in_tissue(img_paths)
    if len(img_paths) == 1:
        return _identify_cell_in_tissue(img_paths[0])

    with napari.gui_qt():
        dataset = _read_images(img_paths)

        viewer = napari.Viewer(ndisplay=3)
        for tissue_img in dataset.keys():
            PARENT_NAME = path.basename(r'{}'.format(tissue_img))
            IMG_NAME = '.'.join(tissue_img.split('/')[-1].split('.')[:-1])
            img = ac.imread(tissue_img)
            if type(dataset[tissue_img][0]) == dict:
                (
                    bounds, centroid_pts, cluster_labels
                ) = _get_roi_scaled_points(dataset[tissue_img])

                viewer.add_image(img, colormap='inferno',
                                 name='tissue_' + PARENT_NAME)
                _label_cells_in_tissue(viewer, img, bounds, centroid_pts,
                                       cluster_labels, IMG_NAME)
            else:
                for cells_data in dataset[tissue_img]:
                    (
                        bounds, centroid_pts, cluster_labels
                    ) = _get_roi_scaled_points(cells_data)

                    viewer.add_image(
                        img, colormap='inferno',
                        name='tissue_' + (
                            PARENT_NAME if 'roi' not in cells_data[0].keys()
                            else IMG_NAME + '_' + cells_data[0]['roi_name'])
                    )
                    _label_cells_in_tissue(viewer, img, bounds, centroid_pts,
                                           cluster_labels, IMG_NAME)


def export_results():
    return None
