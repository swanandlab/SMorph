import json
from os import listdir

import napari
import numpy as np
import tifffile
import smorph.util.autocrop as ac


try:
    on_colab = 'google.colab' in str(get_ipython())
except NameError:
    on_colab = False


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
        for file in listdir(group):
            if not file.startswith('.'):  # skip hidden files
                name = group + '/' + file
                metadata = tifffile.TiffFile(name).pages[
                               0].tags['ImageDescription'].value
                metadata = json.loads(metadata)
                if 'cluster_label' not in metadata.keys():
                    continue
                metadata['name'] = name

                if metadata['parent_image'] in dataset:
                    if (
                        ('roi_name' in metadata.keys())
                        and ('roi' in metadata.keys())
                    ):
                        if type(dataset[metadata['parent_image']][0]) == list:
                            stored = False
                            for i in range(
                                len(dataset[metadata['parent_image']])
                            ):
                                if (
                                    dataset[metadata['parent_image']][i][0][
                                        'roi_name'] == metadata['roi_name']
                                ):
                                    dataset[metadata['parent_image']][
                                        i].append(metadata)
                                    stored = True

                            if not stored:
                                dataset[metadata['parent_image']].append(
                                    [metadata])
                        elif (
                            dataset[metadata['parent_image']][0][
                                'roi_name'] == metadata['roi_name']
                        ):
                            dataset[metadata['parent_image']].append(metadata)
                        else:
                            dataset[metadata['parent_image']] = [
                                dataset[metadata['parent_image']], [metadata]]
                    else:
                        dataset[metadata['parent_image']].append(metadata)
                else:
                    dataset[metadata['parent_image']] = [metadata]

    return dataset


def _model_image_with_labels(viewer, img, name, data):
    viewer.add_image(
        data=img,
        colormap='inferno',
        name=name
    )

    pts, cluster_labels = [], []
    for cell_data in data:
        if 'roi' in cell_data.keys():
            cell_data['centroid'][1] += cell_data['roi'][0]
            cell_data['centroid'][2] += cell_data['roi'][1]
            cell_data['bounds'][1] += cell_data['roi'][0]
            cell_data['bounds'][2] += cell_data['roi'][1]
            cell_data['bounds'][4] += cell_data['roi'][0]
            cell_data['bounds'][5] += cell_data['roi'][1]
        pts.append(cell_data['centroid'])
        cluster_labels.append(cell_data['cluster_label'])
    props = {'cluster_labels': np.array(cluster_labels)}

    viewer.add_points(pts, edge_color='transparent',
                      face_color='transparent',
                      properties=props, text='cluster_labels',
                      name=name+'_cluster_labels')


def label_clusters_spatially(groups_folders):
    dataset = _read_groups_folders(groups_folders)

    if on_colab:
        return

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.dims.ndisplay = 3
        for tissue_img in dataset.keys():
            IMG_NAME = '.'.join(tissue_img.split('/')[-1].split('.')[:-1])
            img = ac.import_confocal_image(tissue_img)
            if type(dataset[tissue_img][0]) == dict:
                _model_image_with_labels(viewer, img, IMG_NAME,
                                            dataset[tissue_img])
            else:
                for cell_data in dataset[tissue_img]:
                    _model_image_with_labels(
                        viewer, img,
                        (IMG_NAME if 'roi' not in cell_data[0].keys()
                            else IMG_NAME + '_' + cell_data[0]['roi_name']),
                        cell_data
                    )


def identify_cell_in_tissue(img_path):
    if on_colab:
        return

    # TODO: try for multiple cells at once too
    with napari.gui_qt():
        IMG_NAME = '.'.join(img_path.split('/')[-1].split('.')[:-1])
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

        img = ac.import_confocal_image(cell_data['parent_image'])

        if 'roi' in cell_data.keys():
            cell_data['centroid'][1] += cell_data['roi'][0]
            cell_data['centroid'][2] += cell_data['roi'][1]
            cell_data['bounds'][1] += cell_data['roi'][0]
            cell_data['bounds'][2] += cell_data['roi'][1]
            cell_data['bounds'][4] += cell_data['roi'][0]
            cell_data['bounds'][5] += cell_data['roi'][1]

        viewer = napari.Viewer(ndisplay=3)
        viewer.add_image(
            data=img,
            colormap='inferno',
            name='tissue_image_'+IMG_NAME
        )

        # segmented image
        minz, miny, minx, maxz, maxy, maxx = cell_data['bounds']
        mask = np.empty(img.shape, dtype=bool)
        mask[minz:maxz+1, miny:maxy+1, minx:maxx+1] = 1
        viewer.add_image(
            data=img*mask,
            colormap='inferno',
            name='bbox_segmented'
        )

        # pts, cluster_labels = [cell_data['centroid']], ['1']
        # props = {'cluster_labels': np.array(cluster_labels)}
        # viewer.add_points(pts, edge_color='transparent',
        #                   face_color='transparent',
        #                   properties=props, text='cluster_labels',
        #                   name=IMG_NAME + '_cluster_labels')

        viewer.add_shapes([[  # back
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
            ]],
            face_color='transparent',
            edge_color='green',
            name='bbox'
        )
        viewer.camera.center = cell_data['centroid']


def export_results():
    return None
