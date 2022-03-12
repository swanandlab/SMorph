import napari

import winsound

import json
import re
from os import (
    getcwd,
    listdir,
    path,
)

import PyQt5
import matplotlib as mpl

import czifile
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import superqt
import zarr

COLOR = 'white'
mpl.rcParams['axes.labelcolor'] = COLOR
mpl.rcParams['xtick.color'] = COLOR
mpl.rcParams['ytick.color'] = COLOR


from pandas import DataFrame
from scipy import ndimage as ndi
from scipy.stats import sem
from magicgui import magicgui
from magicgui.widgets import ComboBox, FloatSlider, Label, Slider
from magicclass import magicclass, field, set_design, wrappers
from magicclass.widgets import Figure, TupleEdit
from matplotlib import gridspec
from skimage import filters
from skimage.measure import label
from vispy.geometry.rect import Rect

from .. import (
    core,
    _preprocess as preprocess,
)
from .._max_rect_in_poly import (
    get_maximal_rectangle,
)
from .._postprocessing import _segment_clump
from ..._image import THRESHOLD_METHODS
from ..._io import (
    df_to_csv,
    savefig,
)
from ....core.api import Groups
from .._roi_extract import (
    select_ROI,
    mask_ROI
)
from ..util import (
    only_name,
    _unwrap_polygon,
)


def _get_ncolors_map(ncolors, cmap_name='plasma_r'):
    cmap = {0: [0,0,0,0], None: [0,0,0,1]}  # initial colors in napari
    colors = plt.cm.get_cmap(cmap_name, ncolors).colors
    cmap.update(dict(zip(range(1, ncolors), colors)))
    return cmap

def longest_contiguous_nonzero(vals):
    """Return longest contiguous nonzero values in 1D array.
    """
    best, run = [], []
    start_best = end_best = start_run = 0

    nonzero_indices = np.nonzero(vals)[0]
    if len(nonzero_indices) == 1:
        start_best = end_best = nonzero_indices[0]
        return [vals[end_best]], end_best, end_best

    for i in range(1, len(vals) + 1):
        run.append(vals[i-1])

        if i == len(vals) or vals[i-1] != vals[i] or vals[i] == 0:
            if len(best) < len(run):
                best = run
                start_best = start_run
                end_best = i - 1

            run = []
            start_run = i

    return best, start_best, end_best


@magicclass(widget_type="none")
class RefineSegmentation:
    @magicclass(widget_type="groupbox")
    class RegionSelector:
        selected_region=field(int, options=dict(max=0))

        @selected_region.connect
        def select_region(self):
            parent = self.__magicclass_parent__
            pipe = parent.__magicclass_parent__.__magicclass_parent__.pipe
            pipe.n_region = self.selected_region.value
            pipe.PROPS = ['vol',
                # 'convex_area',
                # 'equivalent_diameter',
                # 'euler_number',
                # 'extent',
                # 'feret_diameter_max',
                # 'major_axis_length',
                # 'minor_axis_length',
                # 'solidity'
            ]

            minz, miny, minx, maxz, maxy, maxx = pipe.regions[pipe.n_region]['bbox']
            centroid = pipe.regions[pipe.n_region]['centroid']
            if pipe.SCALE is not None:
                centroid = centroid * np.array(pipe.SCALE)
                minz *= pipe.SCALE[0]; maxz *= pipe.SCALE[0]
                miny *= pipe.SCALE[1]; maxy *= pipe.SCALE[1]
                minx *= pipe.SCALE[2]; maxx *= pipe.SCALE[2]

            if self.parent_viewer.dims.ndisplay == 3:
                self.parent_viewer.camera.center = centroid
            elif self.parent_viewer.dims.ndisplay == 2:
                self.parent_viewer.dims.set_current_step(0, np.round(centroid[0]))
                self.parent_viewer.window.qt_viewer.view.camera.set_state(
                    {'rect': Rect(minx, miny, maxx-minx, maxy-miny)})

            data = '<table cellspacing="8">'
            for prop in pipe.PROPS:
                name = prop
                data += '<tr><td><b>' + name + '</b></td><td>' + str(
                    eval(f'pipe.regions[{pipe.n_region}]["{prop}"]')) + '</td></tr>'
            data += '</table>'
            parent.region_props.setText(data)

    @magicclass(widget_type="groupbox")
    class HighPassVolume:
        cutoff=Slider(max=0)
        def vol_cutoff_update(self):
            parent = self.__magicclass_parent__
            pipe = parent.__magicclass_parent__.__magicclass_parent__.pipe
            # self.parent_viewer.add_image(pipe.impreprocessed, scale=pipe.SCALE)
            # self.parent_viewer.add_labels(pipe.labels, rendering='translucent', opacity=.5, scale=pipe.SCALE)
            pipe.LOW_VOLUME_CUTOFF = self.cutoff.value
            pipe.labels.fill(0)
            itr = 1
            filtered_regions = []
            for region in pipe.regions:
                if self.cutoff.value <= region['vol']:
                    minz, miny, minx, maxz, maxy, maxx = region['bbox']
                    pipe.labels[minz:maxz, miny:maxy, minx:maxx] += region['image'] * itr
                    itr += 1
                    filtered_regions.append(region)
            self.parent_viewer.layers['labels'].data = pipe.labels.copy()
            pipe.regions = filtered_regions
            pipe.imsegmented = pipe.impreprocessed * (pipe.labels > 0)
            parent.RegionSelector.selected_region.range = (0, len(pipe.regions)-1)

            pipe.regions = sorted(pipe.regions, key=lambda region: region['vol'])

            reconstructed_labels = np.zeros(pipe.impreprocessed.shape, dtype=int)
            for itr in range(len(pipe.regions)):
                minz, miny, minx, maxz, maxy, maxx = pipe.regions[itr]['bbox']
                reconstructed_labels[minz:maxz, miny:maxy, minx:maxx] += pipe.regions[itr]['image'] * (itr + 1)
            pipe.labels = reconstructed_labels
            pipe.imbinary = reconstructed_labels > 0
            pipe.imsegmented = pipe.impreprocessed * pipe.imbinary
            self.parent_viewer.layers['segmented'].data = pipe.imsegmented.copy()

            try:
                parent.InteractiveSegmentation
            except AttributeError:
                parent.append(InteractiveSegmentation())

            somas_estimates = core.approximate_somas(pipe.imsegmented, pipe.regions)
            pipe.somas_estimates = somas_estimates
            pt_layer = self.parent_viewer.add_points(somas_estimates, face_color='red',
                edge_width=0, blending='translucent',
                opacity=.6, size=5, name='somas_coords', scale=pipe.SCALE
                )
            ray_layer = self.parent_viewer.add_points(
                np.zeros(shape=(1, pipe.imoriginal.ndim)), name='selected point',
                scale=pipe.SCALE, size=5
                )
            selector = np.zeros_like(pipe.impreprocessed)
            selector_layer = self.parent_viewer.add_image(selector, blending='additive', scale=pipe.SCALE)
            self.parent_viewer.layers['labels'].color_mode = 'auto'

            # callback function, called on mouse click when volume layer is active
            @selector_layer.mouse_drag_callbacks.append
            def on_click(layer, event):
                near_point, far_point = layer.get_ray_intersections(
                    event.position,
                    event.view_direction,
                    event.dims_displayed
                )

                if (near_point is not None) and (far_point is not None):
                    ray_points = np.linspace(
                        near_point, far_point, int(np.max(far_point-near_point)),
                        endpoint=False
                        )
                    ray_points = np.round(ray_points).astype(int)  # pixel-coords

                    # select pt at mid pt of most intersecting label at line of sight
                    ray_label_vals = [self.parent_viewer.layers['labels'].data[tuple(coords)] for coords in ray_points]
                    subarr, start, end = longest_contiguous_nonzero(ray_label_vals)

                    if subarr[0] == 0:
                        start = 0; end = len(ray_points) -1
                    pt_index = end if start == end else start + (end - start) // 2

                    mid_pixel = np.take(ray_points, pt_index, axis=0)

                    if ray_points.shape[1] != 0:
                        ray_layer.data = [mid_pixel]


@magicclass(widget_type="groupbox", visible=True)
class InteractiveSegmentation:
    @magicclass(widget_type="tabbed")
    class ClumpSep:
        @magicclass(widget_type="none")
        class Watershed:
            prompt = field(int, widget_type="Label", options={"value": 
                "Edit the `somas_coords` layer using layer controls"})

            def add_selected_point_seed(self):
                coord_to_add = self.parent_viewer.layers['selected point'].data[0]
                self.parent_viewer.layers['somas_coords'].add(coord_to_add)

            def seeded_watershed(self):
                grandparent = self.__magicclass_parent__.__magicclass_parent__
                pipe = grandparent.__magicclass_parent__.__magicclass_parent__.__magicclass_parent__.pipe
                SEARCH_LAYER = "somas_coords"
                layer_names = [layer.name for layer in self.parent_viewer.layers]
                somas_estimates = np.unique(self.parent_viewer.layers[SEARCH_LAYER].data, axis=0)
                filtered_regions, residue = [], []
                separated_clumps = []

                itr = 0
                for region in pipe.regions:
                    minz, miny, minx, maxz, maxy, maxx = region['bbox']
                    ll = np.array([minz, miny, minx])  # lower-left
                    ur = np.array([maxz, maxy, maxx]) - 1  # upper-right
                    inidx = np.all(np.logical_and(ll <= somas_estimates, somas_estimates <= ur), axis=1)
                    somas_coords = somas_estimates[inidx].astype(np.int64)

                    if len(somas_coords) == 0:
                        print('Deleted region:', itr)
                        residue.append(region)
                    elif len(np.unique(somas_coords.astype(int), axis=0)) > 1:  # clumpSep
                        somas_coords = somas_coords.astype(int)
                        somas_coords -= ll
                        im = pipe.impreprocessed[minz:maxz, miny:maxy, minx:maxx].copy()
                        im[~region['image']] = 0
                        markers = np.zeros(region['image'].shape)

                        somas_coords = np.array([x for x in somas_coords if region['image'][tuple(x)] > 0])

                        if len(somas_coords) == 0:  # no marked point ROI
                            print('Deleted region:', itr)

                        if somas_coords.shape[0] == 1:
                            filtered_regions.append(region)
                            continue

                        for i in range(somas_coords.shape[0]):
                            markers[tuple(somas_coords[i])] = i + 1
                            separated_clumps.append(somas_coords[i])

                        labels = _segment_clump(im, markers)
                        separated_regions = core.arrange_regions(labels)
                        for r in separated_regions:
                            r['centroid'] = (minz + r['centroid'][0], miny + r['centroid'][1], minx + r['centroid'][2])
                            r['bbox'] = (minz + r['bbox'][0], miny + r['bbox'][1], minx + r['bbox'][2],
                                minz + r['bbox'][3], miny + r['bbox'][4], minx + r['bbox'][5]
                                )
                            # r.slice = (slice(minz + r.bbox[0], minz + r.bbox[3]),
                            #            slice(miny + r.bbox[1], miny + r.bbox[4]),
                            #         slice(minx + r.bbox[2], minx + r.bbox[5]))
                        print('Splitted clump region:', itr)
                        filtered_regions.extend(separated_regions)
                    else:
                        filtered_regions.append(region)
                    itr += 1

                pipe.filtered_regions = filtered_regions
                pipe.residue = residue
                watershed_results = np.zeros_like(pipe.impreprocessed, dtype=int)
                for itr in range(len(filtered_regions)):
                    minz, miny, minx, maxz, maxy, maxx = filtered_regions[itr]['bbox']
                    watershed_results[minz:maxz, miny:maxy, minx:maxx] += filtered_regions[itr]['image'] * (itr + 1)
                self.parent_viewer.add_labels(
                    watershed_results, opacity=.7,
                    scale=pipe.SCALE, rendering='translucent'
                    )

            def accept_changes(self):
                grandparent = self.__magicclass_parent__.__magicclass_parent__
                greatgrandparent = grandparent.__magicclass_parent__
                pipe = greatgrandparent.__magicclass_parent__.__magicclass_parent__.pipe
                layer_names = [layer.name for layer in self.parent_viewer.layers]
                pipe.filtered_regions = sorted(pipe.filtered_regions, key=lambda region: region['vol'])

                # After all changes (for reproducibility)
                final_soma = np.unique(self.parent_viewer.layers['somas_coords'].data, axis=0)

                pipe.FINAL_PT_ESTIMATES = final_soma

                pipe.regions = regions = pipe.filtered_regions
                watershed_results = np.zeros_like(pipe.impreprocessed, dtype=int)
                for itr in range(len(regions)):
                    minz, miny, minx, maxz, maxy, maxx = regions[itr]['bbox']
                    watershed_results[minz:maxz, miny:maxy, minx:maxx] += regions[itr]['image'] * (itr + 1)

                pipe.labels = watershed_results
                pipe.imsegmented = pipe.impreprocessed * (watershed_results > 0)

                # Update changes for viewer
                self.parent_viewer.layers.remove('watershed_results')

                greatgrandparent.RegionSelector.selected_region.range = (0, len(regions)-1)

                if 'labels' in layer_names:
                    self.parent_viewer.layers['labels'].data = watershed_results
                if 'segmented' in layer_names:
                    self.parent_viewer.layers['segmented'].data = pipe.imsegmented.copy()
            ################

        @magicclass(widget_type="none")
        class Manual:
            prompt = field(int, widget_type="Label", options={"value": 
                "Edit the `labels` layer using layer controls."})

            def reset_changes(self):
                grandparent = self.__magicclass_parent__.__magicclass_parent__
                pipe = grandparent.__magicclass_parent__.__magicclass_parent__.__magicclass_parent__.pipe
                self.parent_viewer.layers["labels"].data = pipe.labels

            def apply_changes(self):
                grandparent = self.__magicclass_parent__.__magicclass_parent__
                pipe = grandparent.__magicclass_parent__.__magicclass_parent__.__magicclass_parent__.pipe
                refined = label(self.parent_viewer.layers['labels'].data)
                self.parent_viewer.add_labels(refined, scale=pipe.SCALE,
                    name="manual_labels", rendering='translucent'
                    )

            def accept_changes(self):
                grandparent = self.__magicclass_parent__.__magicclass_parent__
                greatgrandparent = grandparent.__magicclass_parent__
                pipe = greatgrandparent.__magicclass_parent__.__magicclass_parent__.pipe
                layer_names = [layer.name for layer in self.parent_viewer.layers]
                refined = self.parent_viewer.layers["manual_labels"].data
                bin_refined = refined > 0

                pipe.labels = refined
                pipe.imsegmented = pipe.impreprocessed * bin_refined
                pipe.regions = core.arrange_regions(refined)

                # Update changes for viewer
                self.parent_viewer.layers.remove("manual_labels")

                greatgrandparent.RegionSelector.selected_region.range = (0, len(pipe.regions)-1)

                if 'labels' in layer_names:
                    self.parent_viewer.layers['labels'].data = refined.copy()
                if 'segmented' in layer_names:
                    self.parent_viewer.layers['segmented'].data = pipe.imsegmented.copy()

    
    def confirm_all(self):
        parent = self.__magicclass_parent__
        grandparent = parent.__magicclass_parent__
        greatgrandparent = grandparent.__magicclass_parent__
        pipe = grandparent.__magicclass_parent__.pipe
        viewer = self.parent_viewer

        refined = viewer.layers["labels"].data
        bin_refined = refined > 0
        pipe.labels = refined
        pipe.imsegmented = pipe.impreprocessed * bin_refined

        # After all changes (for reproducibility)
        label_diff = (pipe.imbinary).astype(np.int) - bin_refined.astype(np.int)
        pipe.label_diff = label_diff
        final_soma = np.unique(viewer.layers['somas_coords'].data, axis=0)
        pipe.FINAL_PT_ESTIMATES = final_soma

        # Update changes for viewer
        parent.RegionSelector.selected_region.range = (0, len(pipe.regions)-1)

        viewer.layers['segmented'].data = pipe.imsegmented.copy()

        # Pass the torch
        greatgrandparent.ExportCells.visible = True

        pipe.HIGH_VOLUME_CUTOFF = pipe.regions[-1]['vol']  # filter cell clusters
        pipe.n_region = 0

        try:
            greatgrandparent.ExportCells.bandpass_vol
            viewer.window.remove_dock_widget(
                viewer.window._dock_widgets["Bandpass Volume"])
        except AttributeError:
            pass

        bandpass_vol = superqt.QLabeledRangeSlider()
        bandpass_vol.setRange(0, pipe.regions[-1]['vol'])
        bandpass_vol.setOrientation(1)
        bandpass_vol.setValue([pipe.LOW_VOLUME_CUTOFF, pipe.HIGH_VOLUME_CUTOFF])
        bandpass_vol.setEdgeLabelMode(superqt.sliders._labeled.EdgeLabelMode.NoLabel)
        bandpass_vol.setContentsMargins(25, 5, 25, 5)
        for i in (0, 1):
            # vol_cutoff_slider.children()[i].setAlignment(PyQt5.QtCore.Qt.AlignCenter)
            bandpass_vol.children()[i].setFixedWidth(len(str(int(pipe.regions[-1]['vol']))) * 20)

        def vol_cutoff_update():
            pipe.imsegmented.fill(0)
            pipe.LOW_VOLUME_CUTOFF, pipe.HIGH_VOLUME_CUTOFF = bandpass_vol.value()
            for region in pipe.regions:
                if pipe.LOW_VOLUME_CUTOFF <= region['vol'] <= pipe.HIGH_VOLUME_CUTOFF:
                    minz, miny, minx, maxz, maxy, maxx = region['bbox']
                    segmented_cell = region['image'] * pipe.impreprocessed[minz:maxz, miny:maxy, minx:maxx]
                    segmented_cell = segmented_cell / (segmented_cell.max() - segmented_cell.min())
                    pipe.imsegmented[minz:maxz, miny:maxy, minx:maxx] += segmented_cell
            minz, miny, minx, maxz, maxy, maxx = pipe.regions[pipe.n_region]['bbox']
            viewer.layers['bandpassed'].data = pipe.imsegmented

        bandpass_vol.valueChanged.connect(vol_cutoff_update)
        greatgrandparent.ExportCells.bandpass_vol = bandpass_vol
        viewer.add_image(pipe.imsegmented, colormap='inferno', scale=pipe.SCALE, name='bandpassed')
        if "Bandpass Volume" not in list(viewer.window._dock_widgets.keys()):
            viewer.window.add_dock_widget(
                greatgrandparent.ExportCells.bandpass_vol,
                name="Bandpass Volume", area="bottom"
                )
            viewer.window._dock_widgets["Bandpass Volume"].setFixedHeight(80)
        greatgrandparent.current_index = 4
        vol_cutoff_update()



def _auto_params_deconv(pipe):
    impath = pipe.im_path.lower()
    if impath.split('.')[-1] == 'czi':
        czimeta = czifile.CziFile(impath).metadata(False)
        metadata = czimeta['ImageDocument']['Metadata']
        im_meta = metadata['Information']['Image']
        refr_index = im_meta['ObjectiveSettings']['RefractiveIndex']

        selected_channel = None
        for i in im_meta['Dimensions']['Channels']['Channel']:
            if i['ContrastMethod'] == 'Fluorescence':
                selected_channel = i
        ex_wavelen = selected_channel['ExcitationWavelength']
        em_wavelen = selected_channel['EmissionWavelength']

        selected_detector = None
        for i in metadata['Experiment']['ExperimentBlocks']['AcquisitionBlock'
            ]['MultiTrackSetup']['TrackSetup']['Detectors']['Detector']:  # [channel]['Detectors']['Detector']:
            if i['PinholeDiameter'] > 0:
                selected_detector = i
        pinhole_radius = selected_detector['PinholeDiameter'] / 2 * 1e6

        num_aperture = metadata['Information']['Instrument']['Objectives'][
            'Objective']['LensNA']
        # dim_r = metadata['Scaling']['Items']['Distance'][0]['Value'] * 1e6

        return dict(
            ex_wavelen = ex_wavelen,
            em_wavelen = em_wavelen,
            num_aperture = num_aperture,
            refr_index = refr_index,
            pinhole_radius = pinhole_radius
        )
    return None


@magicclass(widget_type="tabbed")
class Autocrop:
    pipe = None
    IN_DIR = OUT_DIR = IM_FILES = ROI_FILES = ROI_NAME = None

    @magicclass(widget_type="scrollable")
    class LoadDataset:
        @magicclass(widget_type="groupbox")
        class LoadDIR:
            IN_DIR = field('Datasets/Confocal/SAL,DMI, FLX ADN HALO_TREATMENT_28 DAYS/SAL_28_DAYS')
            OUT_DIR = field('Autocropped/')
            IM_FILTER = field('.+HILUS.+\.czi')
            IM_ROI_FILTER = field('.+HILUS.+DAY\.roi')
            ROI_NAME = field('HILUS')

            def load_directory(self):
                files = listdir(self.IN_DIR.value)

                IM_FILTER = self.IM_FILTER.value
                IM_ROI_FILTER = self.IM_ROI_FILTER.value
                ROI_NAME = self.ROI_NAME.value

                IM_FILES = list(filter(re.compile(IM_FILTER).match, files))
                ROI_FILES = list(filter(re.compile(IM_ROI_FILTER).match, files))

                parent = self.__magicclass_parent__
                grandparent = parent.__magicclass_parent__
                grandparent.IM_FILES = IM_FILES
                grandparent.ROI_FILES = ROI_FILES
                grandparent.IN_DIR = self.IN_DIR.value
                grandparent.OUT_DIR = self.OUT_DIR.value
                grandparent.ROI_NAME = ROI_NAME

                parent.LoadImage.IM_NUM.range = (0, len(IM_FILES)-1)
                parent.LoadImage._update_file()

        @magicclass(widget_type="groupbox")
        class LoadImage:
            IM_NUM = field(0)
            IMAGE = field(str)
            ROI = field(str)

            REF_IMAGE = field('Datasets/Confocal/SAL,DMI, FLX ADN HALO_TREATMENT_28 DAYS/FLX_28_DAYS/FLX_MSP3.1M_2_DOUBLE MARK_20X_SEC 1_RIGHT HILUS_28 DAY.czi')
            REF_ROI = field('Datasets/Confocal/SAL,DMI, FLX ADN HALO_TREATMENT_28 DAYS/FLX_28_DAYS/FLX_MSP3.1M_2_DOUBLE MARK_20X_SEC 1_RIGHT HILUS_28 DAY.roi')

            @IM_NUM.connect
            def _update_file(self):
                grandparent = self.__magicclass_parent__.__magicclass_parent__

                if (
                    grandparent.IM_FILES is not None
                    and grandparent.ROI_FILES is not None
                ):
                    self.IMAGE.value = grandparent.IM_FILES[self.IM_NUM.value]
                    self.ROI.value = grandparent.ROI_FILES[self.IM_NUM.value]

            def load_image(self):
                parent = self.__magicclass_parent__
                grandparent = parent.__magicclass_parent__
                pipe = grandparent.pipe

                grandparent.pipe = pipe = core.TissueImage(
                    path.join(grandparent.IN_DIR, self.IMAGE.value),
                    roi_path=path.join(grandparent.IN_DIR, self.ROI.value),
                    roi_name=grandparent.ROI_NAME,
                    ref_im_path=self.REF_IMAGE.value,
                    ref_roi_path=self.REF_ROI.value,
                    out_dir=grandparent.OUT_DIR
                    )
                winsound.MessageBeep(0)

                self.parent_viewer.layers.clear()
                self.parent_viewer.add_image(pipe.imoriginal, scale=pipe.SCALE, colormap='inferno', name='imoriginal')
                self.parent_viewer.add_image(pipe.impreprocessed, scale=pipe.SCALE, colormap='inferno', name='impreprocessed')
                grandparent.current_index = 1
                grandparent.Segmentation.visible = True

                params_deconv = _auto_params_deconv(pipe)

                if params_deconv is not None:
                    deconv_widget = grandparent.Preprocess.Deconvolution
                    deconv_widget.ex_wavelen.value = params_deconv['ex_wavelen']
                    deconv_widget.em_wavelen.value = params_deconv['em_wavelen']
                    deconv_widget.num_aperture.value = params_deconv['num_aperture']
                    deconv_widget.refr_index.value = params_deconv['refr_index']
                    deconv_widget.pinhole_radius.value = params_deconv['pinhole_radius']

    @magicclass(widget_type="scrollable")
    class Preprocess:
        params_preprocess = dict(
            roi=None,
            deconvolution=None,
            bg_subtraction=None,
            clahe=None,
            denoise=None,
            filter=None,
        )

        @magicclass(widget_type="groupbox")
        class ROI:
            skip = field(False)

            @skip.connect
            def _update_params(self):
                if self.skip.value:
                    self.__magicclass_parent__.params_preprocess['roi'] = None

            def crop_roi(self):
                pipe = self.__magicclass_parent__.__magicclass_parent__.pipe
                impreprocessed = pipe.impreprocessed
                roi_path = pipe.ROI_PATH
                roi_name = pipe.ROI_NAME

                select_roi = not(roi_path in (None, '')) and not(roi_name in (None, ''))

                roi_polygon = (np.array([[0, 0], [impreprocessed.shape[0]-1, 0],
                                        [impreprocessed.shape[0]-1, impreprocessed.shape[1]-1],
                                        [0, impreprocessed.shape[1]-1]])
                            if not select_roi else
                            select_ROI(impreprocessed, roi_name, roi_path)
                    )
                pipe.roi_polygon = roi_polygon
                self.parent_viewer.add_shapes(np.fliplr(roi_polygon), shape_type='polygon',
                        name='roi_polygon', scale=pipe.SCALE[-2:])

            def confirm_changes(self):
                pipe = self.__magicclass_parent__.__magicclass_parent__.pipe
                roi_polygon = pipe.roi_polygon
                imoriginal = pipe.imoriginal
                impreprocessed = pipe.impreprocessed

                mask, bounds = mask_ROI(imoriginal, roi_polygon)

                miny, maxy, minx, maxx = get_maximal_rectangle(mask)
                pipe.in_box = (miny, minx, maxy, maxx)

                imoriginal *= np.broadcast_to(mask, imoriginal.shape)
                imoriginal = imoriginal[bounds]  # reduce non-empty

                impreprocessed *= np.broadcast_to(mask, impreprocessed.shape)
                impreprocessed = impreprocessed[bounds]

                pipe.impreprocessed = impreprocessed
                pipe.imoriginal = imoriginal

                roi_params = dict(
                    roi_path=pipe.ROI_PATH,
                    roi_polygon=roi_polygon.astype(int).tolist(),
                    roi_name=pipe.ROI_NAME,
                )
                self.__magicclass_parent__.params_preprocess['roi'] = roi_params
                self.parent_viewer.layers.remove('roi_polygon')
                self.parent_viewer.layers['imoriginal'].data = pipe.imoriginal
                self.parent_viewer.layers['impreprocessed'].data = pipe.impreprocessed

        @magicclass(widget_type="groupbox")
        class Deconvolution:
            skip = field(False)
            ex_wavelen = field(float)
            em_wavelen = field(float)
            num_aperture = FloatSlider(max=1.)
            refr_index = field(float)
            pinhole_radius = field(float)
            pinhole_shape = ComboBox(choices=['round', 'square'])

            DECONV_ITR = field(8)

            @skip.connect
            def _update_params(self):
                if self.skip.value:
                    self.__magicclass_parent__.params_preprocess['deconvolution'] = None

            def deconvolve(self):
                if not self.skip.value:
                    pipe = self.__magicclass_parent__.__magicclass_parent__.pipe
                    params_deconv = dict(
                        iters = self.DECONV_ITR.value,
                        ex_wavelen = self.ex_wavelen.value,
                        em_wavelen = self.em_wavelen.value,
                        num_aperture = self.num_aperture.value,
                        refr_index = self.refr_index.value,
                        pinhole_radius = self.pinhole_radius.value,
                        pinhole_shape = self.pinhole_shape.value
                    )

                    deconvolved = preprocess.deconvolve(pipe, **params_deconv)
                    cmap = self.parent_viewer.layers['impreprocessed'].colormap.name
                    self.parent_viewer.add_image(deconvolved, scale=pipe.SCALE, colormap=cmap)
                    self.__magicclass_parent__.params_preprocess['deconvolution'] = params_deconv

            def confirm_changes(self):
                pipe = self.__magicclass_parent__.__magicclass_parent__.pipe
                pipe.impreprocessed = self.parent_viewer.layers['deconvolved'].data
                self.parent_viewer.layers['imoriginal'].data = pipe.imoriginal
                self.parent_viewer.layers['impreprocessed'].data = pipe.impreprocessed
                self.parent_viewer.layers.remove('deconvolved')

        @magicclass(widget_type="groupbox")
        class BackgroundSubtraction:
            skip = field(False)
            radius = field(50)

            @skip.connect
            def _update_params(self):
                if self.skip.value:
                    self.__magicclass_parent__.params_preprocess['bg_subtraction'] = None

            def subtract_background(self):
                pipe = self.__magicclass_parent__.__magicclass_parent__.pipe
                if not self.skip.value:
                    params_bg_sub = dict(radius=self.radius.value)
                    bg_subtracted = preprocess.subtract_background(pipe.impreprocessed, **params_bg_sub)
                    cmap = self.parent_viewer.layers['impreprocessed'].colormap.name
                    self.parent_viewer.add_image(bg_subtracted, scale=pipe.SCALE, colormap=cmap)
                    self.__magicclass_parent__.params_preprocess['bg_subtraction'] = params_bg_sub

            def confirm_changes(self):
                pipe = self.__magicclass_parent__.__magicclass_parent__.pipe
                pipe.impreprocessed = self.parent_viewer.layers['bg_subtracted'].data
                self.parent_viewer.layers['imoriginal'].data = pipe.imoriginal
                self.parent_viewer.layers['impreprocessed'].data = pipe.impreprocessed
                self.parent_viewer.layers.remove('bg_subtracted')

        @magicclass(widget_type="groupbox")
        class CLAHE:
            skip = field(False)
            clip_limit = FloatSlider(value=.02, max=1)

            @skip.connect
            def _update_params(self):
                if self.skip.value:
                    self.__magicclass_parent__.params_preprocess['clahe'] = None

            def equalize(self):
                pipe = self.__magicclass_parent__.__magicclass_parent__.pipe
                if not self.skip.value:
                    params_clahe = dict(clip_limit=self.clip_limit.value)
                    equalized = preprocess.equalize(pipe.impreprocessed, **params_clahe)
                    cmap = self.parent_viewer.layers['impreprocessed'].colormap.name
                    self.parent_viewer.add_image(equalized, scale=pipe.SCALE, colormap=cmap)
                    self.__magicclass_parent__.params_preprocess['clahe'] = params_clahe

            def confirm_changes(self):
                pipe = self.__magicclass_parent__.__magicclass_parent__.pipe
                pipe.impreprocessed = self.parent_viewer.layers['equalized'].data
                self.parent_viewer.layers['imoriginal'].data = pipe.imoriginal
                self.parent_viewer.layers['impreprocessed'].data = pipe.impreprocessed
                self.parent_viewer.layers.remove('equalized')

        @magicclass(widget_type="groupbox")
        class Denoise:
            skip = field(False)

            @skip.connect
            def _update_params(self):
                if self.skip.value:
                    self.__magicclass_parent__.params_preprocess['denoise'] = None

            def denoise(self):
                pipe = self.__magicclass_parent__.__magicclass_parent__.pipe
                if not self.skip.value:
                    params_denoise = preprocess.calibrate_nlm_denoiser(pipe.impreprocessed)
                    denoised = preprocess.denoise(pipe.impreprocessed, params_denoise)
                    params_denoise['patch_size'] = int(params_denoise['patch_size'])
                    params_denoise['patch_distance'] = int(params_denoise['patch_distance'])
                    cmap = self.parent_viewer.layers['impreprocessed'].colormap.name
                    self.parent_viewer.add_image(denoised, scale=pipe.SCALE, colormap=cmap)
                    self.__magicclass_parent__.params_preprocess['denoise'] = params_denoise

            def confirm_changes(self):
                pipe = self.__magicclass_parent__.__magicclass_parent__.pipe
                pipe.impreprocessed = self.parent_viewer.layers['denoised'].data
                self.parent_viewer.layers['imoriginal'].data = pipe.imoriginal
                self.parent_viewer.layers['impreprocessed'].data = pipe.impreprocessed
                self.parent_viewer.layers.remove('denoised')

        @magicclass(widget_type="groupbox")
        class Filter:
            skip = field(False)
            size = field(2)

            @skip.connect
            def _update_params(self):
                if self.skip.value:
                    self.__magicclass_parent__.params_preprocess['filter'] = None

            def filter(self):
                pipe = self.__magicclass_parent__.__magicclass_parent__.pipe
                if not self.skip.value:
                    params_filter = dict(
                        size=self.size.value
                    )
                    filtered = ndi.median_filter(pipe.impreprocessed, **params_filter)
                    cmap = self.parent_viewer.layers['impreprocessed'].colormap.name
                    self.parent_viewer.add_image(filtered, scale=pipe.SCALE, colormap=cmap)
                    self.__magicclass_parent__.params_preprocess['filter'] = params_filter

            def confirm_changes(self):
                pipe = self.__magicclass_parent__.__magicclass_parent__.pipe
                pipe.impreprocessed = self.parent_viewer.layers['filtered'].data
                self.parent_viewer.layers['imoriginal'].data = pipe.imoriginal
                self.parent_viewer.layers['impreprocessed'].data = pipe.impreprocessed
                self.parent_viewer.layers.remove('filtered')

        def _export_results(self, pipe, params_preprocess):
            with open(path.join(pipe.OUT_DIR, '.params_preprocess.json'), 'w') as out:
                json.dump(params_preprocess, out)
            zarr.save(path.join(pipe.OUT_DIR, '.cache', 'impreprocessed.zarr'), pipe.impreprocessed)

        def save(self):
            self._export_results(self.__magicclass_parent__.pipe, self.params_preprocess)

        def batch_preprocess(self):
            parent = self.__magicclass_parent__
            data = zip(parent.IM_FILES, parent.ROI_FILES)
            params_preprocess = self.params_preprocess.copy()

            for imfile, roifile in data:
                if imfile == parent.LoadDataset.LoadImage.IMAGE.value:
                    continue
                # try:
                pipe = core.TissueImage(
                    path.join(parent.IN_DIR, imfile),
                    roi_path=path.join(parent.IN_DIR, roifile),
                    roi_name=parent.ROI_NAME,
                    ref_im_path=parent.LoadDataset.LoadImage.REF_IMAGE.value,
                    ref_roi_path=parent.LoadDataset.LoadImage.REF_ROI.value,
                    out_dir=parent.OUT_DIR
                    )
                if params_preprocess['roi'] is not None:
                    imoriginal = pipe.imoriginal
                    impreprocessed = pipe.impreprocessed
                    roi_path = pipe.ROI_PATH
                    roi_name = pipe.ROI_NAME

                    select_roi = not(roi_path in (None, '')) and not(roi_name in (None, ''))

                    roi_polygon = (np.array([[0, 0], [impreprocessed.shape[0]-1, 0],
                                            [impreprocessed.shape[0]-1, impreprocessed.shape[1]-1],
                                            [0, impreprocessed.shape[1]-1]])
                                if not select_roi else
                                select_ROI(impreprocessed, roi_name, roi_path)
                        )

                    mask, bounds = mask_ROI(imoriginal, roi_polygon)

                    miny, maxy, minx, maxx = get_maximal_rectangle(mask)
                    pipe.in_box = (miny, minx, maxy, maxx)

                    imoriginal *= np.broadcast_to(mask, imoriginal.shape)
                    imoriginal = imoriginal[bounds]  # reduce non-empty

                    impreprocessed *= np.broadcast_to(mask, impreprocessed.shape)
                    impreprocessed = impreprocessed[bounds]

                    pipe.impreprocessed = impreprocessed
                    pipe.imoriginal = imoriginal

                    roi_params = dict(
                        roi_path=pipe.ROI_PATH,
                        roi_polygon=roi_polygon.astype(int).tolist(),
                        roi_name=pipe.ROI_NAME,
                    )
                    params_preprocess['roi'] = roi_params
                if params_preprocess['deconvolution'] is not None:
                    params_deconv = _auto_params_deconv(pipe)
                    deconv_widget = self.Deconvolution

                    if params_deconv is not None:
                        deconv_widget.ex_wavelen.value = params_deconv['ex_wavelen']
                        deconv_widget.em_wavelen.value = params_deconv['em_wavelen']
                        deconv_widget.num_aperture.value = params_deconv['num_aperture']
                        deconv_widget.refr_index.value = params_deconv['refr_index']
                        deconv_widget.pinhole_radius.value = params_deconv['pinhole_radius']

                    params_deconv = dict(
                        iters = deconv_widget.DECONV_ITR.value,
                        ex_wavelen = deconv_widget.ex_wavelen.value,
                        em_wavelen = deconv_widget.em_wavelen.value,
                        num_aperture = deconv_widget.num_aperture.value,
                        refr_index = deconv_widget.refr_index.value,
                        pinhole_radius = deconv_widget.pinhole_radius.value,
                        pinhole_shape = deconv_widget.pinhole_shape.value
                    )

                    pipe.impreprocessed = preprocess.deconvolve(pipe, **params_deconv)
                    params_preprocess['deconvolution'] = params_deconv
                if params_preprocess['bg_subtraction'] is not None:
                    pipe.impreprocessed = preprocess.subtract_background(pipe.impreprocessed, **params_preprocess['bg_subtraction'])
                if params_preprocess['clahe'] is not None:
                    pipe.impreprocessed = preprocess.equalize(pipe.impreprocessed, **params_preprocess['clahe'])
                if params_preprocess['denoise'] is not None:
                    params_denoise = preprocess.calibrate_nlm_denoiser(pipe.impreprocessed)
                    denoised = preprocess.denoise(pipe.impreprocessed, params_denoise)
                    params_denoise['patch_size'] = int(params_denoise['patch_size'])
                    params_denoise['patch_distance'] = int(params_denoise['patch_distance'])
                    params_preprocess['denoise'] = params_denoise
                if params_preprocess['filter'] is not None:
                    pipe.impreprocessed = ndi.median_filter(pipe.impreprocessed, **params_preprocess['filter'])
                self._export_results(pipe, params_preprocess)
                # except Exception as e:
                #     print(str(e))

        def load_cache(self):
            pipe = self.__magicclass_parent__.pipe
            cache_path = path.join(pipe.OUT_DIR, '.cache', 'impreprocessed.zarr')
            params_path = path.join(pipe.OUT_DIR, '.params_preprocess.json')

            f = open(params_path)
            params_preprocess = json.loads(f.read())
            roi_polygon = np.asarray(params_preprocess['roi']['roi_polygon'])
            pipe.roi_polygon = roi_polygon

            imoriginal = pipe.imoriginal

            mask, bounds = mask_ROI(imoriginal, roi_polygon)

            miny, maxy, minx, maxx = get_maximal_rectangle(mask)
            pipe.in_box = (miny, minx, maxy, maxx)

            imoriginal *= np.broadcast_to(mask, imoriginal.shape)
            imoriginal = imoriginal[bounds]  # reduce non-empty

            pipe.imoriginal = imoriginal
            pipe.impreprocessed = np.asarray(da.from_zarr(cache_path))
            self.params_preprocess = params_preprocess
            self.parent_viewer.layers['imoriginal'].data = pipe.imoriginal
            self.parent_viewer.layers['impreprocessed'].data = pipe.impreprocessed

    @magicclass(widget_type="toolbox")
    class Segmentation:
        @magicclass(widget_type="none")
        class Threshold:
            low_auto_thresh = ComboBox(choices=[None, *THRESHOLD_METHODS])
            low_thresh = FloatSlider(value=.07, max=1)
            high_auto_thresh = ComboBox(choices=[None, *THRESHOLD_METHODS])
            high_thresh = FloatSlider(value=.1, max=1)
            low_delta = FloatSlider(max=1)
            high_delta = FloatSlider(max=1)
            n_steps = field(0)

            def test_thresholds(self):
                pipe = self.__magicclass_parent__.__magicclass_parent__.pipe
                lly, llx, ury, urx = pipe.in_box
                LOW_THRESH, HIGH_THRESH = self.low_thresh.value, self.high_thresh.value
                if self.low_auto_thresh.value is not None:
                    LOW_THRESH = eval(f'filters.threshold_{self.low_auto_thresh.value}(pipe.impreprocessed[:, lly:ury, llx:urx])')
                if self.high_auto_thresh.value is not None:
                    HIGH_THRESH = eval(f'filters.threshold_{self.high_auto_thresh.value}(pipe.impreprocessed[:, lly:ury, llx:urx])')

                results = core._testThresholds(
                    pipe.impreprocessed,
                    LOW_THRESH,
                    HIGH_THRESH,
                    self.low_delta.value,
                    self.high_delta.value,
                    self.n_steps.value
                    )
                past_state = self.parent_viewer.window.qt_viewer.view.camera.get_state()
                self.parent_viewer.layers.clear()

                self.parent_viewer.add_image(pipe.impreprocessed, scale=pipe.SCALE, gamma=.5)

                n_images = len(results)
                for itr in range(n_images):
                    self.parent_viewer.add_labels(**results[itr], scale=pipe.SCALE, rendering='translucent')
                self.parent_viewer.window.qt_viewer.view.camera.set_state(past_state)
                self.low_thresh.value = LOW_THRESH
                self.high_thresh.value = HIGH_THRESH

            def apply_thresholds(self):
                pipe = self.__magicclass_parent__.__magicclass_parent__.pipe
                self.parent_viewer.layers.clear()
                # print(self.low_thresh.value, self.high_thresh.value, self.low_auto_thresh.value, self.high_auto_thresh.value)
                pipe.segment(self.low_thresh.value, self.high_thresh.value, self.low_auto_thresh.value, self.high_auto_thresh.value)
                self.parent_viewer.add_image(pipe.impreprocessed, scale=pipe.SCALE, colormap='inferno', name='unsegmented')
                self.parent_viewer.add_image(pipe.imsegmented, scale=pipe.SCALE, colormap='inferno', name='segmented')
                self.parent_viewer.add_labels(pipe.labels, scale=pipe.SCALE, name='labels',
                    rendering='translucent',
                    color=_get_ncolors_map(len(pipe.regions)+1, cmap_name='plasma_r')
                    )

                parent = self.__magicclass_parent__
                try:
                    parent.RefineSegmentation
                except AttributeError:
                    gui_refine_seg = RefineSegmentation()
                    parent.append(gui_refine_seg)

                gui_refine_seg = parent.RefineSegmentation
                gui_refine_seg.RegionSelector.selected_region.range = (0, len(pipe.regions)-1)
                gui_refine_seg.HighPassVolume.cutoff.max = pipe.regions[-1]['vol']

                try:
                    parent.RefineSegmentation.region_props
                except AttributeError:
                    region_props = PyQt5.QtWidgets.QLabel()
                    parent.RefineSegmentation.region_props = region_props

                if "Region Properties" not in list(self.parent_viewer.window._dock_widgets.keys()):
                    self.parent_viewer.window.add_dock_widget(
                        parent.RefineSegmentation.region_props,
                        name="Region Properties"
                        )
                    self.parent_viewer.window._dock_widgets["Region Properties"].setFixedHeight(270)
                parent.current_index = 1
                parent.__magicclass_parent__.ShowSegmented.visible = True

    @magicclass(widget_type="scrollable", visible=False)
    class ShowSegmented:
        view = field(str, widget_type="ComboBox", options=dict(choices=["single"]))
        obj_index = field(int)
        fig = Figure()

        @view.connect
        @obj_index.connect
        def _plot(self):
            pipe = self.__magicclass_parent__.pipe
            if pipe is not None:
                if self.view.value == 'grid':
                    # Set `BATCH_NO` to view detected objects in paginated 2D MIP views.
                    # pg_size = 30
                    # N_OBJECTS = len(pipe.regions)
                    # N_BATCHES = N_OBJECTS // pg_size + (N_OBJECTS % pg_size > 0)

                    # rows, columns = 10, 3
                    # ax = []

                    # idx = 0
                    # l_obj = self.obj_index.value * pg_size
                    # for obj in range(l_obj, min(pg_size + l_obj, N_OBJECTS)):
                    #     minz, miny, minx, maxz, maxy, maxx = pipe.regions[obj]['bbox']
                    #     ax.append(self.fig.add_subplot(rows, columns, idx+1))
                    #     idx += 1
                    #     ax[-1].set_title(f'Obj {obj}; Vol: {pipe.regions[obj]["vol"]}')
                    #     ax[-1].axis('off')

                    #     extracted_cell = pipe.impreprocessed[minz:maxz, miny:maxy, minx:maxx].copy()
                    #     extracted_cell[~pipe.regions[obj]['image']] = 0.0

                    #     ax.imshow(np.max(extracted_cell, 0), cmap='gray')
                    pass
                else:
                    region = pipe.regions[self.obj_index.value]
                    minz, miny, minx, maxz, maxy, maxx = region['bbox']

                    extracted_cell = pipe.impreprocessed[minz:maxz, miny:maxy, minx:maxx].copy()
                    extracted_cell[~region['image']] = 0.0

                    self.fig.imshow(np.max(extracted_cell, 0))
                    self.fig.ax.axes.xaxis.set_visible(False)
                    self.fig.ax.axes.yaxis.set_visible(False)
                    self.fig.title(f'Volume: {region["vol"]}')

    @magicclass(widget_type="scrollable", visible=False)
    class ExportCells:
        out_dims= field(str, widget_type="ComboBox", options=dict(choices=['both', '3d', 'mip']))
        segment_type=field(str, widget_type="ComboBox", options=dict(choices=['both', 'segmented', 'unsegmented']))

        def export(self):
            pipe = self.__magicclass_parent__.pipe
            pipe.OUT_DIMS, pipe.SEGMENT_TYPE = self.out_dims.value, self.segment_type.value
            pipe._export()

    @magicclass(widget_type="scrollable")
    class SMorph:
        group_dir = field(str)
        LABELS = field(str)
        fig = Figure()

        def analyze(self):
            pipe = self.__magicclass_parent__.pipe
            IMG_TYPE = 'confocal'
            SEGMENTED = True
            CONTRAST_PTILES = (0, 100)
            THRESHOLD_METHOD = .0
            SHOLL_STEP_SIZE = 3
            POLYNOMIAL_DEGREE = 3
            group_dir = self.group_dir.value
            group_dir = pipe.OUT_DIR if group_dir == '' else group_dir

            groups = Groups(
                    [group_dir], image_type=IMG_TYPE, scale=pipe.SCALE,
                    segmented=SEGMENTED, labels=[self.LABELS.value],
                    contrast_ptiles=CONTRAST_PTILES,
                    threshold_method=THRESHOLD_METHOD,
                    sholl_step_size=SHOLL_STEP_SIZE,
                    polynomial_degree=POLYNOMIAL_DEGREE,
                    save_results=True, show_logs=False, fig_format='svg'
                    )
            self.groups = groups

            file_names = groups.file_names
            sholl_step_sz = groups.sholl_step_size
            sholl_polynomial_plots = groups.sholl_polynomial_plots
            polynomial_plots = list(map(lambda x: list(x),
                                        sholl_polynomial_plots))
            group_cnts = groups.group_counts
            labels = groups.labels

            len_polynomial_plots = max(map(len, polynomial_plots))

            polynomial_plots = np.asarray([
                np.pad(x, pad_width=(0, len_polynomial_plots-len(x))) for x in polynomial_plots])

            x = np.arange(sholl_step_sz,
                sholl_step_sz * (len_polynomial_plots + 1),
                sholl_step_sz)

            lft_idx = 0
            err_fn = np.std if min(group_cnts) > 1e5 else sem
            print('Error-bars represent:', err_fn)

            # JASP-friendly data
            jasp_friendly_cols = ['label', 'radius', 'nintersections']
            jasp_friendly = []
            csum_group_cnts = np.cumsum(group_cnts)
            for itercell in range(len(sholl_polynomial_plots)):
                for iterradii, r in enumerate(x):
                    nintersections = (0 if iterradii >= len(sholl_polynomial_plots[itercell])
                        else sholl_polynomial_plots[itercell][iterradii])
                    row = [
                        labels[np.digitize(itercell, csum_group_cnts)],
                        r,
                        nintersections
                    ]
                    jasp_friendly.append(row)

            jasp_friendly = DataFrame(jasp_friendly, columns=jasp_friendly_cols)

            for group_no, group_cnt in enumerate(group_cnts):
                # y = polynomial_plots[lft_idx: lft_idx + group_cnt]
                # e = err_fn(polynomial_plots[lft_idx: lft_idx + group_cnt], axis=0)
                lft_idx += group_cnt

                sns.lineplot(jasp_friendly[jasp_friendly["label"] == labels[group_no]]["radius"],
                    jasp_friendly[jasp_friendly["label"] == labels[group_no]]["nintersections"],
                    ci=68, label=labels[group_no], ax=self.fig.ax
                    )

            self.fig.xlabel("Distance from soma")
            self.fig.ylabel("No. of intersections")
            self.fig.legend()
            fig = plt.gcf()

            cols = np.arange(sholl_step_sz,
                (len_polynomial_plots + 1) * sholl_step_sz,
                sholl_step_sz)

            write_buffer = DataFrame(file_names, columns=['file_name'])
            df_polynomial_plots = DataFrame(polynomial_plots, columns=cols)
            write_buffer[df_polynomial_plots.columns] = df_polynomial_plots

            if groups.save:
                # single_cell_intersections
                OUTFILE = 'sholl_intersections.csv'
                df_to_csv(write_buffer, groups.out_dir, OUTFILE)
                df_to_csv(jasp_friendly, groups.out_dir, 'sholl_intersections_jasp.csv')

                OUTPLOT = f'avg_sholl_plot.{groups.fig_format}'
                savefig(fig, path.join(groups.out_dir, OUTPLOT))
