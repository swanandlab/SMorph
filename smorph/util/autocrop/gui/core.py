import numpy as np
import PyQt5
import superqt

from magicgui import magicgui
from vispy.geometry.rect import Rect


def volume_cutoff(viewer, denoised, segmented, filtered_labels, regions, SCALE):
    PRE_LOW_VOLUME_CUTOFF = 0
    viewer.add_image(denoised, scale=SCALE)
    viewer.add_labels(filtered_labels, rendering='translucent', opacity=.5, scale=SCALE)

    region_props = PyQt5.QtWidgets.QLabel()
    region_inclusivity = np.ones(len(regions), dtype=bool)
    REGION_INCLUSIVITY_LABELS = ['Include Region', 'Exclude Region']

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
        cutoff={'widget_type': 'Slider', 'max': regions[-1]['vol']}
    )
    def vol_cutoff_update(
        cutoff=PRE_LOW_VOLUME_CUTOFF
    ):
        nonlocal filtered_labels, regions, segmented, PRE_LOW_VOLUME_CUTOFF
        layer_names = [layer.name for layer in viewer.layers]
        PRE_LOW_VOLUME_CUTOFF = cutoff
        filtered_labels.fill(0)
        itr = 1
        filtered_regions = []
        for region in regions:
            if cutoff <= region['vol']:
                minz, miny, minx, maxz, maxy, maxx = region['bbox']
                filtered_labels[minz:maxz, miny:maxy, minx:maxx] += region['image'] * itr
                itr += 1
                filtered_regions.append(region)
        viewer.layers[layer_names.index('filtered_labels')].data = filtered_labels
        regions = filtered_regions
        segmented = denoised * (filtered_labels > 0)
        select_region()

    @magicgui(
        auto_call=True,
        selected_region={'maximum': len(regions)-1},
        select_region=dict(widget_type='PushButton', text='Select Region')
    )
    def select_region(
        selected_region=n_region,
        select_region=True  # just for activating the method
    ):
        nonlocal n_region
        n_region = selected_region
        minz, miny, minx, maxz, maxy, maxx = regions[n_region]['bbox']
        centroid = regions[n_region]['centroid']
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
            data += '<tr><td><b>' + name + '</b></td><td>' + str(eval(f'regions[{n_region}]["{prop}"]')) + '</td></tr>'
        data += '</table>'
        region_props.setText(data)


    viewer.window.add_dock_widget(vol_cutoff_update, name='Volume Cutoff', area='bottom')
    viewer.window._dock_widgets['Volume Cutoff'].setFixedHeight(90)
    viewer.window.add_dock_widget(select_region, name='Select Region')
    viewer.window._dock_widgets['Select Region'].setFixedHeight(100)
    viewer.window.add_dock_widget(region_props, name='Region Properties')
    viewer.window._dock_widgets['Select Region'].setFixedHeight(270)
    select_region()