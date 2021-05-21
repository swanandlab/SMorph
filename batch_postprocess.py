import smorph.util.autocrop as ac
from os import getcwd, listdir, mkdir, path
from time import time
from skimage.filters import threshold_otsu
import csv
import json
import czifile, psf, skimage, numpy as np
from skimage.morphology import binary_erosion
from skimage.segmentation import clear_border
from skimage.draw import polygon2mask

ROOT = 'Autocropped'

SECTIONS = [
    'CTRLvDMI-21-ML-Deconv/test'
]

params = {'LOW_THRESH': .055,
          'HIGH_THRESH': .2,
          'SELECT_ROI': True,
          'NAME_ROI': 'Hilus',
          'LOW_VOLUME_CUTOFF': 200,  # filter noise/artifacts
          'HIGH_VOLUME_CUTOFF': 1e9,  # filter cell clusters
          'OUTPUT_TYPE': 'both'
}

# ROI_FILE = r'crop manual\M3_LEFT_ML_CELL CROP\RoiSet_LB_CELLS.zip'
# rois = read_roi_zip(ROI_FILE)

# if rois[list(rois)[1]]['type'] != 'point':
#     raise ValueError('Cannot read points from ROI zip file')

# points = rois[list(rois)[1]]

start = time()

for section in SECTIONS:
    for fol in listdir(ROOT + '/' + section):
        if not fol.startswith('.'):  # skip hidden
            # try:
            SOMA_SELECTED = ROOT + '/' + section + '/' + fol + '/residue'
            reconstructed_labels = None
            reconstructed_labels, parent_path, roi_path = ac.postprocess_segment(SOMA_SELECTED, reconstructed_labels)

            linebuilder = ac._roi_extract._load_ROI(roi_path)

            # Filter
            conservative = True
            reconstructed_residue_labels = clear_border(reconstructed_labels, mask=None if conservative
                                   else ac.core._compute_convex_hull(thresholded))

            if linebuilder is not None:
                X, Y = ac.core._unwrap_polygon(linebuilder)
                min_x, max_x = int(min(X)), int(max(X) + 1)
                min_y, max_y = int(min(Y)), int(max(Y) + 1)
                shape = reconstructed_labels.shape
                roi_mask = np.empty(shape)
                roi_mask[0] = polygon2mask(shape[1:][::-1], list(zip(X, Y))).T

                for i in range(1, shape[0]):
                    roi_mask[i] = roi_mask[0]
                roi_mask = binary_erosion(roi_mask)
                reconstructed_residue_labels = clear_border(reconstructed_residue_labels, mask=roi_mask)

            reconstructed_filtered_regions = ac.arrange_regions(reconstructed_residue_labels)

            CONFOCAL_TISSUE_IMAGE = parent_path

            original = ac.import_confocal_image(CONFOCAL_TISSUE_IMAGE)

            # 2. Non-local means denoising using auto-calibrated parameters
            denoiser = ac.calibrate_nlm_denoiser(original)
            denoise_parameters = denoiser.keywords['denoiser_kwargs']
            denoised = ac.denoise(original, denoise_parameters)

            # czimeta = czifile.CziFile(CONFOCAL_TISSUE_IMAGE).metadata(False)
            # refr_index = czimeta['ImageDocument']['Metadata']['Information']['Image']['ObjectiveSettings']['RefractiveIndex']

            # selected_channel = None
            # for i in czimeta['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel']:
            #     if i['ContrastMethod'] == 'Fluorescence':
            #         selected_channel = i
            # ex_wavelen = selected_channel['ExcitationWavelength']
            # em_wavelen = selected_channel['EmissionWavelength']

            # selected_detector = None
            # for i in czimeta['ImageDocument']['Metadata']['Experiment']['ExperimentBlocks']['AcquisitionBlock']['MultiTrackSetup']['TrackSetup']['Detectors']['Detector']:
            #     if i['PinholeDiameter'] > 0:
            #         selected_detector = i
            # pinhole_radius = selected_detector['PinholeDiameter'] / 2 * 1e6

            # num_aperture = czimeta['ImageDocument']['Metadata']['Information']['Instrument']['Objectives']['Objective']['LensNA']
            # dim_r = czimeta['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][0]['Value'] * 1e6
            # dim_z = czimeta['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][-1]['Value'] * 1e6

            # args = dict(
            #     shape=(3, 3),  # # of samples in z & r direction
            #     dims=(dim_z, dim_r),  # size in z & r direction in microns
            #     ex_wavelen=ex_wavelen,  # nm
            #     em_wavelen=em_wavelen,  # nm
            #     num_aperture=num_aperture,
            #     refr_index=refr_index,
            #     pinhole_radius=pinhole_radius,  # microns
            #     pinhole_shape='square'
            # )
            # obsvol = psf.PSF(psf.ISOTROPIC | psf.CONFOCAL, **args)
            # psf_vol = obsvol.volume()

            # denoised = skimage.restoration.richardson_lucy(denoised, psf_vol, iterations=8)

            LOW_VOLUME_CUTOFF = 0  # filter noise/artifacts
            HIGH_VOLUME_CUTOFF = 1e9  # filter cell clusters

            OUTPUT_OPTION = 'both'  # '3d' for 3D cells, 'mip' for Max Intensity Projections
            SEGMENT_TYPE = 'segmented'

            ac.export_cells(parent_path, LOW_VOLUME_CUTOFF,
                            HIGH_VOLUME_CUTOFF, OUTPUT_OPTION, denoised,
                            reconstructed_filtered_regions, None, SEGMENT_TYPE, '', linebuilder, roi_path=roi_path)

            # DIR = getcwd() + '/Autocropped/'
            # IMAGE_NAME = '.'.join(path.basename(
            #     CONFOCAL_TISSUE_IMAGE).split('.')[:-1])
            # NAME_ROI = params['NAME_ROI']
            # OUT_DIR = DIR + IMAGE_NAME + \
            #         f'{"" if NAME_ROI == "" else "-" + str(NAME_ROI)}/'
            # with open(OUT_DIR + '.params.json', 'w') as out:
            #     json.dump(params, out)
            # except Exception as e:
            #     print(str(e))
                # pass

print('Elapsed:', time() - start, 'secs')
