import smorph.util.autocrop as ac
from os import getcwd, listdir, mkdir, path
from time import time
from skimage.filters import threshold_otsu
import csv
import json
import czifile, psf, skimage

ROOT = 'Datasets'

SECTIONS = [
    'Garima Confocal/SAL,DMI, FLX ADN HALO_TREATMENT_21 DAYS/allImg/HILUS'
]

params = {'LOW_THRESH': .05,
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
    for file in listdir(ROOT + '/' + section):
        if not file.startswith('.'):  # skip hidden files
            try:
                CONFOCAL_TISSUE_IMAGE = ROOT + '/' + section + '/' + file

                original = ac.import_confocal_image(CONFOCAL_TISSUE_IMAGE)

                # 2. Non-local means denoising using auto-calibrated parameters
                denoiser = ac.calibrate_nlm_denoiser(original)
                denoise_parameters = denoiser.keywords['denoiser_kwargs']
                denoised = ac.denoise(original, denoise_parameters)

                czimeta = czifile.CziFile(CONFOCAL_TISSUE_IMAGE).metadata(False)
                refr_index = czimeta['ImageDocument']['Metadata']['Information']['Image']['ObjectiveSettings']['RefractiveIndex']

                selected_channel = None
                for i in czimeta['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel']:
                    if i['ContrastMethod'] == 'Fluorescence':
                        selected_channel = i
                ex_wavelen = selected_channel['ExcitationWavelength']
                em_wavelen = selected_channel['EmissionWavelength']

                selected_detector = None
                for i in czimeta['ImageDocument']['Metadata']['Experiment']['ExperimentBlocks']['AcquisitionBlock']['MultiTrackSetup']['TrackSetup']['Detectors']['Detector']:
                    if i['PinholeDiameter'] > 0:
                        selected_detector = i
                pinhole_radius = selected_detector['PinholeDiameter'] / 2 * 1e6

                num_aperture = czimeta['ImageDocument']['Metadata']['Information']['Instrument']['Objectives']['Objective']['LensNA']
                dim_r = czimeta['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][0]['Value'] * 1e6
                dim_z = czimeta['ImageDocument']['Metadata']['Scaling']['Items']['Distance'][-1]['Value'] * 1e6

                args = dict(
                    shape=(3, 3),  # # of samples in z & r direction
                    dims=(dim_z, dim_r),  # size in z & r direction in microns
                    ex_wavelen=ex_wavelen,  # nm
                    em_wavelen=em_wavelen,  # nm
                    num_aperture=num_aperture,
                    refr_index=refr_index,
                    pinhole_radius=pinhole_radius,  # microns
                    pinhole_shape='square'
                )
                obsvol = psf.PSF(psf.ISOTROPIC | psf.CONFOCAL, **args)
                psf_vol = obsvol.volume()

                denoised = skimage.restoration.richardson_lucy(denoised, psf_vol, iterations=8)

                FILE_ROI = CONFOCAL_TISSUE_IMAGE.replace(CONFOCAL_TISSUE_IMAGE.split('/')[3], 'allRoi')[:-4] + '.roi'
                # FILE_ROI = FILE_ROI.replace(FILE_ROI.split('/')[-1], 'RoiSet MAX_' + FILE_ROI.split('/')[-1])
                print(FILE_ROI)

                params['NAME_ROI'] = params['NAME_ROI'] if params['SELECT_ROI'] else ''
                IMG_NAME = '.'.join(CONFOCAL_TISSUE_IMAGE.split('/')[-1].split('.')[:-1])
                linebuilder = None if not params['SELECT_ROI'] else ac.select_ROI(denoised, IMG_NAME + '-' + params['NAME_ROI'], FILE_ROI)

                if params['SELECT_ROI']:
                    original, denoised = ac.mask_ROI(original, denoised,
                                                        linebuilder)

                # 3. Segmentation
                thresholded = ac.threshold(denoised, params['LOW_THRESH'],
                                            params['HIGH_THRESH'])
                labels = ac.label_thresholded(thresholded)

                # 3.2 Filter segmented individual cells by removing ones in
                # borders (touching the convex hull) discard objects connected
                # to border of approximated tissue, potential partially
                # captured
                filtered_labels = ac.filter_labels(labels, thresholded,
                                                    linebuilder, True)

                regions = ac.arrange_regions(filtered_labels)
                residue_regions = ac.arrange_regions(labels - filtered_labels)

                ac.export_cells(CONFOCAL_TISSUE_IMAGE,
                                params['LOW_VOLUME_CUTOFF'],
                                params['HIGH_VOLUME_CUTOFF'],
                                params['OUTPUT_TYPE'], denoised, regions, residue_regions,
                                seg_type='both', roi_name=params['NAME_ROI'],
                                roi_polygon=linebuilder, roi_path=FILE_ROI)

                DIR = getcwd() + '/Autocropped/'
                IMAGE_NAME = '.'.join(path.basename(
                    CONFOCAL_TISSUE_IMAGE).split('.')[:-1])
                NAME_ROI = params['NAME_ROI']
                OUT_DIR = DIR + IMAGE_NAME + \
                        f'{"" if NAME_ROI == "" else "-" + str(NAME_ROI)}/'
                with open(OUT_DIR + '.params.json', 'w') as out:
                    json.dump(params, out)
            except Exception as e:
                print(str(e))

print('Elapsed:', time() - start, 'secs')
