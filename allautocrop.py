import numpy as np
import smorph.util.autocrop as ac
from os import getcwd, listdir, mkdir, path
from time import time
from skimage.filters import threshold_otsu
import csv
import json
import czifile, psf, skimage

ROOT = 'Datasets'

SECTIONS = [
    'Garima Confocal/SAL,DMI, FLX ADN HALO_TREATMENT_28 DAYS/control_28 days/all'
]

params = {'LOW_THRESH': .07,
          'HIGH_THRESH': .2,
          'SELECT_ROI': True,
          'NAME_ROI': 'HilusMan',
          'LOW_VOLUME_CUTOFF': 200,  # filter noise/artifacts
          'HIGH_VOLUME_CUTOFF': 1e9,  # filter cell clusters
          'OUTPUT_TYPE': 'both'
}

start = time()

for section in SECTIONS:
    for file in listdir(ROOT + '/' + section):
        print(file)
        if not file.startswith('.') and file.endswith('.czi') and 'HILUS' in file:  # skip hidden files
            try:
                CONFOCAL_TISSUE_IMAGE = ROOT + '/' + section + '/' + file

                original = ac.import_confocal_image(CONFOCAL_TISSUE_IMAGE)

                # 2. Non-local means denoising using auto-calibrated parameters
                if original.ndim == 2:
                    original = (original - original.min()) / (original.max() - original.min())
                    
                    original = np.expand_dims(original, 0)

                deconvolved = ac.deconvolve(original, CONFOCAL_TISSUE_IMAGE, iters=10)
                # denoiser = ac.calibrate_nlm_denoiser(deconvolved)
                # denoise_parameters = denoiser.keywords['denoiser_kwargs']
                # print(denoise_parameters)
                # denoised = ac.denoise(deconvolved, denoise_parameters)
                denoised = deconvolved

                FILE_ROI = CONFOCAL_TISSUE_IMAGE[:-4] + '.roi'  # .replace(CONFOCAL_TISSUE_IMAGE.split('/')[3], 'allRoi')[:-4] + '.roi'
                # FILE_ROI = FILE_ROI.replace(FILE_ROI.split('/')[-1], 'RoiSet MAX_' + FILE_ROI.split('/')[-1])
                print(FILE_ROI)

                params['NAME_ROI'] = params['NAME_ROI'] if params['SELECT_ROI'] else ''
                IMG_NAME = '.'.join(CONFOCAL_TISSUE_IMAGE.split('/')[-1].split('.')[:-1])
                linebuilder = None if not params['SELECT_ROI'] else ac.select_ROI(denoised, IMG_NAME + '-' + params['NAME_ROI'], FILE_ROI)

                if params['SELECT_ROI']:
                    original, denoised = ac.mask_ROI(original, denoised,
                                                        linebuilder)

                # 3. Segmentation
                # params['LOW_THRESH'] = params['HIGH_THRESH'] = skimage.filters.threshold_otsu(denoised)
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
