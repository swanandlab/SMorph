import smorph.util.autocrop as ac
from os import getcwd, listdir, mkdir, path
from time import time

ROOT = 'Datasets'

SECTIONS = [
    'autocrop'
    # 'FLX_CZI/SC109.3M_1_SINGLE MARK_FLX_21 DAY',
    # 'FLX_CZI/SC110.1M_2_DOUBLE MARK_FLX_21 DAY',
    # 'FLX_CZI/UN1_2_DOUBLE MARK_FLX_21 DAY',
    # 'FLX_CZI/UN2_3_UNMARKED_FLX_21 DAY'
]

LOW_THRESH = .07
HIGH_THRESH = .3

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

                # ## 2. Non-local means denoising using auto-calibrated parameters
                denoiser = ac.calibrate_nlm_denoiser(original)
                denoise_parameters = denoiser.keywords['denoiser_kwargs']
                denoised = ac.denoise(original, denoise_parameters)

                SELECT_ROI = False
                NAME_ROI = ''
                FILE_ROI = None # CONFOCAL_TISSUE_IMAGE.replace(CONFOCAL_TISSUE_IMAGE.split('/')[1], 'ISOPROTERENOLroi')[:-4] + '-ML.roi'
                print(CONFOCAL_TISSUE_IMAGE)

                NAME_ROI = NAME_ROI if SELECT_ROI else ''
                IMG_NAME = '.'.join(CONFOCAL_TISSUE_IMAGE.split('/')[-1].split('.')[:-1])
                linebuilder = None if not SELECT_ROI else ac.select_ROI(denoised, IMG_NAME + '-' + NAME_ROI, FILE_ROI)

                if SELECT_ROI:
                    original, denoised = ac.mask_ROI(original, denoised, linebuilder)

                # ## 3. Segmentation
                thresholded = ac.threshold(denoised, LOW_THRESH, HIGH_THRESH)
                labels = ac.label_thresholded(thresholded)

                # ### 3.2 Filter segmented individual cells by removing ones in borders (touching the convex hull)
                # discard objects connected to border of approximated tissue, potential partially captured
                filtered_labels = ac.filter_labels(labels, thresholded, linebuilder, False)

                regions = ac.arrange_regions(filtered_labels)

                LOW_VOLUME_CUTOFF = 400  # filter noise/artifacts
                HIGH_VOLUME_CUTOFF = 1e9  # filter cell clusters

                OUTPUT_OPTION = 1  # 0 for 3D cells, 1 for Max Intensity Projections

                ac.export_cells(CONFOCAL_TISSUE_IMAGE, LOW_VOLUME_CUTOFF, HIGH_VOLUME_CUTOFF, OUTPUT_OPTION, original, regions, NAME_ROI, linebuilder)
            except Exception as e:
                print(str(e))

print('Elapsed:', time() - start, 'secs')
