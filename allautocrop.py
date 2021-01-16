import smorph.util.autocrop as ac
from os import getcwd, listdir, mkdir, path
from read_roi import read_roi_zip

ROOT = 'Datasets/SECTION 1'
SECTIONS = ['SECTION 1_M5_RIGHT', 'SECTION 1_M6_LEFT', 'SECTION 1_M6_RIGHT', 'SECTION 1_M7_LEFT']

LOW_THRESH = .06
HIGH_THRESH = .45

# ROI_FILE = r'crop manual\M3_LEFT_ML_CELL CROP\RoiSet_LB_CELLS.zip'
# rois = read_roi_zip(ROI_FILE)

# if rois[list(rois)[1]]['type'] != 'point':
#     raise ValueError('Cannot read points from ROI zip file')

# points = rois[list(rois)[1]]


for section in SECTIONS:
    for file in listdir(ROOT + '/' + section):
        if not file.startswith('.'):  # skip hidden files
            CONFOCAL_TISSUE_IMAGE = ROOT + '/' + section + '/' + file

            original = ac.import_confocal_image(CONFOCAL_TISSUE_IMAGE)

            # ## 2. Non-local means denoising using auto-calibrated parameters
            denoiser = ac.calibrate_nlm_denoiser(original)
            denoise_parameters = denoiser.keywords['denoiser_kwargs']
            print(denoise_parameters)
            denoised = ac.denoise(original, denoise_parameters)

            # ## 3. Segmentation
            # Edge filtering to determine cell domains
            edge_filtered = ac.filter_edges(denoised)

            thresholded = ac.threshold(edge_filtered, LOW_THRESH, HIGH_THRESH)
            labels = ac.label_thresholded(thresholded)

            # ### 3.2 Filter segmented individual cells by removing ones in borders (touching the convex hull)
            # Find convex hull that approximates tissue structure
            convex_hull = ac.compute_convex_hull(thresholded)

            # discard objects connected to border of approximated tissue, potential partially captured
            filtered_labels = ac.filter_labels(labels, convex_hull)

            regions = ac.arrange_regions(filtered_labels)

            LOW_VOLUME_CUTOFF = 1270  # filter noise/artifacts
            HIGH_VOLUME_CUTOFF = 1e9  # filter cell clusters

            OUTPUT_OPTION = 1  # 0 for 3D cells, 1 for Max Intensity Projections

            ac.export_cells(CONFOCAL_TISSUE_IMAGE, LOW_VOLUME_CUTOFF, HIGH_VOLUME_CUTOFF, OUTPUT_OPTION, denoised, regions)
