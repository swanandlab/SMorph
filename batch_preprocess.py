import warnings
warnings.filterwarnings('ignore')

from os import listdir, path
from time import time
import smorph as sm
import smorph.util.autocrop as ac
import numpy as np

from skimage import exposure, restoration
 
ROOT = 'Datasets'
SECTIONS = [
    'Confocal/SAL,DMI, FLX ADN HALO_TREATMENT_28 DAYS/img/HILUS'
]

# REF_IMAGE = 'Datasets/Confocal/SAL,DMI, FLX ADN HALO_TREATMENT_28 DAYS/control_28 days/MSP3.1M_1_SINGLE MARK_CONTROL_28 DAY/CONTROL_MSP3.1M_1_SINGLE MARK_20X_SEC 1_RIGHT HILUS_28 DAYczi.czi'

OUT_DIR = 'Cache/'

DECONV_ITR = 30

start = time()

sm.util._io.mkdir_if_not(OUT_DIR)

for section in SECTIONS:
    for file in listdir(ROOT + '/' + section):
        # print(file)
        if not file.startswith('.') and file.endswith('.czi'):  # skip hidden files
            try:
                CONFOCAL_TISSUE_IMAGE = ROOT + '/' + section + '/' + file

                original = ac.import_confocal_image(CONFOCAL_TISSUE_IMAGE)
                deconvolved = ac.deconvolve(original, CONFOCAL_TISSUE_IMAGE, iters=DECONV_ITR)

                CLIP_LIMIT = .02
                background = restoration.rolling_ball(deconvolved, radius=(min(deconvolved.shape)-1)//2)
                preprocessed = exposure.equalize_adapthist(deconvolved-background, clip_limit=CLIP_LIMIT)
                IMAGE_NAME = '.'.join(path.basename(CONFOCAL_TISSUE_IMAGE).split('.')[:-1])
                np.save(OUT_DIR + IMAGE_NAME + '.npy', preprocessed)
            except Exception as e:
                print(str(e))

print('Elapsed:', time() - start, 'secs')
