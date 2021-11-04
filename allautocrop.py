import numpy as np
import smorph.util.autocrop as ac
from os import getcwd, listdir, mkdir, path
from time import time
from skimage.filters import threshold_otsu
from skimage import exposure
import csv
import json
import czifile, psf, skimage

ROOT = 'Datasets'

SECTIONS = [
    'Confocal/SAL,DMI, FLX ADN HALO_TREATMENT_21 DAYS/allImg/HILUS'
]

params = {'LOW_THRESH': .34,
          'HIGH_THRESH': .4,
          'SELECT_ROI': True,
          'NAME_ROI': 'ML',
          'LOW_VOLUME_CUTOFF': 250,  # filter noise/artifacts
          'HIGH_VOLUME_CUTOFF': 1e9,  # filter cell clusters
          'OUTPUT_TYPE': 'both'
}

ref = None

#installed windows sdk
# !pip install cvxpy cvxopt
import cvxpy
import cvxopt
from shapely.geometry import Polygon
import imagej
import scyjava
scyjava.config.add_option('-Xmx6g')
ij = imagej.init('C:/Program Files (x86)/Fiji.app', headless=False)
from scyjava import jimport
WindowManager = jimport('ij.WindowManager')


def rect2poly(ll, ur):
    """
    Convert rectangle defined by lower left/upper right
    to a closed polygon representation.
    """
    x0, y0 = ll
    x1, y1 = ur

    return [
        [x0, y0],
        [x0, y1],
        [x1, y1],
        [x1, y0],
        [x0, y0]
    ]


def get_intersection(coords):
    """Given an input list of coordinates, find the intersection
    section of corner coordinates. Returns geojson of the
    interesection polygon.
    """
    ipoly = None
    for coord in coords:
        if ipoly is None:
            ipoly = Polygon(coord)
        else:
            tmp = Polygon(coord)
            ipoly = ipoly.intersection(tmp)

    # close the polygon loop by adding the first coordinate again
    first_x = ipoly.exterior.coords.xy[0][0]
    first_y = ipoly.exterior.coords.xy[1][0]
    ipoly.exterior.coords.xy[0].append(first_x)
    ipoly.exterior.coords.xy[1].append(first_y)

    inter_coords = zip(
        ipoly.exterior.coords.xy[0], ipoly.exterior.coords.xy[1])

    inter_gj = {"geometry":
                {"coordinates": [inter_coords],
                 "type": "Polygon"},
                 "properties": {}, "type": "Feature"}

    return inter_gj, inter_coords


def two_pts_to_line(pt1, pt2):
    """
    Create a line from two points in form of
    a1(x) + a2(y) = b
    """
    pt1 = [float(p) for p in pt1]
    pt2 = [float(p) for p in pt2]
    try:
        slp = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
    except ZeroDivisionError:
        slp = 1e5 * (pt2[1] - pt1[1])
    a1 = -slp
    a2 = 1.
    b = -slp * pt1[0] + pt1[1]

    return a1, a2, b


def pts_to_leq(coords):
    """
    Converts a set of points to form Ax = b, but since
    x is of length 2 this is like A1(x1) + A2(x2) = B.
    returns A1, A2, B
    """

    A1 = []
    A2 = []
    B = []
    for i in range(len(coords) - 1):
        pt1 = coords[i]
        pt2 = coords[i + 1]
        a1, a2, b = two_pts_to_line(pt1, pt2)
        A1.append(a1)
        A2.append(a2)
        B.append(b)
    return A1, A2, B


def get_maximal_rectangle(coordinates):
    """
    Find the largest, inscribed, axis-aligned rectangle.
    :param coordinates:
        A list of of [x, y] pairs describing a closed, convex polygon.
    """

    coordinates = np.array(coordinates)
    x_range = np.max(coordinates, axis=0)[0]-np.min(coordinates, axis=0)[0]
    y_range = np.max(coordinates, axis=0)[1]-np.min(coordinates, axis=0)[1]

    scale = np.array([x_range, y_range])
    sc_coordinates = coordinates/scale

    poly = Polygon(sc_coordinates)
    inside_pt = (poly.representative_point().x,
                 poly.representative_point().y)

    A1, A2, B = pts_to_leq(sc_coordinates)

    bl = cvxpy.Variable(2)
    tr = cvxpy.Variable(2)
    br = cvxpy.Variable(2)
    tl = cvxpy.Variable(2)
    obj = cvxpy.Maximize(cvxpy.log(tr[0] - bl[0]) + cvxpy.log(tr[1] - bl[1]))
    constraints = [bl[0] == tl[0],
                   br[0] == tr[0],
                   tl[1] == tr[1],
                   bl[1] == br[1],
                   ]

    for i in range(len(B)):
        if inside_pt[0] * A1[i] + inside_pt[1] * A2[i] <= B[i]:
            constraints.append(bl[0] * A1[i] + bl[1] * A2[i] <= B[i])
            constraints.append(tr[0] * A1[i] + tr[1] * A2[i] <= B[i])
            constraints.append(br[0] * A1[i] + br[1] * A2[i] <= B[i])
            constraints.append(tl[0] * A1[i] + tl[1] * A2[i] <= B[i])

        else:
            constraints.append(bl[0] * A1[i] + bl[1] * A2[i] >= B[i])
            constraints.append(tr[0] * A1[i] + tr[1] * A2[i] >= B[i])
            constraints.append(br[0] * A1[i] + br[1] * A2[i] >= B[i])
            constraints.append(tl[0] * A1[i] + tl[1] * A2[i] >= B[i])

    prob = cvxpy.Problem(obj, constraints)
    prob.solve()#prob.solve(solver=cvxpy.SCIPY, verbose=False, max_iters=1000, reltol=1e-9)

    bottom_left = np.array(bl.value).T * scale
    top_right = np.array(tr.value).T * scale

    return bottom_left, top_right#list(bottom_left[0]), list(top_right[0])

start = time()

for section in SECTIONS:
    for file in listdir(ROOT + '/' + section):
        # print(file)
        if not file.startswith('.') and file.endswith('.czi') and 'HILUS' in file:  # skip hidden files
            try:
                CONFOCAL_TISSUE_IMAGE = ROOT + '/' + section + '/' + file

                original = ac.import_confocal_image(CONFOCAL_TISSUE_IMAGE)

                # 2. Non-local means denoising using auto-calibrated parameters
                if original.ndim == 2:
                    original = (original - original.min()) / (original.max() - original.min())
                    original = np.expand_dims(original, 0)

                deconvolved = ac.deconvolve(original, CONFOCAL_TISSUE_IMAGE, iters=8)
                # denoiser = ac.calibrate_nlm_denoiser(deconvolved)
                # denoise_parameters = denoiser.keywords['denoiser_kwargs']
                # print(denoise_parameters)
                # denoised = ac.denoise(deconvolved, denoise_parameters)
                # Adaptive Equalization
                img_adapteq = exposure.equalize_adapthist(original, clip_limit=0.03)
                if ref is None:
                    ref = img_adapteq
                denoised = skimage.exposure.match_histograms(img_adapteq, ref)

                FILE_ROI = CONFOCAL_TISSUE_IMAGE.replace(CONFOCAL_TISSUE_IMAGE.split('/')[3], 'allRoi')[:-4] + '-ML.roi'
                # FILE_ROI = FILE_ROI.replace(FILE_ROI.split('/')[-1], 'RoiSet MAX_' + FILE_ROI.split('/')[-1])
                print(FILE_ROI)

                params['NAME_ROI'] = params['NAME_ROI'] if params['SELECT_ROI'] else ''
                IMG_NAME = '.'.join(CONFOCAL_TISSUE_IMAGE.split('/')[-1].split('.')[:-1])
                linebuilder = None if not params['SELECT_ROI'] else ac.select_ROI(denoised, IMG_NAME + '-' + params['NAME_ROI'], FILE_ROI)

                if params['SELECT_ROI']:
                    original, denoised = ac.mask_ROI(original, denoised,
                                                        linebuilder)
                
                _, coordinates = get_intersection([linebuilder])
                ll, ur = get_maximal_rectangle(np.array((list(coordinates))))
                ll, ur = np.ceil(ll).astype(int), np.floor(ur).astype(int)
                llx, lly = ll; urx, ury = ur
                llx -= linebuilder[:, 0].min(); urx -= linebuilder[:, 0].min()
                lly -= linebuilder[:, 1].min(); ury -= linebuilder[:, 1].min()

                ij.ui().show('denoised', ij.py.to_java(np.expand_dims(original, axis=1)))
                plugin = '3D Fast Filters'
                args = { 
                    'filter': 'Median',  # StringField
                    'radius_x_unit': .6918883015525974,  # NumericField
                    'radius_x_pix': 1,  # NumericField
                    'radius_y_unit': .6918883015525974,  # NumericField
                    'radius_y_pix': 1,  # NumericField
                    'radius_z_unit': 1.0785801681301463,
                    'radius_z_pix': 1
                }

                ij.py.run_plugin(plugin, args)#, ij1_style=False)
                result = WindowManager.getCurrentImage()
                result = ij.py.from_java(result).to_numpy()

                denoiser = ac.calibrate_nlm_denoiser(result[:, lly:ury, llx:urx])
                denoise_parameters = denoiser.keywords['denoiser_kwargs']
                denoised = ac.denoise(result, denoise_parameters)

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
                                params['OUTPUT_TYPE'], original, regions, residue_regions,
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
