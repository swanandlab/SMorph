import matplotlib.pyplot as plt
import numpy as np
import roifile

from matplotlib.path import Path
from shapely.geometry import LineString


def select_ROI(denoised, name, filename=None):
    if filename is None:
        return _draw_ROI(denoised, name)
    return _load_ROI(filename)


def _load_ROI(filename):
    roi = roifile.roiread(filename)
    roi = roi if type(roi) != list else roi[0]
    polygon_coords = roi.coordinates()
    return polygon_coords


def _draw_ROI(denoised, name):
    class LineBuilder:
        def __init__(self, line):
            self.line = line
            self.xs = list(line.get_xdata())
            self.ys = list(line.get_ydata())
            self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

        def __call__(self, event):
            if event.inaxes != self.line.axes: return
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
            self.line.set_data(self.xs, self.ys)
            self.line.figure.canvas.draw()
            if len(self.xs) > 3:
                L1 = [(self.xs[0], self.ys[0]), (self.xs[1], self.ys[1])]
                L2 = [(self.xs[-2], self.ys[-2]), (self.xs[-1], self.ys[-1])]
                L1, L2 = LineString(L1), LineString(L2)
                res = L1.intersection(L2)
                try:
                    res.xy
                    self.xs[-1], self.ys[-1] = res.x, res.y
                    roi = roifile.ImagejRoi.frompoints(
                        list(zip(linebuilder.xs, linebuilder.ys)))
                    roifile.roiwrite('Results/' + name + '.roi', roi)
                    self.line.figure.canvas.mpl_disconnect(self.cid)
                except:
                    pass


    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(np.max(denoised, 0), cmap='gray')
    ax.set_title('Select ROI')
    l, = ax.plot([], [])  # empty line
    linebuilder = LineBuilder(l)

    plt.show()
    return linebuilder


def mask_ROI(img, linebuilder):
    # Create a binary image (mask) from ROI object.
    y, x = np.mgrid[:img.shape[1], :img.shape[2]]
    if type(linebuilder) != np.ndarray:
        poly_path = Path(list(zip(linebuilder.xs, linebuilder.ys)))
        min_x, max_x = int(min(linebuilder.xs)), int(max(linebuilder.xs) + 1)
        min_y, max_y = int(min(linebuilder.ys)), int(max(linebuilder.ys) + 1)
    else:
        poly_path = Path(linebuilder)
        linebuilder = list(zip(*linebuilder))
        min_x, max_x = int(min(linebuilder[0])), int(max(linebuilder[0]) + 1)
        min_y, max_y = int(min(linebuilder[1])), int(max(linebuilder[1]) + 1)
    coords = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))

    mask = poly_path.contains_points(coords)
    mask = mask.reshape(img.shape[1], img.shape[2])

    out = np.empty(img.shape)
    for i in range(img.shape[0]):
        out[i] = img[i] * mask

    # reduce non-empty
    reduced = out[:, min_y: max_y, min_x:max_x]

    return reduced
