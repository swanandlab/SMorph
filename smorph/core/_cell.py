import matplotlib.pyplot as plt
import napari
import numpy as np
from matplotlib.colors import is_color_like
from skimage.color import rgb2gray
from skimage.draw import (
    ellipsoid,
)
from skimage.measure import (
    marching_cubes,
)

from ._features import _extract_cell_features
from ..util import preprocess_image


class Cell:
    """
    Container object for single cell analysis.

    Parameters
    ----------
    cell_image : ndarray
        RGB or Grayscale image data of cell of nervous system.
    image_type : str
        Neuroimaging technique used to get image data of neuronal cell,
        either 'confocal' or 'DAB'.
    crop_tech : str
        Technique used to crop cell from tissue image,
        either 'manual' or 'auto', by default 'manual'.
    contrast_ptiles : tuple of size 2, optional
        `(low_percentile, hi_percentile)` Contains ends of band of percentile
        values for pixel intensities to which the contrast of cell image
        would be stretched, by default (0, 100)
    threshold_method : str or None, optional
        Automatic single intensity thresholding method to be used for
        obtaining ROI from cell image either of 'otsu', 'isodata', 'li',
        'mean', 'minimum', 'triangle', 'yen'. If None & crop_tech is 'auto' &
        contrast stretch is (0, 100), a single intensity threshold of zero is
        applied, by default 'otsu'
    reference_image : ndarray
        `image` would be standardized to the exposure level of this example.
    sholl_step_size : int, optional
        Difference (in pixels) between concentric Sholl circles, by default 3
    polynomial_degree : int, optional
        Degree of polynomial for fitting regression model on sholl values, by
        default 3

    Attributes
    ----------
    image : ndarray
        RGB or Grayscale image data of cell of nervous system.
    image_type : str
        Neuroimaging technique used to get image data of neuronal cell,
        either 'confocal' or 'DAB'.
    cleaned_image : ndarray
        Thresholded, denoised, boolean transformation of `image` with solid
        soma.
    features : dict
        23 Morphometric features derived of the cell.
    skeleton : ndarray
        2D skeletonized, boolean transformation of `image`.
    convex_hull : ndarray
        2D transformation of `skeleton` representing a convex envelope
        that contains it.

    """
    __slots__ = ('image', 'image_type', 'scale', 'cleaned_image', 'features',
                 'convex_hull', 'skeleton', 'sholl_step_size', '_fork_coords',
                 '_branch_coords', '_branching_struct', '_concentric_coords',
                 '_sholl_intersections', '_padded_skeleton', '_pad_sk_soma',
                 '_sholl_polynomial_model', '_polynomial_sholl_radii',
                 '_non_zero_sholl_intersections')

    def __init__(
        self,
        cell_image,
        image_type,
        scale=1,
        crop_tech='manual',
        contrast_ptiles=(0, 100),
        threshold_method='otsu',
        reference_image=None,
        sholl_step_size=3,
        polynomial_degree=3
    ):
        image = cell_image
        if (cell_image.ndim == 3 and cell_image.shape[-1] == 3):
            image = rgb2gray(cell_image)
        self.image_type = image_type
        self.scale = scale
        self.sholl_step_size = sholl_step_size
        self.image, self.cleaned_image = preprocess_image(
            image, image_type, scale, reference_image, crop_tech,
            contrast_ptiles, threshold_method)
        self.features = _extract_cell_features(
            self, sholl_step_size, polynomial_degree)

    def plot_convex_hull(self):
        """Plots convex hull of the skeleton of the cell."""
        if self.convex_hull.ndim == 2:
            ax = plt.subplots()[1]
            ax.set_axis_off()
            ax.imshow(self.convex_hull)
        else:
            with napari.gui_qt():
                viewer = napari.view_image(self.image, ndisplay=3)
                viewer.add_labels(self.skeleton)
                viewer.add_labels(self.convex_hull)

    def plot_forks(self, highlightcolor='green'):
        """Plots skeleton of the cell with all furcations highlighted.

        Parameters
        ----------
        highlightcolor : str, optional
            Color for highlighting the forks, by default 'green'

        """
        if not is_color_like(highlightcolor):
            print(f'{highlightcolor} is not a valid color. '
                  'Resetting highlight color to green.')
            highlightcolor = 'green'

        fork_coords = self._fork_coords
        if self.skeleton.ndim == 2:
            ax = plt.subplots(figsize=(4, 4))[1]
            ax.set_title('path')
            ax.imshow(self.skeleton, interpolation='nearest')

            for i in fork_coords:
                c = plt.Circle((i[1], i[0]), 0.6, color=highlightcolor)
                ax.add_patch(c)

            ax.set_axis_off()
            plt.tight_layout()
            plt.show()
        else:
            with napari.gui_qt():
                viewer = napari.view_image(self.skeleton, ndisplay=3)
                viewer.add_points(list(fork_coords), size=1, opacity=.25,
                                  symbol='ring', face_color=highlightcolor)

    def plot_branching_structure(
        self,
        colors=['red', 'blue', 'magenta', 'green', 'cyan']
    ):
        """Plots skeleton of the cell with all levels of branching highlighted.

        Parameters
        ----------
        colors : list, optional
            List of colors that distinctively highlight all levels of branching
            -- primary, secondary, tertiary, quaternary, & terminal,
            respectively, by default ['red', 'blue', 'magenta', 'green', 'cyan']

        """
        color_validations = list(map(is_color_like, colors))
        if sum(color_validations) != len(color_validations):
            print(f'{colors} is not a valid list of colors. '
                  'Resetting colors to ["red", "blue", "magenta", '
                  '"green", "cyan"].')
            colors = ['red', 'blue', 'magenta', 'green', 'cyan']

        branching_structure = self._branching_struct
        coords = self._branch_coords
        # store same level branch nodes in single array
        color_branches_coords = []
        for branch_level in branching_structure:
            single_branch_level = []
            for path in branch_level:
                path_coords = []
                for node in path:
                    path_coords.append(coords[node])
                    single_branch_level.extend(path_coords)
            color_branches_coords.append(single_branch_level)

        if self.skeleton.ndim == 2:
            ax = plt.subplots(figsize=(4, 4))[1]
            ax.set_title('path')
            ax.imshow(self.skeleton, interpolation='nearest')

            for j, color_branch in enumerate(color_branches_coords):
                if j > 4:
                    j = 4
                for k in color_branch:
                    c = plt.Circle((k[1], k[0]), 0.5, color=colors[j])
                    ax.add_patch(c)

            ax.set_axis_off()
            plt.tight_layout()
            plt.show()
        else:
            with napari.gui_qt():
                viewer = napari.view_labels(self.skeleton, ndisplay=3)
                for j, color_branch in enumerate(color_branches_coords):
                    if j > 4:
                        j = 4
                    viewer.add_points(color_branch, size=1, symbol='ring',
                                      face_color=colors[j], opacity=.25)

    def plot_sholl_results(self, somacolor='red', radiicolor='red'):
        """Plots results of Sholl Analysis.

        This individually plots cell skeleton with sholl circles & Sholl radii
        vs intersections plot.

        Parameters
        ----------
        somacolor : str, optional
            Color to highlight the skeleton soma, by default 'red'
        radiicolor : str, optional
            Color to highlight the sholl radii, by default 'red'

        """
        if not is_color_like(somacolor):
            print(f'{somacolor} is not a valid color. '
                  'Resetting highlight color to red.')
            somacolor = 'red'

        radius = self.sholl_step_size
        sholl_intersections = self._sholl_intersections

        if self.skeleton.ndim == 2:
            ax = plt.subplots(figsize=(10, 6))[1]
            ax.imshow(self._padded_skeleton)

            # overlay soma on skeleton
            y, x = self._pad_sk_soma
            c = plt.Circle((x, y), 1, color=somacolor, alpha=.9)
            ax.add_patch(c)
            ax.set_axis_off()
            # plot circles on skeleton
            for r in range(radius, (len(sholl_intersections)+1)*radius, radius):
                c = plt.Circle((x, y), r, fill=False, lw=2,
                            ec=radiicolor, alpha=.64)
                ax.add_patch(c)
            plt.tight_layout()
            plt.show()

            # plot sholl graph showing radius vs. n_intersections
            plt.plot(range(radius,
                        (len(sholl_intersections)+1)*radius,
                        radius),
                    sholl_intersections)
            plt.xlabel("Distance from centre")
            plt.ylabel("No. of intersections")
            plt.show()
        else:
            with napari.gui_qt():
                viewer = napari.view_labels(self._padded_skeleton, ndisplay=3)
                viewer.add_points([self._pad_sk_soma], face_color=somacolor,
                                  size=1, symbol='ring', opacity=.25)
                for r in range(radius, (len(sholl_intersections)+1)*radius, radius):
                    el = ellipsoid(r, r, r)
                    center = np.array(tuple(map(lambda d: d//2, el.shape)))
                    vertse, facese, normalse, valuese = marching_cubes(el)
                    sholl_sphere = vertse + r + self._pad_sk_soma - (center+r)
                    viewer.add_surface((sholl_sphere, facese, valuese),
                                       opacity=.05, blending='additive',
                                       colormap='hsv')


    def plot_polynomial_fit(self):
        """Plots original & estimated no. of intersections vs. Sholl radii.

        Plots polynomial regression curve describing the relationship between
        no. of intersections vs. Sholl radii, & observed values from the cell
        skeleton.

        """
        sholl_step_sz = self.sholl_step_size
        last_intersection_idx = np.max(np.nonzero(self._sholl_intersections))
        non_zero_radii = range(sholl_step_sz,
                               (last_intersection_idx + 2) * sholl_step_sz,
                               sholl_step_sz)

        x_ = self._polynomial_sholl_radii
        y_data = self._non_zero_sholl_intersections
        # predict y from the data
        reshaped_x = np.array(non_zero_radii).reshape((-1, 1))

        y_new = self._sholl_polynomial_model.predict(x_)
        # plot the results
        plt.figure(figsize=(4, 3))
        ax = plt.axes()
        ax.scatter(reshaped_x, y_data)
        ax.plot(non_zero_radii, y_new)
        ax.set_xlabel('Radii')
        ax.set_ylabel('No. of intersections')
        ax.axis('tight')
        plt.show()
