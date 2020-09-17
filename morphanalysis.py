import os
import shutil
import copy
import skimage
import sklearn
import scipy
import numpy as np
import pandas as pd
import seaborn as sns

from collections import defaultdict
from itertools import cycle, islice
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import skimage.io as io
from skimage.transform import rescale
from skimage.feature import blob_log
from skimage.morphology import closing, square
from skimage.filters import threshold_otsu
from skimage.measure import label

import skan
from skan import skeleton_to_csgraph

from sklearn import decomposition, linear_model, metrics, preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pandas.plotting import parallel_coordinates
import ipyvolume as ipv


class Cell:
    """
    Extract the individual cell by thresholding and removing background noise. 
    """

    def __init__(self, cell_image, image_type, reference_image=None):
        """
        Args:

        cell_image: RGB cell image
        image_type: 'confocal' or 'DAB'
        reference_image: all images can be standardised to the exposure level of this example cell image

        """

        self.cell_image = cell_image
        self.image_type = image_type
        self.gray_cell_image = skimage.color.rgb2gray(self.cell_image)
        self.inverted_gray_cell_image = skimage.util.invert(
            self.gray_cell_image)
        self.reference_image = reference_image
        self.thresholded_image = self.threshold_image()
        self.inverted_thresholded_image = skimage.util.invert(
            self.thresholded_image)
        self.cleaned_image = self.remove_small_object_noise()
        self.cleaned_image_filled_holes = self.fill_holes()

    def get_blobs(self):
        # Extracts circular blobs in cell image for finding soma later

        if self.image_type == "DAB":
            blobs_log = blob_log(self.inverted_gray_cell_image, min_sigma=6,
                                 max_sigma=20, num_sigma=10, threshold=0.1,
                                 overlap=0.5)
        elif self.image_type == "confocal":
            # assuming cell_image has single color channel
            blobs_log = blob_log(self.cell_image, min_sigma=3, max_sigma=20,
                                 num_sigma=10, threshold=0.1, overlap=0.5)

            def eliminate_border_blobs(blobs_log):
                # find the blobs too close to border so as to eliminate them
                border_x = self.cell_image.shape[1] / 5
                border_y = self.cell_image.shape[0] / 5

                filtered_blobs = blobs_log[(border_x < blobs_log[:, 1]) &
                                           (blobs_log[:, 1] < 4*border_x) &
                                           (border_y < blobs_log[:, 0]) &
                                           (blobs_log[:, 0] < 4*border_y)]

                return filtered_blobs

            blobs_log = eliminate_border_blobs(blobs_log)

            if len(blobs_log) < 1:
                # if none of the blobs remain after border blob elimination,
                # try blob_log with less stringent parameters
                blobs_log = blob_log(self.cell_image, min_sigma=2, max_sigma=20,
                                     num_sigma=10, threshold=0.1, overlap=0.5)
                blobs_log = eliminate_border_blobs(blobs_log)

        return blobs_log

    def centre_of_mass(self, blobs):
        # finds centre of mass of the multiple blobs detected

        # find the blob with highest intensity value
        ixs = np.indices(self.gray_cell_image.shape)

        n_blobs = blobs.shape[0]
        blob_centres = blobs[:, 0:2]
        blob_radii = blobs[:, 2]

        centres = blob_centres[..., np.newaxis, np.newaxis]
        radii = np.square(blob_radii)[:, np.newaxis, np.newaxis]
        # Using the formula for a circle, `x**2 + y**2 < r**2`,
        # generate a mask for all blobs.
        mask = np.square(ixs - centres).sum(axis=1) < radii
        # Calculate the average intensity of pixels under the mask
        blob_intensities = np.full(
            (n_blobs, *ixs.shape[1:]), self.gray_cell_image)
        blob_intensities = (blob_intensities * mask).reshape(mask.shape[0], -1)
        blob_intensities = np.divide(
            blob_intensities.sum(1), (blob_intensities != 0).sum(1))

        if self.image_type == "DAB":
            max_intensity = blob_centres[np.argmin(blob_intensities)]

            return max_intensity

        elif self.image_type == "confocal":
            max_radius = blob_centres[np.argmax(blob_radii)]
            max_intensity = blob_centres[np.argmax(blob_intensities)]
            if len(blob_radii) > len(set(blob_radii)):
                return max_intensity
            return max_radius

    def get_soma(self):
        # calculate pixel position to be attribute as soma

        soma_blobs = self.get_blobs()

        if len(soma_blobs) == 1:
            soma = soma_blobs[0][:2]
        if len(soma_blobs) > 1:
            soma = self.centre_of_mass(soma_blobs)

        return soma

    def threshold_image(self):
        if self.reference_image is not None:
            self.gray_reference_image = skimage.color.rgb2gray(
                self.reference_image)
            self.gray_cell_image = skimage.transform.match_histograms(
                self.gray_cell_image, self.gray_reference_image)

        # Contrast stretching
        p2, p98 = np.percentile(self.gray_cell_image, (2, 98))
        img_rescale = skimage.exposure.rescale_intensity(
            self.gray_cell_image, in_range=(p2, p98))

        thresholded_cell = img_rescale > threshold_otsu(img_rescale)

        if self.image_type == "DAB":
            return thresholded_cell
        elif self.image_type == "confocal":
            return skimage.util.invert(thresholded_cell)

    def label_objects(self):
        bw = closing(self.inverted_thresholded_image, square(1))
        # label image regions
        labelled_image, no_of_objects = skimage.measure.label(
            bw, return_num=True)

        return labelled_image, no_of_objects

    def remove_small_object_noise(self):
        labelled_image, no_of_objects = self.label_objects()
        # TODO: decide removal of unused variable
        labelled_image_1D = labelled_image.reshape(labelled_image.size)
        object_areas = np.bincount(labelled_image_1D[labelled_image_1D != 0])

        largest_object_label = np.argmax(object_areas)
        astrocyte_image = np.where(
            labelled_image == [largest_object_label], 1, 0)

        return astrocyte_image

    def fill_holes(self):
        return scipy.ndimage.binary_fill_holes(self.cleaned_image).astype(int)

    def surface_area(self):
        return np.sum(self.cleaned_image_filled_holes)


class Skeleton:
    """
    Skeletonize the thresholded image and extract relevant features from the 2D skeleton.
    """

    def __init__(self, cell_image, image_type):
        """
        Args:

        cell_image: RGB cell image
        image_type: 'confocal' or 'DAB'

        """

        self.cell_image = cell_image
        self.astrocyte = Cell(cell_image, image_type)
        self.cleaned_image = self.astrocyte.cleaned_image_filled_holes
        self.soma = self.astrocyte.get_soma()
        self.cell_skeleton = self.skeletonization()
        self.soma_on_skeleton = self.get_soma_on_skeleton()
        self.padded_skeleton = self.pad_skeleton()
        self.classify_branching_structure()

    def distance(self, P1, P2):
        # find eucledian distance between two pixel positions
        return ((P1[0] - P2[0])**2 + (P1[1] - P2[1])**2) ** 0.5

    def skeletonization(self):
        # perform skeletonization
        return skimage.morphology.skeletonize(self.cleaned_image)

    def pad_skeleton(self):

        # get all the pixel indices representing skeleton
        skeleton_indices = np.nonzero(self.cell_skeleton)

        # get corner points enclosing skeleton
        x_min, x_max = min(skeleton_indices[1]), max(skeleton_indices[1])
        y_min, y_max = min(skeleton_indices[0]), max(skeleton_indices[0])
        self.bounded_skeleton = self.cell_skeleton[y_min:y_max, x_min:x_max]

        pad_width = max(self.bounded_skeleton.shape)//2
        self.bounded_skeleton_boundary = [x_min, x_max, y_min, y_max]

        # get updated soma position on bounded and padded skeleton
        self.soma_on_bounded_skeleton = self.soma_on_skeleton[0] - \
            y_min, self.soma_on_skeleton[1]-x_min
        self.soma_on_padded_skeleton = self.soma_on_skeleton[0] - \
            y_min+pad_width, self.soma_on_skeleton[1]-x_min+pad_width

        return skimage.util.pad(self.bounded_skeleton, pad_width=pad_width, mode='constant')

    def get_soma_on_skeleton(self):
        skeleton_pixel_coordinates = [(i, j) for (
            i, j), val in np.ndenumerate(self.cell_skeleton) if val != 0]
        soma_on_skeleton = min(skeleton_pixel_coordinates,
                               key=lambda x: self.distance(self.soma, x))

        return soma_on_skeleton

    def total_length(self):
        return np.sum(self.cell_skeleton)

    def avg_process_thickness(self):
        return round((self.astrocyte.surface_area()/self.total_length()), 1)

    def convex_hull(self, plot=False):
        convex_hull = skimage.morphology.convex_hull_image(self.cell_skeleton)
        if plot == True:
            fig, ax = plt.subplots()
            ax.set_axis_off()
            ax.imshow(convex_hull)

        return np.sum(convex_hull)

    def get_no_of_forks(self, plot=False):

        # get the degree for every cell pixel (no. of neighbouring pixels)
        pixel_graph, coordinates, degrees = skeleton_to_csgraph(
            self.cell_skeleton)
        # array of all pixel locations with degree more than 2
        fork_image = np.where(degrees > [2], 1, 0)
        s = scipy.ndimage.generate_binary_structure(2, 2)
        labeled_array, num_forks = scipy.ndimage.label(fork_image, structure=s)

        if plot == True:
            fork_indices = np.where(degrees > [2])
            fork_coordinates = zip(fork_indices[0], fork_indices[1])

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.set_title('path')
            ax.imshow(self.cell_skeleton, interpolation='nearest')

            for i in fork_coordinates:
                c = plt.Circle((i[1], i[0]), 0.6, color='green')
                ax.add_patch(c)

            ax.set_axis_off()
            plt.tight_layout()
            plt.show()

        return num_forks

    def eliminate_loops(self, branch_statistics, paths_list):
        loop_indexes = []
        loop_branch_end_points = []

        # set that keeps track of what elements have been added
        seen = set()
        # eliminate loops from branch statistics
        for branch_no, branch in enumerate(branch_statistics):
            # If element not in seen, add it to both
            current = (branch[0], branch[1])
            if current not in seen:
                seen.add(current)
            elif current in seen:
                # for deleting the loop index from branch statistics
                loop_indexes.append(branch_no)
                # for deleting the paths from paths list by recognizing loop end points
                loop_branch_end_points.append((int(branch[0]), int(branch[1])))

        new_branch_statistics = np.delete(
            branch_statistics, loop_indexes, axis=0)

        # eliminate loops from paths list
        path_indexes = []
        for loop_end_points in loop_branch_end_points:
            for path_no, path in enumerate(paths_list):
                if loop_end_points[0] == path[0] and loop_end_points[1] == path[-1] or loop_end_points[0] == path[-1] and loop_end_points[1] == path[0]:
                    path_indexes.append(path_no)
                    break

        new_paths_list = np.delete(np.array(paths_list), path_indexes, axis=0)

        return new_branch_statistics, new_paths_list

    def branch_structure(self, junctions, branch_statistics, paths_list):
        next_set_junctions = []
        next_set_branches = []
        terminal_branches = []

        for junction in junctions:
            branches_travelled = []
            for branch_no, branch in enumerate(branch_statistics):
                if branch[0] == junction:
                    if branch[3] == 2:
                        next_set_junctions.append(branch[1])
                        for path in paths_list:
                            if branch[0] == path[0] and branch[1] == path[-1] or branch[0] == path[-1] and branch[1] == path[0]:
                                next_set_branches.append(path)
                                branches_travelled.append(branch_no)
                    if branch[3] == 1:
                        for path in paths_list:
                            if branch[0] == path[0] and branch[1] == path[-1] or branch[0] == path[-1] and branch[1] == path[0]:
                                terminal_branches.append(path)
                                next_set_branches.append(path)
                                branches_travelled.append(branch_no)
                elif branch[1] == junction:
                    if branch[3] == 2:
                        next_set_junctions.append(branch[0])
                        for path in paths_list:
                            if branch[0] == path[0] and branch[1] == path[-1] or branch[0] == path[-1] and branch[1] == path[0]:
                                next_set_branches.append(path)
                                branches_travelled.append(branch_no)
                    if branch[3] == 1:
                        for path in paths_list:
                            if branch[0] == path[0] and branch[1] == path[-1] or branch[0] == path[-1] and branch[1] == path[0]:
                                terminal_branches.append(path)
                                next_set_branches.append(path)
                                branches_travelled.append(branch_no)
            branch_statistics = np.delete(
                branch_statistics, branches_travelled, axis=0)

        return next_set_junctions, next_set_branches, terminal_branches, branch_statistics

    def classify_branching_structure(self, plot=False):

        def get_soma_node():
            near = []
            for i in range(skan.csr.Skeleton(self.cell_skeleton).n_paths):
                path_coords = skan.csr.Skeleton(
                    self.cell_skeleton).path_coordinates(i)
                nearest = min(path_coords, key=lambda x: self.distance(
                    self.soma_on_skeleton, x))
                near.append(nearest)

            soma_on_path = min(near, key=lambda x: self.distance(
                self.soma_on_skeleton, x))

            for i, j in enumerate(skan.csr.Skeleton(self.cell_skeleton).coordinates):
                if all(soma_on_path == j):
                    soma_node = [i]
                    break

            return soma_node

        def get_soma_branches(soma_node, paths_list):
            soma_branches = []
            for path in paths_list:
                # print(path)
                if soma_node in path:
                    soma_branches.append(path)
            return soma_branches

        pixel_graph, coordinates, degrees = skeleton_to_csgraph(
            self.cell_skeleton)
        branch_statistics = skan.csr.branch_statistics(pixel_graph)
        paths_list = skan.csr.Skeleton(self.cell_skeleton).paths_list()

        terminal_branches = []
        branching_structure_array = []
        # get branches containing soma node

        soma_node = get_soma_node()
        soma_branches = get_soma_branches(soma_node, paths_list)
        if len(soma_branches) > 2:
            junctions = soma_node
            delete_soma_branch = False
        else:
            # collect first level/primary branches
            junctions = [soma_branches[0][0], soma_branches[0][-1]]
            delete_soma_branch = True

        # eliminate loops in branches and path lists
        branch_statistics, paths_list = self.eliminate_loops(
            branch_statistics, paths_list)

        while True:
            junctions, branches, terminal_branch, branch_statistics = self.branch_structure(
                junctions, branch_statistics, paths_list)
            branching_structure_array.append(branches)
            terminal_branches.extend(terminal_branch)
            if len(junctions) == 0:
                break

        if delete_soma_branch == True:
            branching_structure_array[0].remove(soma_branches[0])

        if plot == True:
            # store same level branch nodes in single array
            color_branches_coords = []
            for branch_level in branching_structure_array:
                single_branch_level = []
                for path in branch_level:
                    path_coords = []
                    for node in path:
                        path_coords.append(coordinates[node])
                        single_branch_level.extend(path_coords)
                color_branches_coords.append(single_branch_level)

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.set_title('path')
            ax.imshow(self.cell_skeleton, interpolation='nearest')

            color_codes = ['red', 'blue', 'magenta', 'green', 'cyan']
            for j, color_branch in enumerate(color_branches_coords):
                if j > 4:
                    j = 4
                for k in color_branch:
                    c = plt.Circle((k[1], k[0]), 0.5, color=color_codes[j])
                    ax.add_patch(c)

            ax.set_axis_off()
            plt.tight_layout()
            plt.show()

        self.branching_structure_array = branching_structure_array
        self.terminal_branches = terminal_branches

    def get_primary_branches(self):
        primary_branches = self.branching_structure_array[0]
        no_of_primary_branches = len(primary_branches)
        avg_length_of_primary_branches = 0 if no_of_primary_branches == 0 else sum(
            map(len, primary_branches))/float(len(primary_branches))

        return primary_branches, no_of_primary_branches, round(avg_length_of_primary_branches, 1)

    def get_secondary_branches(self):
        try:
            secondary_branches = self.branching_structure_array[1]
        except IndexError:
            secondary_branches = []
        no_of_secondary_branches = len(secondary_branches)
        avg_length_of_secondary_branches = 0 if no_of_secondary_branches == 0 else sum(
            map(len, secondary_branches))/float(len(secondary_branches))

        return secondary_branches, no_of_secondary_branches, round(avg_length_of_secondary_branches, 1)

    def get_tertiary_branches(self):
        try:
            tertiary_branches = self.branching_structure_array[2]
        except IndexError:
            tertiary_branches = []
        no_of_tertiary_branches = len(tertiary_branches)
        avg_length_of_tertiary_branches = 0 if no_of_tertiary_branches == 0 else sum(
            map(len, tertiary_branches))/float(len(tertiary_branches))

        return tertiary_branches, no_of_tertiary_branches, round(avg_length_of_tertiary_branches, 1)

    def get_quatenary_branches(self):
        try:
            quatenary_branches = self.branching_structure_array[3:]
        except IndexError:
            quatenary_branches = []
        quatenary_branches = [
            branch for branch_level in quatenary_branches for branch in branch_level]
        no_of_quatenary_branches = len(quatenary_branches)
        avg_length_of_quatenary_branches = 0 if no_of_quatenary_branches == 0 else sum(
            map(len, quatenary_branches))/float(len(quatenary_branches))

        return quatenary_branches, no_of_quatenary_branches, round(avg_length_of_quatenary_branches, 1)

    def get_terminal_branches(self):
        terminal_branches = self.terminal_branches
        no_of_terminal_branches = len(terminal_branches)
        avg_length_of_terminal_branches = 0 if no_of_terminal_branches == 0 else sum(
            map(len, terminal_branches))/float(len(terminal_branches))

        return terminal_branches, no_of_terminal_branches, round(avg_length_of_terminal_branches, 1)


class Sholl:
    """
    Extract radius and no. of intersections for sholl analyses and other relevant features from the resulting sholl plot. 
    """

    def __init__(self, cell_image, image_type, shell_step_size, polynomial_degree=3):
        """
        Args:

        cell_image: RGB cell image
        image_type: 'confocal' or 'DAB'
        shell_step_size: pixel difference between concentric circles for sholl analysis
        polynomial_degree (scalar): degree of polynomial for fitting regression model on sholl values

        """

        self.shell_step_size = shell_step_size
        self.polynomial_degree = polynomial_degree
        self.skeleton = Skeleton(cell_image, image_type)
        self.bounded_skeleton = self.skeleton.bounded_skeleton
        self.soma_on_bounded_skeleton = self.skeleton.soma_on_bounded_skeleton
        self.padded_skeleton = self.skeleton.padded_skeleton
        self.soma_on_padded_skeleton = self.skeleton.soma_on_padded_skeleton
        self.distances_from_soma = self.sholl_results()[0]
        self.no_of_intersections = self.sholl_results()[1]
        self.polynomial_model = self.polynomial_fit()
        self.determination_ratio()

    def concentric_coords_and_values(self):
        # concentric_coordinates: {radius values: [pixel coordinates on that radius]}
        # no_of_intersections: {radius values: no_of_intersection values}

        largest_radius = int(1.3*(np.max([self.soma_on_bounded_skeleton[1], abs(self.soma_on_bounded_skeleton[1]-self.bounded_skeleton.shape[1]),
                                          self.soma_on_bounded_skeleton[0], abs(self.soma_on_bounded_skeleton[0]-self.bounded_skeleton.shape[0])])))
        # {100: [(10,10), ..] , 400: [(20,20), ..]}
        concentric_coordinates = defaultdict(list)
        concentric_coordinates_intensities = defaultdict(list)
        concentric_radiuses = [radius for radius in range(
            self.shell_step_size, largest_radius, self.shell_step_size)]

        for (x, y), value in np.ndenumerate(self.padded_skeleton):
            for radius in concentric_radiuses:
                lhs = (
                    x - self.soma_on_padded_skeleton[0])**2 + (y - self.soma_on_padded_skeleton[1])**2
                if abs((np.sqrt(lhs)-radius)) < 0.9:
                    concentric_coordinates[radius].append((x, y))
                    concentric_coordinates_intensities[radius].append(value)

        # array with intersection values corresponding to radii
        no_of_intersections = defaultdict()
        for radius, val in concentric_coordinates_intensities.items():
            intersec_indicies = []
            indexes = [i for i, x in enumerate(val) if x]
            for index in indexes:
                intersec_indicies.append(concentric_coordinates[radius][index])
            img = np.zeros(self.padded_skeleton.shape)
            intersections = []
            for i, j in enumerate(intersec_indicies):
                img[j] = 1
            label_image = label(img)
            no_of_intersections[radius] = np.amax(label_image)

        return concentric_coordinates, no_of_intersections

    def sholl_results(self, plot=False):
        # return sholl radiuses and corresponding intersection values
        xs, ys = [], []
        concentric_coordinates, no_of_intersections = self.concentric_coords_and_values()
        for rad, val in no_of_intersections.items():
            xs.append(rad)
            ys.append(val)
        order = np.argsort(xs)

        if plot == True:
            astrocyte_skeleton_copy = copy.deepcopy(self.padded_skeleton)
            for radius, coordinates in concentric_coordinates.items():
                for coord in coordinates:
                    cell_image_with_circles = astrocyte_skeleton_copy
                    cell_image_with_circles[coord[0], coord[1]] = 1

            # plot circles on skeleton
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(cell_image_with_circles)
            # overlay soma on skeleton
            y, x = self.soma_on_padded_skeleton
            c = plt.Circle((x, y), 1, color='red')
            ax.add_patch(c)
            ax.set_axis_off()
            plt.tight_layout()
            plt.show()

            # plot sholl graph showing radius vs. no_of_intersections
            plt.plot(self.distances_from_soma, self.no_of_intersections)
            plt.xlabel("Distance from centre")
            plt.ylabel("No. of intersections")
            plt.show()

        return np.array(xs)[order], np.array(ys)[order]

    def polynomial_fit(self, plot=False):
        # Linear polynomial regression to describe the relationship between intersections vs. distance

        # till last non-zero value
        last_intersection_index = np.max(np.nonzero(self.no_of_intersections))
        self.non_zero_no_of_intersections = self.no_of_intersections[:last_intersection_index]
        self.non_zero_distances_from_soma = self.distances_from_soma[:last_intersection_index]

        y_data = self.non_zero_no_of_intersections
        reshaped_x = self.non_zero_distances_from_soma.reshape((-1, 1))

        x_ = preprocessing.PolynomialFeatures(
            degree=self.polynomial_degree, include_bias=False).fit_transform(reshaped_x)
        # create a linear regression model
        polynomial_model = linear_model.LinearRegression().fit(x_, y_data)

        self.polynomial_predicted_no_of_intersections = polynomial_model.predict(
            x_)

        if plot == True:
            # predict y from the data
            x_new = self.non_zero_distances_from_soma
            y_new = polynomial_model.predict(x_)
            # plot the results
            plt.figure(figsize=(4, 3))
            ax = plt.axes()
            ax.scatter(reshaped_x, y_data)
            ax.plot(x_new, y_new)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.axis('tight')
            plt.show()

        return polynomial_model

    def enclosing_radius(self):
        # index of last non-zero value in the array containing radii
        return self.non_zero_distances_from_soma[len(self.non_zero_no_of_intersections) - (self.non_zero_no_of_intersections != 0)[::-1].argmax() - 1]

    def critical_radius(self):
        # radii_array[index of the max value in the array for no_of_intersections (polynomial plot)]
        return self.non_zero_distances_from_soma[np.argmax(self.polynomial_predicted_no_of_intersections)]

    def critical_value(self):
        # local maximum of the polynomial fit (Maximum no. of intersections)
        return round(np.max(self.polynomial_predicted_no_of_intersections), 2)

    def skewness(self):
        # indication of how symmetrical the polynomial distribution is around its mean
        x_ = preprocessing.PolynomialFeatures(degree=3, include_bias=False).fit_transform(
            self.non_zero_no_of_intersections.reshape((-1, 1)))
        return round(scipy.stats.skew(self.polynomial_model.predict(x_)), 2)

    def schoenen_ramification_index(self):
        # ratio between critical value and number of primary branches
        no_of_primary_branches = self.skeleton.get_primary_branches()[1]
        schoenen_ramification_index = self.critical_value()/no_of_primary_branches
        return round(schoenen_ramification_index, 2)

    def semi_log(self):
        # no. of intersections/circumference
        normalized_y = np.log(
            self.non_zero_no_of_intersections/(2*np.pi*self.non_zero_distances_from_soma))
        reshaped_x = self.non_zero_distances_from_soma.reshape((-1, 1))
        model = linear_model.LinearRegression().fit(reshaped_x, normalized_y)

        # predict y from the data
        x_new = self.non_zero_distances_from_soma
        y_new = model.predict(reshaped_x)
        r2 = model.score(reshaped_x, normalized_y)
        regression_intercept = model.intercept_
        regression_coefficient = -model.coef_[0]

        return r2, regression_intercept, regression_coefficient

    def log_log(self):
        # no. of intersections/circumference
        normalized_y = np.log(
            self.non_zero_no_of_intersections/(2*np.pi*self.non_zero_distances_from_soma))
        reshaped_x = self.non_zero_distances_from_soma.reshape((-1, 1))
        normalized_x = np.log(reshaped_x)
        model = linear_model.LinearRegression().fit(normalized_x, normalized_y)

        # predict y from the data
        x_new = normalized_x
        y_new = model.predict(normalized_x)
        r2 = model.score(normalized_x, normalized_y)
        regression_intercept = model.intercept_
        regression_coefficient = -model.coef_[0]

        return r2, regression_intercept, regression_coefficient

    def determination_ratio(self):
        semi_log_r2 = self.semi_log()[0]
        log_log_r2 = self.log_log()[0]
        determination_ratio = semi_log_r2/log_log_r2
        if determination_ratio > 1:
            self.normalization_method = "Semi-log"
        else:
            self.normalization_method = "Log-log"

    def coefficient_of_determination(self):
        # how close the data are to the fitted regression (indicative of the level of explained variability in the data set)
        if self.normalization_method == "Semi-log":
            return round(self.semi_log()[0], 2)
        else:
            return round(self.log_log()[0], 2)

    def regression_intercept(self):
        # Y intercept of the logarithmic plot
        if self.normalization_method == "Semi-log":
            return round(self.semi_log()[1], 2)
        else:
            return round(self.log_log()[1], 2)

    def sholl_regression_coefficient(self):
        # rate of decay of no. of branches
        if self.normalization_method == "Semi-log":
            return round(self.semi_log()[2], 2)
        else:
            return round(self.log_log()[2], 2)


class analyze_cells:
    """
    Extract features of all the cells in different groups and implement PCA and group level sholl analysis.
    """

    def __init__(self, groups_folders, image_type, label, save_features=True, show_sholl_plot=True, shell_step_size=3):
        """
        Args:

        groups_folders (list): name of the input data folders corresponding to different subgroups
        image_type (string): 'confocal' or 'DAB'
        label (dictionary): group labels to be used for pca plots
        save_features (True/False): create text files containing feature values
        show_sholl_plot (True/False): show group sholl plot
        shell_step_size (scalar): difference between concentric circles

        """

        self.show_sholl_plot = show_sholl_plot
        self.image_type = image_type
        self.label = label
        self.shell_step_size = shell_step_size

        dataset = self.read_images(groups_folders)
        self.features = self.get_features(dataset)
        self.feature_names = ['surface_area', 'total_length', 'avg_process_thickness', 'convex_hull', 'no_of_forks', 'no_of_primary_branches', 'no_of_secondary_branches',
                              'no_of_tertiary_branches', 'no_of_quatenary_branches', 'no_of_terminal_branches', 'avg_length_of_primary_branches', 'avg_length_of_secondary_branches',
                              'avg_length_of_tertiary_branches', 'avg_length_of_quatenary_branches', 'avg_length_of_terminal_branches',
                              'critical_radius', 'critical_value', 'enclosing_radius', 'ramification_index', 'skewness', 'coefficient_of_determination',
                              'sholl_regression_coefficient', 'regression_intercept']
        if save_features == True:
            self.save_features()
        if show_sholl_plot == True:
            self.show_avg_sholl_plot(shell_step_size)

    def read_images(self, groups_folders):
        self.file_names = []
        dataset = []
        for group in groups_folders:
            group_data = []
            for file in os.listdir(group):
                if not file.startswith('.'):
                    self.file_names.append((group+'/'+file))
                    image = io.imread(group+'/'+file)
                    group_data.append(image)
            dataset.append(group_data)

        return dataset

    def get_features(self, dataset):
        dataset_features = []
        self.targets = []

        if self.show_sholl_plot == True:
            self.sholl_original_plots = []
            self.sholl_polynomial_plots = []
            self.polynomial_models = []

        self.group_counts = []
        cell_count = 0
        for group_no, group in enumerate(dataset):
            group_cell_count = 0
            for cell_no, cell_image in enumerate(group):

                print(self.file_names[cell_count])

                cell_count += 1
                group_cell_count += 1

                self.targets.append(group_no)

                cell_features = []

                astrocyte = Cell(cell_image, self.image_type)
                skeleton = Skeleton(cell_image, self.image_type)
                sholl = Sholl(cell_image, self.image_type,
                              self.shell_step_size)

                cell_features.append(astrocyte.surface_area())
                cell_features.append(skeleton.total_length())
                cell_features.append(skeleton.avg_process_thickness())
                cell_features.append(skeleton.convex_hull())
                cell_features.append(skeleton.get_no_of_forks())
                for i in range(1, 3):
                    cell_features.append(skeleton.get_primary_branches()[i])
                    cell_features.append(skeleton.get_secondary_branches()[i])
                    cell_features.append(skeleton.get_tertiary_branches()[i])
                    cell_features.append(skeleton.get_quatenary_branches()[i])
                    cell_features.append(skeleton.get_terminal_branches()[i])
                cell_features.append(sholl.critical_radius())
                cell_features.append(sholl.critical_value())
                cell_features.append(sholl.enclosing_radius())
                cell_features.append(sholl.schoenen_ramification_index())
                cell_features.append(sholl.skewness())
                cell_features.append(sholl.coefficient_of_determination())
                cell_features.append(sholl.sholl_regression_coefficient())
                cell_features.append(sholl.regression_intercept())

                if self.show_sholl_plot == True:
                    self.sholl_original_plots.append(
                        (sholl.distances_from_soma, sholl.no_of_intersections))
                    self.sholl_polynomial_plots.append(
                        (sholl.non_zero_distances_from_soma, sholl.non_zero_no_of_intersections))
                    self.polynomial_models.append(sholl.polynomial_model)

                dataset_features.append(cell_features)

            self.group_counts.append(group_cell_count)
        return dataset_features

    def save_features(self):
        directory = os.getcwd()+'/Features'
        if os.path.exists(directory) and os.path.isdir(directory):
            shutil.rmtree(directory)
            os.mkdir(directory)
        else:
            os.mkdir(directory)

        def save_to_file(file_name, feature_name, feature_value):
            path = os.getcwd()+'/Features/'
            with open(path+feature_name+'.txt', 'a') as text_file:
                text_file.write("{} {} \n".format(file_name, feature_value))

        for cell_no, cell_features in enumerate(self.features):
            for feature_no, feature_val in enumerate(cell_features):
                save_to_file(
                    self.file_names[cell_no], self.feature_names[feature_no], feature_val)

    def show_avg_sholl_plot(self, shell_step_size):
        original_plots_file = 'Original plots'
        polynomial_plots_file = 'Polynomial plots'

        directory = os.getcwd()+'/Sholl Results'

        if os.path.exists(directory) and os.path.isdir(directory):
            shutil.rmtree(directory)
            os.mkdir(directory)
        else:
            os.mkdir(directory)

        path = os.getcwd()+'/Sholl Results/'

        largest_radius = []
        no_of_intersections = []

        with open(path+original_plots_file, 'w+') as text_file:
            for cell_no, plot in enumerate(self.sholl_original_plots):
                text_file.write("{} {} {} \n".format(
                    self.file_names[cell_no], plot[0], plot[1]))

                # # get the max radius of each cell, as smallest and mid-level ones can be inferred from shell_step_size
                # largest_radius.append(max(plot[0]))
                # no_of_intersections.append(plot[1])

        with open(path+polynomial_plots_file, 'w+') as text_file:
            for cell_no, plot in enumerate(self.sholl_polynomial_plots):
                text_file.write("{} {} {} \n".format(
                    self.file_names[cell_no], plot[0], plot[1]))

                # get the max radius of each cell, as smallest and mid-level ones can be inferred from shell_step_size
                largest_radius.append(max(plot[0]))
                no_of_intersections.append(plot[1])

        group_radiuses = []
        sholl_intersections = []
        for group_no, count in enumerate(self.group_counts):
            group_count = sum(self.group_counts[:group_no+1])
            group_radius = max(largest_radius[group_count-count:group_count])
            group_radiuses.append(group_radius)

            current_intersections = no_of_intersections[group_count -
                                                        count:group_count]
            current_radiuses = range(
                shell_step_size, group_radius+1, shell_step_size)

            intersection_dict = defaultdict(list)
            for intersections in current_intersections:
                for i, intersection_val in enumerate(intersections):
                    intersection_dict[current_radiuses[i]].append(
                        intersection_val)
            sholl_intersections.append(intersection_dict)

        with open(path+"Sholl values", 'w') as text_file:
            for group_no, group_sholl in enumerate(sholl_intersections):
                text_file.write("Group: {}\n".format(group_no))
                for radius, intersections in group_sholl.items():
                    text_file.write("{} {}\n".format(radius, intersections))

        for group_no, group_sholl in enumerate(sholl_intersections):
            x = []
            y = []
            e = []
            for radius, intersections in group_sholl.items():
                x.append(radius)
                intersections = (
                    intersections + self.group_counts[group_no] * [0])[:self.group_counts[group_no]]
                y.append(np.mean(intersections))
                e.append(scipy.stats.sem(intersections))
            plt.errorbar(x, y, yerr=e, label=self.label[group_no])

        plt.xlabel("Distance from soma")
        plt.ylabel("No. of intersections")
        plt.legend()
        plt.show()

    def pca(self, color_dict, marker):

        self.marker = marker

        def get_cov_ellipse(cov, centre, nstd, **kwargs):
            """
            Return a matplotlib Ellipse patch representing the covariance matrix
            cov centred at centre and scaled by the factor nstd.

            """

            # Find and sort eigenvalues and eigenvectors into descending order
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = eigvals.argsort()[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]

            # The anti-clockwise angle to rotate our ellipse by
            vx, vy = eigvecs[:, 0][0], eigvecs[:, 0][1]
            theta = np.arctan2(vy, vx)

            # Width and height of ellipse to draw
            width, height = nstd * np.sqrt(eigvals)

            return Ellipse(xy=centre, width=width, height=height, angle=np.degrees(theta), **kwargs)

        pca_object = decomposition.PCA(2)

        # Scale data
        scaler = preprocessing.MaxAbsScaler()
        scaler.fit(self.features)
        X = scaler.transform(self.features)

        # fit on data
        pca_object.fit(X)

        # access values and vectors
        self.feature_significance = pca_object.components_

        # variance captured by principal components
        first_component_var = pca_object.explained_variance_ratio_[0]
        second_component_var = pca_object.explained_variance_ratio_[1]

        # transform data
        self.projected = pca_object.transform(X)

        first_component = self.projected[:, 0]
        second_component = self.projected[:, 1]

        with open("pca values", 'w') as text_file:
            text_file.write("First component:\n{}\nSecond component:\n{}".format(
                first_component, second_component))

        no_of_std = 3  # no. of standard deviations to show
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('white')
        for l in np.unique(self.targets):
            ix = np.where(self.targets == l)
            first_component_mean = np.mean(first_component[ix])
            second_component_mean = np.mean(second_component[ix])
            cov = np.cov(first_component, second_component)
            ax.scatter(first_component[ix], second_component[ix],
                       c=color_dict[l], s=40, label=self.label[l], marker=marker[l])
            e = get_cov_ellipse(
                cov, (first_component_mean, second_component_mean), no_of_std, fc=color_dict[l], alpha=0.4)
            ax.add_artist(e)

        plt.xlabel("PC 1 (Variance: %.1f%%)" %
                   (first_component_var*100), fontsize=14)
        plt.ylabel("PC 2 (Variance: %.1f%%)" %
                   (second_component_var*100), fontsize=14)
        plt.legend()
        plt.show()

    def plot_feature_histograms(self):
        # 2 columns each containing 13 figures, total 22 features
        fig, axes = plt.subplots(12, 2, figsize=(15, 12))
        data = np.array(self.features)
        ko = data[np.where(np.array(self.targets) == 0)[0]]  # define ko
        control = data[np.where(np.array(self.targets) == 1)[
            0]]  # define control
        ax = axes.ravel()  # flat axes with numpy ravel

        for i in range(len(self.feature_names)):
            _, bins = np.histogram(data[:, i], bins=40)
            # red color for malignant class
            ax[i].hist(ko[:, i], bins=bins, color='r', alpha=.5)
            # alpha is for transparency in the overlapped region
            ax[i].hist(control[:, i], bins=bins, color='g', alpha=0.3)
            ax[i].set_title(self.feature_names[i], fontsize=9)
            # the x-axis co-ordinates are not so useful, as we just want to look how well separated the histograms are
            ax[i].axes.get_xaxis().set_visible(False)
            ax[i].set_yticks(())

        ax[0].legend(self.marker, loc='best', fontsize=8)
        plt.tight_layout()  # let's make good plots
        plt.show()

    def plot_feature_significance_heatmap(self):
        sorted_significance_order = np.flip(
            np.argsort(abs(self.feature_significance[0])))
        sorted_feature_significance = np.zeros(self.feature_significance.shape)
        sorted_feature_significance[0] = np.array(self.feature_significance[0])[
            sorted_significance_order]
        sorted_feature_significance[1] = np.array(self.feature_significance[1])[
            sorted_significance_order]
        sorted_feature_names = np.array(self.feature_names)[
            sorted_significance_order]

        plt.matshow(np.array(sorted_feature_significance), cmap='gist_heat')
        plt.yticks([0, 1], ['1st Comp', '2nd Comp'], fontsize=10)
        plt.colorbar()
        plt.xticks(range(len(sorted_feature_names)),
                   sorted_feature_names, rotation=65, ha='left')
        plt.show()

    def plot_feature_significance_vectors(self):
        score = self.projected
        coeff = np.transpose(self.feature_significance)
        labels = self.feature_names
        xs = score[:, 0]
        ys = score[:, 1]
        n = coeff.shape[0]
        scalex = 1.0/(xs.max() - xs.min())
        scaley = 1.0/(ys.max() - ys.min())

        plt.figure(figsize=(10, 9))
        ax = plt.axes()
        ax.scatter(xs * scalex, ys * scaley, c=self.targets)
        for i in range(n):
            ax.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
            if labels is None:
                ax.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15,
                        "Var"+str(i+1), color='g', ha='center', va='center')
            else:
                ax.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15,
                        labels[i], color='g', ha='center', va='center')
        ax.set_xlabel("PC {}".format(1))
        ax.set_ylabel("PC {}".format(2))
        plt.show()

    def get_clusters(self, k=None, use_features=True, n_PC=None, plot='parallel'):
        """
        Highly configurable K-Means clustering & visualization of cell data.

        Parameters
        ----------
        k : int or None, default None
            If greater than 1, return k number of clusters. Else if None it is
            autoselected, on basis of maximum Calinski Harabasz Score.
        use_features : bool, default True
            If True, clustering would use original set of morphometric features.
        n_PC : int or None, default None
            If greater than 1, return n_PC number of Principal Components after
            clustering. If None & use_features is False, it's autoselected as
            number of Principal Components calculated.
        plot : str or None, default 'parallel'
            The type of plot user would like to get, either parallel or scatter.

        Returns
        -------
        centers_df
            A DataFrame with normalized center coordinates of each cluster.
        df
            A DataFrame with normalized cell coordinates & the respective
            cluster to which they belong.
        """
        n_cells = len(self.features)
        seed = np.random.randint(0, n_cells)  # for deterministic randomness

        if k != None and (k < 2 or k > n_cells):
            raise ValueError('Number of clusters, k, must be greater than 1 & '
                             'lesser than the total number of cells.')

        def compute_clusters(n_clusters, normalized_features, max_clusters):
            best_variance_ratio, best_k, best_model = 0, 0, None
            K = range(2, max_clusters +
                      1) if n_clusters == None else [n_clusters]

            # Either autoselect least varying K-Means model or use specified k
            for k in K:
                # convergence is costly, better to prespecify k
                model = KMeans(n_clusters=k, random_state=seed).fit(
                    normalized_features)
                variance_ratio = metrics.calinski_harabasz_score(
                    normalized_features, model.labels_)
                if variance_ratio > best_variance_ratio:
                    best_variance_ratio = variance_ratio
                    best_model = model
                    best_k = k

            return best_k, best_model, best_variance_ratio

        if use_features:
            if n_PC == None:
                scaler = preprocessing.MaxAbsScaler()
                features = self.features
                df = pd.DataFrame(scaler.fit(features).transform(features))
            else:
                raise ValueError('Cannot use morphological features & n_PC '
                                 'simultaneously')
        else:
            if not hasattr(self, 'projected'):
                raise RuntimeError('Principal Components must be found before '
                                   'clustering in their feature space.')

            projected = self.projected

            if n_PC is None:
                # autoselect n_PC as number of computed Principal Components
                n_PC = projected.shape[1]

            if 1 < n_PC <= projected.shape[1]:
                df = pd.DataFrame(projected[:, :n_PC])
            else:
                raise ValueError('Number of Principal Components, n_PC, should be '
                                 'greater than 1 & less than or equal to the total '
                                 'number of Principal Components of cells.')

        normalized_feature_vector = np.array(df)

        # TODO: this max_cluster value isn't feasible for large dataset
        k, kmeans_model, variance_ratio = compute_clusters(
            k, normalized_feature_vector, n_cells // 4)

        # creates a dataframe with a column for cluster number
        centers_df = pd.DataFrame(kmeans_model.cluster_centers_)
        centers_df['cluster'] = range(k)

        LABEL_COLOR_MAP = sns.color_palette(None, k)

        def parallel_plot(data):
            plt.figure(figsize=(15, 8)).gca().axes.set_ylim([-3, 3])
            parallel_coordinates(data, 'cluster',
                                 color=LABEL_COLOR_MAP, marker='o')

        def scatter_plot(data, centers):
            l_idx = r_idx = 0
            label_color = [LABEL_COLOR_MAP[l] for l in kmeans_model.labels_]

            if data.shape[1] == 2:
                MARKERS = cycle(['*', 'o', 's', 'p', '+', 'x', '^'])

                plt.scatter(centers[:, 0], centers[:, 1],
                            45, LABEL_COLOR_MAP, 'd')

                for cells in self.group_counts:
                    r_idx += cells
                    plt.scatter(data[l_idx:r_idx, 0], data[l_idx:r_idx, 1], 40,
                                label_color[l_idx:r_idx], next(MARKERS))
                    l_idx += cells

                plt.xlabel('PC1'), plt.ylabel('PC2')
                plt.show()
            elif data.shape[1] == 3:  # ipyvolume 3D scatter plot
                # assuming 2 classes: stab & ctrl
                MARKERS = cycle(['box', 'sphere', 'arrow', 'point_2d',
                                 'square_2d', 'triangle_2d', 'circle_2d'])

                ipv.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                            LABEL_COLOR_MAP, 5, 5.6, marker='diamond')

                for cells in self.group_counts:
                    r_idx += cells
                    ipv.scatter(data[l_idx:r_idx, 0], data[l_idx:r_idx, 1],
                                data[l_idx:r_idx, 2], label_color[l_idx:r_idx],
                                4, 4.6, marker=next(MARKERS))
                    l_idx += cells

                ipv.xyzlabel('PC1', 'PC2', 'PC3')
                ipv.show()

        if plot == 'parallel':
            parallel_plot(centers_df)
        elif plot == 'scatter':
            if n_PC in [2, 3]:
                scatter_plot(normalized_feature_vector,
                             kmeans_model.cluster_centers_)
            else:
                raise ValueError('Scatter plot can only be in 2D or 3D.')
        else:
            raise ValueError('Plot must be either of parallel or scatter.')

        print(f'k = {k} clusters with Variance Ratio = {variance_ratio}')
        print(f'seed = {seed}')
        print('Using', ('principal components',
                        'morphometric features')[use_features])

        df['cluster'] = kmeans_model.labels_

        return centers_df, df
