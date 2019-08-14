import skimage
import sklearn
import scipy
import numpy as np

import skimage.io as io
from skimage.color import rgb2gray
from collections import defaultdict
import math
import matplotlib.pyplot as plt
import csv
import copy
import os
import pylab as pl
from matplotlib import collections as mc
from math import sqrt
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, remove_small_objects
from skimage.color import label2rgb
import matplotlib.patches as mpatches
from scipy.ndimage import label, generate_binary_structure
from skimage.filters import threshold_otsu

import skan
from skan.pre import threshold
from skan import draw
from skan import skeleton_to_csgraph
from skan import Skeleton, summarize



def get_blobs(cell_image):
    inverted_cell_image = skimage.util.invert(cell_image)
    # dab stain
    blobs_log = blob_log(inverted_cell_image, min_sigma=6, max_sigma=20, num_sigma=10, threshold=0.1, overlap=0.5)
    # confocal
#     blobs_log = blob_log(inverted_cell_image, min_sigma=1, max_sigma=30, num_sigma=10, threshold=0.1, overlap=0)
#     blobs_log = blob_dog(inverted_cell_image, max_sigma=30, threshold=0.1, overlap=0)
#     blobs_log = blob_doh(inverted_cell_image, max_sigma=30, threshold=0.2, overlap=0)
    
    blobs_list=[]
    
    for blob in blobs_log:
        blobs_list.append(np.delete(blob, 2))
        
    return blobs_list


def eliminate_border_blobs(cell_image, blobs_log):
    # find the blobs too close to border so as to eliminate them
    blobs_dict = defaultdict()
    for i, blob in enumerate(blobs_log):
        blobs_dict[i] = blob
        y, x, r = blob
        image_border_x, image_border_y = cell_image.shape[0]/4, cell_image.shape[1]/4
        if x < image_border_x or x > 3*image_border_x or y < image_border_y or y > 3*image_border_y:
            blobs_dict.pop(i)
            
    return blobs_dict


def centre_of_mass(cell_image, blobs_dict):
    # find the blob with highest intensity value
    gray_cell_image = skimage.color.rgb2gray(cell_image) 
    ixs = np.indices(gray_cell_image.shape)
    xms = []
    yms = []
    ms = []
    for i, blob in blobs_dict.items():
        y, x, r = blob
        # Define an array of shape `[2, 1, 1]`, containing the center of the blob
        blob_center = np.array([y, x])[:, np.newaxis, np.newaxis]
        # Using the formula for a circle, `x**2 + y**2 < r**2`, generate a mask for this blob.
        mask = ((ixs - blob_center)**2).sum(axis=0) < r**2
        # Calculate the average intensity of pixels under the mask
        blob_avg_est = gray_cell_image[mask].mean()
        yms.append(blob_avg_est*y)
        xms.append(blob_avg_est*x)
        ms.append(blob_avg_est)
        # print(f'Blob {i}: Centre {(x, y)}, average value: {blob_avg_est:.2f}')
    return (sum(yms)/sum(ms), sum(xms)/sum(ms))


def get_soma(cell_image):
    blobs = get_blobs(cell_image)
    soma_blobs = eliminate_border_blobs(cell_image, blobs)
    if len(soma_blobs)==1:
        soma = list(soma_blobs.values())[0][:2]
    if len(soma_blobs)>1:
        soma = centre_of_mass(cell_image, soma_blobs)

    return soma


def label_objects(image):

    invert_thresholded_cell_image = skimage.util.invert(image)
    bw = closing(invert_thresholded_cell_image, square(1))
    # remove artifacts connected to image border
    cleared = clear_border(bw)
    # label image regions
    labelled_image, no_of_objects = label(cleared, return_num=True)
    
    return labelled_image, no_of_objects 


def remove_small_object_noise(image):
    labelled_image, no_of_objects = label_objects(image)
    object_areas = []
    for object_label in range(1, no_of_objects+1):
        object_areas.append(len(np.where(labelled_image==[object_label])[0]))
  
    largest_object_label = np.argmax(object_areas)+1
    astrocyte_image = np.where(labelled_image==[largest_object_label], 1, 0)
    
    return astrocyte_image


def surface_area(image):
    return np.sum(image)


def skeletonization(image):
    # perform skeletonization
    return skimage.morphology.skeletonize(image) 


def get_soma_on_skeleton(soma, astrocyte_skeleton):
    skeleton_pixel_coordinates = [(i, j) for (i, j), val in np.ndenumerate(astrocyte_skeleton) if val!=0]
    soma_on_skeleton = min(skeleton_pixel_coordinates, key=lambda x: distance(soma, x))
    
    return soma_on_skeleton


def plot_skeleton_overlay(cell_image, astrocyte_skeleton):
    fig, ax = plt.subplots()
    draw.overlay_skeleton_2d(cell_image, astrocyte_skeleton, dilate=0, axes=ax);


def total_length(astrocyte_skeleton):
    return np.sum(astrocyte_skeleton)


def convex_hull(astrocyte_skeleton, plot=False):
    convex_hull = skimage.morphology.convex_hull_image(astrocyte_skeleton)
    if plot==True:
        fig, ax = plt.subplots()
        ax.set_axis_off()
        ax.imshow(convex_hull)
        
    return np.sum(convex_hull)


def get_soma_node(soma, astrocyte_skeleton):
    soma_on_skeleton = get_soma_on_skeleton(soma, astrocyte_skeleton)
    near = []
    for i in range(skan.csr.Skeleton(astrocyte_skeleton).n_paths):
        path_coords = skan.csr.Skeleton(astrocyte_skeleton).path_coordinates(i)
        nearest = min(path_coords, key=lambda x: distance(soma_on_skeleton, x))
        near.append(nearest)

    soma_on_path = min(near, key=lambda x: distance(soma_on_skeleton, x))
    soma_node = [i for i,j in enumerate(skan.csr.Skeleton(astrocyte_skeleton).coordinates) if all(soma_on_path==j)]
    return soma_node


def get_no_of_forks(astrocyte_image):
    astrocyte_skeleton = skeletonization(astrocyte_image)
    pixel_graph, coordinates, degrees = skeleton_to_csgraph(astrocyte_skeleton)
    fork_image = np.where(degrees > [2], 1, 0)
    s = generate_binary_structure(2,2)
    labeled_array, num_forks = label(fork_image, structure=s)
    
    return num_forks


def get_soma_branches(soma_node, paths_list):    
    soma_branches=[]
    for path in paths_list:
        if soma_node in path:
            soma_branches.append(path)
    return soma_branches


def eliminate_loops(branch_statistics, paths_list):
    loop_indexes=[]
    loop_branch_end_points=[]
    
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

    new_branch_statistics = np.delete(branch_statistics, loop_indexes, axis=0)

    # eliminate loops from paths list
    path_indexes=[]
    for loop_end_points in loop_branch_end_points:
        for path_no, path in enumerate(paths_list):
            if loop_end_points[0]==path[0] and loop_end_points[1]==path[-1] or loop_end_points[0]==path[-1] and loop_end_points[1]==path[0]:
                path_indexes.append(path_no)
                break

    new_paths_list = np.delete(np.array(paths_list), path_indexes, axis=0)
                    
    return new_branch_statistics, new_paths_list


def branch_structure(junctions, branch_statistics, paths_list):
    next_set_junctions = []
    next_set_branches = []
    terminal_branches=[]
#     print(branch_statistics)
    for junction in junctions:
        branches_travelled = []
        for branch_no, branch in enumerate(branch_statistics):
            if branch[0]==junction:
                if branch[3]==2:
                    next_set_junctions.append(branch[1])
                    for path in paths_list:
                        if branch[0]==path[0] and branch[1]==path[-1] or branch[0]==path[-1] and branch[1]==path[0]:
                            next_set_branches.append(path)
                            branches_travelled.append(branch_no)
                if branch[3]==1:
                    for path in paths_list:
                        if branch[0]==path[0] and branch[1]==path[-1] or branch[0]==path[-1] and branch[1]==path[0]:
                            terminal_branches.append(path)
                            next_set_branches.append(path)
                            branches_travelled.append(branch_no)
            elif branch[1]==junction:
                if branch[3]==2:
                    next_set_junctions.append(branch[0])
                    for path in paths_list:
                        if branch[0]==path[0] and branch[1]==path[-1] or branch[0]==path[-1] and branch[1]==path[0]:
                            next_set_branches.append(path)
                            branches_travelled.append(branch_no)
                if branch[3]==1:
                    for path in paths_list:
                        if branch[0]==path[0] and branch[1]==path[-1] or branch[0]==path[-1] and branch[1]==path[0]:
                            terminal_branches.append(path)
                            next_set_branches.append(path)
                            branches_travelled.append(branch_no)
        branch_statistics = np.delete(branch_statistics, branches_travelled, axis=0)
            
    return next_set_junctions, next_set_branches, terminal_branches, branch_statistics





class Sholl:
    def __init__(self, skeleton, soma):
        self.skeleton = skeleton
        self.soma = soma
        
    def concentric_coords_and_values(self):
        concentric_coordinates = defaultdict(list) # {100: [(10,10), ..] , 400: [(20,20), ..]}
        concentric_coordinates_intensities = defaultdict(list)
        concentric_radiuses = [4, 8, 12, 16, 20, 24, 28, 32, 36]
        # concentric_radiuses = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

        for (x, y), value in np.ndenumerate(self.skeleton):
            for radius in concentric_radiuses:
                lhs = (x - self.soma[1])**2 + (y - self.soma[0])**2
                if abs((math.sqrt(lhs)-radius)) < 0.9:
                    concentric_coordinates[radius].append((x, y))
                    concentric_coordinates_intensities[radius].append(value)

        return concentric_coordinates, concentric_coordinates_intensities
    
    def sholl_results(self, plot=False):
        xs = []
        ys = []
        concentric_coordinates, concentric_intensities = self.concentric_coords_and_values()
        for rad, val in concentric_intensities.items():
            avg = sum(val)
            xs.append(rad)
            ys.append(avg)

        order = np.argsort(xs)
        self.distances_from_soma = np.array(xs)[order]
        self.no_of_intersections = np.array(ys)[order]
        
        if plot==True:
            plt.plot(self.distances_from_soma, self.no_of_intersections)
            plt.xlabel("Distance from centre")
            plt.ylabel("No. of intersections") 
            plt.show()
    
    def polynomial_fit(self, plot=False):
        # Linear
        y_data = self.no_of_intersections
        reshaped_x = self.distances_from_soma.reshape((-1, 1))

        x_ = sklearn.preprocessing.PolynomialFeatures(degree=3, include_bias=False).fit_transform(reshaped_x)
        # create a linear regression model
        self.polynomial_model = sklearn.linear_model.LinearRegression().fit(x_, y_data)
        
        if plot==True:
            # predict y from the data
            x_new = self.distances_from_soma
            y_new = self.polynomial_model.predict(x_)

            # plot the results
            plt.figure(figsize=(4, 3))
            ax = plt.axes()
            ax.scatter(reshaped_x, y_data)
            ax.plot(x_new, y_new)

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.axis('tight')
            plt.show()

        return self.polynomial_model

    def polynomial_equation(self, z):
        return self.polynomial_model.coef_[2] * z**3 + self.polynomial_model.coef_[1] * z**2 + self.polynomial_model.coef_[0]*z + self.polynomial_model.intercept_
        
    def enclosing_radius(self):
        return self.distances_from_soma[len(self.no_of_intersections) - (self.no_of_intersections!=0)[::-1].argmax() - 1]
    
    def critical_radius(self):
        critical_radius = scipy.optimize.fmin(lambda z: -self.polynomial_equation(z), 0)
        return critical_radius[0]
    
    def critical_value(self):
        return self.polynomial_equation(self.critical_radius())
    
    def skewness(self):
        x_ = sklearn.preprocessing.PolynomialFeatures(degree=3, include_bias=False).fit_transform(self.no_of_intersections.reshape((-1, 1)))
        return scipy.stats.skew(self.polynomial_model.predict(x_))
    
    def schoenen_ramification_index(self, no_of_primary_branches):
        return self.critical_value()/no_of_primary_branches
    
    def semi_log(self, plot=False):
        # no. of intersections/circumference
        normalized_y = np.log(self.no_of_intersections/(2*math.pi*self.distances_from_soma))
        reshaped_x = self.distances_from_soma.reshape((-1, 1))
        model = sklearn.linear_model.LinearRegression().fit(reshaped_x, normalized_y)

        # predict y from the data
        x_new = self.distances_from_soma
        y_new = model.predict(reshaped_x)

        if plot==True:
            # plot the results
            plt.figure(figsize=(4, 3))
            ax = plt.axes()
            ax.scatter(reshaped_x, normalized_y)
            ax.plot(x_new, y_new)

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.axis('tight')
            plt.show()

        self.semi_log_r2 = model.score(reshaped_x, normalized_y)
        self.semi_log_regression_intercept = model.intercept_
        self.semi_log_sholl_regression_coefficient = -model.coef_
    
    def log_log(self, plot=False):
    
        # no. of intersections/circumference
        normalized_y = np.log(self.no_of_intersections/(2*math.pi*self.distances_from_soma))
        reshaped_x = self.distances_from_soma.reshape((-1, 1))
        normalized_x = np.log(reshaped_x)
        model = sklearn.linear_model.LinearRegression().fit(normalized_x, normalized_y)

        # predict y from the data
        x_new = normalized_x
        y_new = model.predict(normalized_x)

        if plot==True:

            # plot the results
            plt.figure(figsize=(4, 3))
            ax = plt.axes()
            ax.scatter(normalized_x, normalized_y)
            ax.plot(x_new, y_new)

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.axis('tight')
            plt.show()

        self.log_log_r2 = model.score(normalized_x, normalized_y)
        self.log_log_regression_intercept = model.intercept_
        self.log_log_sholl_regression_coefficient = -model.coef_
    
    def determination_ratio(self):
        determination_ratio = self.semi_log_r2/self.log_log_r2
        if determination_ratio>1:
            self.normalization_method="Semi-log"
        else:
            self.normalization_method="Log-log"
    
    def coefficient_of_determination(self):
        self.determination_ratio()
        if self.normalization_method=="Semi-log":
            return self.semi_log_r2
        else:
            return self.log_log_r2
            
    def sholl_regression_coefficient(self):
        self.determination_ratio()
        if self.normalization_method=="Semi-log":
            return self.semi_log_sholl_regression_coefficient
        else:
            return self.log_log_sholl_regression_coefficient
    
    def regression_intercept(self):
        self.determination_ratio()
        if self.normalization_method=="Semi-log":
            return self.semi_log_regression_intercept
        else:
            return self.log_log_regression_intercept