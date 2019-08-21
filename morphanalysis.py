import skimage
import sklearn
import scipy
import numpy as np

from skimage.transform import rescale
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.decomposition import PCA

import shearlexity
from FFST import shearletTransformSpect

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
from skimage.measure import regionprops
from skimage.morphology import closing, square, remove_small_objects
from skimage.color import label2rgb
import matplotlib.patches as mpatches
from skimage.filters import threshold_otsu

import skan
from skan.pre import threshold
from skan import draw
from skan import skeleton_to_csgraph
from skan import Skeleton, summarize


def distance(P1, P2):
    """
    computing the distance between 2 points
    """
    return ((P1[0] - P2[0])**2 + (P1[1] - P2[1])**2) ** 0.5



class Cell:

    def __init__(self, cell_image, reference_image=None):
        self.cell_image = cell_image
        self.gray_cell_image = skimage.color.rgb2gray(self.cell_image)
        self.inverted_gray_cell_image = skimage.util.invert(self.gray_cell_image)
        self.reference_image = reference_image
        self.threshold_image = self.threshold_image()
        self.inverted_threshold_image = skimage.util.invert(self.threshold_image)
        self.cleaned_image = self.remove_small_object_noise()


    def entropy_complexity(self, plot=False):
        h,c = shearlexity.map_cecp(self.gray_cell_image,3)
        return np.sum(h), np.sum(c)


    def complexity(self, plot=False):
        h,c = shearlexity.map_cecp(self.gray_cell_image,3)
        return np.sum(c)


    def get_blobs(self):
        # dab stain
        blobs_log = blob_log(self.inverted_gray_cell_image, min_sigma=6, max_sigma=20, num_sigma=10, threshold=0.1, overlap=0.5)
        # confocal
    #     blobs_log = blob_log(inverted_cell_image, min_sigma=1, max_sigma=30, num_sigma=10, threshold=0.1, overlap=0)
    #     blobs_log = blob_dog(inverted_cell_image, max_sigma=30, threshold=0.1, overlap=0)
    #     blobs_log = blob_doh(inverted_cell_image, max_sigma=30, threshold=0.2, overlap=0)
        
        blobs_list=[]
        
        # for blob in blobs_log:
        #     blobs_list.append(np.delete(blob, 2))
            
        return blobs_log


    def eliminate_border_blobs(self, blobs_log):
        # find the blobs too close to border so as to eliminate them
        blobs_dict = defaultdict()
        for i, blob in enumerate(blobs_log):
            blobs_dict[i] = blob
            # y, x, r = blob
            # image_border_x, image_border_y = self.cell_image.shape[0]//6, self.cell_image.shape[1]//6

            # if x < image_border_x or x > 3*image_border_x or y < image_border_y or y > 3*image_border_y:
            #     blobs_dict.pop(i)
                
        return blobs_dict


    def centre_of_mass(self, blobs_dict):
        # find the blob with highest intensity value
        ixs = np.indices(self.gray_cell_image.shape)
        # xms = []
        # yms = []
        # ms = []
        blob_intensities=[]
        blob_centres=[]
        for i, blob in blobs_dict.items():
            y, x, r = blob
            # Define an array of shape `[2, 1, 1]`, containing the center of the blob
            blob_center = np.array([y, x])[:, np.newaxis, np.newaxis]
            # Using the formula for a circle, `x**2 + y**2 < r**2`, generate a mask for this blob.
            mask = ((ixs - blob_center)**2).sum(axis=0) < r**2
            # Calculate the average intensity of pixels under the mask
            blob_avg_est = self.gray_cell_image[mask].mean()
            blob_intensities.append(blob_avg_est)
            blob_centres.append((y, x))
            # yms.append(blob_avg_est*y)
            # xms.append(blob_avg_est*x)
            # ms.append(blob_avg_est)
            # print(f'Blob {i}: Centre {(x, y)}, average value: {blob_avg_est:.2f}')

        # return (sum(yms)/sum(ms), sum(xms)/sum(ms))
        return blob_centres[np.argmin(blob_intensities)]


    def get_soma(self):
        blobs = self.get_blobs()
        soma_blobs = self.eliminate_border_blobs(blobs)
        if len(list(soma_blobs.values())) == 0:
            fig, ax = plt.subplots()
            ax.set_axis_off()
            ax.imshow(self.cell_image)

        if len(soma_blobs)==1:
            soma = list(soma_blobs.values())[0][:2]
        if len(soma_blobs)>1:
            soma = self.centre_of_mass(soma_blobs)

        return soma

    def threshold_image(self):

        if self.reference_image is not None:
            self.gray_reference_image = skimage.color.rgb2gray(self.reference_image)
            self.gray_cell_image = skimage.transform.match_histograms(self.gray_cell_image, self.gray_reference_image)

        # Contrast stretching
        p2, p98 = np.percentile(self.gray_cell_image, (2, 98))
        img_rescale = skimage.exposure.rescale_intensity(self.gray_cell_image, in_range=(p2, p98))

        thresholded_cell = img_rescale > threshold_otsu(img_rescale)

        # invert_thresholded_cell = skimage.util.invert(thresholded_cell)

        return thresholded_cell

    def label_objects(self):

        bw = closing(self.inverted_threshold_image, square(1))
        # label image regions
        labelled_image, no_of_objects = skimage.measure.label(bw, return_num=True)
        
        return labelled_image, no_of_objects 


    def remove_small_object_noise(self):
        labelled_image, no_of_objects = self.label_objects()
        object_areas = []
        for object_label in range(1, no_of_objects+1):
        # for object_label in range(no_of_objects):
            object_areas.append(len(np.where(labelled_image==[object_label])[0]))
      
        largest_object_label = np.argmax(object_areas)+1
        astrocyte_image = np.where(labelled_image==[largest_object_label], 1, 0)
        
        return astrocyte_image


    def surface_area(self):
        return np.sum(self.cleaned_image)


class Skeleton:
    def __init__(self, cell_image):

        self.cell_image = cell_image
        self.astrocyte = Cell(cell_image)
        self.cleaned_image = self.astrocyte.cleaned_image
        self.soma = self.astrocyte.get_soma()
        self.cell_skeleton = self.skeletonization()
        self.soma_on_skeleton = self.get_soma_on_skeleton()
        self.padded_skeleton = self.pad_skeleton()
        self.classify_branching_structure()


    def skeletonization(self):
        # perform skeletonization
        return skimage.morphology.skeletonize(self.cleaned_image) 


    def pad_skeleton(self):

        skeleton_indices = np.nonzero(self.cell_skeleton)
        x_min, x_max = min(skeleton_indices[1]), max(skeleton_indices[1])
        y_min, y_max = min(skeleton_indices[0]), max(skeleton_indices[0])
        self.bounded_skeleton = self.cell_skeleton[y_min:y_max, x_min:x_max]
        pad_width = max(self.bounded_skeleton.shape)//3

        self.soma_on_padded_skeleton = self.soma_on_skeleton[0]-y_min+pad_width, self.soma_on_skeleton[1]-x_min+pad_width

        return skimage.util.pad(self.bounded_skeleton, pad_width=pad_width, mode='constant')


    def get_soma_on_skeleton(self):
        skeleton_pixel_coordinates = [(i, j) for (i, j), val in np.ndenumerate(self.cell_skeleton) if val!=0]
        soma_on_skeleton = min(skeleton_pixel_coordinates, key=lambda x: distance(self.soma, x))

        return soma_on_skeleton


    def plot_skeleton_overlay(self):
        fig, ax = plt.subplots()
        draw.overlay_skeleton_2d(self.cell_image, self.cell_skeleton, dilate=0, axes=ax);


    def total_length(self):
        return np.sum(self.cell_skeleton)


    def convex_hull(self, plot=False):
        convex_hull = skimage.morphology.convex_hull_image(self.cell_skeleton)
        if plot==True:
            fig, ax = plt.subplots()
            ax.set_axis_off()
            ax.imshow(convex_hull)
            
        return np.sum(convex_hull)


    def get_no_of_forks(self, plot=False):

        pixel_graph, coordinates, degrees = skeleton_to_csgraph(self.cell_skeleton)
        fork_image = np.where(degrees > [2], 1, 0)
        s = scipy.ndimage.generate_binary_structure(2,2)
        labeled_array, num_forks = scipy.ndimage.label(fork_image, structure=s)
        
        if plot==True:
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


    def branch_structure(self, junctions, branch_statistics, paths_list):
        next_set_junctions = []
        next_set_branches = []
        terminal_branches=[]

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


    def classify_branching_structure(self, plot=False):

        def get_soma_node():

            near = []
            for i in range(skan.csr.Skeleton(self.cell_skeleton).n_paths):
                path_coords = skan.csr.Skeleton(self.cell_skeleton).path_coordinates(i)
                nearest = min(path_coords, key=lambda x: distance(self.soma_on_skeleton, x))
                near.append(nearest)

            soma_on_path = min(near, key=lambda x: distance(self.soma_on_skeleton, x))
            soma_node = [i for i,j in enumerate(skan.csr.Skeleton(self.cell_skeleton).coordinates) if all(soma_on_path==j)]
            return soma_node 

        def get_soma_branches(soma_node, paths_list):    
            soma_branches=[]
            for path in paths_list:
                if soma_node in path:
                    soma_branches.append(path)
            return soma_branches


        pixel_graph, coordinates, degrees = skeleton_to_csgraph(self.cell_skeleton)
        branch_statistics = skan.csr.branch_statistics(pixel_graph)
        paths_list = skan.csr.Skeleton(self.cell_skeleton).paths_list()
        
        terminal_branches = []
        branching_structure_array = []
        # get branches containing soma node

        soma_node = get_soma_node()
        soma_branches = get_soma_branches(soma_node, paths_list)
        if len(soma_branches)>2:
            junctions = soma_node
            delete_soma_branch=False
        else:
            # collect first level/primary branches
            junctions = [soma_branches[0][0], soma_branches[0][-1]]
            delete_soma_branch=True
        
        # eliminate loops in branches and path lists
        branch_statistics, paths_list = self.eliminate_loops(branch_statistics, paths_list)
        
        while True:
            junctions, branches, terminal_branch, branch_statistics = self.branch_structure(junctions, branch_statistics, paths_list)
            branching_structure_array.append(branches)
            terminal_branches.extend(terminal_branch)
            if len(junctions)==0:
                break

        if delete_soma_branch==True:
            branching_structure_array[0].remove(soma_branches[0])

            
        if plot==True:
            # store same level branch nodes in single array 
            color_branches_coords=[]
            for branch_level in branching_structure_array:
                single_branch_level=[]
                for path in branch_level:
                    path_coords=[]
                    for node in path:
                        path_coords.append(coordinates[node])
                        single_branch_level.extend(path_coords)
                color_branches_coords.append(single_branch_level)

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.set_title('path')
            ax.imshow(self.cell_skeleton, interpolation='nearest')

            color_codes = ['red', 'blue', 'magenta', 'green', 'cyan']
            for j, color_branch in enumerate(color_branches_coords):
                if j>4:
                    j=4
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
        avg_length_of_primary_branches = 0 if no_of_primary_branches == 0 else sum(map(len, primary_branches))/float(len(primary_branches))
        return primary_branches, no_of_primary_branches, round(avg_length_of_primary_branches, 1)

    def get_secondary_branches(self):
        try:
            secondary_branches = self.branching_structure_array[1]
        except IndexError:
            secondary_branches=[]
        no_of_secondary_branches = len(secondary_branches)
        avg_length_of_secondary_branches = 0 if no_of_secondary_branches == 0 else sum(map(len, secondary_branches))/float(len(secondary_branches))
        return secondary_branches, no_of_secondary_branches, round(avg_length_of_secondary_branches, 1)

    def get_tertiary_branches(self):
        try:
            tertiary_branches = self.branching_structure_array[2]
        except IndexError:
            tertiary_branches=[]
        no_of_tertiary_branches = len(tertiary_branches)
        avg_length_of_tertiary_branches = 0 if no_of_tertiary_branches == 0 else sum(map(len, tertiary_branches))/float(len(tertiary_branches))
        return tertiary_branches, no_of_tertiary_branches, round(avg_length_of_tertiary_branches, 1)

    def get_quatenary_branches(self):
        try:
            quatenary_branches = self.branching_structure_array[3:]
        except IndexError:
            quatenary_branches=[]
        quatenary_branches = [branch for branch_level in quatenary_branches for branch in branch_level]
        no_of_quatenary_branches = len(quatenary_branches)
        avg_length_of_quatenary_branches = 0 if no_of_quatenary_branches == 0 else sum(map(len, quatenary_branches))/float(len(quatenary_branches))
        return quatenary_branches, no_of_quatenary_branches, round(avg_length_of_quatenary_branches, 1)

    def get_terminal_branches(self):
        terminal_branches = self.terminal_branches
        no_of_terminal_branches = len(terminal_branches)
        avg_length_of_terminal_branches = 0 if no_of_terminal_branches == 0 else sum(map(len, terminal_branches))/float(len(terminal_branches))
        return terminal_branches, no_of_terminal_branches, round(avg_length_of_terminal_branches, 1)



class Sholl:
    def __init__(self, cell_image):

        self.skeleton = Skeleton(cell_image)
        self.bounded_skeleton = self.skeleton.bounded_skeleton
        self.padded_skeleton = self.skeleton.padded_skeleton
        self.soma_on_padded_skeleton = self.skeleton.soma_on_padded_skeleton
        self.sholl_results()
        self.polynomial_model = self.polynomial_fit()

        
    def concentric_coords_and_values(self):

        shell_step_size = (max(self.bounded_skeleton.shape)//2)//10
        largest_radius=max(self.bounded_skeleton.shape)//2

        concentric_coordinates = defaultdict(list) # {100: [(10,10), ..] , 400: [(20,20), ..]}
        concentric_coordinates_intensities = defaultdict(list)
        concentric_radiuses = [radius for radius in range(shell_step_size, largest_radius+1, shell_step_size)]

        for (x, y), value in np.ndenumerate(self.padded_skeleton):
            for radius in concentric_radiuses:
                lhs = (x - self.soma_on_padded_skeleton[0])**2 + (y - self.soma_on_padded_skeleton[1])**2
                if abs((math.sqrt(lhs)-radius)) < 0.9:
                    concentric_coordinates[radius].append((x, y))
                    concentric_coordinates_intensities[radius].append(value)

        return concentric_coordinates, concentric_coordinates_intensities
    
    def sholl_results(self, plot=True):
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
            astrocyte_skeleton_copy = copy.deepcopy(self.padded_skeleton)
            for radius, coordinates in concentric_coordinates.items():
                for coord in coordinates:
                    cell_image_with_circles = astrocyte_skeleton_copy
                    cell_image_with_circles[coord[0],coord[1]]=1


            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(cell_image_with_circles)

            y, x = self.soma_on_padded_skeleton
            c = plt.Circle((x, y), 1, color='red')
            ax.add_patch(c)

            ax.set_axis_off()
            plt.tight_layout()
            plt.show()

            plt.plot(self.distances_from_soma, self.no_of_intersections)
            plt.xlabel("Distance from centre")
            plt.ylabel("No. of intersections") 
            plt.show()
    
    def polynomial_fit(self, plot=False):
        # Linear
        y_data = self.no_of_intersections
        reshaped_x = self.distances_from_soma.reshape((-1, 1))

        x_ = preprocessing.PolynomialFeatures(degree=3, include_bias=False).fit_transform(reshaped_x)
        # create a linear regression model
        self.polynomial_model = linear_model.LinearRegression().fit(x_, y_data)

        self.polynomial_predicted_no_of_intersections = self.polynomial_model.predict(x_)

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

    # def polynomial_equation(self, z):
    #     return self.polynomial_model.coef_[2] * z**3 + self.polynomial_model.coef_[1] * z**2 + self.polynomial_model.coef_[0]*z + self.polynomial_model.intercept_
        
    def enclosing_radius(self):
        return self.distances_from_soma[len(self.no_of_intersections) - (self.no_of_intersections!=0)[::-1].argmax() - 1]
    
    def critical_radius(self):
        return self.distances_from_soma[np.argmax(self.polynomial_predicted_no_of_intersections)]
    
    def critical_value(self):
        return round(max(self.polynomial_predicted_no_of_intersections), 2)
    
    def skewness(self):
        x_ = preprocessing.PolynomialFeatures(degree=3, include_bias=False).fit_transform(self.no_of_intersections.reshape((-1, 1)))
        return round(scipy.stats.skew(self.polynomial_model.predict(x_)), 2)
    
    def schoenen_ramification_index(self):
        no_of_primary_branches = self.skeleton.get_primary_branches()[1]
        schoenen_ramification_index = self.critical_value()/no_of_primary_branches
        return round(schoenen_ramification_index, 2)
    
    def semi_log(self, plot=False):
        # no. of intersections/circumference
        normalized_y = np.log(self.no_of_intersections/(2*math.pi*self.distances_from_soma))
        reshaped_x = self.distances_from_soma.reshape((-1, 1))
        model = linear_model.LinearRegression().fit(reshaped_x, normalized_y)

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
        self.semi_log_sholl_regression_coefficient = -model.coef_[0]
    
    def log_log(self, plot=False):
    
        # no. of intersections/circumference
        normalized_y = np.log(self.no_of_intersections/(2*math.pi*self.distances_from_soma))
        reshaped_x = self.distances_from_soma.reshape((-1, 1))
        normalized_x = np.log(reshaped_x)
        model = linear_model.LinearRegression().fit(normalized_x, normalized_y)

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
        self.log_log_sholl_regression_coefficient = -model.coef_[0]
    
    def determination_ratio(self):
        self.semi_log()
        self.log_log()
        determination_ratio = self.semi_log_r2/self.log_log_r2
        if determination_ratio>1:
            self.normalization_method="Semi-log"
        else:
            self.normalization_method="Log-log"
    
    def coefficient_of_determination(self):
        self.determination_ratio()
        if self.normalization_method=="Semi-log":
            return round(self.semi_log_r2, 2)
        else:
            return round(self.log_log_r2, 2)
            
    def sholl_regression_coefficient(self):
        self.determination_ratio()
        if self.normalization_method=="Semi-log":
            return round(self.semi_log_sholl_regression_coefficient, 2)
        else:
            return round(self.log_log_sholl_regression_coefficient, 2)
    
    def regression_intercept(self):
        self.determination_ratio()
        if self.normalization_method=="Semi-log":
            return round(self.semi_log_regression_intercept, 2)
        else:
            return round(self.log_log_regression_intercept, 2)


class pca:

    def __init__(self, groups_folders):
        self.dataset = self.read_images(groups_folders)
        self.features = self.get_features()
        self.feature_names = ['surface_area', 'total_length', 'convex_hull', 'no_of_forks', 'no_of_primary_branches', 'no_of_secondary_branches', 
                                'no_of_tertiary_branches', 'no_of_quatenary_branches', 'avg_length_of_primary_branches', 'avg_length_of_secondary_branches', 
                                'avg_length_of_tertiary_branches', 'avg_length_of_quatenary_branches', 'avg_length_of_terminal_branches', 
                                'critical_radius', 'critical_value', 'enclosing_radius', 'ramification_index', 'skewness', 'coefficient_of_determination', 
                                'sholl_regression_coefficient', 'regression_intercept']


    def save_features(self, feature_name, feature_value):
        directory = os.getcwd()+'/Features'
        if not os.path.exists(directory):
            os.mkdir(directory)
        path = os.getcwd()+'/Features/'

        with open(path+feature_name+'.txt', 'a') as text_file:
            text_file.write(str(feature_value)+'\n')


    def read_images(self, groups_folders):
        dataset=[]
        for group in groups_folders:
            group_data=[]
            for file in os.listdir(group):
                if not file.startswith('.'):
                    print(group+'/'+file)
                    image = io.imread(group+'/'+file)
                    group_data.append(image)
            dataset.append(group_data)

        return dataset

    def get_features(self):
        dataset_features=[]
        self.targets=[]

        for group_no, group in enumerate(self.dataset):
            group_features=[]

            for cell_no, cell_image in enumerate(group):
                self.targets.append(group_no)

                print(group_no, cell_no)

                cell_features=[]
                astrocyte = Cell(cell_image)
                skeleton = Skeleton(cell_image)
                sholl = Sholl(cell_image)

                print(astrocyte.cell_image.shape)

                # cell_features.append(astrocyte.entropy())
                # cell_features.append(skeleton.complexity())

                cell_features.append(astrocyte.surface_area())
                self.save_features('surface_area', astrocyte.surface_area())
                cell_features.append(skeleton.total_length())
                self.save_features('total_length', skeleton.total_length())
                cell_features.append(skeleton.convex_hull())
                self.save_features('convex_hull', skeleton.convex_hull())
                cell_features.append(skeleton.get_no_of_forks())
                self.save_features('no_of_forks', skeleton.get_no_of_forks())
                cell_features.append(skeleton.get_primary_branches()[1])
                self.save_features('no_of_primary_branches', skeleton.get_primary_branches()[1])
                cell_features.append(skeleton.get_secondary_branches()[1])
                self.save_features('no_of_secondary_branches', skeleton.get_secondary_branches()[1])
                cell_features.append(skeleton.get_tertiary_branches()[1])
                self.save_features('no_of_tertiary_branches', skeleton.get_tertiary_branches()[1])
                cell_features.append(skeleton.get_quatenary_branches()[1])
                self.save_features('no_of_quatenary_branches', skeleton.get_quatenary_branches()[1])
                cell_features.append(skeleton.get_terminal_branches()[1])
                cell_features.append(skeleton.get_primary_branches()[2])
                cell_features.append(skeleton.get_secondary_branches()[2])
                cell_features.append(skeleton.get_tertiary_branches()[2])
                cell_features.append(skeleton.get_quatenary_branches()[2])
                cell_features.append(skeleton.get_terminal_branches()[2])
            
                cell_features.append(sholl.critical_radius())
                cell_features.append(sholl.critical_value())
                cell_features.append(sholl.enclosing_radius())
                cell_features.append(sholl.schoenen_ramification_index())
                cell_features.append(sholl.skewness())
                cell_features.append(sholl.coefficient_of_determination())
                cell_features.append(sholl.sholl_regression_coefficient())
                cell_features.append(sholl.regression_intercept())
                
                group_features.append(cell_features)

            dataset_features.extend(group_features)

        return dataset_features


    def plot(self, color_dict, label, marker):

        pca_object = PCA(2)
        print(self.features)
        # fit on data
        pca_object.fit(self.features)
        # access values and vectors
        self.feature_significance = pca_object.components_
        self.component_variance = pca_object.explained_variance_

        print(pca_object.components_)
        print(pca_object.explained_variance_)
        # transform data
        projected = pca_object.transform(self.features)
        Xax=projected[:,0]
        Yax=projected[:,1]

        fig,ax=plt.subplots(figsize=(7,5))
        fig.patch.set_facecolor('white')
        for l in np.unique(self.targets):
            ix=np.where(self.targets==l)
            ax.scatter(Xax[ix], Yax[ix], c=color_dict[l], s=40, label=label[l], marker=marker[l])
        # for loop ends
        plt.xlabel("1st Principal Component",fontsize=14)
        plt.ylabel("2nd Principal Component",fontsize=14)
        plt.legend()
        plt.show()


        # plt.scatter(projected[:, 0], projected[:, 1], edgecolor='none', alpha=0.5)
        # plt.xlabel('component 1')
        # plt.ylabel('component 2')


    def plot_feature_significance(self):

        plt.matshow(self.feature_significance, cmap='viridis')
        plt.yticks([0,1], ['1st Comp','2nd Comp'], fontsize=10)
        plt.colorbar()
        plt.xticks(range(len(self.feature_names)), self.feature_names, rotation=65, ha='left')
        plt.tight_layout()
        plt.show()






