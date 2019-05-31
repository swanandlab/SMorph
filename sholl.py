import skimage
import skimage.io as io
from skimage.color import rgb2gray
import numpy as np
from collections import defaultdict
import math
import matplotlib.pyplot as plt
import csv
from skimage.filters import threshold_otsu
import cv2
import os

# try sharpening and others in cv2
# high resolution imgs?
# available diseased & non-diseased data?

def extract_cells(result_file, section):

    cell_images = []
    with open(result_file) as csv_file:
        csv_reader = csv.reader(csv_file)
        row_no = 0
        for row in csv_reader:
            if row_no == 0:
                row_no+=1
            else:
                cell = section[int(row[3]):int(row[5]),int(row[2]):int(row[4])]
                cell_images.append(cell)

    return cell_images

def sharpen(cell_image):

    # shapening kernel, it must equal to one eventually
    kernel_sharpening = np.array([[-1,-1,-1], 
                                  [-1, 9,-1],
                                  [-1,-1,-1]])
    # applying the sharpening kernel to the input image
    sharpened = cv2.filter2D(cell_image, -1, kernel_sharpening)

    return sharpened

def sholl(cell_image, file):

    concentric_coordinates = defaultdict(list) # {100: [(10,10), ..] , 400: [(20,20), ..]}
    concentric_coordinates_intensities = defaultdict(list)
    concentric_radiuses = [5, 10, 15, 20, 25, 30, 35, 40]
    # # concentric_radiuses = [15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285]
    # concentric_radiuses = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]

    for (x, y), value in np.ndenumerate(cell_image):
        image_centre = (int(cell_image.shape[0]/2), int(cell_image.shape[1]/2))
        for radius in concentric_radiuses:
            lhs = (x - image_centre[0])**2 + (y - image_centre[1])**2
            if abs((math.sqrt(lhs)-radius)) < 0.9:
                concentric_coordinates[radius].append((x, y))
                concentric_coordinates_intensities[radius].append(cell_image[x, y])

    x = []
    y = []

    for radius, coordinates in concentric_coordinates.items():
        for coord in coordinates:
            cell_image_with_circles = cell_image
            cell_image_with_circles[coord[0],coord[1]]=0

    for radius, intensity_values in concentric_coordinates_intensities.items(): 
        intensity = sum(intensity_values)/(2*(22/7)*radius)
        x.append(radius)
        y.append(intensity)

    # io.imshow(cell_image_with_circles)
    # io.show()

    plt.plot(x, y)

    return concentric_coordinates_intensities

def save_cells(section, section_name, result_file, save_folder):

    resized_section = skimage.transform.resize(section, (512, 704)) # for jpg images
    # io.imshow(resized_section)
    # io.show()

    # thresh = threshold_otsu(resized_section)
    # resized_section = resized_section > thresh

    # black_pixel_coordinates = [(i,j) for (i, j), val in np.ndenumerate(circle) if val==0]
    # io.imsave(fname="", (600,600))

    cell_images = extract_cells(result_file, resized_section)

    for i, cell_image in enumerate(cell_images):
        # io.imshow(cell_image)
        # io.show()
        io.imsave(save_folder+section_name+"_"+(str(i)+".jpg"), cell_image) 


section = io.imread("sample_astrocytes/3.jpg") # section.shape = (1920, 2560, 3)
section_name = "3"
result_file = 'sample_astrocytes/3.csv'
save_folder = "all/"
save_cells(section, section_name, result_file, save_folder)

gray_section = rgb2gray(section) # gray_section.shape = (1920, 2560)
gray_section = skimage.util.invert(gray_section)
resized_section = skimage.transform.resize(gray_section, (512, 704)) # for tif images
# resized_section = skimage.transform.resize(gray_section, (512, 704)) # for jpg images
io.imshow(resized_section)
io.show()

# thresh = threshold_otsu(resized_section)
# resized_section = resized_section > thresh
# black_pixel_coordinates = [(i,j) for (i, j), val in np.ndenumerate(circle) if val==0]
# io.imsave(fname="", (600,600))

# cell_images = extract_cells(result_file, resized_section)
# for i, cell_image in enumerate(cell_images):
#     sholl(cell_image)

for file in os.listdir('.'):
    cell_image = io.imread(file)
    gray_cell_image = rgb2gray(cell_image)
    inverted_cell_image = skimage.util.invert(gray_cell_image)
    sholl(inverted_cell_image, file)

plt.xlabel("Distance from centre")
plt.ylabel("Intensity")
plt.savefig("sholl")



# save small and large astrocytes in respective folders and do their sholl