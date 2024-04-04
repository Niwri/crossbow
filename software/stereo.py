import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# RGB Green (LED) - edit after finding LED green value
GREEN = 0b0000011111100000
# Rows and Cols
ROWS = 144
COLS = 174

# everything in meters
# focal length = 6mm
FOCAL_LENGTH = 6 * (10^-3)

# pixel length, pixel width = 3.6um
PIXEL_LENGTH = 3.6 * 10^(-6)

# baseline distance between center of left and right cameras - dummy variable for now
BASELINE_D = 5

# window/box size (for matching image in pixels) - variable as well
BOX_SIZE = 20

# threshold for how close it should be to green
THRESHOLD = 50

# Code to find the point on both images
# The target is the green LED; so checking for RGB value green 

def calculate_depth(frame1):
    # Array to save index values
    indexArray = []
    indexcount = 0
    # Array to hold colour values
    minArray = np.zeros((ROWS,COLS))
    # Get image data from image 1 --> given in BGR 565 format
    # Algorithm to find the center of green pixels
    for i in range(int(ROWS)):
        for j in range(int(COLS)):
            colour_count = 0
            # How much blue
            colour_count += (frame1[i][j][0] >> 3)
            # How much red
            colour_count += (frame1[i][j][2] >> 3)
            # How much green
            colour_count += abs((GREEN >> 5) - (frame1[i][j][1] >> 2))
            # Create a 2D array representing image with min --> closest to colour green
            minArray[i][j] = colour_count
            # Compare with some threshold
            if colour_count <= THRESHOLD:
                indexcount += 1
                indexArray[indexcount] = (i,j)

    # if we want to compare and check linearity or smth    
    prev_inertia = ((ROWS/2)^2 + (COLS/2)^2)

    for i in range(1, indexcount):
        # Arbitrary number of clusters (from 1 to total number of points)
        kmeans = KMeans(n_clusters=i)
        # fit the array of points
        kmeans.fit(indexArray)
        # compare the current inertia to a certain threshold (for now)
        cur_inertia = kmeans.inertia_
        if cur_inertia <= ((BOX_SIZE/2)^2 + (BOX_SIZE/2)^2):
            break
    
    # we have desired number of clusters n_clusters
    # access the labels assigned to each data point
    labels = kmeans.labels_
    # Count the number of points in each cluster
    unique, counts = np.unique(labels, return_counts = True)

    # Find the cluster with the most points
    cluster_maxpoints = unique[np.argmax(counts)]

    # Get the center of said cluster
    center_cluster = kmeans.cluster_centers_[cluster_maxpoints]

    # center_cluster returns coords so get x and y
    pointx1 = center_cluster[0]
    pointy1 = center_cluster[1]

    distance1 = pointx1 

    # Get image data from image 2
    # Algorithm to find the matching image - sum of squared differences
    # Compare with box of BOX_SIZE length and width and iterate (while keeping pixel index values) and choose minimum value
    #pointx2 = 
    #pointy2 = pointx2
    distance2 = pointx2

    # Calculate the disparity
    disparity = abs(distance1 - distance2)

    depth = (FOCAL_LENGTH/PIXEL_LENGTH) * (BASELINE_D/(disparity * PIXEL_LENGTH))

    return depth 

