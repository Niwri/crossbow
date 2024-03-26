import math

# RGB Green (LED) - edit after finding LED green value
GREEN = 0b0000011111100000

# everything in meters
# focal length = 6mm
FOCAL_LENGTH = 6 * (10^-3)

# pixel length, pixel width = 3.6um
PIXEL_LENGTH = 3.6 * 10^(-6)

# baseline distance between center of left and right cameras - dummy variable for now
BASELINE_D = 5

# window/box size (for matching image in pixels) - variable as well
BOX_SIZE = 20

# Code to find the point on both images
# The target is the green LED; so checking for RGB value green 

def calculate_depth():
    # Get image data from image 1
    # Algorithm to find the center of green pixels 
    #pointx1 =
    #pointy1 = 
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

