from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

# #---------------------------------------------------------------------------------------------------------------------
# PART I - Transforming an image from color to grayscale
# #---------------------------------------------------------------------------------------------------------------------

# Here we import the image file as an array of shape (nx, ny, nz)
image_file = 'Images/original_image.PNG'
input_image = imread(image_file)  # this is the array representation of the input image
[nx, ny, nz] = np.shape(input_image)  # nx: height, ny: width, nz: colors (RGB)

# Extracting each one of the RGB components
r_img, g_img, b_img = input_image[:, :, 0], input_image[:, :, 1], input_image[:, :, 2]

# The following operation will take weights and parameters to convert the color image to grayscale
gamma = 1.400  # a parameter
r_const, g_const, b_const = 0.2126, 0.7152, 0.0722  # weights for the RGB components respectively
grayscale_image = r_const * r_img ** gamma + g_const * g_img ** gamma + b_const * b_img ** gamma

# #---------------------------------------------------------------------------------------------------------------------
# PART II - Applying the Robert operator
# #---------------------------------------------------------------------------------------------------------------------

"""
The kernels Gx and Gy can be thought of as a differential operation in the "input_image" array in the directions x and y 
respectively. These kernels are represented by the following matrices:
note:  I've done it using a padded 3x3 convolution mask. as Robert's Cross is not odd size (2x2 rather than 3x3 or 5x5).
      _               _                   _                _
     |                 |                 |                  |
     | 0.0   0.0  0.0  |                 |  0.0   0.0   0.0 |
Gx = | 0.0   1.0   0.0 |    and     Gy = |  0.0   0.0   1.0 |
     | 0.0   0.0  -1.0 |                 |  0.0  -1.0   0.0 |
     |_               _|                 |_                _|
"""

# Here we define the matrices associated with the Robert filter
Gx = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1.0]])
Gy = np.array([[0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])
[rows, columns] = np.shape(grayscale_image)  # we need to know the shape of the input grayscale image
Robert_filtered_image = np.zeros(shape=(rows, columns))  # initialization of the output image array (all elements are 0)

# Now we "sweep" the image in both x and y directions and compute the output
for i in range(rows - 2):
    for j in range(columns - 2):
        gx = np.sum(np.multiply(Gx, grayscale_image[i:i + 3, j:j + 3]))  # x direction
        gy = np.sum(np.multiply(Gy, grayscale_image[i:i + 3, j:j + 3]))  # y direction
        Robert_filtered_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)  # calculate the "hypotenuse"

# Display the original image and the ٌخلاثقف filtered image
fig = plt.figure(2)
ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)
ax1.imshow(input_image)
ax2.imshow(Robert_filtered_image, cmap=plt.get_cmap('gray'))
fig.show()