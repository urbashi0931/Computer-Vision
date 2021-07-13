This assignment focuses on the implementation of different parts stated.

Please note that, here images come after one by one with seperate window.

Part 1:
a) Here, I copied the factor values from an image
b) Using openCV, I have shown the images

Please note that, the result images come one after one each having different window.

c)Here, I have used the factor 16 image calculated in a section and applied diferent interpolation techniques using built in function.


Part 2:
a) Diagonal shifting an image requires rotation. 

b) Gaussian Kernel is calculated using Gaussian filter equation. Then Gaussian Kernel is convoluted with greyscale image of original image.

c) Here, kernel size is kept fixed at 3x3. For different sigma values respective images are calculated and difference of two filtered images are shown.


Part 3:

In all of the sections of part 3, RGB is converted to grayscale image as a part of preprocessing.

a) Sobel filters are applied to Images with respect to X and Y direction using sobel filter.

b) Orientation is calculated using arctan equation for each pixel using the values from a.
Then the result is alculated for calculating red and green channel of RGB channel 

c) Gradient magnitude is calculated using gradient magnitude equation. 

d) Canny Edge detection is performed using openCV's built in function for cany edge detection.


part 4:

In all of the sections of part 4, RGB is converted to grayscale image as a part of preprocessing.

a) Algorithm for non maximum suppression is applied. 

b) Canny Edge detection is applied on image, using algorithm stated in lecture note.

Here threshold for gradient map is kept at 55.

For hysteresis,

Low pixel= 75
High pixel=255

REFERENCE(S)

1. https://gautamnagrawal.medium.com/rotating-image-by-any-angle-shear-transformation-using-only-numpy-d28d16eb5076
2. https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
3. Professor's Lecture note
 


