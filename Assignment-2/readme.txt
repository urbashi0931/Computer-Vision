Hough Transform

Part 1 a.
Steps Followed in hough transform
1. Create an empty accumulator space of size half of the original image.
2. from 0 to 180 degree, populate the accumulator where thetas are in x direction and rhos are in y direction.
3. Get the local maximum of the hough transform around neigborhood of 2x2.
4. Get a list of hough(rho,theta)

part 2 b.
Display the hough space that is, H(D,theta)

part 1 c.
Display the lines with line function

Harris Corner Detection
Part 2 a

1. get Ix, and Iy
2. Get Ixx, Iyy, Ixy and smooth with gaussian kernel
3. Get response function using equation
      r=det-0.04 x trace *trace
4. thresolding with the equation
 tvalue = 0.01 * input.max()

5. Non-maximum suppression

Part 2 b

Display Ix, Iy and Ixy

Part 2c

Display corner response function

Part 2d
display corner points

Part 3 a

Algorithm followed

1. Get interest points using contour detection
2. get 16 x16 block around each interest point
3. Get sub region of 4 x 4 block
4. Get orientaion matrix of each 4 x 4 block, i. e cell.
5. Count the number of angles at each partition of 45 degree
6. Get a pattern of counts at each cell and merge them

part 3 b

display the keypoints using drawkeypoints

part 3 c

display the matches with randomized color and keypoint location


part 4
a)
Normalization of sift descriptor and thresholding it to 0.2
Normalizing again

b) The first step: Find the Global maximum corner response value, i.e. Fmax=max (f) {F_{max}} = Max\left (f \right).

The second step is to traverse N corner points and calculate that ri {R_i} with the following conditions is stored with the vector r {r}
Ri=minjxi-xj,s.t.f (xi) <crobustf (XJ)

References
1. Professors lecture slides
2. https://topic.alibabacloud.com/a/discussion-on-algorithm-of-anms-non-maximal-value-suppression-algorithm-in-mops_8_8_10233011.html
3. https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
4. http://matthewalunbrown.com/papers/cvpr05.pdf
5. https://getpython.wordpress.com/2019/07/10/corner-detection-using-harris-corner-in-python-programming/








   



