import cv2
import numpy as np


def display(title, img):
    img = img.astype(np.uint8)
    cv2.imshow(title, img)
    cv2.waitKey()


def calculate(kernel, sigma):
    kernel = int(kernel) // 2
    x, y = np.mgrid[-kernel:kernel + 1, -kernel:kernel + 1]
    # normal = 1 / (2.0 * np.pi * sigma ** 2)
    normal = 1 / (np.sqrt(2.0 * np.pi)) * sigma
    value = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    return value


def _convo(img, hk2, wk2, hkernal, wkernal, kernel):
    if len(img.shape) == 3:
        imagep = np.pad(img, ((hk2, hk2), (wk2, wk2), (0, 0)), mode='constant', constant_values=0).astype(np.float32)
    elif len(img.shape) == 2:
        imagep = np.pad(img, ((hk2, hk2), (wk2, wk2)), mode='constant', constant_values=0).astype(np.float32)
    convimg = np.zeros(imagep.shape)
    for i in range(hk2, imagep.shape[0] - hk2):
        for j in range(wk2, imagep.shape[1] - wk2):
            x = imagep[i - hk2:i - hk2 + hkernal, j - wk2:j - wk2 + wkernal]
            x = x.flatten() * kernel.flatten()
            convimg[i][j] = x.sum()
    return convimg


def convo(img, kernel):
    hkernal = kernel.shape[0]
    wkernal = kernel.shape[1]
    hk2 = hkernal // 2
    wk2 = wkernal // 2
    endh = -hk2
    endw = -wk2
    convimg = _convo(img, hk2, wk2, hkernal, wkernal, kernel)
    if wk2 == 0:
        return convimg[hk2:endh, wk2:]
    if hk2 == 0:
        return convimg[hk2:, wk2:endw]
    else:
        return convimg[hk2:endh, wk2:endw]


def _nms(img, degree, i, j):
    vq = 255
    vr = 255
    if (0 <= degree[i, j] < 22.5) or (157.5 <= degree[i, j] <= 180):
        vq = img[i, j + 1]
        vr = img[i, j - 1]
    elif 22.5 <= degree[i, j] < 67.5:
        vq = img[i + 1, j - 1]
        vr = img[i - 1, j + 1]
    elif 67.5 <= degree[i, j] < 112.5:
        vq = img[i + 1, j]
        vr = img[i - 1, j]
    elif 112.5 <= degree[i, j] < 157.5:
        vq = img[i - 1, j - 1]
        vr = img[i + 1, j + 1]
    if vq <= img[i, j] and img[i, j] >= vr:
        return img[i, j]
    else:
        return 0


def nms(img, radian):
    height, width = img.shape
    value = np.zeros((height, width), dtype=np.int32)
    degree = radian * 180. / np.pi
    degree[degree < 0] += 180
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            value[i, j] = _nms(img, degree, i, j)
    return value


def sobel(img):
    norm = lambda value: (value * 255) / value.max()
    equation = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    xaxis = convo(img, equation)
    yaxis = convo(img, np.flip(equation.T, axis=0))
    grad = np.sqrt(np.square(xaxis) + np.square(yaxis))
    orient = np.arctan2(yaxis, xaxis)
    grad = norm(grad)
    return grad, orient, xaxis, yaxis


def hyst(img):
    height, width = img.shape
    for i in range(1, height-1):
        for j in range(1, width-1):
            if img[i, j] == 20:
                try:
                    if ((img[i + 1, j - 1] == 255) or (img[i + 1, j] == 255) or (img[i + 1, j + 1] == 255)
                            or (img[i, j - 1] == 255) or (img[i, j + 1] == 255)
                            or (img[i - 1, j - 1] == 255) or (img[i - 1, j] == 255)
                            or (img[i - 1, j + 1] == 255)):
                        img[i, j] = 255
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


def part_1a_sampling():
    global img_original
    sampling = {}
    for i in [2, 4, 8, 16]:
        sampling[i] = img_original[::i, ::i]
    return sampling


def part_1b_sampling():
    global img_original
    sampling = part_1a_sampling()
    for i in [2, 4, 8, 16]:
        display('part_1b_down_sampling_by_{}'.format(i), sampling[i])


def part_1c_sampling():
    global img_original
    img = part_1a_sampling()[16]
    width = img.shape[0] * 10
    height = img.shape[1] * 10
    xy = (width, height)
    img_resize = [cv2.resize(img, xy, cv2.INTER_NEAREST), cv2.resize(img, xy, cv2.INTER_LINEAR),
                  cv2.resize(img, xy, cv2.INTER_CUBIC)]
    title = ['part_1c_nearest_neighbour', 'part_1c_bilinear_interpolation', 'part_1c_bicubic_interpolation']
    display(title[0], img_resize[0]), display(title[1], img_resize[1]), display(title[2], img_resize[2])


def part_2a_shifts_image_diagonally_towards_top_right_corner():
    def image_shift(img_original, sine, cosine):
        height, width = img_original.shape[0], img_original.shape[1]
        heightn, widthn = round(abs(height * cosine) + abs(width * sine)) + 1, round(abs(width * cosine) + abs(height * sine)) + 1
        output = np.zeros((heightn, widthn, img_original.shape[2]))
        cheight, cwidth = round(((height + 1) / 2) - 1), round(((width + 1) / 2) - 1)
        heightnc, widthnc = round(((heightn + 1) / 2) - 1), round(((widthn + 1) / 2) - 1)
        for i in range(height):
            for j in range(width):
                y, x = height-1-i-cheight, width-1-j-cwidth
                ynew, xnew = heightnc - round(-x*sine + y*cosine), widthnc - round(x*cosine + y*sine)
                output[ynew, xnew, :] = img_original[i, j, :]
        return output

    global part_2a_angle, img_original
    sine = np.sin(np.radians(-part_2a_angle))
    cosine = np.cos(np.radians(-part_2a_angle))
    img_final = image_shift(img_original, sine, cosine)
    display('part_2a_shifts_image_original_diagonally', img_final)


def part_2b_calculates_n_x_n_gaussian_filter():
    global part_2b_kernel, part_2b_sigma, img_gray
    img1 = convo(img_gray, calculate(part_2b_kernel, part_2b_sigma))
    display('part_2b_calculates_n_x_n_gaussian_filter', img1)


def part_2c_calculates_difference_of_gaussian_filtered_images():
    global part_2c_kernel, part_2c_sigma1, part_2c_sigma2, img_gray
    img1 = convo(img_gray, calculate(part_2c_kernel, part_2c_sigma1))
    img2 = convo(img_gray, calculate(part_2c_kernel, part_2c_sigma2))
    img3 = img1 - img2
    display('part_2c_3x3', img1)
    display('part_2c_5x5', img2)
    display('part_2c_difference', img3)


def part_3a_sobel_operators_wrt_x_and_y():
    global img_gray, kernel, sigma
    mat_grad, mat_orien, xaxis, yaxis = sobel(img_gray)
    display('part_3a_sobel_horizontal_edge', xaxis)
    display('part_3a_sobel_vertical_edge', yaxis)


def part_3b_orientation_map():
    global img_gray, kernel, sigma
    mat_grad, mat_orien, xaxis, yaxis = sobel(img_gray)
    display('part_3b_orientation_map', mat_grad)


def part_3c_gradient_magnitude():
    global img_gray, kernel, sigma
    mat_grad, mat_orien, xaxis, yaxis = sobel(img_gray)
    display('part_3c_gradient_magnitude', mat_grad)


def part_3d_opencv_canny_edge_detection():
    global img_gray
    edges = cv2.Canny(img_gray, 100, 200)
    display('part_3d_opencv_canny_edge', edges)


def part_4a_non_maximum_suppression_gradient_map():
    global img_gray, kernel, sigma
    mat_grad, mat_orien, xaxis, yaxis = sobel(img_gray)
    img_nms = nms(mat_grad, mat_orien)
    display('part_4a_non_maximum_suppression', img_nms)


def part_4b_canny_edge_detection():
    def replace(mat_grad):
        global xfactor
        column, row = mat_grad.shape
        for i in range(column):
            for j in range(row):
                if mat_grad[i, j] < xfactor:
                    mat_grad[i, j] = 0

    global img_gray, kernel, sigma
    simg = convo(img_gray, calculate(kernel, sigma))
    mat_grad, mat_orien, xaxis, yaxis = sobel(simg)
    replace(mat_grad)
    img_nms = nms(mat_grad, mat_orien)
    img_final = hyst(img_nms)
    display('part_4b_canny_edge_detection', img_final)


if __name__ == "__main__":
    img_original = cv2.imread('G:/urbosi_image_processing/images/u.jpg')
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    part_2a_angle = 45
    part_2b_kernel = 5
    part_2b_sigma = 1
    part_2c_kernel = 3
    part_2c_sigma1 = 1
    part_2c_sigma2 = 3
    kernel = 3
    sigma = 1
    xfactor = 40

    part_1a_sampling()
    part_1b_sampling()
    part_1c_sampling()
    part_2a_shifts_image_diagonally_towards_top_right_corner()
    part_2b_calculates_n_x_n_gaussian_filter()
    part_2c_calculates_difference_of_gaussian_filtered_images()
    part_3a_sobel_operators_wrt_x_and_y()
    part_3b_orientation_map()
    part_3c_gradient_magnitude()
    part_3d_opencv_canny_edge_detection()
    part_4a_non_maximum_suppression_gradient_map()
    part_4b_canny_edge_detection()
