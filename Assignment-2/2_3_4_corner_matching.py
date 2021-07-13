import cv2
import numpy as np
import random


def show(window_name, img):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class part_2:
    def __init__(self):
        path = 'G:/assignment 2/hough/hough2.png'
        Ix, Iy = part_2.part_2b_gradient(path)
        crs = part_2.part_2c_corner_response(Ix, Iy)
        part_2.part_2d_final_interest_points(crs, path)

    @staticmethod
    def tholding(input):
        tvalue = 0.01 * input.max()
        for (y, x) in ((y, x) for y in range(input.shape[0]) for x in range(input.shape[1])):
            input[y, x] = 0 if input[y, x] > tvalue else 1
        return input

    @staticmethod
    def _nms(input, rho):
        h, w = input.shape
        omg = np.zeros((h, w), dtype=np.int32)
        orient = rho * 180 / np.pi
        orient[orient < 0] += 180

        for (i, j) in ((i, j) for i in range(1, h - 1) for j in range(1, w - 1)):
            fst, sec = 255, 255
            if (0 <= orient[i, j] < 22.5) or (157.5 <= orient[i, j] <= 180):
                fst, sec = input[i, j + 1], input[i, j - 1]
            elif 22.5 <= orient[i, j] < 67.5:
                fst, sec = input[i + 1, j - 1], input[i - 1, j + 1]
            elif 67.5 <= orient[i, j] < 112.5:
                fst, sec = input[i + 1, j], input[i - 1, j]
            elif 112.5 <= orient[i, j] < 157.5:
                fst, sec = input[i - 1, j - 1], input[i + 1, j + 1]
            omg[i, j] = input[i, j] if (fst <= input[i, j]) and (input[i, j] >= sec) else 0

        return omg

    @staticmethod
    def nms(gray):
        Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        gradient_mat = np.sqrt(np.square(Ix) + np.square(Iy))
        theta_mat = np.arctan2(Iy, Ix)
        mergedput = part_2._nms(gradient_mat, theta_mat)
        return mergedput

    @staticmethod
    def part_2b_gradient(path):
        gray = cv2.imread(path, 0)
        Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        IX, IY, IXY = np.square(Ix), np.square(Iy), np.multiply(Ix, Iy)
        show('part_2b_gradient_Ix', IX), show('part_2b_gradient_Iy', IY), show('part_2b_gradient_Ixy', IXY)
        return Ix, Iy

    @staticmethod
    def part_2c_corner_response(Ix, Iy):
        gassian = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
        IXX = cv2.filter2D(Ix, cv2.CV_64F, gassian)
        IYY = cv2.filter2D(Iy, cv2.CV_64F, gassian)
        crs = (IXX * IYY) - (0.04 * (IXX + IYY) ** 2)
        show('part_2c_corner_response', crs)
        return crs

    @staticmethod
    def part_2d_final_interest_points(crs, path):
        hold = part_2.tholding(crs)
        mergedput = part_2.nms(hold)
        img = cv2.imread(path)
        for (y, x) in ((y, x) for y in range(mergedput.shape[0]) for x in range(mergedput.shape[1])):
            if mergedput[y, x] == 1:
                kp = [cv2.KeyPoint(y, x, 0.1)]
                img = cv2.drawKeypoints(img, kp, img, color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        show('part_2d_final_interest_points', img)


class part_3:
    def __init__(self):
        path1, path2 = 'G:/assignment 2/image_sets/yosemite/yosemite1.jpg', 'G:/assignment 2/image_sets/yosemite/yosemite2.jpg'
        kp1, des1, kpimage1 = part_3.kp_and_des(path1)
        kp2, des2, kpimage2 = part_3.kp_and_des(path2)
        clicked = part_3.part_3a_ratio_test(kp1, kp2, des1, des2)
        part_3.part_3b_keypoints(kpimage1, kpimage2)
        part_3.part_3c_matched_keypoints(path1, path2, kp1, kp2, clicked, 'part_3c_matched_keypoints')

    @staticmethod
    def counter(a):
        omg = {0: 0, 0.79: 0, 1.57: 0, 2.36: 0, 3.14: 0, 3.93: 0, 4.71: 0, 5.50: 0}
        for i in a:
            omg[round(i, 2)] += 1
        return omg

    @staticmethod
    def cont_drawing(input, cont, color):
        _cont = np.array(cont)
        for x in range(_cont.shape[0]):
            for y in range(_cont[x].shape[0]):
                for z in range(_cont[x][y].shape[0]):
                    input[cont[x][y][z][1]][cont[x][y][z][0]] = color

    @staticmethod
    def cont_calculate(path, Q=100):
        c = 1
        cx, cy = np.zeros((3, 3), np.double), np.zeros((3, 3), np.double)
        for (i, j) in ((i, j) for i in range(3) for j in range(3)):
            cx[i, j], cy[i, j] = 0, 0
        cx[1, 0], cx[1, 2] = -c, c
        cy[0, 1], cy[2, 1] = -c, c

        cont = []
        img = cv2.imread(path, 0)
        h, w = np.shape(img)[:2]
        xicont, yicont = np.zeros((h, w), np.double), np.zeros((h, w), np.double)

        for (i, j) in ((i, j) for i in range(h) for j in range(w)):
            if j == 0 or j == w - 1 or i == 0 or i == h - 1:
                xicont[i][j], yicont[i][j] = 0, 0
            else:
                xicont[i][j] = (np.multiply(cx, img[i - 1:i + 2, j - 1:j + 2]).sum(axis=1).sum(axis=0))
                yicont[i][j] = (np.multiply(cy, img[i - 1:i + 2, j - 1:j + 2]).sum(axis=1).sum(axis=0))
            a = max(min(np.math.sqrt(np.math.pow(xicont[i][j], 2) + np.math.pow(yicont[i][j], 2)), 255), 0)
            if a > Q:
                cont.append([i, j])
        return cont, xicont, yicont

    @staticmethod
    def orient_calculate(x, y, w, h, xicont, yicont):
        d = 0 if x < 0 or x > w - 1 or y < 0 or y > h - 1 else np.math.atan2(yicont[y][x], xicont[y][x])
        d += np.math.pi
        return d

    @staticmethod
    def block_prepare(x, y, w, h, xicont, yicont):
        block = np.zeros((16, 16), np.double)
        for (j, i) in ((j, i) for j in range(y - 8, y + 8) for i in range(x - 8, x + 8)):
            orient = part_3._orient(part_3.orient_calculate(i, j, w, h, xicont, yicont))
            block[j - y + 8][i - x + 8] = orient
        return block

    @staticmethod
    def aindex(orient):
        orients = [np.math.pi * i for i in [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75]]
        stores = orients
        a = int(np.argmin(np.abs(np.subtract(orients, orient))))
        return a, stores

    @staticmethod
    def _orient(orient):
        a, stores = part_3.aindex(orient)
        return stores[a]

    @staticmethod
    def _orient_spin(orient):
        orientsStrings = ['0', 'π/4', 'π/2', '3π/4', 'π', '5π/4', '3π/2', '7π/4']
        a, _ = part_3.aindex(orient)
        return orientsStrings[a]

    @staticmethod
    def kp_and_des(path):
        img = cv2.imread(path)
        h, w = img.shape[:2]
        cont, xicont, yicont = part_3.cont_calculate(path)
        kp, histo = list(), []

        for i in range(np.shape(cont)[0]):
            pxl_y, pxl_x = cont[i][:2]
            pt = cv2.KeyPoint(pxl_y, pxl_x, 1)
            kp.append(pt)
            blocks = part_3.block_prepare(pxl_x, pxl_y, w, h, xicont, yicont)
            no_of_block, mx_orient, mnQ, mxQ = 0, [], 2, 5
            temp, mx = [], 0

            for block in blocks:
                array = np.matrix.flatten(block)
                count = part_3.counter(array)
                temp.extend(list(count.values()))
                mx, orient_mx = -1, -2 * np.math.pi
                for c in count:
                    if count[c] > mx:
                        mx = count[c]
                        orient_mx = part_3._orient(c)
                mx_orient.append(orient_mx)
                no_of_block += 1

            histo.extend(temp)
            no_orient, mxg = part_3.counter(mx_orient), -2 * np.math.pi
            for c in no_orient:
                if no_orient[c] > mx:
                    mxg = no_orient[c]
            if mnQ < mxg < mxQ:
                cv2.drawKeypoints(img, [pt], img, color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        return kp, histo, img

    @staticmethod
    def part_3a_ratio_test(kp1, kp2, des1, des2):
        temp1 = [set(des1[x:x + 128]) for x in range(0, len(des1), 128)]
        temp2 = [set(des2[x:x + 128]) for x in range(0, len(des2), 128)]

        clicked = list()
        for i in range(len(temp1)):
            omg = {}
            for j in range(len(temp2)):
                if len(temp1[i] - temp2[j]) <= 4:
                    (x1, y1), (x2, y2) = kp1[i].pt, kp2[j].pt
                    dist = np.math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    omg[j] = dist

            tups = sorted(omg.items(), key=lambda item: item[1])
            omgs = {k: v for k, v in tups}
            k = list(omgs.keys())
            if omg[k[0]] >= 0.7 * omg[k[1]]:
                clicked.append((i, k[1]))
        return clicked

    @staticmethod
    def part_3b_keypoints(kpimage1, kpimage2):
        show('part_3b_keypoints', kpimage1)
        show('part_3b_keypoints', kpimage2)

    @staticmethod
    def part_3c_matched_keypoints(path1, path2, kp1, kp2, clicked, title):
        img1, img2 = cv2.imread(path1), cv2.imread(path2)
        (ys, xs), (ye, xe) = img1.shape[:2], img2.shape[:2]
        merged = np.zeros((max(xs, xe), xs + xe, 3), dtype='uint8')
        merged[:ys, :xs, :] = img1
        merged[:ye, xs:xs + xe, :] = img2

        for i in clicked:
            (x1, y1), (x2, y2) = kp1[i[0]].pt, kp2[i[1]].pt
            (x1, y1), (x2, y2) = (int(x1), int(y1)), (int(x2) + ys, int(y2))
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.drawKeypoints(merged, [cv2.KeyPoint(x1, y1, 1)], merged, color)
            cv2.drawKeypoints(merged, [cv2.KeyPoint(x2, y2, 1)], merged, color)
            cv2.line(merged, (x1, y1), (x2, y2), color, 1)
        show(title, merged)


class part_4:
    def __init__(self):
        path, path1, path2 = 'G:/assignment 2/hough/hough2.png', 'G:/assignment 2/image_sets/contrast/contrast1.jpg', 'G:/assignment 2/image_sets/contrast/contrast5.jpg'
        kp1, des1, kpimage1 = part_3.kp_and_des(path1)
        kp2, des2, kpimage2 = part_3.kp_and_des(path2)
        crs = part_4.corner_response(path)
        part_4.part_4a_contrast(path1, path2, kp1, kp2, des1, des2)
        part_4.part_4b_anms(crs, N=500, c=0.2)

    @staticmethod
    def norm(des):
        for i in range(len(des)):
            des[i] = 0 if des[i] / 128 > 0.2 else des[i] / 128
        return des

    @staticmethod
    def corner_response(path):
        gray = cv2.imread(path, 0)
        Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        IX, IY = np.square(Ix), np.square(Iy)
        gassian = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
        IXX = cv2.filter2D(IX, cv2.CV_64F, gassian)
        IYY = cv2.filter2D(IY, cv2.CV_64F, gassian)
        crs = (IXX * IYY)/(IXX + IYY) if np.all(IXX + IYY) else (IXX + IYY)
        return crs

    @staticmethod
    def local_maxima(img, no):
        y, x = img.shape[:2]
        yl, xl = [*range(y)], [*range(x)]
        temp1, temp2 = [yl[i:i + no] for i in range(0, y, no)], [xl[i:i + no] for i in range(0, x, no)]
        mx, sx, sy = -1, -1, -1
        for i, j in zip(temp1, temp2):
            for ix, jx in zip(i, j):
                if img[ix, jx] > mx:
                    mx, sx, sy = img[ix, jx], ix, jx
            for ix, jx in zip(i, j):
                if (ix, jx) != (sx, sy):
                    img[ix, jx] = 0
        return img

    @staticmethod
    def part_4a_contrast(path1, path2, kp1, kp2, des1, des2):
        des11, des22 = part_4.norm(des1), part_4.norm(des2)
        clicked = part_3.part_3a_ratio_test(kp1, kp2, des11, des22)
        part_3.part_3c_matched_keypoints(path1, path2, kp1, kp2, clicked, 'part_4a_contrast')

    @staticmethod
    def part_4b_anms(crs, N, c):
        _crs = part_4.local_maxima(crs, 3)
        corner_points = np.where(_crs > _crs.max() * c)
        responses = crs[corner_points[0], corner_points[1]]
        corner_points = np.hstack((corner_points[0][:, None], corner_points[1][:, None]))
        crsa = []

        for (i, (y, x)) in enumerate(corner_points):
            bigger_neighbors = corner_points[responses > responses[i]]
            radius = np.inf if bigger_neighbors.shape[0] == 0 else np.sum((bigger_neighbors - np.array([y, x])) ** 2, 1).min()
            crsa.append(radius)

        N = min(len(crsa), N)
        p = np.argpartition(-np.asarray(crsa), N)[:N]
        ans = corner_points[p]
        print(ans)
        return ans


part_2()
part_3()
part_4()