import cv2
import numpy as np


def show(window_name, img):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class part_1:
    def __init__(self):
        img = cv2.imread('G:/assignment 2/hough/hough2.png')
        hou, rho, theta = part_1.part_1a_hough_transform(img)
        idxs, hou = part_1.part_1b_hough_space(hou, 3, 11)
        part_1.part_1c_hough_lines(img, idxs, rho, theta)

    @staticmethod
    def accum(img):
        resr, rest = 1, 1
        height, width = img.shape[:2]
        dimg = np.ceil(np.sqrt(height ** 2 + width ** 2))
        rho = np.arange(-dimg, dimg + 1, resr)
        theta = np.deg2rad(np.arange(0, 180, rest))
        hou = np.zeros((len(rho), len(theta)), dtype=np.uint64)
        return hou, dimg, rho, theta

    @staticmethod
    def part_1a_hough_transform(img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 1.5)
        img = cv2.Canny(blur, 100, 200)
        hou, dimg, rho, theta = part_1.accum(img)
        yids, xids = np.nonzero(img)[:2]

        for (i, j) in ((i, j) for i in range(len(xids)) for j in range(len(theta))):
            x, y = xids[i], yids[i]
            _rho = int((x * np.cos(theta[j]) + y * np.sin(theta[j])) + dimg)
            hou[_rho, j] += 1
        return hou, rho, theta

    @staticmethod
    def part_1b_hough_space(hou, pno, sze):
        idxs, sze = [], sze / 2
        hou1 = np.copy(hou)

        for i in range(pno):
            idx = np.argmax(hou1)
            idxh = np.unravel_index(idx, hou1.shape)
            idxs.append(idxh)
            Yid, Xid = idxh[:2]
            xmn = 0 if (Xid - sze) < 0 else (Xid - sze)
            xmx = hou.shape[1] if (Xid + sze + 1) > hou.shape[1] else (Xid + sze + 1)
            mny = 0 if (Yid - sze) < 0 else (Yid - sze)
            mxy = hou.shape[0] if (Yid + sze + 1) > hou.shape[0] else (Yid + sze + 1)

            for (y, x) in ((y, x) for y in range(int(mny), int(mxy)) for x in range(int(xmn), int(xmx))):
                hou1[y, x] = 0
                hou[y, x] = 255 if (x == xmn or x == (xmx - 1)) or (y == mny or y == (mxy - 1)) else hou[y, x]

        show('part_1b_hough_space', hou.astype(np.uint8))
        return idxs, hou

    @staticmethod
    def part_1c_hough_lines(img, idxs, rho, theta):
        for i in range(len(idxs)):
            _rho, _theta = rho[idxs[i][0]], theta[idxs[i][1]]
            a, b = np.cos(_theta), np.sin(_theta)
            xr, yr = a * _rho, b * _rho
            _x, _y = int(xr + 1000 * (-b)), int(yr + 1000 * a)
            __x, __y = int(xr - 1000 * (-b)), int(yr - 1000 * a)
            cv2.line(img, (_x, _y), (__x, __y), (0, 255, 0), 2)
        show('part_1c_hough_lines', img)


part_1()