# image crop
import cv2
import matplotlib.pyplot as plt


class ImgCrop:
    def __init__(self, img):
        self.img = img
        pass

    def crop(self):
        x = int(input())
        y = int(input())
        z = int(input())
        w = int(input())

        img_crop = img[x:y, z:w]
        plt.imshow(img_crop)
        plt.show()
        return img_crop


if __name__ == "__main__":
    img = cv2.imread('D:\git\learngit\lenna.jpg')
    imgCrop = ImgCrop(img)
    imgCrop.crop()

# rotation
import cv2
import matplotlib.pyplot as plt


class Rotation:

    def __init__(self, img):
        self.img = img
        pass

    def rotate(self):
        c = int(input('center_coefficient='))
        angle = int(input('angle='))
        scale = float(input('scale='))

        M = cv2.getRotationMatrix2D((img.shape[1] / c, img.shape[0] / c), angle, scale)
        img_rotate = cv2.warpAffine(img, M, (c * img.shape[1], c * img.shape[0]))
        plt.imshow(img_rotate)
        plt.show()
        return img_rotate


if __name__ == "__main__":
    img = cv2.imread('D:\git\learngit\lenna.jpg')
    Rotation(img).rotate()

# perspective transform
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random


class Perspective_transform:
    def __init__(self, img):
        self.img = img
        pass

    def warp(self):
        height, width, channels = img.shape

        pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
        pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

        M = cv2.getPerspectiveTransform(pts1, pts2)

        dst = cv2.warpPerspective(img, M, (300, 300))
        plt.imshow(dst)
        plt.show()
        return img, dst


if __name__ == "__main__":
    img = cv2.imread('D:\git\learngit\lenna.jpg')
    Perspective_transform(img).warp()

# random perspective transform
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random


class Perspective_transform:
    def __init__(self, img):
        self.img = img
        pass

    def random_warp(self):
        height, width, channels = img.shape

        # warp:
        random_margin = 60
        x1 = random.randint(-random_margin, random_margin)
        y1 = random.randint(-random_margin, random_margin)
        x2 = random.randint(width - random_margin - 1, width - 1)
        y2 = random.randint(-random_margin, random_margin)
        x3 = random.randint(width - random_margin - 1, width - 1)
        y3 = random.randint(height - random_margin - 1, height - 1)
        x4 = random.randint(-random_margin, random_margin)
        y4 = random.randint(height - random_margin - 1, height - 1)

        dx1 = random.randint(-random_margin, random_margin)
        dy1 = random.randint(-random_margin, random_margin)
        dx2 = random.randint(width - random_margin - 1, width - 1)
        dy2 = random.randint(-random_margin, random_margin)
        dx3 = random.randint(width - random_margin - 1, width - 1)
        dy3 = random.randint(height - random_margin - 1, height - 1)
        dx4 = random.randint(-random_margin, random_margin)
        dy4 = random.randint(height - random_margin - 1, height - 1)

        pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
        M_warp = cv2.getPerspectiveTransform(pts1, pts2)
        img_warp = cv2.warpPerspective(img, M_warp, (width, height))
        plt.imshow(img_warp)
        plt.show()
        return M_warp, img_warp


if __name__ == "__main__":
    img = cv2.imread('D:\git\learngit\lenna.jpg')
    Perspective_transform(img).random_warp()

# color shift
import cv2
import numpy as np


class Color_shift:

    def __init__(self, img):
        self.img = img
        pass

    def change(self):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        res = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow('img', img)
        cv2.imshow('mask', mask)
        cv2.imshow('res', res)


if __name__ == "__main__":
    img = cv2.imread('D:\git\learngit\lenna.jpg')
    Color_shift(img).change()