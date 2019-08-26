import cv2
import numpy as np

class medianfilter(object):
    '''
    中值滤波类
    '''
    def __init__(self, img, kernel, paddWay):
        self.img = img
        # 判断kernel是否为奇数大小，不是的话抛出错误
        if (kernel.shape[0] * kernel.shape[1]) % 2 != 0:
            self.kernel = kernel
        else:
            raise AttributeError("Error Kernel Size! Must be ODD!!")
        self.paddWay = paddWay

    def runfilter(self):
        if self.paddWay == 'zero':
            img_padd = self._zeroPadd()
        elif self.paddWay == 'replica':
            img_padd = self._replPadd()
        else:
            raise AttributeError("Error Padding Mode! 'replica' & 'zero' ONLY!!")
        # 获取padding后图像的大小
        # kernel size: krow * kcol
        krow = self.kernel.shape[0]
        kcol = self.kernel.shape[1]
        # kernel center. aka. pixel need to modify
        ctrow = int((krow - 1) / 2)
        ctcol = int((kcol - 1) / 2)

        # 判断图像是否为RGB图像
        if len(self.img.shape) == 3:
            # 考虑到彩色图像，加入通道
            for channel in range(3):

                for irow in range(img_padd.shape[0] - ctrow - 1):
                    for icol in range(img_padd.shape[1] - ctcol - 1):
                        # 缓存kernel范围的像素值
                        tmpixel = np.zeros((krow, kcol))
                        for kr in range(krow):
                            for kc in range(kcol):
                                tmpixel[kr, kc] = img_padd[irow+kr, icol+kc, channel]
                        # 求kernel范围内像素值的中值
                        center = np.median(tmpixel)
                        # 修改kernel中心点的像素值
                        img_padd[irow+ctrow][icol+ctcol][channel] = center.astype(img_padd.dtype)
        else:
            # 如果图像是灰度图，就不需要在channel范围内循环
            for irow in range(img_padd.shape[0] - ctrow - 1):
                for icol in range(img_padd.shape[1] - ctcol - 1):
                    # 　缓存kernel中的像素值
                    tmpixel = np.zeros((krow, kcol))
                    for kr in range(krow):
                        for kc in range(kcol):
                            tmpixel[kr, kc] = img_padd[irow + kr, icol + kc]
                    # 求kernel范围内像素值的中值
                    center = np.median(tmpixel)
                    # 修改kernel中心点的像素值
                    img_padd[irow + ctrow][icol + ctcol] = center.astype(img_padd.dtype)

        return img_padd[0:self.img.shape[0], 0:self.img.shape[1]]

    def _replPadd(self):
        '''
        Edge Padding
        :return: Image after padding
        '''
        krow = self.kernel.shape[0]
        kcol = self.kernel.shape[1]
        padrow = int((krow - 1) / 2)
        padcol = int((kcol - 1) / 2)
        if len(self.img.shape) == 3:
            img_pad = np.pad(self.img, ((padrow, padrow),
                                        (padcol, padcol), (0, 0)), 'edge')
        else:
            img_pad = np.pad(self.img, ((padrow, padrow), (padcol, padcol)), 'edge')
        return img_pad


    def _zeroPadd(self):
        '''
        Zero Padding
        :return: Image after padding
        '''
        krow = self.kernel.shape[0]
        kcol = self.kernel.shape[1]
        padrow = int((krow - 1) / 2)
        padcol = int((kcol - 1) / 2)
        if len(self.img.shape) == 3:
            img_pad = np.pad(self.img, ((padrow, padrow),
                                        (padcol, padcol), (0, 0)),
                             mode='constant', constant_values=0)
        else:
            img_pad = np.pad(self.img, ((padrow, padrow), (padcol, padcol)),
                             mode='constant', constant_values=0)
        return img_pad

def showImg(img):
    cv2.imshow('', img)

    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # img = cv2.imread('lenna_gray.jpg', 0)
    img = cv2.imread('lenna.jpg')
    kernel = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    mdF = medianfilter(img, kernel, 'zero')
    img_padd = mdF.runfilter()
    print(img_padd.shape)
    showImg(img_padd)