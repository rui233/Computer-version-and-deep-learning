import numpy as np
import cv2

# coding: utf-8
import numpy as np
import cv2

left_img = cv2.imread("left.jpg")
left_img = cv2.resize(left_img, (600, 400))
right_img = cv2.imread("right.jpg")
right_img = cv2.resize(right_img, (600, 400))
left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

hessian = 300
surf = cv2.xfeatures2d.SIFT_create(hessian) # 将Hessian Threshold设置为400,阈值越大能检测的特征就越少
kp1, des1 = surf.detectAndCompute(left_gray, None)  # 查找关键点和描述符
kp2, des2 = surf.detectAndCompute(right_gray, None)

# kp1s = np.float32([kp.pt for kp in kp1])
# kp2s = np.float32([kp.pt for kp in kp2])

# 显示特征点
img_with_drawKeyPoint_left = cv2.drawKeypoints(left_gray, kp1, np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("img_with_drawKeyPoint_left", img_with_drawKeyPoint_left)

img_with_drawKeyPoint_right = cv2.drawKeypoints(right_gray, kp2, np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("img_with_drawKeyPoint_right", img_with_drawKeyPoint_right)



'''
BFMatcher简称暴力匹配，意思就是尝试所有可能匹配，实现最佳匹配。

FlannBasedMatcher简称最近邻近似匹配。
是一种近似匹配方法，并不追求完美！，因此速度更快。
可以调整FlannBasedMatcher参数改变匹配精度或改变算法速度。
参考：https://blog.csdn.net/claroja/article/details/83411108
'''
FLANN_INDEX_KDTREE = 0  # 建立FLANN匹配器的参数

indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # 配置索引，密度树的数量为5
searchParams = dict(checks=50)  # 指定递归次数
# FlannBasedMatcher：是目前最快的特征匹配算法（最近邻搜索）
flann = cv2.FlannBasedMatcher(indexParams, searchParams)  # 建立匹配器

# 参考https://docs.opencv.org/master/db/d39/classcv_1_1DescriptorMatcher.html#a378f35c9b1a5dfa4022839a45cdf0e89
'''
int queryIdx –>是测试图像的特征点描述符（descriptor）的下标，同时也是描述符对应特征点（keypoint)的下标。

int trainIdx –> 是样本图像的特征点描述符的下标，同样也是相应的特征点的下标。

int imgIdx –>当样本是多张图像的话有用。

float distance –>代表这一对匹配的特征点描述符（本质是向量）的欧氏距离，数值越小也就说明两个特征点越相像。
'''

matches = flann.knnMatch(des1, des2, k=2)  # 得出匹配的关键点


good = []
# 提取优秀的特征点
for m, n in matches:
    if m.distance < 0.7 * n.distance:  # 如果第一个邻近距离比第二个邻近距离的0.7倍小，则保留
        good.append(m)

src_pts = np.array([kp1[m.queryIdx].pt for m in good])  # 查询图像的特征描述子索引
dst_pts = np.array([kp2[m.trainIdx].pt for m in good])  # 训练(模板)图像的特征描述子索引


# findHomography参考https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780
# 单应矩阵:https://www.cnblogs.com/wangguchangqing/p/8287585.html
H = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)  # 生成变换矩阵

h1, w1 = left_gray.shape[:2]
h2, w2 = right_gray.shape[:2]
shift = np.array([[1.0, 0, w1], [0, 1.0, 0], [0, 0, 1.0]])
# 点积
M = np.dot(shift, H[0])  # 获取左边图像到右边图像的投影映射关系

dst = cv2.warpPerspective(left_img, M, (w1+w2, max(h1, h2)))  # 透视变换，新图像可容纳完整的两幅图
cv2.imshow('left_img', dst)  # 显示，第一幅图已在标准位置
dst[0:h2, w1:w1+w2] = right_img  # 将第二幅图放在右侧
# cv2.imwrite('tiled.jpg',dst_corners)
cv2.imshow('total_img', dst)
cv2.imshow('leftgray', left_img)
cv2.imshow('rightgray', right_img)
cv2.waitKey(0)
cv2.destroyAllWindows()