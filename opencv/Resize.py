# 只是改变图像的尺寸大小，cv2.resize()可以实现这个功能。在缩放时推荐cv2.INTER_AREA，
# 在拓展时推荐cv2.INTER_CUBIC（慢）和cv2.INTER_LINEAR。
# 默认情况下所有改变图像尺寸大小的操作使用的是插值法都是cv2.INTER_LINEAR。
import cv2

img = cv2.imread('./data_and_images/input.jpg')
# 下面的None本应该是输出图像的尺寸，但是因为后面我们设置了缩放因子，所以，这里为None
res = cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
res2 = cv2.resize(img,None,fx=0.75,fy=0.75)
# or
# 这里直接设置输出图像的尺寸，所以不用设置缩放因子
height , width =img.shape[:2]
res = cv2.resize(img,(2*width,2*height),interpolation=cv2.INTER_CUBIC)

while(1):
    cv2.imshow('res',res)
    cv2.imshow('img',img)

    if cv2.waitKey(1)&0xFF == 27:
        break
cv2.destroyAllWindows()

img_scaled = cv2.resize(img,(900,400),interpolation = cv2.INTER_AREA)
cv2.imshow('Scaling - Skewed Size',img_scaled)
cv2.waitKey()
cv2.destroyAllWindows()

