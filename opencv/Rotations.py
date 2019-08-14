import cv2
import numpy as np

image = cv2.imread('images/input.jpg')
height,width = image.shape[:2]
# cv2.getRotationMatrix2D(rotation_center_x,rotation_center_y,angle of rotation,scale)
rotation_matrix = cv2.getRotationMatrix2D((width/2,height/2),90,.5)

rotated_image = cv2.warpAffine(image,rotation_matrix,(width,height))

cv2.imshow('Rotated Image',rotated_image)
cv2.waitKey()
cv2.destroyAllWindows()
