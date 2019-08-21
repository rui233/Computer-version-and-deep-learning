import cv2

image = cv2.imread('./data_and_images/lenna.jpg')
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

cv2.imshow('Grayscale',image)
cv2.waitKey()
cv2.destroyAllWindows()

