import cv2

image = cv2.imread('./data_and_images/input.jpg')

smaller = cv2.pyrDown(image)
larger = cv2.pyrUp(smaller)

cv2.imshow('Smaller',smaller)
cv2.imshow('Larger',larger)
cv2.waitKey()
cv2.destroyAllWindows()