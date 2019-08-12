# making a black square
import  cv2
import numpy as np

image = np.zeros((512,512,3),np.uint8)

image_bw = np.zeros((512,512),np.uint8)

cv2.imshow("Black Rectangle (Color)",image)
cv2.imshow("Black Rectangle (B & W)",image_bw)

cv2.waitKey(0)
cv2.destroyAllWindows()

# draw a line over black sqare
image = np.zeros((512,512,3),np.uint8)
cv2.line(image, (0,0),(511,511),(255,127,0),5)
cv2.imshow("Blue Line",image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# draw a rectangle
image = np.zeros((512,512,3),np.uint8)

cv2.rectangle(image,(100,100),(300,250),(127,50,127),-1)
cv2.imshow("Rectangle",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# draw a circle
image = np.zeros((512,512,3),np.uint8)

cv2.circle(image,(350,350),100,(15,75,50),-1)
cv2.imshow("Circle",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# draw polygons
image = np.zeros((512,512,3),np.uint8)

pts = np.array([[10,50],[400,50],[90,200],[50,500]],np.int32)

pts = pts.reshape((-1,1,2))

cv2.polylines(image,[pts],True,(0,0,255),3)
cv2.imshow("Polygon",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# add text
image = np.zeros((512,512,3),np.uint8)

cv2.putText(image,'Hello World!',(75,290),cv2.FONT_HERSHEY_COMPLEX,2,(100,170,0),3)
cv2.imshow("Hello World!",image)
cv2.waitKey(0)
cv2.destroyAllWindows()


