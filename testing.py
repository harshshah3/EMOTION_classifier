import numpy as np
import cv2
img = cv2.imread('Model_Prediction/Images/test_image.jpg',0)
#img2 = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#img3 = cv2.resize(img2,(48,48)).astype(np.float32)

cv2.imshow('img',img)
cv2.waitKey(3000)
cv2.destroyWindow(img)