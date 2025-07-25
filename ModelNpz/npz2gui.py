import numpy as np
import cv2

data = np.load('lena.npz')
img = data['lena']

# Convert RGB to BGR for OpenCV display
img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imshow('Lena Image', img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()