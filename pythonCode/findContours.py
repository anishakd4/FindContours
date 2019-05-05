import cv2
import sys
import numpy as np
import random

#read input image
image = cv2.imread("../assets/anish.jpg", cv2.IMREAD_GRAYSCALE)

#check if image exists
if image is None:
    print("can not find image")
    sys.exit()

#apply canny to the input image
canny = cv2.Canny(image, 50, 150, apertureSize=3)

#find contours
temp, contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#output image to draw contours on
output = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

#draw contours
for i in range(0, len(contours)):
    cv2.drawContours(output, contours, i, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 2)


#create windows to display images
cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("canny", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("contours", cv2.WINDOW_AUTOSIZE)

#display images
cv2.imshow("image", image)
cv2.imshow("canny", canny)
cv2.imshow("contours", output)

#press esc to exit the program
cv2.waitKey(0)

#close all the opened windows
cv2.destroyAllWindows()