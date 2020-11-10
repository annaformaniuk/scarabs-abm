import numpy as np
import cv2 as cv

gray = cv.imread('frame_6450.jpg', 0)

# remove noise
img = cv.GaussianBlur(gray, (3, 3), 0)

# convolute with proper kernels
laplacian = cv.Laplacian(img, cv.CV_64F)
cv.imwrite('edges_laplacian.jpg', laplacian)
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)  # x
cv.imwrite('edges_sobelx.jpg', sobelx)
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)  # y
cv.imwrite('edges_sobely.jpg', sobely)
canny = cv.Canny(img, 100, 200)
cv.imwrite('edges_canny.jpg', canny)

# Smoothing without removing edges.
gray_filtered = cv.bilateralFilter(img, 7, 50, 50)
edges_filtered = cv.Canny(gray_filtered, 100, 200)
cv.imwrite('edges_canny_filtered.jpg', edges_filtered)

