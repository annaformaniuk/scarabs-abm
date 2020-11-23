import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10
img1 = cv.imread('beetle.JPG', 0)          # queryImage
img2 = cv.imread('full.jpg', 0)  # trainImage

# Grey and sharpened
kernel_sharpening = np.array([[-1, -1, -1],
                                [-1, 9, -1],
                                [-1, -1, -1]])

img1 = cv.bilateralFilter(img1,9,75,75)
img2 = cv.bilateralFilter(img2,9,75,75)

img1 = cv.filter2D(img1, -1, kernel_sharpening)
img2 = cv.filter2D(img2, -1, kernel_sharpening)

# # Initiate SIFT detector
# sift = cv.SIFT()
# Initiate STAR detector
orb = cv.ORB_create()

# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)

# find the keypoints with ORB
kp1 = orb.detect(img1, None)
kp2 = orb.detect(img2, None)

# compute the descriptors with ORB
kp1, des1 = orb.compute(img1, kp1)
kp2, des2 = orb.compute(img2, kp2)

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1, des2)

# Sort them in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)

# Draw first 10 matches.
img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)

plt.imshow(img3), plt.show()
