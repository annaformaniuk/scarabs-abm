import numpy as np
import cv2


def detect_shadow(img_bgr):
    kernel = np.ones((5, 5), np.uint8)
    height, width, depth = img_bgr.shape
    black_img = np.zeros((height, width, 1), dtype="uint8")

    img_hsv: np.ndarray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(src=img_hsv, lowerb=np.array(
        [0, 34, 83]), upperb=np.array([179, 255, 255]))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    dilation = cv2.dilate(closing, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(
        dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    masked_largest_shadow = cv2.fillPoly(
        black_img, pts=[c], color=(255, 255, 255))

    return masked_largest_shadow
