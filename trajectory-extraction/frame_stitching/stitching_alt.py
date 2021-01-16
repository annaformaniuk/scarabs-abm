import cv2
import numpy as np
import matplotlib.pyplot as plt
# import imageio
import imutils
cv2.ocl.setUseOpenCL(False)


def other_stitching(img1_color, img2_color, foreground_mask, background_mask, frame_index):
    # img1 align, img2 ref

    img2_padded = cv2.copyMakeBorder(
        img2_color, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    height, width, depth = img2_color.shape
    background_mask_padded = cv2.copyMakeBorder(
        background_mask, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Convert to grayscale.
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_padded, cv2.COLOR_BGR2GRAY)
    height, width = img2.shape

    img1 = cv2.fastNlMeansDenoising(img1, 30.0, 7, 21)
    img2 = cv2.fastNlMeansDenoising(img2, 30.0, 7, 21)

    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)

    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not reqiured in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, foreground_mask)
    kp2, d2 = orb_detector.detectAndCompute(img2, background_mask_padded)

    # Match features between the two images.
    # We create a Brute Force matcher with
    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)

    # Sort matches on the basis of their Hamming distance.
    matches.sort(key=lambda x: x.distance)

    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches)*90)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(img1_color,
                                          homography, (width, height))

    transformed_mask = cv2.warpPerspective(foreground_mask,
                                           homography, (width, height))

    gray = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2GRAY)
    overlay_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]
    overlay_mask = cv2.erode(
        overlay_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    overlay_mask = cv2.blur(overlay_mask, (3, 3))
    background_mask = 255 - overlay_mask
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    ref_part = (img2_padded * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (transformed_img * (1 / 255.0)) * \
        (overlay_mask * (1 / 255.0))
    dst = np.uint8(cv2.addWeighted(ref_part, 255.0, overlay_part, 255.0, 0.0))

    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    # find contours
    cnts = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    c = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)

    dst = dst[y:y+h, x:x+w]
    overlay_mask = overlay_mask[y:y+h, x:x+w]
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_BGR2GRAY)
    transformed_mask = transformed_mask[y:y+h, x:x+w]
    overlay_mask = cv2.bitwise_and(overlay_mask, transformed_mask)

    cv2.imwrite(str(frame_index) + '.jpg', dst)
    cv2.imshow('overlay_mask', overlay_mask)
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return dst, overlay_mask
