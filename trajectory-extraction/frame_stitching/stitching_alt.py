import cv2
import numpy as np
import matplotlib.pyplot as plt
# import imageio
import imutils
cv2.ocl.setUseOpenCL(False)


def other_stitching(img1_color, img2_color, foreground_mask, background_mask, landscapeReference, ladscapeFront, frame_index, new_centroid, old_centroids):
    # img1 align, img2 ref

    img2_padded = cv2.copyMakeBorder(
        img2_color, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    height_orig, width_orig, depth_orig = img2_color.shape
    background_mask_padded = cv2.copyMakeBorder(
        background_mask, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    landscape_ref_padded = cv2.copyMakeBorder(
        landscapeReference, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=(0, 0, 0))

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

    homography, _ = cv2.estimateAffinePartial2D(p1, p2)

    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv2.warpAffine(img1_color,
                                     homography, (width, height))

    transformed_mask = cv2.warpAffine(foreground_mask,
                                      homography, (width, height))

    transformed_landscape = cv2.warpAffine(ladscapeFront,
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

    dst_orig_shape = dst.shape

    landscape_part = (landscape_ref_padded * (1 / 255.0)) * \
        (background_mask * (1 / 255.0))
    landscape_overlay_part = (transformed_landscape * (1 / 255.0)) * \
        (overlay_mask * (1 / 255.0))
    landscape_dst = np.uint8(cv2.addWeighted(
        landscape_part, 255.0, landscape_overlay_part, 255.0, 0.0))

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
    landscape_dst = landscape_dst[y:y+h, x:x+w]

    print("bounding box", x, y, w, h)
    print("original and new resolutions", dst_orig_shape, dst.shape)

    new_centroid_transformed = cv2.transform(
        np.array([[new_centroid]]), homography)[0]
    new_centroid_tuple = (int(new_centroid_transformed[0][0]), int(
        new_centroid_transformed[0][1]))
    new_centroid_dst = (new_centroid_tuple[0] - x, new_centroid_tuple[1] - y)
    test = dst.copy()
    test = cv2.circle(test, new_centroid_dst, radius=3,
                      color=(0, 0, 255), thickness=-1)

    new_centroids = []
    offset_x = 200 - x
    offset_y = 200 - y
    print("offsets", offset_x, offset_y)

    for centroid in old_centroids:
        centroid_offset = (centroid[0] + offset_x, centroid[1] + offset_y)
        print(centroid, centroid_offset)
        new_centroids.append(centroid_offset)
        test = cv2.circle(test, centroid_offset, radius=3,
                          color=(0, 255, 255), thickness=-1)

    cv2.imshow('stitching result', test)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    new_centroids.append(new_centroid_dst)

    return dst, overlay_mask, landscape_dst, new_centroids


def match_pairwise(img1_color, img2_color, foreground_mask, background_mask, landscapeReference, ladscapeFront, new_centroid):
        # img1 align, img2 ref

    # Convert to grayscale.
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
    height, width = img2.shape

    img1 = cv2.fastNlMeansDenoising(img1, 30.0, 7, 21)
    img2 = cv2.fastNlMeansDenoising(img2, 30.0, 7, 21)

    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(5000)

    # Find keypoints and descriptors.
    # The first arg is the image, second arg is the mask
    #  (which is not reqiured in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, foreground_mask)
    kp2, d2 = orb_detector.detectAndCompute(img2, background_mask)

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

    # homography, _ = cv2.estimateAffinePartial2D(p1, p2)
    (homography, status) = cv2.findHomography(p1, p2, cv2.RANSAC, 4)

    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(img1_color,
                                     homography, (width, height))

    to_transform = np.array([new_centroid], dtype=np.float32)
    new_centroid_transformed = cv2.perspectiveTransform(
        to_transform[np.newaxis], homography)[0]
    new_centroid_tuple = (int(new_centroid_transformed[0][0]), int(
        new_centroid_transformed[0][1]))
    test = transformed_img.copy()
    test = cv2.circle(test, new_centroid_tuple, radius=3,
                      color=(0, 0, 255), thickness=-1)

    cv2.imshow('reference', img2_color)
    cv2.imshow('matched image', test)
    cv2.waitKey()
    cv2.destroyAllWindows

    return transformed_img, new_centroid_tuple


