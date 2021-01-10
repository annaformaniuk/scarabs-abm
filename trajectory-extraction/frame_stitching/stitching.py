import cv2
import numpy as np
import matplotlib.pyplot as plt
# import imageio
import imutils
cv2.ocl.setUseOpenCL(False)


def detect_and_describe(image, mask, method=None):
    """
    Compute key points and feature descriptors using an specific method
    """

    assert method is not None, "You need to define a feature detection method. Values are: 'sift', 'surf'"

    # detect and extract features from the image
    if method == 'sift':
        descriptor = cv2.xfeatures2d.SIFT_create()
    elif method == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()  # 500?

    # get keypoints and descriptors
    if(mask is not None):
        # gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        (kps, features) = descriptor.detectAndCompute(image, mask)
    else:
        (kps, features) = descriptor.detectAndCompute(image, None)

    return (kps, features)


def create_matcher(method, crossCheck):
    "Create and return a Matcher Object"

    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf


def match_keypoints_BF(featuresA, featuresB, method):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    best_matches = bf.match(featuresA, featuresB)

    # Sort the features in order of Hamming distance.
    # The points with small distance (more similarity) are ordered first in the vector
    rawMatches = sorted(best_matches, key=lambda x: x.distance)
    print("Raw matches (Brute force):", len(rawMatches))
    return rawMatches


def match_keypoints_KNN(featuresA, featuresB, ratio, method):
    bf = create_matcher(method, crossCheck=False)
    # compute the raw matches and initialize the list of actual matches
    rawMatches = bf.knnMatch(featuresA, featuresB, 2)
    print("Raw matches (knn):", len(rawMatches))
    matches = []

    # loop over the raw matches
    for m, n in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches


def calculate_homography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
    # convert the keypoints to numpy arrays
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])

    if len(matches) > 4:

        # construct the two sets of points
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])

        # estimate the homography between the sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                         reprojThresh)

        return (matches, H, status, ptsA, ptsB)
    else:
        return None


def stitch_images(img1_color, img2_color, foreground_mask, background_mask):
    # img1 align, img2 ref

    feature_extractor = 'orb'  # one of 'sift', 'surf', 'brisk', 'orb'
    feature_matching = 'bf'

    # Convert to grayscale.
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    # Find keypoints and descriptors.
    kp1, d1 = detect_and_describe(img1, mask=foreground_mask, method=feature_extractor)
    kp2, d2 = detect_and_describe(
        img2, mask=background_mask, method=feature_extractor)

    # Match features between the two images.
    # Sort them on the basis of their Hamming distance.
    matches = match_keypoints_BF(d1, d2, method=feature_matching)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:100],
                            None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    fig = plt.figure(figsize=(16,8))
    plt.imshow(img3)
    plt.show()

    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches)*90)]

    (matches, homography, status, ptsA, ptsB) = calculate_homography(
        kp1, kp2, d1, d2, matches, reprojThresh=4)

    height1, width1, depth1 = img1_color.shape
    height2, width2, depth2 = img2_color.shape
    height = height1 + height2
    width = width1 + width2

    # all this not to crop the left and top sides because they will have negative values
    input_corners = np.array([[0, 0], [width1, 0], [0, height1], [
                             width1, height1]], dtype=np.float32)
    output_corners = cv2.perspectiveTransform(
        input_corners[np.newaxis], homography)
    bounding_rect = cv2.boundingRect(output_corners)  # x,y,w,h

    output_quad = np.array(
        [[0, 0], [width2, 0], [0, height2], [width2, height2]])

    ptsA_new = []
    for point in ptsA:
        ptsA_new.append(np.array(
            [int(point[0] + bounding_rect[0]), int(point[1] + bounding_rect[1])], dtype=np.float32))

    ptsA_new = np.array(ptsA_new, dtype=np.float32)

    # homography_new = cv2.getPerspectiveTransform(ptsA, ptsB)
    (homography_new, status) = cv2.findHomography(ptsA_new, ptsB, cv2.RANSAC, 4)

    # Use this matrix to transform the
    # colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(img1_color,
                                          homography_new, (width, height))

    transformed_mask = cv2.warpPerspective(foreground_mask,
                                          homography_new, (width, height)) 
    
    print("bounding rect", bounding_rect)

    img2_padded = cv2.copyMakeBorder(img2_color, abs(bounding_rect[0]), height - height2 - abs(bounding_rect[0]), abs(
        bounding_rect[1]), width - width2 - abs(bounding_rect[1]), cv2.BORDER_CONSTANT, value=(0, 0, 0))

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
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    c = max(cnts, key=cv2.contourArea)
    (x,y,w,h) = cv2.boundingRect(c)

    dst = dst[y:y+h, x:x+w]
    overlay_mask = overlay_mask[y:y+h, x:x+w]
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_BGR2GRAY)
    transformed_mask = transformed_mask[y:y+h, x:x+w]
    overlay_mask = cv2.bitwise_and(overlay_mask, transformed_mask)

    cv2.imshow('overlay_mask', overlay_mask)
    cv2.imshow('dst', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return dst, overlay_mask
