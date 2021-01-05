import cv2
import numpy as np
import matplotlib.pyplot as plt
# import imageio
import imutils
cv2.ocl.setUseOpenCL(False)



def detectAndDescribe(image, method=None):
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
        descriptor = cv2.ORB_create() # 500?
        
    # get keypoints and descriptors
    (kps, features) = descriptor.detectAndCompute(image, None)
    
    return (kps, features)

def createMatcher(method,crossCheck):
    "Create and return a Matcher Object"
    
    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf

def matchKeyPointsBF(featuresA, featuresB, method):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) 
        
    # Match descriptors.
    best_matches = bf.match(featuresA,featuresB)
    
    # Sort the features in order of Hamming distance.
    # The points with small distance (more similarity) are ordered first in the vector
    rawMatches = sorted(best_matches, key = lambda x:x.distance)
    print("Raw matches (Brute force):", len(rawMatches))
    return rawMatches

def matchKeyPointsKNN(featuresA, featuresB, ratio, method):
    bf = createMatcher(method, crossCheck=False)
    # compute the raw matches and initialize the list of actual matches
    rawMatches = bf.knnMatch(featuresA, featuresB, 2)
    print("Raw matches (knn):", len(rawMatches))
    matches = []

    # loop over the raw matches
    for m,n in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches

def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
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

        return (matches, H, status)
    else:
        return None


def other_stitching(img1_color, img2_color, counter):
    #img1 align, img2 ref

    feature_extractor = 'orb' # one of 'sift', 'surf', 'brisk', 'orb'
    feature_matching = 'bf'

    # Convert to grayscale. 
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY) 
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY) 
    height, width = img2.shape 
    
    # Find keypoints and descriptors. 
    kp1, d1 = detectAndDescribe(img1, method=feature_extractor)
    kp2, d2 = detectAndDescribe(img2, method=feature_extractor)
    
    # Match features between the two images. 
    # Sort them on the basis of their Hamming distance. 
    matches = matchKeyPointsBF(d1, d2, method=feature_matching)
    
    # Take the top 90 % matches forward. 
    matches = matches[:int(len(matches)*90)]
    
    (matches, homography, status) = getHomography(kp1, kp2, d1, d2, matches, reprojThresh=4)
    # Use this matrix to transform the 
    # colored image wrt the reference image. 
    transformed_img = cv2.warpPerspective(img1_color, 
                        homography, (width, height))

    gray = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2GRAY)
    overlay_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]
    overlay_mask = cv2.erode(overlay_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    overlay_mask = cv2.blur(overlay_mask, (3, 3))
    background_mask = 255 - overlay_mask
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    ref_part = (img2_color * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (transformed_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))
    dst = np.uint8(cv2.addWeighted(ref_part, 255.0, overlay_part, 255.0, 0.0))

    cv2.imshow('dst',dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return transformed_img