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
        descriptor = cv2.ORB_create()
        
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
    bf = createMatcher(method, crossCheck=True)
        
    # Match descriptors.
    best_matches = bf.match(featuresA,featuresB)
    
    # Sort the features in order of distance.
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

def get_homography(trainImg, queryImg, counter):
    # select the image id (valid values 1,2,3, or 4)
    feature_extractor = 'orb' # one of 'sift', 'surf', 'brisk', 'orb'
    feature_matching = 'bf'

    # read images and transform them to grayscale
    # Make sure that the train image is the image that will be transformed
    trainImg_gray = cv2.cvtColor(trainImg, cv2.COLOR_RGB2GRAY)

    # Opencv defines the color channel in the order BGR. 
    # Transform it to RGB to be compatible to matplotlib
    queryImg_gray = cv2.cvtColor(queryImg, cv2.COLOR_RGB2GRAY)

    kpsA, featuresA = detectAndDescribe(trainImg_gray, method=feature_extractor)
    kpsB, featuresB = detectAndDescribe(queryImg_gray, method=feature_extractor)

    print("Using: {} feature matcher".format(feature_matching))

    fig = plt.figure(figsize=(16,8))

    if feature_matching == 'bf':
        matches = matchKeyPointsBF(featuresA, featuresB, method=feature_extractor)
        img3 = cv2.drawMatches(trainImg,kpsA,queryImg,kpsB,matches[:100],
                            None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    elif feature_matching == 'knn':
        matches = matchKeyPointsKNN(featuresA, featuresB, ratio=0.75, method=feature_extractor)
        img3 = cv2.drawMatches(trainImg,kpsA,queryImg,kpsB,np.random.choice(matches,100),
                            None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
    plt.imshow(img3)
    plt.show()

    M = getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh=4)
    if M is None:
        print("Error no homography!")
    (matches, H, status) = M
    print(H)

    # return H

    # def apply_homography(image, H):



    #     return corrected

    # def stitch_images(image1, image2):

    # Apply panorama correction
    width = trainImg.shape[1] + queryImg.shape[1]
    height = trainImg.shape[0] + queryImg.shape[0]

    result = cv2.warpPerspective(trainImg, H, (width, height))
    result[0:queryImg.shape[0], 0:queryImg.shape[1]] = queryImg

    plt.figure(figsize=(16,8))
    plt.imshow(result)

    plt.axis('off')
    plt.show()

    # transform the panorama image to grayscale and threshold it 
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    # Finds contours from the binary image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # get the maximum contour area
    c = max(cnts, key=cv2.contourArea)

    # get a bbox from the contour area
    (x, y, w, h) = cv2.boundingRect(c)

    # crop the image to the bbox coordinates
    result = result[y:y + h, x:x + w]

    # show the cropped image
    plt.figure(figsize=(16,8))
    plt.imshow(result)
    cv2.imwrite(str(counter) + '.jpg', result)
    return result


def other_stitching(img1_color, img2_color, counter):
    #img1 align, img2 ref

    # Convert to grayscale. 
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY) 
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY) 
    height, width = img2.shape 
    
    # Create ORB detector with 5000 features. 
    orb_detector = cv2.ORB_create(5000) 
    
    # Find keypoints and descriptors. 
    # The first arg is the image, second arg is the mask 
    #  (which is not reqiured in this case). 
    kp1, d1 = orb_detector.detectAndCompute(img1, None) 
    kp2, d2 = orb_detector.detectAndCompute(img2, None) 
    
    # Match features between the two images. 
    # We create a Brute Force matcher with  
    # Hamming distance as measurement mode. 
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) 
    
    # Match the two sets of descriptors. 
    matches = matcher.match(d1, d2) 
    
    # Sort matches on the basis of their Hamming distance. 
    matches.sort(key = lambda x: x.distance) 
    
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