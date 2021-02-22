import numpy as np
import cv2
import os
import re

MAX_FEATURES = 200
GOOD_MATCH_PERCENT = 0.2

# 1 target, 2 reference
def alignImages(im1, im2, orb):
    """ OPTION ONE WITH ORB, for images of different size """
    # Grey and sharpened
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    im1Sharp = cv2.filter2D(im1, -1, kernel_sharpening)
    im2Sharp = cv2.filter2D(im2, -1, kernel_sharpening)

    if(orb):
        print("ORBBBBBBBBB")
        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(MAX_FEATURES)
        keypoints1, descriptors1 = orb.detectAndCompute(im1Sharp, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2Sharp, None)

        # Match features.
        matcher = cv2.DescriptorMatcher_create(
            cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)

        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]

        # # Draw top matches
        imMatches = cv2.drawMatches(
            im1, keypoints1, im2, keypoints2, matches, None)
        cv2.imwrite("matches.jpg", imMatches)

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

    else:
        """ OPTION TWO WITH OPTICAL FLOW """
        # find the coordinates of good features to track  in base
        base_features = cv2.goodFeaturesToTrack(im1Sharp, 50000, .015, 10)

        # find corresponding features in current photo
        curr_features = np.array([])
        curr_features, pyr_stati, _ = cv2.calcOpticalFlowPyrLK(
            im1Sharp, im2Sharp, base_features, curr_features, flags=1)

        # only add features for which a match was found to the pruned arrays
        base_features_pruned = []
        curr_features_pruned = []
        for index, status in enumerate(pyr_stati):
            if status == 1:
                base_features_pruned.append(base_features[index])
                curr_features_pruned.append(curr_features[index])

        # convert lists to numpy arrays so they can be passed to opencv function
        points1 = np.asarray(base_features_pruned)
        points2 = np.asarray(curr_features_pruned)

    """UNTIL HERE"""

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width = im2.shape
    # im1Reg = cv2.warpPerspective(im1, h, (width, height))
    im1Reg = cv2.warpPerspective(
        im1, h, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

    return im1Reg


if __name__ == '__main__':
    # manual_ref = r"F:\Dokumente\Eftas\IR4MRM\Data\Dortmund2\closer\manual_matching2\matched_manually.tif"
    inputFolder = r"F:\Dokumente\Uni_Msc\Thesis\trajectory_extraction\frames_subset"
    outputFolder = r"F:\Dokumente\Uni_Msc\Thesis\trajectory_extraction\frames_subset"
    folderItems = os.listdir(inputFolder)
    print(folderItems)
    folderItems.sort(key=lambda f: int(re.sub('\D', '', f)))
    print(folderItems)
    jpgs = [fi for fi in folderItems if fi.endswith(".jpg")]
    i = 0

    while i < len(jpgs)-1:
        # Read reference image
        refFilename = inputFolder + "/" + jpgs[i]
        # refFilename = manual_ref
        # print("Reading reference image : ", refFilename)
        imReference = cv2.imread(refFilename, 0)

        # Read image to be aligned
        imFilename = inputFolder + "/" + jpgs[i+1]
        # print("Reading image to align : ", imFilename)
        imTarget = cv2.imread(imFilename, 0)

        # Storing registered image
        imReg = alignImages(imTarget, imReference, orb=False)
        # Write aligned image to disk.
        cv2.imshow('imReg', imReg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        i += 1