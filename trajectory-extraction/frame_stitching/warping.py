import numpy as np
import cv2 as cv

def get_warp_matrix(im1, im2):
    # Convert images to grayscale
    im1_gray = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    im2_gray = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)

    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    im1_gray = cv.filter2D(im1_gray, -1, kernel_sharpening)
    im2_gray = cv.filter2D(im2_gray, -1, kernel_sharpening)

    # Define the motion model
    warp_mode = cv.MOTION_HOMOGRAPHY

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)

    return warp_matrix