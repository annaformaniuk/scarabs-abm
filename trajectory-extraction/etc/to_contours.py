import numpy as np
import cv2
import os
import re
import argparse
from contours.contours_hed import Contours_detector
# python to_contours.py --image_path "F:\Dokumente\Uni_Msc\Thesis\data_backups\270.jpg"


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-image_path", "--image_path", required=True,
                help="Path to the image")
args = vars(ap.parse_args())

print(os.path.isfile(args["image_path"]))
if (os.path.isfile(args["image_path"])):
    image = cv2.imread(args["image_path"])
    # contours = Contours_detector()
    # landscape = contours.detect_landscape(image)

    blackwhite = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('blackwhite', blackwhite)
    cv2.waitKey(0)
    height, width = blackwhite.shape
    black_img = np.zeros((height, width, 1), dtype="uint8")
    black_img = cv2.normalize(blackwhite, 0, 255, cv2.NORM_MINMAX)

    cv2.imwrite('blackwhite.png', blackwhite)

    cv2.imshow('normalized_landscape', blackwhite)
    cv2.waitKey(0)
