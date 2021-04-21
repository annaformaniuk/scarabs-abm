import numpy as np
import cv2
import random
import string
import os
import re
import argparse
# python make_border.py --image_path "F:\Dokumente\Uni_Msc\Thesis\repo\scarabs-abm\trajectory-extraction\480stitched_landscape_last.png"


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-image", "--image_path", required=True,
                help="Path to the image")
args = vars(ap.parse_args())

if (args["image_path"]):
    image = cv2.imread(args["image_path"])
    test = image.copy()
    height, width = image.shape[:2]
    
    for i in range(height):
        for j in range(width):
            if (i < 8 or i > height - 8):
                test[i,j] = 0
            else:
                if (j < 8 or j > width - 8):
                    test[i,j] = 0
            

    cv2.imshow("test", test)
    cv2.imshow("original", image)
    cv2.waitKey()
    cv2.destroyAllWindows()
