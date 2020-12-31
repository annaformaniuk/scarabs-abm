import numpy as np
import cv2 as cv
import random
import string
import os
import re
import argparse
from frame_stitching import stitching
from object_detection.yolo_detect_picture import Yolo_detector

# python video_to_trajectory.py --video_path "F:\Dokumente\Uni_Msc\Thesis\videos\Allogymnopleuri_Rolling from dung pat_201611\resized\cut\Lamarcki_#01_Rolling from dung pat_20161114_cut_720.mp4"


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-video_path", "--video_path", required=True,
                help="Path to the video")
args = vars(ap.parse_args())


if (os.path.isfile(args["video_path"])):
    cap = cv.VideoCapture(args["video_path"])
    yolo = Yolo_detector()

    while True:
        ret, frame = cap.read()
        i = int(cap.get(cv.CAP_PROP_POS_FRAMES))

        if ret:
            if (i == 1) :
                imReference = frame.copy()
                objects = yolo.detect_objects(frame) 
                # received data structure:
                # [{'label': 'Ball', 'box': [342, 174, 417, 234]}, {'label': 'Beetle', 'box': [356, 224, 391, 270]}]
                #x1, y1, x2, y2 = objects[0]["box"]
            # if (i > 1 and i < 6000 and i % 30 == 0):
                #imReference = stitching.stitch_images(imReference, frame, i)
                # warp_matrix = get_warp_matrix(imReference, frame)
                # sz = imReference.shape
                # imAligned = cv.warpPerspective (frame, warp_matrix, (sz[1],sz[0]), flags=cv.INTER_LINEAR + cv.WARP_INVERSE_MAP)

                # imReference = frame.copy()

                # cv.imshow("image", frame)
                # cv.waitKey(0)
            if cv.waitKey(1) & 0xFF == ord('q') or i > 6000:
                break
        else:
            break
    cap.release()
    cv.destroyAllWindows()
