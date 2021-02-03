import numpy as np
import cv2
import random
import string
import os
import re
import argparse
from frame_stitching import stitching, stitching_alt
from object_detection.yolo_detect_picture import Yolo_detector
from frame_stitching.warping import get_warp_matrix
from contours.contours_hed import Contours_detector
from object_detection.shadow_detection import detect_shadow
# python video_to_trajectory.py --video_path "F:\Dokumente\Uni_Msc\Thesis\videos\Allogymnopleuri_Rolling from dung pat_201611\resized\cut\Lamarcki_#01_Rolling from dung pat_20161114_cut_720.mp4"


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-video_path", "--video_path", required=True,
                help="Path to the video")
args = vars(ap.parse_args())


def mask_out_objects(frame, objects):
    height, width, depth = frame.shape

    black_img = np.zeros((height, width, 1), dtype="uint8")
    white_img = 255 - black_img

    masked_objects = white_img.copy()
    for item in objects:
        bounds = item["box"]
        masked_objects[bounds[1]:bounds[3], bounds[0]:bounds[2]
                       ] = black_img[bounds[1]:bounds[3], bounds[0]:bounds[2]]

    # cv2.imshow('mask', masked_objects)
    # cv2.imshow('frame', frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return masked_objects


def get_centroid(bounds):
    x = int((bounds[2] - bounds[0])/2 + bounds[0])
    y = int((bounds[3] - bounds[1])/2 + bounds[1])
    return (x, y)


if (os.path.isfile(args["video_path"])):
    cap = cv2.VideoCapture(args["video_path"])
    yolo = Yolo_detector()
    contours = Contours_detector()
    kernel = np.ones((15, 15), np.uint8)
    beetle_trajectory = []

    while True:
        ret, frame = cap.read()
        i = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if ret:
            if (i == 1):
                imReference = frame.copy()
                # that's what the previously matched frames will become, so that they are not used as reference
                height, width, depth = frame.shape
                background_mask = 255 * np.ones((height, width, 1), np.uint8)
                # connected_homography = [] # alternative stitching with matrix multiplication
                # https://web.archive.org/web/20140115053733/http://cs.bath.ac.uk:80/brown/papers/ijcv2007.pdf
                # H_03 = H_01 * H_12 * H_23.
                # connected_homography = np.matmul(connected_homography, homography)

                objects = yolo.detect_objects(frame)
                # masking out detected objects so that they won't be used as keypoins
                background_mask = mask_out_objects(imReference, objects)
                # masking out the hand with the camera too, hopefully
                largest_shadow = detect_shadow(frame)
                background_mask = cv2.bitwise_and(
                    background_mask, largest_shadow)

                # detecting contours for the landscape and masking out the eroded objects
                background_mask_eroded = cv2.erode(
                    background_mask, kernel, iterations=3)
                landscape = contours.detect_landscape(frame)
                landscapeReference = cv2.bitwise_and(
                    landscape, landscape, mask=background_mask_eroded)

                beetle_bounds = next(
                    (x for x in objects if x["label"] == "Beetle"), None)
                if(beetle_bounds != None):
                    beetle_point = get_centroid(beetle_bounds["box"])
                    beetle_trajectory.append(beetle_point)
                    test = imReference.copy()
                    test = cv2.circle(test, beetle_point, radius=3, color=(
                        0, 0, 255), thickness=-1)
                    cv2.imshow('first frame', test)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

            if (i > 1 and i < 6000 and i % 30 == 0):
                objects = yolo.detect_objects(frame)
                # masking out detected objects so that they won't be used as keypoins
                foreground_mask = mask_out_objects(frame, objects)
                # masking out the hand with the camera too, hopefully
                largest_shadow = detect_shadow(frame)
                foreground_mask = cv2.bitwise_and(
                    foreground_mask, largest_shadow)

                # detecting contours for the landscape and masking out the eroded objects
                foreground_mask_eroded = cv2.erode(
                    foreground_mask, kernel, iterations=3)
                landscape = contours.detect_landscape(frame)
                landscapeFront = cv2.bitwise_and(
                    landscape, landscape, mask=foreground_mask_eroded)

                beetle_bounds = next(
                    (x for x in objects if x["label"] == "Beetle"), None)
                if(beetle_bounds != None):
                    beetle_point = get_centroid(beetle_bounds["box"])

                # finally stitching the images together and replacing variables
                imReference, background_mask, landscapeReference, beetle_trajectory = stitching_alt.other_stitching(
                    frame, imReference, foreground_mask, background_mask, landscapeReference, landscapeFront, i, beetle_point, beetle_trajectory)

            if cv2.waitKey(1) & 0xFF == ord('q') or i > 6000:
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
