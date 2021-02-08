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
import matplotlib.pyplot as plt
from imutils.video import count_frames
# python video_to_displacement_vectors.py --video_path "F:\Dokumente\Uni_Msc\Thesis\videos\Allogymnopleuri_Rolling from dung pat_201611\resized\cut\Lamarcki_#01_Rolling from dung pat_20161114_cut_720.mp4"


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


def get_displacement_vector(first_coords, second_coords):
    scalar_x = second_coords[0] - first_coords[0]
    scalar_y = second_coords[1] - first_coords[1]
    # print(first_coords, second_coords)
    # print(scalar_x, scalar_y)
    return (scalar_x, scalar_y)


def reproduce_trajectory(displacement_vectors):
    vectors_array = np.array(displacement_vectors)
    # print(vectors_array)
    starting_point = vectors_array.sum(axis=0)
    starting_point_array = np.array(starting_point)
    # to start in the middle
    width = abs(starting_point[0]) + abs(starting_point[0])
    height = abs(starting_point[1]) + abs(starting_point[1])

    trajectory = starting_point_array + np.cumsum(vectors_array, axis=0)
    trajectory = np.insert(trajectory, 0, starting_point_array, axis=0)
    print(trajectory)

    (minimal_x, minimal_y) = trajectory.min(axis=0)

    if(minimal_x < 0):
        trajectory[:, 0] += abs(minimal_x)
    if(minimal_y < 0):
        trajectory[:, 1] += abs(minimal_y)
    print(trajectory)

    pts = trajectory.reshape((-1, 1, 2))
    isClosed = False
    color = (0, 0, 255)
    thickness = 5
    black_img = np.zeros((height, width, 3), dtype="uint8")

    black_img = cv2.polylines(black_img, [pts],
                              isClosed, color, thickness)

    cv2.imshow('black_img', black_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('trajectory_reconstruction.png', black_img)


if (os.path.isfile(args["video_path"])):
    cap = cv2.VideoCapture(args["video_path"])
    yolo = Yolo_detector()
    contours = Contours_detector()
    kernel = np.ones((15, 15), np.uint8)
    displacement_vectors = []
    first_coords = None
    second_coords = None
    total_frames_count = count_frames(args["video_path"])
    print('total frames count: ', total_frames_count)

    while True:
        ret, frame = cap.read()
        i = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if ret:
            if (i == 1):
                imReference = frame.copy()
                # that's what the previously matched frames will become, so that they are not used as reference
                height, width, depth = frame.shape
                background_mask = 255 * np.ones((height, width, 1), np.uint8)

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
                    first_coords = get_centroid(beetle_bounds["box"])

            if (i > 1 and i < 6000 and i % 30 == 0):
                height, width, depth = frame.shape
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
                    second_coords = get_centroid(beetle_bounds["box"])

                # finally stitching the images together and replacing variables
                matched_image, matched_coords = stitching_alt.match_pairwise(
                    frame, imReference, foreground_mask, background_mask, landscapeReference, landscapeFront, second_coords)

                # calculate the displacement vector between first_coords and matched_coords
                displacement_vector = get_displacement_vector(
                    first_coords, matched_coords)
                displacement_vectors.append(displacement_vector)

                # plt.gca().invert_yaxis()
                # print(first_coords[0], first_coords[1],
                #            displacement_vector[0], displacement_vector[1])
                # plt.quiver(first_coords[0], first_coords[1],
                #            displacement_vector[0], displacement_vector[1], angles='xy')
                # plt.show()

                # making this frame info to the reference
                imReference = frame
                landscapeReference = landscapeFront
                background_mask = foreground_mask
                first_coords = second_coords

            # if (i == total_frames_count - 1):

            if (i > 600):
                reproduce_trajectory(displacement_vectors)
                break

            if cv2.waitKey(1) & 0xFF == ord('q') or i == total_frames_count:
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
