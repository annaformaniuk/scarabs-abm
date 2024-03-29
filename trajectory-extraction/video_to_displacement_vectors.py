import numpy as np
import cv2
import os
import argparse
import math
from frame_stitching import stitching
from object_detection.yolo_detect_picture import Yolo_detector
from contours.contours_hed import Contours_detector
from object_detection.shadow_detection import detect_shadow
from imutils.video import count_frames
import json

# python video_to_displacement_vectors.py --video_path "F:\Dokumente\Uni_Msc\Thesis\videos\Cut_trajectories\not_processed\Ambiguus_#14_Rolling from dung pat_20161117_cut.mp4" --ball_size 2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-video_path", "--video_path", required=True,
                help="Path to the video")
ap.add_argument("-ball_size", "--ball_size", required=True,
                help="Path to the video")
args = vars(ap.parse_args())

FIRST_FRAME = 93

def mask_out_objects(frame, objects):
    height, width, depth = frame.shape

    black_img = np.zeros((height, width, 1), dtype="uint8")
    white_img = 255 - black_img

    masked_objects = white_img.copy()
    for item in objects:
        bounds = item["box"]
        masked_objects[bounds[1]:bounds[3], bounds[0]:bounds[2]
                       ] = black_img[bounds[1]:bounds[3], bounds[0]:bounds[2]]

    return masked_objects


def get_centroid(bounds):
    x = int((bounds[2] - bounds[0])/2 + bounds[0])
    y = int((bounds[3] - bounds[1])/2 + bounds[1])
    return (x, y)


def get_diagonal(bounds):
    print(bounds)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    diagonal = math.sqrt(width**2 + height**2)
    # print("diagonal:", diagonal)
    return int(diagonal)

def save_geojson(points, vectors_array, name, ball_diagonal, ball_size, i):
    points_json = {
        "properties": [
            {
                "filename": name,
                "ball_pixelsize": ball_diagonal,
                "ball_realsize": float(ball_size)
            }
        ],
        "points": [{"point_coords": [0,0], "frame_number": 0}]
    }

    for num, point in enumerate(points):
        points_json["points"].append({
            "point_coords": point.tolist(),
            "frame_number": i[num],
            "displacement_vector": vectors_array[num].tolist()
        })

    points_json["points"].pop(0)

    # print('ready to save')
    # print(points_json)

    filename = "trajectory_"+ name + ".json"
    with open(filename, 'w') as f:
        json.dump(points_json, f)

def get_displacement_vector(first_coords, second_coords):
    scalar_x = second_coords[0] - first_coords[0]
    scalar_y = second_coords[1] - first_coords[1]

    return (scalar_x, scalar_y)


def reproduce_trajectory(displacement_vectors, diagonals, name, ball_size, i):
    reference_diagonal = diagonals[0]
         
    vectors_array = np.array(displacement_vectors)
    diagonals_array = np.array(diagonals)

    # scaling the displacement vectors to the size of the ball because
    # it's the only constant object in the frames
    scale = reference_diagonal/diagonals_array

    test = scale[:, np.newaxis]

    diagonals_scaled_floats = np.multiply(vectors_array, scale[:, np.newaxis])
    diagonals_scaled = diagonals_scaled_floats.astype(int)

    starting_point = diagonals_scaled.sum(axis=0)
    starting_point_array = np.array(starting_point)
    # to start in the middle
    width = abs(starting_point[0]) + abs(starting_point[0])*10
    height = abs(starting_point[1]) + abs(starting_point[1])

    trajectory = starting_point_array + np.cumsum(diagonals_scaled, axis=0)
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

    # cv2.imshow('black_img', black_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(str(name) + '_trajectory_reconstruction.png', black_img)

    # print(vectors_array)

    vectors_array_full = np.concatenate(([[0,0]],vectors_array))
    print(vectors_array_full)
    save_geojson(trajectory, vectors_array_full, name, reference_diagonal, ball_size, i)


if (os.path.isfile(args["video_path"])):
    print(args)
    cap = cv2.VideoCapture(args["video_path"])
    yolo = Yolo_detector()
    contours = Contours_detector()
    kernel = np.ones((15, 15), np.uint8)
    displacement_vectors = []
    ball_diagonals = []
    frame_counts = []
    first_ball_diagonal = None
    first_coords = None
    second_coords = None
    total_frames_count = count_frames(args["video_path"])
    print('total frames count: ', total_frames_count)

    filename = os.path.splitext(os.path.basename(args["video_path"]))[0]

    while True:
        ret, frame = cap.read()
        i = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if ret:
            if (i == FIRST_FRAME):
                imReference = frame.copy()
                
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

                ball_bounds = next(
                    (x for x in objects if x["label"] == "Ball"), None)
                if(ball_bounds != None):
                    first_ball_diagonal = get_diagonal(ball_bounds["box"])
                # else:
                #     first_ball_diagonal = None
                
                frame_counts.append(i)

                print('first ball!', first_ball_diagonal)

            if (i > FIRST_FRAME and i % 31 == 0):
                print('processing frame', i)
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

                    ball_bounds = next(
                        (x for x in objects if x["label"] == "Ball"), None)
                    ball_diagonal = None
                    if(ball_bounds != None):
                        ball_diagonal = get_diagonal(ball_bounds["box"])
                        ball_diagonals.append(ball_diagonal)
                        frame_counts.append(i)
                    else:
                        ball_diagonal = first_ball_diagonal
                        ball_diagonals.append(ball_diagonal)
                        frame_counts.append(i)
                    

                    # finally stitching the images together and replacing variables
                    matched_image, matched_coords = stitching.match_pairwise(
                        frame, imReference, foreground_mask, background_mask, landscapeReference, landscapeFront, second_coords)

                    # calculate the displacement vector between first_coords and matched_coords
                    displacement_vector = get_displacement_vector(
                        first_coords, matched_coords)
                    displacement_vectors.append(displacement_vector)

                    first_coords = second_coords

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

                ball_diagonals = list(
                    map(lambda x: int(np.average([y for y in ball_diagonals if y != None])) if x == None else x, ball_diagonals))

            if (i > total_frames_count - 10):
                print(ball_diagonals)
                print(frame_counts)
                ball_diagonals = list(
                    map(lambda x: int(np.average([y for y in ball_diagonals if y != None])) if x == None else x, ball_diagonals))
                print(ball_diagonals)
                reproduce_trajectory(displacement_vectors, ball_diagonals, filename, args["ball_size"], frame_counts)
                break

            if cv2.waitKey(1) & 0xFF == ord('q') or i == total_frames_count:
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
