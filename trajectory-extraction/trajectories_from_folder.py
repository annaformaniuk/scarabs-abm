import numpy as np
import os
import argparse
import json
import math
from scipy.stats import chisquare
from video_to_displacement_vectors import process_video
# python trajectories_from_folder.py -video_path "F:\Dokumente\Uni_Msc\Thesis\videos\Cut_trajectories\not_processed" -ball_size 2

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-video_path", "--video_path", required=True,
                    help="path to input videos")
    ap.add_argument("-ball_size", "--ball_size", required=True,
                    help="guess the size of the ball")

    args = vars(ap.parse_args())

    folder_items = os.listdir(args["video_path"])
    videos = [fi for fi in folder_items if fi.endswith(".mp4")]
    i = 0

    print(args["video_path"] + "/" + videos[i])

    while i < len(videos):
        inputs = {
            'video_path': args["video_path"] + "/" + videos[i],
            'ball_size': args["ball_size"]
        }
        print('will process video', inputs['video_path'])

        process_video(inputs)

        i += 1

