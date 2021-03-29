import numpy as np
import os
import argparse
import json
import math
from scipy.stats import chisquare

# python trajectories_to_statistics.py -input "F:\Dokumente\Uni_Msc\Thesis\repo\scarabs-abm\trajectory-extraction\trajectories" -output "F:\Dokumente\Uni_Msc\Thesis\repo\scarabs-abm\trajectory-extraction\trajectories"

def calculate_stats(pts, times, scale, displacement_vectors):
    # first calculate the total length of the trajectory
    apts = np.array(pts) # Make it a numpy array
    lengths = np.sqrt(np.sum(np.diff(apts, axis=0)**2, axis=1)) # Length between corners
    real_lengths = lengths * scale # in cm
    real_total_length = np.sum(real_lengths)
    print('real length of trajectory', real_total_length, 'cm')

    # now the total duration
    times_array = np.array(times)
    times_array[0] = 0
    time_diffs = times_array[1:] - times_array[:-1]
    time_length = np.sum(time_diffs) # in seconds
    print('duration of trajectory', time_length, 'seconds')

    # and the speeds
    speeds = np.divide(real_lengths, time_diffs)
    # print('all speeds during the trajectory, cm/second', speeds)
    average_speed = np.average(speeds)
    print('average speed, cm/second', average_speed)

    # and the headings
    displacement_vectors_ar = np.array(displacement_vectors)

    def heading(row):
        angle = math.atan2(row[1], row[0])*180/math.pi
        # angles should be between 0 and 360
        if (angle < 0):
            angle = angle + 360
        return angle

    headings = np.apply_along_axis(heading, 1, displacement_vectors_ar)
    headings = np.delete(headings, 0)
    # print('headings total', headings)

    # find what heading the beetle chose (10 ?)
    first_headings = headings[:5]
    default_heading = np.average(first_headings)
    print('default heading', default_heading)
    
    # TODO handle circularity

    # Calculate deviations
    heading_deviations = np.subtract(headings, [default_heading]).astype(int)
    # print('heading_deviations', heading_deviations)
    # same bins as in netlogo
    bins = np.arange(0, 361, 30)
    histogram = np.histogram(heading_deviations, bins=bins)
    print('histogram', histogram)
    histogram_stats = chisquare(histogram[0])
    print('histogram stats', histogram_stats)

    return real_total_length, time_length, average_speed

if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-input", "--input_folder", required=True,
                    help="path to input trajectories")
    ap.add_argument("-output", "--output_folder", required=True,
                    help="path to output folder where to save the results")

    args = vars(ap.parse_args())

    folder_items = os.listdir(args["input_folder"])
    trajectories = [fi for fi in folder_items if fi.endswith(".json")]
    i = 0

    while i < len(trajectories):
        with open(args["input_folder"] + "/" + trajectories[i]) as json_file:
            print('reading file', trajectories[i])
            data = json.load(json_file)
            trajectory_list = []
            times_list = []
            displacement_vectors = []
            ball_pixelsize = data['properties'][0]['ball_pixelsize']
            ball_realsize = data['properties'][0]['ball_realsize']
            fps = data['properties'][0]['fps']
            print(fps)
            scale = ball_realsize / ball_pixelsize
            print('scale', scale)

            for point in data['points']:
                trajectory_list.append(point['point_coords'])
                displacement_vectors.append(point['displacement_vector'])
                if (point['frame_number'] == 1):
                    times_list.append(0)
                else:
                    times_list.append(point['frame_number'] / fps)

            something = calculate_stats(trajectory_list, times_list, scale, displacement_vectors)

        i += 1

