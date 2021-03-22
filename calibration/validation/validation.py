import numpy as np
import os
import argparse
import json
import math
from scipy.stats import chisquare

# python validation.py -input_trajectories "F:\Dokumente\Uni_Msc\Thesis\repo\scarabs-abm\calibration\trajectories\validation" -model_stats "F:\Dokumente\Uni_Msc\Thesis\repo\scarabs-abm\calibration\results_default.json"


def calculate_stats(pts, times, scale, displacement_vectors):
    # first calculate the total length of the trajectory
    apts = np.array(pts)  # Make it a numpy array
    # Length between corners
    lengths = np.sqrt(np.sum(np.diff(apts, axis=0)**2, axis=1))
    real_lengths = lengths * scale  # in cm
    real_total_length = np.sum(real_lengths)
    print('real length of trajectory', real_total_length, 'cm')

    # now the total duration
    times_array = np.array(times)
    times_array[0] = 0
    time_diffs = times_array[1:] - times_array[:-1]
    time_length = np.sum(time_diffs)  # in seconds
    print('duration of trajectory', time_length, 'seconds')

    # and the speeds
    speeds = np.divide(real_lengths, time_diffs)

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

    # find what heading the beetle chose (10 ?)
    first_headings = headings[:5]
    default_heading = np.average(first_headings)
    print('default heading', default_heading)

    # TODO handle circularity

    # Calculate deviations
    heading_deviations = np.subtract(headings, [default_heading]).astype(int)
    # same bins as in netlogo
    bins = np.arange(-360, 361, 30)
    histogram = np.histogram(heading_deviations, bins=bins)
    print('histogram', histogram[0])

    # return real_total_length, time_length, average_speed
    return speeds, real_total_length, time_length, histogram[0]


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-input_trajectories", "--input_trajectories", required=True,
                    help="path to input trajectories")
    ap.add_argument("-model_stats", "--model_stats", required=True,
                    help="path to the model output")

    args = vars(ap.parse_args())
    print('args', args)

    # first reading the model statistics
    with open(args["model_stats"]) as json_file:
        print('reading file', args["model_stats"])
        model_stats = json.load(json_file)
        traj_stats = None
        folder_items = os.listdir(args["input_trajectories"])
        trajectories = [fi for fi in folder_items if fi.endswith(".json")]
        i = 0

        traj_full_stats = {
            'amount': len(trajectories),
            'speeds': [],
            'speed_means': [],
            'speed_stds': [],
            'distances': [],
            'distances_means': [],
            'distances_stds': [],
            'durations': [],
            'durations_means': [],
            'durations_stds': [],
            'headings': []
        }

        # then loading and processing all the trajectories
        while i < len(trajectories):
            with open(args["input_trajectories"] + "/" + trajectories[i]) as json_file:
                print('reading file', trajectories[i])
                data = json.load(json_file)
                trajectory_list = []
                times_list = []
                displacement_vectors = []
                ball_pixelsize = data['properties'][0]['ball_pixelsize']
                ball_realsize = data['properties'][0]['ball_realsize']
                fps = data['properties'][0]['fps']
                scale = ball_realsize / ball_pixelsize
                print('scale', scale)

                for point in data['points']:
                    trajectory_list.append(point['point_coords'])
                    displacement_vectors.append(point['displacement_vector'])
                    if (point['frame_number'] == 1):
                        times_list.append(0)
                    else:
                        times_list.append(point['frame_number'] / fps)

                speeds, real_total_length, time_length, heading_deviations = calculate_stats(
                    trajectory_list, times_list, scale, displacement_vectors)

                print('huh', heading_deviations)

                traj_full_stats['headings'].append(
                    heading_deviations)
                traj_full_stats['speeds'].append(speeds)
                traj_full_stats['distances'].append(real_total_length)
                traj_full_stats['durations'].append(time_length)

                traj_full_stats['speed_means'].append(np.mean(speeds))
                traj_full_stats['speed_stds'].append(np.std(speeds))
                # traj_full_stats['distances_means'].append(np.mean(real_total_length))
                # traj_full_stats['distances_stds'].append((np.std(real_total_length))
                # traj_full_stats['durations_means'].append(np.mean(time_length))
                # traj_full_stats['durations_stds'].append((np.std(time_length))
                

            i += 1

            average_hist = np.mean(
                traj_full_stats['headings'], axis=0)

            print('all speeds')
            print(np.array(traj_full_stats['speeds']))

            traj_stats = {
                'mean_speeds': np.mean(traj_full_stats['speed_means']),
                'std_speeds': np.mean(traj_full_stats['speed_stds']),
                'mean_dist': np.mean(traj_full_stats['distances']),
                'std_dist': np.std(traj_full_stats['distances']),
                'mean_time': np.mean(traj_full_stats['durations']),
                'std_time': np.std(traj_full_stats['durations']),
                'chisq': chisquare(average_hist)[0],
                'p': chisquare(average_hist)[1]
            }

        print('here come the model stats', model_stats)
        print('and here come the real stats', traj_stats)

        # TODO validation formula
