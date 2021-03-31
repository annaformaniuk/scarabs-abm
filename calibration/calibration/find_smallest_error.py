import numpy as np
import os
import argparse
import json
import math
import statistics
from scipy.stats import chisquare

# python find_smallest_error.py -model_stats_folder "F:\Dokumente\Uni_Msc\Thesis\repo\scarabs-abm\calibration\2000_6" -validation_trajectories "F:\Dokumente\Uni_Msc\Thesis\repo\scarabs-abm\calibration\trajectories\validation" -calibration_trajectories "F:\Dokumente\Uni_Msc\Thesis\repo\scarabs-abm\calibration\trajectories\training"


def calculate_stats(pts, times, scale, displacement_vectors):
    # first calculate the total length of the trajectory
    apts = np.array(pts)  # Make it a numpy array
    # Length between corners
    lengths = np.sqrt(np.sum(np.diff(apts, axis=0)**2, axis=1))
    real_lengths = lengths * scale  # in cm
    real_total_length = np.sum(real_lengths)
    # print('real length of trajectory', real_total_length, 'cm')

    # now the total duration
    times_array = np.array(times)
    times_array[0] = 0
    time_diffs = times_array[1:] - times_array[:-1]
    time_length = np.sum(time_diffs)  # in seconds
    # print('duration of trajectory', time_length, 'seconds')

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
    # print('default heading', default_heading)

    # TODO handle circularity

    # Calculate deviations
    heading_deviations = np.subtract(headings, [default_heading]).astype(int)
    # same bins as in netlogo
    bins = np.arange(0, 361, 30)
    histogram = np.histogram(heading_deviations, bins=bins)
    # print('histogram', histogram[0])

    # return real_total_length, time_length, average_speed
    return speeds, real_total_length, time_length, histogram[0]


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-model_stats_folder", "--model_stats_folder", required=True,
                    help="path to model outputs that were ran with different parameters")
    ap.add_argument("-validation_trajectories", "--validation_trajectories", required=True,
                    help="path to the trajectories that will be used for validation, needed for normalization")
    ap.add_argument("-calibration_trajectories", "--calibration_trajectories", required=True,
                    help="path to the calibration trajectories for brute force search")

    args = vars(ap.parse_args())
    print('args', args)

    model_stats_folder = os.listdir(args["model_stats_folder"])
    model_stats = [
        fi for fi in model_stats_folder if fi.endswith(".json")]

    model_stats_dicts = []

    i = 0

    # print(len(model_stats_folder))

    while i < len(model_stats_folder)-1:
        # first reading the model statistics
        with open(args["model_stats_folder"] + "/" + model_stats[i]) as json_file:
            model = json.load(json_file)
            model_stats_dicts.append(model)
        i += 1

    print('read all the model stats')

    calibration_stats = None
    calibration_items_folder = os.listdir(args["calibration_trajectories"])
    calibration_trajectories = [
        fi for fi in calibration_items_folder if fi.endswith(".json")]
    # print(trajectories)

    validation_items_folder = os.listdir(args["validation_trajectories"])
    validation_trajectories = [
        fi for fi in validation_items_folder if fi.endswith(".json")]

    i = 0

    all_values = {
        'speeds': [],
        'speed_stds': [],
        'distances': [],
        'distances_stds': [],
        'durations': [],
        'durations_stds': []
    }

    while i < len(validation_trajectories):
        with open(args["validation_trajectories"] + "/" + validation_trajectories[i]) as json_file:
            data = json.load(json_file)
            trajectory_list = []
            times_list = []
            displacement_vectors = []
            ball_pixelsize = data['properties'][0]['ball_pixelsize']
            ball_realsize = data['properties'][0]['ball_realsize']
            fps = data['properties'][0]['fps']
            scale = ball_realsize / ball_pixelsize
            # print('scale', scale)

            for point in data['points']:
                trajectory_list.append(point['point_coords'])
                displacement_vectors.append(point['displacement_vector'])
                if (point['frame_number'] == 1):
                    times_list.append(0)
                else:
                    times_list.append(point['frame_number'] / fps)

            speeds, real_total_length, time_length, heading_deviations = calculate_stats(
                trajectory_list, times_list, scale, displacement_vectors)

            speeds_temp = all_values['speeds']
            if (len(speeds_temp) == 0):
                speeds_temp = speeds
            else:
                np.concatenate([np.array(speeds_temp), np.array(speeds)])

            all_values['speeds'] = speeds_temp
            all_values['speed_stds'].append(np.std(speeds))
            all_values['distances'].append(real_total_length)
            all_values['durations'].append(time_length)

        i += 1

        i = 0

        calibration_full_stats = {
            'amount': len(calibration_trajectories),
            'speeds': [],
            'speed_means': [],
            'speed_stds': [],
            'distances': [],
            'durations': [],
            'headings': [],
            'norm_headings': []
        }

        # then loading and processing all the trajectories
        while i < len(calibration_trajectories):
            with open(args["calibration_trajectories"] + "/" + calibration_trajectories[i]) as json_file:
                # print('reading file', calibration_trajectories[i])
                data = json.load(json_file)
                trajectory_list = []
                times_list = []
                displacement_vectors = []
                ball_pixelsize = data['properties'][0]['ball_pixelsize']
                ball_realsize = data['properties'][0]['ball_realsize']
                fps = data['properties'][0]['fps']
                scale = ball_realsize / ball_pixelsize
                # print('scale', scale)

                for point in data['points']:
                    trajectory_list.append(point['point_coords'])
                    displacement_vectors.append(point['displacement_vector'])
                    if (point['frame_number'] == 1):
                        times_list.append(0)
                    else:
                        times_list.append(point['frame_number'] / fps)

                speeds, real_total_length, time_length, heading_deviations = calculate_stats(
                    trajectory_list, times_list, scale, displacement_vectors)

                calibration_full_stats['headings'].append(
                    heading_deviations)
                calibration_full_stats['norm_headings'].append(
                    (heading_deviations / np.sum(heading_deviations))*100)
                calibration_full_stats['speeds'].append(speeds)
                calibration_full_stats['distances'].append(real_total_length)
                calibration_full_stats['durations'].append(time_length)

                calibration_full_stats['speed_means'].append(np.mean(speeds))
                calibration_full_stats['speed_stds'].append(np.std(speeds))

                speeds_temp = all_values['speeds']
                np.concatenate([np.array(speeds_temp), np.array(speeds)])

                all_values['speeds'] = speeds_temp
                all_values['speed_stds'].append(np.std(speeds))
                all_values['distances'].append(real_total_length)
                all_values['durations'].append(time_length)

            i += 1

            average_hist = np.mean(
                calibration_full_stats['headings'], axis=0)

            # print('just average hist', average_hist)

            average_his_norm = np.mean(
                calibration_full_stats['norm_headings'], axis=0)

            # print('just average_hist_norm', average_his_norm)

            calibration_stats = {
                'mean_speeds': np.mean(calibration_full_stats['speed_means']),
                'std_speeds': np.mean(calibration_full_stats['speed_stds']),
                'mean_dist': np.mean(calibration_full_stats['distances']),
                'std_dist': np.std(calibration_full_stats['distances']),
                'mean_time': np.mean(calibration_full_stats['durations']),
                'std_time': np.std(calibration_full_stats['durations']),
                'heading_deviations': average_hist,
                'heading_deviations_norm': average_his_norm,
                'chisq': chisquare(average_his_norm)[0],
                'p': chisquare(average_his_norm)[1]
            }

    # now compute means for normalization
    all_values['mean_speeds'] = np.mean(np.array(all_values['speeds']))
    all_values['mean_speed_stds'] = np.mean(all_values['speed_stds'])
    all_values['mean_dist'] = np.mean(all_values['distances'])
    all_values['mean_time'] = np.mean(all_values['durations'])

    # print('here come the model stats', model_stats)
    # print('and here come the real stats', calibration_stats)

    # search for the smallest error
    for model_stats in model_stats_dicts:
        model_norm_values = np.array([
            model_stats['mean_speeds']/all_values['mean_speeds'],
            model_stats['std_speeds']/all_values['mean_speeds'],
            model_stats['mean_dist']/all_values['mean_dist'],
            model_stats['std_dist']/all_values['mean_dist'],
            model_stats['mean_time']/all_values['mean_time'],
            model_stats['std_time']/all_values['mean_time']
        ]
            # model_stats['chisq_new']/traj_stats['chisq']
        )
        trajectories_norm_values = np.array([
            calibration_stats['mean_speeds']/all_values['mean_speeds'],
            calibration_stats['std_speeds']/all_values['mean_speeds'],
            calibration_stats['mean_dist']/all_values['mean_dist'],
            calibration_stats['std_dist']/all_values['mean_dist'],
            calibration_stats['mean_time']/all_values['mean_time'],
            calibration_stats['std_time']/all_values['mean_time']
            # calibration_stats['chisq_new']/traj_stats['chisq']
        ]
        )

        rmse_total = rmse(model_norm_values, trajectories_norm_values)
        model_stats['rmse_total'] = rmse_total

    # sort the model runs by the error value
    sorted_model_stats = sorted(
        model_stats_dicts, key=lambda x: x['rmse_total'], reverse=False)

    filename = "total_rmse_results.json"
    with open(filename, 'w') as f:
        json.dump(sorted_model_stats, f)

    print('FINAL OUTPUT')
    print(sorted_model_stats[0])
