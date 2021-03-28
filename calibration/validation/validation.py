import numpy as np
import os
import argparse
import json
import math
import statistics
from scipy.stats import chisquare

# python validation.py -input_trajectories "F:\Dokumente\Uni_Msc\Thesis\repo\scarabs-abm\calibration\trajectories\validation" -model_stats "F:\Dokumente\Uni_Msc\Thesis\repo\scarabs-abm\calibration\results_default.json" -calibration_trajectories "F:\Dokumente\Uni_Msc\Thesis\repo\scarabs-abm\calibration\trajectories\training"


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
    bins = np.arange(0, 361, 30)
    histogram = np.histogram(heading_deviations, bins=bins)
    print('histogram', histogram[0])

    # return real_total_length, time_length, average_speed
    return speeds, real_total_length, time_length, histogram[0]


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-input_trajectories", "--input_trajectories", required=True,
                    help="path to input trajectories")
    ap.add_argument("-model_stats", "--model_stats", required=True,
                    help="path to the model output")
    ap.add_argument("-calibration_trajectories", "--calibration_trajectories", required=True,
                    help="path to the other trajectories to estimate means")

    args = vars(ap.parse_args())
    print('args', args)

    # first reading the model statistics
    with open(args["model_stats"]) as json_file:
        print('reading file', args["model_stats"])
        model_stats = json.load(json_file)
        print('original headings list', type(
            model_stats['heading_deviations']), model_stats['heading_deviations'])
        model_heading_deviations = np.array(model_stats['heading_deviations'])
        print('original headings', model_heading_deviations)
        model_stats['heading_deviations_norm'] = (
            model_heading_deviations / np.sum(model_heading_deviations))*100
        print('normed deviations', model_stats['heading_deviations_norm'])
        model_stats['chisq_new'] = chisquare(
            model_stats['heading_deviations_norm'])[0]
        model_stats['p_new'] = chisquare(
            model_stats['heading_deviations_norm'])[1]

        traj_stats = None
        folder_items = os.listdir(args["input_trajectories"])
        trajectories = [fi for fi in folder_items if fi.endswith(".json")]
        # print(trajectories)

        other_folder_items = os.listdir(args["calibration_trajectories"])
        calibration_trajectories = [
            fi for fi in other_folder_items if fi.endswith(".json")]
        # print(calibration_trajectories)

        # all_trajectories = trajectories.append(calibration_trajectories)
        # print('total trajectories amount', len(all_trajectories))

        i = 0

        all_values = {
            'speeds': [],
            # 'speed_means': [],
            'speed_stds': [],
            'distances': [],
            # 'distances_means': [],
            'distances_stds': [],
            'durations': [],
            # 'durations_means': [],
            'durations_stds': []
        }

        while i < len(calibration_trajectories):
            with open(args["calibration_trajectories"] + "/" + calibration_trajectories[i]) as json_file:
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

        traj_full_stats = {
            'amount': len(trajectories),
            'speeds': [],
            'speed_means': [],
            'speed_stds': [],
            'distances': [],
            'durations': [],
            'headings': [],
            'norm_headings': []
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

                traj_full_stats['headings'].append(
                    heading_deviations)
                traj_full_stats['norm_headings'].append(
                    (heading_deviations / np.sum(heading_deviations))*100)
                traj_full_stats['speeds'].append(speeds)
                traj_full_stats['distances'].append(real_total_length)
                traj_full_stats['durations'].append(time_length)

                traj_full_stats['speed_means'].append(np.mean(speeds))
                traj_full_stats['speed_stds'].append(np.std(speeds))

                speeds_temp = all_values['speeds']
                np.concatenate([np.array(speeds_temp), np.array(speeds)])

                all_values['speeds'] = speeds_temp
                all_values['speed_stds'].append(np.std(speeds))
                all_values['distances'].append(real_total_length)
                all_values['durations'].append(time_length)

            i += 1

            average_hist = np.mean(
                traj_full_stats['headings'], axis=0)

            print('just average hist', average_hist)

            average_his_norm = np.mean(
                traj_full_stats['norm_headings'], axis=0)

            print('just average_his_norm', average_his_norm)

            traj_stats = {
                'mean_speeds': np.mean(traj_full_stats['speed_means']),
                'std_speeds': np.mean(traj_full_stats['speed_stds']),
                'mean_dist': np.mean(traj_full_stats['distances']),
                'std_dist': np.std(traj_full_stats['distances']),
                'mean_time': np.mean(traj_full_stats['durations']),
                'std_time': np.std(traj_full_stats['durations']),
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

        print('here come the model stats', model_stats)
        print('and here come the real stats', traj_stats)
        # print('trajectory stats for normalization are', all_values)

        # validating each result separately
        rmse_mean_speeds_norm = rmse(np.array(model_stats['mean_speeds']/all_values['mean_speeds']), np.array(
            traj_stats['mean_speeds']/all_values['mean_speeds']))
        print("rms error for normalized mean_speed is: " +
              str(rmse_mean_speeds_norm))

        rmse_mean_speeds = rmse(np.array(model_stats['mean_speeds']), np.array(
            traj_stats['mean_speeds']))
        print("rms error for real valies mean_speed is: " + str(rmse_mean_speeds))

        # Normalizing stds by the mean value
        rmse_std_speeds_norm = rmse(np.array(model_stats['std_speeds']/all_values['mean_speeds']), np.array(
            traj_stats['std_speeds']/all_values['mean_speeds']))
        print("rms error for normalized std_speeds is: " +
              str(rmse_std_speeds_norm))

        rmse_std_speeds = rmse(np.array(model_stats['std_speeds']), np.array(
            traj_stats['std_speeds']))
        print("rms error for real values std_speeds is: " + str(rmse_std_speeds))

        rmse_mean_dist_norm = rmse(np.array(model_stats['mean_dist']/all_values['mean_dist']), np.array(
            traj_stats['mean_dist']/all_values['mean_dist']))
        print("rms error for normalized mean_dist is: " + str(rmse_mean_dist_norm))

        rmse_mean_dist = rmse(np.array(model_stats['mean_dist']), np.array(
            traj_stats['mean_dist']))
        print("rms error for real values mean_dist is: " + str(rmse_mean_dist))

        rmse_std_dist_norm = rmse(
            np.array(model_stats['std_dist']/all_values['mean_dist']), np.array(traj_stats['std_dist']/all_values['mean_dist']))
        print("rms error for normalized std_dist is: " + str(rmse_std_dist_norm))

        rmse_std_dist = rmse(
            np.array(model_stats['std_dist']), np.array(traj_stats['std_dist']))
        print("rms error for real values std_dist is: " + str(rmse_std_dist))

        rmse_mean_time_norm = rmse(np.array(model_stats['mean_time']/all_values['mean_time']), np.array(
            traj_stats['mean_time']/all_values['mean_time']))
        print("rms error for normalized mean_time is: " + str(rmse_mean_time_norm))

        rmse_mean_time = rmse(np.array(model_stats['mean_time']), np.array(
            traj_stats['mean_time']))
        print("rms error for real values mean_time is: " + str(rmse_mean_time))

        rmse_std_time_norm = rmse(
            np.array(model_stats['std_time']/all_values['mean_time']), np.array(traj_stats['std_time']/all_values['mean_time']))
        print("rms error for normalized std_time is: " + str(rmse_std_time_norm))

        rmse_std_time = rmse(
            np.array(model_stats['std_time']), np.array(traj_stats['std_time']))
        print("rms error for real values std_time is: " + str(rmse_std_time))

        rmse_chisq_norm = rmse(
            np.array(model_stats['chisq_new']/traj_stats['chisq']), np.array(traj_stats['chisq']/traj_stats['chisq']))
        print("rms error for normalized chisqis: " + str(rmse_chisq_norm))

        rmse_chisq=rmse(
            np.array(model_stats['chisq_new']), np.array(traj_stats['chisq']))
        print("rms error for real values chisqis: " + str(rmse_chisq))

        rmse_p=rmse(np.array(model_stats['p_new']), np.array(traj_stats['p']))
        print("rms error for p is: " + str(rmse_p))
