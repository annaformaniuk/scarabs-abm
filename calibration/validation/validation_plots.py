import numpy as np
import os
import argparse
import json
import math
import statistics
from scipy.stats import chisquare
import matplotlib.pyplot as plt
from validation import validate_stats

# python validation_plots.py -pre_calibration_stats "F:\Dokumente\Uni_Msc\Thesis\repo\scarabs-abm\calibration\validation\chosen_model_stats\results_default.json" -post_calibration_stats "F:\Dokumente\Uni_Msc\Thesis\repo\scarabs-abm\calibration\validation\chosen_model_stats\results_calibrated.json" -validation_trajectories "F:\Dokumente\Uni_Msc\Thesis\repo\scarabs-abm\calibration\trajectories\validation" -calibration_trajectories "F:\Dokumente\Uni_Msc\Thesis\repo\scarabs-abm\calibration\trajectories\training"


if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-validation_trajectories", "--validation_trajectories", required=True,
                    help="path to validation trajectories")
    ap.add_argument("-pre_calibration_stats", "--pre_calibration_stats", required=True,
                    help="path to the model output")
    ap.add_argument("-post_calibration_stats", "--post_calibration_stats", required=True,
                    help="path to the model output")
    ap.add_argument("-calibration_trajectories", "--calibration_trajectories", required=True,
                    help="path to the other trajectories to estimate means")

    args = vars(ap.parse_args())
    print('args', args)

    default_args = {
        'input_trajectories': args['validation_trajectories'],
        'model_stats': args['pre_calibration_stats'],
        'calibration_trajectories': args['calibration_trajectories']
    }

    calibrated_args = {
        'input_trajectories': args['validation_trajectories'],
        'model_stats': args['post_calibration_stats'],
        'calibration_trajectories': args['calibration_trajectories']
    }

    default_results, default_model, validation_traj, traj_full_stats = validate_stats(default_args)
    # print('uncalibrated model results', default_model, validation_traj)

    calibrated_results, calibrated_model, validation_traj, traj_full_stats = validate_stats(calibrated_args)
    # print('calibrated model results', calibrated_model, validation_traj)
    
    print('got both', default_results, calibrated_results)

    default_values = []
    calibrated_values = []

    rmse_names = ['rmse_mean_speeds', 'rmse_std_speeds', 'rmse_mean_dist',
             'rmse_std_dist', 'rmse_mean_time', 'rmse_std_time']
    for name in rmse_names:
        default_values.append(round(default_results[name], 1))
        calibrated_values.append(round(calibrated_results[name], 1))

    fig, ax = plt.subplots()
    ax.bar(rmse_names, default_values, label="Pre-Calibration")
    ax.bar(rmse_names, calibrated_values, label="Post-Calibration")
    fig.suptitle('Root mean square error before and after calibration', fontsize=32)
    ax.legend()
    ax.yaxis.label.set_size(24)
    ax.yaxis.label.set_size(24)

    for i in range(len(default_values)):
        plt.annotate(str(default_values[i]), xy=(
            rmse_names[i], default_values[i]), ha='left', va='bottom', color='blue', fontsize=16)
        plt.annotate(str(calibrated_values[i]), xy=(
            rmse_names[i], calibrated_values[i]), ha='right', va='bottom', color='black', fontsize=16)

    plt.show()

    model_params = ['mean_speeds', 'mean_dist', 'mean_time']
    model_stds = ['std_speeds', 'std_dist', 'std_time']
    params_visualisation = ['Speed', 'Distance', 'Time']

    default_model_params = []
    default_model_stds = []
    calibrated_model_params = []
    calibrated_model_stds = []

    for name in model_params:
        default_model_params.append(round(default_model[name], 1))
        calibrated_model_params.append(round(calibrated_model[name], 1))

    for name in model_stds:
        default_model_stds.append(round(default_model[name], 1))
        calibrated_model_stds.append(round(calibrated_model[name], 1))

    fig2, axs2 = plt.subplots(1, 2, sharey=True)
    fig2.suptitle('Results of uncalibrated and calibrated model runs for speed')
    axs2[0].errorbar([params_visualisation[0]], [default_model_params[0]], [default_model_stds[0]], fmt='ok')
    axs2[1].errorbar([params_visualisation[0]], [calibrated_model_params[0]], [calibrated_model_stds[0]], fmt='ok')

    plt.show()

    fig3, axs3 = plt.subplots(1, 2, sharey=True)
    fig3.suptitle('Results of uncalibrated and calibrated model runs for distances')
    axs3[0].errorbar([params_visualisation[1]], [default_model_params[1]], [default_model_stds[1]], fmt='ok')
    axs3[1].errorbar([params_visualisation[1]], [calibrated_model_params[1]], [calibrated_model_stds[1]], fmt='ok')

    plt.show()

    fig4, axs4 = plt.subplots(1, 2, sharey=True)
    fig4.suptitle('Results of uncalibrated and calibrated model runs for times')
    axs4[0].errorbar([params_visualisation[2]], [default_model_params[2]], [default_model_stds[2]], fmt='ok')
    axs4[1].errorbar([params_visualisation[2]], [calibrated_model_params[2]], [calibrated_model_stds[2]], fmt='ok')

    plt.show()

    traj_stats_names = ['speeds_concatenated', 'distances', 'durations']
    traj_full_stats['speeds_concatenated'] = np.concatenate(traj_full_stats['speeds'])
    traj_stats_values = []
    for name in traj_stats_names:
        traj_stats_values.append(traj_full_stats[name])

    # traj_data = np.concatenate((spread, center))

    fig5, ax5 = plt.subplots(1, 3)
    ax5[0].boxplot(traj_stats_values[0], labels=[traj_stats_names[0]])
    ax5[1].boxplot(traj_stats_values[1], labels=[traj_stats_names[1]])
    ax5[2].boxplot(traj_stats_values[2], labels=[traj_stats_names[2]])
    fig5.suptitle('Real validation trajectories statistics')

    plt.show()
    

    