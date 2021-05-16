import os
import numpy as np
from scipy.stats import chisquare
import json
import matplotlib.pyplot as plt

import pyNetLogo


def run_simulation(experiment, default=False):
    '''run a netlogo model

    Parameters
    ----------
    experiments : dict

    '''

    print('Experiment', experiment)
    netlogo.command('setup')

    save_value = 1

    if (default == False):
        save_value = next(iter(experiment.values()))

        # Set the input parameters
        for key, value in experiment.items():
            if key == 'random-seed':
                # The NetLogo random seed requires a different syntax
                netlogo.command('random-seed {}'.format(value))
            else:
                print('setting', key, value)
                # Otherwise, assume the input parameters are global variables
                netlogo.command('set {0} {1}'.format(key, value))

        netlogo.command('setup')

    # Run for 3000 ticks

    counts = netlogo.repeat_report(
        ['total-mean-speed', 'average-headings', 'total-distances-walked', 'total-durations-walked', 'initial-dance-percentage', 'deviation-dance-percentage', 'free-path-dance-percentage', 'obstacle-dance-percentage'], 3000)

    print('Done')

    # speed preparation
    trimmed = counts['total-mean-speed'].values[counts['total-mean-speed'].values != 0]

    # headings preparation
    last_state_headings = counts['average-headings'].iloc[-1]

    headings_int = last_state_headings.astype(int)

    last_state_headings_norm = (headings_int / np.sum(headings_int))*100

    last_state_headings_norm_int = last_state_headings_norm.astype(int)

    # distance-time preparation
    last_state_dist = counts['total-distances-walked'].iloc[-1]
    last_state_time = counts['total-durations-walked'].iloc[-1]

    result = {
        'chisquare': chisquare(last_state_headings_norm_int),
        'mean_speeds': trimmed.mean(),
        'std_speeds': np.std(trimmed),
        'mean_dist': last_state_dist.mean(),
        'std_dist': np.std(last_state_dist),
        'mean_time': last_state_time.mean(),
        'std_time': np.std(last_state_time),
        'heading_deviations': headings_int.tolist(),
        'heading_deviations_norm': last_state_headings_norm_int.tolist()
    }

    print("sending", save_value, result)
    return save_value, result


if __name__ == '__main__':
    modelfile = os.path.abspath(
        r'F:\Dokumente\Uni_Msc\Thesis\repo\scarabs-abm\abm\code\scarabs_abm.nlogo')

    netlogo = pyNetLogo.NetLogoLink(gui=False)

    print('ready to run the calibrated default one')

    # do the default run first
    netlogo.load_model(modelfile)
    input_value, results = run_simulation(None, True)
    result_json = {
        'mean_speeds': results['mean_speeds'],
        'std_speeds': results['std_speeds'],
        'heading_deviations': results['heading_deviations'],
        'heading_deviations_norm': results['heading_deviations_norm'],
        'chisq': results['chisquare'][0],
        'p': results['chisquare'][1],
        'mean_dist': results['mean_dist'],
        'std_dist': results['std_dist'],
        'mean_time': results['mean_time'],
        'std_time': results['std_time']
    }

    print("got results calibrated", results)

    filename = "results_calibrated_default.json"
    with open(filename, 'w') as f:
        json.dump(result_json, f)

    print('ready to run the more beetles one')

    netlogo.load_model(modelfile)
    input_value, results_beetles = run_simulation(
        {'beetles-at-pile': 10}, False)
    results_beetles_json = {
        'mean_speeds': results_beetles['mean_speeds'],
        'std_speeds': results_beetles['std_speeds'],
        'heading_deviations': results_beetles['heading_deviations'],
        'heading_deviations_norm': results_beetles['heading_deviations_norm'],
        'chisq': results_beetles['chisquare'][0],
        'p': results_beetles['chisquare'][1],
        'mean_dist': results_beetles['mean_dist'],
        'std_dist': results_beetles['std_dist'],
        'mean_time': results_beetles['mean_time'],
        'std_time': results_beetles['std_time']
    }

    print("got results", results_beetles)

    filename = "results_calibrated_more_beetles.json"
    with open(filename, 'w') as f:
        json.dump(results_beetles_json, f)

    print('ready to run the more obstacles one')

    netlogo.load_model(modelfile)
    input_value, results_obstacles = run_simulation(
        {'additional-obstacles': True}, False)
    results_obstacles_json = {
        'mean_speeds': results_obstacles['mean_speeds'],
        'std_speeds': results_obstacles['std_speeds'],
        'heading_deviations': results_obstacles['heading_deviations'],
        'heading_deviations_norm': results_obstacles['heading_deviations_norm'],
        'chisq': results_obstacles['chisquare'][0],
        'p': results_obstacles['chisquare'][1],
        'mean_dist': results_obstacles['mean_dist'],
        'std_dist': results_obstacles['std_dist'],
        'mean_time': results_obstacles['mean_time'],
        'std_time': results_obstacles['std_time']
    }

    print("got results", results_obstacles)

    filename = "results_calibrated_more_obstacles.json"
    with open(filename, 'w') as f:
        json.dump(results_obstacles_json, f)

    # now all the plotting
    model_params = ['mean_speeds', 'mean_dist',
                    'mean_time', 'heading_deviations_norm']
    model_stds = ['std_speeds', 'std_dist', 'std_time']
    params_visualisation = ['Speed', 'Distance', 'Time']
    bins = np.arange(0, 361, 30)
    bins_strings = ['< ' + str(s) for s in bins]

    calibrated_model_params = []
    calibrated_model_stds = []
    more_beetles_model_params = []
    more_beetles_model_stds = []
    more_obstacles_model_params = []
    more_obstacles_model_stds = []

    for name in model_params:
        if (name != 'heading_deviations_norm'):
            calibrated_model_params.append(round(results[name], 1))
            more_beetles_model_params.append(round(results_beetles[name], 1))
            more_obstacles_model_params.append(
                round(results_obstacles[name], 1))
        else:
            print(results[name], results_beetles[name],
                  results_obstacles[name])
            calibrated_model_params.append(results[name])
            more_beetles_model_params.append(results_beetles[name])
            more_obstacles_model_params.append(results_obstacles[name])

    for name in model_stds:
        calibrated_model_stds.append(round(results[name], 1))
        more_beetles_model_stds.append(round(results_beetles[name], 1))
        more_obstacles_model_stds.append(round(results_obstacles[name], 1))

    fig2, axs2 = plt.subplots(1, 2, sharey=True)
    fig2.suptitle(
        'Results of default model and model with more obstacles for journey durations', fontsize=26)
    axs2[0].errorbar(['Duration (default model)'], [calibrated_model_params[2]], [
                     calibrated_model_stds[2]], fmt='ok')
    axs2[1].errorbar(['Duration (more obstacles)'], [more_obstacles_model_params[2]], [
                     more_obstacles_model_stds[2]], fmt='ok')
    axs2[0].set_ylabel('seconds', fontsize=16)
    axs2[1].set_ylabel('seconds', fontsize=16)

    plt.show()
    
    fig3, axs3 = plt.subplots(1, 2, sharey=True)
    fig3.suptitle(
        'Results of heading deviation histograms for default model and model with more beetles', fontsize=18)
    axs3[0].bar(bins_strings[1:], calibrated_model_params[3])
    axs3[0].set_xlabel('Heading deviations (default model)', fontsize=16)
    axs3[0].set_ylabel('Percentage', fontsize=16)
    axs3[1].bar(bins_strings[1:], more_beetles_model_params[3])
    axs3[1].set_xlabel('Heading deviations (more beetles)', fontsize=16)
    axs3[1].set_ylabel('Percentage', fontsize=16)

    plt.show()

    fig4, axs4 = plt.subplots(1, 2, sharey=True)
    fig4.suptitle(
        'Results of heading deviation histograms for default model and model with more obstacles', fontsize=18)
    axs4[0].bar(bins_strings[1:], calibrated_model_params[3])
    axs4[0].set_xlabel('Heading deviations (default model)', fontsize=16)
    axs4[0].set_ylabel('Percentage', fontsize=16)
    axs4[1].bar(bins_strings[1:], more_obstacles_model_params[3])
    axs4[1].set_xlabel('Heading deviations (more obstacles)', fontsize=16)
    axs4[1].set_ylabel('Percentage', fontsize=16)

    plt.show()

    fig0, axs0 = plt.subplots(1, 2, sharey=True)
    fig0.suptitle(
        'Results of default model and model with more beetles for speed', fontsize=26)
    axs0[0].errorbar(['Speed (default model)'], [calibrated_model_params[0]], [
                     calibrated_model_stds[0]], fmt='ok')
    axs0[1].errorbar(['Speed (more beetles)'], [more_beetles_model_params[0]], [
                     more_beetles_model_stds[0]], fmt='ok')
    axs0[0].set_ylabel('centimeters/second', fontsize=16)
    axs0[1].set_ylabel('centimeters/second', fontsize=16)

    plt.show()

    fig1, axs1 = plt.subplots(1, 2, sharey=True)
    fig1.suptitle(
        'Results of default model and model with more beetles for travelled distances', fontsize=26)
    axs1[0].errorbar(['Distance (default model)'], [calibrated_model_params[1]], [
                     calibrated_model_stds[1]], fmt='ok')
    axs1[1].errorbar(['Distance (more beetles)'], [more_beetles_model_params[1]], [
                     more_beetles_model_stds[1]], fmt='ok')
    axs1[0].set_ylabel('centimeters', fontsize=16)
    axs1[1].set_ylabel('centimeters', fontsize=16)

    plt.show()

    fig2, axs2 = plt.subplots(1, 2, sharey=True)
    fig2.suptitle(
        'Results of default model and model with more beetles for journey durations', fontsize=26)
    axs2[0].errorbar(['Duration (default model)'], [calibrated_model_params[2]], [
                     calibrated_model_stds[2]], fmt='ok')
    axs2[1].errorbar(['Duration (more beetles)'], [more_beetles_model_params[2]], [
                     more_beetles_model_stds[2]], fmt='ok')
    axs2[0].set_ylabel('seconds', fontsize=16)
    axs2[1].set_ylabel('seconds', fontsize=16)

    plt.show()

    fig0, axs0 = plt.subplots(1, 2, sharey=True)
    fig0.suptitle(
        'Results of default model and model with more obstacles for speed', fontsize=26)
    axs0[0].errorbar(['Speed (default model)'], [calibrated_model_params[0]], [
                     calibrated_model_stds[0]], fmt='ok')
    axs0[1].errorbar(['Speed (more beetles)'], [more_obstacles_model_params[0]], [
                     more_obstacles_model_stds[0]], fmt='ok')
    axs0[0].set_ylabel('centimeters/second', fontsize=16)
    axs0[1].set_ylabel('centimeters/second', fontsize=16)

    plt.show()

    fig1, axs1 = plt.subplots(1, 2, sharey=True)
    fig1.suptitle(
        'Results of default model and model with more obstacles for travelled distances', fontsize=26)
    axs1[0].errorbar(['Distance (default model)'], [calibrated_model_params[1]], [
                     calibrated_model_stds[1]], fmt='ok')
    axs1[1].errorbar(['Distance (more obstacles)'], [more_obstacles_model_params[1]], [
                     more_obstacles_model_stds[1]], fmt='ok')
    axs1[0].set_ylabel('centimeters', fontsize=16)
    axs1[1].set_ylabel('centimeters', fontsize=16)

    plt.show()

    
