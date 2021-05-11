import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import Pool

import pyNetLogo


def initializer(modelfile):
    '''initialize a subprocess

    Parameters
    ----------
    modelfile : str

    '''

    # we need to set the instantiated netlogo
    # link as a global so run_simulation can
    # use it
    global netlogo

    netlogo = pyNetLogo.NetLogoLink(gui=False)
    netlogo.load_model(modelfile)


def run_simulation(experiment):
    '''run a netlogo model

    Parameters
    ----------
    experiments : dict

    '''

    print('Experiment', experiment)
    netlogo.command('setup')

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
    # Run for 6000 ticks
    counts = netlogo.repeat_report(
        ['total-distances-walked', 'total-durations-walked'], 6000)

    print("Done")

    last_state_dist = counts['total-distances-walked'].iloc[-1]
    last_state_time = counts['total-durations-walked'].iloc[-1]

    mean_dist = last_state_dist.mean()
    std_dist = np.std(last_state_dist)

    mean_time = last_state_time.mean()
    std_time = np.std(last_state_time)

    print(save_value, mean_dist, std_dist, mean_time, std_time)
    return save_value, mean_dist, std_dist, mean_time, std_time


if __name__ == '__main__':
    modelfile = os.path.abspath(
        r'F:\Dokumente\Uni_Msc\Thesis\repo\scarabs-abm\abm\code\scarabs_abm.nlogo')

    netlogo = pyNetLogo.NetLogoLink(gui=False)

    bounds = np.arange(0.1, 2.0, 0.2)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    # add patch roughness!
    problem = {
        'names': ['distance-threshold-impact', 'last-seen-threshold-impact', 'seen-radius-impact', 'heading-memory-impact', 'patch-roughness-impact'],
        'bounds': bounds,
        'mean_dist': [[], [], [], [], []],
        'std_dist': [[], [], [], [], []],
        'mean_time': [[], [], [], [], []],
        'std_time': [[], [], [], [], []]
    }

    for i in range(len(problem['names'])):
        with Pool(4, initializer=initializer, initargs=(modelfile,)) as executor:
            experiments = pd.DataFrame(problem['bounds'],
                                       columns=[problem['names'][i]])
            # placeholders
            result_mean_dist = np.empty_like(problem['bounds'])
            result_std_dist = np.empty_like(problem['bounds'])
            result_mean_time = np.empty_like(problem['bounds'])
            result_std_time = np.empty_like(problem['bounds'])
            for input_value, mean_dist, std_dist, mean_time, std_time in executor.map(run_simulation, experiments.to_dict('records')):
                print("receiving", input_value, mean_dist,
                      std_dist, mean_time, std_time)
                current_index = np.where(
                    problem['bounds'] == input_value)[0][0]
                print(current_index)
                result_mean_dist[current_index] = mean_dist
                result_std_dist[current_index] = std_dist
                result_mean_time[current_index] = mean_time
                result_std_time[current_index] = std_time

            problem['mean_dist'][i] = result_mean_dist
            problem['std_dist'][i] = result_std_dist
            problem['mean_time'][i] = result_mean_time
            problem['std_time'][i] = result_std_time

    fig, (ax1, ax2) = plt.subplots(2)

    ax1.set_xlabel('Multiplication factor', fontsize=16)
    ax1.set_ylabel('Walked distance', fontsize=16)

    # ax.set_ylim([0,20])
    for i in range(len(problem['names'])):
        ax1.errorbar(problem['bounds'], problem['mean_dist'][i],
                     yerr=problem['std_dist'][i], color=colors[i], label=problem['names'][i])
    ax1.legend()

    ax2.set_xlabel('Multiplication factor', fontsize=16)
    ax2.set_ylabel('Walked time', fontsize=16)

    # ax.set_ylim([0,20])
    for i in range(len(problem['names'])):
        ax2.errorbar(problem['bounds'], problem['mean_time'][i],
                     yerr=problem['std_time'][i], color=colors[i], label=problem['names'][i])
    ax2.legend()

    plt.show()
