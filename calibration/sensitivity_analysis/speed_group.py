import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import pyNetLogo


def run_simulation(experiment):
    '''run a netlogo model

    Parameters
    ----------
    experiments : dict

    '''

    print('Experiment', experiment)
    netlogo.command('setup')

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
    # Run for 1000 ticks and return the number of beetles and their mean speeds
    counts = netlogo.repeat_report(
        ['total-mean-speed'], 1000)

    print('Done')

    trimmed = counts['total-mean-speed'].values[counts['total-mean-speed'].values != 0]

    mean = trimmed.mean()
    std = np.std(trimmed)
    print(mean, std)
    return mean, std


if __name__ == '__main__':
    modelfile = os.path.abspath(
        r'F:\Dokumente\Uni_Msc\Thesis\repo\scarabs-abm\abm\code\scarabs_sensitivity_analysis.nlogo')

    netlogo = pyNetLogo.NetLogoLink(gui=False)

    bounds = np.arange(0, 2.0, 0.2)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    problem = {
        'names': ['protonum-width-impact', 'ball-roughness-impact', 'patch-roughness-impact'],
        'bounds': bounds,
        'speeds': [[], [], []],
        'stds': [[], [], []]
    }

    for i in range(len(problem['names'])):
        for value in problem['bounds']:
            netlogo.load_model(modelfile)
            experiment = {problem['names'][i]: value}
            mean, std = run_simulation(experiment)

            problem['speeds'][i].append(mean)
            problem['stds'][i].append(std)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Multiplication factor', fontsize=16)
    ax.set_ylabel('Speed', fontsize=16)

    # ax.set_ylim([0,20])
    for i in range(len(problem['names'])):
        ax.errorbar(problem['bounds'], problem['speeds'][i],
                    yerr=problem['stds'][i], color=colors[i], label=problem['names'][i])
    ax.legend()
    plt.show()
