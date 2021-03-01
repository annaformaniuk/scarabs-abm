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
    counts = netlogo.repeat_report(['total-distances-walked', 'total-durations-walked'], 6000)

    print("Done")

    last_state_dist = counts['total-distances-walked'].iloc[-1]
    last_state_time = counts['total-durations-walked'].iloc[-1]

    mean_dist = last_state_dist.mean()
    std_dist = np.std(last_state_dist)

    mean_time = last_state_time.mean()
    std_time = np.std(last_state_time)

    print(mean_dist, std_dist, mean_time, std_time)
    return mean_dist, std_dist, mean_time, std_time


if __name__ == '__main__':
    modelfile = os.path.abspath(
        r'F:\Dokumente\Uni_Msc\Thesis\repo\scarabs-abm\abm\code\scarabs_sensitivity_analysis.nlogo')

    netlogo = pyNetLogo.NetLogoLink(gui=False)

    bounds = np.arange(0.1, 2.0, 0.2)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    # add patch roughness!
    problem = {
        'names': ['distance-threshold-impact', 'last-seen-threshold-impact', 'seen-radius-impact', 'heading-memory-impact'],
        'bounds': bounds,
        'mean_dist': [[], [], [], []],
        'std_dist': [[], [], [], []],
        'mean_time': [[], [], [], []],
        'std_time': [[], [], [], []]
    }

    
        # with Pool(4, initializer=initializer, initargs=(modelfile,)) as executor:
        # results = []
        # for entry in executor.map(run_simulation, experiments.to_dict('records')):
        #     results.append(entry)
        # results = pd.DataFrame(results)


    for i in range(len(problem['names'])):
        for value in problem['bounds']:
            netlogo.load_model(modelfile)
            experiment = {problem['names'][i]: value}
            mean_dist, std_dist, mean_time, std_time = run_simulation(experiment)

            problem['mean_dist'][i].append(mean_dist)
            problem['std_dist'][i].append(std_dist)
            problem['mean_time'][i].append(mean_time)
            problem['std_time'][i].append(std_time)

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


