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
    counts = netlogo.repeat_report(['initial-dance-percentage', 'deviation-dance-percentage',
                                    'free-path-dance-percentage', 'obstacle-dance-percentage'], 6000)

    print("Done")

    last_state_initial = counts['initial-dance-percentage'].iloc[-1]
    last_state_deviation = counts['deviation-dance-percentage'].iloc[-1]
    last_state_obstacle = counts['obstacle-dance-percentage'].iloc[-1]
    last_state_free = counts['free-path-dance-percentage'].iloc[-1]

    print(last_state_initial, last_state_deviation,
          last_state_obstacle, last_state_free)
    return last_state_initial, last_state_deviation, last_state_obstacle, last_state_free


if __name__ == '__main__':
    modelfile = os.path.abspath(
        r'F:\Dokumente\Uni_Msc\Thesis\repo\scarabs-abm\abm\code\scarabs_sensitivity_analysis.nlogo')

    netlogo = pyNetLogo.NetLogoLink(gui=False)

    bounds = np.arange(0.1, 2.0, 0.2)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    problem = {
        'names': ['spatial-awareness-impact', 'initial-dancing-probability-impact', 'deviation-dancing-probability-impact', 'obstacle-dancing-probability-impact', 'free-dancing-probability-impact'],
        'bounds': bounds,
        'initial_dancing': [[], [], [], [], []],
        'deviation_dancing': [[], [], [], [], []],
        'obstacle_dancing': [[], [], [], [], []],
        'free_dancing': [[], [], [], [], []]
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
            initial, deviation, obstacle, free = run_simulation(
                experiment)

            problem['initial_dancing'][i].append(initial)
            problem['deviation_dancing'][i].append(deviation)
            problem['obstacle_dancing'][i].append(obstacle)
            problem['free_dancing'][i].append(free)
                
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    ax1.set_xlabel('Multiplication factor')
    ax1.set_ylabel('Initial dancings percentage')

    for i in range(len(problem['names'])):
        ax1.plot(problem['bounds'], problem['initial_dancing'][i], color=colors[i], label=problem['names'][i])
    ax1.legend()

    ax2.set_xlabel('Multiplication factor')
    ax2.set_ylabel('Deviation dancings percentage')

    for i in range(len(problem['names'])):
        ax2.errorbar(problem['bounds'], problem['deviation_dancing'][i], color=colors[i], label=problem['names'][i])
    ax2.legend()

    ax3.set_xlabel('Multiplication factor')
    ax3.set_ylabel('Obstacle dancings percentage')

    for i in range(len(problem['names'])):
        ax3.errorbar(problem['bounds'], problem['obstacle_dancing'][i], color=colors[i], label=problem['names'][i])
    ax3.legend()

    ax4.set_xlabel('Multiplication factor')
    ax4.set_ylabel('Free path dancings percentage')

    for i in range(len(problem['names'])):
        ax4.errorbar(problem['bounds'], problem['free_dancing'][i], color=colors[i], label=problem['names'][i])
    ax4.legend()

    plt.show()
