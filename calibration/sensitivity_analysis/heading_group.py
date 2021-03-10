import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import Pool
from scipy.stats import chisquare

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
    # Run for 1000 ticks and return the number of beetles and their mean speeds
    counts = netlogo.repeat_report(
        ['average-headings'], 1000)

    print('Done')
    last_state_headings = counts['average-headings'].iloc[-1]
    print(last_state_headings)

    (chisq, p) = chisquare([16, 18, 16, 14, 12, 12])

    print("sending", save_value, chisq, p)
    return save_value, chisq, p


if __name__ == '__main__':
    modelfile = os.path.abspath(
        r'F:\Dokumente\Uni_Msc\Thesis\repo\scarabs-abm\abm\code\scarabs_sensitivity_analysis.nlogo')

    netlogo = pyNetLogo.NetLogoLink(gui=False)

    bounds = np.arange(0.5, 2.1, 0.5)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    problem = {
        'names': ['spatial-awareness-impact', 'ball-roughness-impact', 'patch-roughness-impact', 'seen-radius-impact'],
        'bounds': bounds,
        'chisq': [[], [], [], []],
        'p': [[], [], [], []]
    }

    for i in range(len(problem['names'])):
        # for value in problem['bounds']:
        #     netlogo.load_model(modelfile)
        #     experiment = {problem['names'][i]: value}
        #     saved_value, chisq, p = run_simulation(experiment)

        #     problem['chisq'][i].append(chisq)
        #     problem['p'][i].append(p)
        with Pool(4, initializer=initializer, initargs=(modelfile,)) as executor:
            experiments = pd.DataFrame(problem['bounds'],
                               columns=[problem['names'][i]])
            # placeholders
            result_chisq = np.empty_like(problem['bounds'])
            result_p = np.empty_like(problem['bounds'])
            for input_value, chisq, p in executor.map(run_simulation, experiments.to_dict('records')):
                print("receiving", input_value, chisq, p)
                current_index = np.where(problem['bounds']==input_value)[0][0]
                print(current_index)
                result_chisq[current_index] = chisq
                result_p[current_index] = p

            problem['chisq'][i] = result_chisq
            problem['p'][i] = result_p

    fig, (ax1, ax2) = plt.subplots(2)

    ax1.set_xlabel('Multiplication factor')
    ax1.set_ylabel('Chisq')

    # ax.set_ylim([0,20])
    for i in range(len(problem['names'])):
        ax1.plot(problem['bounds'], problem['chisq'][i], color=colors[i], label=problem['names'][i])
    ax1.legend()

    ax2.set_xlabel('Multiplication factor')
    ax2.set_ylabel('P')

    # ax.set_ylim([0,20])
    for i in range(len(problem['names'])):
        ax2.plot(problem['bounds'], problem['p'][i], color=colors[i], label=problem['names'][i])
    ax2.legend()

    plt.show()
