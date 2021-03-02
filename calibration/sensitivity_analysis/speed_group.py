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
    # Run for 1000 ticks and return the number of beetles and their mean speeds
    counts = netlogo.repeat_report(
        ['total-mean-speed'], 1000)

    print('Done')

    trimmed = counts['total-mean-speed'].values[counts['total-mean-speed'].values != 0]

    mean = trimmed.mean()
    std = np.std(trimmed)
    print("sending", save_value, mean, std)
    return save_value, mean, std


if __name__ == '__main__':
    modelfile = os.path.abspath(
        r'F:\Dokumente\Uni_Msc\Thesis\repo\scarabs-abm\abm\code\scarabs_sensitivity_analysis.nlogo')

    # netlogo = pyNetLogo.NetLogoLink(gui=False)

    bounds = np.arange(0, 2.0, 0.2)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    problem = {
        'names': ['protonum-width-impact', 'ball-roughness-impact', 'patch-roughness-impact'],
        'bounds': bounds,
        'speeds': [[], [], []],
        'stds': [[], [], []]
    }

    for i in range(len(problem['names'])):
    #     for value in problem['bounds']:
    #         netlogo.load_model(modelfile)
    #         experiment = {problem['names'][i]: value}
    #         mean, std = run_simulation(experiment)

    #         problem['speeds'][i].append(mean)
    #         problem['stds'][i].append(std)

        with Pool(4, initializer=initializer, initargs=(modelfile,)) as executor:
            experiments = pd.DataFrame(problem['bounds'],
                               columns=[problem['names'][i]])
            # placeholders
            result_speeds = np.empty_like(problem['bounds'])
            result_stds = np.empty_like(problem['bounds'])
            for input_value, speed, std in executor.map(run_simulation, experiments.to_dict('records')):
                print("receiving", input_value, speed, std)
                current_index = np.where(problem['bounds']==input_value)[0][0]
                print(current_index)
                result_speeds[current_index] = speed
                result_stds[current_index] = std

            problem['speeds'][i] = result_speeds
            problem['stds'][i] = result_stds

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
