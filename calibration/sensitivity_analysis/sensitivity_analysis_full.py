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

    # Run for 2000 ticks and return the necessary state variables

    counts = netlogo.repeat_report(
        ['total-mean-speed', 'average-headings', 'total-distances-walked', 'total-durations-walked', 'initial-dance-percentage', 'deviation-dance-percentage', 'free-path-dance-percentage', 'obstacle-dance-percentage'], 2000)

    print('Done')

    # speed preparation
    trimmed = counts['total-mean-speed'].values[counts['total-mean-speed'].values != 0]

    # headings preparation
    last_state_headings = counts['average-headings'].iloc[-1]

    # distance-time preparation
    last_state_dist = counts['total-distances-walked'].iloc[-1]
    last_state_time = counts['total-durations-walked'].iloc[-1]

    result = {
        'chisquare': chisquare(last_state_headings),
        'mean_speeds': trimmed.mean(),
        'std_speeds': np.std(trimmed),
        'mean_dist': last_state_dist.mean(),
        'std_dist': np.std(last_state_dist),
        'mean_time': last_state_time.mean(),
        'std_time': np.std(last_state_time),
        'last_state_initial': counts['initial-dance-percentage'].iloc[-1],
        'last_state_deviation': counts['deviation-dance-percentage'].iloc[-1],
        'last_state_obstacle': counts['obstacle-dance-percentage'].iloc[-1],
        'last_state_free': counts['free-path-dance-percentage'].iloc[-1],
    }

    print("sending", save_value, result)
    return save_value, result


if __name__ == '__main__':
    modelfile = os.path.abspath(
        r'F:\Dokumente\Uni_Msc\Thesis\repo\scarabs-abm\abm\code\scarabs_abm.nlogo')

    netlogo = pyNetLogo.NetLogoLink(gui=False)

    bounds = [0.5, 2]  # "1" is run separately as default

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    speed_related = ['protonum-width-impact',
                     'ball-roughness-impact', 'patch-roughness-impact']
    headings_related = ['spatial-awareness-impact', 'ball-roughness-impact',
                        'patch-roughness-impact', 'seen-radius-impact']
    distance_time_related = ['distance-threshold-impact', 'last-seen-threshold-impact',
                             'seen-radius-impact', 'heading-memory-impact', 'patch-roughness-impact']
    dancing_related = ['spatial-awareness-impact', 'initial-dancing-probability-impact',
                       'deviation-dancing-probability-impact', 'obstacle-dancing-probability-impact', 'free-dancing-probability-impact']

    problem = {
        'names': ['protonum-width-impact', 'ball-roughness-impact', 'patch-roughness-impact', 'spatial-awareness-impact', 'seen-radius-impact', 'distance-threshold-impact', 'last-seen-threshold-impact', 'heading-memory-impact', 'initial-dancing-probability-impact', 'deviation-dancing-probability-impact', 'obstacle-dancing-probability-impact', 'free-dancing-probability-impact'],
        'bounds': bounds,
    }

    print('ready to run the default one')

    # do the default run first
    netlogo.load_model(modelfile)
    input_value, results = run_simulation(None, True)
    problem['speeds'] = np.repeat([[0, results['mean_speeds'], 0]], 12, axis=0)
    problem['stds'] = np.repeat([[0, results['std_speeds'], 0]], 12, axis=0)
    problem['chisq'] = np.repeat([[0, results['chisquare'][0], 0]], 12, axis=0)
    problem['p'] = np.repeat([[0, results['chisquare'][1], 0]], 12, axis=0)
    problem['initial_dancing'] = np.repeat(
        [[0, results['last_state_initial'], 0]], 12, axis=0)
    problem['deviation_dancing'] = np.repeat(
        [[0, results['last_state_deviation'], 0]], 12, axis=0)
    problem['obstacle_dancing'] = np.repeat(
        [[0, results['last_state_obstacle'], 0]], 12, axis=0)
    problem['free_dancing'] = np.repeat(
        [[0, results['last_state_free'], 0]], 12, axis=0)
    problem['mean_dist'] = np.repeat(
        [[0, results['mean_dist'], 0]], 12, axis=0)
    problem['std_dist'] = np.repeat([[0, results['std_dist'], 0]], 12, axis=0)
    problem['mean_time'] = np.repeat(
        [[0, results['mean_time'], 0]], 12, axis=0)
    problem['std_time'] = np.repeat([[0, results['std_time'], 0]], 12, axis=0)

    print('ran the default one')
    print(problem)

    for i in range(len(problem['names'])):
        print('going through names', i)
        with Pool(4, initializer=initializer, initargs=(modelfile,)) as executor:
            experiments = pd.DataFrame(problem['bounds'],
                                       columns=[problem['names'][i]])

            for input_value, results in executor.map(run_simulation, experiments.to_dict('records')):
                print("receiving", input_value, results)
                problem['speeds'][i][int(input_value)] = results['mean_speeds']
                problem['stds'][i][int(input_value)] = results['std_speeds']
                problem['chisq'][i][int(input_value)] = results['chisquare'][0]
                problem['p'][i][int(input_value)] = results['chisquare'][1]
                problem['initial_dancing'][i][int(
                    input_value)] = results['last_state_initial']
                problem['deviation_dancing'][i][int(
                    input_value)] = results['last_state_deviation']
                problem['obstacle_dancing'][i][int(
                    input_value)] = results['last_state_obstacle']
                problem['free_dancing'][i][int(
                    input_value)] = results['last_state_free']
                problem['mean_dist'][i][int(
                    input_value)] = results['mean_dist']
                problem['std_dist'][i][int(input_value)] = results['std_dist']
                problem['mean_time'][i][int(
                    input_value)] = results['mean_time']
                problem['std_time'][i][int(input_value)] = results['std_time']

    print('finished running!')

    # Now all the plotting

    # speed: 'total-mean-speed'
    # headings: 'average-headings'
    # distance-time: 'total-distances-walked', 'total-durations-walked'
    # dancing: 'initial-dance-percentage', 'deviation-dance-percentage', 'free-path-dance-percentage', 'obstacle-dance-percentage'

    real_bounds = [0.5, 1, 2]

    '''The speed group'''
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_xlabel('Multiplication factor')
    ax1.set_ylabel('Speed')
    for i in range(len(speed_related)):
        name_index = problem['names'].index(speed_related[i])
        ax1.errorbar(real_bounds, problem['speeds'][name_index],
                     yerr=problem['stds'][name_index], color=colors[i], label=speed_related[i])
    ax1.legend()
    plt.show()

    '''The distance-time group'''
    fig2, (ax2, ax3) = plt.subplots(2)

    ax2.set_xlabel('Multiplication factor')
    ax2.set_ylabel('Walked distance')
    for i in range(len(distance_time_related)):
        name_index = problem['names'].index(distance_time_related[i])
        ax2.errorbar(real_bounds, problem['mean_dist'][name_index],
                     yerr=problem['std_dist'][name_index], color=colors[i], label=distance_time_related[i])
    ax2.legend()

    ax3.set_xlabel('Multiplication factor')
    ax3.set_ylabel('Walked time')
    for i in range(len(distance_time_related)):
        name_index = problem['names'].index(distance_time_related[i])
        ax3.errorbar(real_bounds, problem['mean_time'][name_index],
                     yerr=problem['std_time'][name_index], color=colors[i], label=distance_time_related[i])
    ax3.legend()
    plt.show()

    '''The headings group'''
    fig3, (ax4, ax5) = plt.subplots(2)

    ax4.set_xlabel('Multiplication factor')
    ax4.set_ylabel('Chisq')
    for i in range(len(headings_related)):
        name_index = problem['names'].index(headings_related[i])
        ax4.plot(real_bounds, problem['chisq'][name_index],
                 color=colors[i], label=headings_related[i])
    ax4.legend()

    ax5.set_xlabel('Multiplication factor')
    ax5.set_ylabel('P')
    for i in range(len(headings_related)):
        name_index = problem['names'].index(headings_related[i])
        ax5.plot(real_bounds, problem['p'][name_index],
                 color=colors[i], label=headings_related[i])
    ax5.legend()

    plt.show()

    ''' And dancing related '''
    fig4, ((ax6, ax7), (ax8, ax9)) = plt.subplots(2, 2)

    ax6.set_xlabel('Multiplication factor')
    ax6.set_ylabel('Initial dancings percentage')

    for i in range(len(dancing_related)):
        name_index = problem['names'].index(dancing_related[i])
        ax6.plot(real_bounds, problem['initial_dancing']
                 [name_index], color=colors[i], label=dancing_related[i])
    ax6.legend()

    ax7.set_xlabel('Multiplication factor')
    ax7.set_ylabel('Deviation dancings percentage')

    for i in range(len(dancing_related)):
        name_index = problem['names'].index(dancing_related[i])
        ax7.errorbar(real_bounds, problem['deviation_dancing']
                     [name_index], color=colors[i], label=dancing_related[i])
    ax7.legend()

    ax8.set_xlabel('Multiplication factor')
    ax8.set_ylabel('Obstacle dancings percentage')

    for i in range(len(dancing_related)):
        name_index = problem['names'].index(dancing_related[i])
        ax8.errorbar(real_bounds, problem['obstacle_dancing']
                     [name_index], color=colors[i], label=dancing_related[i])
    ax8.legend()

    ax9.set_xlabel('Multiplication factor')
    ax9.set_ylabel('Free path dancings percentage')

    for i in range(len(dancing_related)):
        name_index = problem['names'].index(dancing_related[i])
        ax9.errorbar(real_bounds, problem['free_dancing']
                     [name_index], color=colors[i], label=dancing_related[i])
    ax9.legend()

    plt.show()
