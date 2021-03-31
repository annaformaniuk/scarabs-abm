import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from multiprocessing import Pool
from scipy.stats import chisquare
import json
import itertools

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

    # Run for 1000 ticks and return the number of beetles and their mean speeds

    counts = netlogo.repeat_report(
        ['total-mean-speed', 'average-headings', 'total-distances-walked', 'total-durations-walked'], 2000)

    print('Done running model')

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
        'std_time': np.std(last_state_time)
    }

    print("sending", experiment, result)
    return experiment, result


if __name__ == '__main__':
    modelfile = os.path.abspath(
        r'F:\Dokumente\Uni_Msc\Thesis\repo\scarabs-abm\abm\code\scarabs_sensitivity_analysis.nlogo')

    netlogo = pyNetLogo.NetLogoLink(gui=False)

    bounds = np.arange(1.0, 2.6, 0.5)  # individual ?

    problem = {
        'names': ['protonum-width-impact', 'ball-roughness-impact', 'patch-roughness-impact', 'seen-radius-impact', 'distance-threshold-impact'],
        'bounds': bounds,
    }

    for protonum_value in bounds:
        for ball_roughness_value in bounds:
            for patch_roughness_value in bounds:
                for seen_radius_value in bounds:
                    # multiprocessing using the last one
                    with Pool(4, initializer=initializer, initargs=(modelfile,)) as executor:
                        experiments = pd.DataFrame(problem['bounds'],
                                                   columns=['distance-threshold-impact'])

                        exp_dict = experiments.to_dict('records')
                        # add values from other loops
                        for exp in exp_dict:
                            exp['protonum-width-impact'] = protonum_value
                            exp['ball-roughness-impact'] = ball_roughness_value
                            exp['patch-roughness-impact'] = patch_roughness_value
                            exp['seen-radius-impact'] = seen_radius_value

                        # this runs in parallel
                        for input_values, results in executor.map(run_simulation, exp_dict):
                            print("receiving", input_values, results)
                            result_json = {
                                'mean_speeds': results['mean_speeds'],
                                'std_speeds': results['std_speeds'],
                                'chisq': results['chisquare'][0],
                                'p': results['chisquare'][1],
                                'mean_dist': results['mean_dist'],
                                'std_dist': results['std_dist'],
                                'mean_time': results['mean_time'],
                                'std_time': results['std_time'],
                                'protonum-width-impact': input_values['protonum-width-impact'],
                                'patch-roughness-impact': input_values['patch-roughness-impact'],
                                'ball-roughness-impact': input_values['ball-roughness-impact'],
                                'distance-threshold-impact': input_values['distance-threshold-impact'],
                                'seen-radius-impact': input_values['seen-radius-impact']
                            }

                            filename = "results_" + str(input_values['protonum-width-impact']) + "_" + str(input_values['ball-roughness-impact']) + "_" + str(
                                input_values['patch-roughness-impact']) + "_" + str(input_values['seen-radius-impact']) + "_" + str(input_values['distance-threshold-impact']) + ".json"
                            with open(filename, 'w') as f:
                                json.dump(result_json, f)

    print('finished running!')
