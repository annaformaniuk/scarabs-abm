import os
import numpy as np
from scipy.stats import chisquare
import json

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

    # Run for 4000 ticks and return the number of beetles and their mean speeds

    counts = netlogo.repeat_report(
        ['total-mean-speed', 'average-headings', 'total-distances-walked', 'total-durations-walked', 'initial-dance-percentage', 'deviation-dance-percentage', 'free-path-dance-percentage', 'obstacle-dance-percentage'], 4000)

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
        'heading_deviations': last_state_headings
    }

    print("sending", save_value, result)
    return save_value, result


if __name__ == '__main__':
    modelfile = os.path.abspath(
        r'F:\Dokumente\Uni_Msc\Thesis\repo\scarabs-abm\abm\code\scarabs_sensitivity_analysis.nlogo')

    netlogo = pyNetLogo.NetLogoLink(gui=False)

    print('ready to run the default one')

    # do the default run first
    netlogo.load_model(modelfile)
    input_value, results = run_simulation(None, True)
    result_json = {
        'mean_speeds': results['mean_speeds'],
        'std_speeds': results['std_speeds'],
        'heading_deviations': list(results['heading_deviations']),
        'chisq': results['chisquare'][0],
        'p': results['chisquare'][1],
        'mean_dist': results['mean_dist'],
        'std_dist': results['std_dist'],
        'mean_time': results['std_time'],
        'std_time': results['std_time']
    }

    filename = "results_default.json"
    with open(filename, 'w') as f:
        json.dump(result_json, f)
