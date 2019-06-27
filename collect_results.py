"""
collect_results.py

Collects the results of experiments which have been run multiple times
and averages the data.
"""

from help_files.help import generate_help_function

import os
import sys
import numpy as np

EXPERIMENT_DIR = 'data/experiments'
PRESET_LIST = ['one_way_histograms', 'two_way_histograms',
               'one_way_reordered', 'two_way_reordered']

print_help = generate_help_function('collect_results')


def average_results(experiment_identifier):
    """
    Takes all runs of an experiment and averages the error summaries.
    Writes a new error summary to the 'data/experiments' folder.

    Error summaries all have the same format so can be loaded into numpy
    arrays and then averaged.
    """

    EPSILONS = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    QUANTILES = [0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    NUM_TESTS = 4

    # Separate array for each epsilon value
    statistics = {eps: np.zeros((12, 2)) for eps in EPSILONS}
    for path in os.listdir(EXPERIMENT_DIR):
        if experiment_identifier not in path or 'summary' in path:
            # Select only experiments matching the identifier and ignore
            # previously generated summaries
            continue

        for epsilon in EPSILONS:
            summary_file = f'{EXPERIMENT_DIR}/{path}/' \
                           f'error_summary_{epsilon}.csv'
            results = np.genfromtxt(summary_file, dtype=float, delimiter=',',
                                    usecols=(1, 2))
            statistics[epsilon] += results  # Sum all corresponding values

    with open(f'{EXPERIMENT_DIR}/{experiment_identifier}_summary.csv', 'w+') \
            as file:
        for eps, stats in statistics.items():
            stats = stats / NUM_TESTS  # Calulate average
            kendall_stats, my_own_algo_stats = np.split(stats, 2, axis=0)

            file.write(f'epsilon={eps},kendall avg,my_own_algo avg,kendall max,'
                       'my_own_algo max\n')
            for q, k, c in zip(QUANTILES, kendall_stats, my_own_algo_stats):
                file.write(f'{q},{k[0]},{c[0]},{k[1]},{c[1]}\n')


if __name__ == '__main__':
    try:
        identifier = sys.argv[1]
    except IndexError:
        print_help()
        quit()

    if identifier == 'help':
        print_help()

    elif identifier == 'preset':
        for ident in PRESET_LIST:
            average_results(ident)

    else:
        average_results(identifier)
