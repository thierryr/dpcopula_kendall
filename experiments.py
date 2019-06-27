"""
experiments.py

Runs experiments on generated synthetic data.

Experiments are split up into steps which can be run separately to save
time.
"""

from DPCopula.Database import Database
from DPCopula.synthetic import kendall_algorithm

from help_files.help import generate_help_function

import numpy as np
import matplotlib.pyplot as plt

import shutil
import sys
import os


# Constants for the experiments
# Quantiles to use for error summary
QUANTILES = [0.5, 0.75, 0.9, 0.95, 0.99, 1.0]

# Maximum value for the x-axis on graphs with the corresponding epsilon
CDF_RANGES = [10000, 3000, 1500, 1250, 750, 500]

# Default values of epsilon to test
# Ensure the correct line is uncommented

# EPSILONS = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0]  # Uncomment for experiments 1, 2
EPSILONS = [1.0]  # Uncomment for experiment 3

GRIDLINE_FMT = {
    'b': True,
    'which': 'both',
    'color': '#CCCCCC',
    'linewidth': 0.5
}

ERRORBAR_FMT = {
    'ls': 'none',
    'capsize': 2,
    'capthick': 1,
    'marker': '.',
    'markersize': 1.5,
    'elinewidth': 1
}


print_help = generate_help_function('experiments')


class Experiment:
    def __init__(self, folder_name):
        self.folder_name = folder_name
        self.exp_path = f'data/experiments/{folder_name}'
        self.original_data_src = 'data/input/adult_data.csv'

    def setup_files(self):
        os.makedirs(self.exp_path, exist_ok=True)
        shutil.copy(self.original_data_src,
                    f'{self.exp_path}/original_data.csv')


class GeneralHistogramExperiment(Experiment):
    """
    Contains the steps to run one-way and two-way histogram experiments.

    Each experiment consists of the following steps:
        1. Create experiment directory and copy input files.
        2. Generate Kendall synthetic data.
        3. Generate the relevant histogram.
        4. Create the error summaries and graphs.

    The input files required are:
        - Original database
        - Synthetic database for each algorithm for each epsilon
        - Attribute order files

    Both one-way and two-way histograms can be generated. This will
    create a file containing all the histogram counts which is used to
    generate the graphs and error summaries.

    The graphs generated are a CDF of the error distribution and a
    boxplot of errors for each value of epsilon.

    The error summaries contain the average error and maximum error for
    each quantile specified in `QUANTILES` for each algorithm.

    To run an experiment from scratch, perform the steps in the above
    order. The experiments are split into these steps as generating the
    histograms and the graphs take a considerable amount of time and not
    all steps need to be run every time if a small change is made to an
    experiment.
    """

    def __init__(self, folder_name, set_id, attr_id, epsilons=EPSILONS,
                 cdf_ranges=CDF_RANGES):
        """
        Set up source files for experiment.

        Parameters:
            folder_name = name of the experiment folder
            set_id      Id of the synthetic data set containing all
                        required files.
            attr_id     Id of the attribute order set.
            epsilons    Values of epsilon to compare. Requires
                        corresponding data files.
            cdf_ranges  Maximum error to plot on cdf graph for
                        corresponding epsilon.

        The folder data/experiments/[folder_name] will be created if it
        doesn't already exist and all files used and created by the
        experiment will be placed in this folder.

        Synthetic data sets are in data/experiment_data/synthetic_sets
        and attribute order sets are found in
        data/experiment_data/attribute_sets.

        Attribute orders are created by the change_order script and
        my_own_algo synthetic data must be generated and place in the correct
        folder manually.
        """

        super().__init__(folder_name)

        self.set_id = set_id
        self.attr_id = attr_id

        self.data_dir = 'data/experiment_data/synthetic_sets/' \
                        f'set_{self.set_id}'
        self.attr_dir = 'data/experiment_data/attribute_sets/' \
                        f'set_{self.attr_id}'
        self.epsilons = epsilons
        self.cdf_ranges = cdf_ranges

    def setup_files(self):
        """
        Copies all data files required for the experiment into
        the experiment folder.

        Generates new Kendall data but reuses my_own_algo data as it takes
        much longer to generate.

        Copies files for all values of epsilon.
        """

        super().setup_files()

        for file in os.listdir(self.data_dir):
            shutil.copy(f'{self.data_dir}/{file}', f'{self.exp_path}/{file}')

        for file in os.listdir(self.attr_dir):
            shutil.copy(f'{self.attr_dir}/{file}', f'{self.exp_path}/{file}')

        original_db = Database()
        original_db.load_from_file(self.original_data_src,
                                   f'{self.exp_path}/adult_columns.csv')
        for epsilon in self.epsilons:
            eps1 = 8 / 9 * epsilon
            eps2 = 1 / 9 * epsilon
            kendall_data = kendall_algorithm(original_db, eps1, eps2)
            kendall_db = Database(kendall_data, original_db.attr_table)
            kendall_db.save_to_file(f'{self.exp_path}/'
                                    f'kendall_synthetic_{epsilon}.csv')

    def generate_one_way_histogram(self, database_file):
        """
        Generates a one-way histogram for a specific data file.

        Parameters:
            database_file = exact filename of the data file without the
                            extension
        This assumes the database file has either of the following formats:
            [algorithm]_synthetic_[epsilon].csv
            original_data.csv

        The file used will have the following format:
        data/experiments/[folder_name]/[database_file].csv

        The histogram will be stored in
        [algorithm]_histogram_[epsilon].csv if the database is synthetic
        or original_histogram.csv if it is the original database. Each
        line has the format '[attribute value],[count]'.

        If the database is my_own_algo synthetic, the columns will be
        reordered as to generate the same histogram order as the
        original and Kendall synthetic data produces.
        """

        database_type = database_file.split('_')[0]

        col_file = f'{self.exp_path}/adult_columns.csv'
        with open(col_file, 'r') as file:
            attr_list = [attr.strip() for attr in file.readlines()]

        if database_type == 'original':
            db = Database()
            db.load_from_file(f'{self.exp_path}/{database_file}.csv',
                              f'{self.exp_path}/adult_columns.csv')

            one_way_hist = db.generate_one_way_hist()

            with open(f'{self.exp_path}/original_histogram.csv', 'w+') as file:
                for attr in attr_list:
                    file.write(f'{attr},{one_way_hist[attr]}\n')

        elif database_type == 'kendall':
            epsilon = database_file.split('_')[2]
            db = Database()
            db.load_from_file(f'{self.exp_path}/{database_file}.csv',
                              f'{self.exp_path}/adult_columns.csv')

            one_way_hist = db.generate_one_way_hist()

            with open(f'{self.exp_path}/kendall_histogram_{epsilon}.csv',
                      'w+') as file:
                for attr in attr_list:
                    file.write(f'{attr},{one_way_hist[attr]}\n')

        elif database_type == 'my_own_algo':
            epsilon = database_file.split('_')[2]

            reordered_data = self.reorder_my_own_algo_data(database_file)

            with open(f'{self.exp_path}/my_own_algo_histogram_{epsilon}.csv',
                      'w+') as file:
                for index, attr in enumerate(attr_list):
                    file.write(f'{attr},{sum(reordered_data[:, index])}\n')

    def generate_two_way_histogram(self, database_file):
        """
        Generates a two-way histogram for a single synthetic database
        specified by the name of the algorithm used and the value of
        epsilon.

        This operates identically to the function generating the one-way
        histogram except the output file will have the format
        '[attribute value 1],[attribute value 2],[count]'.
        """

        database_type = database_file.split('_')[0]

        if 'original' == database_type:
            db = Database()
            db.load_from_file(f'{self.exp_path}/{database_file}.csv',
                              f'{self.exp_path}/adult_columns.csv')

            two_way_hist = db.generate_two_way_hist()

            with open(f'{self.exp_path}/original_histogram.csv',
                      'w+') as file:
                for pair, count in two_way_hist:
                    key1 = Database.avp_to_key(pair[0])
                    key2 = Database.avp_to_key(pair[1])

                    file.write(f'{key1},{key2},{count}\n')

        elif 'kendall' == database_type:
            epsilon = database_file.split('_')[2]

            db = Database()
            db.load_from_file(f'{self.exp_path}/{database_file}.csv',
                              f'{self.exp_path}/adult_columns.csv')

            two_way_hist = db.generate_two_way_hist()

            with open(f'{self.exp_path}/kendall_histogram_{epsilon}.csv',
                      'w+') as file:
                for pair, count in two_way_hist:
                    key1 = Database.avp_to_key(pair[0])
                    key2 = Database.avp_to_key(pair[1])

                    file.write(f'{key1},{key2},{count}\n')

        elif 'my_own_algo' == database_type:
            epsilon = database_file.split('_')[2]

            reordered_data = self.reorder_my_own_algo_data(database_file)

            col_file = f'{self.exp_path}/adult_columns.csv'
            attr_table = Database.get_attrs_from_file(col_file)

            attr_val_pairs = [(attr, val) for attr in attr_table
                              for val in attr_table[attr]]

            with open(col_file, 'r') as file:
                attr_list = [attr.strip() for attr in file.readlines()]

            with open(f'{self.exp_path}/my_own_algo_histogram_{epsilon}.csv',
                      'w+') as file:
                for avp1 in attr_val_pairs:
                    for avp2 in attr_val_pairs:
                        if avp1[0] >= avp2[0]:
                            continue

                        key1 = Database.avp_to_key(avp1)
                        key2 = Database.avp_to_key(avp2)

                        col1 = attr_list.index(key1)
                        col2 = attr_list.index(key2)

                        count = sum(reordered_data[:, col1] &
                                    reordered_data[:, col2])

                        file.write(f'{key1},{key2},{count}\n')

    def generate_results(self):
        """
        Generate the output of the experiment which includes the CDF of
        errors, boxplot of errors and error summary for each epsilon.

        Uses pre-generated histogram files to generate the output.

        Saves all output in the experiment folder. All CDFs are combined
        into a single image and the same is done with all boxplots.
        Each graph shows the errors of both the Kendall and my_own_algo
        algorithm for the corresponding epsilon.

        A separate summary file is created for each epsilon containing
        the average and maximum error for each quantile specified in
        `QUANTILES`.
        """

        original_hist = f'{self.exp_path}/original_histogram.csv'
        kendall_hists = [f'{self.exp_path}/{f}' for f in
                         os.listdir(self.exp_path) if 'kendall_histogram' in f]
        my_own_algo_hists = [f'{self.exp_path}/{f}' for f in
                       os.listdir(self.exp_path) if 'my_own_algo_histogram' in f]

        # Sort histogram files by epsilon value
        def get_eps(f):
            return float(os.path.splitext(f)[0].split('_')[-1])
        kendall_hists.sort(key=get_eps)
        my_own_algo_hists.sort(key=get_eps)

        fig, axes = plt.subplots(2, 3, sharey=True, figsize=(18, 10),
                                 dpi=400)
        box_fig, box_axes = plt.subplots(2, 3, figsize=(9, 9), dpi=400)
        axes = [ax for l in axes for ax in l]  # Flatten list of axes
        box_axes = [ax for l in box_axes for ax in l]

        graph_data = zip(axes, box_axes, self.cdf_ranges, kendall_hists,
                         my_own_algo_hists)
        for axis, box_axis, cdf_range, k, c in graph_data:
            epsilon = round(get_eps(k), 6)

            with open(original_hist) as o_file, open(k) as k_file, \
                    open(c) as c_file:
                o_counts = np.genfromtxt(o_file, delimiter=',', dtype=int,
                                         usecols=-1)
                k_counts = np.genfromtxt(k_file, delimiter=',', dtype=int,
                                         usecols=-1)
                c_counts = np.genfromtxt(c_file, delimiter=',', dtype=int,
                                         usecols=-1)

                k_errors = abs(o_counts - k_counts)
                c_errors = abs(o_counts - c_counts)

            box_axis.set_yscale('log')
            box_axis.boxplot((k_errors, c_errors),
                             labels=('DPCopula', 'my_own_algo'), sym='+')
            box_axis.set_title(f'Boxplot of Errors ε={epsilon}')
            box_axis.set_ylabel('Absolute Error')

            x = np.linspace(0, cdf_range, 1000, endpoint=True)
            y1 = [sum(k_errors <= t) / len(k_errors) for t in x]
            y2 = [sum(c_errors <= t) / len(c_errors) for t in x]

            with open(f'{self.exp_path}/error_summary_{epsilon}.csv',
                      'w+') as file:
                for limit in QUANTILES:
                    highest = np.quantile(k_errors, limit)
                    average = np.mean(k_errors[k_errors <= highest])

                    file.write(f'{limit * 100}%,{average},{highest},'
                               'kendall\n')

                for limit in QUANTILES:
                    highest = np.quantile(c_errors, limit)
                    average = np.mean(c_errors[c_errors <= highest])

                    file.write(f'{limit * 100}%,{average},{highest},'
                               'my_own_algo\n')

            if ((epsilon == 0.5) | (epsilon == 1) | (epsilon == 5)):
                print("TR A", epsilon, args[1])
                np.savetxt(f'{args[1]}_xaxis_epsilon_{epsilon}.txt', x, delimiter=",")
                np.savetxt(f'{args[1]}_yaxis_kendall_epsilon_{epsilon}.txt', y1, delimiter=",")
                np.savetxt(f'{args[1]}_yaxis_my_own_algo_epsilon_{epsilon}.txt', y2, delimiter=",")
            axis.plot(x, y1, label='kendall', linewidth=0.75)
            axis.plot(x, y2, label='my_own_algo', linewidth=0.75)
            axis.set_title(f'ε = {epsilon}')
            axis.set_xlabel('Absolute Error')
            axis.set_ylabel('Cumulative Distribution')
            axis.yaxis.set_tick_params(which='both', labelleft=True)
            axis.legend()


        fig.tight_layout(rect=[0, 0.3, 1, 0.95])
        fig.savefig(f'{self.exp_path}/cdf_graph.png')

        box_fig.tight_layout(rect=[0, 0.3, 1, 0.95])
        box_fig.savefig(f'{self.exp_path}/boxplot.png')

    def reorder_my_own_algo_data(self, database_file):
        my_own_algo_data = np.genfromtxt(f'{self.exp_path}/{database_file}.csv',
                                   delimiter=',', dtype=int)
        reordered_data = np.zeros_like(my_own_algo_data)
        order_data = np.genfromtxt(f'{self.exp_path}/column_order.csv',
                                   delimiter=',', dtype=int)

        for old_i, new_i in enumerate(order_data):
            reordered_data[:, new_i] = my_own_algo_data[:, old_i]

        return reordered_data


def generate_cdf_with_error_bars(folder_name, sets=10, epsilon='1.0'):
    """
    Experiment 3

    Averages the results of 10 two-way histogram experiments each with a
    randomised order, graphs the CDF with error bars.
    """

    MIN_ERROR = 0
    MAX_ERROR = 1000
    STEPS = 400

    INDEX_STEP = 40

    exp_path = f'data/experiments/{folder_name}'

    # Count number of two-way histogram entries
    with open(f'{exp_path}/set_1/original_histogram.csv', 'r') as file:
        for line_num, line in enumerate(file, 1):
            pass

        combinations = line_num

    # Each array column represents a single set of errors
    kendall_errors = np.zeros((combinations, 10))
    my_own_algo_errors = np.zeros((combinations, 10))

    for s in range(sets):
        o_hist_file = f'{exp_path}/set_{s+1}/original_histogram.csv'
        k_hist_file = f'{exp_path}/set_{s+1}/kendall_histogram_{epsilon}.csv'
        c_hist_file = f'{exp_path}/set_{s+1}/my_own_algo_histogram_{epsilon}.csv'
        o_counts = np.genfromtxt(o_hist_file, delimiter=',', dtype=int,
                                 usecols=-1)
        k_counts = np.genfromtxt(k_hist_file, delimiter=',', dtype=int,
                                 usecols=-1)
        c_counts = np.genfromtxt(c_hist_file, delimiter=',', dtype=int,
                                 usecols=-1)

        kendall_errors[:, s] = abs(o_counts - k_counts)
        my_own_algo_errors[:, s] = abs(o_counts - c_counts)

    x = np.linspace(MIN_ERROR, MAX_ERROR, STEPS, endpoint=True)
    k_cumulative = np.zeros((len(x), sets))
    c_cumulative = np.zeros((len(x), sets))

    # Calculate CDF points for each column
    for i, col in enumerate(kendall_errors.T):
        print(f'calculating kendall cdf {i}')
        k_cumulative[:, i] = [sum(col <= t) / len(col) for t in x]

    for i, col in enumerate(my_own_algo_errors.T):
        print(f'calculating my_own_algo cdf {i}')
        c_cumulative[:, i] = [sum(col <= t) / len(col) for t in x]

    # Calculate average CDF by averaging the rows
    y1 = np.mean(k_cumulative, 1)
    y2 = np.mean(c_cumulative, 1)

    fig, ax = plt.subplots(dpi=400)
    ax.grid(**GRIDLINE_FMT)
    ax.set_title('CDF of Two-Way Histogram Count Errors, ε = {epsilon}')
    ax.set_xlabel('Absolute Error')
    ax.set_ylabel('Cumulative Distribution')

    ax.plot(x, y1, linewidth=1, label='DPCopula', color='DeepSkyBlue')
    ax.plot(x, y2, linewidth=1, label='my_own_algo', color='DarkOrange')

    # Calculate error bars
    # Need marker coordinates as well as upper and lower errors for each
    # bar. INDEX_STEP determines how often an error bar appears (1 error
    # bar every INDEX_STEP number of points)
    k_error_x = []
    k_error_y = []
    k_error_lower = []
    k_error_upper = []

    i = INDEX_STEP
    while i < len(x):
        k_error_x.append(x[i])
        k_error_y.append(y1[i])
        k_error_lower.append(y1[i] - np.min(k_cumulative[i]))
        k_error_upper.append(np.max(k_cumulative[i]) - y1[i])
        i += INDEX_STEP

    ax.errorbar(k_error_x, k_error_y, yerr=[k_error_lower, k_error_upper],
                color='DodgerBlue', **ERRORBAR_FMT)

    c_error_x = []
    c_error_y = []
    c_error_lower = []
    c_error_upper = []

    i = INDEX_STEP
    while i < len(x):
        c_error_x.append(x[i])
        c_error_y.append(y2[i])
        c_error_lower.append(y2[i] - np.min(c_cumulative[i]))
        c_error_upper.append(np.max(c_cumulative[i]) - y2[i])
        i += INDEX_STEP

    ax.errorbar(c_error_x, c_error_y, yerr=[c_error_lower, c_error_upper],
                color='OrangeRed', **ERRORBAR_FMT)

    ax.legend()

    fig.savefig(f'{exp_path}/cdf.png')


if __name__ == '__main__':
    # Allow general experiment functionality from command line.

    args = sys.argv
    try:
        if args[2] == 'two_way_errors':
            generate_cdf_with_error_bars(args[1])
            quit()
        else:
            ex = GeneralHistogramExperiment(args[1], args[2], args[3])
            operation = args[4]
    except IndexError:
        print_help()
        quit()

    if args[1] == 'help':
        print_help()

    elif operation == 'setup':
        ex.setup_files()

    elif operation == 'one_way_hist':
        ex.generate_one_way_histogram(args[5])

    elif operation == 'two_way_hist':
        ex.generate_two_way_histogram(args[5])

    elif operation == 'gen_results':
        ex.generate_results()
