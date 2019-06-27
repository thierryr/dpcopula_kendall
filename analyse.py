"""
analyse.py


Generates DP synthetic data and compares the one-way histogram of the
synthetic output with that of the original database.
"""

from datetime import datetime
from argparse import ArgumentParser, RawTextHelpFormatter
from os import makedirs, listdir, path
from shutil import rmtree, copy
from textwrap import dedent

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from DPCopula.Database import Database
from DPCopula.parameters import adult_params as params
from DPCopula.synthetic import kendall_algorithm


# Formatting constants
TIMESTAMP_FMT = '%Y%m%d%H%M%S'
GRIDLINE_FMT = {
    'b': True,
    'which': 'both',
    'color': '#CCCCCC',
    'linewidth': 0.4
}
X_FORMATTER = FuncFormatter(lambda x, pos: str(int(x) if x.is_integer()
                                               else round(x, 4)))
Y_FORMATTER = FuncFormatter(lambda x, pos: str(int(x)))

# Other constants
ANALYSES = ['single', 'batch', 'autobatch', 'cumulative', 'my_own_algo']


def get_batch_directory(batch_id):
    if batch_id is None:
        batch_dir = 'data/test_output'
    else:
        batch_dir = f'data/test_output/batch_{batch_id}'

    return batch_dir


def single_analysis(epsilon, k=8, batch_id=None, replace=False):
    """
    Generates a single set of synthetic data and a one-way histogram
    count of all attribute values.

    epsilon:    the total privacy parameter for the algorithm
    k:          the ratio of epsilon1 to epsilon2
    batch_id:   a str identifier to organise different analyses
    replace:    should replace analyses with same epsilon value

    Results are saved into 'DPCopula/data/test_output/' with each
    analysis with the same batch_id being saved in a batch_[batch_id]
    folder.
    """

    # Pick batch folder
    batch_dir = get_batch_directory(batch_id)
    makedirs(batch_dir, exist_ok=True)

    if replace:
        for data_dir in listdir(batch_dir):
            if 'kendall' not in data_dir:
                continue
            eps = float(data_dir.split('_')[2].replace('-', '.'))
            if round(eps, 6) == round(epsilon, 6):
                rmtree(f'{batch_dir}/{data_dir}')

    epsilon = round(epsilon, 6)  # Ensure no floating point errors
    epsilon1 = k / (k + 1) * epsilon
    epsilon2 = 1 / (k + 1) * epsilon

    epsilon_str = str(epsilon).replace('.', '-')
    timestamp = datetime.now().strftime(TIMESTAMP_FMT)
    output_dir = f'{batch_dir}/kendall_{timestamp}_{epsilon_str}'
    makedirs(output_dir, exist_ok=True)

    print(f'Running DPCopula-Kendall\tε = {epsilon}')

    # Load data and create DP synthetic data
    original_db = Database()
    original_db.load_from_file(params.input_data_file, params.attribute_file)

    synthetic_data = kendall_algorithm(original_db, epsilon1, epsilon2)
    synthetic_db = Database(synthetic_data, original_db.attr_table)

    synthetic_db.save_to_file(f'{output_dir}/synthetic_data.csv')

    # Calculate error statistics
    abs_errors, rel_errors = compare_databases(original_db, synthetic_db,
                                               output_dir)

    # Calculate summary statistics
    with open(f'{output_dir}/summary.txt', 'w+') as file:
        file.write('epsilon,k,avg absolute error,avg relative error,'
                   'median absolute error,median relative error\n')
        file.write(f'{epsilon},{k},{np.mean(abs_errors)},'
                   f'{np.mean(rel_errors)},{np.median(abs_errors)},'
                   f'{np.median(rel_errors)}\n')


def compare_databases(original_db, synthetic_db, output_dir):
    """
    Compare one way histograms of original and synthetic data,
    compute absolute and relative errors and summary statistics.
    """

    abs_errors = []
    rel_errors = []
    with open(f'{output_dir}/histogram_comparison.csv', 'w+') as file:
        file.write('Attribute,Original count,Synthetic count,'
                   'Absolute error,Relative Error\n')
        for attr in sorted(original_db.attr_table.keys()):
            for val in original_db.attr_table[attr]:
                key = f'{attr}_{val}'
                original = original_db.one_way_histogram[key]
                synthetic = synthetic_db.one_way_histogram[key]
                abs_error = abs(synthetic - original)
                rel_error = round(abs_error / original * 100, 2)
                file.write(f'{key},{original},{synthetic},{abs_error},'
                           f'{rel_error}%\n')

                abs_errors.append(abs_error)
                rel_errors.append(rel_error)

    return abs_errors, rel_errors


def batch_analysis(batch_id=None):
    """
    Takes all analyses in a batch and graphs error summary statistics
    against epsilon.

    Produces two log-log plots with one displaying the mean and median
    absolute errors and the other displaying the mean and median
    relative errors (%).

    batch_id: the identifier of single analyses of which to use in the
    batch analysis.

    If batch_id is None, use 'DPCopula/data/test_output/' as the batch
    folder.  The graph is saved in the batch folder as 'error_graph.png'
    """

    epsilons = []
    avg_abs_errors = []
    med_abs_errors = []
    avg_rel_errors = []
    med_rel_errors = []

    # Pick batch folder
    batch_dir = get_batch_directory(batch_id)

    # Collect data
    for data_dir in listdir(batch_dir):
        # Only use folders containing a single analysis
        if 'kendall' not in data_dir:
            continue

        path = f'{batch_dir}/{data_dir}'
        with open(f'{path}/summary.txt', 'r') as file:
            # Summary stats on 2nd line
            summary = [float(stat) for stat in file.readlines()[1].split(',')]

        epsilons.append(summary[0])
        avg_abs_errors.append(summary[2])
        med_abs_errors.append(summary[4])
        avg_rel_errors.append(summary[3])
        med_rel_errors.append(summary[5])

    # Setup plot - two graphs in the same figure
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, dpi=400, figsize=(12, 6))

    # First graph - absolute errors
    ax1.set_title('Absolute Error of Histogram Counts')
    ax1.grid(**GRIDLINE_FMT)
    ax1.loglog(epsilons, avg_abs_errors, '.', color='OrangeRed',
               label='Average Error')
    ax1.loglog(epsilons, med_abs_errors, '.', color='DarkOrange',
               label='Median Error')
    ax1.set_xlabel('ε')
    ax1.set_ylabel('Absolute Error')
    ax1.legend()
    ax1.xaxis.set_major_formatter(X_FORMATTER)
    ax1.yaxis.set_major_formatter(Y_FORMATTER)

    # Second graph - relative errors
    ax2.set_title('Relative Error of Histogram Counts')
    ax2.grid(**GRIDLINE_FMT)
    ax2.loglog(epsilons, avg_rel_errors, '.', color='DodgerBlue',
               label='Average Error')
    ax2.loglog(epsilons, med_rel_errors, '.', color='DeepSkyBlue',
               label='Median Error')
    ax2.set_xlabel('ε')
    ax2.set_ylabel('Relative Error (%)')
    ax2.legend()
    ax2.xaxis.set_major_formatter(X_FORMATTER)
    ax2.yaxis.set_major_formatter(Y_FORMATTER)
    ax2.yaxis.set_tick_params(which='both', labelleft=True)

    f.tight_layout(pad=2, w_pad=3)
    f.savefig(f'{batch_dir}/error_graph.png')


def autobatch_analysis(epsilons=None, k=8, batch_id=None, replace=False):
    """
    Perform multiple single analyses over a range of epsilon values.

    k:          the ratio of epsilon1 to epsilon2
    batch_id:   a str identifier to organise single analyses
    replace:    should replace previous autobatch analyis with same
                batch_id
    """

    if epsilons is None:
        epsilons = np.concatenate([np.arange(0.001, 0.01, 0.0005),
                                   np.arange(0.01, 0.02, 0.001),
                                   np.arange(0.02, 0.1, 0.005),
                                   np.arange(0.1, 1, 0.05),
                                   np.arange(1, 10, 0.5),
                                   [10]])

    for i, epsilon in enumerate(epsilons):
        print(f'{i + 1:02} of {len(epsilons):02}: ', end='')
        single_analysis(epsilon, k=k, batch_id=batch_id, replace=replace)

    batch_analysis(batch_id)


def cumulative_batch_analysis(batch_id=None):
    """
    Creates cumulative distribution graphs of absolute errors for each
    dataset in a batch.

    batch_id:   the identifier of single analyses of which to use in the
                batch analysis.

    Result is saved in the batch folder as 'error_distribution.png'.
    """

    batch_dir = get_batch_directory(batch_id)

    plt.figure(dpi=400)
    plt.title('Cumulative Distribution of Absolute Errors in Histogram Counts')
    plt.xlabel('Absolute Error')
    plt.ylabel('Cumulative Distribution')

    for data_dir in listdir(batch_dir):
        if 'kendall' in data_dir:
            algorithm = 'kendall'
        elif 'my_own_algo' in data_dir:
            algorithm = 'my_own_algo'
        else:
            continue

        path = f'{batch_dir}/{data_dir}'
        with open(f'{path}/histogram_comparison.csv', 'r') as file:
            abs_errors = np.array([int(line.split(',')[3])
                                   for line in file.readlines()[1:]])

        with open(f'{path}/summary.txt', 'r') as file:
            summary = [float(stat) for stat in file.readlines()[1].split(',')]
            epsilon = round(summary[0], 6)

        x = np.append(np.arange(0, max(abs_errors), 0.5), max(abs_errors))
        y = [sum(abs_errors <= t) / len(abs_errors) for t in x]

        plt.grid(**GRIDLINE_FMT)
        plt.plot(x, y, label=f'[{algorithm}] ε = {epsilon}', linewidth=0.75)

    ax = plt.gca()
    # ax.set_xlim(-100, 1100)
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = zip(*sorted(zip(handles, labels),
                                  key=lambda t: float(t[1].split(' ')[-1])))
    # ax.xaxis.set_major_formatter(X_FORMATTER)
    plt.legend(handles, labels, loc='lower right')
    plt.savefig(f'{batch_dir}/error_distribution.png')


def process_my_own_algo_db(db_loc, epsilon, batch_id=None):
    """
    Takes a synthetic database generated by the my_own_algo algorithm and
    generates error statistics to be used in further analyses.

    Copies the synthetic database into the data/test_output folder (in a
    batch if specified) and calculates errors compared to the original
    database. Also generates a summary file with the same format as
    kendall analyses, however since there is no parameter `k`, leave
    that field as 0.
    """

    with open(params.attribute_file, 'r') as file:
        attr_list = [attr.strip() for attr in file.readlines()]

    # Create relevant directories and copy file
    batch_dir = get_batch_directory(batch_id)
    timestamp = datetime.now().strftime(TIMESTAMP_FMT)
    epsilon_str = str(epsilon).replace('.', '-')
    data_dir = f'{batch_dir}/my_own_algo_{timestamp}_{epsilon_str}'
    makedirs(data_dir, exist_ok=True)

    db_file = f'{data_dir}/synthetic_data.csv'
    if not path.exists(db_file):
        copy(db_loc, db_file)

    # Load the synthetic data. This algorithm formats the database as a
    # list of binary attributes with values of 1 if the row has the
    # attribute and 0 otherwise.
    synthetic_data = np.genfromtxt(db_file, dtype=int, delimiter=',')

    original_db = Database()
    original_db.load_from_file(params.input_data_file, params.attribute_file)

    abs_errors = []
    rel_errors = []
    with open(f'{data_dir}/histogram_comparison.csv', 'w+') as file:
        file.write('Attribute,Original count,Synthetic count,'
                   'Absolute error,Relative Error\n')
        for index, attr in enumerate(attr_list):
            original = original_db.one_way_histogram[attr]
            synthetic = sum(synthetic_data[:, index])
            abs_error = abs(synthetic - original)
            rel_error = round(abs_error / original * 100, 2)
            file.write(f'{attr},{original},{synthetic},{abs_error},'
                       f'{rel_error}%\n')

            abs_errors.append(abs_error)
            rel_errors.append(rel_error)

    with open(f'{data_dir}/summary.txt', 'w+') as file:
        file.write('epsilon,k,avg absolute error,avg relative error,'
                   'median absolute error,median relative error\n')
        file.write(f'{epsilon},0,{np.mean(abs_errors)},'
                   f'{np.mean(rel_errors)},{np.median(abs_errors)},'
                   f'{np.median(rel_errors)}\n')


if __name__ == '__main__':
    start = datetime.now()

    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('analysis_type', metavar='type', choices=ANALYSES,
                        help=dedent("""\
            type of analysis to perform:
            single     - generates dp synthetic data with epsilon
                         value(s) specified by -e (default 1.0) and
                         calculates one-way histogram error statistics
            batch      - compares error statistics of all single
                         analyses in a batch
            autobatch  - performs multiple single analyses (specified
                         by -e or a default selection) then performs a
                         batch analysis on these
            cumulative - generates a cumulative distribution graph for
                         all single analyse in a batch
            my_own_algo      - takes a synthetic database generated by the
                         my_own_algo algorithm and generates error statistics
                         to be used in further analyses.
                         Requires a file path (-f) and epsilon value
                         (-e).
            """))
    parser.add_argument('-b', '--batch', dest='batch_id', metavar='id',
                        default=None, help=dedent("""\
            Optional batch identifier to organise single analyses.
            Output directory is `data/test_output/batch_[ID]` if
            specified, otherwise `data/test_output`.\n
            """))
    parser.add_argument('-e', '--eps', dest='epsilons', metavar='E',
                        type=float, nargs='+', help=dedent("""\
            Epsilon value(s) to use for single or autobatch analyses.
            In a single analysis default is 1.0 and in an autobatch
            analysis, default is a range between 0.001 and 10.0.
            Option is not used in batch or cumulative analyses.\n
            """))
    parser.add_argument('-k', '--ratio', dest='k', type=float, default=8.0,
                        help=dedent("""\
            Ratio of epsilon1 to epsilon2 (default 8).
            Option is not used in batch or cumulative analyses.\n
            """))
    parser.add_argument('-r', '--replace', dest='replace', action='store_true',
                        help=dedent("""\
            Should replace previous analyses of similar type.
            Replaces all previous analyses in the same batch with the
            same epsilon value.
            Option is not used in batch or cumulative analyses.\n
            """))
    parser.add_argument('-f', '--file', dest='db_loc', type=str, default=None,
                        help=dedent("""\
            When generating statistics from the my_own_algo algorithm,
            specifiy the location of the synthetic database file.\n
            """))

    args = parser.parse_args()

    # Run analysis
    if args.analysis_type == 'single':
        if args.epsilons is None:
            epsilons = [1.0]
        else:
            epsilons = args.epsilons

        for i, epsilon in enumerate(epsilons):
            print(f'{i + 1:02} of {len(epsilons):02}: ', end='')
            single_analysis(epsilon, k=args.k, batch_id=args.batch_id,
                            replace=args.replace)

    elif args.analysis_type == 'batch':
        batch_analysis(batch_id=args.batch_id)

    elif args.analysis_type == 'autobatch':
        autobatch_analysis(k=args.k, batch_id=args.batch_id,
                           replace=args.replace)

    elif args.analysis_type == 'cumulative':
        cumulative_batch_analysis(batch_id=args.batch_id)

    elif args.analysis_type == 'my_own_algo':
        process_my_own_algo_db(db_loc=args.db_loc, epsilon=args.epsilons[0],
                         batch_id=args.batch_id)

    end = datetime.now()
    delta = end - start
    mins = delta.seconds // 60
    secs = delta.seconds % 60

    if args.analysis_type != 'help':
        print(f'performed {args.analysis_type} analysis in {str(delta)}')
