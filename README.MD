Project: Comparison of different private data generators
==========================================================

Description
-----------
This project aims to compare the any copula-based synthetic data generation
algorithm with the DPCopula-Kendall algorithm developed by Li, Xiong & Jiang.

This implementation of DPCopula is written in Python 3.7.

Additionally, there exists various analysis tools which allow
the comparison of the two algorithms.

DPCopula Algorithm Details
--------------------------
Input: Database and attribute information; Total privacy budget (**ε**);
Internal privacy budget split ratio (**k**).

Output: Synthetic database of identical size.

The algorithm splits its privacy budget into two operations:

1. Generating differentially private marginal histograms of the original
database.
2. Generating a differentially private correlation matrix.

Which are given privacy budgets **ε1** and **ε2** respectively
where **ε1** + **ε2** = **ε**.

The parameter **k** is the ratio of **ε1** and **ε2**.

Initially the EFPA method of privatising histograms was used in the DPCopula
implementation, however this was changed to the Laplace mechanism to allow
easier comparison with other mechanisms. EFPA is still implemented however remains
unused in the code. To change this, change `laplace_mechanism` to `EFPA` in the `get_dp_marginals`
function inside `DPCopula/synthetic.py`

Analysis Tools
--------------
`analyse.py` is a general purpose tool to generate and analyse Kendall
synthetic data. It is capable of producing single or multiple sets of synthetic
data with various epsilon and k (epsilon1 / epsilon2) values and organising
them into 'batches' to be further analysed. There are currently two forms of
analysis available:

1. Comparison of median and average errors against epsilon
2. Generation of a cumulative distribution graph of absolute error

All output of this script will go into the `data/test_output/` folder.

Run `python analyse.py --help` for more info.

`experiment_scripts/collect_graphs.sh` will collect all graphs generated by `analyse.py` into one
folder as hard links, namely `data/test_output/graphs/`.

Experiments
-----------
There also exists specific analyses to compare against the Kendall synthetic data
which provide more useful results. There are three main experiments:

1. Comparison of one-way histogram errors (range of epsilons)
2. Comparison of two-way histogram errors (range of epsilons)
3. Comparison of two-way histogram errors with error bars (ε = 1.0 only)

These experiments are contained in `experiments.py` and since they are rather
technical, helpful scripts to run these experiments can be found in the
`experiment_scripts/` folder.

Each experiment will have its own name and will be contained in its own folder
inside `data/experiments/`. All files used by an experiment will be
automatically copied to its folder and all files generated by an experiment
will be located in its folder.

Running the Experiments
-----------------------
These experiments utilise parallel processing to save on time. Before each you
run each command, ensure the previous command has finished completion.

Experiments 1 and 2 can be customised to use new sets of synthetic data and use
different attribute orderings, however experiment 3 has no customisation
available without changing the source code.

Synthetic sets refer to a collection of synthetic data found in
`data/experiment_data/synthetic_sets/`, which should be generated by the private
synthetic data generator you would like to compare against.
This folder will contain subfolders
named `set_[id]`. To specify a set to use in an experiment, use the id of the
folder (i.e. the folder name without 'set\_').

The same is true for attribute sets except for the fact that these exist in
`data/experiment_data/attribute_sets/`.

**Experiment 1**

```
experiment_scripts/exp_1_gen_hist [experiment name] [synthetic set] [attribute set]
experiment_scripts/exp_1_2_gen_single_output [experiment name] [synthetic set] [attribute set]
```

**Experiment 2**

```
experiment_scripts/exp_2_gen_hist [experiment name] [synthetic set] [attribute set]
experiment_scripts/exp_1_2_gen_single_output [experiment name] [synthetic set] [attribute set]
```

**Experiment 3**

```
experiment_scripts/exp_3_setup
experiment_scripts/exp_3_gen_hist
experiment_scripts/exp_3_gen_output
```

**Miscellaneous**

You can run experiments one and two X times using the
`experiment_scripts/exp_1_2_gen_all_hist` script and once these had completed,
You can run `experiment_scripts/exp_1_2_gen_all_output` which generates the output
for each respective run of each experiment.

These experiments each generate an error summary containing average and maximum
errors for certain quantiles of the error distribution. The
`collect_results.py` script takes all X runs of an experiment and averages
the error summary statistics.

Since all experiments can run with different attribute orderings, the
`change_order.py` script will take the default order and randomise it, creating
a new attribute set as described above.

The `experiments.py`, `collect_results.py` and `change_order.py` scripts all have
more detailed help messages which can be accessed in the `help_files/`
directory or by running the script with a `help` as the only argument.

Experiments 1 and 2 use six preset values of epsilon while experiment 3 only
uses one. Before running any experiments, ensure the correct set of
epsilon values is active in the code (look for the `EPSILONS` variable near the
top of the file).
