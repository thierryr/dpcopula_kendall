"""
synthetic.py

Methods of generating synthetic data.
"""

import numpy as np

# import scipy.interpolate
from scipy.special import comb
from scipy.stats import norm, kendalltau

from DPCopula.privatise import EFPA, laplace_mechanism
from DPCopula.discrete_functions import inverse_marginal_cdf

# import pdb

# import matplotlib.pyplot as plt


def ecdf(data):
    n = len(data)
    x = np.linspace(min(data), max(data), n)
    y = np.zeros(n)

    for i in np.arange(n):
        match_indexes = np.where(data <= x[i])
        y[i] = match_indexes[0].size / (n + 1)

    return x, y


def kendall_algorithm(database, epsilon1, epsilon2):
    # pdb.set_trace()
    m = database.num_attrs

    dp_marginals = get_dp_marginals(database, epsilon1)
    correlation_matrix = get_correlation_matrix(database, epsilon2)

    if not is_positive_definite(correlation_matrix):
        correlation_matrix = make_positive_definite(correlation_matrix)

    gaussian = np.random.multivariate_normal(np.zeros(m), correlation_matrix,
                                             database.num_rows)
    pseudo_data = norm.cdf(gaussian)
    numerical_synthetic_data = np.zeros_like(pseudo_data)

    for attr_id in range(m):
        marginal = dp_marginals[attr_id]
        # cdf = marginal_cdf(marginal)
        inverse_cdf = inverse_marginal_cdf(marginal)
        numerical_synthetic_data[:, attr_id] = [inverse_cdf(x) for x in
                                                pseudo_data[:, attr_id]]
        # cdf_points = get_cdf_points(marginal)
        # cdf = discrete_function_from_points(cdf_points)
        # inverse_cdf_points = get_inverse_points(cdf_points)
        # inverse_cdf = discrete_function_from_points(inverse_cdf_points)

        # print(cdf_points)
        # print(inverse_cdf_points)
        # vals = list(zip(*marginal))[0]
        # x1 = np.linspace(min(vals) - 1, max(vals) + 1, 1000)
        # x2 = np.linspace(0, 1, 1000)
        # y1 = [cdf(t) for t in x1]
        # y2 = [inverse_cdf(t) for t in x2]
        # plt.subplot(121)
        # plt.plot(x1, y1)
        # plt.subplot(122)
        # plt.plot(x2, y2)
        # plt.show()

    synthetic_data = []
    for row in numerical_synthetic_data:
        synthetic_data.append([database.get_attribute_value(i, x)
                               for i, x in enumerate(row)])

    return synthetic_data


def get_dp_marginals(database, epsilon):
    """
    Compute a differentially private marginal histogram for the
    database.
    """
    m = database.num_attrs
    marginals = []

    for attr_num in range(m):
        marginal = database.get_marginal_histogram(attr_num)
        values, counts = zip(*marginal)
        # dp_counts = EFPA(counts, epsilon / m)
        dp_counts = laplace_mechanism(counts, epsilon / m)
        # dp_counts = list(map(round, dp_counts))
        dp_counts = [round(count) if count >= 0 else 0 for count in dp_counts]
        dp_marginal = list(zip(values, dp_counts))
        marginals.append(dp_marginal)

    return marginals


def get_correlation_matrix(database, epsilon):
    """
    Compute differentially private Kendall tau correlation coefficients
    for each pair of attributes in the database.
    """

    m = database.num_attrs
    n = database.num_rows

    num_attr_pairs = comb(m, 2)
    # num_row_pairs = comb(n, 2)
    sensitivity = 4 / (n + 1)
    laplace_scale = num_attr_pairs * sensitivity / epsilon

    corr_matrix = np.identity(m)

    for i in range(m):
        for j in range(i + 1, m):
            i_marginal = database.numerical_data[:, i]
            j_marginal = database.numerical_data[:, j]

            correlation = kendalltau(i_marginal, j_marginal).correlation
            noise = np.random.laplace(0, laplace_scale)
            dp_correlation = correlation + noise
            dp_corr_est = np.sin(np.pi / 2 * dp_correlation)

            corr_matrix[i, j] = corr_matrix[j, i] = dp_corr_est

    return corr_matrix


def is_positive_definite(matrix):
    """
    Checks whether a matrix is positive definite.

    If a matrix has a Cholesky decomposition then it is positive
    definite.
    """

    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return False
    else:
        return True


def make_positive_definite(matrix):
    """
    Transforms a matrix to be positive definite and normalises it so
    that its diagonals are 1.
    """

    eigenvalues, R = np.linalg.eig(matrix)
    eigenvalues = [abs(ev) for ev in eigenvalues]
    D = np.diag(eigenvalues)

    R1 = R @ D @ R.T
    D1 = np.diag([1 / np.sqrt(r) for r in np.diag(R1)])
    normalised_pos_def = D1 @ R1 @ D1

    return normalised_pos_def


# def probability_integral_transform(data):
#     x, p = ecdf(data)
#     cdf = scipy.interpolate.interp1d(x, p)
#     return cdf(data)

# def get_pseudo_copula(database):
#     pseudo_copula_data = []
#     for col_num in range(database.num_attrs):
#         margin = database.numerical_data[:, col_num]
#         marginal_cdf = get_marginal_cdf(margin)
#         pseudo_copula_data.append(list(map(marginal_cdf, margin)))

#     pseudo_copula_data = np.array(pseudo_copula_data)
#     return pseudo_copula_data

# def mle_algorithm(database, epsilon):
#     epsilon1 = epsilon / 2
#     epsilon2 = epsilon - epsilon1
#     m = database.num_attrs

#     marginal_distributions = []

#     for attr_num in range(m):
#         # Get actual marginal histogram in form [(value, count), ...]
#         marginal = database.get_marginal_histogram(attr_num)

#         # Split up into values and the number of times that value occurs
#         values, counts = zip(*marginal)

#         # Add noise to the histogram and round the counts
#         dp_counts = EFPA(counts, epsilon1 / m)
#         dp_counts = list(map(round, dp_counts))

#         data = np.array([v for v, c in zip(values, dp_counts) for i in
#         range(int(c))])
#         U = probability_integral_transform(data)
#         pass

#         # Create a DP marginal histogram
#         # dp_marginal = list(zip(values, dp_counts))

#         # marginal_cdf = get_marginal_cdf(dp_marginal)
#         # pseudo_copula_data = list(map(marginal_cdf, values))
#         # marginal_distributions.append(list(zip(values,
#         pseudo_copula_data)))

#     marginal_distributions = np.array(marginal_distributions)
#     print(marginal_distributions)
