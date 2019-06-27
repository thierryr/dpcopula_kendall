"""
discrete_functions.py


Methods of generating discrete functions from a list of points.

Given a list of points (x_1, y_1), ..., (x_n, y_n) and initial value
y_0, a discrete function built from these points will be defined as
follows:

f(x) = y_0      if x < x_1
f(x) = y_k      if x_1 <= x < x_n where x_k <= x < x_(k+1)
f(x) = y_n      if x >= x_n

"""


def cdf_points(marginal_hist):
    """
    Generate the points for a CDF of a distribution from its marginal
    histogram.
    """
    cumulative = 0
    total = sum(x[1] for x in marginal_hist)
    points = []
    for (val, count) in marginal_hist:
        cumulative += count
        points.append((val, cumulative / (total + 1)))

    return points


def inverse_points(points):
    """
    Generate the points for the inverse CDF of a distribution.

    Takes the points for the CDF and transforms them into the points for
    the inverse.  Due to the discrete nature of the function, the x and
    y coordinates must be re-paired such that the inverse function is
    defined as above.
    """

    inverse_points = []
    next_y = 0
    for x, y in points:
        inverse_points.append((next_y, x))
        next_y = y

    return inverse_points


def function_from_points(points, y_0=0):
    def f(x):
        if x < points[0][0]:
            return y_0
        elif x >= points[-1][0]:
            return points[-1][1]

        for i in range(len(points)):
            if points[i][0] <= x < points[i + 1][0]:
                return points[i][1]

    return f


def marginal_cdf(marginal_hist):
    return function_from_points(cdf_points(marginal_hist))


def inverse_marginal_cdf(marginal_hist):
    points = inverse_points(cdf_points(marginal_hist))
    return function_from_points(points)
