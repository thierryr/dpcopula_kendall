"""
privatise.py

Functions that transform histograms to be differentially private.
Enhanced Fourier Perturbation Algorithm (EFPA) technique is from
http://planete.inrialpes.fr/~ccastel/PAPERS/AcsCC12icdm.pdf
"""

import numpy as np
from math import sqrt
from DPCopula.exponential_mechanism import PrivItem, run_exp_mechanism
import cmath


def EFPA(histogram, epsilon):
    dft_coeffs = np.fft.rfft(histogram)

    # first coefficient is not used
    error_coeffs = [2 * abs(coeff) ** 2 for coeff in dft_coeffs[1:]]

    m = len(dft_coeffs)

    # last coeff has twice the error only if the number of coeffs is odd
    # otherwise it only counts for one times the error.
    if len(histogram) % 2 == 0:
        error_coeffs[-1] /= 2

    priv_items = []
    for k in range(m - 1):
        kept_coeffs = 2 * k + 1
        perturbation_error = sqrt(2) * (kept_coeffs) / (epsilon / 2)
        total_error = sqrt(sum(error_coeffs[k:])) + perturbation_error
        priv_items.append(PrivItem(-total_error, [k, kept_coeffs]))
        # print(perturbation_error, sqrt(sum(error_coeffs[k:])), total_error)

    # For final term, keep all fourier coefficients
    # Perturbation error includes len(histogram) terms
    # Reconstruction error is zero
    k = m - 1
    kept_coeffs = len(histogram)
    priv_items.append(PrivItem(-sqrt(2) * kept_coeffs / (epsilon / 2),
                               [k, kept_coeffs]))
    # print(sqrt(2) * kept_coeffs / (epsilon / 2), 0, sqrt(2) *
    # kept_coeffs / (epsilon / 2))

    picked_item = run_exp_mechanism(priv_items, epsilon / 2)
    lambda_ = sqrt(picked_item.id[1]) / (epsilon / 2)
    picked_k = picked_item.id[0] + 1

    # for item in priv_items:
    #     print(item.q, item.id, item.error)
    # print('Picked item:')
    # print(picked_item.q, picked_item.id, picked_item.error)

    for j in range(m):
        if j < picked_k:
            (magnitude, angle) = cmath.polar(dft_coeffs[j])
            dft_coeffs[j] = cmath.rect(magnitude + np.random.laplace(lambda_),
                                       angle)
        else:
            dft_coeffs[j] = 0
    # print(dft_coeffs)

    return [x.real for x in np.fft.irfft(dft_coeffs, len(histogram))]


def laplace_mechanism(histogram, epsilon):
    lambda_ = 2 / epsilon  # Sensitivity of histogram query is 2
    noisy_histogram = [count + np.random.laplace(0, lambda_)
                       for count in histogram]

    return noisy_histogram


if __name__ == '__main__':
    hist = np.random.randint(0, 10000, 50)
    noisy_hist = EFPA(hist, 0.01)

    print(f'{"Actual Count":>15}{"Noisy Count":>15}{"Absolute Error":>15}'
          f'{"Relative Error":>15}')
    print(f'{"------------":>15}' * 4)
    for h, nh in zip(hist, noisy_hist):
        rounded_nh = round(nh, 2)
        abs_error = round(abs(h - rounded_nh), 2)
        rel_error = round(abs_error / h * 100, 2)
        print(f'{h:>15}{rounded_nh:>15}{abs_error:>15}{f"{rel_error} %":>15}')
