"""
exponential_mechanism.py

Originally from Gergely Acs, Claude Castelluccia and Rui Chen.

"""

import math
import random


class PrivItem:
    def __init__(self, q, id):
        self.id = id
        self.q = q
        self.error = None


def basic(items, f):
    # print(f'f = {f}')
    for item in items:
        item.error = f * item.q

    maximum = max(map(lambda x: x.error, items))

    for item in items:
        item.error = math.exp(item.error - maximum)

    uniform = sum(map(lambda x: x.error, items)) * random.random()
    # print(f'maximum = {maximum}, uniform = {uniform}')
    for item in items:
        # print(f'new uniform = {uniform}')
        uniform -= item.error
        if uniform <= 0:
            break

    return item


def run_exp_mechanism(items, eps):
    return basic(items, eps / 2)
