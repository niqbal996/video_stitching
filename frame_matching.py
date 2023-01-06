import numpy as np


def matcher(seq1, seq2):
    from itertools import product
    min(product(arr1, arr2), key=lambda t: abs(t[0] - t[1]))[0]
    timestamp1 = seq1[2]
    timestamp2 = seq2[2]
