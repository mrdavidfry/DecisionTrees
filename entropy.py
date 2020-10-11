import collections
from math import log2


def entropy(values):
    histogram = collections.Counter(values)
    total = float(len(values))
    probabilities = {label: float(n)/total for label, n in histogram.items()}
    return sum(-p*log2(p)for _, p in probabilities.items())


def IG(full, subsets):
    N = len(full)
    H_subsets = sum(
        float(len(subset))/float(N)*entropy(subset) for subset in subsets
    )
    return entropy(full) - H_subsets
