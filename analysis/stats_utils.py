import itertools

import numpy as np
from scipy import stats


def hodges_lehmann(differences):
    values = np.asarray(differences, dtype=float)
    walsh = np.array([(values[i] + values[j]) / 2.0
                      for i, j in itertools.combinations_with_replacement(range(len(values)), 2)])
    return float(np.median(walsh))


def hl_ci(differences, alpha=0.05):
    values = np.asarray(differences, dtype=float)
    n = len(values)
    if n < 6:
        return float("nan"), float("nan")
    walsh = np.sort(np.array([(values[i] + values[j]) / 2.0
                              for i, j
                              in itertools.combinations_with_replacement(range(n), 2)]))
    total = len(walsh)
    critical = _wilcoxon_critical(n, alpha)
    if critical is None:
        return float("nan"), float("nan")
    lower_index = critical
    upper_index = total - critical - 1
    if lower_index > upper_index:
        return float("nan"), float("nan")
    return float(walsh[lower_index]), float(walsh[upper_index])


def hl_ci_normal(differences, alpha=0.05):
    values = np.asarray(differences, dtype=float)
    n = len(values)
    lower, upper = np.triu_indices(n)
    walsh = np.sort((values[lower] + values[upper]) / 2.0)
    z = stats.norm.ppf(1 - alpha / 2.0)
    k = int(np.floor(n * (n + 1) / 4.0 - z * np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)))
    if k < 1:
        return float(walsh[0]), float(walsh[-1])
    return float(walsh[k - 1]), float(walsh[len(walsh) - k])


def _wilcoxon_critical(n, alpha):
    distribution = np.zeros(n * (n + 1) // 2 + 1)
    distribution[0] = 1.0
    for rank in range(1, n + 1):
        shifted = np.zeros_like(distribution)
        shifted[rank:] = distribution[:-rank]
        distribution = distribution + shifted
    distribution = distribution / distribution.sum()
    cumulative = np.cumsum(distribution)
    candidates = np.nonzero(cumulative <= alpha / 2.0)[0]
    if len(candidates) == 0:
        return None
    return int(candidates[-1]) + 1


def cohens_dz(first, second):
    differences = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    spread = differences.std(ddof=1)
    if spread == 0:
        return float("nan")
    return float(differences.mean() / spread)


def paired_mean_ci(first, second, alpha=0.05):
    differences = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    n = len(differences)
    centre = differences.mean()
    half_width = stats.t.ppf(1 - alpha / 2.0, n - 1) * differences.std(ddof=1) / np.sqrt(n)
    return float(centre), float(centre - half_width), float(centre + half_width)


def cles(first, second):
    a = np.asarray(first, dtype=float)[:, None]
    b = np.asarray(second, dtype=float)[None, :]
    return float(((a > b).sum() + 0.5 * (a == b).sum()) / (a.shape[0] * b.shape[1]))


def cles_ci(first, second, alpha=0.05, draws=10000, seed=20260915):
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    point = cles(a, b)
    generator = np.random.default_rng(seed)
    samples = np.empty(draws)
    for index in range(draws):
        samples[index] = cles(generator.choice(a, len(a), replace=True),
                              generator.choice(b, len(b), replace=True))
    lower, upper = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lower), float(upper)


def wilson_ci(successes, total, alpha=0.05):
    if total == 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(1 - alpha / 2.0)
    proportion = successes / total
    denominator = 1 + z ** 2 / total
    centre = (proportion + z ** 2 / (2 * total)) / denominator
    half_width = (z / denominator) * np.sqrt(proportion * (1 - proportion) / total
                                             + z ** 2 / (4 * total ** 2))
    return float(centre - half_width), float(centre + half_width)


def bh_adjust(pvalues):
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    n = len(values)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty(n)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def cluster_reduce(values, clusters):
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    unique = []
    means = []
    for cluster in dict.fromkeys(clusters.tolist()):
        unique.append(cluster)
        means.append(values[clusters == cluster].mean())
    return np.array(means), unique
