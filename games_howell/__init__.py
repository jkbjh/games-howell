from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.stats
#.studentized_range

@dataclass
class GamesHowellResult:
    pvalue: Any
    statistic: Any
    mean: Any
    n: Any
    df: Any
    var: Any
    k: Any

def games_howell(*samples):
    assert len(samples) > 1  # needs at least 2 samples.
    k = len(samples)
    means = np.zeros((k,))
    var = np.zeros((k,))
    n = np.zeros((k,))

    df = np.zeros((k, k))
    t = np.zeros((k, k))
    sigma = np.zeros((k, k))
    p = np.zeros((k, k))

    for i, sample in enumerate(samples):
        n[i] = len(sample)
        var[i] = np.var(sample, ddof=1)
        means[i] = np.mean(sample)

    #s2 = std ** 2

    for i, sample_i in enumerate(samples):
        for j, sample_j in enumerate(samples):
            df[i, j] = (
                (var[i] / n[i] + var[j] / n[j]) ** 2 /
                (
                    (var[i]/n[i])**2/(n[i]-1) +
                    (var[j]/n[j])**2/(n[j]-1)
                )
            )

            sigma[i, j] = np.sqrt(var[i]/n[i] + var[j]/n[j])

            t[i, j] = (means[i] - means[j]) / sigma[i, j]


    pvals = scipy.stats.studentized_range.sf(np.abs(t.flatten()) * np.sqrt(2), k, df.flatten()).reshape(t.shape)

    eps = np.finfo(np.float32).eps
    pvals = np.where(np.abs(pvals) > eps, pvals, eps)
    assert np.all(pvals >= 0.)

    return GamesHowellResult(
        pvalue=pvals,
        statistic=t,
        mean=means,
        var=var,
        n=n,
        df=df,
        k=k
        )
