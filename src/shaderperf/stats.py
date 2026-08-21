"""Summary statistics for timing runs.

Measurement unit per project-v0.md: per-run GPU time = p50 over the fixed
repeat count; p95 reported alongside. The paired-speedup bootstrap CI here is
the same criterion the headline metric will use (CI lower bound >= 1.10), so
gate #0 exercises the exact yardstick of the real experiment.
"""

from __future__ import annotations

import numpy as np


def summarize(samples_ns: list[int]) -> dict:
    a = np.asarray(samples_ns, dtype=np.float64)
    return {
        "n": int(a.size),
        "p50_ns": float(np.percentile(a, 50)),
        "p95_ns": float(np.percentile(a, 95)),
        "mean_ns": float(a.mean()),
        "std_ns": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "cv": float(a.std(ddof=1) / a.mean()) if a.size > 1 and a.mean() > 0 else None,
        "min_ns": float(a.min()),
        "max_ns": float(a.max()),
    }


def paired_speedup_ci(
    baseline_ns: list[int],
    patched_ns: list[int],
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap CI for speedup = p50(baseline) / p50(patched).

    Resamples each condition's samples with replacement, recomputes the p50
    ratio per replicate, and reports percentile bounds. This is the gate #0 /
    headline certification statistic: certified iff ci_low >= 1.10.
    """
    base = np.asarray(baseline_ns, dtype=np.float64)
    pat = np.asarray(patched_ns, dtype=np.float64)
    rng = np.random.default_rng(seed)

    ratios = np.empty(n_boot)
    for i in range(n_boot):
        b = rng.choice(base, size=base.size, replace=True)
        p = rng.choice(pat, size=pat.size, replace=True)
        ratios[i] = np.percentile(b, 50) / np.percentile(p, 50)

    point = float(np.percentile(base, 50) / np.percentile(pat, 50))
    lo = float(np.percentile(ratios, 100 * alpha / 2))
    hi = float(np.percentile(ratios, 100 * (1 - alpha / 2)))
    return {
        "speedup_point": point,
        "ci_low": lo,
        "ci_high": hi,
        "alpha": alpha,
        "n_boot": n_boot,
        "seed": seed,
        "certified_at_1.10": lo >= 1.10,
    }


def bootstrap_share_difference(
    full_a: list[int],
    var_a: list[int],
    full_b: list[int],
    var_b: list[int],
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap CI for the difference between two ablation cost shares.

    Each of the four conditions is resampled independently. A condition's
    share is ``1 - p50(variant) / p50(full)``, and the reported delta is the
    share for condition A minus the share for condition B.
    """
    full_a_array = np.asarray(full_a, dtype=np.float64)
    var_a_array = np.asarray(var_a, dtype=np.float64)
    full_b_array = np.asarray(full_b, dtype=np.float64)
    var_b_array = np.asarray(var_b, dtype=np.float64)
    rng = np.random.default_rng(seed)

    deltas = np.empty(n_boot)
    for i in range(n_boot):
        full_a_sample = rng.choice(
            full_a_array, size=full_a_array.size, replace=True
        )
        var_a_sample = rng.choice(var_a_array, size=var_a_array.size, replace=True)
        full_b_sample = rng.choice(
            full_b_array, size=full_b_array.size, replace=True
        )
        var_b_sample = rng.choice(var_b_array, size=var_b_array.size, replace=True)
        share_a = 1.0 - np.percentile(var_a_sample, 50) / np.percentile(
            full_a_sample, 50
        )
        share_b = 1.0 - np.percentile(var_b_sample, 50) / np.percentile(
            full_b_sample, 50
        )
        deltas[i] = share_a - share_b

    share_a_point = 1.0 - np.percentile(var_a_array, 50) / np.percentile(
        full_a_array, 50
    )
    share_b_point = 1.0 - np.percentile(var_b_array, 50) / np.percentile(
        full_b_array, 50
    )
    point = float(share_a_point - share_b_point)
    lo = float(np.percentile(deltas, 100 * alpha / 2))
    hi = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
    return {
        "delta_point": point,
        "ci_low": lo,
        "ci_high": hi,
        "alpha": alpha,
        "n_boot": n_boot,
        "seed": seed,
        "distinguishable_from_zero": lo > 0 or hi < 0,
    }
