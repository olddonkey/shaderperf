"""Shared measurement protocol building blocks for gate #0.

Everything here assumes the finding from 2026-08-19 smoke runs: paired
comparisons are only valid when both conditions share one process and one
device and samples are taken in randomized interleaved order.
"""

from __future__ import annotations

import numpy as np

from .harness import ShaderTimer
from .stats import paired_speedup_ci, summarize


def interleaved_compare(
    timer_a: ShaderTimer,
    timer_b: ShaderTimer,
    pairs: int,
    batch: int,
    warmup: int,
    rng: np.random.Generator,
    time_value: float = 1.0,
) -> dict:
    """Randomized-interleaved paired comparison. Returns raw samples + CI."""
    for t in (timer_a, timer_b):
        t.set_uniforms(time_value)
    for _ in range(warmup):
        timer_a.render_once(batch)
        timer_b.render_once(batch)

    wall_a: list[float] = []
    wall_b: list[float] = []
    order_log: list[str] = []
    for _ in range(pairs):
        first_a = bool(rng.random() < 0.5)
        order_log.append("a" if first_a else "b")
        seq = [(timer_a, wall_a), (timer_b, wall_b)]
        if not first_a:
            seq.reverse()
        for timer, sink in seq:
            sink.append(timer.render_once(batch).wall_ns / batch)

    return {
        "a_wall_ns_per_pass": wall_a,
        "b_wall_ns_per_pass": wall_b,
        "first_in_pair": order_log,
        "a_summary": summarize(wall_a),
        "b_summary": summarize(wall_b),
        "speedup_ci": paired_speedup_ci(wall_a, wall_b),
    }


def pixel_equivalence(
    timer_a: ShaderTimer, timer_b: ShaderTimer, time_value: float = 1.0
) -> dict:
    """Exact pixel comparison of the two shaders' rendered output."""
    img_a = timer_a.read_pixels(time_value).astype(np.int16)
    img_b = timer_b.read_pixels(time_value).astype(np.int16)
    diff = np.abs(img_a - img_b)
    return {
        "max_abs_diff": int(diff.max()),
        "differing_pixels": int((diff.max(axis=2) > 0).sum()),
        "total_pixels": int(img_a.shape[0] * img_a.shape[1]),
        "identical": bool(diff.max() == 0),
    }
