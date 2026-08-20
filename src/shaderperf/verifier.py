"""Perceptual-equivalence checks for rendered shader images."""

from __future__ import annotations

from pathlib import Path

import flip_evaluator as flip
import numpy as np
import wgpu

from .harness import GpuContext, ShaderTimer


def flip_mean(ref_rgb: np.ndarray, test_rgb: np.ndarray) -> float:
    """Return mean FLIP error for float32 RGB images in ``[0, 1]``.

    Shader RGB outputs are treated as sRGB-encoded LDR values. The caller is
    responsible for supplying matching ``H x W x 3`` float32 images.
    """

    _error_map, mean_error, _parameters = flip.evaluate(ref_rgb, test_rgb, "LDR")
    return float(mean_error)


def _rgb_channels(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError("expected an H x W image with at least three channels")
    return array[..., :3]


def quantize_ldr(rgba_float: np.ndarray) -> np.ndarray:
    """Apply the verifier's canonical float-to-8-bit RGB quantization.

    Alpha is dropped. RGB is clipped to ``[0, 1]``, rounded to the nearest
    8-bit level with NumPy's ties-to-even rule, and returned as float32. RGB
    input is also accepted so that applying this function twice is idempotent.
    """

    rgb = _rgb_channels(rgba_float).astype(np.float32, copy=False)
    levels = np.rint(np.clip(rgb, 0.0, 1.0) * np.float32(255.0))
    return (levels / np.float32(255.0)).astype(np.float32, copy=False)


def numerical_check(rgba_float: np.ndarray) -> dict:
    """Check finiteness and range over RGB, excluding alpha.

    Alpha is intentionally excluded because ShaderPerf's task domain fixes it
    to 1. The returned ``min`` and ``max`` are computed over all RGB values;
    either is ``None`` when its computed value is non-finite so reports remain
    strict-JSON serializable.
    """

    rgb = _rgb_channels(rgba_float)
    has_nan = bool(np.isnan(rgb).any())
    has_inf = bool(np.isinf(rgb).any())
    raw_minimum = float(np.min(rgb))
    raw_maximum = float(np.max(rgb))
    minimum = raw_minimum if np.isfinite(raw_minimum) else None
    maximum = raw_maximum if np.isfinite(raw_maximum) else None
    in_range = bool(np.all((rgb >= 0.0) & (rgb <= 1.0)))
    passed = not has_nan and not has_inf and in_range
    return {
        "has_nan": has_nan,
        "has_inf": has_inf,
        "min": minimum,
        "max": maximum,
        "in_range": in_range,
        "passed": passed,
    }


def evaluate_sample(
    ref_timer: ShaderTimer, test_timer: ShaderTimer, time_value: float
) -> dict:
    """Render and evaluate one time value using RGBA32F ShaderTimers."""

    for label, timer in (("ref_timer", ref_timer), ("test_timer", test_timer)):
        if timer.texture_format != wgpu.TextureFormat.rgba32float:
            raise ValueError(f"{label} must use rgba32float")

    ref_rgba = ref_timer.read_pixels(time_value)
    test_rgba = test_timer.read_pixels(time_value)
    numerical_ref = numerical_check(ref_rgba)
    numerical_test = numerical_check(test_rgba)

    return {
        "time": float(time_value),
        "flip_mean": flip_mean(quantize_ldr(ref_rgba), quantize_ldr(test_rgba)),
        "numerical_ref": numerical_ref,
        "numerical_test": numerical_test,
    }


def verify(
    ref_path: str | Path,
    test_path: str | Path,
    sample_times,
    tc_times,
    tau_flip: float,
    tau_tc: float | None,
    context: GpuContext | None = None,
    width: int = 1024,
    height: int = 1024,
) -> dict:
    """Run the parameterized perceptual verifier for two shader files."""

    sample_times = [float(value) for value in sample_times]
    tc_times = [float(value) for value in tc_times]
    tau_flip = float(tau_flip)
    if tc_times and tau_tc is None:
        raise ValueError("tau_tc is required when tc_times is not empty")
    tau_tc = None if tau_tc is None else float(tau_tc)

    ctx = context if context is not None else GpuContext()
    timer_kwargs = {
        "width": width,
        "height": height,
        "context": ctx,
        "texture_format": wgpu.TextureFormat.rgba32float,
    }
    ref_timer = ShaderTimer(Path(ref_path), **timer_kwargs)
    test_timer = ShaderTimer(Path(test_path), **timer_kwargs)

    # A time shared by the sample and temporal sets is rendered only once per
    # shader. This keeps the float image as the single source for all checks.
    evaluations: dict[float, dict] = {}

    def at_time(time_value: float) -> dict:
        if time_value not in evaluations:
            evaluations[time_value] = evaluate_sample(
                ref_timer, test_timer, time_value
            )
        return evaluations[time_value]

    samples = []
    for time_value in sample_times:
        result = at_time(time_value)
        passed = bool(
            result["flip_mean"] <= tau_flip
            and result["numerical_test"]["passed"]
        )
        samples.append(result | {"passed": passed})

    temporal_frames = [dict(at_time(time_value)) for time_value in tc_times]
    flip_series = [frame["flip_mean"] for frame in temporal_frames]
    tc_max_delta = max(
        (abs(right - left) for left, right in zip(flip_series, flip_series[1:])),
        default=0.0,
    )
    tc_pass = True if not tc_times else bool(tc_max_delta <= tau_tc)
    tc_numerical_pass = all(
        frame["numerical_test"]["passed"] for frame in temporal_frames
    )
    verdict = bool(
        all(sample["passed"] for sample in samples)
        and tc_pass
        and tc_numerical_pass
    )

    return {
        "reference_path": str(ref_path),
        "test_path": str(test_path),
        "shader_hashes": {
            "reference_sha256": ref_timer.shader_hash,
            "test_sha256": test_timer.shader_hash,
        },
        "resolution": [width, height],
        "thresholds": {"tau_flip": tau_flip, "tau_tc": tau_tc},
        "samples": samples,
        "temporal_frames": temporal_frames,
        "flip_series": flip_series,
        "tc_max_delta": float(tc_max_delta),
        "tc_pass": tc_pass,
        "tc_numerical_pass": tc_numerical_pass,
        "verdict": verdict,
    }
