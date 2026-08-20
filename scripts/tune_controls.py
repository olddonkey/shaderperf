"""Tuning helper: interleaved point estimates + pixel checks for all 5
gate #0 positive-control pairs. Used to size each control's effect into the
15-20% band before the certification run. Not part of the certification
protocol itself.

Run: uv run python scripts/tune_controls.py [--pairs 12] [--batch 32]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from shaderperf.harness import GpuContext, ShaderTimer
from shaderperf.protocol import interleaved_compare, pixel_equivalence

PAIRS = [
    ("raymarch_sdf", "shaders/gate0/raymarch_sdf.wgsl", "shaders/gate0/raymarch_sdf_opt.wgsl"),
    ("mandelbrot_orbit", "shaders/gate0/mandelbrot_orbit.wgsl", "shaders/gate0/mandelbrot_orbit_opt.wgsl"),
    ("fbm_warp", "shaders/gate0/fbm_warp.wgsl", "shaders/gate0/fbm_warp_opt.wgsl"),
    ("voronoi_cells", "shaders/gate0/voronoi_cells.wgsl", "shaders/gate0/voronoi_cells_opt.wgsl"),
    ("plasma_layers", "shaders/gate0/plasma_layers.wgsl", "shaders/gate0/plasma_layers_opt.wgsl"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--only", type=str, default=None)
    args = ap.parse_args()

    root = Path(__file__).parent.parent
    ctx = GpuContext()
    rng = np.random.default_rng(1)

    for name, base, opt in PAIRS:
        if args.only and args.only not in name:
            continue
        ta = ShaderTimer(root / base, context=ctx)
        tb = ShaderTimer(root / opt, context=ctx)
        r = interleaved_compare(ta, tb, args.pairs, args.batch, warmup=4, rng=rng)
        px = pixel_equivalence(ta, tb)
        ci = r["speedup_ci"]
        print(
            f"{name:18s} base p50={r['a_summary']['p50_ns'] / 1e6:6.3f} ms  "
            f"speedup={ci['speedup_point']:.4f}x  "
            f"CI [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]  "
            f"pixels: max_diff={px['max_abs_diff']} "
            f"differing={px['differing_pixels']}/{px['total_pixels']}"
        )


if __name__ == "__main__":
    main()
