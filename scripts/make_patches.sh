#!/bin/bash
# Regenerate the gate #0 known-optimization patches from their baselines.
# Patches are derived mechanically so the exact diff is reproducible:
#   raymarch_sdf:     insert early loop termination at the marked site
#   mandelbrot_orbit: delete ballast blocks (trig-identity zero terms)
#   fbm_warp:         delete micro block (sub-quantization noise octaves)
#   voronoi_cells:    delete ballast blocks (trig-identity zero terms)
#   plasma_layers:    delete ballast blocks (trig-identity zero terms)
#   voronoi_cells_snr4: preserve structure; reduce BALLAST_CELLS 12 -> 4
#   mandelbrot_orbit_snr8: preserve structure; reduce BALLAST_ITERS 25 -> 8
set -euo pipefail
cd "$(dirname "$0")/../shaders/gate0"

sed 's|// EARLY-EXIT-SITE|if (hit \|\| t > TMAX) { break; }|' \
    raymarch_sdf.wgsl > raymarch_sdf_opt.wgsl

sed '/BALLAST-BEGIN/,/BALLAST-END/d' mandelbrot_orbit.wgsl > mandelbrot_orbit_opt.wgsl

sed '/MICRO-BEGIN/,/MICRO-END/d' fbm_warp.wgsl > fbm_warp_opt.wgsl

sed '/BALLAST-BEGIN/,/BALLAST-END/d' voronoi_cells.wgsl > voronoi_cells_opt.wgsl

sed '/BALLAST-BEGIN/,/BALLAST-END/d' plasma_layers.wgsl > plasma_layers_opt.wgsl

sed 's/const BALLAST_CELLS: i32 = 12;/const BALLAST_CELLS: i32 = 4;/' \
    voronoi_cells.wgsl > voronoi_cells_snr4.wgsl

sed 's/const BALLAST_ITERS: i32 = 25;/const BALLAST_ITERS: i32 = 8;/' \
    mandelbrot_orbit.wgsl > mandelbrot_orbit_snr8.wgsl

echo "patches regenerated:"
ls -1 *_opt.wgsl *_snr*.wgsl
