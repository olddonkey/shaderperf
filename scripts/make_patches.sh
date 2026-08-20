#!/bin/bash
# Regenerate the gate #0 known-optimization patches from their baselines.
# Patches are derived mechanically so the exact diff is reproducible:
#   raymarch_sdf:     insert early loop termination at the marked site
#   mandelbrot_orbit: delete ballast blocks (trig-identity zero terms)
#   fbm_warp:         delete micro block (sub-quantization noise octaves)
#   voronoi_cells:    delete ballast blocks (trig-identity zero terms)
#   plasma_layers:    delete ballast blocks (trig-identity zero terms)
set -euo pipefail
cd "$(dirname "$0")/../shaders/gate0"

sed 's|// EARLY-EXIT-SITE|if (hit \|\| t > TMAX) { break; }|' \
    raymarch_sdf.wgsl > raymarch_sdf_opt.wgsl

sed '/BALLAST-BEGIN/,/BALLAST-END/d' mandelbrot_orbit.wgsl > mandelbrot_orbit_opt.wgsl

sed '/MICRO-BEGIN/,/MICRO-END/d' fbm_warp.wgsl > fbm_warp_opt.wgsl

sed '/BALLAST-BEGIN/,/BALLAST-END/d' voronoi_cells.wgsl > voronoi_cells_opt.wgsl

sed '/BALLAST-BEGIN/,/BALLAST-END/d' plasma_layers.wgsl > plasma_layers_opt.wgsl

echo "patches regenerated:"
ls -1 *_opt.wgsl
