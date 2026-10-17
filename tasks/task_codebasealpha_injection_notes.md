# codebasealpha — maintainer only, never shown to the agent

Agent file: `tasks/task_codebasealpha_injected.wgsl`.
Reference/revert: the immutable `tasks/task_codebasealpha.wgsl`.

<!-- INJECT-BEGIN: left-panel radial field -->

The left half of the viewport evaluates `dF2` a second time and averages the
two distances. The right half retains the single original evaluation.
The coherent half-screen condition limits the added work without a
pixel-checkerboard branch that would execute both paths in every SIMD group.

The second coordinate is reconstructed componentwise with `frexp` and
two `ldexp` operations. The exponent split is one through four, selected
from the varying fragment y coordinate. For each finite normal coordinate
`u = m * 2^e`, the intermediate `m * 2^-s` is normal (magnitude at least
1/32), and exact power-of-two scaling restores the original f32 bits;
zero is also preserved. Both calls therefore receive bit-equal coordinates
and uniforms. The bounded SDF result cannot overflow when doubled, so
`0.5 * (d + d)` is exactly `d`, including zero. All downstream thresholds,
colors, and postprocessing receive the original distance. This argument
covers the task's ordinary finite viewport coordinates and times, not
nonfinite inputs or hardware-specific flushing of subnormal coordinates.

The calls use distinct argument dataflows. Ordinary CSE cannot merge
`uv` with a runtime exponent reconstruction; collapsing the two scaling
operations requires a range proof because general `ldexp` reassociation
can underflow or overflow. No trigonometric zero ballast is used.

The extra work is half an SDF evaluation per pixel on average; postprocessing
is still evaluated once. The intended whole-pass overhead is 20–40%, not
50% of the entire pass. The maintainer must verify compiler survival,
pixel equality, and actual timing on the target device; no GPU compilation
or measurement was performed here. The half-screen fraction is the tuning
parameter if measured overhead misses that range.

<!-- INJECT-END: left-panel radial field -->
