# Cherry — maintainer only, never shown to the agent

Agent file: `tasks/task_cherry_injected.wgsl`.
Reference/revert: the immutable `tasks/task_cherry.wgsl`.

<!-- INJECT-BEGIN: first-petal panel field -->

`petal_field` extracts the original per-petal expressions without changing
their arithmetic or the five-petal loop. For `i == 0`, a second evaluation of
both petal and cut fields is averaged with the first.

The second phase uses `frexp(phi)` followed by two `ldexp` operations, with an
exponent split from the varying fragment x coordinate (one through four).
For a finite normal phase, write `phi = m * 2^e`, with `0.5 <= abs(m) < 1`.
The intermediate `m * 2^-s` lies between 1/32 and 1/2 in magnitude and is
normal. Power-of-two scaling is exact here: `(m * 2^-s) * 2^(e+s)` restores
the original f32 bits (also preserving zero). Consequently both field calls
receive the same bits and produce the same finite, bounded field values.
For these values `0.5 * (x + x)` returns exactly `x`: doubling cannot
overflow and introduces no rounding. The min/max reductions remain unchanged.
This argument covers the normal finite time/viewport domain of this task,
including the visible times; it makes no claim for nonfinite uniforms.

The second phase has a different dataflow from the first. Folding the
varying exponent split across the two `ldexp` calls requires a range proof
(in general such reassociation can underflow or overflow); ordinary CSE
cannot merge the two call arguments. There is no trigonometric zero ballast
and no change to the image, geometry, epsilon, or sample count.

One extra petal evaluation per five originals adds 20% to the dominant
field work, plus the decomposition/scaling overhead. The intended total
overhead is 20–40%, subject to the maintainer's compiler/timing gate. No GPU
compilation or timing was performed in this unit; compiler survival and
the actual overhead must be measured, not inferred from this operation count.
If retuning is needed, change the number of petals selected by the condition.

<!-- INJECT-END: first-petal panel field -->
