# Evaluation tasks (route-B sprint: 4 tasks)

Stratified by construction (plan ratio halved for the sprint): 2 natural-
headroom + 2 injected-degradation. Entry conditions, per the frozen plan:

- **Natural-headroom task**: ported from a real, permissively-licensed
  source (CC0/MIT/Apache-2.0/CC-BY only; Shadertoy default BY-NC-SA is NOT
  acceptable); a reference optimization must be measured on this device at
  a paired speedup CI lower bound >= 1.15 before the task enters the set.
  The reference solution is a *reference*, never called an upper bound.
- **Injected-degradation task**: a real permissive shader, ported, then a
  known performance degradation is injected (over-iteration, removed early
  exit, defeated-CSE double evaluation, unhoistable redundant work). The
  reference solution is the revert; its measured speedup must also clear
  CI low >= 1.15. Degradations must survive the compiler (measured, not
  assumed).

Every task records in registry.json: source URL/author/license (with the
exact location of the license statement), popularity + contamination flag
("possibly in training data" for high-exposure sources), port provenance
(what changed in porting), reference-solution sha256 + measured headroom CI
+ date, visible times, sealed-commitment id. Task sources live in
tasks/<task_id>.wgsl with the original header retained; reference solutions
in tasks/<task_id>_ref.wgsl are NEVER shown to the agent.

The authoritative mapping is `registry.json`: natural tasks expose the original
port and use a separate `_ref.wgsl`; injected tasks expose `_injected.wgsl` and
use the immutable original port as the reference/revert. Gears is a backup,
not one of the four registered evaluation tasks. `headroom` and `sealed_task`
remain null until the maintainer measures and commits them.

Frozen measurement settings (batch / duty): Tunnel 4 / 0.3, Dots3 96 / 0.7,
Cherry 72 / 0.7, codebasealpha 384 / 0.7. Gears, if promoted, uses 4 / 0.3;
it remains outside the registry while its contingency reference is measured.
The loader requires a positive integer batch and numeric `0 < duty <= 1`.

The maintainer's GPU null measurement showed why the light-task settings are
paired: Cherry at batch 48 / duty 0.3 falsely measured 0.82x with 39% CV;
batch 72 / duty 0.7 gave 0.9977x [0.9659, 1.0527] with 4.2% CV. Small submits
and low duty let GPU clock ramp dominate; retain the frozen per-task pair.

Agent-visible material per trajectory is **only that entry's `agent_file`**.
Each entry's `never_shown` list includes its reference and, where applicable,
its injection notes. `shaderperf.tasks_registry.load_registry(root=None)`
validates the manifest and returns these lists as sets; callers must enforce
that exclusion when assembling agent material. Sidecar notes are tracked
maintainer material, never agent input.
