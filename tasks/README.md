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

Agent-visible material per trajectory: tasks/<task_id>.wgsl only.
