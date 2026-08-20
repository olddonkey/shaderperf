# shaderperf

Measurement harness for studying whether coding agents can honestly optimize
real shaders for GPU time under a perceptual-equivalence constraint.

**Status: pre-gate #0.** No agent code exists yet, by design: the first
deliverable is a measurement-validity experiment (null runs + known-effect
positive controls, certified with the same paired-CI criterion the eventual
headline metric uses). Agent experiments only start after that gate passes.

## Stack

- WGSL fragment shaders, rendered headless via [wgpu-py](https://github.com/pygfx/wgpu-py) (Metal backend on macOS)
- Python 3.12, managed with `uv` (`uv sync`); `uv.lock` pins the environment
- Candidate measurement device: Apple M5 (integrated GPU). The device is
  frozen only if gate #0 passes on it.

## Usage

Measure one shader's per-pass GPU time distribution:

```
uv run python -m shaderperf.measure shaders/dev/fbm_smoke.wgsl \
    --warmup 10 --repeats 30 --batch 32 --out results/run.json
```

Paired comparison with randomized interleaving (the gate #0 mechanism):

```
uv run python -m shaderperf.compare baseline.wgsl patched.wgsl \
    --pairs 30 --batch 32 --out results/compare.json
```

Certification criterion: bootstrap CI of the p50 speedup ratio; a patch is
certified iff the CI lower bound >= 1.10.

## Measurement findings so far (2026-08-19, M5, wgpu-py 0.32)

Raw data in `results/`.

1. **Pass-boundary timestamp queries are unreliable on Metal/Apple Silicon**:
   negative deltas observed. GPU timestamps are recorded for diagnostics only;
   the primary signal is wall clock around submit+wait.
2. **Batching is required**: one sample = one submit containing N=32 separate
   render passes (separate passes, not N draws in one pass — Apple TBDR
   hidden-surface removal would eliminate overdraw). Wall-clock CV drops from
   ~32% (single pass) to ~4% (batched).
3. **Cross-process comparison is invalid**: a null comparison across two
   processes produced a false, CI-certified 1.26x "speedup" (~26% drift).
   All paired comparisons must interleave both conditions in one process on
   one shared device (`shaderperf.compare` does this).
4. With interleaving, the criterion behaves correctly on a smoke pair:
   null (shader vs itself) → 1.0013x, CI [0.944, 1.031], not certified;
   known-faster variant → 1.56x, CI [1.539, 1.593], certified. Note this
   smoke effect (56%) is far larger than the 15–20% effects gate #0 must
   certify; the smoke variant also alters the image, which is fine for pure
   timing but real positive controls must be known *honest* optimizations.
5. **Sustained full load has no steady state on the M5**: run 1's sentinel
   p50 rose monotonically by ~19% over ~50s. A fixed 90s full-load preheat in
   run 2 instead induced thermal oscillation, with 5–14% CV across sentinel
   blocks. Gate #0 therefore uses duty-cycled sampling at ~30% to keep average
   power below the throttle threshold.

## Gate #0 protocol

Each run evaluates three measurement-validity criteria:

- **(a) Positive-control sensitivity:** certify at least 4 of 5 known honest
  optimizations at a paired-bootstrap p50 speedup CI lower bound >= 1.10.
- **(b) Null specificity:** produce zero false certifications on null
  comparisons, checked in both directions.
- **(c) Within-run stability:** keep every sentinel-block p50 within +/-10%
  of the run's median sentinel p50.

Every positive control must also pass the pixel gate: maximum absolute channel
difference <= 1/255 (one 8-bit level), with no more than 1% of pixels differing.

The overall gate requires at least 3 passing runs across at least 2 distinct
UTC dates, using one fresh process per run. Run a certification and aggregate
all reports with:

```
uv run python -m shaderperf.gate0 --label day1-evening --duty 0.3
uv run python -m shaderperf.gate0_report
```

## Layout

- `src/shaderperf/harness.py` — device/pipeline setup, batched timed rendering
- `src/shaderperf/measure.py` — single-shader distribution CLI
- `src/shaderperf/compare.py` — interleaved paired-comparison CLI
- `src/shaderperf/stats.py` — summaries + paired bootstrap speedup CI
- `shaders/dev/` — self-written development/smoke shaders
- `results/` — raw run data (committed; every JSON embeds environment metadata)
