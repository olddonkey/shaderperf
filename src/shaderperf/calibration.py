"""Generate blinded perceptual-calibration image pairs.

The generator renders RGBA8 images because the human labeler should judge the
same quantized LDR pixels that are written to PNG.  FLIP is computed
consistently from those uint8 RGB pixels converted to float32 in ``[0, 1]``.
GPU and Pillow imports are intentionally delayed until ``generate`` runs so
text transforms and CLI help remain CPU-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


ROOT = Path(__file__).resolve().parents[2]
SHADER_DIR = ROOT / "shaders" / "calibration"
DEFAULT_OUTPUT = ROOT / "results" / "calibration"
WIDTH = 1024
HEIGHT = 1024


TransformName = Literal[
    "safe_ballast_removal", "knob", "posterize", "constant_light"
]


@dataclass(frozen=True)
class VariantSpec:
    """One declared source transformation and its calibration metadata."""

    variant_id: str
    kind: str
    transform: TransformName
    severity: int
    knob_name: str | None = None
    knob_value: int | None = None
    posterize_levels: float | None = None


@dataclass(frozen=True)
class ShaderSpec:
    """Fixed time samples and variants for one calibration-only shader."""

    times: tuple[float, ...]
    variants: tuple[VariantSpec, ...]


def _common_variants(knob_name: str, knob_values: tuple[int, int, int]):
    return (
        VariantSpec(
            "safe_ballast", "known_safe", "safe_ballast_removal", 0
        ),
        *(
            VariantSpec(
                f"{knob_name.lower()}_{value}",
                "graded_degradation",
                "knob",
                severity,
                knob_name=knob_name,
                knob_value=value,
            )
            for severity, value in enumerate(knob_values, start=1)
        ),
        *(
            VariantSpec(
                f"posterize_{int(levels)}",
                "graded_degradation",
                "posterize",
                severity,
                posterize_levels=levels,
            )
            for severity, levels in enumerate((24.0, 8.0, 4.0), start=1)
        ),
    )


# This table is the pre-declared pair design.  It yields 50 pairs and seven
# known-safe pairs: sky/detail/waves use two times, while the static edge scene
# uses one.  Variant order is declared here; presentation order is seeded.
VARIANTS: dict[str, ShaderSpec] = {
    "cal_sky": ShaderSpec(
        times=(0.0, 3.25),
        variants=_common_variants("CLOUD_OCTAVES", (4, 2, 1)),
    ),
    "cal_edges": ShaderSpec(
        times=(1.0,),
        variants=_common_variants("MARCH_STEPS", (40, 24, 12))
        + (
            VariantSpec(
                "constant_light",
                "graded_degradation",
                "constant_light",
                1,
            ),
        ),
    ),
    "cal_detail": ShaderSpec(
        times=(0.0, 2.0),
        variants=_common_variants("DETAIL_OCTAVES", (5, 3, 1)),
    ),
    "cal_waves": ShaderSpec(
        times=(0.35, 2.4),
        variants=_common_variants("WAVE_TERMS", (10, 6, 3)),
    ),
}


def _replace_one(source: str, pattern: str, replacement: str, marker: str) -> str:
    transformed, count = re.subn(pattern, replacement, source, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"expected exactly one {marker}; found {count}")
    return transformed


def rewrite_knob(source: str, name: str, value: int) -> str:
    """Return source with exactly one exact-format i32 const rewritten."""

    if value < 0:
        raise ValueError("knob value must be non-negative")
    pattern = rf"^const {re.escape(name)}: i32 = [0-9]+;$"
    return _replace_one(
        source, pattern, f"const {name}: i32 = {value};", f"{name} const line"
    )


def remove_safe_block(source: str) -> str:
    """Delete the single SAFE-BEGIN..SAFE-END block, including marker lines."""

    pattern = r"^[ \t]*// SAFE-BEGIN\n[\s\S]*?^[ \t]*// SAFE-END\n?"
    return _replace_one(source, pattern, "", "SAFE block")


def apply_posterize(source: str, levels: float) -> str:
    """Replace the single posterization marker with an LDR quantizer."""

    if levels <= 0.0:
        raise ValueError("posterize levels must be positive")
    literal = f"{levels:.1f}"
    return _replace_one(
        source,
        r"^(?P<indent>[ \t]*)// POSTERIZE-SITE$",
        rf"\g<indent>col = floor(col * {literal}) / {literal};",
        "POSTERIZE-SITE line",
    )


def apply_constant_light(source: str) -> str:
    """Replace the edge shader's flat-light marker with a diffuse override."""

    return _replace_one(
        source,
        r"^(?P<indent>[ \t]*)// FLATLIGHT-SITE$",
        r"\g<indent>diffuse = 0.65;",
        "FLATLIGHT-SITE line",
    )


def apply_variant(source: str, variant: VariantSpec) -> str:
    """Apply one declared transform without touching the filesystem."""

    if variant.transform == "safe_ballast_removal":
        return remove_safe_block(source)
    if variant.transform == "knob":
        if variant.knob_name is None or variant.knob_value is None:
            raise ValueError("knob transform requires knob_name and knob_value")
        return rewrite_knob(source, variant.knob_name, variant.knob_value)
    if variant.transform == "posterize":
        if variant.posterize_levels is None:
            raise ValueError("posterize transform requires posterize_levels")
        return apply_posterize(source, variant.posterize_levels)
    if variant.transform == "constant_light":
        return apply_constant_light(source)
    raise ValueError(f"unknown transform: {variant.transform}")


def opaque_pair_id(shader: str, variant_id: str, time_value: float) -> str:
    """Return a stable identifier that reveals no experimental metadata."""

    key = f"{shader}|{variant_id}|{time_value:.17g}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]


def pair_plan(seed: int) -> list[dict]:
    """Build the complete deterministically shuffled pair plan."""

    pairs = []
    for shader, shader_spec in VARIANTS.items():
        for variant in shader_spec.variants:
            for time_value in shader_spec.times:
                pairs.append(
                    {
                        "pair_id": opaque_pair_id(
                            shader, variant.variant_id, time_value
                        ),
                        "shader": shader,
                        "variant_id": variant.variant_id,
                        "kind": variant.kind,
                        "severity": variant.severity,
                        "time": time_value,
                        "expected": (
                            "acceptable"
                            if variant.kind == "known_safe"
                            else "unknown"
                        ),
                        "_variant": variant,
                    }
                )
    random.Random(seed).shuffle(pairs)
    return pairs


def generate(output_dir: Path, seed: int) -> dict:
    """Render all declared pairs and write PNGs plus a blinded manifest."""

    import numpy as np
    import wgpu
    from PIL import Image

    from .harness import GpuContext, ShaderTimer
    from .verifier import flip_mean

    output_dir.mkdir(parents=True, exist_ok=True)
    context = GpuContext()
    reference_timers: dict[str, ShaderTimer] = {}
    sources: dict[str, str] = {}
    manifest_pairs = []
    plan = pair_plan(seed)

    for index, planned in enumerate(plan, start=1):
        shader = planned["shader"]
        shader_path = SHADER_DIR / f"{shader}.wgsl"
        if shader not in sources:
            sources[shader] = shader_path.read_text()
            reference_timers[shader] = ShaderTimer(
                shader_path,
                width=WIDTH,
                height=HEIGHT,
                context=context,
                texture_format=wgpu.TextureFormat.rgba8unorm,
            )

        variant = planned["_variant"]
        variant_source = apply_variant(sources[shader], variant)
        variant_timer = ShaderTimer(
            shader_path,
            width=WIDTH,
            height=HEIGHT,
            context=context,
            texture_format=wgpu.TextureFormat.rgba8unorm,
            source_override=variant_source,
        )
        time_value = planned["time"]
        ref_rgba = reference_timers[shader].read_pixels(time_value)
        var_rgba = variant_timer.read_pixels(time_value)

        pair_id = planned["pair_id"]
        Image.fromarray(ref_rgba, "RGBA").save(
            output_dir / f"{pair_id}_ref.png"
        )
        Image.fromarray(var_rgba, "RGBA").save(
            output_dir / f"{pair_id}_var.png"
        )
        ref_rgb = ref_rgba[..., :3].astype(np.float32) / np.float32(255.0)
        var_rgb = var_rgba[..., :3].astype(np.float32) / np.float32(255.0)
        row = {key: value for key, value in planned.items() if key != "_variant"}
        row["flip_mean"] = flip_mean(ref_rgb, var_rgb)
        manifest_pairs.append(row)
        print(f"[{index:02d}/{len(plan):02d}] rendered pair {pair_id}")

    manifest = {
        "schema_version": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed,
        "resolution": [WIDTH, HEIGHT],
        "flip_input": "rgba8unorm RGB converted to float32/255",
        "pairs": manifest_pairs,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser(
        "generate", help="render the declared calibration pairs"
    )
    generate_parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUTPUT, help="output directory"
    )
    generate_parser.add_argument(
        "--seed", type=int, default=0, help="deterministic presentation seed"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        manifest = generate(args.out, args.seed)
        print(f"wrote {len(manifest['pairs'])} pairs to {args.out}")
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
